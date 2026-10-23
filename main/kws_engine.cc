#include "kws_engine.h"
#include "model_data.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "nvs.h"
#include "nvs_flash.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

#include "tensorflow/lite/micro/memory_planner/linear_memory_planner.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

static const char *TAG = "KWS_ENGINE";

// Tensor Arena: 2 MB in PSRAM for LinearMemoryPlanner (requested ~1.15 MB)
constexpr size_t kTensorArenaSize = 2 * 1024 * 1024;
static uint8_t *s_tensor_arena = nullptr;

static const tflite::Model *s_model = nullptr;
static tflite::MicroInterpreter *s_interpreter = nullptr;
alignas(tflite::MicroInterpreter) static uint8_t
    s_interpreter_buf[sizeof(tflite::MicroInterpreter)];
static TfLiteTensor *s_input_tensor = nullptr;
static TfLiteTensor *s_output_tensor = nullptr;
static bool s_kws_ready = false;

// MicroMutableOpResolver for operators used by the model
static tflite::MicroMutableOpResolver<12> s_op_resolver;

// Precomputed Mel filterbank constants
static constexpr int FFT_SIZE = 512;
static constexpr int HOP_SIZE =
    320; // 20ms at 16kHz -> (16000-512)/320 + 1 = 49 frames

// Static working buffers to prevent Stack Protection Fault in FreeRTOS tasks
static float s_fft_buf[FFT_SIZE * 2];
static float s_power_spectrum[FFT_SIZE / 2 + 1];
static int8_t s_features[KWS_INPUT_FEATURES_LEN];

// =========================================================
// On-Device Continual Learning & NVS Persistence
// =========================================================
#define KWS_LABEL_CHUEY_DUAY 0
#define KWS_LABEL_OTHER 1

#define KWS_EMBEDDING_DIM 64
#define KWS_NUM_CLASSES 2

#define NVS_NAMESPACE "kws_model"
#define NVS_KEY_WEIGHTS "dense_w"
#define NVS_KEY_BIAS "dense_b"
#define NVS_KEY_COUNT "train_cnt"
#define NVS_KEY_MAGIC "kws_magic"
#define KWS_NVS_MAGIC_VAL                                                      \
  0x4B575305 // Magic tag for KWS Dense layer (v4: 10-Persona 1,000-sample
             // balanced model)

// Trainable Dense classification head parameters (Float32 in RAM for smooth
// gradient learning)
static float s_learned_weights[KWS_NUM_CLASSES][KWS_EMBEDDING_DIM];
static float s_learned_bias[KWS_NUM_CLASSES];
static uint32_t s_train_sample_count = 0;
static bool s_weights_customized = false;

// Last audio window embedding (64 float values)
static float s_last_audio_embedding[KWS_EMBEDDING_DIM];
static bool s_last_audio_embedding_valid = false;

// Tensor index and quantization params for GlobalAveragePooling2D
static int s_embedding_tensor_idx = -1;
static float s_embedding_scale = 1.0f;
static int32_t s_embedding_zp = 0;

static void init_factory_weights(void);

extern "C" esp_err_t kws_engine_save_to_nvs(void) {
  nvs_handle_t handle;
  esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READWRITE, &handle);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to open NVS namespace: %s", esp_err_to_name(err));
    return err;
  }

  err = nvs_set_blob(handle, NVS_KEY_WEIGHTS, s_learned_weights,
                     sizeof(s_learned_weights));
  if (err == ESP_OK) {
    err = nvs_set_blob(handle, NVS_KEY_BIAS, s_learned_bias,
                       sizeof(s_learned_bias));
  }
  if (err == ESP_OK) {
    err = nvs_set_u32(handle, NVS_KEY_COUNT, s_train_sample_count);
  }
  if (err == ESP_OK) {
    err = nvs_set_u32(handle, NVS_KEY_MAGIC, KWS_NVS_MAGIC_VAL);
  }
  if (err == ESP_OK) {
    err = nvs_commit(handle);
  }
  nvs_close(handle);

  if (err == ESP_OK) {
    ESP_LOGI(TAG,
             "💾 [NVS SAVED] Stored updated model weights into Flash NVS "
             "(Samples trained: %lu)",
             (unsigned long)s_train_sample_count);
  } else {
    ESP_LOGE(TAG, "Failed to write weights to NVS: %s", esp_err_to_name(err));
  }
  return err;
}

extern "C" esp_err_t kws_engine_load_from_nvs(void) {
  nvs_handle_t handle;
  esp_err_t err = nvs_open(NVS_NAMESPACE, NVS_READONLY, &handle);
  if (err != ESP_OK) {
    return ESP_ERR_NOT_FOUND;
  }

  uint32_t magic = 0;
  err = nvs_get_u32(handle, NVS_KEY_MAGIC, &magic);
  if (err != ESP_OK || magic != KWS_NVS_MAGIC_VAL) {
    nvs_close(handle);
    return ESP_ERR_NOT_FOUND;
  }

  size_t w_size = sizeof(s_learned_weights);
  size_t b_size = sizeof(s_learned_bias);
  err = nvs_get_blob(handle, NVS_KEY_WEIGHTS, s_learned_weights, &w_size);
  if (err == ESP_OK) {
    err = nvs_get_blob(handle, NVS_KEY_BIAS, s_learned_bias, &b_size);
  }
  if (err == ESP_OK) {
    nvs_get_u32(handle, NVS_KEY_COUNT, &s_train_sample_count);
  }
  nvs_close(handle);

  if (err == ESP_OK && w_size == sizeof(s_learned_weights) &&
      b_size == sizeof(s_learned_bias)) {
    s_weights_customized = true;
    ESP_LOGI(TAG,
             "🎉 [NVS LOADED] Restored trained model weights from Flash NVS! "
             "(Trained samples: %lu)",
             (unsigned long)s_train_sample_count);
    return ESP_OK;
  }
  return ESP_FAIL;
}

extern "C" esp_err_t kws_engine_reset_weights(void) {
  init_factory_weights();
  s_train_sample_count = 0;
  s_weights_customized = false;

  nvs_handle_t handle;
  if (nvs_open(NVS_NAMESPACE, NVS_READWRITE, &handle) == ESP_OK) {
    nvs_erase_all(handle);
    nvs_commit(handle);
    nvs_close(handle);
  }
  ESP_LOGI(TAG, "Model weights reset to original factory pretrained state");
  return ESP_OK;
}

extern "C" uint32_t kws_engine_get_train_count(void) {
  return s_train_sample_count;
}

static void init_factory_weights(void) {
  if (!s_model)
    return;

  const auto *subgraphs = s_model->subgraphs();
  if (!subgraphs || subgraphs->size() == 0)
    return;
  const auto *sg = subgraphs->Get(0);
  const auto *ops = sg->operators();

  bool found_fc = false;
  for (size_t i = 0; i < ops->size(); i++) {
    const auto *op = ops->Get(i);
    int opcode_idx = op->opcode_index();
    auto opcode = s_model->operator_codes()->Get(opcode_idx)->builtin_code();
    if (opcode == tflite::BuiltinOperator_FULLY_CONNECTED) {
      int w_tensor_idx = op->inputs()->Get(1);
      int b_tensor_idx = (op->inputs()->size() > 2) ? op->inputs()->Get(2) : -1;

      auto *w_tensor = sg->tensors()->Get(w_tensor_idx);
      int w_buf_idx = w_tensor->buffer();
      const auto *w_buf = s_model->buffers()->Get(w_buf_idx);
      const int8_t *raw_w = (const int8_t *)w_buf->data()->data();
      float w_scale =
          (w_tensor->quantization() && w_tensor->quantization()->scale() &&
           w_tensor->quantization()->scale()->size() > 0)
              ? w_tensor->quantization()->scale()->Get(0)
              : 0.01f;
      int32_t w_zp =
          (w_tensor->quantization() && w_tensor->quantization()->zero_point() &&
           w_tensor->quantization()->zero_point()->size() > 0)
              ? w_tensor->quantization()->zero_point()->Get(0)
              : 0;

      const int32_t *raw_b = nullptr;
      float b_scale = w_scale * s_embedding_scale;
      if (b_tensor_idx >= 0) {
        auto *b_tensor = sg->tensors()->Get(b_tensor_idx);
        int b_buf_idx = b_tensor->buffer();
        const auto *b_buf = s_model->buffers()->Get(b_buf_idx);
        raw_b = (const int32_t *)b_buf->data()->data();
        if (b_tensor->quantization() && b_tensor->quantization()->scale() &&
            b_tensor->quantization()->scale()->size() > 0) {
          b_scale = b_tensor->quantization()->scale()->Get(0);
        }
      }

      for (int c = 0; c < KWS_NUM_CLASSES; c++) {
        for (int d = 0; d < KWS_EMBEDDING_DIM; d++) {
          int8_t rw = raw_w[c * KWS_EMBEDDING_DIM + d];
          s_learned_weights[c][d] = (float)(rw - w_zp) * w_scale;
        }
        if (raw_b) {
          s_learned_bias[c] = (float)raw_b[c] * b_scale;
        } else {
          s_learned_bias[c] = 0.0f;
        }
      }
      found_fc = true;
      ESP_LOGI(TAG,
               "Initialized factory dense weights from FlatBuffer (b0=%.3f, "
               "b1=%.3f)",
               s_learned_bias[0], s_learned_bias[1]);
      break;
    }
  }

  if (!found_fc) {
    ESP_LOGW(TAG, "FULLY_CONNECTED op not found in FlatBuffer, using zero "
                  "initialization");
    std::memset(s_learned_weights, 0, sizeof(s_learned_weights));
    std::memset(s_learned_bias, 0, sizeof(s_learned_bias));
  }
}

extern "C" esp_err_t kws_engine_init(void) {
  if (s_kws_ready && s_interpreter != nullptr) {
    return ESP_OK;
  }

  // Ensure NVS Flash is ready
  esp_err_t nvs_err = nvs_flash_init();
  if (nvs_err == ESP_ERR_NVS_NO_FREE_PAGES ||
      nvs_err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }

  tflite::InitializeTarget();

  s_model = tflite::GetModel(g_kws_model_data);
  if (s_model->version() != TFLITE_SCHEMA_VERSION) {
    ESP_LOGE(TAG, "Model schema version mismatch: %ld vs %d",
             s_model->version(), TFLITE_SCHEMA_VERSION);
    return ESP_FAIL;
  }

  // Register required operators (only once)
  static bool s_ops_registered = false;
  if (!s_ops_registered) {
    s_op_resolver.AddConv2D();
    s_op_resolver.AddDepthwiseConv2D();
    s_op_resolver.AddMaxPool2D();
    s_op_resolver.AddMean();
    s_op_resolver.AddFullyConnected();
    s_op_resolver.AddSoftmax();
    s_op_resolver.AddMul();
    s_op_resolver.AddAdd();
    s_op_resolver.AddPad();
    s_op_resolver.AddReshape();
    s_op_resolver.AddQuantize();
    s_op_resolver.AddDequantize();
    s_ops_registered = true;
  }

  // Allocate tensor arena in PSRAM (32 MB available on ESP32-P4)
  if (!s_tensor_arena) {
    s_tensor_arena = (uint8_t *)heap_caps_malloc(
        kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    if (!s_tensor_arena) {
      ESP_LOGW(TAG, "PSRAM full, allocating Tensor Arena in Internal RAM...");
      s_tensor_arena = (uint8_t *)heap_caps_malloc(
          kTensorArenaSize, MALLOC_CAP_8BIT | MALLOC_CAP_INTERNAL);
    }
  }

  if (!s_tensor_arena) {
    ESP_LOGE(TAG, "Failed to allocate tensor arena (%d bytes)",
             (int)kTensorArenaSize);
    return ESP_ERR_NO_MEM;
  }

  if (s_interpreter != nullptr) {
    s_interpreter->~MicroInterpreter();
    s_interpreter = nullptr;
  }

  static tflite::LinearMemoryPlanner s_linear_memory_planner;
  tflite::MicroAllocator *allocator = tflite::MicroAllocator::Create(
      s_tensor_arena, kTensorArenaSize, &s_linear_memory_planner);
  if (!allocator) {
    ESP_LOGE(TAG, "Failed to create MicroAllocator with LinearMemoryPlanner");
    return ESP_FAIL;
  }

  s_interpreter = new (s_interpreter_buf)
      tflite::MicroInterpreter(s_model, s_op_resolver, allocator);

  TfLiteStatus allocate_status = s_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    ESP_LOGE(TAG, "AllocateTensors() failed");
    s_interpreter->~MicroInterpreter();
    s_interpreter = nullptr;
    s_kws_ready = false;
    return ESP_FAIL;
  }

  s_input_tensor = s_interpreter->input(0);
  s_output_tensor = s_interpreter->output(0);
  if (!s_input_tensor || !s_output_tensor) {
    ESP_LOGE(TAG, "Failed to get input/output tensors");
    s_interpreter->~MicroInterpreter();
    s_interpreter = nullptr;
    s_kws_ready = false;
    return ESP_FAIL;
  }

  // Find GlobalAveragePooling2D (Mean) embedding tensor index and quantization
  // parameters
  const auto *subgraphs = s_model->subgraphs();
  if (subgraphs && subgraphs->size() > 0) {
    const auto *sg = subgraphs->Get(0);
    const auto *tensors = sg->tensors();
    for (size_t i = 0; i < tensors->size(); i++) {
      const auto *t = tensors->Get(i);
      if (t && t->name()) {
        const char *name = t->name()->c_str();
        if (strstr(name, "global_average_pooling") || strstr(name, "Mean")) {
          s_embedding_tensor_idx = (int)i;
          if (t->quantization() && t->quantization()->scale() &&
              t->quantization()->scale()->size() > 0) {
            s_embedding_scale = t->quantization()->scale()->Get(0);
            s_embedding_zp = t->quantization()->zero_point()->Get(0);
          }
          ESP_LOGI(TAG,
                   "Identified Embedding Layer [%d]: %s (scale=%.6f, zp=%ld)",
                   (int)i, name, s_embedding_scale, (long)s_embedding_zp);
          break;
        }
      }
    }
  }

  // Initialize base pretrained weights from FlatBuffer
  init_factory_weights();

  // Check if Flash NVS has previous user-trained weights
  if (kws_engine_load_from_nvs() == ESP_OK) {
    ESP_LOGI(TAG,
             "Active on-device model loaded with %lu custom training samples",
             (unsigned long)s_train_sample_count);
  } else {
    ESP_LOGI(TAG, "Using base factory weights (no on-device training yet)");
  }

  s_kws_ready = true;

  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "TinyML KWS Engine Initialized (ESP32-P4)");
  ESP_LOGI(TAG, "Model: 2-Class Porjai+EdgeTTS [ช่วยด้วย / OTHER]");
  ESP_LOGI(TAG, "Input: dims=[%d,%d,%d,%d], scale=%.6f, zp=%d",
           s_input_tensor->dims->data[0], s_input_tensor->dims->data[1],
           s_input_tensor->dims->data[2], s_input_tensor->dims->data[3],
           s_input_tensor->params.scale, s_input_tensor->params.zero_point);
  ESP_LOGI(TAG, "Output: dims=[%d,%d], scale=%.6f, zp=%d",
           s_output_tensor->dims->data[0], s_output_tensor->dims->data[1],
           s_output_tensor->params.scale, s_output_tensor->params.zero_point);
  ESP_LOGI(TAG, "========================================");

  return ESP_OK;
}

extern "C" esp_err_t kws_engine_run_features(const int8_t *features,
                                             float *prob_help,
                                             float *prob_other) {
  if (!s_kws_ready || !s_interpreter || !s_input_tensor || !s_output_tensor) {
    ESP_LOGE(TAG, "KWS engine not initialized");
    return ESP_ERR_INVALID_STATE;
  }

  if (!features) {
    return ESP_ERR_INVALID_ARG;
  }

  // Copy features to model input tensor
  std::memcpy(s_input_tensor->data.int8, features, KWS_INPUT_FEATURES_LEN);

  // Invoke model
  if (s_interpreter->Invoke() != kTfLiteOk) {
    ESP_LOGE(TAG, "Model Invoke() failed");
    return ESP_FAIL;
  }

  // Pure Model Mode: Read direct Softmax output from TFLite model
  float scale = s_output_tensor->params.scale;
  int32_t zero_point = s_output_tensor->params.zero_point;

  int8_t raw_help = s_output_tensor->data.int8[KWS_LABEL_CHUEY_DUAY];
  int8_t raw_other = s_output_tensor->data.int8[KWS_LABEL_OTHER];

  float p_help = (raw_help - zero_point) * scale;
  float p_other = (raw_other - zero_point) * scale;

  p_help = std::max(0.0f, std::min(1.0f, p_help));
  p_other = std::max(0.0f, std::min(1.0f, p_other));

  float sum = p_help + p_other;
  if (sum > 1e-4f) {
    p_help /= sum;
    p_other /= sum;
  }

  if (prob_help)
    *prob_help = p_help;
  if (prob_other)
    *prob_other = p_other;

  return ESP_OK;
}

extern "C" esp_err_t kws_engine_train_step(int target_class,
                                           float *out_new_help_conf,
                                           float *out_new_other_conf) {
  if (!s_last_audio_embedding_valid) {
    ESP_LOGW(TAG, "No valid audio embedding to train on");
    return ESP_ERR_INVALID_STATE;
  }
  if (target_class < 0 || target_class >= KWS_NUM_CLASSES) {
    return ESP_ERR_INVALID_ARG;
  }

  // Current forward pass on the saved embedding
  float z0 = s_learned_bias[0];
  float z1 = s_learned_bias[1];
  for (int i = 0; i < KWS_EMBEDDING_DIM; i++) {
    z0 += s_learned_weights[0][i] * s_last_audio_embedding[i];
    z1 += s_learned_weights[1][i] * s_last_audio_embedding[i];
  }
  float diff = z0 - z1;
  if (diff > 16.0f)
    diff = 16.0f;
  if (diff < -16.0f)
    diff = -16.0f;
  float p0 = 1.0f / (1.0f + std::exp(-diff));
  float p1 = 1.0f - p0;

  // Target probabilities: class 0 = [1, 0], class 1 = [0, 1]
  float y0 = (target_class == 0) ? 1.0f : 0.0f;
  float y1 = (target_class == 1) ? 1.0f : 0.0f;

  // Gradient error: (target - prediction)
  float err0 = y0 - p0;
  float err1 = y1 - p1;

  // Learning rate: 0.08 for responsive personalization
  constexpr float kLearningRate = 0.08f;

  for (int i = 0; i < KWS_EMBEDDING_DIM; i++) {
    s_learned_weights[0][i] += kLearningRate * err0 * s_last_audio_embedding[i];
    s_learned_weights[1][i] += kLearningRate * err1 * s_last_audio_embedding[i];
  }
  s_learned_bias[0] += kLearningRate * err0;
  s_learned_bias[1] += kLearningRate * err1;

  s_train_sample_count++;
  s_weights_customized = true;

  // Recompute forward pass to evaluate post-training confidence
  z0 = s_learned_bias[0];
  z1 = s_learned_bias[1];
  for (int i = 0; i < KWS_EMBEDDING_DIM; i++) {
    z0 += s_learned_weights[0][i] * s_last_audio_embedding[i];
    z1 += s_learned_weights[1][i] * s_last_audio_embedding[i];
  }
  diff = z0 - z1;
  if (diff > 16.0f)
    diff = 16.0f;
  if (diff < -16.0f)
    diff = -16.0f;
  float new_p0 = 1.0f / (1.0f + std::exp(-diff));
  float new_p1 = 1.0f - new_p0;

  if (out_new_help_conf)
    *out_new_help_conf = new_p0;
  if (out_new_other_conf)
    *out_new_other_conf = new_p1;

  ESP_LOGI(TAG,
           "🧠 [ON-DEVICE TRAINED] Target=%s | Help: %.1f%% -> %.1f%% | Other: "
           "%.1f%% -> %.1f%% (Total Samples: %lu)",
           (target_class == 0) ? "1. ช่วยด้วย (HELP)" : "2. คำอื่นๆ (OTHER)",
           p0 * 100.0f, new_p0 * 100.0f, p1 * 100.0f, new_p1 * 100.0f,
           (unsigned long)s_train_sample_count);

  // Persist immediately to Flash NVS partition
  kws_engine_save_to_nvs();

  return ESP_OK;
}

#include "dsps_fft2r.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static inline float hz_to_mel(float hz) {
  return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
  return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

static void extract_mel_spectrogram(const int16_t *pcm, size_t total_samples,
                                    size_t window_offset, int8_t *features,
                                    float input_scale, int32_t input_zp) {
  // 512 Hann window
  static float s_hann_window[FFT_SIZE];
  static bool s_hann_inited = false;
  if (!s_hann_inited) {
    for (int i = 0; i < FFT_SIZE; i++) {
      s_hann_window[i] = 0.5f * (1.0f - std::cos(2.0f * (float)M_PI * (float)i /
                                                 (float)(FFT_SIZE - 1)));
    }
    s_hann_inited = true;
  }

  // Mel filterbank center frequencies (20 Hz - 8000 Hz)
  static int s_bin_points[KWS_SPECTROGRAM_BINS + 2];
  static bool s_bins_inited = false;
  if (!s_bins_inited) {
    float mel_min = hz_to_mel(20.0f);
    float mel_max = hz_to_mel((float)KWS_AUDIO_SAMPLE_RATE / 2.0f);
    float mel_step = (mel_max - mel_min) / (KWS_SPECTROGRAM_BINS + 1);
    for (int i = 0; i < KWS_SPECTROGRAM_BINS + 2; i++) {
      float hz = mel_to_hz(mel_min + i * mel_step);
      s_bin_points[i] =
          std::min(FFT_SIZE / 2, (int)std::round((float)FFT_SIZE * hz /
                                                 (float)KWS_AUDIO_SAMPLE_RATE));
    }
    s_bins_inited = true;
  }

  // Ensure DSP FFT tables are initialized
  if (!dsps_fft2r_initialized) {
    dsps_fft2r_init_fc32(NULL, FFT_SIZE);
  }

  for (int f = 0; f < KWS_SPECTROGRAM_FRAMES; f++) {
    size_t frame_start = window_offset + f * HOP_SIZE;

    for (int i = 0; i < FFT_SIZE; i++) {
      float sample = 0.0f;
      if (frame_start + i < total_samples) {
        sample = (float)pcm[frame_start + i] / 32768.0f;
      }
      s_fft_buf[2 * i] = sample * s_hann_window[i];
      s_fft_buf[2 * i + 1] = 0.0f;
    }

    // Compute FFT via hardware DSP
    dsps_fft2r_fc32(s_fft_buf, FFT_SIZE);
    dsps_bit_rev_fc32(s_fft_buf, FFT_SIZE);

    // Power spectrum for bins 0 .. 256
    for (int k = 0; k <= FFT_SIZE / 2; k++) {
      float r = s_fft_buf[2 * k];
      float im = s_fft_buf[2 * k + 1];
      s_power_spectrum[k] = (r * r + im * im);
    }

    // Apply 40 Mel triangular filterbanks
    for (int m = 0; m < KWS_SPECTROGRAM_BINS; m++) {
      float mel_energy = 0.0f;
      int start_bin = s_bin_points[m];
      int center_bin = s_bin_points[m + 1];
      int end_bin = s_bin_points[m + 2];

      for (int k = start_bin; k < center_bin && k <= FFT_SIZE / 2; k++) {
        float weight =
            (float)(k - start_bin) / std::max(1, center_bin - start_bin);
        mel_energy += s_power_spectrum[k] * weight;
      }
      for (int k = center_bin; k < end_bin && k <= FFT_SIZE / 2; k++) {
        float weight = (float)(end_bin - k) / std::max(1, end_bin - center_bin);
        mel_energy += s_power_spectrum[k] * weight;
      }

      // Log energy
      float log_energy = std::log(mel_energy + 1e-6f);

      // Quantize to INT8 for TFLM input tensor
      int q_val = (int)std::round(log_energy / input_scale) + input_zp;
      q_val = std::max(-128, std::min(127, q_val));

      features[f * KWS_SPECTROGRAM_BINS + m] = (int8_t)q_val;
    }
  }
}

extern "C" esp_err_t kws_engine_process_audio(const int16_t *pcm_samples,
                                              size_t num_samples,
                                              float *out_confidence,
                                              bool *out_is_help) {
  if (!pcm_samples || num_samples == 0) {
    return ESP_ERR_INVALID_ARG;
  }

  if (!s_kws_ready || !s_input_tensor || !s_output_tensor) {
    esp_err_t err = kws_engine_init();
    if (err != ESP_OK || !s_kws_ready || !s_input_tensor || !s_output_tensor) {
      ESP_LOGW(TAG, "KWS engine is not ready, skipping audio inference");
      return ESP_ERR_INVALID_STATE;
    }
  }

  float input_scale = s_input_tensor->params.scale;
  int32_t input_zp = s_input_tensor->params.zero_point;

  float max_prob_help = 0.0f;
  float min_prob_other = 1.0f;

  // 1-second window is 16,000 samples
  constexpr size_t WINDOW_SAMPLES = KWS_AUDIO_TOTAL_SAMPLES;
  constexpr size_t STEP_SAMPLES = 4000; // 250ms slide step

  size_t num_windows = 1;
  if (num_samples > WINDOW_SAMPLES) {
    num_windows = 1 + (num_samples - WINDOW_SAMPLES) / STEP_SAMPLES;
    if (num_windows > 5)
      num_windows = 5; // Cap at 5 windows
  }

  float best_window_emb[KWS_EMBEDDING_DIM];
  bool best_emb_found = false;

  for (size_t w = 0; w < num_windows; w++) {
    size_t window_offset =
        (num_samples > WINDOW_SAMPLES) ? (w * STEP_SAMPLES) : 0;
    extract_mel_spectrogram(pcm_samples, num_samples, window_offset, s_features,
                            input_scale, input_zp);

    float p_help = 0.0f;
    float p_other = 0.0f;
    esp_err_t ret = kws_engine_run_features(s_features, &p_help, &p_other);
    if (ret == ESP_OK) {
      if (!best_emb_found || p_help > max_prob_help) {
        max_prob_help = p_help;
        min_prob_other = p_other;
        std::memcpy(best_window_emb, s_last_audio_embedding,
                    sizeof(best_window_emb));
        best_emb_found = true;
      }
    }
  }

  if (best_emb_found) {
    std::memcpy(s_last_audio_embedding, best_window_emb,
                sizeof(s_last_audio_embedding));
    s_last_audio_embedding_valid = true;
  }

  if (out_confidence)
    *out_confidence = max_prob_help;
  if (out_is_help)
    *out_is_help = (max_prob_help >= KWS_DEFAULT_THRESHOLD);

  ESP_LOGI(TAG,
           "KWS Result -> Help (ช่วยด้วย): %.1f%%, Other: %.1f%% (%u window%s) | "
           "%s [Learned samples: %lu]",
           max_prob_help * 100.0f, min_prob_other * 100.0f,
           (unsigned)num_windows, (num_windows > 1) ? "s" : "",
           (max_prob_help >= KWS_DEFAULT_THRESHOLD) ? "EMERGENCY DETECTED!"
                                                    : "NORMAL",
           (unsigned long)s_train_sample_count);

  return ESP_OK;
}

extern "C" void kws_engine_deinit(void) {
  if (s_interpreter) {
    s_interpreter->~MicroInterpreter();
    s_interpreter = nullptr;
  }
  if (s_tensor_arena) {
    free(s_tensor_arena);
    s_tensor_arena = nullptr;
  }
  s_model = nullptr;
  s_input_tensor = nullptr;
  s_output_tensor = nullptr;
  s_kws_ready = false;
  ESP_LOGI(TAG, "KWS engine deinitialized");
}
