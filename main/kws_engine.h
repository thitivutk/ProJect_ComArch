#ifndef KWS_ENGINE_H
#define KWS_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =========================================================
 * TinyML Keyword Spotting (KWS) Configuration
 * Model: 2-Class DS-CNN (Porjai + Edge-TTS)
 * Classes: [0: "ช่วยด้วย (HELP)", 1: "OTHER"]
 * =========================================================
 */

#define KWS_AUDIO_SAMPLE_RATE   16000
#define KWS_AUDIO_DURATION_SEC  1
#define KWS_AUDIO_TOTAL_SAMPLES (KWS_AUDIO_SAMPLE_RATE * KWS_AUDIO_DURATION_SEC) // 16,000 samples

#define KWS_SPECTROGRAM_FRAMES  49
#define KWS_SPECTROGRAM_BINS    40
#define KWS_INPUT_FEATURES_LEN  (KWS_SPECTROGRAM_FRAMES * KWS_SPECTROGRAM_BINS) // 1,960 elements

#define KWS_DEFAULT_THRESHOLD   0.75f  // Probability threshold to trigger HELP (0.0 - 1.0)

/**
 * @brief Initialize TensorFlow Lite Micro KWS engine.
 *        Allocates Tensor Arena and sets up interpreter.
 * 
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t kws_engine_init(void);

/**
 * @brief Run inference on pre-extracted INT8 feature map (49x40).
 * 
 * @param features Pointer to INT8 features array of length 1,960.
 * @param prob_help Pointer to store probability of "ช่วยด้วย" (0.0 to 1.0).
 * @param prob_other Pointer to store probability of "OTHER" (0.0 to 1.0).
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t kws_engine_run_features(const int8_t *features, float *prob_help, float *prob_other);

/**
 * @brief End-to-end processing: Extract Mel-spectrogram features from 16kHz PCM audio
 *        and run TFLM inference.
 * 
 * @param pcm_samples Pointer to raw 16kHz 16-bit mono PCM audio.
 * @param num_samples Number of samples (ideally 16,000 for 1 second).
 * @param out_confidence Output confidence of "ช่วยด้วย" (0.0 to 1.0).
 * @param out_is_help Output flag, true if confidence >= KWS_DEFAULT_THRESHOLD.
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t kws_engine_process_audio(const int16_t *pcm_samples, size_t num_samples, 
                                   float *out_confidence, bool *out_is_help);

/**
 * @brief Perform an on-device training step using the audio features from the latest
 *        kws_engine_process_audio() run.
 * 
 * @param target_class Target label: 0 for "ช่วยด้วย" (HELP), 1 for "คำอื่นๆ" (OTHER).
 * @param out_new_help_conf Output updated confidence of "ช่วยด้วย" (0.0 to 1.0).
 * @param out_new_other_conf Output updated confidence of "คำอื่นๆ" (0.0 to 1.0).
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t kws_engine_train_step(int target_class, float *out_new_help_conf, float *out_new_other_conf);

/**
 * @brief Save current learned weights to Flash NVS partition.
 *        Persists across reboots and firmware re-flashing!
 * @return ESP_OK on success, error code otherwise.
 */
esp_err_t kws_engine_save_to_nvs(void);

/**
 * @brief Load learned weights from Flash NVS partition.
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if none saved, error code otherwise.
 */
esp_err_t kws_engine_load_from_nvs(void);

/**
 * @brief Reset weights back to factory pretrained model and erase NVS saved weights.
 * @return ESP_OK on success.
 */
esp_err_t kws_engine_reset_weights(void);

/**
 * @brief Get total number of on-device training samples performed.
 */
uint32_t kws_engine_get_train_count(void);

/**
 * @brief Release KWS engine resources.
 */
void kws_engine_deinit(void);

#ifdef __cplusplus
}
#endif

#endif // KWS_ENGINE_H
