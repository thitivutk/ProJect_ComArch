#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "audio_processor.h"
#include "http_client.h"
#include "microphone.h"
#include "stt_client.h"
#include "wifi_manager.h"

#include "camera.h"
#include "display.h"
#include "alert_recorder.h"
#include "benchmark.h"
#include "kws_engine.h"

static const char *TAG = "MAIN";

/*
 * =========================================================
 * Recording configuration
 * =========================================================
 */

#define MAX_RECORDING_BYTES                                                    \
  (AUDIO_SAMPLE_RATE * (AUDIO_MAX_RECORDING_MS / 1000) * sizeof(int16_t))

/*
 * =========================================================
 * Application
 * =========================================================
 */

void app_main(void) {
  printf("\n");
  printf("========================================\n");
  printf("       ESP32-P4 Voice + Camera\n");
  printf("========================================\n\n");

  esp_err_t ret;

  /*
   * =====================================================
   * 1. Initialize DISPLAY
   * =====================================================
   */

  ESP_LOGI(TAG, "Initializing display...");

  ret = display_init();

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Display initialization failed: %s", esp_err_to_name(ret));

    return;
  }

  ESP_LOGI(TAG, "Display initialization PASSED");

  /*
   * =====================================================
   * 2. Initialize CAMERA
   * =====================================================
   *
   * camera_init() จะ:
   *
   * - start BSP camera
   * - เปิด V4L2
   * - setup camera buffers
   * - start V4L2 stream
   * - สร้าง camera task
   * - รับ frame ต่อเนื่อง
   * - scale ด้วย PPA
   * - ส่ง frame ไป display
   *
   * ดังนั้น main task ไม่ต้องรับ frame เอง
   */

  ESP_LOGI(TAG, "Initializing camera...");

  ret = camera_init();

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Camera initialization failed: %s", esp_err_to_name(ret));

    display_deinit();

    return;
  }

  ESP_LOGI(TAG, "Camera initialization PASSED");

  ESP_LOGI(TAG, "========================================");

  ESP_LOGI(TAG, "CAMERA + DISPLAY RUNNING");

  ESP_LOGI(TAG, "Camera starts automatically");

  ESP_LOGI(TAG, "No Camera App required");

  ESP_LOGI(TAG, "========================================");

  /*
   * =====================================================
   * 3. Initialize WiFi
   * =====================================================
   */

  ESP_LOGI(TAG, "Initializing WiFi...");

  bool wifi_ready = false;
  ret = wifi_manager_init();

  if (ret != ESP_OK) {
    ESP_LOGW(TAG, "WiFi initialization failed (%s). Continuing in Offline TinyML mode...", esp_err_to_name(ret));
  } else {
    /*
     * 4. Start WiFi
     */
    ESP_LOGI(TAG, "Starting WiFi...");
    ret = wifi_manager_start();

    if (ret != ESP_OK) {
      ESP_LOGW(TAG, "WiFi start failed (%s). Continuing in Offline TinyML mode...", esp_err_to_name(ret));
    } else {
      /*
       * 5. Wait for WiFi
       */
      ESP_LOGI(TAG, "Waiting for WiFi connection...");
      ret = wifi_manager_wait_connected(15000);

      if (ret != ESP_OK) {
        ESP_LOGW(TAG, "WiFi connection timeout (%s), continuing with On-Device AI...", esp_err_to_name(ret));
      } else {
        ESP_LOGI(TAG, "WiFi connection established!");
        wifi_ready = true;
      }
    }
  }

  /*
   * =====================================================
   * 6. Internet test (Only if WiFi is ready)
   * =====================================================
   */
  if (wifi_ready) {
    ESP_LOGI(TAG, "Starting Internet test...");
    ret = http_client_test();
    if (ret == ESP_OK) {
      ESP_LOGI(TAG, "Internet test PASSED");
    } else {
      ESP_LOGW(TAG, "Internet test FAILED: %s", esp_err_to_name(ret));
    }
  } else {
    ESP_LOGW(TAG, "WiFi not connected, skipping Internet test");
  }

  /*
   * =====================================================
   * 7. Initialize microphone
   * =====================================================
   */

  ESP_LOGI(TAG, "Initializing microphone...");

  ret = microphone_init();

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Microphone initialization failed: %s", esp_err_to_name(ret));

    camera_deinit();
    display_deinit();

    return;
  }

  ESP_LOGI(TAG, "Microphone initialization PASSED");

  /*
   * =====================================================
   * 8. Initialize audio processor
   * =====================================================
   */

  ESP_LOGI(TAG, "Initializing audio processor...");

  ret = audio_processor_init();

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Audio processor initialization failed: %s",
             esp_err_to_name(ret));

    microphone_deinit();
    camera_deinit();
    display_deinit();

    return;
  }

  ESP_LOGI(TAG, "Audio processor initialization PASSED");

  /*
   * =====================================================
   * 9. Initialize STT client (Data Collection Mode)
   * =====================================================
   */
  ESP_LOGI(TAG, "Initializing STT client for data collection...");
  ret = stt_client_init();
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "STT client initialization failed: %s", esp_err_to_name(ret));
    microphone_deinit();
    camera_deinit();
    display_deinit();
    return;
  }
  ESP_LOGI(TAG, "STT client initialization PASSED");

  /*
   * =====================================================
   * 9.1 Initialize TinyML KWS Engine (On-Device "ช่วยด้วย")
   * =====================================================
   */
  ESP_LOGI(TAG, "Initializing TinyML KWS Engine...");
  ret = kws_engine_init();
  if (ret != ESP_OK) {
    ESP_LOGW(TAG, "KWS engine init failed: %s (will rely on STT Server)", esp_err_to_name(ret));
  } else {
    ESP_LOGI(TAG, "TinyML KWS Engine PASSED");
  }

  /*
   * =====================================================
   * 10. VAD configuration
   * =====================================================
   */

  ESP_LOGI(TAG, "========================================");

  ESP_LOGI(TAG, "Voice Activity Detection");

  ESP_LOGI(TAG, "Sample rate       : %d Hz", AUDIO_SAMPLE_RATE);

  ESP_LOGI(TAG, "Frame size        : %d samples", AUDIO_FRAME_SAMPLES);

  ESP_LOGI(TAG, "Frame duration    : %d ms",
           (AUDIO_FRAME_SAMPLES * 1000) / AUDIO_SAMPLE_RATE);

  ESP_LOGI(TAG, "VAD threshold     : %.2f", AUDIO_VAD_THRESHOLD);

  ESP_LOGI(TAG, "Silence timeout   : %d ms", AUDIO_SILENCE_TIMEOUT_MS);

  ESP_LOGI(TAG, "Maximum recording : %d ms", AUDIO_MAX_RECORDING_MS);

  ESP_LOGI(TAG, "Maximum buffer    : %d bytes", MAX_RECORDING_BYTES);

  ESP_LOGI(TAG, "========================================");

  /*
   * =====================================================
   * 11. Allocate recording buffer
   * =====================================================
   */

  int16_t *recording_buffer =
      (int16_t *)heap_caps_malloc(MAX_RECORDING_BYTES, MALLOC_CAP_8BIT);

  if (recording_buffer == NULL) {
    ESP_LOGE(TAG, "Failed to allocate recording buffer (%d bytes)",
             MAX_RECORDING_BYTES);

    microphone_deinit();
    camera_deinit();
    display_deinit();

    return;
  }

  ESP_LOGI(TAG, "Recording buffer allocated: %d bytes", MAX_RECORDING_BYTES);

  /*
   * =====================================================
   * 12. Audio frame buffers
   * =====================================================
   */

  int16_t audio_buffer[AUDIO_FRAME_SAMPLES];

  int16_t filtered_buffer[AUDIO_FRAME_SAMPLES];

  size_t bytes_read = 0;

  /*
   * =====================================================
   * 13. STT output buffer
   * =====================================================
   */

  char recognized_text[256];

  /*
   * =====================================================
   * 14. VAD state & Pre-roll Ring Buffer
   * =====================================================
   */

  bool recording = false;

  size_t recorded_bytes = 0;

  int64_t speech_start_time = 0;

  int64_t last_speech_time = 0;

  /*
   * Pre-roll ring buffer (8 frames = 256 ms)
   * เก็บเสียงช่วงก่อนหน้า VAD trigger ไม่ให้พยัญชนะต้นคำขาดหาย
   */
  static int16_t pre_roll_buffer[AUDIO_PREROLL_FRAMES][AUDIO_FRAME_SAMPLES];

  size_t pre_roll_index = 0;

  size_t pre_roll_count = 0;

  /*
   * =====================================================
   * 15. Continuous VAD
   * =====================================================
   *
   * Camera:
   *
   *     camera task
   *
   * Voice:
   *
   *     main task
   *
   * ทั้งสองทำงานพร้อมกัน
   */

  ESP_LOGI(TAG, "========================================");

  ESP_LOGI(TAG, "System ready");

  ESP_LOGI(TAG, "Camera + Display : RUNNING");

  ESP_LOGI(TAG, "WiFi             : CONNECTED");

  ESP_LOGI(TAG, "Voice            : READY");

  ESP_LOGI(TAG, "Waiting for speech...");

  ESP_LOGI(TAG, "========================================");

  /*
   * Start alert recorder after all systems are initialized and ready
   */
  alert_recorder_init();

  /*
   * Elevate Voice/VAD task priority to 12 (Hard Real-Time I/O stream)
   */
  vTaskPrioritySet(NULL, 12);

  /*
   * Single-thread vs multi-thread benchmark (~70 s) - DISABLED
   */
  // #if BENCH_ENABLE
  // bench_start();
  // #endif


  int64_t last_read_us = 0;

  /*
   * =====================================================
   * Main voice loop
   * =====================================================
   */

  while (1) {
    /*
     * -------------------------------------------------
     * Read microphone
     * -------------------------------------------------
     */

    ret = microphone_read(audio_buffer, sizeof(audio_buffer), &bytes_read);

    int64_t read_done_us = esp_timer_get_time();

    if (last_read_us > 0) {
      bench_record(BENCH_M_AUDIO_PERIOD, read_done_us - last_read_us);
    }

    last_read_us = read_done_us;

    if (ret != ESP_OK) {
      ESP_LOGE(TAG, "Microphone read failed: %s", esp_err_to_name(ret));

      break;
    }

    /*
     * Ignore empty microphone frames.
     */

    if (bytes_read == 0) {
      vTaskDelay(pdMS_TO_TICKS(1));

      continue;
    }

    /*
     * -------------------------------------------------
     * VAD
     * -------------------------------------------------
     */

    float rms = 0.0f;

    bool speech =
        audio_processor_is_speech(audio_buffer, AUDIO_FRAME_SAMPLES, &rms);

    /*
     * -------------------------------------------------
     * Audio processing
     * -------------------------------------------------
     */

    float energy = 0.0f;

    ret = audio_processor_process(audio_buffer, filtered_buffer,
                                  AUDIO_FRAME_SAMPLES, &energy);

    if (ret != ESP_OK) {
      ESP_LOGE(TAG, "Audio processing failed: %s", esp_err_to_name(ret));

      break;
    }

    bench_record(BENCH_M_AUDIO_PROC, esp_timer_get_time() - read_done_us);

#if BENCH_ENABLE
    /*
     * -------------------------------------------------
     * Benchmark running: fixed synthetic workload
     * instead of the speech state machine
     * -------------------------------------------------
     */

    if (bench_is_running()) {
      bench_feed_audio(audio_buffer, AUDIO_FRAME_SAMPLES);

      bench_step();

      if (!bench_is_running()) {
        recording = false;
        recorded_bytes = 0;
        pre_roll_count = 0;
        pre_roll_index = 0;

        ESP_LOGI(TAG, "Waiting for speech...");
      }

      vTaskDelay(pdMS_TO_TICKS(1));

      continue;
    }
#endif

    /*
     * -------------------------------------------------
     * Current time
     * -------------------------------------------------
     */

    int64_t now = esp_timer_get_time() / 1000;

    /*
     * =================================================
     * STATE 1
     *
     * Waiting for speech
     * =================================================
     */

    if (!recording) {
      if (speech) {
        recording = true;

        speech_start_time = now;

        last_speech_time = now;

        recorded_bytes = 0;

        /*
         * Copy pre-roll buffer into recording_buffer
         * เพื่อให้ได้เสียงพยัญชนะ/สระต้นคำที่พูดเร็ว (ย้อนหลัง ~256ms)
         */
        for (size_t i = 0; i < pre_roll_count; i++) {
          size_t idx =
              (pre_roll_index + AUDIO_PREROLL_FRAMES - pre_roll_count + i) %
              AUDIO_PREROLL_FRAMES;

          if (recorded_bytes + AUDIO_FRAME_BYTES <= MAX_RECORDING_BYTES) {
            memcpy(((uint8_t *)recording_buffer) + recorded_bytes,

                   pre_roll_buffer[idx],

                   AUDIO_FRAME_BYTES);

            recorded_bytes += AUDIO_FRAME_BYTES;
          }
        }

        pre_roll_count = 0;
        pre_roll_index = 0;

        ESP_LOGI(TAG, "========================================");

        ESP_LOGI(TAG, "Speech detected! (Pre-roll: %u bytes)",
                 (unsigned)recorded_bytes);

        ESP_LOGI(TAG, "Starting recording...");

        ESP_LOGI(TAG, "RMS = %.2f", rms);

        ESP_LOGI(TAG, "========================================");
      } else {
        /*
         * บันทึก frame ความเงียบลง pre-roll ring buffer ไว้ล่วงหน้า
         */
        memcpy(pre_roll_buffer[pre_roll_index], audio_buffer,
               sizeof(audio_buffer));

        pre_roll_index = (pre_roll_index + 1) % AUDIO_PREROLL_FRAMES;

        if (pre_roll_count < AUDIO_PREROLL_FRAMES) {
          pre_roll_count++;
        }

        static int64_t last_log = 0;

        if ((now - last_log) >= 2000) {
          ESP_LOGI(TAG, "Waiting for speech... RMS=%.2f", rms);

          last_log = now;
        }
      }
    }

    /*
     * =================================================
     * STATE 2
     *
     * Recording
     * =================================================
     */

    if (recording) {
      /*
       * -------------------------------------------------
       * Check maximum recording time
       * -------------------------------------------------
       */

      int64_t recording_time = now - speech_start_time;

      if (recording_time >= AUDIO_MAX_RECORDING_MS) {
        ESP_LOGW(TAG, "Recording exceeded %d ms", AUDIO_MAX_RECORDING_MS);

        ESP_LOGW(TAG, "Discarding audio...");

        recording = false;

        recorded_bytes = 0;

        pre_roll_count = 0;
        pre_roll_index = 0;

        ESP_LOGI(TAG, "Waiting for speech...");

        continue;
      }

      /*
       * -------------------------------------------------
       * Store raw PCM
       * -------------------------------------------------
       */

      if (recorded_bytes + bytes_read <= MAX_RECORDING_BYTES) {
        memcpy(((uint8_t *)recording_buffer) + recorded_bytes,

               audio_buffer,

               bytes_read);

        recorded_bytes += bytes_read;
      } else {
        ESP_LOGW(TAG, "Recording buffer full");

        recording = false;

        recorded_bytes = 0;

        pre_roll_count = 0;
        pre_roll_index = 0;

        continue;
      }

      /*
       * -------------------------------------------------
       * Speech detected
       * -------------------------------------------------
       */

      if (speech) {
        last_speech_time = now;
      }

      /*
       * -------------------------------------------------
       * Silence duration
       * -------------------------------------------------
       */

      int64_t silence_time = now - last_speech_time;

      /*
       * -------------------------------------------------
       * Silence timeout
       * -------------------------------------------------
       */

      if (silence_time >= AUDIO_SILENCE_TIMEOUT_MS) {
        recording = false;

        float duration =
            ((float)recorded_bytes) / (AUDIO_SAMPLE_RATE * sizeof(int16_t));

        ESP_LOGI(TAG, "========================================");

        ESP_LOGI(TAG, "Speech finished");

        ESP_LOGI(TAG, "Silence duration : %lld ms", silence_time);

        ESP_LOGI(TAG, "Audio size       : %u bytes", (unsigned)recorded_bytes);

        ESP_LOGI(TAG, "Audio duration   : %.2f sec", duration);

        ESP_LOGI(TAG, "========================================");

        /*
         * =================================================
         * 1. On-Device TinyML Inference (Display Model %)
         * =================================================
         */
        float kws_confidence = 0.0f;
        bool kws_is_help = false;
        kws_engine_process_audio((const int16_t *)recording_buffer,
                                 recorded_bytes / sizeof(int16_t),
                                 &kws_confidence, &kws_is_help);

        ESP_LOGI(TAG, "--------------------------------------------------------");
        ESP_LOGI(TAG, "🤖 [ON-DEVICE TINYML] ผลการวิเคราะห์เสียง (โมเดล 10 คน):");
        ESP_LOGI(TAG, "   - ช่วยด้วย (Help) : %.1f%%", kws_confidence * 100.0f);
        ESP_LOGI(TAG, "   - คำอื่นๆ (Other) : %.1f%%", (1.0f - kws_confidence) * 100.0f);
        
        if (kws_is_help) {
            ESP_LOGW(TAG, "🚨🚨🚨 [EMERGENCY DETECTED] ตรวจพบคำว่า: \"ช่วยด้วย (HELP)\" (ความมั่นใจ: %.1f%%) 🚨🚨🚨",
                     kws_confidence * 100.0f);
            alert_recorder_trigger();
        } else {
            ESP_LOGI(TAG, "🟢 [SPEECH] จัดเป็น: \"คำอื่นๆ (OTHER)\" (ความมั่นใจ: %.1f%%)",
                     (1.0f - kws_confidence) * 100.0f);
        }
        ESP_LOGI(TAG, "--------------------------------------------------------");

#if 0
        /* Data collection mode disabled - running pure on-device KWS testing */
        char recognized_text[256];
        memset(recognized_text, 0, sizeof(recognized_text));
        stt_client_transcribe((const uint8_t *)recording_buffer, recorded_bytes, recognized_text, sizeof(recognized_text));
#endif

        /*
         * -------------------------------------------------
         * Reset recording buffer
         * -------------------------------------------------
         */

        recorded_bytes = 0;

        pre_roll_count = 0;
        pre_roll_index = 0;

        ESP_LOGI(TAG, "Waiting for speech...");
      }
    }

    /*
     * -------------------------------------------------
     * Small delay
     * -------------------------------------------------
     */

    vTaskDelay(pdMS_TO_TICKS(1));
  }

  /*
   * =====================================================
   * Cleanup
   * =====================================================
   */

  ESP_LOGW(TAG, "Main application loop stopped");

  /*
   * Free recording buffer.
   *
   * IMPORTANT:
   *
   * The buffer was allocated with
   * heap_caps_malloc(), therefore use
   * heap_caps_free().
   */

  if (recording_buffer != NULL) {
    heap_caps_free(recording_buffer);

    recording_buffer = NULL;
  }

  /*
   * Stop microphone.
   */

  microphone_deinit();

  /*
   * Stop camera.
   */

  camera_deinit();

  /*
   * Stop display.
   */

  display_deinit();

  /*
   * =====================================================
   * Keep application alive
   * =====================================================
   */

  while (1) {
    vTaskDelay(pdMS_TO_TICKS(10000));
  }
}