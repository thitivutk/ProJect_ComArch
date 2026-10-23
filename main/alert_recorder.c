#include "alert_recorder.h"
#include "benchmark.h"
#include "camera.h"
#include "display.h"
#include "driver/jpeg_encode.h"
#include "esp_heap_caps.h"
#include "esp_http_client.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

static const char *TAG = "ALERT_REC";

#define ALERT_SERVER_URL "http://172.20.10.13:5000/upload_alert_frame"

/* Continuous 21 JPEG Frames in PSRAM:
 * 10 Pre-Frames + 1 Trigger Frame + 10 Post-Frames
 * Total RAM: 21 * 128 KB = ~2.6 MB (Compared to ~25.6 MB if raw 1024x600!) */
static alert_frame_t pre_buffers[ALERT_PRE_FRAMES_COUNT];
static alert_frame_t trigger_buffer;
static alert_frame_t post_buffers[ALERT_POST_FRAMES_COUNT];

/* Freeze snapshot of pre-frames head/count when triggered */
static int frozen_pre_head = 0;
static int frozen_pre_count = 0;

static int pre_head = 0;
static int pre_count = 0;
static volatile bool alert_ready = false;
static volatile bool alert_triggered = false;
static volatile bool post_capture_active = false;
static int64_t current_alert_id = 0;

#define ALERT_CAPTURE_PERIOD_MS 100

static TaskHandle_t capture_task_handle = NULL;
static TaskHandle_t upload_task_handle = NULL;

/* Capture task stop request (benchmark switches it between cores) */
static volatile bool capture_task_stop = false;
static SemaphoreHandle_t capture_stop_sem = NULL;

/* Next post-frame slot while post_capture_active */
static int post_index = 0;
static int64_t last_capture_us = 0;

static jpeg_encoder_handle_t jpeg_enc_handle = NULL;
static jpeg_encode_cfg_t jpeg_enc_cfg;

/* Helper: Hardware JPEG encode from RGB565 buffer */
static bool encode_rgb565_to_jpeg(const uint8_t *src_rgb565, int width,
                                  int height, uint8_t *dst_jpeg, size_t dst_max,
                                  size_t *out_len) {
  if (jpeg_enc_handle == NULL || src_rgb565 == NULL || dst_jpeg == NULL ||
      out_len == NULL) {
    return false;
  }

  jpeg_enc_cfg.width = width;
  jpeg_enc_cfg.height = height;

  uint32_t src_size = (uint32_t)width * (uint32_t)height * 2;
  uint32_t codeced_size = 0;

  esp_err_t err = jpeg_encoder_process(jpeg_enc_handle, &jpeg_enc_cfg,
                                       (uint8_t *)src_rgb565, src_size,
                                       dst_jpeg, dst_max, &codeced_size);
  if (err == ESP_OK && codeced_size > 0) {
    *out_len = (size_t)codeced_size;
    return true;
  } else {
    ESP_LOGE(TAG, "Hardware JPEG encode failed: %s (size=%lu)",
             esp_err_to_name(err), (unsigned long)codeced_size);
    return false;
  }
}

/* HTTP upload helper using persistent client */
static bool send_frame_http(esp_http_client_handle_t client, int64_t alert_id,
                            int index, int total, const char *type,
                            const uint8_t *data, size_t len) {
  if (client == NULL || data == NULL || len == 0)
    return false;

  char val_buf[32];
  snprintf(val_buf, sizeof(val_buf), "%lld", alert_id);
  esp_http_client_set_header(client, "X-Alert-ID", val_buf);

  snprintf(val_buf, sizeof(val_buf), "%d", index);
  esp_http_client_set_header(client, "X-Frame-Index", val_buf);

  snprintf(val_buf, sizeof(val_buf), "%d", total);
  esp_http_client_set_header(client, "X-Frame-Total", val_buf);

  esp_http_client_set_header(client, "X-Frame-Type", type);

  snprintf(val_buf, sizeof(val_buf), "%d", ALERT_WIDTH);
  esp_http_client_set_header(client, "X-Frame-Width", val_buf);

  snprintf(val_buf, sizeof(val_buf), "%d", ALERT_HEIGHT);
  esp_http_client_set_header(client, "X-Frame-Height", val_buf);

  esp_http_client_set_header(client, "Content-Type", "image/jpeg");
  esp_http_client_set_post_field(client, (const char *)data, len);

  esp_err_t err = esp_http_client_perform(client);
  if (err == ESP_OK) {
    int status = esp_http_client_get_status_code(client);
    if (status == 200) {
      ESP_LOGI(TAG, "[ALERT] Frame %2d/%d (%s, %u bytes) sent OK", index + 1,
               total, type, (unsigned)len);
      return true;
    } else {
      ESP_LOGW(TAG, "[ALERT] Frame %d upload HTTP status %d", index + 1,
               status);
    }
  } else {
    ESP_LOGE(TAG, "[ALERT] Frame %d upload failed: %s", index + 1,
             esp_err_to_name(err));
  }
  return false;
}

/* Background upload task - uploads ALL 21 pre-recorded continuous JPEG frames
 */
static void alert_upload_task(void *arg) {
  while (1) {
    /* Direct-to-Task Notification: Wait until signaled by alert_capture_task */
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    int64_t t_start = esp_timer_get_time() / 1000;
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "[ALERT] Uploading 21 continuous JPEG frames (ID: %lld)...",
             current_alert_id);
    ESP_LOGI(TAG, "========================================");

    esp_http_client_config_t config = {
        .url = ALERT_SERVER_URL,
        .method = HTTP_METHOD_POST,
        .timeout_ms = 4000,
        .keep_alive_enable = true,
    };

    esp_http_client_handle_t client = esp_http_client_init(&config);
    if (client == NULL) {
      ESP_LOGE(TAG, "Failed to initialize HTTP client for alert batch");
      alert_triggered = false;
      continue;
    }

    int sent_count = 0;
    int frame_idx = 0;

    /* 1. Upload 10 Pre-Frames (in chronological order) */
    for (int i = 0; i < ALERT_PRE_FRAMES_COUNT; i++) {
      int slot =
          (frozen_pre_head + ALERT_PRE_FRAMES_COUNT - frozen_pre_count + i) %
          ALERT_PRE_FRAMES_COUNT;
      if (pre_buffers[slot].data != NULL && pre_buffers[slot].len > 0) {
        if (send_frame_http(client, current_alert_id, frame_idx++,
                            ALERT_TOTAL_COUNT, "pre", pre_buffers[slot].data,
                            pre_buffers[slot].len)) {
          sent_count++;
        }
      }
    }

    /* 2. Upload Trigger Frame */
    if (trigger_buffer.data != NULL && trigger_buffer.len > 0) {
      if (send_frame_http(client, current_alert_id, frame_idx++,
                          ALERT_TOTAL_COUNT, "trigger", trigger_buffer.data,
                          trigger_buffer.len)) {
        sent_count++;
      }
    }

    /* 3. Upload 10 Post-Frames (already recorded continuously in PSRAM) */
    for (int i = 0; i < ALERT_POST_FRAMES_COUNT; i++) {
      if (post_buffers[i].data != NULL && post_buffers[i].len > 0) {
        if (send_frame_http(client, current_alert_id, frame_idx++,
                            ALERT_TOTAL_COUNT, "post", post_buffers[i].data,
                            post_buffers[i].len)) {
          sent_count++;
        }
      }
    }

    esp_http_client_cleanup(client);

    int64_t elapsed_ms = (esp_timer_get_time() / 1000) - t_start;
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "[ALERT COMPLETE] Sent %d/%d JPEG frames in %lld ms!",
             sent_count, ALERT_TOTAL_COUNT, elapsed_ms);
    ESP_LOGI(TAG, "========================================");

    alert_triggered = false; // Resume normal pre-roll buffer
  }
}

/* Encode one frame and record the encode time for the benchmark */
static bool capture_jpeg(const uint8_t *frame, int w, int h,
                         alert_frame_t *slot) {
  int64_t t0 = esp_timer_get_time();
  bool ok = encode_rgb565_to_jpeg(frame, w, h, slot->data,
                                  ALERT_JPEG_MAX_BYTES, &slot->len);
  bench_record(BENCH_M_ALERT_JPEG, esp_timer_get_time() - t0);
  return ok;
}

/* One capture step, run every ALERT_CAPTURE_PERIOD_MS:
 * - When idle: encodes one rolling pre-frame to JPEG
 * - When triggered: encodes the next of 10 post-frames; after the last one
 *   the upload task is woken */
static void alert_capture_step(void) {
  if (!alert_ready) {
    return;
  }

  int64_t now = esp_timer_get_time();
  if (last_capture_us > 0) {
    bench_record(BENCH_M_ALERT_PERIOD, now - last_capture_us);
  }
  last_capture_us = now;

  int disp_w = display_get_width();
  int disp_h = display_get_height();
  if (disp_w <= 0)
    disp_w = ALERT_WIDTH;
  if (disp_h <= 0)
    disp_h = ALERT_HEIGHT;

  if (post_capture_active) {
    /* Record 10 post-frames continuously right after trigger */
    const uint8_t *frame = camera_get_frame();
    if (frame != NULL && post_buffers[post_index].data != NULL) {
      capture_jpeg(frame, disp_w, disp_h, &post_buffers[post_index]);
    }

    post_index++;
    if (post_index >= ALERT_POST_FRAMES_COUNT) {
      post_index = 0;
      post_capture_active = false;

      /* All 21 continuous JPEG frames recorded in PSRAM! Wake up upload task
       * via Direct Notification */
      if (upload_task_handle != NULL) {
        xTaskNotifyGive(upload_task_handle);
      }
    }
  } else if (!alert_triggered) {
    /* Rolling pre-frames */
    const uint8_t *frame = camera_get_frame();
    if (frame != NULL && pre_buffers[pre_head].data != NULL) {
      if (capture_jpeg(frame, disp_w, disp_h, &pre_buffers[pre_head])) {
        pre_head = (pre_head + 1) % ALERT_PRE_FRAMES_COUNT;
        if (pre_count < ALERT_PRE_FRAMES_COUNT) {
          pre_count++;
        }
      }
    }
  }
}

/* Continuous capture task: one step every 100 ms (10 FPS) */
static void alert_capture_task(void *arg) {
  while (!capture_task_stop) {
    alert_capture_step();
    vTaskDelay(pdMS_TO_TICKS(ALERT_CAPTURE_PERIOD_MS));
  }

  capture_task_handle = NULL;
  xSemaphoreGive(capture_stop_sem);
  vTaskDelete(NULL);
}

esp_err_t alert_recorder_task_start(int core_id) {
  if (capture_task_handle != NULL) {
    return ESP_OK;
  }

  capture_task_stop = false;
  xSemaphoreTake(capture_stop_sem, 0);

  if (xTaskCreatePinnedToCore(alert_capture_task, "alert_cap", 4096, NULL, 5,
                              &capture_task_handle, core_id) != pdPASS) {
    capture_task_handle = NULL;
    ESP_LOGE(TAG, "Failed to create alert capture task");
    return ESP_FAIL;
  }

  return ESP_OK;
}

void alert_recorder_task_stop(void) {
  if (capture_task_handle == NULL) {
    return;
  }

  capture_task_stop = true;
  if (xSemaphoreTake(capture_stop_sem, pdMS_TO_TICKS(1000)) != pdTRUE) {
    ESP_LOGE(TAG, "Alert capture task did not stop");
    return;
  }
  capture_task_stop = false;
}

void alert_recorder_poll(void) {
  if (capture_task_handle != NULL) {
    return;
  }

  int64_t now = esp_timer_get_time();
  if (last_capture_us == 0 ||
      now - last_capture_us >= ALERT_CAPTURE_PERIOD_MS * 1000LL) {
    alert_capture_step();
  }
}

esp_err_t alert_recorder_init(void) {
  ESP_LOGI(TAG, "========================================");
  ESP_LOGI(TAG, "Initializing Hardware JPEG Alert Recorder (1024x600)...");

  /* 1. Initialize Hardware JPEG Encoder Engine */
  jpeg_encode_engine_cfg_t encode_eng_cfg = {
      .intr_priority = 0,
      .timeout_ms = 100,
  };
  esp_err_t ret = jpeg_new_encoder_engine(&encode_eng_cfg, &jpeg_enc_handle);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to create hardware JPEG encoder engine: %s",
             esp_err_to_name(ret));
    return ret;
  }

  jpeg_enc_cfg.src_type = JPEG_ENCODE_IN_FORMAT_RGB565;
  jpeg_enc_cfg.sub_sample = JPEG_DOWN_SAMPLING_YUV422;
  jpeg_enc_cfg.image_quality = 70;
  jpeg_enc_cfg.width = ALERT_WIDTH;
  jpeg_enc_cfg.height = ALERT_HEIGHT;

  ESP_LOGI(TAG,
           "Allocating 21 continuous JPEG buffers in PSRAM (%d KB each, total "
           "~%.1f MB)...",
           ALERT_JPEG_MAX_BYTES / 1024,
           (float)(ALERT_TOTAL_COUNT * ALERT_JPEG_MAX_BYTES) /
               (1024.0f * 1024.0f));

  /* 2. Allocate 10 Pre-Buffers in PSRAM */
  for (int i = 0; i < ALERT_PRE_FRAMES_COUNT; i++) {
    pre_buffers[i].data = (uint8_t *)heap_caps_aligned_calloc(
        64, 1, ALERT_JPEG_MAX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    pre_buffers[i].len = 0;
    if (pre_buffers[i].data == NULL) {
      ESP_LOGE(TAG, "Failed to allocate pre_buffer %d in PSRAM", i);
      return ESP_ERR_NO_MEM;
    }
  }

  /* 3. Allocate 1 Trigger Buffer in PSRAM */
  trigger_buffer.data = (uint8_t *)heap_caps_aligned_calloc(
      64, 1, ALERT_JPEG_MAX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  trigger_buffer.len = 0;
  if (trigger_buffer.data == NULL) {
    ESP_LOGE(TAG, "Failed to allocate trigger_buffer in PSRAM");
    return ESP_ERR_NO_MEM;
  }

  /* 4. Allocate 10 Post-Buffers in PSRAM */
  for (int i = 0; i < ALERT_POST_FRAMES_COUNT; i++) {
    post_buffers[i].data = (uint8_t *)heap_caps_aligned_calloc(
        64, 1, ALERT_JPEG_MAX_BYTES, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
    post_buffers[i].len = 0;
    if (post_buffers[i].data == NULL) {
      ESP_LOGE(TAG, "Failed to allocate post_buffer %d in PSRAM", i);
      return ESP_ERR_NO_MEM;
    }
  }

  capture_stop_sem = xSemaphoreCreateBinary();
  if (capture_stop_sem == NULL) {
    ESP_LOGE(TAG, "Failed to create capture stop semaphore");
    return ESP_ERR_NO_MEM;
  }

  xTaskCreatePinnedToCore(alert_upload_task, "alert_upl", 8192, NULL, 2,
                          &upload_task_handle, 0);

  alert_ready = true;

  ret = alert_recorder_task_start(0);
  if (ret != ESP_OK) {
    return ret;
  }

  ESP_LOGI(TAG,
           "Continuous JPEG Alert Recorder ready (21 frames active, 1024x600)");
  ESP_LOGI(TAG, "========================================");
  return ESP_OK;
}

esp_err_t alert_recorder_trigger(void) {
  if (!alert_ready) {
    ESP_LOGW(TAG, "Alert recorder not ready");
    return ESP_ERR_INVALID_STATE;
  }

  if (alert_triggered || post_capture_active) {
    ESP_LOGW(TAG, "Alert already in progress");
    return ESP_ERR_INVALID_STATE;
  }

  current_alert_id = esp_timer_get_time() / 1000;
  alert_triggered = true;

  /* Freeze current pre-frames indices */
  frozen_pre_head = pre_head;
  frozen_pre_count = pre_count;

  /* Capture Trigger Frame immediately */
  const uint8_t *curr = camera_get_frame();
  if (curr != NULL && trigger_buffer.data != NULL) {
    int disp_w = display_get_width();
    int disp_h = display_get_height();
    if (disp_w <= 0)
      disp_w = ALERT_WIDTH;
    if (disp_h <= 0)
      disp_h = ALERT_HEIGHT;
    encode_rgb565_to_jpeg(curr, disp_w, disp_h, trigger_buffer.data,
                          ALERT_JPEG_MAX_BYTES, &trigger_buffer.len);
  }

  /* Start capturing 10 post-frames continuously without network interruption */
  post_index = 0;
  post_capture_active = true;

  ESP_LOGI(TAG, "****************************************");
  ESP_LOGI(TAG, "EMERGENCY ALERT TRIGGERED! ID: %lld", current_alert_id);
  ESP_LOGI(TAG, "Capturing 10 Pre + 1 Trigger + 10 Post continuous JPEG frames "
                "(1024x600)...");
  ESP_LOGI(TAG, "****************************************");

  return ESP_OK;
}

bool alert_recorder_is_ready(void) { return alert_ready; }
