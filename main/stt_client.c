#include "stt_client.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_err.h"
#include "esp_http_client.h"
#include "esp_log.h"

static const char *TAG = "STT";

/*
 * =========================================================
 * Python STT Server
 * =========================================================
 */

#define STT_SERVER_URL "http://172.20.10.13:5000/stt"

/*
 * =========================================================
 * HTTP Response buffer
 * =========================================================
 */

typedef struct {

  char *buffer;

  size_t buffer_size;

  size_t data_len;

} stt_response_t;

/*
 * =========================================================
 * HTTP Event Handler
 * =========================================================
 */

static esp_err_t http_event_handler(esp_http_client_event_t *evt) {
  stt_response_t *response = (stt_response_t *)evt->user_data;

  if (response == NULL) {

    return ESP_OK;
  }

  switch (evt->event_id) {

    /*
     * -------------------------------------------------
     * Receive response data
     * -------------------------------------------------
     */

  case HTTP_EVENT_ON_DATA:

    if (evt->data == NULL || evt->data_len <= 0) {

      break;
    }

    /*
     * Check buffer space
     */

    if (response->data_len + evt->data_len >= response->buffer_size) {

      ESP_LOGE(TAG, "Response buffer too small");

      break;
    }

    /*
     * Copy response
     */

    memcpy(response->buffer + response->data_len,

           evt->data,

           evt->data_len);

    response->data_len += evt->data_len;

    /*
     * Keep string terminated
     */

    response->buffer[response->data_len] = '\0';

    break;

  default:

    break;
  }

  return ESP_OK;
}

/*
 * =========================================================
 * Initialize
 * =========================================================
 */

esp_err_t stt_client_init(void) {
  ESP_LOGI(TAG, "Initializing STT client...");

  ESP_LOGI(TAG, "STT server: %s", STT_SERVER_URL);

  ESP_LOGI(TAG, "Audio format: PCM 16-bit / 16000 Hz / Mono");

  ESP_LOGI(TAG, "STT client initialized");

  return ESP_OK;
}

/*
 * =========================================================
 * Send PCM → Python STT
 * =========================================================
 */

esp_err_t stt_client_transcribe(const uint8_t *audio, size_t audio_len,
                                char *text, size_t text_size) {
  /*
   * -----------------------------------------------------
   * Validate input
   * -----------------------------------------------------
   */

  if (audio == NULL) {

    ESP_LOGE(TAG, "Audio buffer is NULL");

    return ESP_ERR_INVALID_ARG;
  }

  if (audio_len == 0) {

    ESP_LOGE(TAG, "Audio length is zero");

    return ESP_ERR_INVALID_ARG;
  }

  if (text == NULL) {

    ESP_LOGE(TAG, "Text buffer is NULL");

    return ESP_ERR_INVALID_ARG;
  }

  if (text_size == 0) {

    ESP_LOGE(TAG, "Text buffer size is zero");

    return ESP_ERR_INVALID_ARG;
  }

  /*
   * -----------------------------------------------------
   * Clear output
   * -----------------------------------------------------
   */

  text[0] = '\0';

  /*
   * -----------------------------------------------------
   * Response structure
   * -----------------------------------------------------
   */

  stt_response_t response = {

      .buffer = text,

      .buffer_size = text_size,

      .data_len = 0

  };

  /*
   * -----------------------------------------------------
   * Display information
   * -----------------------------------------------------
   */

  float duration = (float)audio_len / (16000.0f * 2.0f);

  ESP_LOGI(TAG, "========================================");

  ESP_LOGI(TAG, "Sending audio to PC STT server");

  ESP_LOGI(TAG, "URL      : %s", STT_SERVER_URL);

  ESP_LOGI(TAG, "PCM size : %u bytes", (unsigned)audio_len);

  ESP_LOGI(TAG, "Duration : %.2f sec", duration);

  ESP_LOGI(TAG, "========================================");

  /*
   * -----------------------------------------------------
   * Configure HTTP client
   * -----------------------------------------------------
   */

  esp_http_client_config_t config = {

      .url = STT_SERVER_URL,

      .method = HTTP_METHOD_POST,

      .timeout_ms = 15000,

      .event_handler = http_event_handler,

      .user_data = &response

  };

  esp_http_client_handle_t client = esp_http_client_init(&config);

  if (client == NULL) {

    ESP_LOGE(TAG, "Failed to initialize HTTP client");

    return ESP_FAIL;
  }

  /*
   * -----------------------------------------------------
   * Content type
   * -----------------------------------------------------
   *
   * Raw PCM data
   */

  esp_http_client_set_header(client, "Content-Type",
                             "application/octet-stream");

  /*
   * -----------------------------------------------------
   * Set PCM body
   * -----------------------------------------------------
   */

  esp_err_t ret =
      esp_http_client_set_post_field(client, (const char *)audio, audio_len);

  if (ret != ESP_OK) {

    ESP_LOGE(TAG, "Failed to set POST data: %s", esp_err_to_name(ret));

    esp_http_client_cleanup(client);

    return ret;
  }

  /*
   * -----------------------------------------------------
   * Send request
   * -----------------------------------------------------
   */

  ESP_LOGI(TAG, "POST /stt ...");

  ret = esp_http_client_perform(client);

  if (ret != ESP_OK) {

    ESP_LOGE(TAG, "HTTP request failed: %s", esp_err_to_name(ret));

    esp_http_client_cleanup(client);

    return ret;
  }

  /*
   * -----------------------------------------------------
   * HTTP status
   * -----------------------------------------------------
   */

  int status = esp_http_client_get_status_code(client);

  ESP_LOGI(TAG, "HTTP status: %d", status);

  /*
   * -----------------------------------------------------
   * Cleanup HTTP client
   * -----------------------------------------------------
   */

  esp_http_client_cleanup(client);

  /*
   * -----------------------------------------------------
   * Check HTTP result
   * -----------------------------------------------------
   */

  if (status != 200) {

    ESP_LOGE(TAG, "STT server returned HTTP %d", status);

    text[0] = '\0';

    return ESP_FAIL;
  }

  /*
   * -----------------------------------------------------
   * Check response
   * -----------------------------------------------------
   */

  if (response.data_len == 0) {

    ESP_LOGW(TAG, "STT returned empty text");

    text[0] = '\0';

    return ESP_OK;
  }

  /*
   * -----------------------------------------------------
   * Remove trailing newline
   * -----------------------------------------------------
   */

  while (response.data_len > 0 && (text[response.data_len - 1] == '\n' ||

                                   text[response.data_len - 1] == '\r')) {

    text[response.data_len - 1] = '\0';

    response.data_len--;
  }

  /*
   * -----------------------------------------------------
   * Display recognized text
   * -----------------------------------------------------
   */

  ESP_LOGI(TAG, "========================================");

  ESP_LOGI(TAG, "STT result:");

  ESP_LOGI(TAG, "%s", text);

  ESP_LOGI(TAG, "========================================");

  return ESP_OK;
}