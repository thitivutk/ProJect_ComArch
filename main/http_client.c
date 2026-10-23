#include "http_client.h"

#include <stdio.h>
#include <inttypes.h>

#include "esp_log.h"
#include "esp_http_client.h"

static const char *TAG = "HTTP";

/* =========================================================
 * HTTP Event Handler
 * ========================================================= */

static esp_err_t http_event_handler(
    esp_http_client_event_t *evt)
{
    switch (evt->event_id) {

    case HTTP_EVENT_ERROR:

        ESP_LOGE(TAG,
                 "HTTP_EVENT_ERROR");

        break;

    case HTTP_EVENT_ON_CONNECTED:

        ESP_LOGI(TAG,
                 "HTTP connected");

        break;

    case HTTP_EVENT_HEADER_SENT:

        ESP_LOGI(TAG,
                 "HTTP header sent");

        break;

    case HTTP_EVENT_ON_HEADER:

        ESP_LOGI(TAG,
                 "HEADER: %s = %s",
                 evt->header_key,
                 evt->header_value);

        break;

    case HTTP_EVENT_ON_DATA:

        ESP_LOGI(TAG,
                 "Received %d bytes",
                 evt->data_len);

        /*
         * evt->data ไม่จำเป็นต้องลงท้ายด้วย '\0'
         *
         * ดังนั้นใช้ fwrite()
         */

        if (evt->data != NULL &&
            evt->data_len > 0) {

            fwrite(
                evt->data,
                1,
                evt->data_len,
                stdout
            );

            printf("\n");
        }

        break;

    case HTTP_EVENT_ON_FINISH:

        ESP_LOGI(TAG,
                 "HTTP transfer finished");

        break;

    case HTTP_EVENT_DISCONNECTED:

        ESP_LOGI(TAG,
                 "HTTP disconnected");

        break;

    case HTTP_EVENT_REDIRECT:

        ESP_LOGW(TAG,
                 "HTTP redirect");

        break;

    default:

        break;
    }

    return ESP_OK;
}

/* =========================================================
 * HTTP Internet Test
 * ========================================================= */

esp_err_t http_client_test(void)
{
    ESP_LOGI(TAG,
             "========================================");

    ESP_LOGI(TAG,
             "Testing Internet connection");

    ESP_LOGI(TAG,
             "========================================");

    esp_http_client_config_t config = {

        .url =
            "http://example.com",

        .method =
            HTTP_METHOD_GET,

        .timeout_ms =
            10000,

        .event_handler =
            http_event_handler,

        .buffer_size =
            4096,

        .buffer_size_tx =
            2048,

        .disable_auto_redirect =
            false,
    };

    ESP_LOGI(TAG,
             "URL: %s",
             config.url);

    ESP_LOGI(TAG,
             "Sending HTTP GET...");

    esp_http_client_handle_t client =
        esp_http_client_init(&config);

    if (client == NULL) {

        ESP_LOGE(TAG,
                 "Failed to initialize HTTP client");

        return ESP_FAIL;
    }

    esp_err_t err =
        esp_http_client_perform(client);

    if (err == ESP_OK) {

        int status =
            esp_http_client_get_status_code(client);

        int64_t content_length =
            esp_http_client_get_content_length(client);

        ESP_LOGI(TAG,
                 "HTTP GET successful");

        ESP_LOGI(TAG,
                 "Status = %d",
                 status);

        ESP_LOGI(TAG,
                 "Content-Length = %" PRId64,
                 content_length);

        if (status == 200) {

            ESP_LOGI(TAG,
                     "Internet connection OK");

        } else {

            ESP_LOGW(TAG,
                     "HTTP returned status %d",
                     status);
        }

    } else {

        ESP_LOGE(TAG,
                 "HTTP GET failed: %s",
                 esp_err_to_name(err));
    }

    esp_http_client_cleanup(client);

    ESP_LOGI(TAG,
             "========================================");

    return err;
}