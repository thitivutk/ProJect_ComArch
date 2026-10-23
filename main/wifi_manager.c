#include "wifi_manager.h"

#include <stdio.h>

#include "freertos/FreeRTOS.h"
#include "freertos/event_groups.h"

#include "nvs_flash.h"

#include "driver/gpio.h"
#include "esp_err.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_netif.h"
#include "esp_wifi.h"

static const char *TAG = "WIFI";

/* =========================================================
 * Configuration
 * ========================================================= */

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT BIT1

#define WIFI_MAX_RETRY 10

#define WIFI_SSID "taantor"
#define WIFI_PASSWORD "12345678"

/* =========================================================
 * Private variables
 * ========================================================= */

static EventGroupHandle_t wifi_event_group;

static int wifi_retry_count = 0;

/* =========================================================
 * WiFi Event Handler
 * ========================================================= */

static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data) {
  /* -----------------------------------------------------
   * WiFi started
   * ----------------------------------------------------- */

  if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {

    ESP_LOGI(TAG, "WiFi started");

    esp_err_t err = esp_wifi_connect();

    if (err != ESP_OK && err != ESP_ERR_WIFI_CONN) {

      ESP_LOGE(TAG, "esp_wifi_connect failed: %s", esp_err_to_name(err));
    }
  }

  /* -----------------------------------------------------
   * WiFi disconnected
   * ----------------------------------------------------- */

  else if (event_base == WIFI_EVENT &&
           event_id == WIFI_EVENT_STA_DISCONNECTED) {

    wifi_event_sta_disconnected_t *event =
        (wifi_event_sta_disconnected_t *)event_data;

    const char *reason_str = "UNKNOWN";
    switch (event->reason) {
    case 201:
      reason_str = "NO_AP_FOUND (ไม่พบสัญญาณ WiFi '" WIFI_SSID "' - โปรดตรวจเช็คว่าเปิด "
                   "Hotspot/Router และปล่อยย่าน 2.4GHz หรือไม่)";
      break;
    case 202:
      reason_str = "AUTH_FAIL (รหัสผ่าน WiFi ไม่ถูกต้อง)";
      break;
    case 203:
      reason_str = "ASSOC_FAIL (เชื่อมต่อ Access Point ไม่สำเร็จ)";
      break;
    case 204:
      reason_str = "HANDSHAKE_TIMEOUT (หมดเวลา Handshake)";
      break;
    case 205:
      reason_str = "CONNECTION_FAIL";
      break;
    default:
      break;
    }
    ESP_LOGW(TAG, "WiFi disconnected, reason=%d (%s)", event->reason,
             reason_str);

    xEventGroupClearBits(wifi_event_group, WIFI_CONNECTED_BIT);

    wifi_retry_count++;
    ESP_LOGI(TAG, "Retrying WiFi connection to '%s'... (attempt %d)", WIFI_SSID,
             wifi_retry_count);

    vTaskDelay(pdMS_TO_TICKS(1500));
    esp_err_t err = esp_wifi_connect();
    if (err != ESP_OK && err != ESP_ERR_WIFI_CONN) {
      ESP_LOGW(TAG, "Reconnect call returned: %s", esp_err_to_name(err));
    }
  }

  /* -----------------------------------------------------
   * Got IP
   * ----------------------------------------------------- */

  else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {

    ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;

    wifi_retry_count = 0;

    ESP_LOGI(TAG, "========================================");

    ESP_LOGI(TAG, "WiFi connected!");

    ESP_LOGI(TAG, "IP Address : " IPSTR, IP2STR(&event->ip_info.ip));

    ESP_LOGI(TAG, "Netmask    : " IPSTR, IP2STR(&event->ip_info.netmask));

    ESP_LOGI(TAG, "Gateway    : " IPSTR, IP2STR(&event->ip_info.gw));

    ESP_LOGI(TAG, "========================================");

    xEventGroupSetBits(wifi_event_group, WIFI_CONNECTED_BIT);
  }
}

/* =========================================================
 * Initialize WiFi
 * ========================================================= */

esp_err_t wifi_manager_init(void) {
  esp_err_t ret;

  /* -----------------------------------------------------
   * NVS
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "0. Initializing NVS");

  ret = nvs_flash_init();

  if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
      ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {

    ESP_LOGI(TAG, "NVS erase and re-init");

    ESP_ERROR_CHECK(nvs_flash_erase());

    ret = nvs_flash_init();
  }

  if (ret != ESP_OK) {
    return ret;
  }

  /* -----------------------------------------------------
   * Network interface
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "1. Initializing network interface");

  ret = esp_netif_init();

  if (ret != ESP_OK) {
    return ret;
  }

  /* -----------------------------------------------------
   * Event loop
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "2. Creating event loop");

  ret = esp_event_loop_create_default();

  if (ret != ESP_OK && ret != ESP_ERR_INVALID_STATE) {

    return ret;
  }

  /* -----------------------------------------------------
   * Event group
   * ----------------------------------------------------- */

  wifi_event_group = xEventGroupCreate();

  if (wifi_event_group == NULL) {

    ESP_LOGE(TAG, "Failed to create WiFi event group");

    return ESP_ERR_NO_MEM;
  }

  /* -----------------------------------------------------
   * Create WiFi STA
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "3. Creating WiFi STA");

  esp_netif_t *sta_netif = esp_netif_create_default_wifi_sta();

  if (sta_netif == NULL) {

    ESP_LOGE(TAG, "Failed to create default WiFi STA");

    return ESP_FAIL;
  }

  /* -----------------------------------------------------
   * WiFi init
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "4. Resetting and waiting for ESP32-C6 coprocessor...");
  gpio_reset_pin(GPIO_NUM_54);
  gpio_set_direction(GPIO_NUM_54, GPIO_MODE_OUTPUT);
  gpio_set_level(GPIO_NUM_54, 0);
  vTaskDelay(pdMS_TO_TICKS(100));
  gpio_set_level(GPIO_NUM_54, 1);
  vTaskDelay(pdMS_TO_TICKS(2000));

  ESP_LOGI(TAG, "Initializing WiFi driver (SDIO)...");

  wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();

  ret = esp_wifi_init(&cfg);

  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "esp_wifi_init failed: %s", esp_err_to_name(ret));
    return ret;
  }

  /* -----------------------------------------------------
   * Register WiFi event
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "5. Registering WiFi event handler");

  ret = esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID,
                                   &wifi_event_handler, NULL);

  if (ret != ESP_OK) {
    return ret;
  }

  ret = esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP,
                                   &wifi_event_handler, NULL);

  if (ret != ESP_OK) {
    return ret;
  }

  /* -----------------------------------------------------
   * STA mode
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "6. Setting WiFi STA mode");

  ret = esp_wifi_set_mode(WIFI_MODE_STA);

  if (ret != ESP_OK) {
    return ret;
  }

  /* -----------------------------------------------------
   * WiFi configuration
   * ----------------------------------------------------- */

  ESP_LOGI(TAG, "7. Configuring WiFi");

  wifi_config_t wifi_config = {
      .sta =
          {
              .ssid = WIFI_SSID,
              .password = WIFI_PASSWORD,
          },
  };

  ret = esp_wifi_set_config(WIFI_IF_STA, &wifi_config);

  if (ret != ESP_OK) {
    return ret;
  }

  return ESP_OK;
}

/* =========================================================
 * Start WiFi
 * ========================================================= */

esp_err_t wifi_manager_start(void) {
  ESP_LOGI(TAG, "8. Starting WiFi");

  return esp_wifi_start();
}

/* =========================================================
 * Wait for connection
 * ========================================================= */

esp_err_t wifi_manager_wait_connected(uint32_t timeout_ms) {
  EventBits_t bits =
      xEventGroupWaitBits(wifi_event_group, WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
                          pdFALSE, pdFALSE, pdMS_TO_TICKS(timeout_ms));

  if (bits & WIFI_CONNECTED_BIT) {
    return ESP_OK;
  }

  if (bits & WIFI_FAIL_BIT) {
    return ESP_FAIL;
  }

  return ESP_ERR_TIMEOUT;
}

bool wifi_manager_is_connected(void) {
  if (wifi_event_group == NULL) {
    return false;
  }
  EventBits_t bits = xEventGroupGetBits(wifi_event_group);
  return (bits & WIFI_CONNECTED_BIT) != 0;
}
