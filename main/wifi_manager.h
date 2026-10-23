#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <stdbool.h>
#include "esp_err.h"


/*
 * Initialize:
 * - NVS
 * - Network interface
 * - Event loop
 * - WiFi STA
 * - WiFi event handlers
 */
esp_err_t wifi_manager_init(void);

/*
 * Start WiFi
 */
esp_err_t wifi_manager_start(void);

/*
 * Wait until WiFi gets an IP address.
 *
 * timeout_ms:
 *     timeout in milliseconds
 */
esp_err_t wifi_manager_wait_connected(
    uint32_t timeout_ms
);

/*
 * Check if WiFi is currently connected with valid IP
 */
bool wifi_manager_is_connected(void);

#endif