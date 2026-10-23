#ifndef MICROPHONE_H
#define MICROPHONE_H

#include <stddef.h>
#include <stdint.h>
#include "esp_err.h"

/**
 * Initialize built-in microphone
 */
esp_err_t microphone_init(void);

/**
 * Read PCM audio samples from microphone
 *
 * @param buffer       Destination buffer
 * @param buffer_size  Buffer size in bytes
 * @param bytes_read   Number of bytes actually read
 */
esp_err_t microphone_read(
    int16_t *buffer,
    size_t buffer_size,
    size_t *bytes_read
);

/**
 * Number of I2S RX overflows since boot (audio samples lost
 * because microphone_read() was not called in time).
 */
uint32_t microphone_get_overflow_count(void);

/**
 * Deinitialize microphone
 */
void microphone_deinit(void);

#endif