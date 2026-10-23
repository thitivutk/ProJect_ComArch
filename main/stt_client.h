#ifndef STT_CLIENT_H
#define STT_CLIENT_H

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

/*
 * =========================================================
 * ESP32-P4 → PC → Python → Google Speech Recognition
 * =========================================================
 */

/**
 * Initialize STT client.
 */
esp_err_t stt_client_init(void);

/**
 * Send PCM audio to Python STT server
 * and receive recognized text.
 *
 * Audio format:
 *      16-bit signed PCM
 *      16000 Hz
 *      Mono
 *
 * audio:
 *      PCM audio buffer
 *
 * audio_len:
 *      Number of bytes
 *
 * text:
 *      Output buffer
 *
 * text_size:
 *      Size of output buffer
 */
esp_err_t stt_client_transcribe(
    const uint8_t *audio,
    size_t audio_len,
    char *text,
    size_t text_size
);

#endif
