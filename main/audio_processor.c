#include "audio_processor.h"

#include <math.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#include "esp_log.h"
#include "esp_err.h"

#include "dsps_fft2r.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

static const char *TAG = "AUDIO_PROC";

static bool initialized = false;


/*
 * =========================================================
 * Initialize FFT
 * =========================================================
 */

esp_err_t audio_processor_init(void)
{
    if (initialized) {
        return ESP_OK;
    }

    ESP_LOGI(
        TAG,
        "Initializing audio processor..."
    );

    ESP_LOGI(
        TAG,
        "FFT size: %d",
        AUDIO_FRAME_SAMPLES
    );

    ESP_LOGI(
        TAG,
        "Speech band: %d - %d Hz",
        AUDIO_LOW_FREQ,
        AUDIO_HIGH_FREQ
    );

    ESP_LOGI(
        TAG,
        "VAD threshold: %.2f",
        AUDIO_VAD_THRESHOLD
    );

    ESP_LOGI(
        TAG,
        "Silence timeout: %d ms",
        AUDIO_SILENCE_TIMEOUT_MS
    );

    ESP_LOGI(
        TAG,
        "Maximum recording: %d ms",
        AUDIO_MAX_RECORDING_MS
    );


    esp_err_t ret =
        dsps_fft2r_init_fc32(
            NULL,
            AUDIO_FRAME_SAMPLES
        );


    if (ret != ESP_OK) {

        ESP_LOGE(
            TAG,
            "FFT initialization failed: %s",
            esp_err_to_name(ret)
        );

        return ret;
    }


    initialized = true;


    ESP_LOGI(
        TAG,
        "Audio processor initialized"
    );


    return ESP_OK;
}


/*
 * =========================================================
 * VAD
 * =========================================================
 *
 * Calculate RMS of PCM samples.
 *
 * RMS =
 *
 * sqrt(
 *     sum(sample^2) / N
 * )
 *
 * ถ้า RMS >= threshold
 * ถือว่ามีเสียง
 */

bool audio_processor_is_speech(
    const int16_t *samples,
    size_t sample_count,
    float *rms
)
{
    if (samples == NULL ||
        sample_count == 0) {

        if (rms != NULL) {
            *rms = 0.0f;
        }

        return false;
    }


    double sum_square = 0.0;


    for (size_t i = 0;
         i < sample_count;
         i++) {

        float sample =
            (float)samples[i];

        sum_square +=
            (double)sample *
            (double)sample;
    }


    float calculated_rms =
        sqrtf(
            (float)(
                sum_square /
                (double)sample_count
            )
        );


    if (rms != NULL) {
        *rms = calculated_rms;
    }


    return (
        calculated_rms >=
        AUDIO_VAD_THRESHOLD
    );
}


/*
 * =========================================================
 * Process audio frame
 * =========================================================
 */

esp_err_t audio_processor_process(
    const int16_t *input,
    int16_t *output,
    size_t samples,
    float *energy
)
{
    if (!initialized) {
        return ESP_ERR_INVALID_STATE;
    }


    if (input == NULL ||
        output == NULL ||
        samples != AUDIO_FRAME_SAMPLES) {

        return ESP_ERR_INVALID_ARG;
    }


    /*
     * FFT buffer
     *
     * Format:
     *
     * [real, imag, real, imag, ...]
     */

    static float fft_buffer[
        AUDIO_FRAME_SAMPLES * 2
    ];


    /*
     * Calculate RMS
     */

    float sum_square = 0.0f;


    for (size_t i = 0;
         i < samples;
         i++) {

        float sample =
            (float)input[i];

        sum_square +=
            sample * sample;
    }


    float rms =
        sqrtf(
            sum_square /
            (float)samples
        );


    if (energy != NULL) {
        *energy = rms;
    }


    /*
     * Copy PCM → FFT buffer
     *
     * Apply Hann window
     */

    for (size_t i = 0;
         i < samples;
         i++) {

        float window =
            0.5f -
            0.5f *
            cosf(
                2.0f *
                M_PI *
                (float)i /
                (float)(samples - 1)
            );


        fft_buffer[2 * i] =
            ((float)input[i] /
             32768.0f) *
            window;


        fft_buffer[2 * i + 1] =
            0.0f;
    }


    /*
     * Forward FFT
     */

    dsps_fft2r_fc32(
        fft_buffer,
        AUDIO_FRAME_SAMPLES
    );


    dsps_bit_rev_fc32(
        fft_buffer,
        AUDIO_FRAME_SAMPLES
    );


    /*
     * Frequency resolution
     *
     * 16000 / 512
     *
     * = 31.25 Hz/bin
     */

    const float freq_resolution =
        (float)AUDIO_SAMPLE_RATE /
        (float)AUDIO_FRAME_SAMPLES;


    /*
     * Remove frequencies outside
     * speech range.
     */

    for (size_t k = 0;
         k < AUDIO_FRAME_SAMPLES / 2;
         k++) {

        float freq =
            (float)k *
            freq_resolution;


        if (freq < AUDIO_LOW_FREQ ||
            freq > AUDIO_HIGH_FREQ) {

            fft_buffer[2 * k] =
                0.0f;

            fft_buffer[2 * k + 1] =
                0.0f;


            /*
             * Mirror frequency
             */

            size_t mirror =
                AUDIO_FRAME_SAMPLES - k;


            if (mirror <
                AUDIO_FRAME_SAMPLES) {

                fft_buffer[
                    2 * mirror
                ] = 0.0f;

                fft_buffer[
                    2 * mirror + 1
                ] = 0.0f;
            }
        }
    }


    /*
     * Inverse FFT
     *
     * Conjugate
     */

    for (size_t i = 0;
         i < AUDIO_FRAME_SAMPLES;
         i++) {

        fft_buffer[
            2 * i + 1
        ] =
            -fft_buffer[
                2 * i + 1
            ];
    }


    dsps_fft2r_fc32(
        fft_buffer,
        AUDIO_FRAME_SAMPLES
    );


    dsps_bit_rev_fc32(
        fft_buffer,
        AUDIO_FRAME_SAMPLES
    );


    /*
     * Convert FFT result back to PCM
     */

    for (size_t i = 0;
         i < AUDIO_FRAME_SAMPLES;
         i++) {

        float value =
            fft_buffer[2 * i] /
            (float)AUDIO_FRAME_SAMPLES;


        value *= 32768.0f;


        if (value > 32767.0f) {
            value = 32767.0f;
        }


        if (value < -32768.0f) {
            value = -32768.0f;
        }


        output[i] =
            (int16_t)value;
    }


    return ESP_OK;
}