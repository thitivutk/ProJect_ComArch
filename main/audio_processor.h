#ifndef AUDIO_PROCESSOR_H
#define AUDIO_PROCESSOR_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "esp_err.h"

/*
 * =========================================================
 * Audio configuration
 * =========================================================
 */

#define AUDIO_SAMPLE_RATE       16000
#define AUDIO_FRAME_SAMPLES     512
#define AUDIO_FRAME_BYTES       (AUDIO_FRAME_SAMPLES * sizeof(int16_t))

/*
 * Speech frequency range
 *
 * ใช้สำหรับ audio processing / FFT
 */
#define AUDIO_LOW_FREQ          100
#define AUDIO_HIGH_FREQ         4000

/*
 * =========================================================
 * Voice Activity Detection
 * =========================================================
 *
 * Frame = 512 samples
 * Sample rate = 16000 Hz
 *
 * 512 / 16000 = 32 ms
 *
 * ถ้ามีเสียง → เริ่มบันทึก
 *
 * ถ้าเงียบต่อเนื่อง 600 ms
 * → จบการบันทึก
 *
 * ถ้าบันทึกต่อเนื่องถึง 5000 ms
 * → ยกเลิกและทิ้งเสียงชุดนั้น
 */

/*
 * RMS threshold
 *
 * ค่านี้เป็นค่าเริ่มต้น
 * สามารถปรับตามสภาพไมค์/เสียงรบกวนได้
 */
#define AUDIO_VAD_THRESHOLD      500.0f

/*
 * Silence duration before stopping
 */
#define AUDIO_SILENCE_TIMEOUT_MS 600

/*
 * Pre-roll buffer before speech (8 frames * 32ms = 256 ms ~ 0.26 sec)
 * เพื่อไม่ให้เสียงพยัญชนะต้นคำขาดหายกรณีพูดเร็ว
 */
#define AUDIO_PREROLL_FRAMES     8

/*
 * Maximum recording duration
 */
#define AUDIO_MAX_RECORDING_MS   5000

/*
 * =========================================================
 * Audio processor
 * =========================================================
 */

esp_err_t audio_processor_init(void);

/*
 * Process one PCM frame
 *
 * input:
 *     raw microphone PCM
 *
 * output:
 *     filtered PCM
 *
 * energy:
 *     RMS energy
 */
esp_err_t audio_processor_process(
    const int16_t *input,
    int16_t *output,
    size_t samples,
    float *energy
);

/*
 * =========================================================
 * VAD
 * =========================================================
 */

/*
 * ตรวจว่า frame นี้มีเสียงหรือไม่
 *
 * return:
 *     true  = มีเสียง
 *     false = เงียบ
 */
bool audio_processor_is_speech(
    const int16_t *samples,
    size_t sample_count,
    float *rms
);

#endif