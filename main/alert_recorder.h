#ifndef ALERT_RECORDER_H
#define ALERT_RECORDER_H

#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define ALERT_WIDTH             1024
#define ALERT_HEIGHT            600
#define ALERT_JPEG_MAX_BYTES    (256 * 1024) // 256 KB per JPEG frame slot in PSRAM

#define ALERT_PRE_FRAMES_COUNT  10
#define ALERT_POST_FRAMES_COUNT 10
#define ALERT_TOTAL_COUNT       (ALERT_PRE_FRAMES_COUNT + 1 + ALERT_POST_FRAMES_COUNT)

typedef struct {
    uint8_t *data;
    size_t   len;
} alert_frame_t;

/**
 * @brief Initialize the alert recorder (allocates ring buffers in PSRAM after system startup)
 */
esp_err_t alert_recorder_init(void);

/**
 * @brief Trigger emergency capture upon detecting voice keyword "ช่วยด้วย"
 */
esp_err_t alert_recorder_trigger(void);

/**
 * @brief Check if alert recorder is ready
 */
bool alert_recorder_is_ready(void);

/**
 * @brief Start the capture task pinned to core_id (0, 1 or tskNO_AFFINITY).
 *        No-op if it is already running. Used by the benchmark.
 */
esp_err_t alert_recorder_task_start(int core_id);

/**
 * @brief Stop the capture task (buffers are kept).
 */
void alert_recorder_task_stop(void);

/**
 * @brief Single-thread mode: run one capture step if 100 ms have passed.
 *        Does nothing while the capture task is running.
 */
void alert_recorder_poll(void);

#ifdef __cplusplus
}
#endif

#endif // ALERT_RECORDER_H
