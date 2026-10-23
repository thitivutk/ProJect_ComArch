#ifndef CAMERA_H
#define CAMERA_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif


/*
 * =========================================================
 * Camera initialization
 * =========================================================
 *
 * Initialize the ESP32-P4 camera pipeline.
 *
 * This function:
 *
 *   - starts BSP camera
 *   - opens V4L2 camera device
 *   - configures RGB565
 *   - allocates camera buffers
 *   - allocates display buffers
 *   - initializes PPA
 *   - starts V4L2 streaming
 *   - starts camera background task
 *
 * After this function returns ESP_OK,
 * the camera starts capturing continuously.
 *
 */
esp_err_t camera_init(void);


/*
 * =========================================================
 * Camera task control (used by the benchmark)
 * =========================================================
 *
 * camera_task_start(core) : start the background camera
 *                           task pinned to core (0, 1 or
 *                           tskNO_AFFINITY). No-op if running.
 *
 * camera_task_stop()      : stop the task, stream stays on.
 *
 * camera_poll(timeout_ms) : single-thread mode. Handle at most
 *                           one frame, waiting up to timeout_ms
 *                           for it (0 = never block). Only
 *                           works while the task is stopped.
 *                           Returns true if a frame was shown.
 */
esp_err_t camera_task_start(int core_id);

void camera_task_stop(void);

bool camera_poll(uint32_t timeout_ms);


/*
 * =========================================================
 * Camera capture
 * =========================================================
 *
 * The camera is continuously capturing in the
 * background.
 *
 * This function waits until at least one processed
 * frame is available.
 *
 * NOTE:
 *
 * The application normally does NOT need to call
 * this function because camera_init() already starts
 * the background camera task and that task sends
 * frames directly to the display.
 *
 */
esp_err_t camera_capture(void);


/*
 * =========================================================
 * Get latest frame
 * =========================================================
 *
 * Return the latest processed frame.
 *
 * Format:
 *
 *     RGB565
 *
 * Resolution:
 *
 *     BSP_LCD_H_RES x BSP_LCD_V_RES
 *
 * IMPORTANT:
 *
 * The returned pointer is owned by the camera pipeline.
 *
 * Do NOT:
 *
 *     free()
 *     heap_caps_free()
 *     modify()
 *
 * the returned buffer.
 *
 */
const uint8_t *camera_get_frame(void);


/*
 * =========================================================
 * Get frame size
 * =========================================================
 *
 * Return the size of the current processed frame
 * in bytes.
 *
 * For RGB565:
 *
 *     width x height x 2
 *
 */
size_t camera_get_frame_size(void);


/*
 * =========================================================
 * Get frame width
 * =========================================================
 *
 * Returns the output/display frame width.
 *
 */
uint32_t camera_get_width(void);


/*
 * =========================================================
 * Get frame height
 * =========================================================
 *
 * Returns the output/display frame height.
 *
 */
uint32_t camera_get_height(void);


/*
 * =========================================================
 * Release frame
 * =========================================================
 *
 * Mark the current frame as consumed.
 *
 * IMPORTANT:
 *
 * This does NOT free any memory.
 *
 * The camera pipeline owns all frame buffers.
 *
 */
void camera_release_frame(void);


/*
 * =========================================================
 * Camera deinitialization
 * =========================================================
 *
 * Stop the camera background task,
 * stop V4L2 streaming,
 * unmap camera buffers,
 * close the V4L2 device,
 * and free display buffers.
 *
 */
void camera_deinit(void);


#ifdef __cplusplus
}
#endif

#endif