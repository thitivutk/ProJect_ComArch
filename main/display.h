#ifndef DISPLAY_H
#define DISPLAY_H

#include <stdint.h>
#include <stddef.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Initialize LCD display and LVGL.
 */
esp_err_t display_init(void);

/**
 * Show an RGB565 frame on the LCD.
 *
 * The frame must contain:
 *
 * BSP_LCD_H_RES x BSP_LCD_V_RES
 *
 * RGB565 = 2 bytes per pixel.
 */
esp_err_t display_show_frame(
    const uint8_t *frame,
    size_t frame_size
);

/**
 * Clear the display.
 */
esp_err_t display_clear(void);

/**
 * Deinitialize display.
 */
void display_deinit(void);

/**
 * Get active display width & height.
 */
int display_get_width(void);
int display_get_height(void);

/*
 * Render loop control (used by the benchmark)
 *
 * display_render_takeover()   : suspend the BSP "taskLVGL".
 * display_render_task_start() : run lv_timer_handler() in our own
 *                               task pinned to core_id (0, 1 or
 *                               tskNO_AFFINITY).
 * display_render_task_stop()  : stop that task.
 * display_render_poll()       : single-thread mode: call from the
 *                               main loop, renders when LVGL is due.
 * display_render_release()    : stop our task, resume "taskLVGL".
 */
esp_err_t display_render_takeover(void);
esp_err_t display_render_task_start(int core_id);
void display_render_task_stop(void);
void display_render_poll(void);
void display_render_release(void);

#ifdef __cplusplus
}
#endif

#endif /* DISPLAY_H */