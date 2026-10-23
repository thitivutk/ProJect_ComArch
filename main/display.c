#include "display.h"

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_timer.h"

#include "bsp/esp-bsp.h"
#include "lvgl.h"

#include "benchmark.h"

static const char *TAG = "DISPLAY";

/*
 * =========================================================
 * Render loop control (benchmark)
 *
 * Normally the BSP's "taskLVGL" runs lv_timer_handler().
 * The benchmark suspends it and runs the same loop either
 * in its own task ("lvgl_bench", placeable on a core) or
 * from the single-thread main loop (display_render_poll).
 * Same priority / stack / max sleep as esp_lvgl_port.
 * =========================================================
 */
#define DISPLAY_RENDER_TASK_NAME     "lvgl_bench"
#define DISPLAY_RENDER_TASK_PRIORITY 4
#define DISPLAY_RENDER_TASK_STACK    7168
#define DISPLAY_RENDER_MAX_SLEEP_MS  500

static TaskHandle_t lvgl_port_task = NULL;
static TaskHandle_t render_task = NULL;
static volatile bool render_task_stop = false;
static SemaphoreHandle_t render_stop_sem = NULL;
static int64_t render_next_us = 0;
static int64_t render_start_us = 0;

/*
 * =========================================================
 * Display state
 * =========================================================
 */
static bool display_initialized = false;
static int active_lcd_w = BSP_LCD_H_RES;
static int active_lcd_h = BSP_LCD_V_RES;

/*
 * LVGL image object
 */
static lv_obj_t *camera_image = NULL;

/*
 * =========================================================
 * Display framebuffer
 * =========================================================
 */
static uint8_t *display_framebuffer = NULL;
static size_t display_framebuffer_size = 0;

/*
 * =========================================================
 * LVGL image descriptor
 * =========================================================
 */
static lv_image_dsc_t camera_image_descriptor;

int display_get_width(void)
{
    return active_lcd_w;
}

int display_get_height(void)
{
    return active_lcd_h;
}

static size_t display_get_frame_size(void)
{
    return (size_t)active_lcd_w * (size_t)active_lcd_h * 2;
}

/*
 * Time real screen renders only (not idle lv_timer_handler calls).
 * Runs inside lv_timer_handler(), in whichever task drives LVGL.
 */
static void display_render_event_cb(lv_event_t *e)
{
    lv_event_code_t code = lv_event_get_code(e);

    if (code == LV_EVENT_RENDER_START)
    {
        render_start_us = esp_timer_get_time();
    }
    else if (code == LV_EVENT_RENDER_READY && render_start_us > 0)
    {
        bench_record(BENCH_M_LVGL_RENDER, esp_timer_get_time() - render_start_us);
        render_start_us = 0;
    }
}

/*
 * =========================================================
 * Display initialization
 * =========================================================
 */
esp_err_t display_init(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Initializing display...");

    /*
     * 1. Start BSP display
     */
    lv_display_t *display = bsp_display_start();
    if (display == NULL)
    {
        ESP_LOGE(TAG, "bsp_display_start() failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "bsp_display_start() OK");

    /*
     * 2. Turn on backlight
     */
    esp_err_t ret = bsp_display_backlight_on();
    if (ret != ESP_OK)
    {
        ESP_LOGE(TAG, "bsp_display_backlight_on() failed: %s", esp_err_to_name(ret));
        return ret;
    }

    /*
     * 3. Query true hardware LCD dimensions from LVGL
     */
    int32_t hor_res = lv_display_get_horizontal_resolution(display);
    int32_t ver_res = lv_display_get_vertical_resolution(display);
    if (hor_res > 0) active_lcd_w = (int)hor_res;
    if (ver_res > 0) active_lcd_h = (int)ver_res;

    display_framebuffer_size = display_get_frame_size();
    ESP_LOGI(TAG, "Hardware LCD resolution: %d x %d", active_lcd_w, active_lcd_h);
    ESP_LOGI(TAG, "Pixel format: RGB565");
    ESP_LOGI(TAG, "Framebuffer size: %u bytes", (unsigned)display_framebuffer_size);

    /*
     * 4. Allocate framebuffer in PSRAM
     */
    display_framebuffer = (uint8_t *)heap_caps_malloc(
        display_framebuffer_size,
        MALLOC_CAP_8BIT | MALLOC_CAP_SPIRAM
    );
    if (display_framebuffer == NULL)
    {
        display_framebuffer = (uint8_t *)heap_caps_malloc(
            display_framebuffer_size,
            MALLOC_CAP_8BIT
        );
    }

    if (display_framebuffer == NULL)
    {
        ESP_LOGE(TAG, "Failed to allocate framebuffer: %u bytes", (unsigned)display_framebuffer_size);
        bsp_display_backlight_off();
        return ESP_ERR_NO_MEM;
    }

    memset(display_framebuffer, 0, display_framebuffer_size);

    /*
     * 5. Lock LVGL and setup full-screen canvas (0 margins, 0 padding)
     */
    if (!bsp_display_lock(1000))
    {
        ESP_LOGE(TAG, "Failed to acquire LVGL lock");
        heap_caps_free(display_framebuffer);
        display_framebuffer = NULL;
        display_framebuffer_size = 0;
        bsp_display_backlight_off();
        return ESP_ERR_TIMEOUT;
    }

    lv_obj_t *screen = lv_screen_active();
    lv_obj_remove_style_all(screen);
    lv_obj_set_style_pad_all(screen, 0, LV_PART_MAIN);
    lv_obj_set_style_margin_all(screen, 0, LV_PART_MAIN);
    lv_obj_set_style_border_width(screen, 0, LV_PART_MAIN);
    lv_obj_set_style_outline_width(screen, 0, LV_PART_MAIN);
    lv_obj_set_size(screen, active_lcd_w, active_lcd_h);
    lv_obj_set_pos(screen, 0, 0);
    lv_obj_remove_flag(screen, LV_OBJ_FLAG_SCROLLABLE);

    /*
     * 6. Create and configure image object
     */
    camera_image = lv_image_create(screen);
    if (camera_image == NULL)
    {
        ESP_LOGE(TAG, "lv_image_create() failed");
        bsp_display_unlock();
        heap_caps_free(display_framebuffer);
        display_framebuffer = NULL;
        display_framebuffer_size = 0;
        bsp_display_backlight_off();
        return ESP_ERR_NO_MEM;
    }

    lv_obj_remove_style_all(camera_image);
    lv_obj_set_style_pad_all(camera_image, 0, LV_PART_MAIN);
    lv_obj_set_style_margin_all(camera_image, 0, LV_PART_MAIN);
    lv_obj_set_style_border_width(camera_image, 0, LV_PART_MAIN);
    lv_obj_set_style_outline_width(camera_image, 0, LV_PART_MAIN);
    lv_obj_set_size(camera_image, active_lcd_w, active_lcd_h);
    lv_obj_set_pos(camera_image, 0, 0);
    lv_obj_remove_flag(camera_image, LV_OBJ_FLAG_SCROLLABLE);
    lv_image_set_inner_align(camera_image, LV_IMAGE_ALIGN_STRETCH);

    /*
     * 7. Initialize image descriptor
     */
    memset(&camera_image_descriptor, 0, sizeof(camera_image_descriptor));
    camera_image_descriptor.header.magic = LV_IMAGE_HEADER_MAGIC;
    camera_image_descriptor.header.cf = LV_COLOR_FORMAT_RGB565;
    camera_image_descriptor.header.w = active_lcd_w;
    camera_image_descriptor.header.h = active_lcd_h;
    camera_image_descriptor.header.stride = active_lcd_w * 2;
    camera_image_descriptor.data_size = display_framebuffer_size;
    camera_image_descriptor.data = display_framebuffer;

    lv_image_set_src(camera_image, &camera_image_descriptor);

    lv_display_add_event_cb(display, display_render_event_cb, LV_EVENT_ALL, NULL);

    bsp_display_unlock();

    display_initialized = true;
    ESP_LOGI(TAG, "Display initialized successfully (%d x %d)", active_lcd_w, active_lcd_h);
    ESP_LOGI(TAG, "========================================");

    return ESP_OK;
}

/*
 * =========================================================
 * Show camera frame
 * =========================================================
 */
esp_err_t display_show_frame(
    const uint8_t *frame,
    size_t frame_size
)
{
    if (!display_initialized)
    {
        return ESP_ERR_INVALID_STATE;
    }

    if (frame == NULL)
    {
        ESP_LOGE(TAG, "display_show_frame(): frame is NULL");
        return ESP_ERR_INVALID_ARG;
    }

    size_t expected_size = display_get_frame_size();
    if (frame_size < expected_size)
    {
        ESP_LOGE(TAG, "Frame too small: %u < %u", (unsigned)frame_size, (unsigned)expected_size);
        return ESP_ERR_INVALID_SIZE;
    }

    if (!bsp_display_lock(0))
    {
        return ESP_ERR_TIMEOUT;
    }

    camera_image_descriptor.data = (const void *)frame;
    camera_image_descriptor.data_size = expected_size;

    lv_image_set_src(camera_image, &camera_image_descriptor);
    lv_obj_invalidate(camera_image);

    bsp_display_unlock();

    return ESP_OK;
}

/*
 * =========================================================
 * Clear display
 * =========================================================
 */
esp_err_t display_clear(void)
{
    if (!display_initialized)
    {
        return ESP_ERR_INVALID_STATE;
    }

    if (!bsp_display_lock(0))
    {
        return ESP_ERR_TIMEOUT;
    }

    if (display_framebuffer != NULL)
    {
        memset(display_framebuffer, 0, display_framebuffer_size);
    }

    if (camera_image != NULL)
    {
        lv_obj_invalidate(camera_image);
    }

    bsp_display_unlock();

    ESP_LOGI(TAG, "Display cleared");
    return ESP_OK;
}

/*
 * =========================================================
 * Render loop control (benchmark)
 * =========================================================
 */

/* One lv_timer_handler() call. Returns ms until LVGL needs to run again. */
static uint32_t display_render_once(void)
{
    if (!bsp_display_lock(0))
    {
        return 1;
    }

    uint32_t next_ms = lv_timer_handler();

    bsp_display_unlock();

    if (next_ms == LV_NO_TIMER_READY || next_ms > DISPLAY_RENDER_MAX_SLEEP_MS)
    {
        next_ms = DISPLAY_RENDER_MAX_SLEEP_MS;
    }

    return next_ms;
}

static void display_render_task(void *arg)
{
    (void)arg;

    while (!render_task_stop)
    {
        uint32_t next_ms = display_render_once();

        /* Same pacing as esp_lvgl_port: at least one tick */
        TickType_t wait = pdMS_TO_TICKS(next_ms);
        vTaskDelay(wait > 0 ? wait : 1);
    }

    render_task = NULL;
    xSemaphoreGive(render_stop_sem);
    vTaskDelete(NULL);
}

esp_err_t display_render_takeover(void)
{
    if (lvgl_port_task != NULL)
    {
        return ESP_OK;
    }

    TaskHandle_t port_task = xTaskGetHandle("taskLVGL");
    if (port_task == NULL)
    {
        ESP_LOGE(TAG, "taskLVGL not found");
        return ESP_ERR_NOT_FOUND;
    }

    if (render_stop_sem == NULL)
    {
        render_stop_sem = xSemaphoreCreateBinary();
        if (render_stop_sem == NULL)
        {
            return ESP_ERR_NO_MEM;
        }
    }

    /* Hold the LVGL lock so taskLVGL is not inside lv_timer_handler() */
    if (!bsp_display_lock(1000))
    {
        return ESP_ERR_TIMEOUT;
    }
    vTaskSuspend(port_task);
    bsp_display_unlock();

    lvgl_port_task = port_task;
    render_next_us = 0;

    ESP_LOGI(TAG, "taskLVGL suspended, render loop taken over");
    return ESP_OK;
}

esp_err_t display_render_task_start(int core_id)
{
    if (lvgl_port_task == NULL)
    {
        return ESP_ERR_INVALID_STATE;
    }

    if (render_task != NULL)
    {
        return ESP_OK;
    }

    render_task_stop = false;
    xSemaphoreTake(render_stop_sem, 0);

    if (xTaskCreatePinnedToCore(display_render_task,
                                DISPLAY_RENDER_TASK_NAME,
                                DISPLAY_RENDER_TASK_STACK,
                                NULL,
                                DISPLAY_RENDER_TASK_PRIORITY,
                                &render_task,
                                core_id) != pdPASS)
    {
        render_task = NULL;
        ESP_LOGE(TAG, "Failed to create render task");
        return ESP_FAIL;
    }

    return ESP_OK;
}

void display_render_task_stop(void)
{
    if (render_task == NULL)
    {
        return;
    }

    render_task_stop = true;

    if (xSemaphoreTake(render_stop_sem, pdMS_TO_TICKS(1000)) != pdTRUE)
    {
        ESP_LOGE(TAG, "Render task did not stop");
        return;
    }

    render_task_stop = false;
}

void display_render_poll(void)
{
    if (lvgl_port_task == NULL || render_task != NULL)
    {
        return;
    }

    int64_t now = esp_timer_get_time();
    if (now < render_next_us)
    {
        return;
    }

    uint32_t next_ms = display_render_once();
    render_next_us = esp_timer_get_time() + (int64_t)next_ms * 1000;
}

void display_render_release(void)
{
    display_render_task_stop();

    if (lvgl_port_task != NULL)
    {
        vTaskResume(lvgl_port_task);
        lvgl_port_task = NULL;
        ESP_LOGI(TAG, "taskLVGL resumed");
    }
}


/*
 * =========================================================
 * Display deinitialization
 * =========================================================
 */
void display_deinit(void)
{
    if (!display_initialized)
    {
        return;
    }

    ESP_LOGI(TAG, "Deinitializing display...");

    if (bsp_display_lock(1000))
    {
        if (camera_image != NULL)
        {
            lv_obj_del(camera_image);
            camera_image = NULL;
        }
        bsp_display_unlock();
    }

    bsp_display_backlight_off();

    if (display_framebuffer != NULL)
    {
        heap_caps_free(display_framebuffer);
        display_framebuffer = NULL;
    }

    display_framebuffer_size = 0;
    memset(&camera_image_descriptor, 0, sizeof(camera_image_descriptor));

    display_initialized = false;
    ESP_LOGI(TAG, "Display deinitialized");
}
