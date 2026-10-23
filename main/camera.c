#include "camera.h"

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#include "linux/videodev2.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_err.h"
#include "camera.h"

#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>

#include "linux/videodev2.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_heap_caps.h"

#include "esp_timer.h"

#include "esp_video_device.h"
#include "esp_video_ioctl.h"

#include "driver/ppa.h"

#include "bsp/esp-bsp.h"

#include "benchmark.h"
#include "display.h"


static const char *TAG = "CAMERA";


/*
 * =========================================================
 * Configuration
 * =========================================================
 */

#define CAMERA_NUM_BUFS        2

#define CAMERA_TASK_STACK_SIZE 8192
#define CAMERA_TASK_PRIORITY   15
#define CAMERA_TASK_CORE       1

/* Task-mode DQBUF wait, short enough to notice a stop request */
#define CAMERA_DQBUF_TIMEOUT_MS 200

#define CAMERA_CACHE_ALIGNMENT 64


#define ALIGN_UP(num, align) \
    (((num) + ((align) - 1)) & ~((align) - 1))


/*
 * =========================================================
 * Camera state
 * =========================================================
 */

static bool camera_initialized = false;

static int video_fd = -1;


/*
 * =========================================================
 * V4L2 camera buffers
 * =========================================================
 */

static uint8_t *camera_buffers[CAMERA_NUM_BUFS];

static size_t camera_buffer_size = 0;


/*
 * =========================================================
 * Camera format
 * =========================================================
 */

static uint32_t camera_width = 0;

static uint32_t camera_height = 0;

static uint32_t camera_pixel_format = 0;


/*
 * =========================================================
 * Display buffers
 *
 * Camera frame is converted by PPA into one of these
 * buffers before being passed to LVGL.
 * =========================================================
 */

static uint8_t *display_buffers[CAMERA_NUM_BUFS];

static size_t display_buffer_size = 0;


/*
 * =========================================================
 * Current frame
 * =========================================================
 */

static volatile const uint8_t *frame_buffer = NULL;

static volatile size_t frame_size = 0;

static volatile bool frame_ready = false;


/*
 * =========================================================
 * Video task
 * =========================================================
 */

static TaskHandle_t video_task_handle = NULL;

static volatile bool video_task_stop = false;

/* True only after VIDIOC_STREAMON succeeds. */
static bool video_streaming = false;


/*
 * =========================================================
 * PPA
 * =========================================================
 */

static ppa_client_handle_t ppa_handle = NULL;


/*
 * =========================================================
 * Stop semaphore
 * =========================================================
 */

static SemaphoreHandle_t video_stop_sem = NULL;


/*
 * =========================================================
 * Current display buffer
 *
 * The camera alternates between two output buffers.
 * =========================================================
 */

static uint8_t current_display_buffer = 0;


/*
 * =========================================================
 * PPA initialization
 * =========================================================
 */

static esp_err_t camera_ppa_init(void)
{
    if (ppa_handle != NULL)
    {
        return ESP_OK;
    }


    ppa_client_config_t config = {
        .oper_type = PPA_OPERATION_SRM,
    };


    esp_err_t ret =
        ppa_register_client(
            &config,
            &ppa_handle
        );


    if (ret != ESP_OK)
    {
        ESP_LOGE(
            TAG,
            "ppa_register_client failed: %s",
            esp_err_to_name(ret)
        );

        return ret;
    }


    ESP_LOGI(
        TAG,
        "PPA initialized"
    );


    return ESP_OK;
}


/*
 * =========================================================
 * Open V4L2 camera
 * =========================================================
 */

static esp_err_t camera_open_device(void)
{
    ESP_LOGI(
        TAG,
        "Opening camera device: %s",
        BSP_CAMERA_DEVICE
    );


    /*
     * Open V4L2 device.
     */

    video_fd =
        open(
            BSP_CAMERA_DEVICE,
            O_RDONLY
        );


    if (video_fd < 0)
    {
        ESP_LOGE(
            TAG,
            "Failed to open camera device"
        );

        ESP_LOGE(
            TAG,
            "errno = %d",
            errno
        );

        ESP_LOGE(
            TAG,
            "error = %s",
            strerror(errno)
        );

        return ESP_FAIL;
    }


    ESP_LOGI(
        TAG,
        "Camera device opened"
    );

    ESP_LOGI(
        TAG,
        "video_fd = %d",
        video_fd
    );


    /*
     * =====================================================
     * Query capability
     * =====================================================
     */

    struct v4l2_capability capability;

    memset(
        &capability,
        0,
        sizeof(capability)
    );


    if (
        ioctl(
            video_fd,
            VIDIOC_QUERYCAP,
            &capability
        ) != 0
    )
    {
        ESP_LOGE(
            TAG,
            "VIDIOC_QUERYCAP failed: errno=%d (%s)",
            errno,
            strerror(errno)
        );

        close(video_fd);

        video_fd = -1;

        return ESP_FAIL;
    }


    ESP_LOGI(
        TAG,
        "Camera driver : %s",
        capability.driver
    );

    ESP_LOGI(
        TAG,
        "Camera card   : %s",
        capability.card
    );

    ESP_LOGI(
        TAG,
        "Capabilities  : 0x%08lx",
        (unsigned long)capability.capabilities
    );


    /*
     * =====================================================
     * Get current format
     * =====================================================
     */

    struct v4l2_format format;

    memset(
        &format,
        0,
        sizeof(format)
    );


    format.type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;


    if (
        ioctl(
            video_fd,
            VIDIOC_G_FMT,
            &format
        ) != 0
    )
    {
        ESP_LOGE(
            TAG,
            "VIDIOC_G_FMT failed: errno=%d (%s)",
            errno,
            strerror(errno)
        );

        close(video_fd);

        video_fd = -1;

        return ESP_FAIL;
    }


    camera_width =
        format.fmt.pix.width;

    camera_height =
        format.fmt.pix.height;

    camera_pixel_format =
        format.fmt.pix.pixelformat;


    ESP_LOGI(
        TAG,
        "Initial format: %lu x %lu",
        (unsigned long)camera_width,
        (unsigned long)camera_height
    );

    ESP_LOGI(
        TAG,
        "Initial pixel format: 0x%08lx",
        (unsigned long)camera_pixel_format
    );


    /*
     * =====================================================
     * Request RGB565
     * =====================================================
     */

    if (camera_pixel_format != V4L2_PIX_FMT_RGB565)
    {
        struct v4l2_format s_format;
        memset(
            &s_format,
            0,
            sizeof(s_format)
        );

        s_format.type =
            V4L2_BUF_TYPE_VIDEO_CAPTURE;

        s_format.fmt.pix.width =
            camera_width;

        s_format.fmt.pix.height =
            camera_height;

        s_format.fmt.pix.pixelformat =
            V4L2_PIX_FMT_RGB565;

        if (
            ioctl(
                video_fd,
                VIDIOC_S_FMT,
                &s_format
            ) != 0
        )
        {
            ESP_LOGE(
                TAG,
                "VIDIOC_S_FMT RGB565 failed: errno=%d (%s)",
                errno,
                strerror(errno)
            );

            close(video_fd);

            video_fd = -1;

            return ESP_FAIL;
        }
    }
    else
    {
        ESP_LOGI(
            TAG,
            "Camera is already in RGB565 format, skipping VIDIOC_S_FMT"
        );
    }


    /*
     * =====================================================
     * Read actual format back
     * =====================================================
     */

    memset(
        &format,
        0,
        sizeof(format)
    );


    format.type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;


    if (
        ioctl(
            video_fd,
            VIDIOC_G_FMT,
            &format
        ) != 0
    )
    {
        ESP_LOGE(
            TAG,
            "VIDIOC_G_FMT final failed: errno=%d (%s)",
            errno,
            strerror(errno)
        );

        close(video_fd);

        video_fd = -1;

        return ESP_FAIL;
    }


    camera_width =
        format.fmt.pix.width;

    camera_height =
        format.fmt.pix.height;

    camera_pixel_format =
        format.fmt.pix.pixelformat;


    ESP_LOGI(
        TAG,
        "Final camera resolution: %lu x %lu",
        (unsigned long)camera_width,
        (unsigned long)camera_height
    );

    ESP_LOGI(
        TAG,
        "Final pixel format: 0x%08lx",
        (unsigned long)camera_pixel_format
    );


    /*
     * Verify RGB565.
     */

    if (
        camera_pixel_format !=
        V4L2_PIX_FMT_RGB565
    )
    {
        ESP_LOGE(
            TAG,
            "Camera did not accept RGB565"
        );

        ESP_LOGE(
            TAG,
            "Actual format = 0x%08lx",
            (unsigned long)camera_pixel_format
        );

        close(video_fd);

        video_fd = -1;

        return ESP_ERR_NOT_SUPPORTED;
    }


    return ESP_OK;
}


/*
 * =========================================================
 * Setup V4L2 buffers
 * =========================================================
 */

static esp_err_t camera_setup_buffers(void)
{
    struct v4l2_requestbuffers request;

    memset(
        &request,
        0,
        sizeof(request)
    );


    request.count =
        CAMERA_NUM_BUFS;

    request.type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;

    request.memory =
        V4L2_MEMORY_MMAP;


    if (
        ioctl(
            video_fd,
            VIDIOC_REQBUFS,
            &request
        ) != 0
    )
    {
        ESP_LOGE(
            TAG,
            "VIDIOC_REQBUFS failed: errno=%d (%s)",
            errno,
            strerror(errno)
        );

        return ESP_FAIL;
    }


    if (
        request.count <
        CAMERA_NUM_BUFS
    )
    {
        ESP_LOGE(
            TAG,
            "Camera returned only %u buffers",
            request.count
        );

        return ESP_ERR_NO_MEM;
    }


    /*
     * Map every camera buffer.
     */

    for (
        int i = 0;
        i < CAMERA_NUM_BUFS;
        i++
    )
    {
        struct v4l2_buffer buffer;

        memset(
            &buffer,
            0,
            sizeof(buffer)
        );


        buffer.type =
            V4L2_BUF_TYPE_VIDEO_CAPTURE;

        buffer.memory =
            V4L2_MEMORY_MMAP;

        buffer.index =
            i;


        if (
            ioctl(
                video_fd,
                VIDIOC_QUERYBUF,
                &buffer
            ) != 0
        )
        {
            ESP_LOGE(
                TAG,
                "VIDIOC_QUERYBUF failed: %d",
                i
            );

            return ESP_FAIL;
        }


        if (
            i == 0
        )
        {
            camera_buffer_size =
                buffer.length;
        }


        camera_buffers[i] =
            mmap(
                NULL,
                buffer.length,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,
                video_fd,
                buffer.m.offset
            );


        if (
            camera_buffers[i] ==
            MAP_FAILED
        )
        {
            camera_buffers[i] =
                NULL;

            ESP_LOGE(
                TAG,
                "mmap failed for buffer %d: errno=%d (%s)",
                i,
                errno,
                strerror(errno)
            );

            return ESP_FAIL;
        }


        ESP_LOGI(
            TAG,
            "Camera buffer %d mapped: %u bytes",
            i,
            (unsigned)buffer.length
        );


        /*
         * Queue buffer.
         */

        if (
            ioctl(
                video_fd,
                VIDIOC_QBUF,
                &buffer
            ) != 0
        )
        {
            ESP_LOGE(
                TAG,
                "VIDIOC_QBUF failed: %d",
                i
            );

            return ESP_FAIL;
        }
    }


    ESP_LOGI(
        TAG,
        "V4L2 buffers ready"
    );


    return ESP_OK;
}


/*
 * =========================================================
 * Allocate display buffers
 * =========================================================
 */

static esp_err_t camera_allocate_display_buffers(void)
{
    const size_t alignment =
        CAMERA_CACHE_ALIGNMENT;


    /*
     * RGB565 = 2 bytes / pixel
     */

    int cur_lcd_w = display_get_width();
    int cur_lcd_h = display_get_height();
    if (cur_lcd_w <= 0) cur_lcd_w = BSP_LCD_H_RES;
    if (cur_lcd_h <= 0) cur_lcd_h = BSP_LCD_V_RES;

    size_t required_size =
        (
            (size_t)cur_lcd_w *
            (size_t)cur_lcd_h *
            2
        );

    display_buffer_size =
        ALIGN_UP(
            required_size,
            alignment
        );

    ESP_LOGI(
        TAG,
        "Display resolution: %dx%d",
        cur_lcd_w,
        cur_lcd_h
    );

    ESP_LOGI(
        TAG,
        "Display buffer size: %u bytes",
        (unsigned)display_buffer_size
    );


    /*
     * Allocate two buffers.
     */

    for (
        int i = 0;
        i < CAMERA_NUM_BUFS;
        i++
    )
    {
        display_buffers[i] =
            heap_caps_aligned_calloc(
                alignment,
                1,
                display_buffer_size,
                MALLOC_CAP_SPIRAM
            );


        if (
            display_buffers[i] ==
            NULL
        )
        {
            ESP_LOGE(
                TAG,
                "Failed to allocate display buffer %d",
                i
            );


            /*
             * Cleanup already allocated buffers.
             */

            for (
                int j = 0;
                j < i;
                j++
            )
            {
                if (
                    display_buffers[j] !=
                    NULL
                )
                {
                    heap_caps_free(
                        display_buffers[j]
                    );

                    display_buffers[j] =
                        NULL;
                }
            }


            return ESP_ERR_NO_MEM;
        }


        ESP_LOGI(
            TAG,
            "Display buffer %d allocated: %u bytes",
            i,
            (unsigned)display_buffer_size
        );
    }

    current_display_buffer =
        0;

    return ESP_OK;
}


/*
 * =========================================================
 * Process frame
 * =========================================================
 */

static esp_err_t camera_process_frame(
    uint8_t *input,
    uint32_t input_width,
    uint32_t input_height,
    uint8_t *output,
    size_t output_size
)
{
    if (
        input == NULL ||
        output == NULL
    )
    {
        return ESP_ERR_INVALID_ARG;
    }


    /*
     * =====================================================
     * Camera frame -> LCD
     *
     * Do exactly ONE PPA operation.
     *
     * The output is directly the LCD resolution.
     * =====================================================
     */

    uint32_t lcd_w = (uint32_t)display_get_width();
    uint32_t lcd_h = (uint32_t)display_get_height();
    if (lcd_w == 0) lcd_w = BSP_LCD_H_RES;
    if (lcd_h == 0) lcd_h = BSP_LCD_V_RES;

    uint32_t crop_w = input_width;
    uint32_t crop_h = input_height;
    uint32_t offset_x = 0;
    uint32_t offset_y = 0;

    uint32_t ideal_crop_w = (input_height * lcd_w) / lcd_h;
    if (ideal_crop_w <= input_width) {
        crop_w = ideal_crop_w & ~1;
        offset_x = (input_width - crop_w) / 2;
    } else {
        uint32_t ideal_crop_h = (input_width * lcd_h) / lcd_w;
        crop_h = ideal_crop_h & ~1;
        offset_y = (input_height - crop_h) / 2;
    }

    float scale_x = (float)lcd_w / (float)crop_w;
    float scale_y = (float)lcd_h / (float)crop_h;

    ppa_srm_oper_config_t config = {
        .in.buffer =
            input,

        .in.pic_w =
            input_width,

        .in.pic_h =
            input_height,

        .in.block_w =
            crop_w,

        .in.block_h =
            crop_h,

        .in.block_offset_x =
            offset_x,

        .in.block_offset_y =
            offset_y,

        .in.srm_cm =
            PPA_SRM_COLOR_MODE_RGB565,

        .out.buffer =
            output,

        .out.buffer_size =
            output_size,

        .out.pic_w =
            lcd_w,

        .out.pic_h =
            lcd_h,

        .out.block_offset_x =
            0,

        .out.block_offset_y =
            0,

        .out.srm_cm =
            PPA_SRM_COLOR_MODE_RGB565,

        .rotation_angle =
            PPA_SRM_ROTATION_ANGLE_0,

        .scale_x =
            scale_x,

        .scale_y =
            scale_y,

        .rgb_swap =
            0,

        .byte_swap =
            0,

        .mode =
            PPA_TRANS_MODE_BLOCKING,
    };


    /*
     * Camera rotation.
     */

    switch (
        BSP_CAMERA_ROTATION
    )
    {
        case 0:

            config.rotation_angle =
                PPA_SRM_ROTATION_ANGLE_0;

            break;


        case 90:

            config.rotation_angle =
                PPA_SRM_ROTATION_ANGLE_90;

            break;


        case 180:

            config.rotation_angle =
                PPA_SRM_ROTATION_ANGLE_180;

            break;


        case 270:

            config.rotation_angle =
                PPA_SRM_ROTATION_ANGLE_270;

            break;


        default:

            ESP_LOGW(
                TAG,
                "Unsupported camera rotation: %d",
                BSP_CAMERA_ROTATION
            );

            config.rotation_angle =
                PPA_SRM_ROTATION_ANGLE_0;

            break;
    }


    /*
     * Rotation 90/270 changes the effective dimensions.
     */

    if (
        BSP_CAMERA_ROTATION == 90 ||
        BSP_CAMERA_ROTATION == 270
    )
    {
        config.scale_x =
            (
                (float)BSP_LCD_H_RES /
                (float)input_height
            );

        config.scale_y =
            (
                (float)BSP_LCD_V_RES /
                (float)input_width
            );
    }


    /*
     * Run PPA.
     */

    esp_err_t ret =
        ppa_do_scale_rotate_mirror(
            ppa_handle,
            &config
        );


    if (ret != ESP_OK)
    {
        ESP_LOGE(
            TAG,
            "PPA processing failed: %s",
            esp_err_to_name(ret)
        );

        return ret;
    }


    return ESP_OK;
}


/*
 * =========================================================
 * Receive camera frame
 * =========================================================
 */

static esp_err_t camera_receive_frame(
    struct v4l2_buffer *buffer,
    bool quiet
)
{
    memset(
        buffer,
        0,
        sizeof(*buffer)
    );


    buffer->type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;

    buffer->memory =
        V4L2_MEMORY_MMAP;


    if (
        ioctl(
            video_fd,
            VIDIOC_DQBUF,
            buffer
        ) != 0
    )
    {
        if (!quiet)
        {
            ESP_LOGE(
                TAG,
                "VIDIOC_DQBUF failed: errno=%d (%s)",
                errno,
                strerror(errno)
            );
        }

        return ESP_FAIL;
    }


    return ESP_OK;
}


/*
 * This module is the sole owner of video_fd.  STREAMOFF is issued before
 * releasing MMAP buffers so a task blocked in VIDIOC_DQBUF can exit safely.
 */
static void camera_stop_stream(void)
{
    if (!video_streaming || video_fd < 0)
    {
        return;
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (ioctl(video_fd, VIDIOC_STREAMOFF, &type) != 0)
    {
        ESP_LOGE(
            TAG,
            "VIDIOC_STREAMOFF failed: errno=%d (%s)",
            errno,
            strerror(errno)
        );
        return;
    }

    video_streaming = false;
    ESP_LOGI(TAG, "V4L2 camera stream stopped");
}


/*
 * =========================================================
 * DQBUF timeout
 *
 * Task mode   : wait up to CAMERA_DQBUF_TIMEOUT_MS so a
 *               stopping task can notice video_task_stop.
 * Polled mode : 0 ms, the single-thread loop must never
 *               block on the camera.
 * =========================================================
 */

static int dqbuf_timeout_ms = -1;

static void camera_set_dqbuf_timeout(int timeout_ms)
{
    if (video_fd < 0 || dqbuf_timeout_ms == timeout_ms)
    {
        return;
    }

    struct timeval tv = {
        .tv_sec = timeout_ms / 1000,
        .tv_usec = (timeout_ms % 1000) * 1000,
    };

    if (ioctl(video_fd, VIDIOC_S_DQBUF_TIMEOUT, &tv) != 0)
    {
        ESP_LOGW(
            TAG,
            "VIDIOC_S_DQBUF_TIMEOUT failed: errno=%d (%s)",
            errno,
            strerror(errno)
        );
        return;
    }

    dqbuf_timeout_ms = timeout_ms;
}


/*
 * =========================================================
 * Handle one frame
 *
 * DQBUF -> PPA scale -> display -> QBUF
 *
 * Shared by the camera task (multi-thread) and
 * camera_poll() (single-thread).
 *
 * Returns true if a frame was processed.
 * =========================================================
 */

static int64_t last_frame_us = 0;

static bool camera_handle_one_frame(bool quiet_timeout)
{
    struct v4l2_buffer buffer;

    if (
        camera_receive_frame(
            &buffer,
            quiet_timeout || video_task_stop
        ) != ESP_OK
    )
    {
        return false;
    }

    /*
     * Check index.
     */

    if (
        buffer.index >=
        CAMERA_NUM_BUFS
    )
    {
        ESP_LOGE(
            TAG,
            "Invalid camera buffer index: %u",
            buffer.index
        );

        if (ioctl(video_fd, VIDIOC_QBUF, &buffer) != 0)
        {
            ESP_LOGE(
                TAG,
                "VIDIOC_QBUF invalid buffer failed: errno=%d (%s)",
                errno,
                strerror(errno)
            );
        }

        return false;
    }


    uint8_t camera_index =
        (uint8_t)buffer.index;


    /*
     * Use the other display buffer.
     */

    uint8_t display_index =
        current_display_buffer;

    current_display_buffer ^= 1;


    /*
     * Process frame with hardware PPA.
     */

    int64_t t_ppa = esp_timer_get_time();

    esp_err_t ret =
        camera_process_frame(
            camera_buffers[camera_index],

            camera_width,

            camera_height,

            display_buffers[
                display_index
            ],

            display_buffer_size
        );

    bool processed =
        (ret == ESP_OK);

    if (processed)
    {
        bench_record(BENCH_M_CAM_PPA, esp_timer_get_time() - t_ppa);
    }


    if (processed)
    {
        /*
         * Update latest frame.
         */

        frame_buffer =
            display_buffers[
                display_index
            ];


        int cur_w = display_get_width();
        int cur_h = display_get_height();
        if (cur_w <= 0) cur_w = BSP_LCD_H_RES;
        if (cur_h <= 0) cur_h = BSP_LCD_V_RES;

        frame_size = (size_t)cur_w * (size_t)cur_h * 2;


        /*
         * Send to display (waits for the LVGL lock).
         */

        int64_t t_show = esp_timer_get_time();

        ret =
            display_show_frame(
                (
                    const uint8_t *
                )display_buffers[
                    display_index
                ],

                frame_size
            );

        bench_record(BENCH_M_CAM_SHOW, esp_timer_get_time() - t_show);


        if (
            ret != ESP_OK
        )
        {
            ESP_LOGW(
                TAG,
                "display_show_frame failed: %s",
                esp_err_to_name(ret)
            );
        }
    }


    /*
     * Return camera buffer.
     */

    if (
        ioctl(
            video_fd,
            VIDIOC_QBUF,
            &buffer
        ) != 0
    )
    {
        ESP_LOGE(
            TAG,
            "VIDIOC_QBUF failed: errno=%d (%s)",
            errno,
            strerror(errno)
        );
    }


    /*
     * Benchmark: frame-to-frame period.
     */

    if (processed)
    {
        int64_t t_end = esp_timer_get_time();

        if (last_frame_us > 0)
        {
            bench_record(BENCH_M_CAM_PERIOD, t_end - last_frame_us);
        }

        last_frame_us = t_end;
    }


    return processed;
}


/*
 * =========================================================
 * Video task
 * =========================================================
 */

static void camera_video_task(
    void *arg
)
{
    (void)arg;


    ESP_LOGI(
        TAG,
        "Camera video task started (core %d)",
        (int)xPortGetCoreID()
    );


    while (
        !video_task_stop
    )
    {
        if (!camera_handle_one_frame(false))
        {
            if (video_task_stop)
            {
                break;
            }

            vTaskDelay(
                pdMS_TO_TICKS(10)
            );
        }
    }


    /*
     * Clear the handle before notifying, so a restart
     * right after camera_task_stop() sees the task gone.
     */

    video_task_handle =
        NULL;


    ESP_LOGI(
        TAG,
        "Camera video task stopped"
    );


    if (
        video_stop_sem != NULL
    )
    {
        xSemaphoreGive(
            video_stop_sem
        );
    }


    vTaskDelete(NULL);
}


/*
 * =========================================================
 * Start / stop video task
 * =========================================================
 */

esp_err_t camera_task_start(int core_id)
{
    if (!video_streaming)
    {
        return ESP_ERR_INVALID_STATE;
    }

    if (video_task_handle != NULL)
    {
        return ESP_OK;
    }


    camera_set_dqbuf_timeout(CAMERA_DQBUF_TIMEOUT_MS);

    video_task_stop = false;

    /* Drop a stale stop notification, if any. */
    xSemaphoreTake(video_stop_sem, 0);


    BaseType_t task_ret =
        xTaskCreatePinnedToCore(
            camera_video_task,

            "camera_video",

            CAMERA_TASK_STACK_SIZE,

            NULL,

            CAMERA_TASK_PRIORITY,

            &video_task_handle,

            core_id
        );


    if (task_ret != pdPASS)
    {
        ESP_LOGE(
            TAG,
            "Failed to create camera video task"
        );

        video_task_handle = NULL;

        return ESP_FAIL;
    }


    return ESP_OK;
}


void camera_task_stop(void)
{
    if (video_task_handle == NULL)
    {
        return;
    }


    video_task_stop = true;


    if (xSemaphoreTake(
        video_stop_sem,
        pdMS_TO_TICKS(1000)
    ) != pdTRUE)
    {
        ESP_LOGE(
            TAG,
            "Camera video task did not stop"
        );
        return;
    }


    video_task_stop = false;
}


/*
 * =========================================================
 * Poll one frame (single-thread mode)
 * =========================================================
 */

bool camera_poll(uint32_t timeout_ms)
{
    if (
        !camera_initialized ||
        !video_streaming ||
        video_task_handle != NULL
    )
    {
        return false;
    }


    camera_set_dqbuf_timeout((int)timeout_ms);


    return camera_handle_one_frame(true);
}


/*
 * =========================================================
 * Camera initialization
 * =========================================================
 */

esp_err_t camera_init(void)
{
    if (camera_initialized)
    {
        return ESP_ERR_INVALID_STATE;
    }

    ESP_LOGI(
        TAG,
        "========================================"
    );

    ESP_LOGI(
        TAG,
        "Initializing camera..."
    );

    ESP_LOGI(
        TAG,
        "BSP_CAMERA_DEVICE = %s",
        BSP_CAMERA_DEVICE
    );


    /*
     * =====================================================
     * 1. Start BSP camera
     * =====================================================
     */

    ESP_LOGI(
        TAG,
        "Calling bsp_camera_start(NULL)..."
    );


    esp_err_t ret =
        bsp_camera_start(NULL);


    ESP_LOGI(
        TAG,
        "bsp_camera_start returned: %s (0x%x)",
        esp_err_to_name(ret),
        ret
    );


    if (
        ret != ESP_OK
    )
    {
        ESP_LOGE(
            TAG,
            "bsp_camera_start failed"
        );

        goto cleanup_ppa;
    }


    /*
     * Allow driver/device registration.
     */

    vTaskDelay(
        pdMS_TO_TICKS(100)
    );


    /*
     * =====================================================
     * 2. Check V4L2 device
     * =====================================================
     */

    ESP_LOGI(
        TAG,
        "Checking V4L2 device: %s",
        BSP_CAMERA_DEVICE
    );


    if (
        false && access(
            BSP_CAMERA_DEVICE,
            F_OK
        ) != 0
    )
    {
        ESP_LOGE(
            TAG,
            "V4L2 device does not exist"
        );

        ESP_LOGE(
            TAG,
            "errno = %d (%s)",
            errno,
            strerror(errno)
        );

        return ESP_FAIL;
    }


    ESP_LOGI(
        TAG,
        "V4L2 device exists"
    );


    /*
     * =====================================================
     * 3. Initialize PPA
     * =====================================================
     */

    ret =
        camera_ppa_init();


    if (
        ret != ESP_OK
    )
    {
        ESP_LOGE(
            TAG,
            "PPA initialization failed: %s",
            esp_err_to_name(ret)
        );

        return ret;
    }


    /*
     * =====================================================
     * 4. Allocate display buffers
     * =====================================================
     */

    ret =
        camera_allocate_display_buffers();


    if (
        ret != ESP_OK
    )
    {
        ESP_LOGE(
            TAG,
            "Display buffer allocation failed: %s",
            esp_err_to_name(ret)
        );

        goto cleanup_display;
    }


    /*
     * =====================================================
     * 5. Open camera
     * =====================================================
     */

    ret =
        camera_open_device();


    if (
        ret != ESP_OK
    )
    {
        ESP_LOGE(
            TAG,
            "Camera open failed: %s",
            esp_err_to_name(ret)
        );

        goto cleanup_display;
    }


    /*
     * =====================================================
     * 6. Setup V4L2 buffers
     * =====================================================
     */

    ret =
        camera_setup_buffers();


    if (
        ret != ESP_OK
    )
    {
        ESP_LOGE(
            TAG,
            "Camera buffer setup failed: %s",
            esp_err_to_name(ret)
        );

        goto cleanup_device;
    }


    /*
     * =====================================================
     * 7. Create stop semaphore
     * =====================================================
     */

    video_stop_sem =
        xSemaphoreCreateBinary();


    if (
        video_stop_sem == NULL
    )
    {
        ESP_LOGE(
            TAG,
            "Failed to create video semaphore"
        );

        ret =
            ESP_ERR_NO_MEM;

        goto cleanup_camera_buffers;
    }


    /*
     * =====================================================
     * 8. Start V4L2 stream
     * =====================================================
     */

    enum v4l2_buf_type type =
        V4L2_BUF_TYPE_VIDEO_CAPTURE;


    if (
        ioctl(
            video_fd,
            VIDIOC_STREAMON,
            &type
        ) != 0
    )
    {
        ESP_LOGE(
            TAG,
            "VIDIOC_STREAMON failed"
        );

        ESP_LOGE(
            TAG,
            "errno = %d (%s)",
            errno,
            strerror(errno)
        );

        ret =
            ESP_FAIL;

        goto cleanup_semaphore;
    }


    ESP_LOGI(
        TAG,
        "V4L2 camera stream started"
    );

    video_streaming = true;


    /*
     * =====================================================
     * 9. Start video task
     * =====================================================
     */

    video_task_stop =
        false;


    frame_ready =
        false;


    current_display_buffer =
        0;


    if (camera_task_start(CAMERA_TASK_CORE) != ESP_OK)
    {
        camera_stop_stream();

        ret = ESP_FAIL;

        goto cleanup_semaphore;
    }


    /*
     * =====================================================
     * 10. Camera ready
     * =====================================================
     */

    camera_initialized = true;

    ESP_LOGI(
        TAG,
        "========================================"
    );

    ESP_LOGI(
        TAG,
        "Camera initialized successfully"
    );

    ESP_LOGI(
        TAG,
        "Camera device : %s",
        BSP_CAMERA_DEVICE
    );

    ESP_LOGI(
        TAG,
        "Camera size   : %lu x %lu",
        (unsigned long)camera_width,
        (unsigned long)camera_height
    );

    ESP_LOGI(
        TAG,
        "Display size  : %d x %d",
        BSP_LCD_H_RES,
        BSP_LCD_V_RES
    );

    ESP_LOGI(
        TAG,
        "Pixel format  : RGB565"
    );

    ESP_LOGI(
        TAG,
        "Stream        : RUNNING"
    );

    ESP_LOGI(
        TAG,
        "========================================"
    );

    return ESP_OK;


/*
 * =========================================================
 * Error cleanup
 * =========================================================
 */

cleanup_semaphore:

    if (video_stop_sem != NULL)
    {
        vSemaphoreDelete(video_stop_sem);
        video_stop_sem = NULL;
    }


cleanup_camera_buffers:

    for (int i = 0; i < CAMERA_NUM_BUFS; i++)
    {
        if (camera_buffers[i] != NULL)
        {
            munmap(camera_buffers[i], camera_buffer_size);
            camera_buffers[i] = NULL;
        }
    }


cleanup_device:

    if (video_fd >= 0)
    {
        close(video_fd);
        video_fd = -1;
    }


cleanup_display:
    for (
        int i = 0;
        i < CAMERA_NUM_BUFS;
        i++
    )
    {
        if (
            display_buffers[i] !=
            NULL
        )
        {
            heap_caps_free(
                display_buffers[i]
            );

            display_buffers[i] =
                NULL;
        }
    }


cleanup_ppa:

    if (ppa_handle != NULL)
    {
        ppa_unregister_client(ppa_handle);
        ppa_handle = NULL;
    }


    return ret;
}


/*
 * =========================================================
 * Camera capture
 * =========================================================
 */

esp_err_t camera_capture(void)
{
    if (
        !camera_initialized
    )
    {
        return ESP_ERR_INVALID_STATE;
    }


    /*
     * Camera continuously captures in
     * background.
     *
     * Wait until at least one processed
     * frame is available.
     */

    int timeout_ms =
        1000;


    while (
        !frame_ready &&
        timeout_ms > 0
    )
    {
        vTaskDelay(
            pdMS_TO_TICKS(10)
        );

        timeout_ms -=
            10;
    }


    if (
        !frame_ready
    )
    {
        return ESP_ERR_TIMEOUT;
    }


    return ESP_OK;
}


/*
 * =========================================================
 * Get frame
 * =========================================================
 */

const uint8_t *camera_get_frame(void)
{
    return (
        const uint8_t *
    )frame_buffer;
}


/*
 * =========================================================
 * Get frame size
 * =========================================================
 */

size_t camera_get_frame_size(void)
{
    return frame_size;
}


/*
 * =========================================================
 * Get width
 * =========================================================
 */

uint32_t camera_get_width(void)
{
    return BSP_LCD_H_RES;
}


/*
 * =========================================================
 * Get height
 * =========================================================
 */

uint32_t camera_get_height(void)
{
    return BSP_LCD_V_RES;
}


/*
 * =========================================================
 * Release frame
 * =========================================================
 */

void camera_release_frame(void)
{
    /*
     * The buffer is owned by the camera/display
     * pipeline.
     *
     * Do NOT free it.
     */

    frame_ready =
        false;
}


/*
 * =========================================================
 * Camera deinitialization
 * =========================================================
 */

void camera_deinit(void)
{
    if (
        !camera_initialized
    )
    {
        return;
    }


    ESP_LOGI(
        TAG,
        "Stopping camera..."
    );


    /*
     * Tell video task to stop.
     *
     * In single-thread (polled) mode there is no task.
     */

    bool task_running =
        (video_task_handle != NULL);

    video_task_stop =
        true;


    /* STREAMOFF wakes a pending VIDIOC_DQBUF before MMAP buffers are freed. */
    camera_stop_stream();


    /*
     * Wait for task.
     */

    if (
        task_running &&
        video_stop_sem != NULL
    )
    {
        if (xSemaphoreTake(
            video_stop_sem,
            pdMS_TO_TICKS(1000)
        ) != pdTRUE)
        {
            ESP_LOGE(
                TAG,
                "Camera video task did not stop; V4L2 buffers remain mapped"
            );
            return;
        }
    }


    /*
     * =====================================================
     * Unmap V4L2 buffers
     * =====================================================
     */

    for (
        int i = 0;
        i < CAMERA_NUM_BUFS;
        i++
    )
    {
        if (
            camera_buffers[i] !=
            NULL
        )
        {
            munmap(
                camera_buffers[i],
                camera_buffer_size
            );

            camera_buffers[i] =
                NULL;
        }
    }


    /*
     * =====================================================
     * Close V4L2 device
     * =====================================================
     */

    if (
        video_fd >= 0
    )
    {
        close(
            video_fd
        );

        video_fd =
            -1;
    }


    /*
     * =====================================================
     * Free display buffers
     * =====================================================
     */

    for (
        int i = 0;
        i < CAMERA_NUM_BUFS;
        i++
    )
    {
        if (
            display_buffers[i] !=
            NULL
        )
        {
            heap_caps_free(
                display_buffers[i]
            );

            display_buffers[i] =
                NULL;
        }
    }


    /*
     * =====================================================
     * Delete semaphore
     * =====================================================
     */

    if (
        video_stop_sem != NULL
    )
    {
        vSemaphoreDelete(
            video_stop_sem
        );

        video_stop_sem =
            NULL;
    }


    if (ppa_handle != NULL)
    {
        esp_err_t ret = ppa_unregister_client(ppa_handle);

        if (ret != ESP_OK)
        {
            ESP_LOGW(
                TAG,
                "ppa_unregister_client failed: %s",
                esp_err_to_name(ret)
            );
        }

        ppa_handle = NULL;
    }


    /*
     * =====================================================
     * Reset state
     * =====================================================
     */

    frame_buffer =
        NULL;

    frame_size =
        0;

    frame_ready =
        false;

    video_task_handle =
        NULL;

    video_task_stop =
        false;

    video_streaming =
        false;

    camera_initialized =
        false;


    ESP_LOGI(
        TAG,
        "Camera deinitialized"
    );
}
