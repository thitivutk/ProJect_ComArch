#include "microphone.h"

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "driver/i2s_std.h"
#include "driver/i2c_master.h"

#include "esp_attr.h"
#include "esp_log.h"
#include "esp_err.h"

#include "esp_codec_dev.h"
#include "esp_codec_dev_defaults.h"

#include "bsp/esp-bsp.h"

static const char *TAG = "MIC";

/* =========================================================
 * ESP32-P4 Built-in Audio Configuration
 * ========================================================= */

#define MIC_I2S_NUM        I2S_NUM_0

#define MIC_I2C_NUM        0

#define MIC_I2C_SCL_IO     8
#define MIC_I2C_SDA_IO     7

#define MIC_I2S_MCK_IO     13
#define MIC_I2S_BCK_IO     12
#define MIC_I2S_WS_IO      10
#define MIC_I2S_DOUT_IO    9
#define MIC_I2S_DIN_IO     11

#define MIC_SAMPLE_RATE    16000
#define MIC_BITS           16

#define MIC_DMA_DESC_NUM   8
#define MIC_DMA_FRAME_NUM  96

/* ES8311 */
#define MIC_PA_CTRL_IO     53

#define MIC_MCLK_MULTIPLE  256

/* =========================================================
 * Handles
 * ========================================================= */

static i2s_chan_handle_t i2s_rx_handle = NULL;

static i2c_master_bus_handle_t i2c_bus_handle = NULL;

static esp_codec_dev_handle_t codec_handle = NULL;

/*
 * Number of I2S RX queue overflows = audio lost because
 * the application did not read in time.
 */
static volatile uint32_t rx_overflow_count = 0;


static bool IRAM_ATTR microphone_on_recv_q_ovf(
    i2s_chan_handle_t handle,
    i2s_event_data_t *event,
    void *user_ctx
)
{
    rx_overflow_count++;

    return false;
}


/* =========================================================
 * I2S initialization
 * ========================================================= */

static esp_err_t microphone_i2s_init(void)
{
    ESP_LOGI(TAG, "Initializing I2S...");

    i2s_chan_config_t chan_cfg =
        I2S_CHANNEL_DEFAULT_CONFIG(
            MIC_I2S_NUM,
            I2S_ROLE_MASTER
        );

    chan_cfg.dma_desc_num = MIC_DMA_DESC_NUM;
    chan_cfg.dma_frame_num = MIC_DMA_FRAME_NUM;
    chan_cfg.auto_clear = true;

    ESP_ERROR_CHECK(
        i2s_new_channel(
            &chan_cfg,
            NULL,
            &i2s_rx_handle
        )
    );

    i2s_std_config_t std_cfg = {

        .clk_cfg =
            I2S_STD_CLK_DEFAULT_CONFIG(
                MIC_SAMPLE_RATE
            ),

        .slot_cfg =
            I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
                MIC_BITS,
                I2S_SLOT_MODE_MONO
            ),

        .gpio_cfg = {

            .mclk = MIC_I2S_MCK_IO,

            .bclk = MIC_I2S_BCK_IO,

            .ws = MIC_I2S_WS_IO,

            .dout = MIC_I2S_DOUT_IO,

            .din = MIC_I2S_DIN_IO,

            .invert_flags = {

                .mclk_inv = false,

                .bclk_inv = false,

                .ws_inv = false,
            },
        },
    };

    std_cfg.clk_cfg.mclk_multiple =
        MIC_MCLK_MULTIPLE;

    ESP_ERROR_CHECK(
        i2s_channel_init_std_mode(
            i2s_rx_handle,
            &std_cfg
        )
    );

    /* Must be registered before the channel is enabled */
    i2s_event_callbacks_t cbs = {
        .on_recv_q_ovf = microphone_on_recv_q_ovf,
    };

    ESP_ERROR_CHECK(
        i2s_channel_register_event_callback(
            i2s_rx_handle,
            &cbs,
            NULL
        )
    );

    ESP_ERROR_CHECK(
        i2s_channel_enable(
            i2s_rx_handle
        )
    );

    ESP_LOGI(
        TAG,
        "I2S initialized: %d Hz / %d-bit",
        MIC_SAMPLE_RATE,
        MIC_BITS
    );

    return ESP_OK;
}


/* =========================================================
 * I2C initialization
 * ========================================================= */

static esp_err_t microphone_i2c_init(void)
{
    ESP_LOGI(TAG, "Getting shared BSP I2C bus handle...");

    i2c_bus_handle = bsp_i2c_get_handle();

    if (i2c_bus_handle == NULL) {
        esp_err_t ret = bsp_i2c_init();
        if (ret != ESP_OK) {
            ESP_LOGE(TAG, "bsp_i2c_init failed: %s", esp_err_to_name(ret));
            return ret;
        }
        i2c_bus_handle = bsp_i2c_get_handle();
    }

    return (i2c_bus_handle != NULL) ? ESP_OK : ESP_FAIL;
}


/* =========================================================
 * ES8311 initialization
 * ========================================================= */

static esp_err_t microphone_codec_init(void)
{
    ESP_LOGI(TAG, "Initializing ES8311 codec...");

    audio_codec_i2c_cfg_t i2c_cfg = {

        .port = MIC_I2C_NUM,

        .addr = ES8311_CODEC_DEFAULT_ADDR,

        .bus_handle = i2c_bus_handle,
    };

    const audio_codec_ctrl_if_t *ctrl_if =
        audio_codec_new_i2c_ctrl(&i2c_cfg);

    if (ctrl_if == NULL) {

        ESP_LOGE(
            TAG,
            "Failed to create I2C codec interface"
        );

        return ESP_FAIL;
    }


    audio_codec_i2s_cfg_t i2s_cfg = {

        .port = MIC_I2S_NUM,

        .rx_handle = i2s_rx_handle,

        .tx_handle = NULL,
    };

    const audio_codec_data_if_t *data_if =
        audio_codec_new_i2s_data(&i2s_cfg);

    if (data_if == NULL) {

        ESP_LOGE(
            TAG,
            "Failed to create I2S codec interface"
        );

        return ESP_FAIL;
    }


    const audio_codec_gpio_if_t *gpio_if =
        audio_codec_new_gpio();

    if (gpio_if == NULL) {

        ESP_LOGE(
            TAG,
            "Failed to create GPIO codec interface"
        );

        return ESP_FAIL;
    }


    es8311_codec_cfg_t codec_cfg = {

        .ctrl_if = ctrl_if,

        .gpio_if = gpio_if,

        .codec_mode =
            ESP_CODEC_DEV_WORK_MODE_ADC,

        .master_mode = false,

        .use_mclk = true,

        .pa_pin = MIC_PA_CTRL_IO,

        .pa_reverted = false,

        .hw_gain = {

            .pa_voltage = 5.0,

            .codec_dac_voltage = 3.3,
        },

        .mclk_div =
            MIC_MCLK_MULTIPLE,
    };


    const audio_codec_if_t *codec_if =
        es8311_codec_new(&codec_cfg);

    if (codec_if == NULL) {

        ESP_LOGE(
            TAG,
            "Failed to create ES8311 interface"
        );

        return ESP_FAIL;
    }


    esp_codec_dev_cfg_t dev_cfg = {

        .dev_type =
            ESP_CODEC_DEV_TYPE_IN,

        .codec_if = codec_if,

        .data_if = data_if,
    };


    codec_handle =
        esp_codec_dev_new(&dev_cfg);

    if (codec_handle == NULL) {

        ESP_LOGE(
            TAG,
            "Failed to create codec device"
        );

        return ESP_FAIL;
    }


    esp_codec_dev_sample_info_t sample_cfg = {

        .bits_per_sample = MIC_BITS,

        .channel = 1,

        .channel_mask = 0x01,

        .sample_rate = MIC_SAMPLE_RATE,
    };


    int ret =
        esp_codec_dev_open(
            codec_handle,
            &sample_cfg
        );

    if (ret != ESP_CODEC_DEV_OK) {

        ESP_LOGE(
            TAG,
            "Failed to open codec: %d",
            ret
        );

        return ESP_FAIL;
    }


    /*
     * Microphone gain
     *
     * เริ่มที่ 18 dB เหมือนตัวอย่าง
     */

    esp_codec_dev_set_in_gain(
        codec_handle,
        18
    );

    ESP_LOGI(
        TAG,
        "ES8311 microphone initialized"
    );

    return ESP_OK;
}


/* =========================================================
 * Public: microphone_init()
 * ========================================================= */

esp_err_t microphone_init(void)
{
    ESP_LOGI(TAG, "========================================");

    ESP_LOGI(TAG, "Initializing microphone");

    ESP_LOGI(TAG, "========================================");


    esp_err_t ret;


    ret = microphone_i2s_init();

    if (ret != ESP_OK) {

        ESP_LOGE(
            TAG,
            "I2S initialization failed"
        );

        return ret;
    }


    ret = microphone_i2c_init();

    if (ret != ESP_OK) {

        ESP_LOGE(
            TAG,
            "I2C initialization failed"
        );

        return ret;
    }


    ret = microphone_codec_init();

    if (ret != ESP_OK) {

        ESP_LOGE(
            TAG,
            "Codec initialization failed"
        );

        return ret;
    }


    ESP_LOGI(
        TAG,
        "Microphone initialization complete"
    );

    return ESP_OK;
}


/* =========================================================
 * Public: microphone_read()
 * ========================================================= */

esp_err_t microphone_read(
    int16_t *buffer,
    size_t buffer_size,
    size_t *bytes_read
)
{
    if (codec_handle == NULL) {

        return ESP_ERR_INVALID_STATE;
    }

    if (buffer == NULL) {

        return ESP_ERR_INVALID_ARG;
    }


    int ret =
        esp_codec_dev_read(
            codec_handle,
            buffer,
            buffer_size
        );


    if (ret != ESP_CODEC_DEV_OK) {

        ESP_LOGE(
            TAG,
            "Microphone read failed: %d",
            ret
        );

        return ESP_FAIL;
    }


    if (bytes_read != NULL) {

        *bytes_read = buffer_size;
    }


    return ESP_OK;
}


/* =========================================================
 * Public: microphone_get_overflow_count()
 * ========================================================= */

uint32_t microphone_get_overflow_count(void)
{
    return rx_overflow_count;
}


/* =========================================================
 * Public: microphone_deinit()
 * ========================================================= */

void microphone_deinit(void)
{
    if (codec_handle != NULL) {

        esp_codec_dev_close(
            codec_handle
        );

        esp_codec_dev_delete(
            codec_handle
        );

        codec_handle = NULL;
    }


    if (i2s_rx_handle != NULL) {

        i2s_channel_disable(
            i2s_rx_handle
        );

        i2s_del_channel(
            i2s_rx_handle
        );

        i2s_rx_handle = NULL;
    }


    i2c_bus_handle = NULL;


    ESP_LOGI(
        TAG,
        "Microphone deinitialized"
    );
}