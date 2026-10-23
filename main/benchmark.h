#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * =========================================================
 * Single-thread vs Multi-thread benchmark
 * =========================================================
 *
 * Core split (both 2-core modes):
 *   Core 0 = mic + VAD + KWS + alert JPEG
 *   Core 1 = camera + PPA + LVGL render
 *
 * After boot the benchmark runs 4 modes back-to-back:
 *
 *   1. MULTI-THREAD, 2 CORES : one task per job (main, alert_cap on
 *                              Core 0; camera_video, lvgl_bench on Core 1),
 *                              fixed-priority preemptive scheduling
 *   2. MULTI-THREAD, 2 CORES, ROUND-ROBIN: same tasks and cores, all
 *                              at BENCH_RR_PRIORITY (time-sliced per tick)
 *   3. SINGLE-THREAD, 2 CORES: one loop per core (main on Core 0,
 *                              core1_loop on Core 1)
 *   4. SINGLE-THREAD, 1 CORE : one loop in main does everything (Core 0)
 *
 * Each mode: warm-up -> measure. At the end one result table
 * (columns: 1 Core | 2 Core | 2 Core MT | 2 Core MT RR) is printed and
 * the normal application (real VAD + KWS + alerts) continues.
 *
 * Set BENCH_ENABLE to 0 to build the normal application only.
 */
#define BENCH_ENABLE 0

/* 1 = also print a detailed report after each design */
#define BENCH_VERBOSE 0

#define BENCH_WARMUP_MS     3000
#define BENCH_PHASE_MS      20000

/* Synthetic KWS workload: one inference on the last 1 s of audio */
#define BENCH_KWS_PERIOD_MS 1000

/*
 * Deadlines (0 = none)
 *
 * Audio period : I2S DMA holds 8 x 96 samples = 48 ms of audio.
 *                If the loop takes longer, samples are lost.
 * Camera period: the display pipeline runs at ~15 FPS (66 ms),
 *                > 100 ms between frames = frame(s) skipped.
 * Alert period : capture every 100 ms, 120 ms allowed.
 */
#define BENCH_DEADLINE_AUDIO_PERIOD_US  48000
#define BENCH_DEADLINE_CAM_PERIOD_US    100000
#define BENCH_DEADLINE_ALERT_PERIOD_US  120000

/* Core 1 single loop: max wait for a camera frame before rendering */
#define BENCH_CORE1_FRAME_WAIT_MS 20

/*
 * Round-robin mode: every app task gets this same priority, so
 * FreeRTOS time-slices them every tick instead of preempting.
 */
#define BENCH_RR_PRIORITY 5

/* Run order */
typedef enum {
    BENCH_MODE_MULTI_2CORE = 0,   /* priority-based (normal priorities) */
    BENCH_MODE_MULTI_2CORE_RR,    /* same tasks, all equal priority     */
    BENCH_MODE_SINGLE_2CORE,
    BENCH_MODE_SINGLE_1CORE,
    BENCH_MODE_COUNT
} bench_mode_t;

typedef enum {
    BENCH_M_AUDIO_PERIOD = 0, /* mic read -> next mic read          */
    BENCH_M_AUDIO_PROC,       /* VAD + filter of one 32 ms frame    */
    BENCH_M_KWS_INFER,        /* TinyML inference (1 s window)      */
    BENCH_M_CAM_PERIOD,       /* frame -> next frame (FPS)          */
    BENCH_M_CAM_PPA,          /* PPA scale camera -> LCD size       */
    BENCH_M_CAM_SHOW,         /* hand frame to LVGL (incl. lock wait) */
    BENCH_M_ALERT_PERIOD,     /* alert capture -> next capture      */
    BENCH_M_ALERT_JPEG,       /* hardware JPEG encode of one frame  */
    BENCH_M_LVGL_RENDER,      /* lv_timer_handler() (screen render) */
    BENCH_M_COUNT
} bench_metric_t;

/* Record one sample (microseconds). Ignored outside a measure window. */
void bench_record(bench_metric_t metric, int64_t us);

/* Start the 3-mode benchmark sequence. Call from the main task. */
void bench_start(void);

/* True while the benchmark sequence is running. */
bool bench_is_running(void);

/* Feed mic samples into the 1 s ring used by the synthetic KWS load. */
void bench_feed_audio(const int16_t *samples, size_t count);

/* Call once per main-loop iteration while bench_is_running(). */
void bench_step(void);

#ifdef __cplusplus
}
#endif

#endif /* BENCHMARK_H */
