#include "benchmark.h"

#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"

#include "alert_recorder.h"
#include "camera.h"
#include "display.h"
#include "kws_engine.h"
#include "microphone.h"

#if !configGENERATE_RUN_TIME_STATS || !configUSE_TRACE_FACILITY
#error "Benchmark needs CONFIG_FREERTOS_GENERATE_RUN_TIME_STATS=y and CONFIG_FREERTOS_USE_TRACE_FACILITY=y"
#endif

static const char *TAG = "BENCH";

/*
 * =========================================================
 * Response-time statistics
 * =========================================================
 */

/* 1 ms histogram buckets: [0,1) ... [255,256) ms + overflow */
#define BENCH_HIST_MAX_MS 256

typedef struct {
    uint32_t count;
    uint32_t miss;
    int64_t sum_us;
    int64_t min_us;
    int64_t max_us;
    uint32_t hist[BENCH_HIST_MAX_MS + 1];
} bench_stat_t;

typedef struct {
    uint32_t count;
    uint32_t miss;
    float avg_ms;
    float p99_ms;
    float max_ms;
    float total_ms;   /* time spent in this step during the phase */
} bench_stat_summary_t;

static const char *const s_metric_names[BENCH_M_COUNT] = {
    [BENCH_M_AUDIO_PERIOD] = "audio loop period",
    [BENCH_M_AUDIO_PROC]   = "audio VAD+filter",
    [BENCH_M_KWS_INFER]    = "KWS inference",
    [BENCH_M_CAM_PERIOD]   = "camera frame period",
    [BENCH_M_CAM_PPA]      = "PPA scale",
    [BENCH_M_CAM_SHOW]     = "hand frame to LVGL",
    [BENCH_M_ALERT_PERIOD] = "alert capture period",
    [BENCH_M_ALERT_JPEG]   = "alert JPEG encode",
    [BENCH_M_LVGL_RENDER]  = "LVGL render",
};

static const int64_t s_deadline_us[BENCH_M_COUNT] = {
    [BENCH_M_AUDIO_PERIOD] = BENCH_DEADLINE_AUDIO_PERIOD_US,
    [BENCH_M_CAM_PERIOD]   = BENCH_DEADLINE_CAM_PERIOD_US,
    [BENCH_M_ALERT_PERIOD] = BENCH_DEADLINE_ALERT_PERIOD_US,
};

/* Which jobs belong to which core in the 2-core design */
static const bench_metric_t s_core0_metrics[] = {
    BENCH_M_AUDIO_PERIOD, BENCH_M_AUDIO_PROC, BENCH_M_KWS_INFER,
    BENCH_M_ALERT_PERIOD, BENCH_M_ALERT_JPEG,
};

static const bench_metric_t s_core1_metrics[] = {
    BENCH_M_CAM_PERIOD, BENCH_M_CAM_PPA, BENCH_M_CAM_SHOW, BENCH_M_LVGL_RENDER,
};

#define ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))

/* Per-design detail report, off by default (summary table only) */
#define VPRINT(...) do { if (BENCH_VERBOSE) printf(__VA_ARGS__); } while (0)

static bench_stat_t s_stats[BENCH_M_COUNT];
static portMUX_TYPE s_stats_lock = portMUX_INITIALIZER_UNLOCKED;
static volatile bool s_recording = false;

/*
 * =========================================================
 * CPU load (FreeRTOS run-time stats)
 * =========================================================
 */

#define BENCH_MAX_TASKS 80

typedef struct {
    TaskStatus_t tasks[BENCH_MAX_TASKS];
    UBaseType_t count;
    int64_t time_us;
} bench_cpu_snapshot_t;

static bench_cpu_snapshot_t s_snap_start;
static bench_cpu_snapshot_t s_snap_end;

/* Tasks that belong to the application (everything else = system) */
static const char *const s_app_tasks[] = {
    "main", "camera_video", "alert_cap", "alert_upl", "lvgl_bench", "core1_loop",
};

/*
 * =========================================================
 * Benchmark sequence state
 * =========================================================
 */

typedef enum {
    BENCH_STATE_IDLE = 0,
    BENCH_STATE_WARMUP,
    BENCH_STATE_MEASURE,
} bench_state_t;

typedef struct {
    bool valid;
    float wall_s;
    float core_load[portNUM_PROCESSORS];      /* from idle time        */
    float core_app_pct[portNUM_PROCESSORS];   /* app tasks pinned here */
    float any_app_pct;                        /* app tasks, no affinity */
    float total_load;
    float app_pct;
    float cam_fps;
    float render_fps;
    uint32_t i2s_overflow;
    float task_pct[ARRAY_LEN(s_app_tasks)];   /* -1 = task not present */
    int task_core[ARRAY_LEN(s_app_tasks)];    /* -1 = no affinity      */
    float sys_pct;
    bench_stat_summary_t m[BENCH_M_COUNT];
} bench_mode_summary_t;

static const char *const s_mode_names[BENCH_MODE_COUNT] = {
    [BENCH_MODE_MULTI_2CORE]  = "2 Core MultiThread (priority)",
    [BENCH_MODE_MULTI_2CORE_RR] = "2 Core MultiThread (round-robin)",
    [BENCH_MODE_SINGLE_2CORE] = "2 Core (single-thread per core)",
    [BENCH_MODE_SINGLE_1CORE] = "1 Core (single-thread)",
};

static volatile bool s_running = false;
static bench_state_t s_state = BENCH_STATE_IDLE;
static bench_mode_t s_mode = BENCH_MODE_MULTI_2CORE;
static int64_t s_state_start_us = 0;
static int64_t s_next_kws_us = 0;
static uint32_t s_ovf_start = 0;
static bench_mode_summary_t s_summary[BENCH_MODE_COUNT];
static UBaseType_t s_main_priority = 12;

/* Synthetic KWS workload: ring of the last 1 s of mic audio */
static int16_t *s_ring = NULL;
static int16_t *s_kws_buf = NULL;
static size_t s_ring_pos = 0;
static size_t s_ring_fill = 0;

/* Core 1 single-thread loop (camera + LVGL) */
static TaskHandle_t s_core1_task = NULL;
static volatile bool s_core1_stop = false;
static SemaphoreHandle_t s_core1_stop_sem = NULL;

/*
 * =========================================================
 * Statistics helpers
 * =========================================================
 */

void bench_record(bench_metric_t metric, int64_t us)
{
    if (!s_recording || metric >= BENCH_M_COUNT || us < 0) {
        return;
    }

    uint32_t bucket = (uint32_t)(us / 1000);
    if (bucket > BENCH_HIST_MAX_MS) {
        bucket = BENCH_HIST_MAX_MS;
    }

    bench_stat_t *s = &s_stats[metric];

    taskENTER_CRITICAL(&s_stats_lock);
    if (s->count == 0 || us < s->min_us) {
        s->min_us = us;
    }
    if (us > s->max_us) {
        s->max_us = us;
    }
    s->count++;
    s->sum_us += us;
    s->hist[bucket]++;
    if (s_deadline_us[metric] > 0 && us > s_deadline_us[metric]) {
        s->miss++;
    }
    taskEXIT_CRITICAL(&s_stats_lock);
}

static void stats_reset(void)
{
    taskENTER_CRITICAL(&s_stats_lock);
    memset(s_stats, 0, sizeof(s_stats));
    taskEXIT_CRITICAL(&s_stats_lock);
}

/* Upper edge of the 1 ms bucket that holds the pct-th percentile */
static float stat_percentile_ms(const bench_stat_t *s, uint32_t pct)
{
    if (s->count == 0) {
        return 0.0f;
    }

    uint32_t target = (s->count * pct + 99) / 100;
    uint32_t cum = 0;

    for (int i = 0; i <= BENCH_HIST_MAX_MS; i++) {
        cum += s->hist[i];
        if (cum >= target) {
            int64_t edge_us = (int64_t)(i + 1) * 1000;
            if (i == BENCH_HIST_MAX_MS || edge_us > s->max_us) {
                edge_us = s->max_us;
            }
            return edge_us / 1000.0f;
        }
    }

    return s->max_us / 1000.0f;
}

static bench_stat_summary_t stat_summarize(const bench_stat_t *s)
{
    bench_stat_summary_t out = {
        .count = s->count,
        .miss = s->miss,
    };

    if (s->count > 0) {
        out.avg_ms = (float)s->sum_us / (float)s->count / 1000.0f;
        out.p99_ms = stat_percentile_ms(s, 99);
        out.max_ms = s->max_us / 1000.0f;
        out.total_ms = s->sum_us / 1000.0f;
    }

    return out;
}

/*
 * =========================================================
 * CPU helpers
 * =========================================================
 */

static void cpu_snapshot(bench_cpu_snapshot_t *snap)
{
    configRUN_TIME_COUNTER_TYPE total;
    snap->count = uxTaskGetSystemState(snap->tasks, BENCH_MAX_TASKS, &total);
    snap->time_us = esp_timer_get_time();

    if (snap->count == 0) {
        ESP_LOGW(TAG, "More than %d tasks (%u), raise BENCH_MAX_TASKS",
                 BENCH_MAX_TASKS, (unsigned)uxTaskGetNumberOfTasks());
    }
}

static bool is_app_task(const char *name)
{
    for (size_t i = 0; i < ARRAY_LEN(s_app_tasks); i++) {
        if (strcmp(name, s_app_tasks[i]) == 0) {
            return true;
        }
    }
    return false;
}

static const TaskStatus_t *find_task(const bench_cpu_snapshot_t *snap, TaskHandle_t handle)
{
    for (UBaseType_t i = 0; i < snap->count; i++) {
        if (snap->tasks[i].xHandle == handle) {
            return &snap->tasks[i];
        }
    }
    return NULL;
}

typedef struct {
    const char *name;
    BaseType_t core;
    UBaseType_t prio;
    uint64_t delta_us;
    bool app;
} cpu_row_t;

static void report_cpu(bench_mode_summary_t *sum)
{
    static cpu_row_t rows[BENCH_MAX_TASKS];
    int row_count = 0;

    const double wall_us = (double)(s_snap_end.time_us - s_snap_start.time_us);

    TaskHandle_t idle[portNUM_PROCESSORS];
    uint64_t idle_delta[portNUM_PROCESSORS] = {0};
    for (int c = 0; c < portNUM_PROCESSORS; c++) {
        idle[c] = xTaskGetIdleTaskHandleForCore(c);
    }

    for (UBaseType_t i = 0; i < s_snap_end.count; i++) {
        const TaskStatus_t *end = &s_snap_end.tasks[i];
        const TaskStatus_t *start = find_task(&s_snap_start, end->xHandle);
        uint64_t delta = (uint64_t)end->ulRunTimeCounter -
                         (start ? (uint64_t)start->ulRunTimeCounter : 0);

        bool is_idle = false;
        for (int c = 0; c < portNUM_PROCESSORS; c++) {
            if (end->xHandle == idle[c]) {
                idle_delta[c] = delta;
                is_idle = true;
            }
        }
        if (is_idle) {
            continue;
        }

        rows[row_count++] = (cpu_row_t){
            .name = end->pcTaskName,
            .core = xTaskGetCoreID(end->xHandle),
            .prio = end->uxCurrentPriority,
            .delta_us = delta,
            .app = is_app_task(end->pcTaskName),
        };
    }

    /* Sort by CPU time, highest first */
    for (int i = 1; i < row_count; i++) {
        cpu_row_t key = rows[i];
        int j = i - 1;
        while (j >= 0 && rows[j].delta_us < key.delta_us) {
            rows[j + 1] = rows[j];
            j--;
        }
        rows[j + 1] = key;
    }

    VPRINT("CPU LOAD PER TASK            type core prio  cpu%% (of 1 core)\n");

    double app_pct = 0.0;
    double sys_pct = 0.0;
    double small_pct = 0.0;
    int small_count = 0;

    for (int c = 0; c < portNUM_PROCESSORS; c++) {
        sum->core_app_pct[c] = 0.0f;
    }
    sum->any_app_pct = 0.0f;
    for (size_t t = 0; t < ARRAY_LEN(s_app_tasks); t++) {
        sum->task_pct[t] = -1.0f;
        sum->task_core[t] = -1;
    }

    for (int i = 0; i < row_count; i++) {
        double pct = 100.0 * (double)rows[i].delta_us / wall_us;

        if (rows[i].app) {
            app_pct += pct;
            for (size_t t = 0; t < ARRAY_LEN(s_app_tasks); t++) {
                if (strcmp(rows[i].name, s_app_tasks[t]) == 0) {
                    sum->task_pct[t] = (float)pct;
                    sum->task_core[t] = (rows[i].core == tskNO_AFFINITY) ? -1 : (int)rows[i].core;
                }
            }
            if (rows[i].core >= 0 && rows[i].core < portNUM_PROCESSORS) {
                sum->core_app_pct[rows[i].core] += (float)pct;
            } else {
                sum->any_app_pct += (float)pct;
            }
        } else {
            sys_pct += pct;
        }

        if (!rows[i].app && pct < 0.1) {
            small_pct += pct;
            small_count++;
            continue;
        }

        char core_str[12];
        if (rows[i].core == tskNO_AFFINITY) {
            snprintf(core_str, sizeof(core_str), "any");
        } else {
            snprintf(core_str, sizeof(core_str), "%d", (int)rows[i].core);
        }

        VPRINT("  %-26s %-4s %4s %4u  %7.2f\n",
               rows[i].name,
               rows[i].app ? "APP" : "SYS",
               core_str,
               (unsigned)rows[i].prio,
               pct);
    }

    if (small_count > 0) {
        VPRINT("  %-26s %-4s %4s %4s  %7.2f\n",
               "(other system tasks)", "SYS", "-", "-", small_pct);
    }

    VPRINT("  ----------------------------------------------------------\n");
    VPRINT("  %-41s  %7.2f\n", "ALL APP TASKS", app_pct);
    VPRINT("  %-41s  %7.2f\n", "ALL SYSTEM TASKS", sys_pct);

    double total = 0.0;
    for (int c = 0; c < portNUM_PROCESSORS; c++) {
        double load = 100.0 - 100.0 * (double)idle_delta[c] / wall_us;
        if (load < 0.0) {
            load = 0.0;
        }
        sum->core_load[c] = (float)load;
        total += load;
    }
    total /= portNUM_PROCESSORS;

    sum->total_load = (float)total;
    sum->app_pct = (float)app_pct;
    sum->sys_pct = (float)sys_pct;
}

/*
 * =========================================================
 * Per-core report
 * =========================================================
 */

static void print_metric_row(const bench_stat_summary_t *s, bench_metric_t m)
{
    if (s_deadline_us[m] > 0) {
        VPRINT("    %-22s %7lu %8.2f %8.2f %8.2f %9.1f %5lu\n",
               s_metric_names[m], (unsigned long)s->count,
               s->avg_ms, s->p99_ms, s->max_ms,
               s_deadline_us[m] / 1000.0f, (unsigned long)s->miss);
    } else {
        VPRINT("    %-22s %7lu %8.2f %8.2f %8.2f %9s %5s\n",
               s_metric_names[m], (unsigned long)s->count,
               s->avg_ms, s->p99_ms, s->max_ms, "-", "-");
    }
}

static void report_per_core(bench_mode_summary_t *sum)
{
    for (int m = 0; m < BENCH_M_COUNT; m++) {
        sum->m[m] = stat_summarize(&s_stats[m]);
    }

    sum->cam_fps = sum->m[BENCH_M_CAM_PPA].count / sum->wall_s;
    sum->render_fps = sum->m[BENCH_M_LVGL_RENDER].count / sum->wall_s;

    /* In SINGLE-1CORE the video jobs run on Core 0 too */
    const bool video_on_core0 = (s_mode == BENCH_MODE_SINGLE_1CORE);

    VPRINT("PER-CORE RESPONSE TIME (ms)  count      avg      p99      max  deadline  miss\n");

    VPRINT("  [Core 0 : audio + KWS + alert]  load %.2f %%  (app tasks %.2f %%)\n",
           sum->core_load[0], sum->core_app_pct[0]);
    for (size_t i = 0; i < ARRAY_LEN(s_core0_metrics); i++) {
        print_metric_row(&sum->m[s_core0_metrics[i]], s_core0_metrics[i]);
    }
    VPRINT("    KWS runs: %lu   I2S overflow (audio lost): %lu\n",
           (unsigned long)sum->m[BENCH_M_KWS_INFER].count,
           (unsigned long)sum->i2s_overflow);

    if (video_on_core0) {
        VPRINT("  [Core 0 : camera + LVGL (same core, same loop)]\n");
    } else {
        VPRINT("  [Core 1 : camera + LVGL]  load %.2f %%  (app tasks %.2f %%)\n",
               sum->core_load[1], sum->core_app_pct[1]);
    }
    for (size_t i = 0; i < ARRAY_LEN(s_core1_metrics); i++) {
        print_metric_row(&sum->m[s_core1_metrics[i]], s_core1_metrics[i]);
    }
    VPRINT("    Camera FPS: %.1f   LVGL render/s: %.1f\n", sum->cam_fps, sum->render_fps);

    if (sum->any_app_pct > 0.0f) {
        VPRINT("  [No affinity] app tasks %.2f %%\n", sum->any_app_pct);
    }

    VPRINT("  Core 0 load: %.2f %%   Core 1 load: %.2f %%   Total (2 cores): %.2f %%\n",
           sum->core_load[0], sum->core_load[1], sum->total_load);
}

/*
 * =========================================================
 * Final summary tables
 *
 * Columns: 1 Core | 2 Core | 2 Core MT | 2 Core MT RR
 * =========================================================
 */

static const bench_mode_t s_col_modes[] = {
    BENCH_MODE_SINGLE_1CORE, BENCH_MODE_SINGLE_2CORE,
    BENCH_MODE_MULTI_2CORE, BENCH_MODE_MULTI_2CORE_RR,
};

static const char *const s_col_names[] = {
    "1 Core", "2 Core", "2 Core MT", "2 Core MT RR",
};

/* Steps (jobs) that do the work, with the core they run on in 2-core modes */
typedef struct {
    const char *label;
    bench_metric_t metric;
    int core_2c;
} bench_step_t;

static const bench_step_t s_steps[] = {
    { "Mic VAD + filter (every 32 ms)",   BENCH_M_AUDIO_PROC,  0 },
    { "KWS inference TinyML (1x / s)",    BENCH_M_KWS_INFER,   0 },
    { "JPEG encode alert (every 100 ms)", BENCH_M_ALERT_JPEG,  0 },
    { "PPA scale camera -> LCD size",     BENCH_M_CAM_PPA,     1 },
    { "Hand frame to LVGL (wait lock)",   BENCH_M_CAM_SHOW,    1 },
    { "LVGL render to LCD",               BENCH_M_LVGL_RENDER, 1 },
};

#define SUMMARY_LINE \
    "  -------------------------------------------------------------------------------------------------------------\n"

static int step_core(const bench_step_t *step, bench_mode_t mode)
{
    return (mode == BENCH_MODE_SINGLE_1CORE) ? 0 : step->core_2c;
}

/* % of the measured time spent in this step */
static float step_share_pct(const bench_mode_summary_t *s, bench_metric_t m)
{
    if (s->wall_s <= 0.0f) {
        return 0.0f;
    }
    return s->m[m].total_ms / (s->wall_s * 1000.0f) * 100.0f;
}

static void summary_header(const char *title, const char *unit)
{
    printf("\n  %s\n", title);
    printf("  %-38s", unit);
    for (size_t i = 0; i < ARRAY_LEN(s_col_names); i++) {
        printf(" %17s", s_col_names[i]);
    }
    printf("\n");
    printf(SUMMARY_LINE);
}

static void summary_section(const char *label)
{
    printf("  [%s]\n", label);
}

/* One cell: value, optionally followed by the core it ran on */
static void summary_cell(bool valid, float value, int decimals, int core)
{
    char text[32];

    if (!valid) {
        snprintf(text, sizeof(text), "n/a");
    } else if (core >= 0) {
        snprintf(text, sizeof(text), "%.*f (C%d)", decimals, value, core);
    } else {
        snprintf(text, sizeof(text), "%.*f", decimals, value);
    }

    printf(" %17s", text);
}

typedef enum {
    SUM_STEP_AVG,       /* arg = step index   */
    SUM_STEP_MAX,
    SUM_STEP_SHARE,
    SUM_CORE_SHARE,     /* arg = core         */
    SUM_METRIC_AVG,     /* arg = metric       */
    SUM_METRIC_MAX,
    SUM_CAM_FPS,
    SUM_I2S_OVF,
    SUM_CORE_LOAD,      /* arg = core         */
    SUM_TOTAL_LOAD,
    SUM_TASK_PCT,       /* arg = app task idx */
    SUM_SYS_PCT,
} summary_value_t;

static void summary_row(const char *label, summary_value_t kind, int arg, int decimals)
{
    printf("  %-38s", label);

    for (size_t i = 0; i < ARRAY_LEN(s_col_modes); i++) {
        bench_mode_t mode = s_col_modes[i];
        const bench_mode_summary_t *s = &s_summary[mode];
        bool valid = s->valid;
        float v = 0.0f;
        int core = -1;

        switch (kind) {
        case SUM_STEP_AVG:
            v = s->m[s_steps[arg].metric].avg_ms;
            core = step_core(&s_steps[arg], mode);
            break;
        case SUM_STEP_MAX:
            v = s->m[s_steps[arg].metric].max_ms;
            core = step_core(&s_steps[arg], mode);
            break;
        case SUM_STEP_SHARE:
            v = step_share_pct(s, s_steps[arg].metric);
            core = step_core(&s_steps[arg], mode);
            break;
        case SUM_CORE_SHARE:
            for (size_t k = 0; k < ARRAY_LEN(s_steps); k++) {
                if (step_core(&s_steps[k], mode) == arg) {
                    v += step_share_pct(s, s_steps[k].metric);
                }
            }
            break;
        case SUM_METRIC_AVG:
            v = s->m[arg].avg_ms;
            break;
        case SUM_METRIC_MAX:
            v = s->m[arg].max_ms;
            break;
        case SUM_CAM_FPS:
            v = s->cam_fps;
            break;
        case SUM_I2S_OVF:
            v = (float)s->i2s_overflow;
            break;
        case SUM_CORE_LOAD:
            v = s->core_load[arg];
            break;
        case SUM_TOTAL_LOAD:
            v = s->total_load;
            break;
        case SUM_TASK_PCT:
            v = s->task_pct[arg];
            core = s->task_core[arg];
            if (v < 0.0f) {
                /* Task does not exist in this design */
                printf(" %17s", "-");
                continue;
            }
            break;
        case SUM_SYS_PCT:
            v = s->sys_pct;
            break;
        }

        summary_cell(valid, v, decimals, core);
    }

    printf("\n");
}

static void print_comparison(void)
{
    printf("\n");
    printf("#############################################################################################\n");
    printf("#   BENCHMARK RESULT   (each design measured %d s, same workload)                           #\n",
           BENCH_PHASE_MS / 1000);
    printf("#   1 Core             : one loop does everything on Core 0                                 #\n");
    printf("#   2 Core             : one loop per core (C0 audio+KWS+alert, C1 camera+LCD)              #\n");
    printf("#   2 Core MT          : one task per job, same core split, fixed priority (preemptive)     #\n");
    printf("#                        C0: main 12, alert_cap 5 | C1: camera_video 15, lvgl_bench 4       #\n");
    printf("#   2 Core MT RR       : same tasks and cores, all priority %d (round-robin every tick)      #\n",
           BENCH_RR_PRIORITY);
    printf("#############################################################################################\n");

    summary_header("", "");

    summary_section("1) TIME PER STEP (avg ms)  (C0/C1 = core it ran on)");
    for (size_t k = 0; k < ARRAY_LEN(s_steps); k++) {
        summary_row(s_steps[k].label, SUM_STEP_AVG, (int)k, 2);
    }

    printf(SUMMARY_LINE);
    summary_section("2) TOTAL TIME PER CYCLE (avg ms)");
    summary_row("Audio cycle (must be <= 32 ms)",   SUM_METRIC_AVG, BENCH_M_AUDIO_PERIOD, 2);
    summary_row("Video cycle (1 frame on LCD)",     SUM_METRIC_AVG, BENCH_M_CAM_PERIOD, 2);
    summary_row("Alert cycle (target 100 ms)",      SUM_METRIC_AVG, BENCH_M_ALERT_PERIOD, 2);
    summary_row("Camera FPS",                       SUM_CAM_FPS, 0, 1);
    summary_row("Audio lost (I2S overflow, count)", SUM_I2S_OVF, 0, 0);

    printf(SUMMARY_LINE);
    summary_section("3) CPU LOAD (%)");
    summary_row("Core 0",                           SUM_CORE_LOAD, 0, 1);
    summary_row("Core 1",                           SUM_CORE_LOAD, 1, 1);
    summary_row("Total (2 cores)",                  SUM_TOTAL_LOAD, 0, 1);

    printf(SUMMARY_LINE);
    summary_section("4) CPU PER TASK (% of 1 core, '-' = not used)");
    for (size_t t = 0; t < ARRAY_LEN(s_app_tasks); t++) {
        if (strcmp(s_app_tasks[t], "alert_upl") == 0) {
            continue;  /* upload is disabled during the benchmark */
        }
        summary_row(s_app_tasks[t], SUM_TASK_PCT, (int)t, 1);
    }
    summary_row("system (ISP, timer, WiFi)",        SUM_SYS_PCT, 0, 1);

    printf("#############################################################################################\n\n");
}

/*
 * =========================================================
 * Synthetic KWS workload
 * =========================================================
 */

void bench_feed_audio(const int16_t *samples, size_t count)
{
    if (!s_running || s_ring == NULL || samples == NULL) {
        return;
    }

    for (size_t i = 0; i < count; i++) {
        s_ring[s_ring_pos] = samples[i];
        s_ring_pos = (s_ring_pos + 1) % KWS_AUDIO_TOTAL_SAMPLES;
    }

    s_ring_fill += count;
    if (s_ring_fill > KWS_AUDIO_TOTAL_SAMPLES) {
        s_ring_fill = KWS_AUDIO_TOTAL_SAMPLES;
    }
}

static void run_synthetic_kws(void)
{
    if (s_ring == NULL || s_kws_buf == NULL || s_ring_fill < KWS_AUDIO_TOTAL_SAMPLES) {
        return;
    }

    /* Oldest sample is at s_ring_pos: copy ring into chronological order */
    size_t tail = KWS_AUDIO_TOTAL_SAMPLES - s_ring_pos;
    memcpy(s_kws_buf, &s_ring[s_ring_pos], tail * sizeof(int16_t));
    memcpy(&s_kws_buf[tail], s_ring, s_ring_pos * sizeof(int16_t));

    float confidence = 0.0f;
    bool is_help = false;

    int64_t t0 = esp_timer_get_time();
    kws_engine_process_audio(s_kws_buf, KWS_AUDIO_TOTAL_SAMPLES, &confidence, &is_help);
    bench_record(BENCH_M_KWS_INFER, esp_timer_get_time() - t0);
}

/*
 * =========================================================
 * Core 1 single-thread loop (camera + LVGL)
 * =========================================================
 */

static void core1_loop_task(void *arg)
{
    (void)arg;

    while (!s_core1_stop) {
        int64_t t0 = esp_timer_get_time();

        /* Sleeps up to BENCH_CORE1_FRAME_WAIT_MS waiting for a frame */
        bool got_frame = camera_poll(BENCH_CORE1_FRAME_WAIT_MS);

        display_render_poll();

        /* Camera not available: don't spin and starve IDLE1 */
        if (!got_frame && esp_timer_get_time() - t0 < 1000) {
            vTaskDelay(1);
        }
    }

    s_core1_task = NULL;
    xSemaphoreGive(s_core1_stop_sem);
    vTaskDelete(NULL);
}

static void core1_loop_start(void)
{
    if (s_core1_task != NULL) {
        return;
    }

    if (s_core1_stop_sem == NULL) {
        s_core1_stop_sem = xSemaphoreCreateBinary();
        if (s_core1_stop_sem == NULL) {
            ESP_LOGE(TAG, "No memory for core1_loop semaphore");
            return;
        }
    }

    s_core1_stop = false;
    xSemaphoreTake(s_core1_stop_sem, 0);

    /* Same priority as camera_video */
    if (xTaskCreatePinnedToCore(core1_loop_task, "core1_loop", 8192, NULL, 15,
                                &s_core1_task, 1) != pdPASS) {
        s_core1_task = NULL;
        ESP_LOGE(TAG, "Failed to create core1_loop task");
    }
}

static void core1_loop_stop(void)
{
    if (s_core1_task == NULL) {
        return;
    }

    s_core1_stop = true;
    if (xSemaphoreTake(s_core1_stop_sem, pdMS_TO_TICKS(1000)) != pdTRUE) {
        ESP_LOGE(TAG, "core1_loop did not stop");
        return;
    }
    s_core1_stop = false;
}

/*
 * =========================================================
 * Mode switching
 * =========================================================
 */

static void stop_all_workers(void)
{
    core1_loop_stop();
    camera_task_stop();
    alert_recorder_task_stop();
    display_render_task_stop();
}

static void set_task_priority(const char *name, UBaseType_t priority)
{
    TaskHandle_t handle = xTaskGetHandle(name);
    if (handle != NULL) {
        vTaskPrioritySet(handle, priority);
    }
}

static void apply_mode(bench_mode_t mode)
{
    stop_all_workers();

    /* main (the caller) back to its normal priority */
    vTaskPrioritySet(NULL, s_main_priority);

    switch (mode) {
    case BENCH_MODE_MULTI_2CORE:
        /* Core 0: main + alert_cap, Core 1: camera_video + lvgl_bench */
        display_render_task_start(1);
        camera_task_start(1);
        alert_recorder_task_start(0);
        break;

    case BENCH_MODE_MULTI_2CORE_RR:
        /* Same tasks and cores, but no task can preempt another */
        display_render_task_start(1);
        camera_task_start(1);
        alert_recorder_task_start(0);
        set_task_priority("lvgl_bench", BENCH_RR_PRIORITY);
        set_task_priority("camera_video", BENCH_RR_PRIORITY);
        set_task_priority("alert_cap", BENCH_RR_PRIORITY);
        vTaskPrioritySet(NULL, BENCH_RR_PRIORITY);
        break;

    case BENCH_MODE_SINGLE_2CORE:
        /* Core 0: main loop (audio + KWS + alert), Core 1: core1_loop */
        core1_loop_start();
        break;

    case BENCH_MODE_SINGLE_1CORE:
        /* Nothing to start: bench_step() polls everything from main */
        break;

    default:
        break;
    }
}

static void enter_mode(bench_mode_t mode)
{
    s_mode = mode;

    printf("\n>>> BENCH [%d/%d] %s: warm-up %d ms, measure %d ms ...\n",
           (int)mode + 1, (int)BENCH_MODE_COUNT, s_mode_names[mode],
           BENCH_WARMUP_MS, BENCH_PHASE_MS);

    apply_mode(mode);

    s_state = BENCH_STATE_WARMUP;
    s_state_start_us = esp_timer_get_time();
}

static void begin_measure(void)
{
    stats_reset();
    s_ovf_start = microphone_get_overflow_count();
    cpu_snapshot(&s_snap_start);

    s_state = BENCH_STATE_MEASURE;
    s_state_start_us = s_snap_start.time_us;
    s_recording = true;
}

static void end_measure(void)
{
    s_recording = false;
    cpu_snapshot(&s_snap_end);

    bench_mode_summary_t *sum = &s_summary[s_mode];
    memset(sum, 0, sizeof(*sum));
    sum->valid = true;
    sum->wall_s = (s_snap_end.time_us - s_snap_start.time_us) / 1e6f;
    sum->i2s_overflow = microphone_get_overflow_count() - s_ovf_start;

    /* Detailed per-design report only with BENCH_VERBOSE */
    VPRINT("\n==========================================================================\n");
    VPRINT(" BENCH [%d/%d] %s   (measured %.1f s)\n",
           (int)s_mode + 1, (int)BENCH_MODE_COUNT, s_mode_names[s_mode], sum->wall_s);
    VPRINT("==========================================================================\n");
    report_cpu(sum);
    VPRINT("--------------------------------------------------------------------------\n");
    report_per_core(sum);
    VPRINT("==========================================================================\n");

    printf("    done: %s (%.1f s)\n", s_mode_names[s_mode], sum->wall_s);
}

static void finish(void)
{
    print_comparison();

    /* Back to the normal application layout */
    stop_all_workers();
    vTaskPrioritySet(NULL, s_main_priority);
    camera_task_start(1);
    alert_recorder_task_start(0);
    display_render_release();

    esp_log_level_set("KWS_ENGINE", ESP_LOG_INFO);

    heap_caps_free(s_ring);
    heap_caps_free(s_kws_buf);
    s_ring = NULL;
    s_kws_buf = NULL;

    s_state = BENCH_STATE_IDLE;
    s_running = false;

    ESP_LOGI(TAG, "Benchmark finished, normal application resumed");
}

/*
 * =========================================================
 * Public API
 * =========================================================
 */

bool bench_is_running(void)
{
    return s_running;
}

void bench_start(void)
{
    if (s_running) {
        return;
    }

    s_ring = heap_caps_malloc(KWS_AUDIO_TOTAL_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    s_kws_buf = heap_caps_malloc(KWS_AUDIO_TOTAL_SAMPLES * sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (s_ring == NULL || s_kws_buf == NULL) {
        ESP_LOGW(TAG, "No memory for KWS ring buffer, KWS load disabled");
        heap_caps_free(s_ring);
        heap_caps_free(s_kws_buf);
        s_ring = NULL;
        s_kws_buf = NULL;
    }
    s_ring_pos = 0;
    s_ring_fill = 0;

    memset(s_summary, 0, sizeof(s_summary));
    s_main_priority = uxTaskPriorityGet(NULL);

    /* One KWS result log per second would flood the benchmark output */
    esp_log_level_set("KWS_ENGINE", ESP_LOG_WARN);

    printf("\n##########################################################################\n");
    printf("#   BENCHMARK START : %d designs                                          #\n",
           (int)BENCH_MODE_COUNT);
    printf("#   1) MULTI-THREAD 2 cores (priority)                                   #\n");
    printf("#   2) MULTI-THREAD 2 cores (round-robin, all priority %d)                #\n",
           BENCH_RR_PRIORITY);
    printf("#   3) SINGLE-THREAD 2 cores  4) SINGLE-THREAD 1 core                    #\n");
    printf("#   Core 0: mic+VAD every 32 ms, KWS every %d ms, alert JPEG 100 ms    #\n",
           BENCH_KWS_PERIOD_MS);
    printf("#   Core 1: camera -> PPA -> LVGL.  Alerts/upload are disabled.          #\n");
    printf("##########################################################################\n");

    /* Replace the BSP LVGL task with our own render loop (placeable per mode) */
    display_render_takeover();

    s_running = true;
    s_next_kws_us = esp_timer_get_time() + BENCH_KWS_PERIOD_MS * 1000LL;
    enter_mode((bench_mode_t)0);
}

void bench_step(void)
{
    if (!s_running) {
        return;
    }

    switch (s_mode) {
    case BENCH_MODE_SINGLE_1CORE:
        /* One loop does all the work itself */
        camera_poll(0);
        display_render_poll();
        alert_recorder_poll();
        break;

    case BENCH_MODE_SINGLE_2CORE:
        /* Core 0 loop: alert capture (camera + LVGL are on core1_loop) */
        alert_recorder_poll();
        break;

    default:
        break;
    }

    int64_t now = esp_timer_get_time();

    if (now >= s_next_kws_us) {
        s_next_kws_us = now + BENCH_KWS_PERIOD_MS * 1000LL;
        run_synthetic_kws();
        now = esp_timer_get_time();
    }

    switch (s_state) {
    case BENCH_STATE_WARMUP:
        if (now - s_state_start_us >= BENCH_WARMUP_MS * 1000LL) {
            begin_measure();
        }
        break;

    case BENCH_STATE_MEASURE:
        if (now - s_state_start_us >= BENCH_PHASE_MS * 1000LL) {
            end_measure();

            if (s_mode + 1 < BENCH_MODE_COUNT) {
                enter_mode((bench_mode_t)(s_mode + 1));
            } else {
                finish();
            }
        }
        break;

    default:
        break;
    }
}
