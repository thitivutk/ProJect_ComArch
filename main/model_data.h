#ifndef MODEL_DATA_H
#define MODEL_DATA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// 2-Class TinyML Model for ESP32-P4 (Porjai + Edge-TTS)
// Classes: [0: "ช่วยด้วย (HELP)", 1: "OTHER"]
#define KWS_MODEL_NUM_CLASSES 2
#define KWS_LABEL_CHUEY_DUAY  0
#define KWS_LABEL_OTHER       1

extern const char* KWS_LABELS[KWS_MODEL_NUM_CLASSES];
extern const uint8_t g_kws_model_data[];
extern const unsigned int g_kws_model_data_len;

#ifdef __cplusplus
}
#endif

#endif // MODEL_DATA_H
