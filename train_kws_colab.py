# =============================================================================
# Google Colab Training Script for ESP32-P4 TinyML KWS ("ช่วยด้วย" vs "คำอื่นๆ")
# =============================================================================

# 1. แตกไฟล์ ZIP
!unzip -q -o dataset_ready.zip

import os, glob, wave
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from google.colab import files

print("TensorFlow Version:", tf.__version__)

# 2. ค้นหาไฟล์เสียงทั้ง 2 คลาส (Recursive ค้นหาเจอแน่นอนไม่ว่าจะแตกไว้โฟลเดอร์ไหน)
help_files = sorted(glob.glob('**/1_chuey_duay/*.wav', recursive=True))
other_files = sorted(glob.glob('**/2_other_words/*.wav', recursive=True) + glob.glob('**/3_unrecognized/*.wav', recursive=True))

print(f"\n📊 ตรวจพบข้อมูลในชุด Dataset:")
print(f"   - Class 0 [ช่วยด้วย]     : {len(help_files)} ไฟล์")
print(f"   - Class 1 [ไม่ใช่ช่วยด้วย]: {len(other_files)} ไฟล์")
print(f"   - รวมทั้งหมด              : {len(help_files) + len(other_files)} ไฟล์")

if len(help_files) == 0 or len(other_files) == 0:
    raise RuntimeError("❌ ไม่พบไฟล์เสียง กรุณาตรวจสอบว่าอัปโหลด 'dataset_ready.zip' เรียบร้อยแล้ว")

# 3. สกัด Mel-Spectrogram ขนาด 49x40 (สูตรคณิตศาสตร์ตรงกับ ESP32-P4 DSP 100%)
SAMPLE_RATE = 16000
AUDIO_DURATION = 1.0
N_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION) # 16,000 samples
N_FFT, HOP_LENGTH, N_MELS, N_FRAMES = 512, 320, 40, 49
FMIN, FMAX = 20.0, 8000.0

def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

mel_min, mel_max = hz_to_mel(FMIN), hz_to_mel(FMAX)
mel_step = (mel_max - mel_min) / (N_MELS + 1)
bin_points = np.array([min(N_FFT//2, int(round(N_FFT * mel_to_hz(mel_min + i * mel_step) / SAMPLE_RATE))) for i in range(N_MELS + 2)])
hann_window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / (N_FFT - 1)))

# โหลดตัวอย่างเสียง Ambient Noise จากไมค์จริงสำหรับผสมเสียงสังเคราะห์ให้เนื้อเสียงสมจริง
noise_files = sorted(glob.glob('**/3_unrecognized/*.wav', recursive=True))
noise_samples_list = []
for nf in noise_files:
    try:
        with wave.open(nf, 'rb') as w:
            s = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
            if len(s) >= N_SAMPLES:
                noise_samples_list.append(s[:N_SAMPLES])
    except Exception:
        pass

def extract_mel(file_path):
    with wave.open(file_path, 'rb') as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    samples = np.pad(samples, (0, max(0, N_SAMPLES - len(samples))))[:N_SAMPLES]
    
    # ถ้าเป็นเสียงสังเคราะห์ (syn_*) ให้เติมเสียง Ambient Noise ไมโครโฟนเบาๆ (10-15%) เพื่อเชื่อมช่องว่างความถี่ไมค์จริง
    if 'syn_' in os.path.basename(file_path):
        if noise_samples_list:
            bg_noise = noise_samples_list[np.random.randint(0, len(noise_samples_list))]
            samples = samples * 0.88 + bg_noise * 0.12
        else:
            samples = samples + np.random.normal(0, 0.005, size=N_SAMPLES)
            
    spec = np.zeros((N_FRAMES, N_MELS), dtype=np.float32)
    for f in range(N_FRAMES):
        frame = np.pad(samples[f*HOP_LENGTH : f*HOP_LENGTH + N_FFT], (0, max(0, N_FFT - len(samples[f*HOP_LENGTH : f*HOP_LENGTH + N_FFT]))))
        power = np.abs(np.fft.rfft(frame * hann_window, n=N_FFT)) ** 2
        for m in range(N_MELS):
            sb, cb, eb = bin_points[m], bin_points[m+1], bin_points[m+2]
            w1 = (np.arange(sb, cb) - sb) / max(1, cb - sb)
            w2 = (eb - np.arange(cb, min(eb, N_FFT//2 + 1))) / max(1, eb - cb)
            energy = np.sum(power[sb:cb] * w1) + np.sum(power[cb:min(eb, N_FFT//2 + 1)] * w2)
            spec[f, m] = np.log(energy + 1e-6)
    return np.expand_dims(spec, axis=-1)

print("\n⏳ กำลังสกัด Mel-Spectrogram ทั้งหมด 2,000 ไฟล์ (ประมาณ 30-45 วินาที)...")
X = [extract_mel(f) for f in help_files] + [extract_mel(f) for f in other_files]
y = [0] * len(help_files) + [1] * len(other_files)
X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

# สุ่ม Shuffle และแบ่งข้อมูล 80% Train, 20% Validation
idx = np.arange(len(X))
np.random.seed(42)
np.random.shuffle(idx)
X, y = X[idx], y[idx]

val_count = int(0.2 * len(X))
X_train, y_train = X[val_count:], y[val_count:]
X_val, y_val = X[:val_count], y[:val_count]

print(f"✅ ข้อมูลพร้อมเทรน: Train = {len(X_train)} ชุด | Validation = {len(X_val)} ชุด")

# 4. โมเดล TinyML CNN (64-dim Embedding Layer + 2-Class Dense)
model = models.Sequential([
    layers.Input(shape=(49, 40, 1), name="input_spectrogram"),
    layers.Conv2D(16, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.SeparableConv2D(32, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.SeparableConv2D(64, (3, 3), padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.GlobalAveragePooling2D(name="global_average_pooling2d"), # 64-dim Vector สำหรับ On-Device Learning
    layers.Dropout(0.25),
    layers.Dense(2, activation='softmax', name="dense_output")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# บันทึกโมเดลรอบที่แม่นยำที่สุดไว้เสมอ (เทรนยาวๆ ไปจนครบ Epochs ไม่มีการตัดจบกลางคัน)
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    'best_kws_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)
lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,
    min_lr=1e-5,
    verbose=1
)

print("\n🚀 เริ่มต้นการเทรนโมเดลยาวๆ (80 Epochs เต็ม ไม่มีหยุดกลางคัน)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=80,
    batch_size=32,
    callbacks=[checkpoint_cb, lr_cb],
    verbose=1
)

# โหลดค่าน้ำหนักรอบที่ทำ Accuracy ได้สูงสุดมาใช้ทำ INT8
if os.path.exists('best_kws_model.keras'):
    print("\n🏆 โหลดโมเดลรอบที่แม่นยำที่สุด (Best Accuracy) มาแปลงเป็น INT8...")
    model = tf.keras.models.load_model('best_kws_model.keras')

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n🎯 ผลการประเมินโมเดลรอบที่ดีที่สุด: Validation Accuracy = {val_acc * 100:.2f}%")

# 5. Full INT8 Quantization (สำหรับ TFLM บน ESP32-P4)
print("\n⚙️ กำลังทำการแปลงโมเดลเป็น Full INT8 Quantized...")

def rep_data():
    for i in range(min(150, len(X_train))):
        yield [X_train[i:i+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = rep_data
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

# สร้างไฟล์ model_data.c รูปแบบ C Array พร้อมใช้กับ ESP-IDF
c_code = [
    '#include "model_data.h"\n',
    'const char* KWS_LABELS[KWS_MODEL_NUM_CLASSES] = {"ช่วยด้วย (HELP)", "OTHER"};\n',
    'const uint8_t g_kws_model_data[] __attribute__((aligned(4))) = {'
]
hex_b = [f"0x{b:02x}" for b in tflite_model]
for i in range(0, len(hex_b), 12):
    c_code.append("    " + ", ".join(hex_b[i:i+12]) + ("," if i+12 < len(hex_b) else ""))
c_code.append('};\n')
c_code.append(f'const unsigned int g_kws_model_data_len = {len(tflite_model)};\n')

with open('model_data.c', 'w', encoding='utf-8') as f:
    f.write('\n'.join(c_code))

print(f"\n🎉 สำเร็จเรียบร้อย! ขนาดโมเดล: {len(tflite_model)/1024:.2f} KB")
print("⬇️ กำลังดาวน์โหลดไฟล์ 'model_data.c' ลงเครื่องคอมพิวเตอร์ของคุณอัตโนมัติ...")
files.download('model_data.c')
