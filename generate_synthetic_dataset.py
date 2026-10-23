"""
Synthetic Thai Voice Dataset Generator using Microsoft Edge-TTS
Balances the dataset to exactly 1,000 : 1,000 (2,000 files total):
  - Class 0 (ช่วยด้วย + ตัวแปรเสริม): 251 User + 749 Synthetic (10 Personas) = 1,000 files
  - Class 1 (คำอื่นๆ 160+ วลี): 221 User + 749 Synthetic (10 Personas) + 30 Noise = 1,000 files
"""

import asyncio
import os
import sys
import uuid
import glob
import wave
import miniaudio
import numpy as np
import edge_tts

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# 10 Distinct Speaker Personas
PERSONAS = [
    {"id": "p01_girl_child",   "voice": "th-TH-PremwadeeNeural", "base_pitch": 45,  "base_rate": 15, "desc": "เด็กหญิง (Child Girl)"},
    {"id": "p02_girl_teen",    "voice": "th-TH-PremwadeeNeural", "base_pitch": 25,  "base_rate": 10, "desc": "วัยรุ่นหญิง (Teen Girl)"},
    {"id": "p03_female_young", "voice": "th-TH-PremwadeeNeural", "base_pitch": 5,   "base_rate": 0,  "desc": "หญิงสาว (Young Adult Female)"},
    {"id": "p04_female_soft",  "voice": "th-TH-PremwadeeNeural", "base_pitch": -15, "base_rate": -5, "desc": "หญิงเสียงนุ่ม (Soft Adult Female)"},
    {"id": "p05_female_elder", "voice": "th-TH-PremwadeeNeural", "base_pitch": -30, "base_rate": -15, "desc": "หญิงสูงวัย (Elderly Female)"},

    {"id": "p06_boy_child",    "voice": "th-TH-NiwatNeural",     "base_pitch": 40,  "base_rate": 15, "desc": "เด็กชาย (Child Boy)"},
    {"id": "p07_boy_teen",     "voice": "th-TH-NiwatNeural",     "base_pitch": 20,  "base_rate": 12, "desc": "วัยรุ่นชาย (Teen Boy)"},
    {"id": "p08_male_young",   "voice": "th-TH-NiwatNeural",     "base_pitch": 0,   "base_rate": 0,  "desc": "ชายหนุ่ม (Young Adult Male)"},
    {"id": "p09_male_deep",    "voice": "th-TH-NiwatNeural",     "base_pitch": -25, "base_rate": -10, "desc": "ชายเสียงทุ้ม (Deep Voice Male)"},
    {"id": "p10_male_elder",   "voice": "th-TH-NiwatNeural",     "base_pitch": -38, "base_rate": -18, "desc": "ชายสูงวัย (Elderly Male)"},
]

# Varied HELP keywords (all contain "ช่วยด้วย" with natural extensions)
HELP_VARIANTS = [
    "ช่วยด้วย",
    "ช่วยด้วยครับ",
    "ช่วยด้วยค่ะ",
    "ช่วยด้วย ช่วยด้วย",
    "ช่วยด้วยเร็ว",
    "ช่วยด้วยนะ",
    "ช่วยด้วยนะคะ",
    "ช่วยด้วยนะครับ",
    "ช่วยด้วยจ้า",
    "ช่วยด้วยที",
    "ช่วยด้วยเถอะ",
    "โอ๊ยช่วยด้วย",
    "ช่วยด้วยใครก็ได้",
    "ช่วยด้วย ช่วยด้วย เร็ว",
    "เร็วช่วยด้วย",
    "ช่วยด้วยหนูด้วย",
    "ช่วยด้วยครับผม",
    "ช่วยด้วยด่วน",
    "ช่วยด้วยตรงนี้",
    "ช่วยด้วยช่วยที"
]

# 160+ Daily Life Words & Common Phrases
DAILY_LIFE_WORDS = [
    # ทักทาย / สนทนา
    "สวัสดี", "สวัสดีครับ", "สวัสดีค่ะ", "ขอบคุณ", "ขอบคุณครับ", "ขอบคุณค่ะ", "ขอโทษ", "ขอโทษครับ", "ขอโทษค่ะ",
    "บาย", "บ๊ายบาย", "เป็นไงบ้าง", "สบายดีไหม", "ยินดีด้วย", "ฮัลโหล", "ครับผม", "ค่ะคุณ", "ใช่", "ไม่ใช่",
    "ไม่เป็นไร", "โอเค", "แน่นอน", "ได้ยินไหม", "ว่าไง", "เอ่อ", "อืม", "จริงเหรอ", "เข้าใจแล้ว", "ไม่เข้าใจ",
    "ฝันดีนะ", "ไปก่อนนะ", "เจอกันพรุ่งนี้", "โชคดีนะ",

    # อาหาร / เครื่องดื่ม / ซื้อของ
    "กินข้าว", "หิวน้ำ", "กินน้ำ", "กับข้าว", "อาหาร", "กินอะไรดี", "กาแฟ", "ชานม", "อิ่มแล้ว", "ต้มยำ",
    "ข้าวผัด", "ส้มตำ", "ก๋วยเตี๋ยว", "ช้อนส้อม", "น้ำแข็ง", "ขนมปัง", "ผลไม้", "กินยา", "ซื้อขนม", "น้ำเปล่า",
    "ไปกินข้าวกัน", "สั่งอาหาร", "คิดเงินด้วย", "เช็คบิล", "ร้านอาหาร", "อร่อยมาก", "ไม่อร่อย", "เผ็ดจัง",

    # กิจวัตรประจำวัน / ในบ้าน
    "ตื่นนอน", "ไปนอน", "อาบน้ำ", "แปรงฟัน", "แต่งตัว", "เปิดไฟ", "ปิดไฟ", "เปิดแอร์", "ปิดแอร์",
    "เปิดพัดลม", "ปิดพัดลม", "เปิดประตู", "ปิดประตู", "เปิดหน้าต่าง", "ปิดหน้าต่าง", "ดูทีวี", "ซักผ้า",
    "ล้างจาน", "ทำความสะอาด", "นอนหลับ", "ฝันดี", "กุญแจ", "รีโมท", "โทรศัพท์", "ที่ชาร์จ", "ห้องน้ำ",
    "ห้องนอน", "ห้องครัว", "ดูหนัง", "ฟังเพลง", "ทำงานบ้าน", "ให้อาหารหมา", "ให้อาหารแมว", "ทิ้งขยะ",

    # การเดินทาง / การทำงาน / ข้างนอก
    "ไปทำงาน", "ไปเรียน", "กลับบ้าน", "รถติด", "ฝนตก", "แดดร้อน", "ขึ้นรถ", "ลงรถ", "ขับรถ", "ถึงแล้ว",
    "รอแป๊บ", "เดินเล่น", "ตลาด", "ซื้อของ", "เซเว่น", "ข้ามถนน", "ปั๊มน้ำมัน", "ทางด่วน", "ติดไฟแดง",
    "จอดรถ", "รถเมล์", "มอเตอร์ไซค์", "แท็กซี่", "สถานีรถไฟ", "กำลังไป", "ถึงไหนแล้ว", "หลงทาง",

    # อารมณ์ / ความรู้สึก / สุขภาพ
    "ทำอะไรอยู่", "อยู่ไหน", "กี่โมงแล้ว", "เท่าไหร่", "ร้อนจัง", "หนาวจัง", "เหนื่อยจัง", "ง่วงนอน",
    "ปวดหัว", "ดีจัง", "ชอบมาก", "ไม่ชอบ", "คิดถึง", "รอก่อน", "เร็วๆ หน่อย", "เดี๋ยวนะ", "อะไรนะ",
    "สนุกดี", "ตกใจหมด", "ปวดท้อง", "เป็นไข้", "เจ็บคอ", "เวียนหัว", "เมื่อยตัว", "หายไวๆ นะ",

    # คำลวงที่ใช้คำว่า "ช่วย" แต่ไม่ใช่ "ช่วยด้วย" (Hard Negatives)
    "ช่วยหน่อย", "ช่วยที", "ช่วยดู", "ช่วยคิด", "ช่วยหยิบ", "ช่วยเปิด", "ช่วยปิด", "ช่วยถือหน่อย",
    "ช่วยสอนหน่อย", "ช่วยบอกทาง", "ช่วยงาน", "ช่วยฟังที",

    # คำลวงที่ใช้คำว่า "ด้วย" แต่ไม่ใช่ "ช่วยด้วย" (Hard Negatives)
    "ไปด้วย", "เอาด้วย", "ไปด้วยกัน", "ขอไปด้วย", "คนสวยด้วย", "รวยด้วย", "ไปด้วยสิ", "ยินดีด้วยนะ",
    "เห็นด้วย", "ทำด้วยกัน",

    # คำลวงเสียงสระ อวย/อวด/อวน คล้ายเสียงคำว่า ช่วยด้วย (Hard Negatives)
    "กล้วย", "สวย", "รวย", "ถ้วย", "มวย", "บ๊วย", "พวย", "ดวง", "หน่วย", "ห้วย", "ขวด", "นวม", "หมวด", "รวดเร็ว"
]

TEMP_DIR = os.path.join("dataset", "temp_tts")
os.makedirs(TEMP_DIR, exist_ok=True)
SEM = asyncio.Semaphore(12)

async def generate_clip(text, voice, pitch_str, rate_str, out_wav_path, txt_path=None):
    if os.path.exists(out_wav_path):
        return True
    async with SEM:
        temp_mp3 = os.path.join(TEMP_DIR, f"temp_{uuid.uuid4().hex}.mp3")
        try:
            comm = edge_tts.Communicate(text, voice, pitch=pitch_str, rate=rate_str)
            await comm.save(temp_mp3)
            
            decoded = miniaudio.decode_file(temp_mp3, nchannels=1, sample_rate=16000)
            samples = np.frombuffer(decoded.samples, dtype=np.int16)
            
            with wave.open(out_wav_path, 'wb') as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(16000)
                w.writeframes(samples.tobytes())
            
            if txt_path:
                with open(txt_path, 'w', encoding='utf-8') as tf:
                    tf.write(text)
            return True
        except Exception as e:
            try:
                comm = edge_tts.Communicate(text, voice, pitch="+0Hz", rate=rate_str)
                await comm.save(temp_mp3)
                decoded = miniaudio.decode_file(temp_mp3, nchannels=1, sample_rate=16000)
                samples = np.frombuffer(decoded.samples, dtype=np.int16)
                with wave.open(out_wav_path, 'wb') as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(16000)
                    w.writeframes(samples.tobytes())
                if txt_path:
                    with open(txt_path, 'w', encoding='utf-8') as tf:
                        tf.write(text)
                return True
            except Exception as e2:
                print(f"  Error generating '{text}' ({voice}): {e2}", flush=True)
                return False
        finally:
            if os.path.exists(temp_mp3):
                try: os.remove(temp_mp3)
                except: pass

async def main():
    print("=" * 65, flush=True)
    print("2,000 Balanced Dataset Generator (1,000 Help vs 1,000 Not-Help)", flush=True)
    print("=" * 65, flush=True)
    
    help_dir = os.path.join("dataset", "1_chuey_duay")
    other_dir = os.path.join("dataset", "2_other_words")
    noise_dir = os.path.join("dataset", "3_unrecognized")
    os.makedirs(help_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    os.makedirs(noise_dir, exist_ok=True)

    # Count real user files
    user_help_count = len([f for f in glob.glob(os.path.join(help_dir, "*.wav")) if not os.path.basename(f).startswith("syn_")])
    user_other_count = len([f for f in glob.glob(os.path.join(other_dir, "*.wav")) if not os.path.basename(f).startswith("syn_")])
    noise_count = len(glob.glob(os.path.join(noise_dir, "*.wav")))

    target_help_total = 1000
    target_class1_total = 1000
    
    needed_syn_help = target_help_total - user_help_count
    needed_syn_other = target_class1_total - noise_count - user_other_count

    print(f"Current Human/Noise Audio Files:")
    print(f"  - User Help Files:  {user_help_count}")
    print(f"  - User Other Files: {user_other_count}")
    print(f"  - Noise Files:      {noise_count}")
    print(f"\nTarget Synthetic Files to Generate for 2,000 Total Balance:")
    print(f"  - Synthetic Help:  {needed_syn_help} files (Total Help -> {target_help_total})")
    print(f"  - Synthetic Other: {needed_syn_other} files (Total Not-Help -> {target_class1_total})")

    # Clean old synthetic files
    print("\nCleaning up old synthetic files...", flush=True)
    for f in glob.glob(os.path.join(help_dir, "syn_*.wav")):
        try: os.remove(f)
        except: pass
    for f in glob.glob(os.path.join(other_dir, "syn_*.wav")):
        try: os.remove(f)
        except: pass
    for f in glob.glob(os.path.join(other_dir, "syn_*.txt")):
        try: os.remove(f)
        except: pass

    # 1. Build tasks for HELP (Diverse "ช่วยด้วย" variants)
    pitch_jitters = [-12, -6, 0, +6, +12]
    rate_jitters = [-10, -5, 0, +5, +10]
    jitter_pairs = [(p, r) for p in pitch_jitters for r in rate_jitters]

    help_tasks = []
    clip_idx = 0
    while len(help_tasks) < needed_syn_help:
        text = HELP_VARIANTS[clip_idx % len(HELP_VARIANTS)]
        p = PERSONAS[(clip_idx // len(HELP_VARIANTS) + clip_idx) % len(PERSONAS)]
        pj, rj = jitter_pairs[clip_idx % len(jitter_pairs)]
        
        cur_pitch = p["base_pitch"] + pj
        cur_rate = p["base_rate"] + rj
        pitch_str = f"{cur_pitch:+d}Hz"
        rate_str = f"{cur_rate:+d}%"
        
        out_path = os.path.join(help_dir, f"syn_{p['id']}_{clip_idx:04d}.wav")
        help_tasks.append(generate_clip(text, p["voice"], pitch_str, rate_str, out_path))
        clip_idx += 1

    print(f"\n[1/2] Generating {len(help_tasks)} diverse 'ช่วยด้วย' clips across 10 personas...", flush=True)
    res_help = await asyncio.gather(*help_tasks)
    succ_help = sum(1 for r in res_help if r)
    print(f">> Finished 'ช่วยด้วย': {succ_help}/{len(help_tasks)} files generated!", flush=True)

    # 2. Build tasks for OTHER (160+ Daily Life Words & Hard Negatives)
    other_tasks = []
    other_idx = 0
    while len(other_tasks) < needed_syn_other:
        word = DAILY_LIFE_WORDS[other_idx % len(DAILY_LIFE_WORDS)]
        p = PERSONAS[(other_idx // len(DAILY_LIFE_WORDS) + other_idx) % len(PERSONAS)]
        
        jit = (other_idx % 5) - 2
        cur_pitch = p["base_pitch"] + jit * 4
        cur_rate = p["base_rate"] + jit * 3
        pitch_str = f"{cur_pitch:+d}Hz"
        rate_str = f"{cur_rate:+d}%"
        
        out_path = os.path.join(other_dir, f"syn_other_{p['id']}_{other_idx:04d}.wav")
        txt_path = os.path.join(other_dir, f"syn_other_{p['id']}_{other_idx:04d}.txt")
        other_tasks.append(generate_clip(word, p["voice"], pitch_str, rate_str, out_path, txt_path))
        other_idx += 1

    print(f"\n[2/2] Generating {len(other_tasks)} diverse 'คำอื่นๆ' clips across 10 personas...", flush=True)
    res_other = await asyncio.gather(*other_tasks)
    succ_other = sum(1 for r in res_other if r)
    print(f">> Finished 'คำอื่นๆ': {succ_other}/{len(other_tasks)} files generated!", flush=True)

    # Cleanup temp
    if os.path.exists(TEMP_DIR):
        try: os.rmdir(TEMP_DIR)
        except: pass

    # Final tally
    final_help = len(glob.glob(os.path.join(help_dir, "*.wav")))
    final_other = len(glob.glob(os.path.join(other_dir, "*.wav")))
    final_noise = len(glob.glob(os.path.join(noise_dir, "*.wav")))
    final_class1 = final_other + final_noise

    print("\n" + "=" * 65, flush=True)
    print("FINAL 2,000 BALANCED DATASET SUMMARY:", flush=True)
    print(f"  Class 0 [ช่วยด้วย]      : {final_help} files", flush=True)
    print(f"  Class 1 [ไม่ใช่ช่วยด้วย]: {final_class1} files ({final_other} words + {final_noise} noise)", flush=True)
    print(f"  Total Audio Files       : {final_help + final_class1} files", flush=True)
    print("=" * 65, flush=True)

if __name__ == "__main__":
    asyncio.run(main())
