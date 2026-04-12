import numpy as np
import subprocess
import os
from ecapa_engine import ECAPAEngine
from ecapa_machine import ECAPAMachine

# Cấu hình
TEST_FILE = "my_test_voice/pharse_2/myvoice_Recording_testpass_3.m4a"
MASTER_FILE = "my_test_voice/pharse_2/my_voice_eng_1.m4a"

def load_audio_ffmpeg(path):
    """Đọc audio bằng ffmpeg chuẩn xác"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file: {path}")
    command = [
        '/usr/bin/ffmpeg', '-i', path, '-ac', '1', '-ar', '16000',
        '-f', 'f32le', '-hide_banner', '-loglevel', 'error', 'pipe:1'
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate()
    return np.frombuffer(out, dtype=np.float32)

def run_simulation():
    print("\n" + "="*60)
    print(" [MÔ PHỎNG] XÁC THỰC LUỒNG ÂM THANH (STREAMING)")
    print("="*60)

    temp_engine = ECAPAEngine()
    master_audio = load_audio_ffmpeg(MASTER_FILE)
    master_sig = temp_engine.get_embedding(master_audio[:48000])

    authenticator = ECAPAMachine(master_signature=master_sig,threshold=0.4)

    print(f"[*] Đang mô phỏng truyền âm thanh từ file: {os.path.basename(TEST_FILE)}")
    test_audio = load_audio_ffmpeg(TEST_FILE)
    
    # Chia nhỏ âm thanh thành các đoạn 0.5 giây (8000 samples)
    chunk_size = 8000 
    total_samples = len(test_audio)
    
    print(f"{'Thời điểm':<15} | {'Trạng thái':<20} | {'Điểm số (Cosine)':<12} | {'Độ trễ'}")
    print("-" * 70)

    for i in range(0, total_samples, chunk_size):
        current_chunk = test_audio[i : i + chunk_size]
        elapsed_sec = (i + len(current_chunk)) / 16000 # Giây hiện tại
        
        # Đẩy đoạn âm thanh vào bộ xác thực
        results = authenticator.feed_audio(current_chunk)
        
        # Nếu có embedding mới (khi cửa sổ trượt vừa khớp)
        if results:
            for res in results:
                score = res['score']
                latency = res['latency_ms']
                if res['authenticated']:
                    status = "✅ MỞ CỬA (PASS)"
                else:
                    status = "❌ CHẶN LẠI (FAIL)"
                print(f"{elapsed_sec:5.1f} giây     | {status:<20} | {score:.4f}      | {latency:.1f} ms")
        else:
            # Chỉ in trạng thái tích lũy nếu chưa đạt 3 giây đầu
            if elapsed_sec < 3.0:
                print(f"{elapsed_sec:5.1f} giây     | ⏳ Đang tích lũy...  | ---")

    print("\n" + "="*60)
    print(" KẾT THÚC MÔ PHỎNG")
    print("="*60)

if __name__ == "__main__":
    run_simulation()
