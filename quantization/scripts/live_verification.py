import torch
import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
import huggingface_hub
import urllib.error
_orig_download = huggingface_hub.hf_hub_download
def _patched_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else "")
    if "custom.py" in filename:
        # Nếu nó đòi custom.py, tạo ngay 1 file rỗng và trả về đường dẫn
        dummy_path = os.path.abspath("dummy_custom.py")
        if not os.path.exists(dummy_path):
            with open(dummy_path, "w") as f:
                f.write("# File giả mạo để lừa SpeechBrain bỏ qua lỗi 404\n")
        return dummy_path
        # Các file khác (weights, config) thì vẫn tải bình thường
    return _orig_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_download

import numpy as np
import librosa
import os
from speechbrain.inference.speaker import EncoderClassifier
from torch.nn import CosineSimilarity

# ==========================================
# CẤU HÌNH THỬ NGHIỆM THỰC TẾ
# ==========================================
VOICE_DIR = "my_test_voice"
# File giọng "chủ nhân" dùng để đăng ký ban đầu (Enrollment)
INIT_VOICE = os.path.join(VOICE_DIR, "testing_init_my_voice.m4a")

# Danh sách các file cần kiểm tra (Verification)
TEST_FILES = [
    "hey_celia.m4a",             # Giọng thật của chủ
    "command_open.m4a",          # Giọng thật của chủ
    "testing_another_person.m4a"  # Giọng người khác
]

# Ngưỡng chấp nhận (Threshold) - Thường ECAPA nằm khoảng 0.25 - 0.35
THRESHOLD = 0.3

cos_sim = CosineSimilarity(dim=-1)

def load_audio(path):
    """
    Đọc file âm thanh (hỗ trợ .m4a qua librosa)
    Chuyển về chuẩn 16000Hz, Mono để AI xử lý chính xác.
    """
    try:
        # librosa tự động xử lý m4a nếu hệ thống có ffmpeg/av
        signal, _ = librosa.load(path, sr=16000)
        return torch.tensor(signal).unsqueeze(0)
    except Exception as e:
        print(f"[ERROR] Không thể đọc file {path}: {e}")
        return None

def run_live_test(model_name="ECAPA-TDNN PyTorch (Original)"):
    print(f"\n" + "="*60)
    print(f" BẮT ĐẦU TEST XÁC THỰC: {model_name}")
    print("="*60)

    # 1. Tải mô hình gốc (Bản chưa nén)
    # Ghi chú: Sau này khi có bản ONNX, ta sẽ thêm logic load ONNX Runtime tại đây.
    print("[1/3] Đang nạp mô hình AI...")
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", 
                                           savedir="pretrained_models/ecapa")

    # 2. Tạo "Vân tay giọng nói" gốc
    print(f"[2/3] Đang trích xuất vân tay từ file gốc: {os.path.basename(INIT_VOICE)}")
    sig_init = load_audio(INIT_VOICE)
    if sig_init is None: return
    
    with torch.no_grad():
        emb_init = model.encode_batch(sig_init)

    # 3. Chạy vòng lặp kiểm tra
    print("[3/3] Đang tiến hành so sánh đối soát...")
    print("-" * 65)
    print(f"{'Tên File':<30} | {'Độ tương đồng':<15} | {'Kết quả'}")
    print("-" * 65)

    for f_name in TEST_FILES:
        f_path = os.path.join(VOICE_DIR, f_name)
        sig_test = load_audio(f_path)
        
        if sig_test is not None:
            with torch.no_grad():
                emb_test = model.encode_batch(sig_test)
            
            # Tính độ tương đồng Cosine (từ -1 đến 1)
            score = cos_sim(emb_init, emb_test).item()
            
            status = "CHẤP NHẬN ✅" if score >= THRESHOLD else "TỪ CHỐI ❌"
            print(f"{f_name:<30} | {score:<15.4f} | {status}")

    print("-" * 65)
    print(f"Ngưỡng bảo mật hiện tại: {THRESHOLD}")

if __name__ == "__main__":
    # Hiện tại chạy test với bản PyTorch gốc
    # Sau này sẽ bổ sung các bản nén v1, v2, v3 để so sánh ngay tại đây
    run_live_test()
