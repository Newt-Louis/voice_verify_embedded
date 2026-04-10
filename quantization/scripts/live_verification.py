import torch, psutil
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
VOICE_DIR = "my_test_voice/pharse_2"
ENROLLMENT_FILES = [
    "my_voice_eng_1.m4a",
    "my_voice_eng_2.m4a",
    "my_voice_vie_1.m4a",
    "my_voice_vie_2.m4a"
]
# Danh sách các file cần kiểm tra (Verification)
TEST_FILES = [
    "eng_1.m4a",
    "eng_2.m4a",
    "vie_1.m4a",
    "vie_2.m4a",
    "myvoice_Recording_testpass_1.m4a",
    "myvoice_Recording_testpass_2.m4a",
    "myvoice_Recording_testpass_3.m4a",
]

# Ngưỡng chấp nhận (Threshold) - Thường ECAPA nằm khoảng 0.25 - 0.35
THRESHOLD = 0.4

cos_sim = CosineSimilarity(dim=-1)

def get_memory_usage():
    """Trả về lượng RAM (tính bằng MB) mà tiến trình Python này đang sử dụng."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def load_audio(path):
    """
    Đọc file âm thanh (hỗ trợ .m4a qua librosa)
    Chuyển về chuẩn 16000Hz, Mono để AI xử lý chính xác.
    """
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            signal, _ = librosa.load(path, sr=16000)
        return torch.tensor(signal).unsqueeze(0)
    except Exception as e:
        print(f"[ERROR] Không thể đọc file {path}: {e}")
        return None

def run_live_test(model_name="ECAPA-TDNN PyTorch (Original)"):
    print(f"\n" + "="*60)
    print(f" BẮT ĐẦU TEST XÁC THỰC: {model_name}")
    print("="*60)
    ram_baseline = get_memory_usage()
    print(f"[*] RAM Cơ bản (Chưa nạp gì): {ram_baseline:.2f} MB")
    # 1. Tải mô hình gốc (Bản chưa nén)
    # Ghi chú: Sau này khi có bản ONNX, ta sẽ thêm logic load ONNX Runtime tại đây.
    print("[1/3] Đang nạp mô hình AI...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa",
        run_opts={"device": "cpu"}
    )
    ram_after_load = get_memory_usage()
    print(f"[*] RAM sau khi nạp mô hình: {ram_after_load:.2f} MB")
    print(f"    -> MÔ HÌNH NGỐN (Tĩnh): {ram_after_load - ram_baseline:.2f} MB")
    # 2. Tạo "Vân tay giọng nói" gốc
    print(f"[2/3] Đang trích xuất vân tay từ file gốc: {len(ENROLLMENT_FILES)}")
    enrollment_embeddings = []
    peak_ram = ram_after_load
    for f_name in ENROLLMENT_FILES:
        f_path = os.path.join(VOICE_DIR, f_name)
        sig = load_audio(f_path)
        if sig is not None:
            with torch.no_grad():
                emb = model.encode_batch(sig)
                enrollment_embeddings.append(emb)
            current_ram = get_memory_usage()
            if current_ram > peak_ram:
                peak_ram = current_ram
    if not enrollment_embeddings:
        print("❌ Lỗi: Không đọc được file mẫu nào. Dừng hệ thống.")
        return
    # sig_init = load_audio(INIT_VOICE)
    # if sig_init is None: return
    #
    # with torch.no_grad():
    #     emb_init = model.encode_batch(sig_init)
    master_signature = torch.mean(torch.stack(enrollment_embeddings), dim=0)

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
            current_ram = get_memory_usage()
            if current_ram > peak_ram:
                peak_ram = current_ram
            score = cos_sim(master_signature, emb_test).item()
            
            status = "CHẤP NHẬN ✅" if score >= THRESHOLD else "TỪ CHỐI ❌"
            print(f"{f_name:<30} | {score:<15.4f} | {status}")

    print("-" * 65)
    print(f"Ngưỡng bảo mật hiện tại: {THRESHOLD}")
    print("\n=== TỔNG KẾT TÀI NGUYÊN (RAM) ===")
    print(f"- Lượng RAM cao nhất hệ thống đã dùng (Peak RAM): {peak_ram:.2f} MB")
    print(f"- Lượng RAM THỰC TẾ phục vụ AI (Peak - Baseline): {peak_ram - ram_baseline:.2f} MB")
if __name__ == "__main__":
    run_live_test()
