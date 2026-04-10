import torch
import onnxruntime as ort
import numpy as np
import os
import time
import psutil
import subprocess
import json
import torchaudio

# ==========================================
# 0. PATCH LỖI THƯ VIỆN & CẤU HÌNH
# ==========================================
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

import huggingface_hub
_orig_download = huggingface_hub.hf_hub_download
def _patched_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    for dep in ['local_dir_use_symlinks', 'force_filename']:
        kwargs.pop(dep, None)
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else "")
    if "custom.py" in filename:
        dummy_path = os.path.abspath("dummy_custom.py")
        if not os.path.exists(dummy_path):
            with open(dummy_path, "w") as f:
                f.write("# File giả mạo để lừa SpeechBrain bỏ qua lỗi 404\n")
        return dummy_path
    return _orig_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_download

from speechbrain.inference.speaker import EncoderClassifier

# Đường dẫn
VOICE_DIR = "my_test_voice/pharse_2"
STRANGER_FILES = ["eng_1.m4a", "eng_2.m4a", "vie_1.m4a", "vie_2.m4a"]
OWNER_FILES = [
    "myvoice_Recording_testpass_1.m4a",
    "myvoice_Recording_testpass_2.m4a",
    "myvoice_Recording_testpass_3.m4a"
]
ALL_FILES = STRANGER_FILES + OWNER_FILES

ONNX_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
REPORT_PATH = "benchmarks/quantization_report.json"

def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)

def load_audio(path):
    try:
        command = [
            '/usr/bin/ffmpeg', '-i', path, '-ac', '1', '-ar', '16000',
            '-f', 'f32le', '-hide_banner', '-loglevel', 'error', 'pipe:1'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        return torch.tensor(np.frombuffer(out, dtype=np.float32)).unsqueeze(0)
    except:
        return None

def run_test():
    print("\n" + "="*60)
    print(" [PHASE 0] TEST ĐỐI SOÁT: PYTORCH VS ONNX FP32")
    print("="*60)
    
    # 1. Đo Baseline RAM
    base_ram = get_ram()
    
    # 2. Load PyTorch Model
    print("[*] Đang nạp mô hình PyTorch (SpeechBrain)...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="pretrained_models/ecapa"
    )
    
    # 3. Load ONNX Model
    print("[*] Đang nạp mô hình ONNX...")
    onnx_start_ram = get_ram()
    session = ort.InferenceSession(ONNX_PATH, providers=['CPUExecutionProvider'])
    onnx_static_ram = get_ram() - onnx_start_ram
    
    # 4. Thực hiện Inference và đo lường
    latencies = []
    cosine_diffs = []
    
    peak_ram = get_ram()
    
    print(f"\n{'File':<35} | {'PyTorch':<10} | {'ONNX':<10} | {'Diff'}")
    print("-" * 75)
    
    for f_name in ALL_FILES:
        f_path = os.path.join(VOICE_DIR, f_name)
        sig = load_audio(f_path)
        if sig is None: continue
        
        # A. Chạy PyTorch
        with torch.no_grad():
            feats = classifier.mods.compute_features(sig)
            feats = classifier.mods.mean_var_norm(feats, torch.ones(1))
            
            t0 = time.perf_counter()
            pt_emb = classifier.mods.embedding_model(feats).squeeze().numpy()
            pt_time = (time.perf_counter() - t0) * 1000
            
        # B. Chạy ONNX
        t1 = time.perf_counter()
        onnx_outputs = session.run(None, {'input': feats.numpy()})
        onnx_emb = onnx_outputs[0].flatten()
        onnx_time = (time.perf_counter() - t1) * 1000
        
        # C. So sánh (Cosine Similarity)
        norm_a = np.linalg.norm(pt_emb)
        norm_b = np.linalg.norm(onnx_emb)
        cos_sim = np.dot(pt_emb, onnx_emb) / (norm_a * norm_b)
        accuracy_gap = float(1.0 - cos_sim) # Ép kiểu sang float Python
        
        latencies.append(float(onnx_time))
        cosine_diffs.append(accuracy_gap)
        
        print(f"{f_name:<35} | {pt_time:6.2f}ms | {onnx_time:6.2f}ms | {accuracy_gap:.8f}")
        
        current_ram = get_ram()
        if current_ram > peak_ram: peak_ram = current_ram

    # 5. Tổng hợp dữ liệu
    final_data = {
        "version": "v0_float32",
        "storage_mb": float(os.path.getsize(ONNX_PATH) / (1024 * 1024)),
        "static_ram_mb": float(onnx_static_ram),
        "peak_ram_mb": float(peak_ram - base_ram),
        "latency_ms": float(np.mean(latencies)) if latencies else 0.0,
        "accuracy_drop": float(np.mean(cosine_diffs)) if cosine_diffs else 0.0
    }
    
    print("\n" + "="*60)
    print(f" KẾT QUẢ BASELINE (MỐC 0)")
    print("="*60)
    print(f"- Dung lượng file: {final_data['storage_mb']:.2f} MB")
    print(f"- RAM Tĩnh (ONNX): {final_data['static_ram_mb']:.2f} MB")
    print(f"- RAM Đỉnh (Peak):  {final_data['peak_ram_mb']:.2f} MB")
    print(f"- Độ trễ TB (ms):  {final_data['latency_ms']:.2f} ms")
    print(f"- Sai lệch Cosine: {final_data['accuracy_drop']:.12f}")
    
    # Lưu vào JSON
    history = {}
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, 'r') as f: history = json.load(f)
        except: pass
    
    history["v0_float32"] = final_data
    with open(REPORT_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n[*] Đã lưu báo cáo vào: {REPORT_PATH}")

if __name__ == "__main__":
    run_test()
