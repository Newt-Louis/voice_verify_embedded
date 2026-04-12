import torch, os, time, psutil, subprocess, json, torchaudio
import onnxruntime as ort
import numpy as np

# ==========================================
# CẤU HÌNH
# ==========================================
VOICE_DIR = "my_test_voice/pharse_2"
STRANGER_FILES = ["eng_1.m4a", "eng_2.m4a", "vie_1.m4a", "vie_2.m4a"]
OWNER_FILES = [
    "myvoice_Recording_testpass_1.m4a",
    "myvoice_Recording_testpass_2.m4a",
    "myvoice_Recording_testpass_3.m4a"
]
ALL_FILES = STRANGER_FILES + OWNER_FILES

FP32_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
FP16_PATH = "quantization/exports/ecapa/v1_fp32/ecapa_fp16.onnx"
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

def extract_feats(sig):
    # Sử dụng torchaudio để trích xuất fbank (Giống ecapa_engine.py)
    fbank = torchaudio.compliance.kaldi.fbank(
        sig, num_mel_bins=80, sample_frequency=16000, frame_length=25, frame_shift=10
    )
    # Norm
    mean = torch.mean(fbank, dim=0, keepdim=True)
    std = torch.std(fbank, dim=0, keepdim=True)
    fbank_norm = (fbank - mean) / (std + 1e-5)
    return fbank_norm.unsqueeze(0).numpy().astype(np.float32)

def run_test():
    print("\n" + "="*60)
    print(" [PHASE 1] TEST ĐỐI SOÁT: ONNX FP32 VS ONNX FP16")
    print("="*60)
    
    base_ram = get_ram()
    
    # 1. Load FP32 Model (Lấy làm chuẩn đối soát)
    print("[*] Loading FP32 Session...")
    sess_fp32 = ort.InferenceSession(FP32_PATH, providers=['CPUExecutionProvider'])
    
    # 2. Load FP16 Model
    print("[*] Loading FP16 Session...")
    start_ram = get_ram()
    sess_fp16 = ort.InferenceSession(FP16_PATH, providers=['CPUExecutionProvider'])
    static_ram_fp16 = get_ram() - start_ram
    
    latencies = []
    cosine_diffs = []
    peak_ram = get_ram()
    
    print(f"\n{'File':<35} | {'FP32 Time':<10} | {'FP16 Time':<10} | {'Cos Sim'}")
    print("-" * 80)
    
    for f_name in ALL_FILES:
        f_path = os.path.join(VOICE_DIR, f_name)
        sig = load_audio(f_path)
        if sig is None: continue
        
        feats = extract_feats(sig)
        
        # A. Chạy FP32
        t0 = time.perf_counter()
        emb_fp32 = sess_fp32.run(None, {'input': feats})[0].flatten()
        fp32_time = (time.perf_counter() - t0) * 1000
        
        # B. Chạy FP16
        t1 = time.perf_counter()
        emb_fp16 = sess_fp16.run(None, {'input': feats})[0].flatten()
        fp16_time = (time.perf_counter() - t1) * 1000
        
        # C. So sánh Cosine Similarity
        cos_sim = np.dot(emb_fp32, emb_fp16) / (np.linalg.norm(emb_fp32) * np.linalg.norm(emb_fp16))
        accuracy_drop = 1.0 - cos_sim
        
        latencies.append(fp16_time)
        cosine_diffs.append(accuracy_drop)
        
        print(f"{f_name:<35} | {fp32_time:8.2f}ms | {fp16_time:8.2f}ms | {cos_sim:.8f}")
        
        if get_ram() > peak_ram: peak_ram = get_ram()

    # Tổng hợp
    final_data = {
        "version": "v1_fp16",
        "storage_mb": float(os.path.getsize(FP16_PATH) / (1024 * 1024)),
        "static_ram_mb": float(static_ram_fp16),
        "peak_ram_mb": float(peak_ram - base_ram),
        "latency_ms": float(np.mean(latencies)),
        "accuracy_drop": float(np.mean(cosine_diffs))
    }
    
    print("\n" + "="*60)
    print(f" KẾT QUẢ MỐC 1 (FLOAT16)")
    print("="*60)
    print(f"- Dung lượng:   {final_data['storage_mb']:.2f} MB (Giảm ~50%)")
    print(f"- RAM Tĩnh:     {final_data['static_ram_mb']:.2f} MB")
    print(f"- RAM Đỉnh:     {final_data['peak_ram_mb']:.2f} MB")
    print(f"- Độ trễ TB:    {final_data['latency_ms']:.2f} ms")
    print(f"- Accuracy Drop: {final_data['accuracy_drop']:.12f}")
    
    # Lưu vào JSON
    history = {}
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, 'r') as f: history = json.load(f)
    
    history["v1_fp16"] = final_data
    with open(REPORT_PATH, 'w') as f:
        json.dump(history, f, indent=4)

if __name__ == "__main__":
    run_test()
