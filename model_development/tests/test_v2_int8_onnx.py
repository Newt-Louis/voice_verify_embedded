import torch, gc, os, time, psutil, subprocess, json, torchaudio
import onnxruntime as ort
import numpy as np

# ==========================================
# CẤU HÌNH TEST MỐC 2 (STATIC VS DYNAMIC)
# ==========================================
VOICE_DIR = "my_test_voice/pharse_2"
ALL_FILES = [
    "eng_1.m4a", "eng_2.m4a", "vie_1.m4a", "vie_2.m4a",
    "myvoice_Recording_testpass_1.m4a",
    "myvoice_Recording_testpass_2.m4a",
    "myvoice_Recording_testpass_3.m4a"
]

FP32_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
INT8_STATIC_PATH = "quantization/exports/ecapa/v2_int8_calib/ecapa_int8_static.onnx"
INT8_DYNAMIC_PATH = "quantization/exports/ecapa/v2_int8_fast/ecapa_int8_dynamic.onnx"
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
    fbank = torchaudio.compliance.kaldi.fbank(
        sig, num_mel_bins=80, sample_frequency=16000, frame_length=25, frame_shift=10
    )
    mean = torch.mean(fbank, dim=0, keepdim=True)
    std = torch.std(fbank, dim=0, keepdim=True)
    fbank_norm = (fbank - mean) / (std + 1e-5)
    return fbank_norm.unsqueeze(0).numpy().astype(np.float32)


def evaluate_model(model_path, version_name, fp32_baseline_embs, test_data):
    """Hàm đo lường độc lập cho từng model để RAM không bị cộng dồn"""
    # Ép dọn dẹp bộ nhớ trước khi nạp model mới
    gc.collect()
    time.sleep(0.5)
    base_ram = get_ram()

    # Đo Static RAM (Sau khi khởi tạo)
    sess = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
    static_ram_mb = get_ram() - base_ram

    latencies = []
    cosine_diffs = []
    peak_ram = get_ram()

    # Bắt đầu Inference Loop
    for item in test_data:
        feats = item['feats']

        t0 = time.perf_counter()
        emb = sess.run(None, {'input': feats})[0].flatten()
        latencies.append((time.perf_counter() - t0) * 1000)

        # Cập nhật Peak RAM liên tục
        current_ram = get_ram()
        if current_ram > peak_ram:
            peak_ram = current_ram

        # Tính Accuracy Drop so với FP32
        baseline_emb = fp32_baseline_embs[item['name']]
        cos_sim = np.dot(baseline_emb, emb) / (np.linalg.norm(baseline_emb) * np.linalg.norm(emb))
        cosine_diffs.append(1.0 - cos_sim)

    # Hủy Session để trả lại RAM cho hệ thống
    del sess
    gc.collect()

    return {
        "version": version_name,
        "storage_mb": float(os.path.getsize(model_path) / (1024 * 1024)),
        "static_ram_mb": float(static_ram_mb),
        "peak_ram_mb": float(peak_ram - base_ram),
        "latency_ms": float(np.mean(latencies)),
        "accuracy_drop": float(np.mean(cosine_diffs))
    }


def run_test():
    print("\n" + "=" * 70)
    print(" [MỐC 2] KIỂM TRA TOÀN DIỆN: ĐỘ TRỄ, RAM & ĐỘ CHÍNH XÁC (INT8)")
    print("=" * 70)

    # 1. Nạp sẵn dữ liệu âm thanh (tránh đo nhầm thời gian/RAM đọc ổ đĩa)
    print("[*] Đang nạp và trích xuất Fbank dữ liệu test...")
    test_data = []
    for f_name in ALL_FILES:
        f_path = os.path.join(VOICE_DIR, f_name)
        sig = load_audio(f_path)
        if sig is not None:
            test_data.append({"name": f_name, "feats": extract_feats(sig)})

    # 2. Chạy FP32 để lấy Ground Truth (Làm mốc so sánh Accuracy)
    print("[*] Đang chạy bản FP32 Gốc để lấy Baseline...")
    gc.collect()
    sess_fp32 = ort.InferenceSession(FP32_PATH, providers=['CPUExecutionProvider'])
    fp32_embs = {}
    for item in test_data:
        fp32_embs[item['name']] = sess_fp32.run(None, {'input': item['feats']})[0].flatten()
    del sess_fp32
    gc.collect()  # Trả RAM về nguyên trạng

    # 3. Đo lường bản Dynamic
    print("\n[*] Đang đo lường bản DYNAMIC INT8...")
    dynamic_report = evaluate_model(INT8_DYNAMIC_PATH, "v2_int8_dynamic", fp32_embs, test_data)

    # 4. Đo lường bản Static
    print("[*] Đang đo lường bản STATIC INT8...")
    static_report = evaluate_model(INT8_STATIC_PATH, "v2_int8_static", fp32_embs, test_data)

    # ==============================
    # IN KẾT QUẢ THEO FORMAT YÊU CẦU
    # ==============================
    for data, title in [(dynamic_report, "DYNAMIC (INT8)"), (static_report, "STATIC (INT8)")]:
        print("\n" + "=" * 60)
        print(f" KẾT QUẢ MỐC 2 - {title}")
        print("=" * 60)
        print(f"- Dung lượng:    {data['storage_mb']:.2f} MB")
        print(f"- RAM Tĩnh:      {data['static_ram_mb']:.2f} MB")
        print(f"- RAM Đỉnh:      {data['peak_ram_mb']:.2f} MB")
        print(f"- Độ trễ TB:     {data['latency_ms']:.2f} ms")
        print(f"- Accuracy Drop: {data['accuracy_drop']:.12f}")

    # ==============================
    # LƯU LỊCH SỬ JSON
    # ==============================
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    history = {}
    if os.path.exists(REPORT_PATH):
        try:
            with open(REPORT_PATH, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            pass  # Bỏ qua nếu file hỏng

    history[dynamic_report["version"]] = dynamic_report
    history[static_report["version"]] = static_report

    with open(REPORT_PATH, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"\n✅ Đã cập nhật kết quả vào: {REPORT_PATH}")


if __name__ == "__main__":
    run_test()
