import os,time,psutil,subprocess,gc,numpy as np

# 1. Import các "Vũ khí" của Hugging Face
from transformers import WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================
# Trỏ vào đúng thư mục chứa các file ONNX vừa xuất
MODEL_DIR = "quantization/exports/whisper/v0_float32"
# Chọn 1 file test giọng chủ nhân
TEST_AUDIO = "my_test_voice/pharse_2/myvoice_Recording_testpass_3.m4a"


def get_ram():
    """Đo RAM hiện tại của tiến trình (MB)"""
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_audio_for_whisper(path):
    """
    Đọc audio bằng FFmpeg và xuất ra numpy array 1D.
    Whisper Processor cực kỳ thích định dạng 1D Numpy chuẩn 16kHz này.
    """
    try:
        command = [
            '/usr/bin/ffmpeg', '-i', path, '-ac', '1', '-ar', '16000',
            '-f', 'f32le', '-hide_banner', '-loglevel', 'error', 'pipe:1'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        # Lưu ý: Whisper cần mảng 1D, nên không dùng .unsqueeze(0) như ECAPA
        return np.frombuffer(out, dtype=np.float32)
    except Exception as e:
        print(f"[!] Lỗi đọc file {path}: {e}")
        return None


def run_benchmark():
    print("\n" + "=" * 70)
    print(" [MỐC 0] TEST & BENCHMARK WHISPER SMALL (ONNX FP32)")
    print("=" * 70)

    # 1. Nạp âm thanh trước để tránh tính thời gian I/O ổ đĩa
    print(f"[*] Đang nạp file audio: {os.path.basename(TEST_AUDIO)}")
    audio_data = load_audio_for_whisper(TEST_AUDIO)
    if audio_data is None:
        return

    # 2. Ép dọn RAM hệ thống trước khi bắt đầu đo
    gc.collect()
    time.sleep(1)
    base_ram = get_ram()

    # 3. Nạp Model & Processor
    print("[*] Đang nạp Whisper Processor và Model ONNX FP32 vào RAM...")

    # Processor làm nhiệm vụ: Âm thanh -> Fbank -> Model -> Số ID -> Text
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Optimum tự động tìm và nạp 2 file encoder/decoder trong thư mục
    # Thêm use_cache=False vì chúng ta chưa export bản decoder_with_past
    model = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_DIR)

    static_ram = get_ram() - base_ram
    print(f"    -> Đã nạp xong! (RAM Tĩnh chiếm: {static_ram:.2f} MB)")

    # 4. Tiền xử lý âm thanh (Đưa âm thanh qua bộ lọc Fbank của Whisper)
    # return_tensors="pt" báo cho nó trả về định dạng PyTorch tensor giả lập
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt",return_attention_mask=True)
    input_features = inputs.input_features
    attention_mask = inputs.attention_mask

    print("\n[*] Đang chạy suy luận (Inference/Generation)...")
    peak_ram = get_ram()
    t0 = time.perf_counter()

    # Vòng lặp Autoregressive chạy ngầm trong hàm .generate()
    # Nó tự gọi Encoder 1 lần, và gọi Decoder nhiều lần
    generated_ids = model.generate(
        input_features,
        attention_mask=attention_mask,
        language="vi",
        task="transcribe",
        use_cache=True)

    # Cập nhật Peak RAM ngay sau khi chạy xong hàm sinh văn bản
    current_ram = get_ram()
    if current_ram > peak_ram:
        peak_ram = current_ram

    latency = (time.perf_counter() - t0) * 1000

    # 5. Giải mã ID thành Text Tiếng Việt
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Tính tổng dung lượng thư mục Model
    total_size_mb = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR) if
                        os.path.isfile(os.path.join(MODEL_DIR, f))) / (1024 * 1024)

    # ==============================
    # IN KẾT QUẢ THEO FORMAT
    # ==============================
    print("\n" + "=" * 60)
    print(f" KẾT QUẢ MỐC 0 PROPER (WHISPER FLOAT32)")
    print("=" * 60)
    print(f"- Text Dịch Ra:  \"{transcription.strip()}\"")
    print(f"- Dung lượng:    {total_size_mb:.2f} MB")
    print(f"- RAM Tĩnh:      {static_ram:.2f} MB")
    print(f"- RAM Đỉnh:      {peak_ram - base_ram:.2f} MB")
    print(f"- Độ trễ TB:     {latency:.2f} ms")
    print(f"- Accuracy Drop: 0.000000000000 (Ground Truth)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()