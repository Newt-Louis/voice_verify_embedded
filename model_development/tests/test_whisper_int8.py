import os
import time
import psutil
import subprocess
import numpy as np
import gc
from transformers import WhisperProcessor, GenerationConfig
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN MỚI NHẤT
# ==========================================
MODEL_DIR = "quantization/exports/whisper/v2_int8_dynamic"
TEST_AUDIO = "my_test_voice/pharse_2/myvoice_Recording_testpass_3.m4a"


def get_ram():
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def load_audio_for_whisper(path):
    try:
        command = [
            '/usr/bin/ffmpeg', '-i', path, '-ac', '1', '-ar', '16000',
            '-f', 'f32le', '-hide_banner', '-loglevel', 'error', 'pipe:1'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        return np.frombuffer(out, dtype=np.float32)
    except Exception as e:
        print(f"[!] Lỗi đọc file {path}: {e}")
        return None


def run_benchmark():
    print("\n" + "=" * 70)
    print(" [MỐC 2] TEST & BENCHMARK WHISPER SMALL (ONNX INT8 DYNAMIC)")
    print("=" * 70)

    print(f"[*] Đang nạp file audio: {os.path.basename(TEST_AUDIO)}")
    audio_data = load_audio_for_whisper(TEST_AUDIO)
    if audio_data is None: return

    gc.collect()
    time.sleep(1)
    base_ram = get_ram()

    print("[*] Đang nạp Processor và Model INT8...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # Optimum sẽ tự động đọc file ort_config.json và biết nạp các file *_quantized.onnx
    model = ORTModelForSpeechSeq2Seq.from_pretrained(MODEL_DIR,
                                                     encoder_file_name="encoder_model_quantized.onnx",
                                                     decoder_file_name="decoder_model_quantized.onnx",
                                                     decoder_with_past_file_name="decoder_with_past_model_quantized.onnx",
                                                     use_merged = True
                                                     )
    model.generation_config = GenerationConfig.from_pretrained("openai/whisper-small")
    static_ram = get_ram() - base_ram
    print(f"    -> Đã nạp xong! (RAM Tĩnh chiếm: {static_ram:.2f} MB)")

    # Lấy attention_mask để vá lỗi "Ảo giác khoảng lặng"
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
    input_features = inputs.input_features
    attention_mask = inputs.attention_mask

    print("\n[*] Đang chạy suy luận (Inference)...")
    peak_ram = get_ram()
    t0 = time.perf_counter()

    # THỦ THUẬT: Mồi từ vựng (Prompt Engineering cho Audio)
    prompt_text = "Hệ thống siêu quản gia Celia. Các lệnh cơ bản: bật, tắt, xác thực, mở cửa."
    prompt_ids = processor.get_prompt_ids(prompt_text, return_tensors="pt")

    generated_ids = model.generate(
        input_features,
        attention_mask=attention_mask,  # Chặn tiếng ồn khoảng lặng
        prompt_ids=prompt_ids,  # Bơm mồi từ vựng
        language="vi",  # Ép tiếng Việt
        task="transcribe",  # Chép chính tả
        use_cache=True  # Bật KV-Cache
    )

    current_ram = get_ram()
    if current_ram > peak_ram:
        peak_ram = current_ram

    latency = (time.perf_counter() - t0) * 1000
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    total_size_mb = sum(os.path.getsize(os.path.join(MODEL_DIR, f)) for f in os.listdir(MODEL_DIR) if
                        os.path.isfile(os.path.join(MODEL_DIR, f))) / (1024 * 1024)

    print("\n" + "=" * 60)
    print(f" KẾT QUẢ MỐC 2 (WHISPER INT8 DYNAMIC)")
    print("=" * 60)
    print(f"- Text Dịch Ra:  \"{transcription.strip()}\"")
    print(f"- Dung lượng:    {total_size_mb:.2f} MB (Chứa file dự phòng)")
    print(f"- RAM Tĩnh:      {static_ram:.2f} MB")
    print(f"- RAM Đỉnh:      {peak_ram - base_ram:.2f} MB")
    print(f"- Độ trễ TB:     {latency:.2f} ms")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()