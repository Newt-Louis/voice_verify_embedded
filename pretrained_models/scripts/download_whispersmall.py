import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def download_and_save_whisper():
    # Thư mục đích
    MODEL_NAME = "openai/whisper-small"
    SAVE_DIR = "pretrained_models/whisper-small"

    # Tạo thư mục nếu chưa có
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f" BẮT ĐẦU TẢI MÔ HÌNH: {MODEL_NAME}")
    print(" Cảnh báo: File gốc nặng khoảng ~1GB, có thể mất vài phút...")

    try:
        print("\n[1/2] Đang tải Processor (Feature Extractor & Tokenizer)...")
        processor = WhisperProcessor.from_pretrained(MODEL_NAME)
        processor.save_pretrained(SAVE_DIR)

        print("\n[2/2] Đang tải mô hình cốt lõi (Weights)...")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        model.save_pretrained(SAVE_DIR)
        print("✅ Đã lưu Mô hình thành công!")

    except Exception as e:
        print(f"\n❌ Lỗi trong quá trình tải: {e}")


if __name__ == "__main__":
    download_and_save_whisper()