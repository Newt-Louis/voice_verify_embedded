import os
from onnxruntime.quantization import quantize_dynamic, QuantType

# LUÔN LUÔN BẮT ĐẦU TỪ MỐC 0 (Source of Truth)
FP32_MODEL_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
INT8_MODEL_PATH = "quantization/exports/ecapa/v2_int8_fast/ecapa_int8_dynamic.onnx"

def convert_fp32_to_int8_dynamic():
    print(f"[*] Đang chuẩn bị ép cân xuống INT8 Dynamic từ bản gốc FP32...")
    if not os.path.exists(FP32_MODEL_PATH):
        print(f"[!] LỖI: Không tìm thấy Source at {FP32_MODEL_PATH}")
        return

    os.makedirs(os.path.dirname(INT8_MODEL_PATH), exist_ok=True)

    print("[*] Đang chạy onnxruntime Dynamic Quantization...")
    # Dynamic Quantization: Chỉ nén trọng số (Weights), kích hoạt (Activations) xử lý linh hoạt
    quantize_dynamic(
        model_input=FP32_MODEL_PATH,
        model_output=INT8_MODEL_PATH,
        weight_type=QuantType.QUInt8,
    )

    size_mb = os.path.getsize(INT8_MODEL_PATH) / (1024 * 1024)
    print(f"✅ HOÀN TẤT! Đã lưu mô hình INT8 Dynamic tại:\n    {INT8_MODEL_PATH} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    convert_fp32_to_int8_dynamic()
