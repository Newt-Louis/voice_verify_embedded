import onnx
from onnxconverter_common import float16
import os

# Đường dẫn từ Mốc 0
FP32_MODEL_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
# Đường dẫn xuất ra Mốc 1 chuẩn
FP16_MODEL_PATH = "quantization/exports/ecapa/v1_fp16/ecapa_fp16_proper.onnx"

def convert_fp32_to_fp16_onnx():
    print(f"[*] Đang nạp mô hình ONNX FP32 gốc từ:\n    {FP32_MODEL_PATH}")
    if not os.path.exists(FP32_MODEL_PATH):
        print("[!] Không tìm thấy file FP32. Hãy đảm bảo bạn đã xuất Mốc 0 thành công.")
        return

    # 1. Đọc file ONNX 32-bit vào RAM
    model_fp32 = onnx.load(FP32_MODEL_PATH)

    print("[*] Đang nhờ Chuyên gia ép cân (onnxconverter) xử lý đồ thị...")
    # 2. Ép xuống 16-bit. 
    print("[*] Đang lọc các node ASP nhạy cảm để đưa vào danh sách chặn (Block list)...")
    # Lấy danh sách tất cả các node có chứa từ 'asp'
    asp_nodes = [n.name for n in model_fp32.graph.node if 'asp' in n.name.lower()]
    print(f"[*] Đã tìm thấy {len(asp_nodes)} node ASP. Sẽ giữ chúng ở định dạng Float32.")

    # 2. Ép xuống 16-bit. 
    model_fp16 = float16.convert_float_to_float16(
        model_fp32, 
        keep_io_types=True,
        node_block_list=asp_nodes
    )

    # 3. Lưu lại
    os.makedirs(os.path.dirname(FP16_MODEL_PATH), exist_ok=True)
    onnx.save(model_fp16, FP16_MODEL_PATH)
    
    size_mb = os.path.getsize(FP16_MODEL_PATH) / (1024 * 1024)
    print(f"✅ HOÀN TẤT! Đã lưu mô hình FP16 tại:\n    {FP16_MODEL_PATH} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    convert_fp32_to_fp16_onnx()
