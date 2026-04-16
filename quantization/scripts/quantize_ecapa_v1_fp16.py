import onnx
from onnxconverter_common import float16
import os

INPUT_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
OUTPUT_PATH = "quantization/exports/ecapa/v1_fp32/ecapa_fp16.onnx"

def quantize_fp16_selective():
    print(f"[*] Loading FP32 model: {INPUT_PATH}")
    model = onnx.load(INPUT_PATH)
    
    # Tìm tất cả các node liên quan đến ASP để đưa vào danh sách đen
    # Các node này thường nhạy cảm với kiểu dữ liệu và dễ gây lỗi Type mismatch
    asp_nodes = [n.name for n in model.graph.node if 'asp' in n.name.lower()]
    print(f"[*] Found {len(asp_nodes)} ASP-related nodes. Adding them to block list...")

    print("[*] Converting to FLOAT16 with ASP node exclusion...")
    model_fp16 = float16.convert_float_to_float16(
        model,
        keep_io_types=True,
        node_block_list=asp_nodes
    )
    
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    onnx.save(model_fp16, OUTPUT_PATH)
    
    size_old = os.path.getsize(INPUT_PATH) / 1e6
    size_new = os.path.getsize(OUTPUT_PATH) / 1e6
    print(f"✅ SUCCESS: {OUTPUT_PATH} ({size_new:.2f} MB vs {size_old:.2f} MB)")

if __name__ == "__main__":
    quantize_fp16_selective()
