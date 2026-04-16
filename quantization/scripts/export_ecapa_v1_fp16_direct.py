import torch
import os
import sys

# Đưa thư mục hiện tại vào path để import sb_patch
sys.path.append(os.getcwd())
from model_development.sb_patch import apply_patches
apply_patches()

from speechbrain.inference.speaker import EncoderClassifier

SAVE_DIR = "pretrained_models/ecapa"
OUTPUT_PATH = "quantization/exports/ecapa/v1_fp32/ecapa_fp16.onnx"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def export_fp16():
    print("[*] Loading PyTorch model for direct FP16 export...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=SAVE_DIR
    )
    
    # 1. Chuyển mô hình sang FP16
    model = classifier.mods.embedding_model
    model.eval()
    
    # Một số layer trong ASP của ECAPA có thể không hỗ trợ half() trên CPU (như Pooling)
    # Tuy nhiên, ta cứ thử convert toàn bộ xem export có xử lý được không.
    # Nếu lỗi, ta sẽ chuyển sang chiến thuật 'selective cast'.
    try:
        model.half()
        print("[*] Model converted to Half precision.")
        dtype = torch.float16
        dummy_input = torch.randn(1, 300, 80, dtype=dtype)
    except Exception as e:
        print(f"[!] Warning: model.half() failed on CPU: {e}. Falling back to float export then cast.")
        # Nếu model.half() lỗi trên CPU, ta vẫn để model là float32 
        # nhưng ta sẽ sử dụng ONNX export với tham số convert.
        return False

    print("[*] Exporting using legacy API with float16 dummy input...")
    from torch.onnx import utils
    
    utils.export(
        model,
        (dummy_input,),
        OUTPUT_PATH,
        export_params=True,
        opset_version=14, # Thử opset 14 cho FP16 tốt hơn
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'time'},
            'output': {0: 'batch_size'}
        }
    )
    
    if os.path.exists(OUTPUT_PATH):
        print(f"✅ SUCCESS DIRECT EXPORT: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1e6:.2f} MB)")
        return True
    return False

if __name__ == "__main__":
    if not export_fp16():
        print("[!] Direct export failed. Trying Selective Casting next...")
