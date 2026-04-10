import torch
import os
import torchaudio

# PATCH
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

SAVE_DIR = "pretrained_models/ecapa"
OUTPUT_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def export():
    print("[*] Loading model...")
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=SAVE_DIR
    )
    # Lấy mô hình gốc
    model = classifier.mods.embedding_model
    model.eval()

    # Input: (Batch, Time, Features)
    dummy_input = torch.randn(1, 300, 80)

    print("[*] Exporting using legacy API...")
    
    # Ép sử dụng legacy exporter bằng cách gọi qua torch.onnx._export
    from torch.onnx import utils
    
    utils.export(
        model,
        (dummy_input,),
        OUTPUT_PATH,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 1: 'time'},
            'output': {0: 'batch_size'}
        }
    )
    
    if os.path.exists(OUTPUT_PATH):
        print(f"✅ SUCCESS: {OUTPUT_PATH} ({os.path.getsize(OUTPUT_PATH)/1e6:.2f} MB)")

if __name__ == "__main__":
    export()
