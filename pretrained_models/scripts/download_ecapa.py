import torch
import torchaudio
import huggingface_hub
import os
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

_orig_download = huggingface_hub.hf_hub_download


def _patched_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')

    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else "")
    if "custom.py" in filename:
        dummy_path = os.path.abspath("dummy_custom.py")
        if not os.path.exists(dummy_path):
            with open(dummy_path, "w") as f:
                f.write("# File giả mạo\n")
        return dummy_path
    return _orig_download(*args, **kwargs)
huggingface_hub.hf_hub_download = _patched_download

from speechbrain.inference.speaker import EncoderClassifier


def download_and_save_model():
    SAVE_DIR = "pretrained_models/ecapa"

    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=SAVE_DIR,
        run_opts={"device": "cpu"}
    )

    print("\n✅ HOÀN TẤT!")
    print(f"Toàn bộ file mô hình đã được trích xuất an toàn tại: {SAVE_DIR}")


if __name__ == "__main__":
    download_and_save_model()