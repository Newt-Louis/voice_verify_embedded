import torchaudio
import os
import huggingface_hub

def apply_patches():
    # Patch Torchaudio
    if not hasattr(torchaudio, 'list_audio_backends'):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    
    # Patch HuggingFace
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
                    f.write("# File giả mạo\n")
            return dummy_path
        return _orig_download(*args, **kwargs)
    huggingface_hub.hf_hub_download = _patched_download
    print("[*] Patches applied successfully.")
