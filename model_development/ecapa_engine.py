import numpy as np, os
import onnxruntime as ort
import torchaudio
import torch

INTERNAL_MODEL_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
class ECAPAEngine:
    def __init__(self, provider='CPUExecutionProvider'):
        if not os.path.exists(INTERNAL_MODEL_PATH):
            raise FileNotFoundError(f"[FATAL] Engine Core không tìm thấy mô hình tại: {INTERNAL_MODEL_PATH}")
        self.session = ort.InferenceSession(INTERNAL_MODEL_PATH, providers=[provider])
        self.input_name = self.session.get_inputs()[0].name
        print(f"[*] Engine Core loaded: {INTERNAL_MODEL_PATH}")

    def _extract_fbank(self, waveform):
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            num_mel_bins=80,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10
        )
        return fbank.unsqueeze(0).numpy()

    def _mean_var_norm(self, features):
        mean = np.mean(features, axis=1, keepdims=True)
        std = np.std(features, axis=1, keepdims=True)
        return (features - mean) / (std + 1e-5)

    def get_embedding(self, audio_data):
        feats = self._extract_fbank(audio_data)
        feats_norm = self._mean_var_norm(feats).astype(np.float32)
        ort_outs = self.session.run(None, {self.input_name: feats_norm})
        return ort_outs[0].flatten()
