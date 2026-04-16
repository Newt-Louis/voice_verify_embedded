import numpy as np
import onnxruntime as ort
import os
import torch
import torchaudio
import subprocess
from onnxruntime.quantization import CalibrationDataReader

# Patch Torchaudio (vết sẹo từ Mốc 0)
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

class ECAPACalibrationReader(CalibrationDataReader):
    def __init__(self, audio_dir, input_name):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith(('.m4a', '.wav'))]
        self.input_name = input_name
        self.data_list = iter(self._preprocess())

    def _load_audio_ffmpeg(self, path):
        command = [
            '/usr/bin/ffmpeg', '-i', path, '-ac', '1', '-ar', '16000',
            '-f', 'f32le', '-hide_banner', '-loglevel', 'error', 'pipe:1'
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, _ = process.communicate()
        return torch.tensor(np.frombuffer(out, dtype=np.float32)).unsqueeze(0)

    def _extract_feats(self, sig):
        fbank = torchaudio.compliance.kaldi.fbank(
            sig, num_mel_bins=80, sample_frequency=16000, frame_length=25, frame_shift=10
        )
        mean = torch.mean(fbank, dim=0, keepdim=True)
        std = torch.std(fbank, dim=0, keepdim=True)
        fbank_norm = (fbank - mean) / (std + 1e-5)
        return fbank_norm.unsqueeze(0).numpy().astype(np.float32)

    def _preprocess(self):
        calib_data = []
        # Chỉ lấy tối đa 20 mẫu để calibration cho nhanh
        for f_path in self.audio_files[:20]:
            sig = self._load_audio_ffmpeg(f_path)
            if sig is not None:
                feats = self._extract_feats(sig)
                calib_data.append({self.input_name: feats})
        return calib_data

    def get_next(self):
        return next(self.data_list, None)

def quantize_int8_static(model_fp16_path, output_int8_path):
    print(f"[*] Đang nạp mô hình FP16 gốc: {model_fp16_path}")
    
    # 1. Khởi tạo reader với dữ liệu thực tế
    audio_dir = "my_test_voice/pharse_2"
    input_name = "input" # Theo cấu hình ONNX đã biết
    dr = ECAPACalibrationReader(audio_dir, input_name)
    
    # 2. Thực hiện lượng tử hóa Static
    import onnxruntime
    from onnxruntime.quantization import quantize_static, QuantType
    
    print("[*] Bắt đầu quá trình nén INT8 Static (Yêu cầu Calibration)...")
    quantize_static(
        model_input=model_fp16_path,
        model_output=output_int8_path,
        calibration_data_reader=dr,
        quant_format=onnxruntime.quantization.QuantFormat.QDQ, # Chuẩn nén hiện đại
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8
    )
    
    size_old = os.path.getsize(model_fp16_path) / 1e6
    size_new = os.path.getsize(output_int8_path) / 1e6
    print(f"✅ HOÀN TẤT MỐC 2! File INT8: {output_int8_path} ({size_new:.2f} MB vs {size_old:.2f} MB)")

if __name__ == "__main__":
    FP32_PATH = "quantization/exports/ecapa/v0_float32/ecapa_fp32.onnx"
    INT8_PATH = "quantization/exports/ecapa/v2_int8_calib/ecapa_int8_static.onnx"
    os.makedirs(os.path.dirname(INT8_PATH), exist_ok=True)
    quantize_int8_static(FP32_PATH, INT8_PATH)
