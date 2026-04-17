import os
import subprocess
import tempfile
import wave
import numpy as np

class WhisperEngine:
    def __init__(self, model_path, executable_path):
        self.model_path = model_path
        self.executable_path = executable_path
        
        if not os.path.exists(self.model_path):
            print(f"[Whisper] Cảnh báo: Không tìm thấy mô hình tại {self.model_path}")
        if not os.path.exists(self.executable_path):
            print(f"[Whisper] Cảnh báo: Không tìm thấy executable tại {self.executable_path}")

    def transcribe(self, audio_data: np.ndarray):
        """
        Nhận vào mảng numpy float32 (16kHz), lưu ra file tạm và gọi whisper.cpp
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            tmp_wav_path = tmp_wav.name
            self._save_wav(tmp_wav_path, audio_data)
        
        try:
            # Lệnh chạy whisper.cpp:
            # -m: model, -f: file, -nt: no timestamps, -l: language (vi/en/auto)
            cmd = [
                self.executable_path,
                "-m", self.model_path,
                "-f", tmp_wav_path,
                "-nt", 
                "-l", "en", # Mặc định tiếng Anh cho Demo
                "-t", "4"   # Sử dụng 4 threads
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                text = result.stdout.strip()
                return text
            else:
                print(f"[Whisper] Lỗi: {result.stderr}")
                return ""
        finally:
            if os.path.exists(tmp_wav_path):
                os.remove(tmp_wav_path)

    def _save_wav(self, file_path, audio_data):
        # Đảm bảo dữ liệu ở định dạng int16 cho WAV standard
        # Whisper.cpp có thể đọc float32 nhưng WAV chuẩn thường dùng int16
        audio_int16 = (audio_data * 32767).astype(np.int16)
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
