import os
import threading
import numpy as np
from audio_processor import AudioProcessor
from whisper_engine import WhisperEngine
# Giả sử chúng ta import ECAPAEngine từ file có sẵn hoặc copy vào đây
# Để đơn giản cho app development, tôi sẽ import từ model_development nếu được, 
# hoặc copy logic sang đây để app chạy độc lập.

class AppEngine:
    def __init__(self, whisper_model, whisper_exe, ecapa_model=None):
        self.audio_proc = AudioProcessor()
        self.whisper = WhisperEngine(whisper_model, whisper_exe)
        self.ecapa = None
        if ecapa_model:
            # Sẽ khởi tạo ECAPAEngine ở đây
            pass
        
        self.is_auth_enabled = False
        self.is_running = False
        
        # Callback để gửi text về UI
        self.ui_callback = None

    def start(self, ui_callback):
        self.ui_callback = ui_callback
        self.is_running = True
        self.audio_proc.start()
        
        # Chạy vòng lặp xử lý trong thread riêng
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self.process_thread.start()

    def stop(self):
        self.is_running = False
        self.audio_proc.stop()

    def _process_loop(self):
        def on_segment(segment):
            # 1. Transcribe bằng Whisper
            text = self.whisper.transcribe(segment)
            if text and self.ui_callback:
                self.ui_callback("transcription", text)
            
            # 2. Authenticate bằng ECAPA (nếu bật)
            if self.is_auth_enabled and self.ecapa:
                embedding = self.ecapa.get_embedding(segment)
                # Logic so sánh với user_profile sẽ ở đây
                self.ui_callback("auth", "User Verified" if True else "Unknown")

        self.audio_proc.process_loop(on_segment)

    def set_auth(self, enabled):
        self.is_auth_enabled = enabled
        print(f"[Engine] Authentication {'ON' if enabled else 'OFF'}")
