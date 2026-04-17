import numpy as np
import sounddevice as sd
import threading
import queue
import time

class AudioProcessor:
    def __init__(self, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.stream = None
        
        # Buffer cho Whisper (3s window, 1s slide)
        self.window_size = int(sample_rate * 3)
        self.slide_size = int(sample_rate * 1)
        self.audio_buffer = np.zeros(0, dtype=np.float32)
        
        # Hooks cho VAD và NS
        self.vad_enabled = True
        self.ns_enabled = True

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(f"[Audio] Status: {status}")
        # Đưa dữ liệu vào queue
        self.audio_queue.put(indata.copy().flatten())

    def start(self):
        self.is_running = True
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self._audio_callback
        )
        self.stream.start()
        print("[Audio] Đã bắt đầu luồng ghi âm.")

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        print("[Audio] Đã dừng luồng ghi âm.")

    def process_loop(self, callback_on_segment):
        """
        Vòng lặp xử lý: Lấy dữ liệu từ queue, áp dụng NS/VAD, 
        và gọi callback khi có đủ dữ liệu cho Whisper.
        """
        while self.is_running:
            try:
                chunk = self.audio_queue.get(timeout=1)
            except queue.Empty:
                continue
            
            # 1. Noise Suppression (NS) - Placeholder
            processed_chunk = self.apply_ns(chunk)
            
            # 2. Voice Activity Detection (VAD)
            if self.apply_vad(processed_chunk):
                # Nếu có tiếng người, đưa vào buffer trượt
                self.audio_buffer = np.append(self.audio_buffer, processed_chunk)
                
                # Kiểm tra nếu đủ độ dài cửa sổ (3s)
                if len(self.audio_buffer) >= self.window_size:
                    # Gửi segment đi xử lý (Whisper/ECAPA)
                    segment = self.audio_buffer[:self.window_size]
                    callback_on_segment(segment)
                    
                    # Trượt cửa sổ: Giữ lại (Window - Slide) dữ liệu
                    self.audio_buffer = self.audio_buffer[self.slide_size:]
            else:
                # Nếu không có tiếng người, xóa bớt buffer cũ để tránh lag
                if len(self.audio_buffer) > 0:
                    self.audio_buffer = self.audio_buffer[len(processed_chunk):]

    def apply_ns(self, chunk):
        # TODO: Tích hợp thư viện NS thực tế
        return chunk

    def apply_vad(self, chunk):
        # Đơn giản: Kiểm tra năng lượng (RMS)
        rms = np.sqrt(np.mean(chunk**2))
        # Ngưỡng năng lượng (cần điều chỉnh tùy môi trường)
        threshold = 0.01 
        return rms > threshold
