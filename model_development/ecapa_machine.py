import numpy as np
import time
from ecapa_engine import ECAPAEngine

class ECAPAMachine:
    """
    [CÁI MÁY CHẠY - MACHINE]
    Nhiệm vụ: Quản lý băng chuyền dữ liệu (Buffer), cắt cửa sổ trượt (Sliding Window).
    Máy này sẽ được "lắp" vào Middleware hoặc Service của ứng dụng sau này.
    """
    def __init__(self, master_signature, threshold=0.30):
        print(f"[*] Starting ECAPA Machine...")
        # 1. Lắp Lõi (Engine) vào Máy
        self.engine = ECAPAEngine()
        
        # 2. Cài đặt thông số nhận diện
        self.master_sig = master_signature
        self.threshold = threshold
        
        # 3. Cấu hình Băng chuyền Cửa sổ trượt
        self.sample_rate = 16000
        self.window_size_sec = 3.0
        self.stride_sec = 1.0
        
        self.window_samples = int(self.window_size_sec * self.sample_rate) # 48.000
        self.stride_samples = int(self.stride_sec * self.sample_rate) # 16.000
        
        # Cái xô (Buffer) chứa âm thanh
        self.audio_buffer = np.array([], dtype=np.float32)

    def _cosine_similarity(self, a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def feed_audio(self, audio_chunk):
        """
        Nạp âm thanh vào máy. Trả về kết quả xác thực nếu cửa sổ trượt đầy.
        """
        self.audio_buffer = np.concatenate((self.audio_buffer, audio_chunk))
        results = []
        
        while len(self.audio_buffer) >= self.window_samples:
            current_window = self.audio_buffer[:self.window_samples]
            
            # Thực hiện xác thực qua Lõi
            start_time = time.perf_counter()
            signature = self.engine.get_embedding(current_window)
            score = self._cosine_similarity(self.master_sig, signature)
            latency = (time.perf_counter() - start_time) * 1000
            
            # Trả về kết quả cho tầng Middleware/App xử lý tiếp
            is_authenticated = score >= self.threshold
            results.append({
                "authenticated": is_authenticated,
                "score": float(score),
                "latency_ms": latency
            })
            
            # Trượt băng chuyền
            self.audio_buffer = self.audio_buffer[self.stride_samples:]
            
        return results

# ==========================================
# MÔ PHỎNG CÁCH APP SỬ DỤNG MÁY NÀY
# ==========================================
if __name__ == "__main__":
    dummy_master = np.random.randn(192).astype(np.float32)
    machine = ECAPAMachine(master_signature=dummy_master)
    
    print("\n[APP] Đang đẩy audio vào máy...")
    for i in range(10):
        fake_chunk = np.random.randn(8000).astype(np.float32) 
        res = machine.feed_audio(fake_chunk)
        if res:
            for r in res:
                print(f" -> KẾT QUẢ MÁY TRẢ VỀ: Auth={r['authenticated']} | Score={r['score']:.4f}")
