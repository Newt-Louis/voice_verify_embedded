import onnxruntime as ort
import numpy as np
import os

# Mô hình đã lượng tử hóa (Int8) để ép nhẹ nhất có thể
MODEL_PATH = "model_quantized_int8.onnx"

class VoiceAIEngine:
    def __init__(self):
        # Sử dụng CPU vì chúng ta nhắm tới hệ thống nhúng/mobile (ít tốn pin hơn GPU/NPU nếu mô hình nhỏ)
        self.session = None
        if os.path.exists(MODEL_PATH):
            self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
            print("[AI Engine] Đã nạp mô hình offline hoàn tất.")
        else:
            print("[AI Engine] Cảnh báo: Chưa tìm thấy file mô hình tại " + MODEL_PATH)

    def extract_vector(self, audio_data: np.ndarray):
        """
        Chuyển đổi âm thanh trực tiếp thành Vector đặc trưng (Embedding)
        """
        if self.session is None:
            return None
        
        # Tiền xử lý dữ liệu âm thanh (Hardcore: Chỉnh trực tiếp vào Tensor)
        # Giả định đầu vào là (1, 80, T) - Mel Spectrogram
        inputs = {self.session.get_inputs()[0].name: audio_data}
        outputs = self.session.run(None, inputs)
        return outputs[0]

# Chế độ chờ lệnh (Background Logic)
def background_listener():
    print("[Service] Đang chạy ngầm để chờ câu lệnh xác thực...")
    # Logic: Luôn lắng nghe buffer âm thanh siêu nhỏ, nếu có tiếng động thì mới đánh thức AI
    pass

if __name__ == "__main__":
    engine = VoiceAIEngine()
