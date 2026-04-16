import os
import sys
import requests
from tqdm import tqdm

def download_file(url, destination):
    """Tải file với thanh tiến trình."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(destination))
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("LỖI: Có lỗi xảy ra trong quá trình tải.")
        return False
    return True

def main():
    # Model: Whisper Small Multilingual
    # URL: https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
    model_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "../../pretrained_models/ggml-whispersmall-multilingual")
    target_path = os.path.join(target_dir, "ggml-small.bin")
    
    # Đảm bảo thư mục tồn tại
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Bắt đầu tải model Whisper-Small (GGML) từ: {model_url}")
    print(f"Lưu tại: {target_path}")
    
    if os.path.exists(target_path):
        print(f"File đã tồn tại: {target_path}. Bỏ qua tải xuống.")
        sys.exit(0)
        
    try:
        success = download_file(model_url, target_path)
        if success:
            print("\nTải xuống THÀNH CÔNG!")
        else:
            print("\nTải xuống THẤT BẠI!")
    except Exception as e:
        print(f"\nĐã xảy ra lỗi: {e}")

if __name__ == "__main__":
    main()
