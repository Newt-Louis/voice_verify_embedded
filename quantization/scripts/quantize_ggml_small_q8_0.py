import os
import subprocess
import sys
import time

def main():
    # Cấu hình đường dẫn
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    quantize_tool = os.path.join(project_root, "whisper.cpp/build/bin/whisper-quantize")
    source_model = os.path.join(project_root, "pretrained_models/ggml-whispersmall-multilingual/ggml-small.bin")
    output_dir = os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q8_0")
    output_model = os.path.join(output_dir, "ggml-model-q8_0.bin")
    
    # Đảm bảo thư mục output tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(source_model):
        print(f"LỖI: Không tìm thấy model gốc tại {source_model}")
        sys.exit(1)
        
    if not os.path.exists(quantize_tool):
        print(f"LỖI: Không tìm thấy công cụ quantize tại {quantize_tool}. Hãy chắc chắn bạn đã build whisper.cpp thành công.")
        sys.exit(1)
        
    print(f"--- BẮT ĐẦU LƯỢNG TỬ HÓA Q8_0 ---")
    print(f"Nguồn: {source_model}")
    print(f"Đích: {output_model}")
    
    start_time = time.time()
    
    # Lệnh thực thi: ./whisper-quantize <input_file> <output_file> <type>
    cmd = [quantize_tool, source_model, output_model, "q8_0"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if os.path.exists(output_model):
            size_mb = os.path.getsize(output_model) / (1024 * 1024)
            print(f"--- HOÀN THÀNH ---")
            print(f"Thời gian thực hiện: {duration:.2f} giây")
            print(f"Kích thước model mới: {size_mb:.2f} MB")
        else:
            print("LỖI: Script chạy xong nhưng không tìm thấy file output.")
            
    except subprocess.CalledProcessError as e:
        print(f"LỖI khi chạy quantize: {e}")
        print(f"Chi tiết: {e.stderr}")
        sys.exit(1)

if __name__ == "__main__":
    main()
