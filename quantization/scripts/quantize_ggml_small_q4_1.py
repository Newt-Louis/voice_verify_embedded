import os
import subprocess
import time

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    
    quantize_tool = os.path.join(project_root, "whisper.cpp/build/bin/whisper-quantize")
    source_model = os.path.join(project_root, "pretrained_models/ggml-whispersmall-multilingual/ggml-small.bin")
    output_dir = os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q4_1")
    output_model = os.path.join(output_dir, "ggml-model-q4_1.bin")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"--- BẮT ĐẦU LƯỢNG TỬ HÓA Q4_1 ---")
    cmd = [quantize_tool, source_model, output_model, "q4_1"]
    
    start_time = time.time()
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        size_mb = os.path.getsize(output_model) / (1024 * 1024)
        print(f"--- HOÀN THÀNH ---")
        print(f"Thời gian: {duration:.2f} giây | Kích thước: {size_mb:.2f} MB")
    except subprocess.CalledProcessError as e:
        print(f"LỖI: {e.stderr}")

if __name__ == "__main__":
    main()
