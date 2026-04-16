import os
import subprocess
import re
import json
from datetime import datetime

def run_command(cmd):
    """Chạy lệnh và trả về stdout, stderr."""
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    stdout, stderr = process.communicate()
    return stdout, stderr

def parse_benchmark_output(stdout, stderr):
    """Bóc tách thông tin từ output của whisper-cli."""
    results = {
        "peak_ram_mb": 0,
        "init_ms": 0,
        "transcribe_ms": 0,
        "transcript": ""
    }
    
    ram_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    if ram_match:
        results["peak_ram_mb"] = int(ram_match.group(1)) / 1024
        
    load_time_match = re.search(r"load time\s*=\s*([\d.]+)\s*ms", stderr)
    if load_time_match:
        results["init_ms"] = float(load_time_match.group(1))
        
    full_time_match = re.search(r"total time\s*=\s*([\d.]+)\s*ms", stderr)
    if full_time_match:
        results["transcribe_ms"] = float(full_time_match.group(1))
        
    transcript_lines = []
    for line in stdout.split('\n'):
        if "-->" in line:
            content = re.sub(r'\[.*\]\s*', '', line).strip()
            if content:
                transcript_lines.append(content)
    results["transcript"] = " ".join(transcript_lines)
    
    return results

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    whisper_cli = os.path.join(project_root, "whisper.cpp/build/bin/whisper-cli")
    base_model = os.path.join(project_root, "pretrained_models/ggml-whispersmall-multilingual/ggml-small.bin")
    q8_0_model = os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q8_0/ggml-model-q8_0.bin")
    q5_1_model = os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q5_1/ggml-model-q5_1.bin")
    
    # Sử dụng file audio my_voice_vie_1.m4a làm test case chuẩn
    test_audio_m4a = os.path.join(project_root, "my_test_voice/pharse_2/my_voice_vie_1.m4a")
    test_audio_wav = os.path.join(project_root, "model_development/tests/tmp_test.wav")
    
    # Dictionary Prompt tiếng Việt
    keywords = [
        "giọng nói", "xác thực", "danh tính", "chìa khóa", "bật", "mở", "tắt",
        "thiết bị", "thông minh", "người dùng", "hệ thống", "an ninh", "bảo mật",
        "alo", "xin chào", "bluetooth", "wifi", "mạng", "kết nối", "server", "máy chủ",
        "xe tự hành", "điều khiển", "ra lệnh", "tiếng Việt", "trí tuệ nhân tạo",
        "đăng nhập", "truy cập", "mật khẩu", "mã hóa", "dữ liệu", "offline", "cảm biến"
    ]
    initial_prompt = ", ".join(keywords)

    print(f"--- ĐANG CHUYỂN ĐỔI ÂM THANH: {os.path.basename(test_audio_m4a)} ---")
    subprocess.run(["ffmpeg", "-y", "-i", test_audio_m4a, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", test_audio_wav], 
                   capture_output=True)
    
    models = [
        {"id": "M0", "name": "Base (FP16)", "path": base_model},
        {"id": "M1", "name": "Q8_0 (8-bit)", "path": q8_0_model},
        {"id": "M2", "name": "Q5_1 (5-bit)", "path": q5_1_model}
    ]
    
    all_results = []
    
    for m in models:
        if not os.path.exists(m["path"]):
            print(f"BỎ QUA {m['name']}: Không tìm thấy file model.")
            continue
            
        print(f"--- ĐANG BENCHMARK MODEL: {m['name']} ---")
        
        # Chạy với các tham số tối ưu hóa
        cmd = (
            f'/usr/bin/time -v {whisper_cli} -m {m["path"]} -f {test_audio_wav} '
            f'-l vi -t 4 --prompt "{initial_prompt}" --beam-size 5 '
            f'--entropy-thold 2.4 --logprob-thold -1.0'
        )
        
        stdout, stderr = run_command(cmd)
        
        res = parse_benchmark_output(stdout, stderr)
        res["model_id"] = m["id"]
        res["model_name"] = m["name"]
        res["file_size_mb"] = os.path.getsize(m["path"]) / (1024 * 1024)
        all_results.append(res)
        
        print(f"   RAM: {res['peak_ram_mb']:.2f} MB")
        print(f"   Time: {res['transcribe_ms']:.2f} ms")
        print(f"   Result: {res['transcript']}")
        print("-" * 30)
    
    # Xuất báo cáo tổng hợp
    report_path = os.path.join(project_root, "benchmarks/quantization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "results": all_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nĐã lưu báo cáo tổng hợp tại: {report_path}")
    if os.path.exists(test_audio_wav): os.remove(test_audio_wav)

if __name__ == "__main__":
    main()
