import os
import subprocess
import re
import json
from datetime import datetime

def run_command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    stdout, stderr = process.communicate()
    return stdout, stderr

def parse_benchmark_output(stdout, stderr):
    results = {"peak_ram_mb": 0, "transcribe_ms": 0, "transcript": ""}
    ram_match = re.search(r"Maximum resident set size \(kbytes\): (\d+)", stderr)
    if ram_match: results["peak_ram_mb"] = int(ram_match.group(1)) / 1024
    full_time_match = re.search(r"total time\s*=\s*([\d.]+)\s*ms", stderr)
    if full_time_match: results["transcribe_ms"] = float(full_time_match.group(1))
    transcript_lines = []
    for line in stdout.split('\n'):
        if "-->" in line:
            content = re.sub(r'\[.*\]\s*', '', line).strip()
            if content: transcript_lines.append(content)
    results["transcript"] = " ".join(transcript_lines)
    return results

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    whisper_cli = os.path.join(project_root, "whisper.cpp/build/bin/whisper-cli")
    q5_1_model = os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q5_1/ggml-model-q5_1.bin")
    test_audio_m4a = os.path.join(project_root, "my_test_voice/pharse_2/my_voice_vie_1.m4a")
    test_audio_wav = os.path.join(project_root, "model_development/tests/tmp_test_q5.wav")
    
    keywords = ["giọng nói", "xác thực", "danh tính", "chìa khóa", "bật", "mở", "tắt", "thiết bị", "thông minh", "người dùng"]
    initial_prompt = ", ".join(keywords)

    print(f"--- BENCHMARK Q5_1 VỚI BEAM-SIZE = 1 ---")
    subprocess.run(["ffmpeg", "-y", "-i", test_audio_m4a, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", test_audio_wav], capture_output=True)
    
    # beam-size 1 (Greedy Search)
    cmd = f'/usr/bin/time -v {whisper_cli} -m {q5_1_model} -f {test_audio_wav} -l vi -t 4 --prompt "{initial_prompt}" --beam-size 1'
    stdout, stderr = run_command(cmd)
    res = parse_benchmark_output(stdout, stderr)
    
    print(f"   RAM: {res['peak_ram_mb']:.2f} MB")
    print(f"   Time: {res['transcribe_ms']:.2f} ms")
    print(f"   Result: {res['transcript']}")
    
    report_path = os.path.join(project_root, "benchmarks/q5_1_beam1_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "model": "Q5_1", "beam_size": 1, "results": res}, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(test_audio_wav): os.remove(test_audio_wav)

if __name__ == "__main__":
    main()
