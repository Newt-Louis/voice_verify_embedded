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
    
    models = [
        {"id": "M0", "name": "Base (FP16)", "path": os.path.join(project_root, "pretrained_models/ggml-whispersmall-multilingual/ggml-small.bin")},
        {"id": "M1", "name": "Q8_0 (8-bit)", "path": os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q8_0/ggml-model-q8_0.bin")},
        {"id": "M2", "name": "Q5_1 (5-bit)", "path": os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q5_1/ggml-model-q5_1.bin")},
        {"id": "M3", "name": "Q5_0 (5-bit)", "path": os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q5_0/ggml-model-q5_0.bin")},
        {"id": "M4", "name": "Q4_1 (4-bit)", "path": os.path.join(project_root, "quantization/ggml-whispersmall-multilingual/q4_1/ggml-model-q4_1.bin")}
    ]
    
    test_audio_m4a = os.path.join(project_root, "my_test_voice/pharse_2/my_voice_vie_1.m4a")
    test_audio_wav = os.path.join(project_root, "model_development/tests/tmp_final.wav")
    
    keywords = ["giọng nói", "xác thực", "danh tính", "chìa khóa", "bật", "mở", "tắt", "thiết bị", "thông minh"]
    initial_prompt = ", ".join(keywords)

    subprocess.run(["ffmpeg", "-y", "-i", test_audio_m4a, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", test_audio_wav], capture_output=True)
    
    final_results = []
    
    for m in models:
        if not os.path.exists(m["path"]): continue
        
        for beam in [5, 1]:
            print(f"--- BENCHMARK: {m['name']} | Beam: {beam} ---")
            cmd = (
                f'/usr/bin/time -v {whisper_cli} -m {m["path"]} -f {test_audio_wav} '
                f'-l vi -t 4 --prompt "{initial_prompt}" --beam-size {beam} '
                f'--entropy-thold 2.4 --logprob-thold -1.0'
            )
            stdout, stderr = run_command(cmd)
            res = parse_benchmark_output(stdout, stderr)
            res.update({"model_name": m["name"], "beam_size": beam, "file_size_mb": os.path.getsize(m["path"])/(1024*1024)})
            final_results.append(res)
            print(f"   RAM: {res['peak_ram_mb']:.2f} MB | Result: {res['transcript']}")

    report_path = os.path.join(project_root, "benchmarks/final_quantization_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "data": final_results}, f, ensure_ascii=False, indent=2)
    
    if os.path.exists(test_audio_wav): os.remove(test_audio_wav)
    print(f"\nBáo cáo cuối cùng đã sẵn sàng tại: {report_path}")

if __name__ == "__main__":
    main()
