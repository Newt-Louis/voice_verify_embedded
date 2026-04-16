# Security Voice Embedded (Project History & Lessons Learned)

## 🎯 Project Overview
This project aims to build a robust Voice Authentication and Speech-to-Text system for resource-constrained devices (Android Mobile, Car PC).

## 📈 Evolution & Milestones

### Phase 1: ECAPA-TDNN (Speaker Verification) - [SUCCESS]
- Successfully quantized ECAPA-TDNN using ONNX.
- Achieved **RAM < 20MB** and **Model Size ~18MB (INT8)**.
- This phase established the baseline for embedding-based speaker verification.

### Phase 2: Whisper-Small STT (Research & Pivot)

#### 🧪 Attempt 1: PyTorch/ONNX Quantization (Legacy Research)
- **Method:** Directly exporting `openai/whisper-small` from safetensors to ONNX via PyTorch/Hugging Face Optimum.
- **Goal:** Use standard ONNX Runtime for inference.
- **Results:**
    - **FP32 Model Size:** ~1.1 GB (Encoder: 337MB | Decoder: 739MB).
    - **Outcome:** FAILED to meet RAM constraints (< 400MB).
- **Lessons Learned:** 
    - The PyTorch/ONNX runtime overhead and the raw model structure were too heavy for the target 1GB RAM hardware.
    - Standard quantization didn't squeeze the model enough while maintaining Vietnamese accuracy on typical STT engines.

#### 🚀 Attempt 2: Whisper.cpp & GGML/GGUF (Current Path - SUCCESS)
- **Method:** Switched to `whisper.cpp` (C++ implementation) and GGML quantization.
- **Tooling:** Built `whisper-quantize` using CMake from the `whisper.cpp` submodule.
- **Quantized Models (Multilingual):**
    - **Q8_0:** ~252MB | Peak RAM ~437MB.
    - **Q5_1:** ~181MB | Peak RAM **~366MB** (PASSED threshold).
    - **Q4_1:** ~153MB | Peak RAM **~338MB** (PASSED threshold).
- **Outcome:** Successfully achieved the goal of running Whisper-Small STT under 400MB RAM.

## 🛠️ Technical Stack
- **Verification:** ECAPA-TDNN (ONNX INT8).
- **Transcription:** Whisper Small (GGML Q5_1 via whisper.cpp).
- **VAD:** Silero VAD (ONNX).
- **Audio Processing:** FFmpeg (C/C++ integration).

## ⚠️ Critical Notes
- **Never** install libraries into the system Python; always use `.venv`.
- **Always** prioritize `whisper.cpp` for STT on embedded devices due to its superior memory management (memory-mapped files).
