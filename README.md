# Security Voice Embedded (Project History & Lessons Learned)

## 🎯 Project Overview
This project aims to build a robust Voice Authentication and Speech-to-Text system for resource-constrained devices (Android Mobile, Car PC).

## 📈 Evolution & Milestones

### Phase 1: ECAPA-TDNN (Speaker Verification) - [SUCCESS]
- Successfully quantized ECAPA-TDNN using ONNX.
- Achieved **RAM < 20MB** and **Model Size ~18MB (INT8)**.

### Phase 2: Whisper-Small STT (Research & Pivot)

#### 🧪 Attempt 1: PyTorch/ONNX Quantization (Legacy Research)
- **Method:** Exporting from safetensors to ONNX.
- **Outcome:** FAILED (RAM > 1GB, too heavy for embedded).

#### 🚀 Attempt 2: Whisper.cpp & GGML (Optimization Success)
- **Method:** Used `whisper.cpp` with GGML quantization.
- **Results:** 
    - **Q5_0 (Beam 1):** ~167MB Model | **~351MB Peak RAM**.
- **Outcome:** SUCCESS in resource management.

#### 🔀 Attempt 3: Strategic Pivot at Step 4 (Current)
- **Observation:** Whisper-Small Multilingual struggles with Vietnamese accuracy (hallucinations) despite prompting. English accuracy remains superior.
- **Pivot:** 
    - **Current Path:** Proceed to **Phase 3 (Deployment)** using the Q5_0 model as an English STT Demo to showcase RAM efficiency and hardware stability.
    - **Deferred Path:** Fine-tuning with a dedicated Vietnamese dataset will be required for production-grade Vietnamese STT, followed by re-quantization.

## 🛠️ Technical Stack
- **Verification:** ECAPA-TDNN (ONNX INT8).
- **Transcription:** Whisper Small (GGML Q5_0 via whisper.cpp).
- **VAD:** Silero VAD (ONNX).
- **Audio Processing:** FFmpeg.

## ⚠️ Lessons Learned
1. **Tool Choice:** `whisper.cpp` is mandatory for < 400MB RAM targets.
2. **Model Limitations:** Multilingual models in small sizes have a strong English bias; Vietnamese STT requires specialized fine-tuning for high accuracy.
3. **Efficiency vs. Search:** Using `beam-size = 1` (Greedy Search) is highly recommended for short commands to save ~100MB RAM.
