import os, time, random, psutil, torch, torchaudio, librosa, librosa.display
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
import huggingface_hub
original_download = huggingface_hub.hf_hub_download
def patched_download(*args, **kwargs):
    if 'use_auth_token' in kwargs:
        kwargs['token'] = kwargs.pop('use_auth_token')
    for deprecated_arg in ['local_dir_use_symlinks', 'force_filename']:
        kwargs.pop(deprecated_arg, None)
    filename = kwargs.get('filename')
    if filename is None and len(args) > 1:
        filename = args[1]
    try:
        return original_download(*args, **kwargs)
    except Exception as e:
        if filename == "custom.py" and "404" in str(e):
            dummy_path = os.path.abspath("dummy_custom.py")
            with open(dummy_path, "w") as f: pass
            return dummy_path
        raise e
huggingface_hub.hf_hub_download = patched_download

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.manifold import TSNE
from torch.nn import CosineSimilarity
from speechbrain.inference.speaker import EncoderClassifier
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (CHỈNH SỬA Ở ĐÂY)
# ==========================================
# File để vẽ biểu đồ trực quan
AUDIO_FILE = "vox1_test_wav/wav/id10270/5r0dWxy17C8/00001.wav"
VOXCELEB_DIR = "vox1_test_wav/wav"

NUM_SPEAKERS = 5
FILES_PER_SPEAKER = 10
device = torch.device("cpu")
cos_sim = CosineSimilarity(dim=0, eps=1e-6)

def print_ram(stage_name):
    mem_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    print(f"[{stage_name}] RAM tiêu thụ: {mem_mb:.2f} MB")

def load_audio_tensor(file_path):
    """Fix lỗi ImportError: TorchCodec is required for load_with_torchcodec"""
    sig, _ = librosa.load(file_path, sr=16000)
    return torch.tensor(sig).unsqueeze(0)

# ==========================================
# MODULE 1: TRỰC QUAN HÓA ÂM THANH
# ==========================================
def generate_visual_report(file_path):
    print("\n--- PHẦN 1: TRỰC QUAN HÓA ĐẶC TRƯNG ÂM THANH ---")
    print(f"Đang phân tích phổ âm cho file: {file_path}")
    y, sr = librosa.load(file_path, sr=16000)
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform (Sóng âm thanh gốc)')

    plt.subplot(3, 1, 2)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=80, fmax=8000)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis='mel', sr=sr, fmax=8000)
    plt.title('Mel-Spectrogram (Phổ âm - Dữ liệu đầu vào cho AI)')

    plt.subplot(3, 1, 3)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    plt.plot(librosa.times_like(f0), f0, color='cyan', linewidth=2)
    plt.title('Pitch Contour (Đường cong cao độ giọng nói)')

    plt.tight_layout()
    output_img = "report_audio_viz.png"
    plt.savefig(output_img)
    plt.close()
    print(f"-> Đã lưu thành công biểu đồ vào file: {output_img} (Sẵn sàng chèn vào báo cáo)")


# ==========================================
# MODULE 2: TẢI MÔ HÌNH VÀ TẠO BÀI THI
# ==========================================
def load_all_models():
    print("\n--- PHẦN 2: KHỞI TẠO VÀ ĐO LƯỜNG TÀI NGUYÊN MÔ HÌNH ---")
    print_ram("Base (Chưa load)")

    print("1. Đang tải X-Vector (Kaldi)...")
    xvector = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb",
                                             savedir="pretrained_models/xvect", run_opts={"device": device})
    print_ram("Sau khi load X-Vector")

    print("2. Đang tải ECAPA-TDNN...")
    ecapa = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir="pretrained_models/ecapa", run_opts={"device": device})
    print_ram("Sau khi load ECAPA-TDNN")

    print("3. Đang tải Wav2Vec2 (Cảnh báo: Rất nặng)...")
    w2v2_ext = AutoFeatureExtractor.from_pretrained("anton-l/wav2vec2-base-superb-sv")
    w2v2_mod = AutoModelForAudioXVector.from_pretrained("anton-l/wav2vec2-base-superb-sv").to(device)
    print_ram("Sau khi load Wav2Vec2")

    return xvector, ecapa, (w2v2_ext, w2v2_mod)

def get_speaker_files():
    speaker_files = {}
    speakers = [s for s in os.listdir(VOXCELEB_DIR) if os.path.isdir(os.path.join(VOXCELEB_DIR, s))][:NUM_SPEAKERS]
    for spk in speakers:
        files = []
        for root, _, filenames in os.walk(os.path.join(VOXCELEB_DIR, spk)):
            files.extend([os.path.join(root, f) for f in filenames if f.endswith(".wav")])
        if len(files) >= FILES_PER_SPEAKER:
            speaker_files[spk] = random.sample(files, FILES_PER_SPEAKER)
    return speaker_files

def prepare_test_data(speaker_files):
    pairs = []
    # Positive pairs
    for files in speaker_files.values():
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pairs.append((files[i], files[j], 1))
    # Negative pairs
    spk_list = list(speaker_files.keys())
    for i in range(len(spk_list)):
        for j in range(i + 1, len(spk_list)):
            pairs.append((random.choice(speaker_files[spk_list[i]]), random.choice(speaker_files[spk_list[j]]), 0))
    random.shuffle(pairs)
    return pairs[:150]


# ==========================================
# MODULE 3: BENCHMARK (ĐO ĐỘ TRỄ & ĐỘ CHÍNH XÁC)
# ==========================================
def compute_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr[np.nanargmin(np.absolute((1 - tpr) - fpr))]


def run_benchmark(model, name, pairs, is_wav2vec=False, w2v2_ext=None):
    print(f"\n[Testing] Đang chạy đánh giá: {name}...")
    total_time = 0
    scores, labels = [], []

    for file1, file2, label in pairs:
        sig1 = load_audio_tensor(file1)
        sig2 = load_audio_tensor(file2)

        start = time.time()
        with torch.no_grad():
            if is_wav2vec:
                in1 = w2v2_ext(sig1.numpy()[0], sampling_rate=16000, return_tensors="pt")
                emb1 = model(**in1).embeddings.squeeze()
                in2 = w2v2_ext(sig2.numpy()[0], sampling_rate=16000, return_tensors="pt")
                emb2 = model(**in2).embeddings.squeeze()
            else:
                emb1 = model.encode_batch(sig1).squeeze()
                emb2 = model.encode_batch(sig2).squeeze()
        total_time += (time.time() - start)

        scores.append(cos_sim(emb1, emb2).item())
        labels.append(label)

    avg_latency = (total_time / (len(pairs) * 2)) * 1000
    eer = compute_eer(labels, scores)
    print(f"  -> Độ trễ (Latency): {avg_latency:.2f} ms / file")
    print(f"  -> Tỉ lệ lỗi (EER):  {eer * 100:.2f} %")


def generate_tsne_report(model, model_name, speaker_files, is_wav2vec=False, w2v2_ext=None):
    print(f"\n[T-SNE] Đang ép Vector và vẽ Phân cụm cho: {model_name}...")
    embeddings, labels = [], []
    speaker_names = list(speaker_files.keys())

    for spk_idx, spk in enumerate(speaker_names):
        for file_path in speaker_files[spk]:
            sig = load_audio_tensor(file_path)
            with torch.no_grad():
                if is_wav2vec:
                    in1 = w2v2_ext(sig.numpy()[0], sampling_rate=16000, return_tensors="pt")
                    emb = model(**in1).embeddings.squeeze().numpy()
                else:
                    emb = model.encode_batch(sig).squeeze().numpy()
            embeddings.append(emb)
            labels.append(spk_idx)

    # Chạy T-SNE để ép 256 chiều xuống 2 chiều
    tsne = TSNE(n_components=2, random_state=42, perplexity=15)
    embeddings_2d = tsne.fit_transform(np.array(embeddings))

    # Vẽ và lưu biểu đồ
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10', alpha=0.8, s=100)
    plt.legend(handles=scatter.legend_elements()[0], labels=speaker_names, title="Danh tính (Speakers)")
    plt.title(f'T-SNE Clustering - {model_name}')

    # Rút gọn tên file cho đẹp
    short_name = model_name.split()[0].replace("-", "")
    output_img = f"report_tsne_{short_name}.png"
    plt.savefig(output_img)
    plt.close()
    print(f"  -> Đã lưu biểu đồ vào: {output_img}")


if __name__ == "__main__":
    print("==================================================")
    print(" HỆ THỐNG ĐÁNH GIÁ MÔ HÌNH XÁC THỰC GIỌNG NÓI ")
    print("==================================================")

    generate_visual_report(AUDIO_FILE)
    xvect, ecapa, (w2v_ext, w2v_mod) = load_all_models()

    print("\n--- PHẦN 3: ĐÁNH GIÁ ĐỘ CHÍNH XÁC (BENCHMARK) ---")
    speaker_files = get_speaker_files()
    test_pairs = prepare_test_data(speaker_files)

    # 1. Chạy Benchmark EER
    run_benchmark(xvect, "X-Vector (Cổ điển)", test_pairs)
    run_benchmark(ecapa, "ECAPA-TDNN (Cân bằng Edge)", test_pairs)
    run_benchmark(w2v_mod, "Wav2Vec2 (Transformer Nặng)", test_pairs, is_wav2vec=True, w2v2_ext=w2v_ext)

    print("\n--- PHẦN 4: VẼ BIỂU ĐỒ PHÂN CỤM VECTOR ---")
    # 2. Vẽ biểu đồ T-SNE
    generate_tsne_report(xvect, "X-Vector", speaker_files)
    generate_tsne_report(ecapa, "ECAPA-TDNN", speaker_files)
    generate_tsne_report(w2v_mod, "Wav2Vec2", speaker_files, is_wav2vec=True, w2v2_ext=w2v_ext)

    print("\n==================================================")
    print(" ĐÃ HOÀN THÀNH TOÀN BỘ! Vui lòng kiểm tra các file ảnh .png trong thư mục.")