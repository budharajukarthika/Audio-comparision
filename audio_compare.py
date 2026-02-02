import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
audio1, sr1 = librosa.load("audio1.wav.ogg", sr=None)
audio2, sr2 = librosa.load("audio2.wav.ogg", sr=None)
min_len = min(len(audio1), len(audio2))
audio1 = audio1[:min_len]
audio2 = audio2[:min_len]
time = np.linspace(0, min_len / sr1, min_len)
audio1_norm = (audio1 - np.mean(audio1)) / np.std(audio1)
audio2_norm = (audio2 - np.mean(audio2)) / np.std(audio2)
similarity_score = np.corrcoef(audio1_norm, audio2_norm)[0, 1]
plt.figure(figsize=(10, 4))
plt.plot(time, audio1, label="Reference (ref)", color="blue")
plt.plot(time, audio2, label="Pattern (pat)", color="orange")
plt.title("Audio Signal Comparison")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
fft_ref = np.abs(np.fft.fft(audio1))
fft_pat = np.abs(np.fft.fft(audio2))
fft_ref = fft_ref[:min_len // 2]
fft_pat = fft_pat[:min_len // 2]
similarity_score = 1 - cosine(fft_ref, fft_pat)
print(f"Similarity Score: {similarity_score:.4f}")