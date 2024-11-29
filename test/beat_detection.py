import librosa
import matplotlib.pyplot as plt
import numpy as np
import librosa.display

# Load the audio file
audio_file = "C:\\Users\\KRISHU\\Desktop\\butterfly.wav"
y, sr = librosa.load(audio_file)

# Run the beat tracker
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
tempo = float(tempo)

# Print estimated tempo
print(f"Estimated tempo: {tempo:.2f} beats per minute")

# Convert beat frames to timestamps in seconds
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
print("Beat timestamps:", beat_times)

# 1. Plot the audio waveform
plt.figure(figsize=(14, 5))
plt.plot(y, label='Waveform', color='lightblue')
for beat in beat_times:
    plt.axvline(x=beat * sr, color='red', linestyle='--', linewidth=1)
plt.title(f'Audio Waveform with Beat Tracking (Tempo: {tempo:.2f} BPM)')
plt.xlabel('Samples')  # 横轴：样本数
plt.ylabel('Amplitude')  # 纵轴：幅度
plt.legend()
plt.grid()
plt.show()

# 2. Frequency analysis: Calculate and plot the short-time Fourier transform (STFT)
D = librosa.stft(y)
DB = librosa.amplitude_to_db(np.abs(D), ref=np.max)
plt.figure(figsize=(14, 5))
img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.title('Spectrogram (STFT)')
plt.xlabel('Time (s)')  # 横轴：时间（秒）
plt.ylabel('Frequency (Hz)')  # 纵轴：频率（赫兹）
plt.colorbar(format='%+2.0f dB')
plt.show()

# 3. Calculate and plot MFCC
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
plt.figure(figsize=(14, 5))
img_mfcc = librosa.display.specshow(mfccs, sr=sr, x_axis='time', cmap='coolwarm')
plt.title('MFCC')
plt.xlabel('Time (s)')  # 横轴：时间（秒）
plt.ylabel('MFCC Coefficients')  # 纵轴：MFCC系数
plt.colorbar()
plt.show()

# 4. Calculate and plot RMS energy
rms = librosa.feature.rms(y=y)
plt.figure(figsize=(14, 5))
plt.plot(librosa.times_like(rms), rms[0], label='RMS Energy', color='green')
plt.title('RMS Energy Over Time')
plt.xlabel('Time (s)')  # 横轴：时间（秒）
plt.ylabel('RMS Energy')  # 纵轴：均方根能量
plt.legend()
plt.grid()
plt.show()
