import librosa
import soundfile as sf

# Đọc file âm thanh
file_path = './Music-Human/Justin-Bieber-Love-Yourself-_PURPOSE-The-Movement_.wav'
y, sr = librosa.load(file_path)

# Thử điều chỉnh các tham số margin
y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(3.0, 3.0))


# Lưu giọng hát và âm nhạc thành các file âm thanh riêng biệt
harmonic_path = 'harmonic_audio.wav'
percussive_path = 'percussive_audio.wav'

sf.write(harmonic_path, y_harmonic, sr)
sf.write(percussive_path, y_percussive, sr)

print("Đã tách giọng hát và âm nhạc và lưu thành công!")
