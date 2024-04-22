import librosa
import soundfile as sf

# Đọc file âm thanh
file_path = './Music/justin_cold_water.wav'
y, sr = librosa.load(file_path)

# Tách riêng giọng hát và âm nhạc
y_harmonic, y_percussive = librosa.effects.hpss(y)

# Lưu giọng hát và âm nhạc thành các file âm thanh riêng biệt
harmonic_path = 'harmonic_audio.wav'
percussive_path = 'percussive_audio.wav'

sf.write(harmonic_path, y_harmonic, sr)
sf.write(percussive_path, y_percussive, sr)

print("Đã tách giọng hát và âm nhạc và lưu thành công!")
