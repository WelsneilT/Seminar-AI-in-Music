import numpy as np
import scipy.io.wavfile as wav

# Hàm tạo file âm thanh giả định
def create_fake_audio(filename, duration):
    # Tạo một chuỗi số ngẫu nhiên để biểu thị giọng hát
    voice_data = np.random.randint(low=-32767, high=32767, size=int(duration * 44100), dtype=np.int16)
    # Ghi chuỗi số vào file WAV
    wav.write(filename, 44100, voice_data)

# Tạo các file âm thanh giả định
create_fake_audio("singer1.wav", 5)  # Giọng hát của ca sĩ 1
create_fake_audio("singer2.wav", 5)  # Giọng hát của ca sĩ 2
create_fake_audio("singer3.wav", 5)  # Giọng hát của ca sĩ 3
create_fake_audio("singer4.wav", 5) 
# Cơ sở dữ liệu giọng hát
database = {
    "singer1": "singer1.wav",
    "singer2": "singer2.wav",
    "singer3": "singer3.wav"
}

# Tiếp tục với phần còn lại của mã để phân biệt giọng nói
# Định nghĩa các hàm extract_features(), compare_features(), identify_singer(), ...
