import numpy as np
import librosa

def extract_features(audio_file):
    try:
        # Sử dụng librosa để trích xuất đặc trưng âm thanh
        y, sr = librosa.load(audio_file)
        # Trích xuất các đặc trưng MFCC (Mel-frequency cepstral coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr)
        # Trung bình các đặc trưng trên toàn bộ file âm thanh
        averaged_mfccs = np.mean(mfccs.T, axis=0)
        return averaged_mfccs
    except Exception as e:
        print(f"Error occurred while processing {audio_file}: {str(e)}")
        return None

def match_voices(audio_file_1, audio_file_2):
    # Trích xuất đặc trưng của hai file âm thanh
    features_1 = extract_features(audio_file_1)
    features_2 = extract_features(audio_file_2)
    
    if features_1 is None or features_2 is None:
        print("Unable to compare voices due to invalid audio files.")
        return
    
    # Sử dụng một phương pháp so sánh như hệ số tương quan để so sánh các đặc trưng
    correlation = np.corrcoef(features_1, features_2)[0, 1]
    
    # Thông tin về mức độ khớp giữa hai file âm thanh
    print("Correlation between the two audio files:", correlation)
    
    # Xác định ngưỡng cho mức độ khớp
    threshold = 0.9
    
    # Kiểm tra xem mức độ khớp có vượt qua ngưỡng hay không
    if correlation > threshold:
        print("The voices in the two audio files match.")
    else:
        print("The voices in the two audio files do not match.")

# Hàm phân biệt giọng nói
def identify_singer(audio_file, database):
    features = extract_features(audio_file)
    if features is None:
        print("Unable to identify singer due to invalid audio file.")
        return
    
    distances = {}
    for singer, features_db in database.items():
        # So sánh đặc trưng giữa file âm thanh đầu vào và cơ sở dữ liệu
        correlation = np.corrcoef(features, features_db)[0, 1]
        distances[singer] = correlation
    
    # In ra giá trị tương quan với từng ca sĩ trong cơ sở dữ liệu
    print("Correlation with each singer in the database:")
    for singer, correlation in distances.items():
        print(f"{singer}: {correlation}")
    
    singer_identified = max(distances, key=distances.get)
    return singer_identified

# Cơ sở dữ liệu mẫu
database = {
    "singer1": extract_features("./Music/set_fire_to_the_rain.wav"), #Adele
    "singer2": extract_features("./Music/delicate.wav"), #Taylor
    "singer3": extract_features("./Music/justin_cold_water.wav"), #Justin
    "singer4": extract_features("./Music/justin_love_yourself.wav")
}

# Đường dẫn đến file âm thanh bạn muốn xác định ca sĩ
audio_file_to_identify = "./Music/AI-Bieber_Test.wav"

# Xác định ca sĩ
singer_identified = identify_singer(audio_file_to_identify, database)
print(f"Ca sĩ được xác định: {singer_identified}")
