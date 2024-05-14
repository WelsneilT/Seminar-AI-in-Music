import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Hàm trích xuất đặc trưng MFCC
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # Tăng số lượng MFCC
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Lấy trung bình theo thời gian cho tất cả đặc trưng
    mfccs = np.mean(mfccs, axis=1)
    chroma = np.mean(chroma, axis=1)
    spectral_contrast = np.mean(spectral_contrast, axis=1)

    # Kết hợp tất cả đặc trưng thành một vector duy nhất
    features = np.hstack([mfccs, chroma, spectral_contrast])
    return features

# Chuẩn bị dữ liệu
def prepare_dataset(audio_files, labels):
    features = []
    for file in audio_files:
        mfcc = extract_features(file)
        if mfcc is not None:
            features.append(mfcc)
    return np.array(features), np.array(labels)

# Danh sách các file âm thanh và nhãn tương ứng (0: Human, 1: AI)
audio_files = ["./Music-Human/Justin-Bieber-Love-Yourself-_PURPOSE-The-Movement_.wav", 
               "./Music-Human/Justin-Bieber-Off-My-Face-_Visualizer_.wav" ,
               "./Music-Human/Justin-Bieber-Ghost.wav" ,
               "./Music-AI/That-Should-Be-Me-Justin-Bieber-Older-_AI-Cover_.wav", 
               "./Music-AI/Someone-You-Loved-Justin-Bieber-_AI-Cover_.wav"]  
labels = [0, 0, 0, 1, 1 ]  # 0 for human, 1 for AI

# Xử lý dữ liệu
features, labels = prepare_dataset(audio_files, labels)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Tạo và huấn luyện mô hình SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Đánh giá mô hình
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Dự đoán một mẫu mới
def predict_new_sample(new_audio_file):
    new_feature = extract_features(new_audio_file)
    new_feature = scaler.transform([new_feature])  # Lưu ý: scaler yêu cầu input là 2D array
    prediction = model.predict(new_feature)
    return "Human" if prediction[0] == 0 else "AI"

# Kiểm tra một file mới
new_audio_file1 = "./Music-AI/Easy-on-me-Justin-Bieber-_-Ai-cover-_-lyrics-_Adele_.wav"
new_audio_file2 = "./Music-AI/Nothing_s-Gonna-Change-My-Love-For-You-Lyrics-Justin-Bieber-_AI-Cover_.wav"
result = predict_new_sample(new_audio_file1)
print(f"The voice in '{new_audio_file1}' is identified as: {result}")
