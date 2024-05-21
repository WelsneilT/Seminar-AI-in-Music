import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Hàm trích xuất đặc trưng
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)


    # Lấy trung bình theo thời gian cho tất cả đặc trưng
    mfccs = np.mean(mfccs, axis=1)
    chroma = np.mean(chroma, axis=1)
    spectral_contrast = np.mean(spectral_contrast, axis=1)
    tonnetz = np.mean(tonnetz, axis=1)
    zcr = np.mean(zcr, axis=1)

    # Kết hợp tất cả đặc trưng thành một vector duy nhất, bao gồm tempo
    features = np.hstack([mfccs, chroma, spectral_contrast, tonnetz, zcr])
    return features

# Tải dữ liệu
human_dir = "./Music-Human"
ai_dir = "./Music-AI"
audio_files = []
labels = []

# Lấy tất cả các tệp từ thư mục human và gán nhãn là 0
for f in os.listdir(human_dir):
    if f.endswith('.wav'):
        audio_files.append(os.path.join(human_dir, f))
        labels.append(0)

# Lấy tất cả các tệp từ thư mục AI và gán nhãn là 1
for f in os.listdir(ai_dir):
    if f.endswith('.wav'):
        audio_files.append(os.path.join(ai_dir, f))
        labels.append(1)

# Trích xuất đặc trưng và gán nhãn
X = []
y = []

for file in audio_files:
    features = extract_features(file)
    X.append(features)

X = np.array(X)
y = np.array(labels)

# Áp dụng SMOTE để xử lý imbalance
smote = SMOTE(random_state=100)
X_res, y_res = smote.fit_resample(X, y)

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Định nghĩa các mô hình
models = {
    'RandomForest': RandomForestClassifier(random_state=100),
    'SVM': SVC(kernel='linear', probability=True, random_state=100),
    'GradientBoosting': GradientBoostingClassifier(random_state=100)
}

# Huấn luyện và đánh giá mô hình sử dụng cross-validation
for name, model in models.items():
    skf = StratifiedKFold(n_splits=5)
    accuracies = []
    
    for train_index, test_index in skf.split(X_train, y_train):
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]
        
        model.fit(X_train_fold, y_train_fold)
        y_pred = model.predict(X_test_fold)
        accuracies.append(accuracy_score(y_test_fold, y_pred))
    
    avg_accuracy = np.mean(accuracies)
    print(f'Cross-Validated Accuracy with {name}: {avg_accuracy * 100:.2f}%')

# Tuning tham số hyperparameters cho mô hình tốt nhất (ví dụ cho SVM)
param_grid = {
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf']
    }
}

# Thực hiện grid search cho SVM
grid_search = GridSearchCV(estimator=models['SVM'], param_grid=param_grid['SVM'], cv=5, n_jobs=-1, verbose=2, error_score='raise')
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Đánh giá mô hình tốt nhất trên tập kiểm tra
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Optimized Accuracy with SVM: {accuracy * 100:.2f}%')

# Đánh giá trên nhiều lần chia train-test
splits = 10
split_accuracies = []

for i in range(splits):
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_res, y_res, test_size=0.2, random_state=100 + i)
    best_model.fit(X_train_split, y_train_split)
    y_pred_split = best_model.predict(X_test_split)
    split_accuracies.append(accuracy_score(y_test_split, y_pred_split))

final_avg_accuracy = np.mean(split_accuracies)
print(f'Final Averaged Accuracy with SVM: {final_avg_accuracy * 100:.2f}%')

# Hàm dự đoán mẫu mới
def predict_new_sample(new_audio_file):
    new_feature = extract_features(new_audio_file)
    prediction = best_model.predict(new_feature)
    return "Human" if prediction[0] == 0 else "AI"

# Sử dụng hàm dự đoán mẫu mới
new_audio_file1 = "./Music/Justin-Bieber-When-i-was-your-man-AI-COVER.wav"
new_audio_file2 = "./Music/Justin Bieber - Come Around Me (Audio).mp3"
result1 = predict_new_sample(new_audio_file1)
result2 = predict_new_sample(new_audio_file2)
print(f"The voice in '{new_audio_file1}' is identified as: {result1}")
print(f"The voice in '{new_audio_file2}' is identified as: {result2}")