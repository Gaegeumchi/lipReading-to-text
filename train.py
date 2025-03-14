import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# 🔹 데이터 로드
DATASET_PATH = "dataset/processed/"
lip_features = np.load(DATASET_PATH + "lip_features.npy")
with open(DATASET_PATH + "labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f]

# 🔹 텍스트 라벨 인코딩 (문장을 정수로 변환)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# 🔹 PyTorch 데이터셋 클래스 정의
class LipReadingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 🔹 데이터셋 & 데이터로더 생성
dataset = LipReadingDataset(lip_features, labels_encoded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 🔹 LSTM 기반 모델 정의
class LipReadingModel(nn.Module):
    def __init__(self, input_dim=42, hidden_dim=128, output_dim=100, num_layers=2):
        super(LipReadingModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return self.softmax(out)

# 🔹 모델 초기화 및 학습 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LipReadingModel(input_dim=42, output_dim=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 모델 학습 루프
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

# 🔹 모델 저장
torch.save(model.state_dict(), "lip_reading_model.pth")
print("✅ 학습 완료! 모델 저장됨: lip_reading_model.pth")
