import cv2
import mediapipe as mp
import json
import numpy as np
import os
from tqdm import tqdm

# 🔹 Mediapipe 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 🔹 데이터 폴더 경로
DATASET_FOLDER = "dataset/"  # 여기에 MP4 + JSON 파일이 있음

# 🔹 입술 좌표 및 라벨을 저장할 리스트
lip_data = []
labels = []

# 🔹 모든 JSON 파일 탐색
json_files = [f for f in os.listdir(DATASET_FOLDER) if f.endswith(".json")]

for json_file in tqdm(json_files, desc="📂 Processing JSON Files"):
    json_path = os.path.join(DATASET_FOLDER, json_file)

    # 🔹 JSON 파일 로드
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # 🔹 MP4 영상 파일명 가져오기
    video_name = dataset[0]["Video_info"]["video_Name"]
    video_path = os.path.join(DATASET_FOLDER, video_name)

    # 🔹 MP4 파일 존재 확인
    if not os.path.exists(video_path):
        print(f"❌ 파일 없음: {video_path}")
        continue

    # 🔹 동영상 처리 시작
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # FPS 가져오기
    sentences = dataset[0]["Sentence_info"]

    print(f"🎥 Processing Video: {video_name} (FPS: {fps})")

    # 🔹 프레임별 입술 좌표 추출
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 초 단위 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
                # 입술 좌표만 추출 (468개 중 입술 관련 랜드마크 인덱스)
                lip_indices = [
                    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308
                ]
                lip_points = np.array([(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in lip_indices])

                # 🔹 현재 프레임이 해당 문장 범위 내에 있는지 확인
                label = None
                for sentence in sentences:
                    if sentence["start_time"] <= frame_time <= sentence["end_time"]:
                        label = sentence["sentence_text"]
                        break

                if label:
                    lip_data.append(lip_points.flatten())  # 벡터화
                    labels.append(label)

    cap.release()

# 🔹 NumPy로 변환 및 저장
lip_data = np.array(lip_data)
np.save("lip_features.npy", lip_data)
with open("labels.txt", "w", encoding="utf-8") as f:
    for label in labels:
        f.write(label + "\n")

print("✅ 모든 데이터 저장 완료: lip_features.npy, labels.txt")