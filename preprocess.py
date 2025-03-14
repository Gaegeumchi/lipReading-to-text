import cv2
import mediapipe as mp
import json
import numpy as np
import os
from tqdm import tqdm

# 🔹 Mediapipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    refine_landmarks=True  # 더 정밀한 랜드마크 감지
)


# 🔹 데이터 폴더 설정
VIDEO_FOLDER = "dataset/original/"  # original data location
JSON_FOLDER = "dataset/labeld/"      # labeld data location
SAVE_FOLDER = "dataset/processed/"   # processed data location

os.makedirs(SAVE_FOLDER, exist_ok=True)

lip_data = []
labels = []

json_files = [f for f in os.listdir(JSON_FOLDER) if f.endswith(".json")]

for json_file in tqdm(json_files, desc="📂 Processing JSON Files"):
    json_path = os.path.join(JSON_FOLDER, json_file)
    video_name = json_file.replace(".json", ".mp4")  # MP4와 매칭
    video_path = os.path.join(VIDEO_FOLDER, video_name)

    # load labeld data
    with open(json_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if not os.path.exists(video_path):
        print(f"❌ 영상 파일 없음: {video_path}")
        continue

    # 🔹 동영상 처리 시작
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    sentences = dataset[0]["Sentence_info"]

    print(f"🎥 Processing Video: {video_name} (FPS: {fps})")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
        frame_resized = cv2.resize(frame, (640, 640))  # 정사각형으로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for landmarks in results.multi_face_landmarks:
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
                    lip_data.append(lip_points.flatten())
                    labels.append(label)

    cap.release()

# Convert into NumPy
lip_data = np.array(lip_data)
np.save(os.path.join(SAVE_FOLDER, "lip_features.npy"), lip_data)
with open(os.path.join(SAVE_FOLDER, "labels.txt"), "w", encoding="utf-8") as f:
    for label in labels:
        f.write(label + "\n")

print("✅ Saved Data: dataset/processed/lip_features.npy, dataset/processed/labels.txt")