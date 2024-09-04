import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import numpy as np

def read_allowed_vehicles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        allowed_vehicles = file.read().splitlines()
    return allowed_vehicles

def main():
    model_pl_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8s_Plates\best.pt"
    model_sym_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8s_symbols\best.pt"
    model_other_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8_Spectransport\best.pt"

    # Создание объектов моделей
    model_pl = YOLO(model_pl_path)
    model_sym = YOLO(model_sym_path)
    model_other = YOLO(model_other_path)

    allowed_vehicles_file = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\allowed_vehicles.txt"
    allowed_vehicles = read_allowed_vehicles(allowed_vehicles_file)

    st.title("Распознавание объектов на видео")

    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "avi"])

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results_other = model_other(frame)
            result = results_other[0]

            if result.boxes is not None:  # Проверяем наличие обнаруженных объектов
                for box in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, score, class_id = map(int, box)
                    cls = model_other.names[class_id]
                    label = f"{cls}: {score:.2f}"

                    if cls in allowed_vehicles:
                        color = (0, 255, 0)  # Зеленый для разрешенных транспортных средств
                    else:
                        color = (0, 0, 255)  # Красный для запрещенных

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            stframe.image(frame, channels="BGR")
        
        cap.release()
        os.remove(tfile.name)

if __name__ == "__main__":
    main()
