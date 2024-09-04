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

    uploaded_video = st.file_uploader("Загрузите видео", type=["mp4", "avi", "mov"])

    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results_pl = model_pl(frame)
            results_sym = model_sym(frame)
            results_other = model_other(frame)

            # Объединение всех результатов
            results_all = results_pl + results_sym + results_other

            for result in results_all:
                for box in result.boxes.data.cpu().numpy():
                    x1, y1, x2, y2, score, class_id = box
                    label = result.names[int(class_id)]
                    if label in allowed_vehicles:
                        color = (0, 255, 0)
                    else:
                        color = (0, 0, 255)

                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            stframe.image(frame, channels='BGR')

        cap.release()
        os.remove(tfile.name)

if __name__ == "__main__":
    main()
