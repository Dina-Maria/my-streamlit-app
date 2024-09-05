import streamlit as st
import cv2
import torch
from PIL import Image
import streamlit as st
import time
import numpy as np
from ultralytics import YOLO

# Пути к моделям и файлу разрешенных номеров
model_vehicle_class_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8s_Plates\best.pt"
model_plate_number_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8s_symbols\best.pt"
model_symbols_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8_Spectransport\best.pt"
allowed_vehicles_file = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\allowed_vehicles.txt"

# Функция для чтения списка разрешенных номеров


def read_allowed_vehicles(file_path):
    with open(file_path, 'r') as file:
        allowed_vehicles = file.read().splitlines()
    return allowed_vehicles


# Загрузка моделей
model_vehicle_class = torch.hub.load(
    'ultralytics/yolov8', 'custom', path=model_vehicle_class_path)
model_plate_number = torch.hub.load(
    'ultralytics/yolov8', 'custom', path=model_plate_number_path)
model_symbols = torch.hub.load(
    'ultralytics/yolov8', 'custom', path=model_symbols_path)
allowed_vehicles = read_allowed_vehicles(allowed_vehicles_file)

# Обработка видео


def process_video(video_file):
    # Открытие видео файла
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Распознавание класса транспортного средства
        results_vehicle_class = model_vehicle_class(frame)
        class_detected = results_vehicle_class.pred[0, -1]

        # Распознавание номера
        results_plate_number = model_plate_number(frame)
        plate_number = "".join(results_plate_number.pandas().xyxy[0]['name'])

        # Распознавание символов
        results_symbols = model_symbols(frame)
        symbols_detected = results_symbols.pandas().xyxy[0]['name']

        # Проверка разрешения на въезд
        if plate_number in allowed_vehicles:
            access_status = "Въезд разрешен"
        else:
            access_status = "Въезд запрещен"

        # Отображение результатов в Streamlit
        st.image(
            frame, caption=f"Класс: {class_detected}, Номер: {plate_number}, Символы: {symbols_detected}, Статус: {access_status}")

        # Задержка для отображения кадров видео
        time.sleep(0.05)

    cap.release()


# UI Streamlit
st.title('Распознавание транспортных средств и проверка допуска')

uploaded_video = st.file_uploader(
    "Загрузите видео", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    process_video(uploaded_video.name)

st.text('Этот проект позволяет распознавать класс транспортного средства, номер и символы, а также проверять допуск по номеру.')
