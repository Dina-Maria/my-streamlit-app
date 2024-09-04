import streamlit as st
from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image
import time



def read_allowed_vehicles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        allowed_vehicles = file.read().splitlines()
    return allowed_vehicles

# Загрузка моделей
model_pl_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8s_Plates\best.pt"
model_sym_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8s_symbols\best.pt"
model_sp_tr_path = r"c:\Users\alex9\Desktop\проект КСК ИТ-2\yolov8_Spectransport\best.pt"

# Проверка существования файлов
if not os.path.isfile(model_pl_path):
    raise FileNotFoundError(f"'{model_pl_path}' does not exist")
if not os.path.isfile(model_sym_path):
    raise FileNotFoundError(f"'{model_sym_path}' does not exist")
if not os.path.isfile(model_sp_tr_path):
    raise FileNotFoundError(f"'{model_sp_tr_path}' does not exist")

# Загрузка моделей
model_pl = YOLO(model_pl_path)
model_sym = YOLO(model_sym_path)
model_sp_tr = YOLO(model_sp_tr_path)

print("Модели загружены успешно")

letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T',
           'X', 'Y', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
# Укажите путь к вашему текстовому файлу
allowed_vehicles_file = r"C:\Users\alex9\Desktop\проект КСК ИТ-2\allowed_vehicles.txt"
allowed_vehicles = read_allowed_vehicles(
    allowed_vehicles_file)  # Чтение номеров из файла


def adjust_contrast_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean, std = np.mean(gray), np.std(gray)

    target_mean, target_std = 200, 70

    alpha = target_std / std
    beta = target_mean - mean * alpha

    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    return adjusted_image


def detect_objects(image):
    img = np.array(image)

    # Распознавание типа транспортного средства
    results_sp_tr = model_sp_tr(img)
    classes = results_sp_tr.pred[0].numpy()[:, -1]

    if "другие авто" not in classes:  # Предполагается, что "другие авто" - это класс для общего транспорта
        st.write(
            "Автомобиль не относится к категории 'другие авто'. Ожидание специального транспорта.")
        return

    # Распознавание номерного знака если это "другие авто"
    adjusted_image = adjust_contrast_brightness(img)
    results_np = model_pl(adjusted_image)

    if results_np[0].boxes.shape[0] == 0:
        st.write("Номерной знак не найден.")
        return

    for box in results_np[0].boxes:
        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
        plate = img[int(ymin):int(ymax), int(xmin):int(xmax)]

        # Распознавание символов на номерном знаке
        plate_img = adjust_contrast_brightness(plate)
        results_char = model_sym(plate_img)

        recognized_chars = []
        for char_box in results_char[0].boxes:
            char_xmin, char_ymin, char_xmax, char_ymax = char_box.xyxy[0].tolist(
            )
            char = plate[int(char_ymin):int(char_ymax),
                         int(char_xmin):int(char_xmax)]
            char_res = model_sym(char)
            char_class = np.argmax(char_res[0].probs.numpy())
            recognized_chars.append(letters[char_class])

        recognized_plate = ''.join(recognized_chars)
        st.write(f"Распознанный номерной знак: {recognized_plate}")

        # Проверка на разрешение въезда
        if recognized_plate in allowed_vehicles:
            st.write("Въезд разрешен.")
        else:
            st.write("Въезд запрещен.")


def main():
    st.title("Распознавание номерных знаков и типов транспорта")
    uploaded_file = st.file_uploader(
        "Загрузите изображение", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Загруженное изображение",
                 use_column_width=True)
        if st.button("Распознать"):
            detect_objects(image)


if __name__ == "__main__":
    main()
