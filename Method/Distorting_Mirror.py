"""
    Distorting mirror V0.9
"""

import cv2
import math


def MaxFrame(frame):
    height, width, n = frame.shape
    center_x, center_y = width / 2, height / 2
    radius = 400
    real_radius = int(radius / 2.0)
    new_data = frame.copy()

    for i in range(width):
        for j in range(height):
            tx = i - center_x
            ty = j - center_y

            distance = tx * tx + ty * ty

            if distance < radius * radius:
                new_x = int(tx / 2.0)
                new_y = int(ty / 2.0)

                new_x = int(new_x * (math.sqrt(distance) / real_radius))
                new_y = int(new_y * (math.sqrt(distance) / real_radius))

                new_x = int(new_x + center_x)
                new_y = int(new_y + center_y)

                if new_x < width and new_y < height:
                    new_data[j, i][0] = frame[new_y, new_x][0]
                    new_data[j, i][1] = frame[new_y, new_x][1]
                    new_data[j, i][2] = frame[new_y, new_x][2]

    return new_data


def MinFrame(frame, compress=8):
    height, width, n = frame.shape
    center_x, center_y = width / 2, height / 2
    radius = 400
    radius = int(radius / 2.0)

    new_data = frame.copy()

    for i in range(width):
        for j in range(height):
            tx = i - center_x
            ty = j - center_y

            theta = math.atan2(ty, tx)
            radius = math.sqrt((tx, tx) + (ty * ty))

            new_x = int(center_x + (math.sqrt(radius) * compress * math.cos(theta)))
            new_y = int(center_y + (math.sqrt(radius) * compress * math.sin(theta)))

            if 0 > new_x > width:
                new_x = 0
            if 0 > new_y > height:
                new_y = 0

            if new_x < width and new_y < height:
                new_data[j, i][0] = frame[new_y, new_x][0]
                new_data[j, i][1] = frame[new_y, new_x][1]
                new_data[j, i][2] = frame[new_y, new_x][2]

    return new_data


def apply_mirror_effect(frame, effect_type="max"):
    if effect_type == "max":
        return MaxFrame(frame)
    elif effect_type == "min":
        return MinFrame(frame)
    else:
        raise ValueError("Unknown effect type: choose 'max' or 'min'.")


# 图片
def process_image(image_path, effect_type="max"):
    frame = cv2.imread(image_path)
    if frame is not None:
        frame = apply_mirror_effect(frame, effect_type)
        cv2.imshow('HaHa Mirror', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to read image.")


# 视频
def process_video(effect_type="max"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = apply_mirror_effect(frame, effect_type)
        cv2.imshow('HaHa Mirror', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# 哈哈镜主函数
def distorting_mirror_main(mode="video", image_path=None, effect_type="max"):
    if mode == "video":
        process_video(effect_type)
    elif mode == "image" and image_path is not None:
        process_image(image_path, effect_type)
    else:
        print("Invalid mode or missing image path.")