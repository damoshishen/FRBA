"""
    人脸识别方法
"""
import cv2
import joblib


def load_and_preprocess_image(image_path, height, width):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 确保图像大小与训练时一致
    image_resized = cv2.resize(image, (width, height))

    # 展开图像为一维向量
    image_flattened = image_resized.flatten()

    return image_flattened


def recognize_face(image_path, model_path='../Model/face_recognition_model.pkl'):
    # 加载训练好的模型
    model = joblib.load(model_path)

    # 假设你已经知道训练数据的高度和宽度
    height, width = 112, 92  # 请根据实际情况调整

    # 预处理新图像
    image = load_and_preprocess_image(image_path, height, width)

    # 扩展维度以匹配模型输入
    image = image.reshape(1, -1)

    # 进行预测
    y_pred = model.predict(image)

    return y_pred