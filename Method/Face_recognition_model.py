"""
    This file stores different model training methods.
"""

import numpy as np
import joblib
import os
import face_recognition
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


"""
    GET DATA
"""


def Change_ATA_datasets(data_path, save_path):
    # '../Data/ATA/s'
    datasetFaces = []

    for person in range(1, 41):
        temp = []

        for pose in range(1, 11):
            data = plt.imread(data_path + str(person) + '/' + str(pose) + '.pgm')
            temp.append(data)

        datasetFaces.append(np.array(temp))

    datasetFaces = np.array(datasetFaces)
    np.save(save_path, datasetFaces)

    print('Total number of datasets:', len(datasetFaces))
    print('Dataset size:', datasetFaces.shape)


def load_data(data_path):
    data = np.load(data_path)

    num_people, num_images, height, width = data.shape

    data = data.reshape(num_people * num_images, height, width)

    labels = np.repeat(np.arange(num_people), num_images)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


"""
    PCA+SVM
"""


def train_ML_model(X_train, y_train):
    height, width = X_train.shape[1], X_train.shape[2]
    X_train_new = X_train.reshape(X_train.shape[0], height * width)

    # 构建PCA和SVM管道
    pca = PCA(n_components=150, whiten=True, random_state=42)
    svm = SVC(kernel='rbf', class_weight='balanced')

    model = make_pipeline(StandardScaler(), pca, svm)

    # 训练模型
    model.fit(X_train_new, y_train)

    joblib.dump(model, 'face_recognition_model.pkl')
    print("训练结束！")


"""
    Test
"""


def model_test(X_test, y_test, model_name='svm', model_path='../Model/face_recognition_model.pkl'):
    if model_name == 'svm':
        height, width = X_test.shape[1], X_test.shape[2]
        X_test = X_test.reshape(X_test.shape[0], height * width)

        model = joblib.load(model_path)

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f'Test accuracy: {accuracy:.3f}')
    else:
        print('该模型不存在！')
