"""
    功能测试
"""
from Method.Distorting_Mirror import distorting_mirror_main
from Method.Style_Change import choose_style
from Method.Face_recognition_model import Change_ATA_datasets, load_data, train_ML_model, model_test
from Method.Face_recognization_method import recognize_face

if __name__ == '__main__':
    # 哈哈镜
    # distorting_mirror_main(mode='video', effect_type='min')

    # 风格转换 --复古/黑白/素描
    # retro_style_main(img_path='test_img.jpg')
    # choose_style(style_name='youhua', img_path='test_img.jpg')

    # ATA人脸数据集转换
    # Change_ATA_datasets(data_path='../Data/ATA/s', save_path='../Data/datasetFacesORL.npy')

    # 测试数据集与机器学习训练
    # X_train, X_test, y_train, y_test = load_data(data_path='../Data/datasetFacesORL.npy')
    # print(X_train.shape)

    # train_ML_model(X_train, y_train)
    # model_test(X_test, y_test)

    result = recognize_face(image_path='test_img.jpg')
    print(f'Recognition result: {result}')