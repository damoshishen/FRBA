"""
    Image style change
        1.Nostalgic style
        2.Black and white outline
        3.Sketch style
"""
import cv2
import numpy as np
import random
from PIL import Image
from PIL import ImageEnhance


def retro_style(img):
    img2 = img.copy()
    height, width, n = img.shape

    for i in range(height):
        for j in range(width):
            b, g, r = img[i, j]

            # 计算新的图像中的RGB值
            B = int(0.272 * r + 0.534 * g + 0.131 * b)
            G = int(0.349 * r + 0.686 * g + 0.168 * b)
            R = int(0.393 * r + 0.769 * g + 0.189 * b)

            # 约束图像像素值，防止溢出
            img2[i, j] = [max(0, min(B, 255)), max(0, min(G, 255)), max(0, min(R, 255))]

    return img2


def resize_image(img, max_size=800):
    height, width = img.shape[:2]
    if max(height, width) > max_size:
        scaling_factor = max_size / float(max(height, width))
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def retro_style_main(img_path):
    img = cv2.imread(img_path)
    retro_img = retro_style(img)
    resized_img = resize_image(retro_img)
    cv2.imshow("Retro Style Image", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 缩放函数
def resize_with_aspect_ratio(image, max_size):
    h, w = image.shape[:2]
    if h > w:
        new_h = max_size
        new_w = int(w * (max_size / h))
    else:
        new_w = max_size
        new_h = int(h * (max_size / w))
    return cv2.resize(image, (new_w, new_h))


def edge_filter(img, max_size=800):
    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用中值滤波平滑图像
    blurred_img = cv2.medianBlur(gray_img, 7)
    # 使用拉普拉斯算子进行边缘检测
    edges = cv2.Laplacian(blurred_img, cv2.CV_8U, ksize=5)

    # 阈值处理
    _, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_TOZERO_INV)

    # 调整输出图像大小并保持长宽比
    resized_thresh = resize_with_aspect_ratio(thresh, max_size)

    # 显示边缘检测结果
    cv2.imshow("Edge Filter", resized_thresh)


def edge_filter_main(img_path, max_size=800):
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    # 应用边缘检测滤波器
    edge_filter(img, max_size)

    # 等待按键事件并关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def sketch_style(img, maxsize=800):
    height, width, _ = img.shape  # 获取图像的高度和宽度
    gray0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    # 创建一个与原图像大小相同的黑色图像
    img2 = np.zeros((height, width), dtype='uint8')

    # 使用加权和的方法进行图像反转，创建类似负片效果的图像
    gray1 = cv2.addWeighted(gray0, -1, img2, 0, 255, 0)

    # 使用高斯模糊平滑图像，调整模糊内核大小和标准差
    gray1 = cv2.GaussianBlur(gray1, (15, 15), 0)

    # 增加对比度
    gray1 = cv2.convertScaleAbs(gray1, alpha=1.5, beta=0)

    # 将原始灰度图和模糊后的负片图像混合，调整混合权重
    dst = cv2.addWeighted(gray0, 0.7, gray1, 0.3, 0)

    # 反转颜色，使之变为白底黑画
    dst = cv2.bitwise_not(dst)

    # 调整输出图像大小并保持长宽比
    dst = resize_with_aspect_ratio(dst, maxsize)

    # 显示素描效果图像
    cv2.imshow('sketch_img', dst)


def sketch_style_main(img_path, max_size=800):
    # 读取输入图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return

    # 应用素描风格处理
    sketch_style(img, max_size)

    # 等待按键事件并关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def oil_style(img):
    height, width, n = img.shape
    output = np.zeros((height, width, n), dtype='uint8')
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            rand_choice = random.randint(0, 2)
            if rand_choice == 0:
                output[i, j] = img[i + 1, j]
            elif rand_choice == 1:
                output[i, j] = img[i - 1, j]
            else:
                output[i, j] = img[i, j - 1]
    return output


def color_add(img):
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Color(image)
    image_colored = enhancer.enhance(2.0)
    return cv2.cvtColor(np.array(image_colored), cv2.COLOR_RGB2BGR)


def oil_style_main(img_path, maxsize=800):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return
    oil_img = oil_style(img)
    final_img = color_add(oil_img)

    final_img = resize_with_aspect_ratio(final_img, maxsize)

    cv2.imshow("Oil Painting Style Image", final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 风格选择器
def choose_style(style_name, img_path):
    if style_name == 'fugu':
        retro_style_main(img_path)
    elif style_name == 'heibai':
        edge_filter_main(img_path)
    elif style_name == 'sumiao':
        sketch_style_main(img_path)
    elif style_name == 'youhua':
        oil_style_main(img_path)
    else:
        print("没有该风格，请重新选择！")
