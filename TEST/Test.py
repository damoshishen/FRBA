"""
    功能测试
"""
from Method.Distorting_Mirror import distorting_mirror_main
from Method.Style_Change import choose_style

if __name__ == '__main__':
    # 哈哈镜
    distorting_mirror_main(mode='video', effect_type='min')

    # 风格转换 --复古/黑白/素描
    # retro_style_main(img_path='test_img.jpg')
    # choose_style(style_name='sumiao', img_path='test_img.jpg')
