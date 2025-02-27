import os
import cv2
import numpy as np
import yaml
import random
from pathlib import Path

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """画一个边界框"""
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def visualize_dataset():
    # 读取yaml配置文件
    with open('/Users/zhanghaining/2022/yolov5/data/face.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    # 获取类别名称
    names = data['names']
    
    # 设置图片和标签路径
    img_path = '/Users/zhanghaining/2022/dataset/WIDERFACE/train/images'
    label_path = '/Users/zhanghaining/2022/dataset/WIDERFACE/train/labels'
    
    # 获取所有图片文件
    img_files = sorted(os.listdir(img_path))
    
    for img_file in img_files:
        if not img_file.endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        # 读取图片
        img = cv2.imread(os.path.join(img_path, img_file))
        if img is None:
            print(f"无法读取图片: {img_file}")
            continue
            
        # 获取对应的标签文件
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_file_path = os.path.join(label_path, label_file)
        
        height, width = img.shape[:2]
        
        if os.path.exists(label_file_path):
            # 读取标签文件
            with open(label_file_path, 'r') as f:
                labels = f.readlines()
                
            # 绘制每个标注框
            for label in labels:
                label = label.strip().split()
                if len(label) == 5:  # class, x_center, y_center, width, height
                    cls_id = int(label[0])
                    # 将相对坐标转换为绝对坐标
                    x_center, y_center = float(label[1]) * width, float(label[2]) * height
                    w, h = float(label[3]) * width, float(label[4]) * height
                    
                    # 计算边界框的左上角和右下角坐标
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    # 绘制边界框和类别名称
                    plot_one_box([x1, y1, x2, y2], img, label=names[cls_id], color=(0, 255, 0))
        
        # 显示图片
        cv2.imshow('Dataset Review', img)
        key = cv2.waitKey(0)
        
        # 按'q'退出，按其他键继续
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize_dataset()
