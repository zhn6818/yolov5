#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将labelme标注格式转换为YOLOv5训练格式
支持Error和Success两类标签
"""

import os
import json
import shutil
from pathlib import Path
import argparse

def convert_labelme_to_yolo(labelme_dir, output_dir, class_names):
    """
    将labelme格式转换为YOLOv5格式
    
    Args:
        labelme_dir: labelme数据目录
        output_dir: 输出目录
        class_names: 类别名称列表
    """
    
    # 创建输出目录结构
    output_path = Path(output_dir)
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    
    # 创建训练和验证目录
    for split in ["train", "val"]:
        (images_dir / split).mkdir(parents=True, exist_ok=True)
        (labels_dir / split).mkdir(parents=True, exist_ok=True)
    
    # 获取所有JSON文件
    json_files = list(Path(labelme_dir).glob("*.json"))
    
    # 按8:2比例分割训练集和验证集
    train_count = int(len(json_files) * 0.8)
    train_files = json_files[:train_count]
    val_files = json_files[train_count:]
    
    print(f"总数据量: {len(json_files)}")
    print(f"训练集: {len(train_files)}")
    print(f"验证集: {len(val_files)}")
    
    # 处理训练集
    for json_file in train_files:
        process_single_file(json_file, images_dir / "train", labels_dir / "train", class_names)
    
    # 处理验证集
    for json_file in val_files:
        process_single_file(json_file, images_dir / "val", labels_dir / "val", class_names)
    
    print("转换完成！")
    print(f"输出目录: {output_dir}")

def process_single_file(json_file, images_output_dir, labels_output_dir, class_names):
    """
    处理单个JSON文件
    
    Args:
        json_file: JSON文件路径
        images_output_dir: 图片输出目录
        labels_output_dir: 标签输出目录
        class_names: 类别名称列表
    """
    
    # 读取JSON文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取图片文件名
    image_filename = data['imagePath']
    image_path = json_file.parent / image_filename
    
    if not image_path.exists():
        print(f"警告: 图片文件不存在: {image_path}")
        return
    
    # 复制图片文件
    shutil.copy2(image_path, images_output_dir / image_filename)
    
    # 创建YOLO格式的标签文件
    label_filename = image_filename.rsplit('.', 1)[0] + '.txt'
    label_path = labels_output_dir / label_filename
    
    # 获取图片尺寸
    img_height = data['imageHeight']
    img_width = data['imageWidth']
    
    # 转换标注格式
    yolo_labels = []
    for shape in data['shapes']:
        label = shape['label']
        points = shape['points']
        
        # 检查标签是否在类别列表中
        if label not in class_names:
            print(f"警告: 未知标签 '{label}' 在文件 {json_file}")
            continue
        
        # 获取类别索引
        class_id = class_names.index(label)
        
        # 计算边界框坐标
        x1, y1 = points[0]
        x2, y2 = points[1]
        
        # 转换为YOLO格式 (x_center, y_center, width, height) 归一化到0-1
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = abs(x2 - x1) / img_width
        height = abs(y2 - y1) / img_height
        
        # 确保坐标在0-1范围内
        x_center = max(0, min(1, x_center))
        y_center = max(0, min(1, y_center))
        width = max(0, min(1, width))
        height = max(0, min(1, height))
        
        yolo_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # 写入标签文件
    with open(label_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(yolo_labels))

def main():
    parser = argparse.ArgumentParser(description='将labelme格式转换为YOLOv5格式')
    parser.add_argument('--input', type=str, required=True, help='labelme数据目录')
    parser.add_argument('--output', type=str, required=True, help='输出目录')
    parser.add_argument('--classes', nargs='+', default=['Error', 'Success'], help='类别名称列表')
    
    args = parser.parse_args()
    
    # 如果没有指定参数，使用默认值
    if args.input == 'default':
        args.input = '/Users/zhanghaining/2022/dataset/DataSet/tong/data'
    if args.output == 'default':
        args.output = '/Users/zhanghaining/2022/dataset/DataSet/tong/yolo_dataset'
    
    print(f"输入目录: {args.input}")
    print(f"输出目录: {args.output}")
    print(f"类别: {args.classes}")
    
    convert_labelme_to_yolo(args.input, args.output, args.classes)

if __name__ == "__main__":
    main()
