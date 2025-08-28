#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
扩充Tong数据集
将每个文件复制5份，增加训练数据量
"""

import os
import shutil
import glob
from pathlib import Path

def expand_dataset(source_dir, target_dir, copies=5):
    """
    扩充数据集，将每个文件复制指定份数
    
    Args:
        source_dir: 源目录
        target_dir: 目标目录
        copies: 复制份数
    """
    
    # 创建目标目录
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有文件
    all_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.txt']:
        all_files.extend(glob.glob(os.path.join(source_dir, ext)))
    
    if not all_files:
        print(f"在 {source_dir} 中没有找到文件")
        return 0
    
    print(f"在 {source_dir} 中找到 {len(all_files)} 个文件")
    
    # 复制文件
    total_copied = 0
    for file_path in all_files:
        file_name = os.path.basename(file_path)
        name_without_ext = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]
        
        # 复制原文件
        target_path = os.path.join(target_dir, file_name)
        shutil.copy2(file_path, target_path)
        total_copied += 1
        
        # 复制额外份数
        for i in range(1, copies):
            new_name = f"{name_without_ext}_copy{i}{ext}"
            target_path = os.path.join(target_dir, new_name)
            shutil.copy2(file_path, target_path)
            total_copied += 1
    
    print(f"总共复制了 {total_copied} 个文件到 {target_dir}")
    return total_copied

def expand_tong_dataset():
    """扩充Tong数据集"""
    
    base_dir = "/Users/zhanghaining/2022/dataset/DataSet/tong"
    source_dataset = os.path.join(base_dir, "yolo_dataset")
    expanded_dataset = os.path.join(base_dir, "yolo_dataset_expanded")
    
    print("=== Tong数据集扩充工具 ===")
    print(f"源数据集: {source_dataset}")
    print(f"扩充后数据集: {expanded_dataset}")
    print("每个文件将复制5份")
    
    # 检查源数据集是否存在
    if not os.path.exists(source_dataset):
        print(f"错误：源数据集不存在: {source_dataset}")
        return
    
    # 扩充训练集
    print("\n1. 扩充训练集...")
    train_source = os.path.join(source_dataset, "images", "train")
    train_target = os.path.join(expanded_dataset, "images", "train")
    train_copied = expand_dataset(train_source, train_target, copies=5)
    
    # 扩充训练标签
    print("\n2. 扩充训练标签...")
    train_labels_source = os.path.join(source_dataset, "labels", "train")
    train_labels_target = os.path.join(expanded_dataset, "labels", "train")
    train_labels_copied = expand_dataset(train_labels_source, train_labels_target, copies=5)
    
    # 扩充验证集
    print("\n3. 扩充验证集...")
    val_source = os.path.join(source_dataset, "images", "val")
    val_target = os.path.join(expanded_dataset, "images", "val")
    val_copied = expand_dataset(val_source, val_target, copies=5)
    
    # 扩充验证标签
    print("\n4. 扩充验证标签...")
    val_labels_source = os.path.join(source_dataset, "labels", "val")
    val_labels_target = os.path.join(expanded_dataset, "labels", "val")
    val_labels_copied = expand_dataset(val_labels_source, val_labels_target, copies=5)
    
    # 统计结果
    print(f"\n=== 扩充完成 ===")
    print(f"训练图片: {train_copied} 个")
    print(f"训练标签: {train_labels_copied} 个")
    print(f"验证图片: {val_copied} 个")
    print(f"验证标签: {val_labels_copied} 个")
    
    # 计算数据量增长
    original_train = len(glob.glob(os.path.join(train_source, "*.jpg")))
    original_val = len(glob.glob(os.path.join(val_source, "*.jpg")))
    
    expanded_train = len(glob.glob(os.path.join(train_target, "*.jpg")))
    expanded_val = len(glob.glob(os.path.join(val_target, "*.jpg")))
    
    print(f"\n数据量增长:")
    print(f"训练集: {original_train} → {expanded_train} (增长 {expanded_train/original_train:.1f}倍)")
    print(f"验证集: {original_val} → {expanded_val} (增长 {expanded_val/original_val:.1f}倍)")
    
    # 创建新的配置文件
    create_expanded_config(expanded_dataset)
    
    print(f"\n=== 使用说明 ===")
    print(f"1. 新的数据集位置: {expanded_dataset}")
    print(f"2. 更新训练配置文件: data/tong_expanded.yaml")
    print(f"3. 使用新配置训练:")
    print(f"   python train.py --data data/tong_expanded.yaml --weights yolov5s.pt --epochs 1000 --batch-size 4 --img 320 --hyp data/hyp_tong_long_targets.yaml --freeze 0 20 --patience 200")

def create_expanded_config(expanded_dataset):
    """创建扩充后的数据集配置文件"""
    
    config_content = f"""# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
# Tong数据集配置文件 (扩充版)
# 每个文件复制5份，数据量显著增加

# 训练和验证数据集路径
train: {expanded_dataset}/images/train
val: {expanded_dataset}/images/val

# 类别数量
nc: 2

# 类别名称
names:
  0: Error
  1: Success

# 数据集描述
# 这是扩充后的Tong数据集，每个原始文件复制了5份
# 数据来源于labelme标注，已转换为YOLO格式
# 训练集和验证集按8:2比例分割，然后每个文件复制5份
# 扩充后数据量显著增加，有助于改善训练效果
"""
    
    config_file = "data/tong_expanded.yaml"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"扩充后的配置文件已创建: {config_file}")

if __name__ == "__main__":
    expand_tong_dataset()
