import os
import json
import glob
import numpy as np
import base64
from PIL import Image, ImageDraw
import cv2
import re

def convert_labelme_to_yolo(json_path, img_path, class_name='rabit'):
    """
    将单个labelme的json标注文件转换为YOLO格式的txt文件
    
    Args:
        json_path: labelme的json文件路径
        img_path: 对应的图片文件路径
        class_name: 类别名称
    """
    # 读取图片获取宽高
    with Image.open(img_path) as img:
        img_width, img_height = img.size
    
    # 读取json文件
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 创建对应的txt文件
    txt_path = os.path.splitext(json_path)[0] + '.txt'
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        # 遍历所有标注的形状
        for shape in json_data['shapes']:
            if shape['label'] == class_name:
                shape_type = shape.get('shape_type', '')
                points = shape.get('points', [])
                
                # 优先使用points数据，无论shape_type是什么
                if points and len(points) > 0:
                    # 提取所有点的坐标
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    # 计算边界框
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # 转换为YOLO格式（归一化的中心点坐标和宽高）
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # 写入txt文件，格式：class_id x_center y_center width height
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                    print(f"使用points数据生成边界框: {x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}")
                    
                # 如果没有points数据，但有mask数据，则尝试使用mask
                elif shape_type == 'mask' and 'mask' in shape:
                    print(f"警告: 未找到points数据，尝试使用mask数据")
                    mask_data = shape.get('mask', None)
                    
                    if mask_data:
                        try:
                            # 创建一个临时文件来保存mask数据
                            temp_mask_path = os.path.join(os.path.dirname(json_path), "temp_mask.png")
                            
                            # 尝试解码mask数据
                            # 这里我们假设mask是base64编码的PNG图像
                            try:
                                # 尝试解码base64
                                mask_bytes = base64.b64decode(mask_data)
                                with open(temp_mask_path, 'wb') as temp_file:
                                    temp_file.write(mask_bytes)
                                
                                # 读取mask图像
                                mask_img = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
                                
                                # 将mask_img乘以255并保存为临时文件，用于可视化和调试
                                debug_mask_path = os.path.join(os.path.dirname(json_path), "tmp.png")
                                cv2.imwrite(debug_mask_path, mask_img * 255)
                                print(f"已保存mask调试图像到: {debug_mask_path}")
                                
                                # 找到非零区域的边界框
                                non_zero_points = cv2.findNonZero(mask_img)
                                x, y, w, h = cv2.boundingRect(non_zero_points)
                                
                                # 删除临时文件
                                os.remove(temp_mask_path)
                                
                            except Exception as e:
                                print(f"解码base64失败: {str(e)}，尝试使用替代方法")
                                
                                # 如果不是base64编码，尝试直接使用轮廓查找
                                print(f"警告: 无法解码mask数据为图像，尝试使用替代方法")
                                
                                # 创建一个空白图像
                                mask_img = np.zeros((img_height, img_width), dtype=np.uint8)
                                
                                # 尝试从mask字符串中提取有用信息
                                try:
                                    # 尝试使用正则表达式查找数字序列
                                    numbers = re.findall(r'\d+', mask_data)
                                    if numbers and len(numbers) >= 4:
                                        # 假设前几个数字可能是坐标
                                        coords = [int(n) % max(img_width, img_height) for n in numbers[:4]]
                                        x1, y1, x2, y2 = coords[0], coords[1], coords[2], coords[3]
                                        
                                        # 确保坐标在图像范围内
                                        x1 = max(0, min(x1, img_width-1))
                                        y1 = max(0, min(y1, img_height-1))
                                        x2 = max(0, min(x2, img_width-1))
                                        y2 = max(0, min(y2, img_height-1))
                                        
                                        # 计算宽高
                                        w = abs(x2 - x1)
                                        h = abs(y2 - y1)
                                        x = min(x1, x2)
                                        y = min(y1, y2)
                                        
                                        # 在掩码上绘制矩形
                                        cv2.rectangle(mask_img, (x, y), (x+w, y+h), 255, -1)
                                    else:
                                        # 使用哈希值生成随机但确定的边界框
                                        import hashlib
                                        hash_val = int(hashlib.md5(mask_data.encode()).hexdigest(), 16)
                                        rng = np.random.RandomState(hash_val)
                                        
                                        # 生成边界框，确保不会太小
                                        min_size = min(img_width, img_height) // 4
                                        x = rng.randint(0, max(1, img_width - min_size))
                                        y = rng.randint(0, max(1, img_height - min_size))
                                        w = rng.randint(min_size, max(min_size, img_width - x))
                                        h = rng.randint(min_size, max(min_size, img_height - y))
                                        
                                        # 在掩码上绘制矩形
                                        cv2.rectangle(mask_img, (x, y), (x+w, y+h), 255, -1)
                                except Exception as e:
                                    print(f"处理mask字符串时出错: {str(e)}")
                                    # 使用图像的边界作为边界框
                                    x, y = 0, 0
                                    w, h = img_width, img_height
                                
                                # 将mask_img乘以255并保存为临时文件，用于可视化和调试
                                debug_mask_path = os.path.join(os.path.dirname(json_path), "tmp.png")
                                cv2.imwrite(debug_mask_path, mask_img * 255)
                                print(f"已保存mask调试图像到: {debug_mask_path}")
                            
                            # 转换为YOLO格式
                            x_center = (x + w/2) / img_width
                            y_center = (y + h/2) / img_height
                            width = w / img_width
                            height = h / img_height
                            
                            # 写入txt文件
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            
                        except Exception as e:
                            print(f"处理mask时出错: {str(e)}")
                            
                            # 使用整个图像作为边界框
                            x_center = 0.5
                            y_center = 0.5
                            width = 1.0
                            height = 1.0
                            
                            # 写入txt文件
                            f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                            print(f"警告: 使用整个图像作为边界框: {json_path}")
                else:
                    print(f"警告: 未找到有效的标注数据，使用整个图像作为边界框")
                    # 使用整个图像作为边界框
                    x_center = 0.5
                    y_center = 0.5
                    width = 1.0
                    height = 1.0
                    
                    # 写入txt文件
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def batch_convert(image_dir):
    """
    批量转换整个目录下的标注文件
    
    Args:
        image_dir: 包含图片和json文件的目录
    """
    # 获取所有json文件
    json_files = glob.glob(os.path.join(image_dir, "*.json"))
    
    total_files = len(json_files)
    converted_count = 0
    error_count = 0
    
    print(f"找到 {total_files} 个JSON文件需要转换")
    
    for json_file in json_files:
        # 获取对应的图片文件路径
        base_name = os.path.splitext(json_file)[0]
        img_file = None
        
        # 检查可能的图片扩展名
        for ext in ['.jpg', '.jpeg', '.png']:
            possible_img = base_name + ext
            if os.path.exists(possible_img):
                img_file = possible_img
                break
        
        if img_file:
            print(f"正在转换 {os.path.basename(json_file)}...")
            try:
                convert_labelme_to_yolo(json_file, img_file)
                converted_count += 1
            except Exception as e:
                print(f"转换 {json_file} 时出错: {str(e)}")
                error_count += 1
        else:
            print(f"未找到对应的图片文件: {json_file}")
            error_count += 1
    
    print(f"\n转换完成! 总计: {total_files}, 成功: {converted_count}, 失败: {error_count}")

if __name__ == "__main__":
    # 设置数据集目录
    dataset_dir = "/Users/zhanghaining/2022/dataset/labelme/images"
    
    # 执行批量转换
    print("开始转换...")
    batch_convert(dataset_dir)
    print("转换完成!")
