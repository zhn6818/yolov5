import os
import json
import glob
import numpy as np
import base64
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_annotations(image_dir, output_dir=None, class_name='rabit'):
    """
    可视化标注数据，包括矩形框和mask
    
    Args:
        image_dir: 包含图片和json文件的目录
        output_dir: 输出可视化结果的目录，如果为None则显示而不保存
        class_name: 类别名称
    """
    # 如果指定了输出目录，确保它存在
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 获取所有json文件
    json_files = glob.glob(os.path.join(image_dir, "*.json"))
    
    total_files = len(json_files)
    print(f"找到 {total_files} 个JSON文件需要可视化")
    
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
        
        if not img_file:
            print(f"未找到对应的图片文件: {json_file}")
            continue
        
        print(f"正在可视化 {os.path.basename(json_file)}...")
        
        try:
            # 读取图片
            img = cv2.imread(img_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB
            img_height, img_width = img.shape[:2]
            
            # 读取json文件
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # 创建一个副本用于绘制
            img_with_annotations = img.copy()
            
            # 遍历所有标注的形状
            for shape in json_data['shapes']:
                if shape['label'] == class_name:
                    shape_type = shape.get('shape_type', '')
                    points = shape.get('points', [])
                    
                    # 计算边界框
                    if points and len(points) > 0:
                        # 提取所有点的坐标
                        x_coords = [int(p[0]) for p in points]
                        y_coords = [int(p[1]) for p in points]
                        
                        # 计算边界框
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # 计算宽高
                        box_width = x_max - x_min
                        box_height = y_max - y_min
                        
                        # 绘制矩形框
                        cv2.rectangle(img_with_annotations, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                        
                        # 添加标签文本
                        cv2.putText(img_with_annotations, f"{class_name}", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # 如果有mask数据，尝试可视化
                        if shape_type == 'mask' and 'mask' in shape:
                            mask_data = shape.get('mask', None)
                            
                            if mask_data:
                                try:
                                    # 创建一个临时文件来保存mask数据
                                    temp_mask_path = os.path.join(image_dir, "temp_mask.png")
                                    
                                    # 尝试解码base64
                                    mask_bytes = base64.b64decode(mask_data)
                                    with open(temp_mask_path, 'wb') as temp_file:
                                        temp_file.write(mask_bytes)
                                    
                                    # 读取mask图像
                                    mask_img = cv2.imread(temp_mask_path, cv2.IMREAD_GRAYSCALE)
                                    
                                    # 如果成功读取mask
                                    if mask_img is not None:
                                        # 调整mask大小以匹配边界框
                                        resized_mask = cv2.resize(mask_img, (box_width, box_height))
                                        
                                        # 创建一个与原图相同大小的空白mask
                                        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                                        
                                        # 确保边界框在图像范围内
                                        if x_min >= 0 and y_min >= 0 and x_max < img_width and y_max < img_height:
                                            # 将调整大小后的mask放置在边界框位置
                                            full_mask[y_min:y_max, x_min:x_max] = resized_mask
                                            
                                            # 创建红色mask覆盖
                                            red_mask = np.zeros_like(img_with_annotations)
                                            red_mask[full_mask > 0] = [255, 0, 0]  # 红色
                                            
                                            # 应用半透明效果
                                            alpha = 0.3  # 透明度
                                            mask_indices = full_mask > 0
                                            img_with_annotations[mask_indices] = cv2.addWeighted(
                                                img_with_annotations[mask_indices], 
                                                1 - alpha,
                                                red_mask[mask_indices], 
                                                alpha, 
                                                0
                                            )
                                            
                                            # 保存调整后的mask用于调试（可选）
                                            debug_mask_path = os.path.join(image_dir, "resized_mask.png")
                                            cv2.imwrite(debug_mask_path, full_mask)
                                            print(f"已保存调整后的mask到: {debug_mask_path}")
                                    
                                    # 删除临时文件
                                    if os.path.exists(temp_mask_path):
                                        os.remove(temp_mask_path)
                                    
                                except Exception as e:
                                    print(f"处理mask时出错: {str(e)}")
            
            # 显示或保存结果
            if output_dir:
                output_path = os.path.join(output_dir, f"vis_{os.path.basename(img_file)}")
                
                # 直接使用OpenCV保存图像，保持原始尺寸
                img_with_annotations_bgr = cv2.cvtColor(img_with_annotations, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, img_with_annotations_bgr)
                print(f"已保存可视化结果到: {output_path}")
            else:
                # 使用matplotlib显示图像
                plt.figure(figsize=(12, 8))
                plt.imshow(img_with_annotations)
                plt.title(f"Annotations for {os.path.basename(img_file)}")
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        
        except Exception as e:
            print(f"可视化 {json_file} 时出错: {str(e)}")

if __name__ == "__main__":
    # 设置数据集目录
    image_dir = "/Users/zhanghaining/2022/dataset/labelme/images"
    
    # 设置输出目录（如果不想保存，可以设为None）
    output_dir = "/Users/zhanghaining/2022/dataset/labelme/visualization"
    
    # 执行可视化
    print("开始可视化...")
    visualize_annotations(image_dir, output_dir)
    print("可视化完成!")
