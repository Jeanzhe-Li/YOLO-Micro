# 随即剪裁与亮度调整
# YOLO自带数据增强，训练过程中并没有使用此增强代码

import os
import random
import traceback
from PIL import Image, ImageEnhance
from datetime import datetime

# ====================== 全局配置 ======================
IMG_DIR = "../yolo_dataset/train/images"
LABEL_DIR = "../yolo_dataset/train/labels"
LOG_DIR = "../log"
AUG_PROB = 0.5  # 数据增强概率
MIN_CROP_RATIO = 0.5  # 最小裁剪比例
BRIGHTNESS_RANGES = {  # 亮度调整范围
    'reduce': (0.5, 0.9),
    'increase': (1.1, 1.5)
}


def augmentation():
    processed_count = 0
    success_count = 0
    error_log = []

    time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_name = time + ".txt"
    log = None
    if os.path.exists(LOG_DIR):
        log = open(os.path.join(LOG_DIR, log_name), 'a')
        log.write(time + "：\n")

    # 遍历图片目录
    for img_name in os.listdir(IMG_DIR):
        try:
            # 跳过已增强的文件
            if '_2.' in img_name:
                continue

            img_path = os.path.join(IMG_DIR, img_name)

            # 检查文件格式
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # 数据增强概率判断
            if random.random() > AUG_PROB:
                continue

            processed_count += 1
            base_name = os.path.splitext(img_name)[0]
            new_img_name = f"{base_name}_2{os.path.splitext(img_name)[1]}"
            new_label_name = f"{base_name}_2.txt"

            # 打开原始图片（添加异常处理）
            try:
                img = Image.open(img_path)
                original_mode = img.mode
                # 统一转换为可处理模式
                if original_mode not in ['L', 'RGB', 'RGBA']:
                    img = img.convert('RGB')
                    original_mode = 'RGB'
            except Exception as e:
                error_log.append(f"打开图片失败 [{img_name}]: {str(e)}")
                continue

            W, H = img.size
            print(f"\n处理图片：{img_name}")
            if log:
                log.write(f"\n处理图片：{img_name}\n")

            # 选择增强类型
            if random.random() < 0.5:
                # ========== 亮度调整处理 ==========
                # 随机选择调整方向
                direction = random.choice(['reduce', 'increase'])
                min_val, max_val = BRIGHTNESS_RANGES[direction]
                factor = round(random.uniform(min_val, max_val), 2)

                # 执行亮度调整
                try:
                    enhancer = ImageEnhance.Brightness(img)
                    enhanced_img = enhancer.enhance(factor)
                    if original_mode == 'L':  # 灰度模式
                        enhanced_img = enhanced_img.convert('L')

                    enhanced_img.save(os.path.join(IMG_DIR, new_img_name))
                except Exception as e:
                    error_log.append(f"亮度调整失败 [{img_name}]: {str(e)}")
                    continue

                # 输出调整信息
                print(f"└── 操作类型：亮度调整")
                print(f"    ├── 调整方向：{'亮度减弱' if direction == 'reduce' else '亮度增强'}")
                print(f"    └── 调整系数：{factor:.2f}")
                if log:
                    log.write(f"└── 操作类型：亮度调整\n")
                    log.write(f"    ├── 调整方向：{'亮度减弱' if direction == 'reduce' else '亮度增强'}\n")
                    log.write(f"    └── 调整系数：{factor:.2f}\n")

                # 处理标签文件
                orig_label = os.path.join(LABEL_DIR, f"{base_name}.txt")
                new_label = os.path.join(LABEL_DIR, new_label_name)
                try:
                    if os.path.exists(orig_label):
                        with open(orig_label, 'r') as f_in, open(new_label, 'w') as f_out:
                            f_out.write(f_in.read())
                except Exception as e:
                    error_log.append(f"标签处理失败 [{img_name}]: {str(e)}")
                    os.remove(os.path.join(IMG_DIR, new_img_name))  # 回滚生成的图片
                    continue

            else:
                # ========== 随机裁剪处理 ==========
                # 计算裁剪尺寸
                crop_w = int(W * random.uniform(MIN_CROP_RATIO, 1.0))
                crop_h = int(H * random.uniform(MIN_CROP_RATIO, 1.0))

                # 计算有效裁剪区域
                max_left = max(0, W - crop_w)
                max_top = max(0, H - crop_h)
                left = random.randint(0, max_left)
                top = random.randint(0, max_top)

                try:
                    # 执行裁剪并保存
                    cropped_img = img.crop((left, top, left + crop_w, top + crop_h))
                    if original_mode == 'L':
                        cropped_img = cropped_img.convert('L')
                    cropped_img.save(os.path.join(IMG_DIR, new_img_name))
                except Exception as e:
                    error_log.append(f"图片裁剪失败 [{img_name}]: {str(e)}")
                    continue

                # 输出裁剪信息
                print(f"└── 操作类型：随机裁剪")
                print(f"    ├── 原图尺寸：{W}x{H}")
                print(f"    ├── 裁剪区域：X[{left}-{left + crop_w}] Y[{top}-{top + crop_h}]")
                print(f"    └── 新图尺寸：{crop_w}x{crop_h}")
                if log:
                    log.write(f"└── 操作类型：随机裁剪\n")
                    log.write(f"    ├── 原图尺寸：{W}x{H}\n")
                    log.write(f"    ├── 裁剪区域：X[{left}-{left + crop_w}] Y[{top}-{top + crop_h}]\n")
                    log.write(f"    └── 新图尺寸：{crop_w}x{crop_h}\n")

                # 处理标签文件
                new_labels = []
                orig_label = os.path.join(LABEL_DIR, f"{base_name}.txt")
                try:
                    if os.path.exists(orig_label):
                        with open(orig_label, 'r') as f:
                            for line_num, line in enumerate(f, 1):
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    error_log.append(f"标签格式非法 [{img_name} line {line_num}]")
                                    continue

                                try:
                                    cls_id, x_center, y_center, width, height = map(float, parts)
                                except ValueError:
                                    error_log.append(f"数值转换失败 [{img_name} line {line_num}]")
                                    continue

                                # 坐标转换
                                abs_x = x_center * W
                                abs_y = y_center * H

                                # 中心点检查
                                if left <= abs_x <= left + crop_w and top <= abs_y <= top + crop_h:
                                    new_x = (abs_x - left) / crop_w
                                    new_y = (abs_y - top) / crop_h
                                    new_w = (width * W) / crop_w
                                    new_h = (height * H) / crop_h

                                    new_labels.append(
                                        f"{int(cls_id)} {new_x:.6f} {new_y:.6f} {new_w:.6f} {new_h:.6f}\n")

                    # 写入新标签
                    with open(os.path.join(LABEL_DIR, new_label_name), 'w') as f:
                        f.writelines(new_labels)
                except Exception as e:
                    error_log.append(f"标签处理失败 [{img_name}]: {str(e)}")
                    os.remove(os.path.join(IMG_DIR, new_img_name))  # 回滚生成的图片
                    continue

            success_count += 1

        except Exception as e:
            error_log.append(f"未知错误 [{img_name}]: {str(e)}\n{traceback.format_exc()}")
            continue

    # 输出统计信息
    print("\n" + "=" * 40)
    print(f"处理完成！共处理 {processed_count} 张图片")
    print(f"成功处理：{success_count} 张")
    print(f"失败处理：{len(error_log)} 次")
    if log:
        log.write("\n" + "=" * 40 + "\n")
        log.write(f"处理完成！共处理 {processed_count} 张图片\n")
        log.write(f"成功处理：{success_count} 张\n")
        log.write(f"失败处理：{len(error_log)} 次\n")

    # 输出错误日志
    if error_log:
        print("\n错误日志：")
        for i, error in enumerate(error_log[:5], 1):
            print(f"{i}. {error}")
        if len(error_log) > 5:
            print(f"（仅显示前5条错误，共 {len(error_log)} 条错误）")

        if log:
            log.write("\n错误日志：")
            for i, error in enumerate(error_log[:], 1):
                log.write(f"{i}. {error}")

    if log:
        log.close()

if __name__ == "__main__":
    augmentation()