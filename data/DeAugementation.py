# 取消数据增强
import os

img_dir = "./yolo_dataset/train/images"
label_dir = "./yolo_dataset/train/labels"

#删除增强的图片和对应的标注文件
def de_augmentation():
    count = 0
    for img_name in os.listdir(img_dir):
        if '_2.' in img_name:
            count += 1
            os.remove(os.path.join(img_dir,img_name))
            base_name = os.path.splitext(img_name)[0]
            label_name = f"{base_name}.txt"
            if os.path.exists(os.path.join(label_dir,label_name)):
                os.remove(os.path.join(label_dir,label_name))
            print(f"已删除图片“{img_name}”和其对应的标注文件。")
    print(f"\n删除了{count}张图片和其对应的标注文件。")

if __name__ == "__main__":
    de_augmentation()