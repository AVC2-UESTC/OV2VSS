from PIL import Image
import os

image_sizes = set()

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size

# 指定要遍历的文件夹路径
folder_path = '/home/lixinhao/vss/data/coco/val2017'  # 替换为实际的文件夹路径

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path):
        try:
            size = get_image_size(file_path)
            image_sizes.add(size)
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")

print(f"Total number of unique image sizes: {len(image_sizes)},{image_sizes}")
