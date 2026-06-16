"""
将彩色图片转换为灰度图。

读取 /data/ori_img 下的所有彩色图片，转换为灰度图（黑白图），
保存到 /data/black_img 目录。
"""

import os
from PIL import Image
from tqdm import tqdm

# 输入/输出目录
INPUT_DIR = os.path.join("data", "ori_img")
OUTPUT_DIR = os.path.join("data", "black_img")

# 支持的图片格式
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def convert_to_grayscale(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    """
    将 input_dir 中所有彩色图片转换为灰度图，保存至 output_dir。

    参数:
        input_dir: 彩色图片所在目录
        output_dir: 灰度图输出目录
    """
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 获取所有图片文件
    files = os.listdir(input_dir)
    image_files = [
        f for f in files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"在 {input_dir} 中未找到图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片，开始转换...")

    for filename in tqdm(image_files, desc="转换进度"):
        input_path = os.path.join(input_dir, filename)

        # 读取彩色图片并转为灰度图
        image = Image.open(input_path)
        grayscale_image = image.convert("L")

        # 保存灰度图（保持原格式后缀）
        output_path = os.path.join(output_dir, filename)
        grayscale_image.save(output_path)

    print(f"转换完成！灰度图已保存到 {output_dir}")


if __name__ == "__main__":
    convert_to_grayscale()
