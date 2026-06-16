"""
批量将灰度图上色（还原为彩色图）。

加载模型权重，读取 /data/black_img 下的灰度图，
使用 Pix2Pix 生成器推理生成 ab 颜色通道，合并 LAB 后转回 RGB，
保存到 /result 目录。

自动处理任意尺寸/长宽比的输入：通过居中 + 黑色填充补成正方形，
缩放至 256×256 后送入模型，输出时裁剪并还原为原始尺寸，
避免画面拉伸变形。
"""

import os
import torch
import numpy as np
from PIL import Image
from generator import Generator
from torchvision.utils import save_image
from torchvision.transforms.functional import crop, resize as tv_resize
from kornia.color import lab_to_rgb, rgb_to_lab
from tqdm import tqdm

# 路径配置
INPUT_DIR = os.path.join("data", "black_img")
OUTPUT_DIR = "result"
MODEL_PATH = "gen-70.pth"
MODEL_INPUT_SIZE = 256

# 设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 支持的图片格式
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(model_path: str) -> Generator:
    """加载训练好的生成器模型。"""
    model = Generator(inChannels=1, outChannels=2)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    print(f"模型已从 {model_path} 加载到 {DEVICE}")
    return model


def preprocess_square(image: Image.Image) -> tuple[torch.Tensor, dict]:
    """
    将任意尺寸的灰度图居中填充为正方形，缩放至模型输入尺寸，
    并提取与训练一致的 LAB L 通道。

    参数:
        image: PIL 灰度图（mode 'L'）

    返回:
        l_norm:  形状 (1, 1, 256, 256)，L 通道归一化到 [-1, 1]（与训练一致）
        info:    还原用信息
    """
    orig_w, orig_h = image.size

    # 以长边为边长，居中粘贴到正方形黑色画布
    max_side = max(orig_w, orig_h)
    square = Image.new("L", (max_side, max_side), 0)
    left = (max_side - orig_w) // 2
    top = (max_side - orig_h) // 2
    square.paste(image, (left, top))

    # 缩放至模型输入尺寸
    square = square.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)

    # 记录还原所需信息
    scale = MODEL_INPUT_SIZE / max_side
    info = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "crop_left": int(round(left * scale)),
        "crop_top": int(round(top * scale)),
        "crop_w": int(round(orig_w * scale)),
        "crop_h": int(round(orig_h * scale)),
    }

    # 灰度图转 3 通道伪 RGB，再提取真正的 LAB L 通道（与 dataloader 训练预处理一致）
    gray_np = np.array(square)  # (256, 256), [0, 255]
    # 复制为 3 通道伪 RGB
    rgb_np = np.stack([gray_np] * 3, axis=-1)  # (256, 256, 3)
    rgb_tensor = (
        torch.tensor(rgb_np, dtype=torch.float32)
        .permute(2, 0, 1)     # (3, 256, 256)
        .unsqueeze(0)          # (1, 3, 256, 256)
        / 255.0                # [0, 1]
    )

    # 用 kornia rgb_to_lab 提取真实 LAB L 通道（与训练完全一致）
    # 输入必须是 [0, 1] RGB
    lab = rgb_to_lab(rgb_tensor)           # (1, 3, 256, 256), L∈[0,100], ab∈[-128,128]
    l_channel = lab[:, 0:1, :, :]          # (1, 1, 256, 256)

    # 归一化到 [-1, 1] — 与 dataloader 的 RGB2LAB 变换一致
    l_norm = l_channel / 50.0 - 1.0

    return l_norm, info


def postprocess_restore(rgb_tensor: torch.Tensor, info: dict) -> torch.Tensor:
    """
    将模型输出的 256×256 结果裁剪并缩放回原始尺寸。
    """
    cropped = crop(
        rgb_tensor,
        info["crop_top"],
        info["crop_left"],
        info["crop_h"],
        info["crop_w"],
    )
    restored = tv_resize(cropped, (info["orig_h"], info["orig_w"]))
    return restored


def colorize_batch(model: Generator, input_dir: str, output_dir: str):
    """对 input_dir 中所有灰度图逐张上色，结果保存到 output_dir。"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    files = os.listdir(input_dir)
    image_files = [
        f for f in files
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_files:
        print(f"在 {input_dir} 中未找到图片文件。")
        return

    print(f"找到 {len(image_files)} 张灰度图，开始上色...")

    with torch.no_grad():
        for filename in tqdm(image_files, desc="上色进度"):
            input_path = os.path.join(input_dir, filename)

            # 1. 读取灰度图
            image = Image.open(input_path).convert("L")

            # 2. 预处理：padding + 提取与训练一致的 LAB L 通道
            l_norm, info = preprocess_square(image)
            l_norm = l_norm.to(DEVICE)

            # 3. 模型推理（输入 L∈[-1,1]，输出 ab∈[-1,1]）
            ab = model(l_norm) * 128           # ab ∈ [-128, 128]

            # 4. 反归一化 L：[-1, 1] → [0, 100]
            l = (l_norm + 1.0) * 50.0          # L ∈ [0, 100]

            # 5. LAB → RGB
            lab_combined = torch.cat((l, ab), dim=1)
            rgb = lab_to_rgb(lab_combined).cpu()

            # 6. 裁剪 + 缩放回原始尺寸
            rgb_restored = postprocess_restore(rgb, info)

            # 7. 保存
            output_path = os.path.join(output_dir, filename)
            save_image(rgb_restored, output_path)

    print(f"上色完成！结果已保存到 {output_dir}")


if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    colorize_batch(model, INPUT_DIR, OUTPUT_DIR)
