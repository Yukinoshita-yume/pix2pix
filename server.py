"""
Pix2Pix 图片上色 Web 服务。

启动后访问 http://localhost:5000 即可使用。
"""

import os
import io
import zipfile
import uuid
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from torchvision.utils import save_image
from torchvision.transforms.functional import crop, resize as tv_resize
from kornia.color import lab_to_rgb, rgb_to_lab

from generator import Generator

# ---- 配置 ----
MODEL_PATH = "gen-70.pth"
MODEL_INPUT_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UPLOAD_FOLDER = Path("uploads")
RESULT_FOLDER = Path("static/results")
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

UPLOAD_FOLDER.mkdir(exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# ---- Flask 应用 ----
app = Flask(__name__, static_folder="static", static_url_path="/static")
CORS(app)

# ---- 加载模型 ----
print(f"[Server] 加载模型: {MODEL_PATH}, device: {DEVICE}")
model = Generator(inChannels=1, outChannels=2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print("[Server] 模型加载完毕，服务就绪。")


def preprocess_image(image: Image.Image) -> tuple[torch.Tensor, dict]:
    """
    灰度图预处理：居中填方 → 缩放到 256 → 提取 LAB L 通道 → 归一化到 [-1,1]。
    与 dataloader 训练预处理完全一致。
    返回 (l_norm, info)。
    """
    orig_w, orig_h = image.size
    max_side = max(orig_w, orig_h)

    # 居中填充为正方形
    square = Image.new("L", (max_side, max_side), 0)
    left = (max_side - orig_w) // 2
    top = (max_side - orig_h) // 2
    square.paste(image, (left, top))
    square = square.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)

    scale = MODEL_INPUT_SIZE / max_side
    info = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "crop_left": int(round(left * scale)),
        "crop_top": int(round(top * scale)),
        "crop_w": int(round(orig_w * scale)),
        "crop_h": int(round(orig_h * scale)),
    }

    # 转 3 通道伪 RGB → 提取 LAB L 通道
    gray_np = np.array(square)
    rgb_np = np.stack([gray_np] * 3, axis=-1)
    rgb_tensor = (
        torch.tensor(rgb_np, dtype=torch.float32)
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )
    lab = rgb_to_lab(rgb_tensor)
    l_channel = lab[:, 0:1, :, :]
    l_norm = l_channel / 50.0 - 1.0  # → [-1, 1]

    return l_norm, info


def postprocess_restore(rgb_tensor: torch.Tensor, info: dict) -> Image.Image:
    """裁剪填充区域 → 恢复原始尺寸 → 返回 PIL Image。"""
    cropped = crop(rgb_tensor, info["crop_top"], info["crop_left"],
                   info["crop_h"], info["crop_w"])
    restored = tv_resize(cropped, (info["orig_h"], info["orig_w"]))

    # tensor → PIL
    arr = restored.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr)


def run_colorize(image: Image.Image) -> Image.Image:
    """对单张灰度图执行上色，返回 RGB PIL Image。"""
    l_norm, info = preprocess_image(image)
    l_norm = l_norm.to(DEVICE)

    with torch.no_grad():
        ab = model(l_norm) * 128
        l = (l_norm + 1.0) * 50.0
        lab_combined = torch.cat((l, ab), dim=1)
        rgb = lab_to_rgb(lab_combined).cpu()

    return postprocess_restore(rgb, info)


# ---- API 路由 ----

@app.route("/api/colorize", methods=["POST"])
def api_colorize_single():
    """单张图片上色，直接返回 PNG。"""
    if "image" not in request.files:
        return jsonify({"error": "未上传图片"}), 400

    file = request.files["image"]
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"不支持的格式: {ext}"}), 400

    try:
        image = Image.open(file.stream).convert("L")
        result = run_colorize(image)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/colorize-batch", methods=["POST"])
def api_colorize_batch():
    """批量上色，返回 ZIP 压缩包。"""
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "未上传图片"}), 400

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue

            try:
                image = Image.open(file.stream).convert("L")
                result = run_colorize(image)

                img_buf = io.BytesIO()
                # 保存为同格式
                fmt = ext.lstrip(".").upper()
                fmt = "JPEG" if fmt in ("JPG", "JPEG") else fmt
                result.save(img_buf, format=fmt)
                img_buf.seek(0)
                zf.writestr(file.filename, img_buf.read())
            except Exception as e:
                print(f"[Warn] 处理 {file.filename} 失败: {e}")

    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip",
                     download_name="colorized_results.zip",
                     as_attachment=True)


@app.route("/")
def index():
    """前端页面。"""
    return app.send_static_file("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
