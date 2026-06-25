"""
Pix2Pix 图片上色 Web 服务 —— 含登录注册、模型管理。
启动: python server.py
"""

import io
import json
import os
import sqlite3
import zipfile
from datetime import datetime
from functools import wraps
from pathlib import Path

import numpy as np
import torch
from flask import Flask, g, jsonify, request, send_file, session
from flask_cors import CORS
from kornia.color import lab_to_rgb, rgb_to_lab
from PIL import Image
from torchvision.transforms.functional import crop, resize as tv_resize

from generator import Generator

# ── 配置 ──────────────────────────────────────────────
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)
CONFIG_PATH = Path("model_config.json")
DB_PATH = Path("users.db")
MODEL_INPUT_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
UPLOAD_FOLDER = Path("uploads")
ORIGINALS_DIR = Path("uploads/originals")
RESULTS_DIR = Path("uploads/results")
UPLOAD_FOLDER.mkdir(exist_ok=True)
ORIGINALS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.secret_key = os.urandom(24).hex()
CORS(app, supports_credentials=True)


# ── 数据库 ────────────────────────────────────────────
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(exc):
    g.pop("db", None)

def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            quota_used INTEGER DEFAULT 0,
            quota_limit INTEGER DEFAULT 50,
            quota_date TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            original_name TEXT,
            original_path TEXT,
            result_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # 兼容旧表：quota_date
    user_cols = [r[1] for r in db.execute("PRAGMA table_info(users)").fetchall()]
    if "quota_date" not in user_cols:
        db.execute("ALTER TABLE users ADD COLUMN quota_date TEXT DEFAULT ''")
    # 兼容旧表：缺少新字段则重建
    cols = [r[1] for r in db.execute("PRAGMA table_info(history)").fetchall()]
    if "original_path" not in cols:
        db.execute("DROP TABLE IF EXISTS history")
        db.execute("""
            CREATE TABLE history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                original_name TEXT,
                original_path TEXT,
                result_path TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
    db.commit()
    db.close()

init_db()


# ── 模型配置管理 ──────────────────────────────────────
def load_model_config():
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return {"active": None}

def save_model_config(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

def scan_models():
    """扫描 models/ 目录，返回可用模型列表。"""
    models = []
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.iterdir()):
            if f.suffix in (".pth", ".pt"):
                stat = f.stat()
                models.append({
                    "name": f.name,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M"),
                })
    return models

def get_active_model():
    cfg = load_model_config()
    active = cfg.get("active")
    # 如果配置中的模型不存在，自动选第一个
    available = scan_models()
    if active and any(m["name"] == active for m in available):
        return active
    elif available:
        active = available[0]["name"]
        cfg["active"] = active
        save_model_config(cfg)
        return active
    return None

# 全局模型与当前模型名
current_model_name = None
model = None

def load_model_by_name(name: str):
    """加载指定名称的模型。"""
    global current_model_name, model
    path = MODELS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"模型文件不存在: {path}")
    gen = Generator(inChannels=1, outChannels=2)
    gen.load_state_dict(torch.load(str(path), map_location=DEVICE))
    gen.to(DEVICE)
    gen.eval()
    model = gen
    current_model_name = name
    # 更新配置
    cfg = load_model_config()
    cfg["active"] = name
    save_model_config(cfg)
    print(f"[Model] 已切换模型: {name}")

def init_model():
    """启动时加载活动模型。"""
    name = get_active_model()
    if name:
        load_model_by_name(name)
    else:
        print("[Model] 未找到任何模型，请先上传。")

init_model()


# ── 鉴权装饰器 ────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "请先登录"}), 401
        return f(*args, **kwargs)
    return wrapper

def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "请先登录"}), 401
        if session.get("role") != "admin":
            return jsonify({"error": "需要管理员权限"}), 403
        return f(*args, **kwargs)
    return wrapper


# ── 图片预处理 / 推理（与训练一致） ──────────────────
def preprocess_image(image: Image.Image):
    orig_w, orig_h = image.size
    max_side = max(orig_w, orig_h)
    square = Image.new("L", (max_side, max_side), 0)
    left = (max_side - orig_w) // 2
    top = (max_side - orig_h) // 2
    square.paste(image, (left, top))
    square = square.resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.LANCZOS)

    scale = MODEL_INPUT_SIZE / max_side
    info = {
        "orig_w": orig_w, "orig_h": orig_h,
        "crop_left": int(round(left * scale)),
        "crop_top": int(round(top * scale)),
        "crop_w": int(round(orig_w * scale)),
        "crop_h": int(round(orig_h * scale)),
    }
    gray_np = np.array(square)
    rgb_np = np.stack([gray_np] * 3, axis=-1)
    rgb_tensor = (
        torch.tensor(rgb_np, dtype=torch.float32)
        .permute(2, 0, 1).unsqueeze(0) / 255.0
    )
    lab = rgb_to_lab(rgb_tensor)
    l_channel = lab[:, 0:1, :, :]
    l_norm = l_channel / 50.0 - 1.0
    return l_norm, info

def postprocess_restore(rgb_tensor, info):
    cropped = crop(rgb_tensor, info["crop_top"], info["crop_left"],
                   info["crop_h"], info["crop_w"])
    restored = tv_resize(cropped, (info["orig_h"], info["orig_w"]))
    arr = restored.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((arr * 255).astype(np.uint8))

def run_colorize(image: Image.Image):
    l_norm, info = preprocess_image(image)
    l_norm = l_norm.to(DEVICE)
    with torch.no_grad():
        ab = model(l_norm) * 128
        l = (l_norm + 1.0) * 50.0
        rgb = lab_to_rgb(torch.cat((l, ab), dim=1)).cpu()
    return postprocess_restore(rgb, info)


# ═══════════ API：认证 ═══════════════════════════════
@app.route("/api/auth/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    if not username or not password:
        return jsonify({"error": "用户名和密码不能为空"}), 400
    role = "admin" if data.get("is_admin") else "user"
    db = get_db()
    try:
        db.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                   (username, password, role))
        db.commit()
    except sqlite3.IntegrityError:
        return jsonify({"error": "用户名已存在"}), 409
    return jsonify({"ok": True, "msg": "注册成功"})


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    if not user or user["password"] != password:
        return jsonify({"error": "用户名或密码错误"}), 401
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    return jsonify({"ok": True, "user": {"id": user["id"], "username": user["username"], "role": user["role"]}})


@app.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/me")
def api_me():
    if "user_id" not in session:
        return jsonify({"user": None})
    db = get_db()
    user = db.execute("SELECT id, username, role, quota_used, quota_limit FROM users WHERE id = ?",
                      (session["user_id"],)).fetchone()
    if not user:
        session.clear()
        return jsonify({"user": None})
    return jsonify({"user": dict(user)})


# ═══════════ API：模型管理 ═══════════════════════════
@app.route("/api/models")
def api_models():
    models_list = scan_models()
    active = get_active_model()
    return jsonify({"models": models_list, "active": active})


@app.route("/api/admin/models/upload", methods=["POST"])
@admin_required
def api_upload_model():
    if "model" not in request.files:
        return jsonify({"error": "未上传文件"}), 400
    file = request.files["model"]
    name = file.filename
    if not name.endswith((".pth", ".pt")):
        return jsonify({"error": "仅支持 .pth / .pt 文件"}), 400
    file.save(str(MODELS_DIR / name))
    # 如果还没有活动模型，自动激活
    cfg = load_model_config()
    if not cfg.get("active"):
        cfg["active"] = name
        save_model_config(cfg)
        load_model_by_name(name)
    return jsonify({"ok": True, "name": name})


@app.route("/api/admin/models/activate", methods=["POST"])
@login_required
def api_activate_model():
    data = request.get_json() or {}
    name = data.get("name")
    if not name:
        return jsonify({"error": "缺少模型名称"}), 400
    if not (MODELS_DIR / name).exists():
        return jsonify({"error": "模型文件不存在"}), 404
    try:
        load_model_by_name(name)
    except Exception as e:
        return jsonify({"error": f"加载失败: {e}"}), 500
    return jsonify({"ok": True, "active": name})


@app.route("/api/admin/models/<name>", methods=["DELETE"])
@admin_required
def api_delete_model(name):
    path = MODELS_DIR / name
    if not path.exists():
        return jsonify({"error": "文件不存在"}), 404
    cfg = load_model_config()
    if cfg.get("active") == name:
        return jsonify({"error": "不能删除正在使用的模型，请先切换到其他模型"}), 400
    path.unlink()
    return jsonify({"ok": True})



# ═══════════ API：管理员用户管理 ═══════════════════════
@app.route("/api/admin/users")
@admin_required
def api_admin_users():
    db = get_db()
    users = db.execute("SELECT id, username, role, quota_used, quota_limit, created_at FROM users ORDER BY id").fetchall()
    return jsonify({"users": [dict(u) for u in users]})

@app.route("/api/admin/users/<int:uid>", methods=["PATCH"])
@admin_required
def api_update_user(uid):
    data = request.get_json() or {}
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (uid,)).fetchone()
    if not user:
        return jsonify({"error": "用户不存在"}), 404
    new_role = data.get("role", user["role"])
    new_limit = data.get("quota_limit", user["quota_limit"])
    new_used = data.get("quota_used", user["quota_used"])
    db.execute("UPDATE users SET role=?, quota_limit=?, quota_used=? WHERE id=?",
               (new_role, new_limit, new_used, uid))
    db.commit()
    return jsonify({"ok": True})

@app.route("/api/admin/users/<int:uid>", methods=["DELETE"])
@admin_required
def api_delete_user(uid):
    if uid == session["user_id"]:
        return jsonify({"error": "不能删除自己"}), 400
    db = get_db()
    db.execute("DELETE FROM users WHERE id = ?", (uid,))
    db.commit()
    return jsonify({"ok": True})

# ═══════════ API：图片上色 ═══════════════════════════
def daily_reset_quota():
    """如果日期变更，将当前用户的配额清零。"""
    today = datetime.now().strftime("%Y-%m-%d")
    db = get_db()
    row = db.execute("SELECT quota_date FROM users WHERE id = ?",
                     (session["user_id"],)).fetchone()
    if row and row["quota_date"] != today:
        db.execute("UPDATE users SET quota_used = 0, quota_date = ? WHERE id = ?",
                   (today, session["user_id"]))
        db.commit()

def check_quota():
    db = get_db()
    daily_reset_quota()
    user = db.execute("SELECT quota_used, quota_limit FROM users WHERE id = ?",
                      (session["user_id"],)).fetchone()
    if user and user["quota_used"] >= user["quota_limit"]:
        return False
    return True

def save_history(original_name, original_img, result_img):
    """保存原图与结果到本地文件夹，写入数据库记录。返回记录 dict。"""
    uid = session["user_id"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_name = f"user_{uid}_{ts}_{original_name}"
    orig_path = ORIGINALS_DIR / safe_name
    result_path = RESULTS_DIR / safe_name
    # 转 PNG 保存
    original_img.save(str(orig_path), format="PNG")
    result_img.save(str(result_path), format="PNG")
    db = get_db()
    cur = db.execute(
        "INSERT INTO history (user_id, original_name, original_path, result_path) VALUES (?, ?, ?, ?)",
        (uid, original_name, str(orig_path), str(result_path)))
    db.commit()
    return {
        "id": cur.lastrowid,
        "original_name": original_name,
        "original_url": f"/uploads/originals/{safe_name}",
        "result_url": f"/uploads/results/{safe_name}",
    }

@app.route("/api/history")
@login_required
def api_history():
    db = get_db()
    rows = db.execute(
        "SELECT * FROM history WHERE user_id = ? ORDER BY id DESC LIMIT 50",
        (session["user_id"],)).fetchall()
    records = []
    for r in rows:
        rec = dict(r)
        name = Path(rec["original_path"] or "").name
        rec["original_url"] = f"/uploads/originals/{name}" if name else ""
        rec["result_url"] = f"/uploads/results/{name}" if name else ""
        records.append(rec)
    return jsonify({"history": records})

# 静态文件：serve 上传的图片
@app.route("/uploads/<sub>/<filename>")
def serve_upload(sub, filename):
    folder = ORIGINALS_DIR if sub == "originals" else RESULTS_DIR
    return send_file(folder / filename, mimetype="image/png")

@app.route("/api/colorize", methods=["POST"])
@login_required
def api_colorize_single():
    if not check_quota():
        return jsonify({"error": "今日配额已用完"}), 429
    if "image" not in request.files:
        return jsonify({"error": "未上传图片"}), 400
    file = request.files["image"]
    fname = file.filename or "image.png"
    ext = Path(fname).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": f"不支持的格式: {ext}"}), 400
    try:
        img_bytes = file.read()
        gray_img = Image.open(io.BytesIO(img_bytes)).convert("L")
        original_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        result = run_colorize(gray_img)
        # 扣配额
        db = get_db()
        db.execute("UPDATE users SET quota_used = quota_used + 1 WHERE id = ?",
                   (session["user_id"],))
        db.commit()
        # 保存到本地 + 数据库
        save_history(fname, original_rgb, result)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/colorize-batch", methods=["POST"])
@login_required
def api_colorize_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "未上传图片"}), 400
    if not check_quota():
        return jsonify({"error": "今日配额已用完"}), 429
    zip_buf = io.BytesIO()
    count = 0
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in files:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in ALLOWED_EXTENSIONS:
                continue
            try:
                img_bytes = file.read()
                original_img = Image.open(io.BytesIO(img_bytes)).convert("L")
                original_rgb = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                result = run_colorize(original_img)
                # 保存结果到 zip
                img_buf = io.BytesIO()
                fmt = ext.lstrip(".").upper()
                fmt = "JPEG" if fmt in ("JPG", "JPEG") else fmt
                result.save(img_buf, format=fmt)
                img_buf.seek(0)
                zf.writestr(file.filename, img_buf.read())
                # 存本地历史
                save_history(file.filename or "image.png", original_rgb, result)
                count += 1
            except Exception as e:
                print(f"[Warn] {file.filename}: {e}")
    if count == 0:
        return jsonify({"error": "没有成功处理任何图片"}), 500
    # 扣配额
    db = get_db()
    db.execute("UPDATE users SET quota_used = quota_used + ? WHERE id = ?",
               (count, session["user_id"]))
    db.commit()
    zip_buf.seek(0)
    return send_file(zip_buf, mimetype="application/zip",
                     download_name="colorized_results.zip", as_attachment=True)

# ═══════════ 前端 ════════════════════════════════════
@app.route("/")
def index():
    return app.send_static_file("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
