#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Banana Ripeness Classification — Web UI
========================================
基于 YOLOv8-cls 训练的香蕉成熟度分类模型，提供 Gradio Web 界面。

启动方式:
    python app.py
    python app.py --port 7860 --server_name 127.0.0.1
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import gradio as gr

# ── 配置 ───────────────────────────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "runs", "classify", "banana_ripeness", "weights", "best.pt")

# 6 个香蕉成熟度类别 (与训练数据文件夹名一致)
CLASS_NAMES = ["freshripe", "freshunripe", "overripe", "ripe", "rotten", "unripe"]

CLASS_LABELS_CN = {
    "freshripe":    "新鲜成熟",
    "freshunripe":  "新鲜未熟",
    "overripe":     "过熟",
    "ripe":         "成熟",
    "rotten":       "腐烂",
    "unripe":       "未熟",
}

CLASS_COLORS = {
    "freshripe":    "#4CAF50",
    "freshunripe":  "#8BC34A",
    "overripe":     "#FF9800",
    "ripe":         "#FFEB3B",
    "rotten":       "#795548",
    "unripe":       "#2196F3",
}

CLASS_DESCRIPTIONS = {
    "freshripe":    "香蕉已成熟，色泽鲜黄，适合立即食用。",
    "freshunripe":  "香蕉新鲜但尚未成熟，建议放置 2-4 天待熟。",
    "overripe":     "香蕉已过度成熟，表皮出现褐斑，需尽快食用或加工。",
    "ripe":         "香蕉完全成熟，甜度最佳，推荐食用。",
    "rotten":       "香蕉已腐烂变质，不可食用，请丢弃。",
    "unripe":       "香蕉未成熟，颜色青绿，需等待成熟。",
}

# ── 模型加载 ────────────────────────────────────────────────────────────

_YOLO_MODEL = None
_MODEL_LOADED = False
_LOAD_ERROR = None


def get_model():
    """懒加载 YOLO 分类模型"""
    global _YOLO_MODEL, _MODEL_LOADED, _LOAD_ERROR

    if _MODEL_LOADED:
        return _YOLO_MODEL, _LOAD_ERROR

    _MODEL_LOADED = True

    if not os.path.exists(MODEL_PATH):
        _LOAD_ERROR = f"模型文件未找到: {MODEL_PATH}\n请先运行 train_banana.py 训练模型。"
        return None, _LOAD_ERROR

    try:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(MODEL_PATH)
        return _YOLO_MODEL, None
    except Exception as e:
        _LOAD_ERROR = f"模型加载失败: {e}"
        return None, _LOAD_ERROR


# ── 分类函数 ────────────────────────────────────────────────────────────

def classify_banana(image: np.ndarray):
    """对上传的香蕉图像进行分类"""
    if image is None:
        return None, "### 请上传一张香蕉图片。", [], ""

    model, error = get_model()
    if model is None:
        return image, f"### 模型未就绪\n\n{error}", [], ""

    # 推理
    results = model(image, verbose=False)

    if results is None or len(results) == 0:
        return image, "### 分类失败\n\n模型未能返回结果，请重试。", [], ""

    # 获取 top-3 预测
    probs = results[0].probs
    if probs is None:
        return image, "### 分类失败\n\n无法获取概率分布。", [], ""

    top3_indices = probs.top5[:3]
    top3_confs = probs.top5conf[:3].tolist()

    # 主预测
    top1_idx = top3_indices[0]
    top1_name = results[0].names[top1_idx]
    top1_conf = top3_confs[0]
    top1_cn = CLASS_LABELS_CN.get(top1_name, top1_name)
    top1_color = CLASS_COLORS.get(top1_name, "#888888")
    top1_desc = CLASS_DESCRIPTIONS.get(top1_name, "")

    # 构建结果 Markdown
    summary = f"""
## 检测结果

<div style="text-align: center; padding: 20px; margin: 10px 0;
            background: {top1_color}22; border: 2px solid {top1_color};
            border-radius: 16px;">

  <div style="font-size: 3em; margin: 10px 0;">🍌</div>
  <h2 style="color: {top1_color}; margin: 8px 0;">{top1_cn}</h2>
  <p style="font-size: 1.3em; color: #555; margin: 4px 0;">
    置信度: <b>{top1_conf:.1%}</b>
  </p>
  <p style="color: #777; font-size: 0.95em; margin: 8px 16px;">
    {top1_desc}
  </p>

</div>
"""

    # 构建 top-3 表格
    table_data = []
    for idx, conf in zip(top3_indices, top3_confs):
        name = results[0].names[idx]
        cn_name = CLASS_LABELS_CN.get(name, name)
        color = CLASS_COLORS.get(name, "#888888")
        bar_len = int(conf * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        table_data.append([cn_name, f"{conf:.1%}", bar, color])

    # 结果图片标注
    annotated = _annotate_result(image, top1_cn, top1_conf, top1_color)

    return annotated, summary, table_data, top1_name


def _annotate_result(image: np.ndarray, label_cn: str, confidence: float,
                     color_hex: str) -> np.ndarray:
    """在结果图像底部添加分类标签条"""
    img = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(img)

    text = f"{label_cn}  ({confidence:.1%})"
    font = _get_font(24)

    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = len(text) * 14, 30

    # 底部标签条
    from PIL import ImageColor
    color_rgb = ImageColor.getrgb(color_hex)
    bar_h = th + 20
    bar_y = img.height - bar_h

    # 半透明背景
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle([0, bar_y, img.width, img.height],
                           fill=(*color_rgb, 200))

    # 文字
    tx = (img.width - tw) // 2
    ty = bar_y + (bar_h - th) // 2
    overlay_draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    img = img.convert("RGBA")
    img = Image.alpha_composite(img, overlay)
    return np.array(img.convert("RGB"))


def _get_font(size: int = 18):
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                continue
    return ImageFont.load_default()


# ── UI 构建 ─────────────────────────────────────────────────────────────

def create_ui():
    theme = gr.themes.Soft(
        primary_hue="yellow",
        secondary_hue="amber",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    css = """
    .banana-title h1 {
        font-size: 2.2em; font-weight: 800;
        background: linear-gradient(135deg, #f9a825, #fdd835, #fff176);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0;
    }
    footer { display: none !important; }
    """

    with gr.Blocks(title="Banana Ripeness Classification", css=css) as app:

        # 标题
        gr.HTML("""
        <div class="banana-title" style="text-align: center; padding: 10px 0;">
            <h1>Banana Ripeness Classification</h1>
            <p style="color: #888; font-size: 1em;">
                YOLOv8-cls · 6 类成熟度 · freshripe / freshunripe / overripe / ripe / rotten / unripe
            </p>
        </div>
        """)

        # 模型状态
        model_status = gr.Markdown("")

        with gr.Row(equal_height=False):
            # 左侧: 输入
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Banana Image",
                    type="numpy", height=380,
                    sources=["upload", "webcam", "clipboard"],
                )
                classify_btn = gr.Button(
                    "Classify Ripeness",
                    variant="primary", size="lg",
                )

                gr.Markdown("""
                ---
                ### 成熟度类别
                | 类别 | 含义 |
                |------|------|
                | 🟢 freshripe | 新鲜成熟 |
                | 🔵 freshunripe | 新鲜未熟 |
                | 🟠 overripe | 过熟 |
                | 🟡 ripe | 成熟 |
                | 🟤 rotten | 腐烂 |
                | 🔷 unripe | 未熟 |
                """)

            # 右侧: 结果
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Classification Result",
                    type="numpy", height=380,
                )
                result_summary = gr.Markdown(
                    "*Upload a banana image and click Classify to see the result.*"
                )
                top3_table = gr.Dataframe(
                    headers=["类别", "置信度", "置信度条", "颜色"],
                    datatype=["str", "str", "str", "str"],
                    label="Top-3 Predictions",
                    interactive=False,
                )

        # 示例区
        gr.Markdown("### Dataset Samples")
        with gr.Row():
            sample_gr = gr.Gallery(
                label="Training Data Examples (from dataset)",
                columns=6, rows=1, height=160,
                allow_preview=False,
            )

        # 事件
        classify_btn.click(
            fn=classify_banana,
            inputs=[input_image],
            outputs=[output_image, result_summary, top3_table, gr.Textbox(visible=False)],
        )

        # 启动时加载模型状态
        app.load(fn=_get_model_status, outputs=[model_status])

    return app, theme


def _get_model_status():
    model, error = get_model()
    if model is not None:
        return f"**Status:** Model loaded. `{MODEL_PATH}`"
    elif error:
        return f"**Status:** {error}"
    return "**Status:** Unknown error."


# ── 主入口 ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Banana Ripeness Classification Web UI")
    parser.add_argument("--port", type=int, default=7861, help="Web server port")
    parser.add_argument("--share", action="store_true", help="Create public sharing link")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Server hostname")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"{'='*60}")
    print(f"  Banana Ripeness Classification — Web UI")
    print(f"  Model: {MODEL_PATH}")
    print(f"{'='*60}")

    app, theme = create_ui()

    print(f"\n  Starting at: http://{args.server_name}:{args.port}")
    print(f"  Press Ctrl+C to stop.\n")

    app.launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
        show_error=True,
        theme=theme,
    )


if __name__ == "__main__":
    main()
