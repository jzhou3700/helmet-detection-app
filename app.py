import io

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from detector.image_detector import ImageDetector
from config import YOLO_CONFIG

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="安全帽检测系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    h1 { color: #1f77b4; text-align: center; }
    .stMetric { background-color: #f0f2f6; padding: 15px; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ 安全帽检测系统")
st.markdown("---")
st.markdown("""
### 🎯 系统说明
上传一张照片，系统将自动检测图中每个人是否佩戴了安全帽，并在图像上标注结果。
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 配置参数")

    confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="YOLO模型检测置信度的最小值，越高越严格",
    )

    iou_threshold = st.slider(
        "IOU阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="非最大值抑制(NMS)的IOU阈值",
    )

    st.markdown("---")
    st.info("""
    ### 💡 使用提示
    - 置信度越高，检测结果越严格
    - 建议 GPU 环境下运行以加快速度
    """)

# ── File uploader ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "📤 上传图片",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="支持的格式: JPG, JPEG, PNG, BMP, WEBP",
)

# ── Detection ─────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    st.markdown("---")

    # Decode uploaded image → BGR numpy array
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("❌ 无法解码图片，请检查文件是否损坏或格式是否正确。")
        st.stop()

    col_orig, col_result = st.columns(2)

    with col_orig:
        st.subheader("📷 原始图片")
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

    with st.spinner("🔄 正在加载模型并检测，请稍候…"):
        try:
            detector = ImageDetector(
                model_name=YOLO_CONFIG["helmet_detector_model"],
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold,
            )
            result = detector.detect(image_bgr)
        except Exception as e:
            import traceback
            st.error(f"❌ 检测出错: {e}")
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())
            st.stop()

    with col_result:
        st.subheader("🔍 检测结果")
        annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, use_column_width=True)

    # ── Stats ─────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📊 检测统计")

    total = result["helmet_count"] + result["no_helmet_count"]
    helmet_rate = (result["helmet_count"] / total * 100) if total > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 检测总人数", total)
    c2.metric("✅ 佩戴头盔", result["helmet_count"])
    c3.metric("⚠️ 未佩戴头盔", result["no_helmet_count"])
    c4.metric("📈 头盔佩戴率", f"{helmet_rate:.1f}%")

    if result["no_helmet_count"] > 0:
        st.warning(f"⚠️ 检测到 **{result['no_helmet_count']}** 人未佩戴头盔！")
    elif total > 0:
        st.success("✅ 所有检测到的人员均已佩戴头盔！")
    else:
        st.info("ℹ️ 图片中未检测到人员或头盔。")

    # ── Download annotated image ──────────────────────────────────────────────
    st.markdown("---")
    annotated_pil = Image.fromarray(annotated_rgb)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    st.download_button(
        label="📥 下载标注结果图片",
        data=buf.getvalue(),
        file_name=f"helmet_detection_{Path(uploaded_file.name).stem}.png",
        mime="image/png",
    )

else:
    st.info("👆 请上传一张图片开始检测")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### ✨ 功能特性
        - 🖼️ 支持多种图片格式（JPG / PNG / BMP / WEBP）
        - 🤖 基于 YOLO 深度学习直接检测头盔
        - 📊 实时显示佩戴率统计
        - 🎨 彩色边界框标注
        - 📥 支持下载标注结果
        """)

    with col2:
        st.markdown("""
        ### 📋 标注说明
        - 🟢 **绿色框** = 佩戴头盔
        - 🔴 **红色框** = 未佩戴头盔
        - 标签显示类别与置信度分数
        - 左上角汇总总人数统计
        """)
