import io
import os
import tempfile

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from detector.image_detector import ImageDetector
from detector.yolo_detector import YOLODetector
from detector.video_processor import VideoProcessor
from config import YOLO_CONFIG
from config import VIDEO_CONFIG, USE_TRAINED_HELMET_MODEL

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
当前模型只会标注**骑电动车的人员/乘员**是否佩戴安全帽；未骑电动车的普通行人不会被框出。
""")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 配置参数")

    confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.35,
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
    st.caption("推荐参数：置信度 0.35~0.5，IOU 0.4~0.5。当前默认适合头盔漏检与误检平衡场景。")
    st.info("""
    ### 💡 使用提示
    - 置信度越高，检测结果越严格
    - 建议 GPU 环境下运行以加快速度
    """)

# ── Mode tabs ─────────────────────────────────────────────────────────────────
image_tab, video_tab = st.tabs(["🖼️ 图片检测", "🎥 视频检测"])

with image_tab:
    st.caption("说明：图片检测仅统计模型识别到的骑电动车人员/乘员，普通行人不会被纳入统计。")
    uploaded_files = st.file_uploader(
        "📤 上传图片（支持批量上传）",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        help="支持的格式: JPG, JPEG, PNG, BMP, WEBP；可一次选择多张图片进行批量处理",
        key="image_uploader",
    )

    if uploaded_files:
        # Initialize detector once for all images
        with st.spinner("🔄 正在加载模型，请稍候…"):
            try:
                detector = ImageDetector(
                    model_path=YOLO_CONFIG["helmet_detector_model"],
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                )
            except Exception as e:
                import traceback

                st.error(f"❌ 模型加载出错: {e}")
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())
                st.stop()

        total_all = 0
        helmet_all = 0
        no_helmet_all = 0

        for idx, uploaded_file in enumerate(uploaded_files):
            st.markdown("---")
            st.subheader(f"📁 图片 {idx + 1} / {len(uploaded_files)}: {uploaded_file.name}")

            # Decode uploaded image -> BGR numpy array
            file_bytes = np.frombuffer(uploaded_file.getvalue(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if image_bgr is None:
                st.error(f"❌ 无法解码图片 **{uploaded_file.name}**，请检查文件是否损坏或格式是否正确。")
                continue

            col_orig, col_result = st.columns(2)

            with col_orig:
                st.subheader("📷 原始图片")
                st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_column_width=True)

            with st.spinner(f"🔄 正在检测 {uploaded_file.name}…"):
                try:
                    result = detector.detect(image_bgr)
                except Exception as e:
                    import traceback

                    st.error(f"❌ 检测出错: {e}")
                    with st.expander("查看错误详情"):
                        st.code(traceback.format_exc())
                    continue

            with col_result:
                st.subheader("🔍 检测结果")
                annotated_rgb = cv2.cvtColor(result["annotated_image"], cv2.COLOR_BGR2RGB)
                st.image(annotated_rgb, use_column_width=True)

            total = result["helmet_count"] + result["no_helmet_count"]
            helmet_rate = (result["helmet_count"] / total * 100) if total > 0 else 0

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🛵 检测到的骑行人员", total)
            c2.metric("✅ 戴头盔骑行人员", result["helmet_count"])
            c3.metric("⚠️ 未戴头盔骑行人员", result["no_helmet_count"])
            c4.metric("📈 骑行人员头盔佩戴率", f"{helmet_rate:.1f}%")

            if result["no_helmet_count"] > 0:
                st.warning(f"⚠️ 检测到 **{result['no_helmet_count']}** 名骑行人员未佩戴头盔！")
            elif total > 0:
                st.success("✅ 所有检测到的骑行人员均已佩戴头盔！")
            else:
                st.info("ℹ️ 图片中未检测到骑电动车人员/乘员，或未识别到可用目标。")

            annotated_pil = Image.fromarray(annotated_rgb)
            buf = io.BytesIO()
            annotated_pil.save(buf, format="PNG")
            st.download_button(
                label="📥 下载标注结果图片",
                data=buf.getvalue(),
                file_name=f"helmet_detection_{Path(uploaded_file.name).stem}.png",
                mime="image/png",
                key=f"download_{idx}",
            )

            total_all += total
            helmet_all += result["helmet_count"]
            no_helmet_all += result["no_helmet_count"]

        if len(uploaded_files) > 1:
            st.markdown("---")
            st.subheader("📊 批量检测汇总统计")
            overall_rate = (helmet_all / total_all * 100) if total_all > 0 else 0
            s1, s2, s3, s4 = st.columns(4)
            s1.metric("🖼️ 处理图片数", len(uploaded_files))
            s2.metric("🛵 累计骑行人员", total_all)
            s3.metric("✅ 累计戴头盔骑行人员", helmet_all)
            s4.metric("📈 整体骑行人员头盔佩戴率", f"{overall_rate:.1f}%")
    else:
        st.info("👆 请上传一张或多张图片开始检测")

with video_tab:
    st.caption("说明：视频检测仅统计模型识别到的骑电动车人员/乘员，普通行人不会被框出或计数。")
    uploaded_video = st.file_uploader(
        "📤 上传视频",
        type=VIDEO_CONFIG["supported_formats"],
        accept_multiple_files=False,
        help="支持 MP4 / AVI / MOV / MKV / FLV / WMV",
        key="video_uploader",
    )

    max_frames = st.number_input(
        "最大处理帧数（0 = 全部帧）",
        min_value=0,
        value=0,
        step=100,
        help="用于加速测试。设置较小值可以更快看到结果。",
        key="video_max_frames",
    )

    enable_live_preview = st.checkbox(
        "实时预览打框视频",
        value=True,
        help="处理过程中在网页里实时显示带标注的画面。",
        key="video_live_preview",
    )

    preview_every_n_frames = st.slider(
        "预览刷新间隔（每N帧更新一次）",
        min_value=1,
        max_value=30,
        value=3,
        step=1,
        help="数值越小越流畅，但会占用更多资源。",
        key="video_preview_stride",
    )

    if uploaded_video is None:
        st.info("👆 请上传视频开始检测")
    else:
        tmp_input_path = None
        output_video_path = None
        try:
            suffix = Path(uploaded_video.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_video.read())
                tmp_input_path = tmp_file.name

            output_video_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
                dir=tempfile.gettempdir(),
            ).name

            with st.spinner("🔄 正在加载视频检测模型，请稍候…"):
                detector = YOLODetector(
                    person_model=YOLO_CONFIG["person_detector_model"],
                    helmet_model=YOLO_CONFIG["helmet_detector_model"],
                    use_trained_helmet=USE_TRAINED_HELMET_MODEL,
                    confidence_threshold=confidence_threshold,
                    iou_threshold=iou_threshold,
                )
                processor = VideoProcessor(detector)

            progress_bar = st.progress(0.0)
            status_text = st.empty()
            live_preview_placeholder = st.empty() if enable_live_preview else None

            if enable_live_preview:
                st.caption("实时预览：正在显示处理中的标注画面")

            def progress_callback(current: int, total: int):
                if total <= 0:
                    progress_bar.progress(0.0)
                    status_text.write("处理中...")
                    return
                p = min(current / total, 1.0)
                progress_bar.progress(p)
                status_text.write(f"处理进度：{current}/{total} 帧 ({p * 100:.1f}%)")

            def frame_callback(frame_bgr: np.ndarray, current: int, total: int):
                if not enable_live_preview or live_preview_placeholder is None:
                    return
                if current % preview_every_n_frames != 0 and current != total:
                    return

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                live_preview_placeholder.image(
                    frame_rgb,
                    channels="RGB",
                    use_column_width=True,
                    caption=f"实时预览：第 {current}/{total} 帧",
                )

            with st.spinner("🎬 正在处理视频..."):
                stats = processor.process_video(
                    input_path=tmp_input_path,
                    output_path=output_video_path,
                    max_frames=max_frames,
                    progress_callback=progress_callback,
                    frame_callback=frame_callback,
                )

            progress_bar.progress(1.0)
            status_text.write("✅ 处理完成")

            st.markdown("---")
            st.subheader("📈 视频检测统计")
            total_persons = stats["total_persons"]
            no_helmet_count = stats["no_helmet_count"]
            helmet_count = total_persons - no_helmet_count
            helmet_rate = (helmet_count / total_persons * 100) if total_persons > 0 else 0
            detection_instances = stats.get("total_detection_instances", 0)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🎬 处理帧数", stats["total_frames"])
            m2.metric("🛵 去重后骑行人员", total_persons)
            m3.metric("⚠️ 去重后未戴头盔骑行人员", no_helmet_count)
            m4.metric("📈 去重后骑行人员头盔佩戴率", f"{helmet_rate:.1f}%")

            if detection_instances:
                st.caption(f"说明：视频汇总已按跨帧同一骑行人员去重；逐帧检测实例总数为 {detection_instances}。")

            violations = stats.get("frames_with_violations", {})
            if violations:
                st.warning(f"⚠️ 检测到 **{len(violations)}** 个存在未戴头盔骑行人员的违规帧（下表为逐帧情况，汇总人数已去重）。")
                with st.expander("查看违规帧明细"):
                    rows = [
                        {"帧号": frame_idx, "未戴头盔骑行人员": count}
                        for frame_idx, count in sorted(violations.items())
                    ]
                    st.dataframe(rows, use_container_width=True)
            else:
                st.success("✅ 未发现未戴头盔的骑行人员。")

            if output_video_path and os.path.exists(output_video_path):
                st.markdown("---")
                with st.expander("🎥 检测结果视频与下载", expanded=True):
                    st.video(output_video_path)
                    with open(output_video_path, "rb") as f:
                        st.download_button(
                            label="📥 下载检测结果视频",
                            data=f,
                            file_name=f"helmet_detection_{Path(uploaded_video.name).stem}.mp4",
                            mime="video/mp4",
                        )
            else:
                st.error("❌ 视频输出失败，请重试。")

        except Exception as e:
            import traceback

            st.error(f"❌ 视频检测出错: {e}")
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc())
        finally:
            if tmp_input_path and os.path.exists(tmp_input_path):
                try:
                    os.remove(tmp_input_path)
                except OSError:
                    pass

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✨ 功能特性
    - 🖼️ 支持多种图片格式（JPG / PNG / BMP / WEBP）
    - 🎥 支持视频上传检测（MP4 / AVI / MOV / MKV / FLV / WMV）
    - 📦 支持批量上传图片，一次处理多张
    - 🤖 基于 YOLO 深度学习检测骑电动车人员/乘员的头盔佩戴情况
    - 📊 提供骑行人员、违规人数与佩戴率统计
    """)

with col2:
    st.markdown("""
    ### 📋 标注说明
    - 🟢 **绿色框** = 佩戴头盔
    - 🔴 **红色框** = 未佩戴头盔
    - 标签显示置信度分数
    - 视频左上角显示逐帧统计
    """)
