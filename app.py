import streamlit as st
import cv2
import numpy as np
from pathlib import Path
import tempfile
from detector.yolo_detector import YOLODetector
from detector.video_processor import VideoProcessor
import os
import csv
from io import StringIO


def generate_csv_report(stats, helmet_rate):
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["电动车乘客安全帽检测系统 - 检测报告"])
    writer.writerow([])
    
    writer.writerow(["=== 基本统计信息 ==="])
    writer.writerow(["指标", "数值", "说明"])
    writer.writerow(["总检测帧数", stats["total_frames"], "视频中处理的总帧数"])
    writer.writerow(["总检测行人数", stats["total_persons"], "检测到的所有行人"])
    writer.writerow(["未佩戴头盔人数", stats["no_helmet_count"], "需要警告的行人"])
    writer.writerow(["正确佩戴头盔人数", stats["total_persons"] - stats["no_helmet_count"], "佩戴正确的行人"])
    writer.writerow(["头盔佩戴率(%)", f"{helmet_rate:.2f}%", "正确佩戴的百分比"])
    writer.writerow([])
    
    writer.writerow(["=== 风险分析 ==="])
    writer.writerow(["指标", "数值"])
    writer.writerow(["含违规帧数", len(stats['frames_with_violations'])])
    writer.writerow(["最大违规帧人数", max(stats['frames_with_violations'].values()) if stats['frames_with_violations'] else 0])
    writer.writerow(["平均每帧人数", f"{stats['total_persons']/stats['total_frames']:.2f}"])
    writer.writerow(["平均每帧违规数", f"{stats['no_helmet_count']/stats['total_frames']:.2f}"])
    writer.writerow([])
    
    if stats['detections_per_frame']:
        writer.writerow(["=== 帧级检测数据 ==="])
        writer.writerow(["帧号", "检测人数", "未佩戴头盔", "佩戴率(%)"])
        
        for detection in stats['detections_per_frame']:
            helmet_rate_frame = (detection['persons'] - detection['no_helmet']) / detection['persons'] * 100 if detection['persons'] > 0 else 0
            writer.writerow([
                detection["frame"],
                detection["persons"],
                detection["no_helmet"],
                f"{helmet_rate_frame:.2f}"
            ])
    
    return output.getvalue()


st.set_page_config(
    page_title="安全帽检测系统",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .warning-text {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 18px;
    }
    .success-text {
        color: #09ab3b;
        font-weight: bold;
        font-size: 16px;
    }
    h1 {
        color: #1f77b4;
        text-align: center;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ 电动车乘客安全帽检测系统")
st.markdown("---")
st.markdown("""
### 🎯 系统说明
本系统基于YOLO深度学习模型，能够自动检测电动车乘客是否正确佩戴安全帽。
支持视频文件上传、实时检测标注和统计分析。
""")

with st.sidebar:
    st.header("⚙️ 配置参数")
    
    confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="YOLO模型检测置信度的最小值，越高越严格"
    )
    
    iou_threshold = st.slider(
        "IOU阈值",
        min_value=0.1,
        max_value=1.0,
        value=0.45,
        step=0.05,
        help="非最大值抑制(NMS)的IOU阈值"
    )
    
    max_frames = st.number_input(
        "最大处理帧数 (0=全部)",
        min_value=0,
        value=0,
        step=100,
        help="处理视频的最大帧数，0表示处理全部帧"
    )
    
    st.markdown("---")
    st.info("""
    ### 💡 使用提示
    - 置信度越高，检测结果越严格
    - 处理帧数越少，速度越快
    - 建议GPU环境下运行以加快速度
    """)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📤 上传视频")
    uploaded_file = st.file_uploader(
        "选择视频文件",
        type=["mp4", "avi", "mov", "mkv", "flv", "wmv"],
        help="支持的格式: MP4, AVI, MOV, MKV, FLV, WMV"
    )

with col2:
    st.subheader("📊 实时统计")
    stats_placeholder = st.empty()

if uploaded_file is not None:
    st.markdown("---")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_video_path = tmp_file.name
    
    try:
        with st.spinner("🔄 正在初始化YOLO模型..."):
            detector = YOLODetector(
                model_name="yolov8n.pt",
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
        
        st.success("✅ 模型加载成功")
        
        processor = VideoProcessor(detector)
        
        st.info("🎬 正在处理视频，请耐心等待...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_counter = st.empty()
        
        output_video_path = tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=".mp4",
            dir=tempfile.gettempdir()
        ).name
        
        def progress_callback(current, total):
            progress_bar.progress(min(current / total, 1.0))
            frame_counter.write(f"处理进度: {current}/{total} 帧 ({100*current/total:.1f}%)")
        
        stats = processor.process_video(
            input_path=tmp_video_path,
            output_path=output_video_path,
            max_frames=max_frames,
            progress_callback=progress_callback
        )
        
        progress_bar.progress(1.0)
        status_text.write("✅ 处理完成!")
        frame_counter.empty()
        
        st.markdown("---")
        st.subheader("📈 检测统计结果")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 总检测行人数",
                stats["total_persons"],
                help="视频中检测到的所有行人"
            )
        
        with col2:
            st.metric(
                "⚠️ 未佩戴头盔人数",
                f"{stats['no_helmet_count']}",
                help="需要警告的行人数量"
            )
        
        with col3:
            helmet_rate = (
                (stats['total_persons'] - stats['no_helmet_count']) / 
                stats['total_persons'] * 100
            ) if stats['total_persons'] > 0 else 0
            st.metric(
                "✅ 头盔佩戴率",
                f"{helmet_rate:.1f}%",
                help="正确佩戴头盔的比率"
            )
        
        with col4:
            st.metric(
                "🎬 处理总帧数",
                stats["total_frames"],
                help="视频的总帧数"
            )
        
        st.markdown("---")
        st.subheader("🎥 检测结果视频（带标注）")
        
        if os.path.exists(output_video_path):
            st.video(output_video_path)
        else:
            st.error("❌ 输出视频文件不存在")
        
        st.markdown("---")
        st.subheader("📋 详细检测日志")
        
        if "frames_with_violations" in stats and stats["frames_with_violations"]:
            violation_count = len(stats['frames_with_violations'])
            st.warning(
                f"⚠️ 检测到 **{violation_count}** 个包含未佩戴头盔的帧"
            )
            
            with st.expander(f"查看违规帧详情 (共{violation_count}帧)"):
                violations_df = []
                for frame_idx, persons in sorted(stats['frames_with_violations'].items()):
                    violations_df.append({
                        "帧号": frame_idx,
                        "未佩戴头盔人数": persons
                    })
                
                st.dataframe(violations_df, use_container_width=True)
        else:
            st.success("✅ 所有检测到的行人都正确佩戴了头盔！")
        
        st.markdown("---")
        st.subheader("📊 详细报告")
        
        report_tab1, report_tab2 = st.tabs(["统计摘要", "帧级数据"])
        
        with report_tab1:
            report_col1, report_col2 = st.columns(2)
            
            with report_col1:
                st.write(f"""
                ### 📌 基本信息
                - **视频总帧数**: {stats['total_frames']} 帧
                - **检测总人数**: {stats['total_persons']} 人
                - **未佩戴头盔**: {stats['no_helmet_count']} 人
                - **佩戴头盔**: {stats['total_persons'] - stats['no_helmet_count']} 人
                - **佩戴率**: {helmet_rate:.2f}%
                """)
            
            with report_col2:
                st.write(f"""
                ### ⚠️ 风险指标
                - **含违规帧数**: {len(stats['frames_with_violations'])} 帧
                - **最多违规帧**: {max(stats['frames_with_violations'].values()) if stats['frames_with_violations'] else 0} 人
                - **平均每帧人数**: {stats['total_persons']/stats['total_frames']:.2f} 人/帧
                - **平均每帧违规**: {stats['no_helmet_count']/stats['total_frames']:.2f} 人/帧
                """)
        
        with report_tab2:
            if stats['detections_per_frame']:
                detections_data = []
                for detection in stats['detections_per_frame']:
                    helmet_rate_frame = (detection['persons']-detection['no_helmet'])/detection['persons']*100 if detection['persons'] > 0 else 0
                    detections_data.append({
                        "帧号": detection["frame"],
                        "检测行人数": detection["persons"],
                        "未佩戴头盔数": detection["no_helmet"],
                        "佩戴率(%)": f"{helmet_rate_frame:.1f}"
                    })
                
                st.dataframe(detections_data, use_container_width=True, height=400)
        
        st.markdown("---")
        st.subheader("💾 下载结果")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if os.path.exists(output_video_path):
                with open(output_video_path, "rb") as video_file:
                    st.download_button(
                        label="📥 下载检测结果视频",
                        data=video_file,
                        file_name=f"helmet_detection_{Path(uploaded_file.name).stem}.mp4",
                        mime="video/mp4",
                        help="下载带有检测框和标注的视频文件"
                    )
        
        with col2:
            csv_content = generate_csv_report(stats, helmet_rate)
            st.download_button(
                label="📥 下载检测统计报告(CSV)",
                data=csv_content,
                file_name=f"helmet_detection_report_{Path(uploaded_file.name).stem}.csv",
                mime="text/csv",
                help="下载详细的统计分析报告"
            )
        
    except Exception as e:
        st.error(f"❌ 处理出错: {str(e)}")
        st.info("💡 请检查：\n- 视频格式是否正确\n- 视频文件是否损坏\n- 磁盘空间是否充足")
        
        import traceback
        with st.expander("查看错误详情"):
            st.code(traceback.format_exc())
    
    finally:
        if os.path.exists(tmp_video_path):
            try:
                os.remove(tmp_video_path)
            except OSError:
                pass

else:
    st.info("👈 请在上方上传视频文件开始检测")
    
    st.markdown("---")
    st.subheader("📚 使用指南")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ 功能特性
        - 🎥 支持多种视频格式
        - 🤖 基于YOLO深度学习检测
        - 📊 详细的统计分析
        - 🎨 实时检测标注
        - 📥 结果导出功能
        """)
    
    with col2:
        st.markdown("""
        ### 📋 标注说明
        - 🟢 **绿色框** = 佩戴头盔
        - 🔴 **红色框** = 未佩戴头盔
        - 标签显示置信度分数
        - 左上角显示实时统计
        """)
