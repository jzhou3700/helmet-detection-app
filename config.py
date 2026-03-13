"""项目配置文件"""

# YOLO模型配置 - 已更新以支持HuggingFace
YOLO_CONFIG = {
    # 头盔检测模型（来自HuggingFace，已训练）
    "helmet_detector_model": "tdcdpd/Helmet_Detection",

    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
}

VIDEO_CONFIG = {
    "supported_formats": ["mp4", "avi", "mov", "mkv", "flv", "wmv"],
    "max_file_size_mb": 500,
    "output_format": "mp4",
}

STREAMLIT_CONFIG = {
    "page_title": "安全帽检测系统",
    "layout": "wide",
    "theme": "light",
}

COLOR_CONFIG = {
    "helmet": (0, 255, 0),
    "no_helmet": (0, 0, 255),
    "text": (255, 255, 255),
    "stats": (0, 255, 0),
}

# HuggingFace模型选项
HELMET_MODELS = {
    "tdcdpd-helmet": "tdcdpd/Helmet_Detection",
}
