"""项目配置文件"""

# YOLO模型配置 - 已更新以支持HuggingFace
YOLO_CONFIG = {
    # 行人检测模型（标准COCO）
    "person_detector_model": "yolov8n.pt",

    # 头盔检测模型（来自HuggingFace，已训练）
    "helmet_detector_model": "keremberke/yolov8m-helmet-detection",

    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
}

# 是否使用已训练的头盔检测模型（推荐True）
USE_TRAINED_HELMET_MODEL = True

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

MODEL_OPTIONS = {
    "nano": "yolov8n.pt",
    "small": "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large": "yolov8l.pt",
}

# HuggingFace模型选项
HELMET_MODELS = {
    "yolov8m-helmet": "keremberke/yolov8m-helmet-detection",
    "yolov8s-helmet": "keremberke/yolov8s-helmet-detection",
}
