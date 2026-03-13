"""辅助函数集合"""

import cv2
import numpy as np
from typing import Tuple, List
import os

def resize_frame(frame: np.ndarray, max_width: int = 1280, max_height: int = 720) -> np.ndarray:
    """调整画面大小（保持纵横比）"""
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    
    if scale < 1.0:
        new_width = int(width * scale)
        new_height = int(height * scale)
        frame = cv2.resize(frame, (new_width, new_height))
    
    return frame

def get_video_info(video_path: str) -> dict:
    """获取视频信息"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    info = {
        "fps": int(fps),
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "duration_seconds": int(total_frames / fps) if fps > 0 else 0
    }
    
    cap.release()
    return info

def format_time(seconds: int) -> str:
    """格式化时间"""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def save_frame(frame: np.ndarray, path: str, quality: int = 95):
    """保存画面为图像文件"""
    dir_name = os.path.dirname(path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
