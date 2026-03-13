import cv2
import numpy as np
from pathlib import Path
from typing import Callable, Dict, Tuple
from detector.yolo_detector import YOLODetector

class VideoProcessor:
    """视频处理器 - 逐帧检测并标注"""
    
    COLORS = {
        "person_helmet": (0, 255, 0),
        "person_no_helmet": (0, 0, 255),
        "unknown": (255, 165, 0)
    }
    
    def __init__(self, detector: YOLODetector):
        """初始化视频处理器"""
        self.detector = detector
    
    def process_video(self,
                     input_path: str,
                     output_path: str,
                     max_frames: int = 0,
                     progress_callback: Callable = None) -> Dict:
        """处理视频文件"""
        print(f"🎬 开始处理视频...")
        print(f"   输入: {input_path}")
        print(f"   使用已训练头盔检测: {self.detector.use_trained_helmet}")
        
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps} FPS")
        print(f"   总帧数: {total_frames}")
        
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise ValueError(f"无法创建输出视频: {output_path}")
        
        stats = {
            "total_frames": total_frames,
            "total_persons": 0,
            "no_helmet_count": 0,
            "frames_with_violations": {},
            "detections_per_frame": []
        }
        
        frame_idx = 0
        
        while frame_idx < total_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            detections = self.detector.detect(frame)
            persons = detections["persons"]
            
            no_helmet_in_frame = 0
            for person in persons:
                # 使用已训练的头盔检测模型（如果启用）
                if self.detector.use_trained_helmet:
                    has_helmet = self.detector.detect_helmet_trained(frame, person["bbox"])
                else:
                    # 备选：使用启发式方法
                    has_helmet = self._check_helmet_heuristic(frame, person["bbox"])
                
                if not has_helmet:
                    no_helmet_in_frame += 1
                    stats["no_helmet_count"] += 1
                
                self._draw_bbox(frame, person["bbox"], has_helmet, person["confidence"])
            
            stats["total_persons"] += len(persons)
            
            if no_helmet_in_frame > 0:
                stats["frames_with_violations"][frame_idx] = no_helmet_in_frame
            
            stats["detections_per_frame"].append({
                "frame": frame_idx,
                "persons": len(persons),
                "no_helmet": no_helmet_in_frame
            })
            
            self._add_stats_text(frame, frame_idx, len(persons), no_helmet_in_frame, total_frames)
            
            out.write(frame)
            
            if progress_callback:
                progress_callback(frame_idx + 1, total_frames)
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        print(f"✅ 视频处理完成!")
        return stats
    
    def _check_helmet_heuristic(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """启发式检查是否佩戴头盔"""
        x1, y1, x2, y2 = [int(x) for x in bbox]
        person_height = y2 - y1
        
        if person_height <= 0 or x1 < 0 or y1 < 0:
            return False
        
        head_y1 = y1
        head_y2 = int(y1 + person_height * 0.25)
        head_y2 = max(head_y2, y1 + 30)
        
        head_y2 = min(head_y2, frame.shape[0])
        x1 = max(x1, 0)
        x2 = min(x2, frame.shape[1])
        
        if head_y2 <= head_y1 or x1 >= x2:
            return False
        
        try:
            head_roi = frame[head_y1:head_y2, x1:x2]
            
            if head_roi.size == 0:
                return False
            
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            
            lower_dark = np.array([0, 0, 0])
            upper_dark = np.array([180, 255, 100])
            
            dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
            dark_ratio = np.sum(dark_mask > 0) / dark_mask.size
            
            lower_bright = np.array([0, 0, 150])
            upper_bright = np.array([180, 50, 255])
            
            bright_mask = cv2.inRange(hsv, lower_bright, upper_bright)
            bright_ratio = np.sum(bright_mask > 0) / bright_mask.size
            
            has_helmet = (dark_ratio > 0.25) or (bright_ratio > 0.2)
            
            return has_helmet
        
        except Exception as e:
            return False
    
    def _draw_bbox(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                   has_helmet: bool, confidence: float):
        """在画面上绘制边界框"""
        x1, y1, x2, y2 = [int(x) for x in bbox]
        
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        
        color = self.COLORS["person_helmet"] if has_helmet else self.COLORS["person_no_helmet"]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        label = f"{'✓ Helmet' if has_helmet else '✗ NO Helmet'} {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - label_size[1] - baseline - 5)),
            (x1 + label_size[0] + 5, y1),
            color,
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    def _add_stats_text(self, frame: np.ndarray, frame_idx: int,
                        total_persons: int, no_helmet: int, total_frames: int):
        """在画面上添加统计文本"""
        bg_color = (0, 0, 0)
        text_color = (0, 255, 0)
        warning_color = (0, 0, 255)
        
        stats_text = [
            f"Frame: {frame_idx}/{total_frames}",
            f"Persons: {total_persons}",
            f"No Helmet: {no_helmet}"
        ]
        
        y_offset = 30
        for i, text in enumerate(stats_text):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(
                frame,
                (5, y_offset + i * 25 - text_size[1] - 5),
                (10 + text_size[0], y_offset + i * 25 + 5),
                bg_color,
                -1
            )
            
            color = warning_color if "No Helmet" in text and no_helmet > 0 else text_color
            
            cv2.putText(
                frame,
                text,
                (10, y_offset + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
