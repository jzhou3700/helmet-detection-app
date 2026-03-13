import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict
import warnings

warnings.filterwarnings('ignore')

class YOLODetector:
    """YOLO目标检测器 - 用于检测行人"""
    
    def __init__(self, model_name: str = "yolov8n.pt", 
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """初始化YOLO检测器"""
        print(f"🔧 初始化YOLO检测器...")
        print(f"   模型: {model_name}")
        print(f"   置信度: {confidence_threshold}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   设备: {self.device}")
        
        self.model = YOLO(model_name)
        self.model.to(self.device)
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        self.class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorbike",
        }
        
        print(f"✅ YOLO模型加载完成!")
        
    def detect(self, frame: np.ndarray) -> Dict:
        """对单个画面进行检测"""
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        detections = {
            "persons": [],
            "helmets": [],
            "raw_results": results
        }
        
        if len(results) > 0:
            result = results[0]
            
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    if class_id == 0:
                        detection = {
                            "class_id": class_id,
                            "class_name": "person",
                            "confidence": confidence,
                            "bbox": [int(x) for x in bbox]
                        }
                        detections["persons"].append(detection)
        
        return detections
