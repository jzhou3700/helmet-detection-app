import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

class YOLODetector:
    """YOLO目标检测器 - 使用头盔检测模型直接检测"""

    def __init__(self,
                 helmet_model: str = "tdcdpd/Helmet_Detection",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        初始化YOLO检测器

        Args:
            helmet_model: 头盔检测模型（来自HuggingFace）
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        print(f"🔧 初始化YOLO检测器...")
        print(f"   头盔检测模型: {helmet_model}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   设备: {self.device}")

        print(f"   正在从HuggingFace加载头盔检测模型...")
        self.helmet_model = YOLO(f"huggingface://{helmet_model}")
        self.helmet_model.to(self.device)
        print(f"✅ 头盔检测模型加载成功!")

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        print(f"✅ YOLO检测器初始化完成!")

    def detect(self, frame: np.ndarray) -> Dict:
        """对单个画面进行头盔检测，直接返回每个检测结果及是否佩戴头盔"""
        results = self.helmet_model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )

        detections = {
            "persons": [],
            "raw_results": results
        }

        if len(results) > 0:
            result = results[0]

            helmet_class_ids = {
                cid for cid, name in result.names.items()
                if "helmet" in name.lower() and not name.lower().startswith("no")
            }

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    bbox = box.xyxy[0].cpu().numpy()

                    detection = {
                        "class_id": class_id,
                        "class_name": result.names[class_id],
                        "confidence": confidence,
                        "bbox": [int(x) for x in bbox],
                        "has_helmet": class_id in helmet_class_ids,
                    }
                    detections["persons"].append(detection)

        return detections

