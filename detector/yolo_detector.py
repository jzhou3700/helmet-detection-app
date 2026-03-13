import torch
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import numpy as np
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')

class YOLODetector:
    """YOLO目标检测器 - 支持行人检测和已训练的头盔检测"""

    def __init__(self,
                 person_model: str = "yolov8n.pt",
                 helmet_model: str = "keremberke/yolov8m-helmet-detection",
                 use_trained_helmet: bool = True,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45):
        """
        初始化YOLO检测器

        Args:
            person_model: 行人检测模型
            helmet_model: 头盔检测模型（来自HuggingFace）
            use_trained_helmet: 是否使用已训练的头盔检测模型
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
        """
        print(f"🔧 初始化YOLO检测器...")
        print(f"   行人检测模型: {person_model}")
        print(f"   使用已训练头盔模型: {use_trained_helmet}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   设备: {self.device}")

        # 加载行人检测模型
        self.person_model = YOLO(person_model)
        self.person_model.to(self.device)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_trained_helmet = use_trained_helmet

        # 如果启用已训练的头盔检测模型
        if use_trained_helmet:
            print(f"   正在从HuggingFace Hub下载头盔检测模型...")
            print(f"   模型仓库: {helmet_model}")
            try:
                # 使用 huggingface_hub 直接从 HuggingFace Hub 下载已训练的头盔检测模型权重
                local_model_path = hf_hub_download(
                    repo_id=helmet_model,
                    filename="best.pt"
                )
                print(f"   模型已下载至: {local_model_path}")
                self.helmet_model = YOLO(local_model_path)
                self.helmet_model.to(self.device)
                print(f"✅ 头盔检测模型加载成功!")
            except Exception as e:
                print(f"⚠️  无法从HuggingFace加载头盔模型: {e}")
                print(f"   将使用启发式方法进行头盔检测")
                self.use_trained_helmet = False

        self.class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
        }

        print(f"✅ YOLO检测器初始化完成!")

    def detect(self, frame: np.ndarray) -> Dict:
        """对单个画面进行行人检测"""
        results = self.person_model(
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

    def detect_helmet_trained(self, frame: np.ndarray, person_bbox: Tuple[int, int, int, int]) -> bool:
        """
        使用已训练的头盔检测模型检测头盔

        Args:
            frame: 完整的视频帧
            person_bbox: 行人的边界框 [x1, y1, x2, y2]

        Returns:
            True: 检测到头盔，False: 未检测到头盔
        """
        if not self.use_trained_helmet:
            return False

        x1, y1, x2, y2 = [int(x) for x in person_bbox]

        # 确保坐标有效
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return False

        try:
            # 提取行人区域
            person_roi = frame[y1:y2, x1:x2]

            if person_roi.size == 0:
                return False

            # 使用已训练的头盔检测模型进行推理（较低阈值以捕获候选）
            results = self.helmet_model(
                person_roi,
                conf=0.3,
                verbose=False
            )

            if len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_name = result.names[int(box.cls[0])]
                        confidence = float(box.conf[0])

                        # 检查是否检测到helmet类别（较高阈值以减少误判）
                        if "helmet" in class_name.lower() and confidence > 0.4:
                            return True

            return False

        except Exception as e:
            print(f"⚠️  头盔检测异常: {e}")
            return False
