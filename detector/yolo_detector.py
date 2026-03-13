import torch
from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Optional
import warnings

warnings.filterwarnings('ignore')

class YOLODetector:
    """YOLO目标检测器 - 用于检测行人和头盔"""
    
    def __init__(self,
                 person_model: str = "yolov8n.pt",
                 helmet_model: str = "keremberke/yolov8m-helmet-detection",
                 use_trained_helmet: bool = True,
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 helmet_inference_conf: float = 0.3,
                 helmet_accept_conf: float = 0.4,
                 # Legacy parameter kept for backwards compatibility
                 model_name: Optional[str] = None):
        """
        初始化YOLO检测器

        Args:
            person_model: 行人检测模型（标准COCO，用于定位行人）
            helmet_model: 头盔检测模型的HuggingFace repo ID 或本地路径
            use_trained_helmet: True=使用已训练头盔检测模型，False=使用启发式
            confidence_threshold: 行人检测置信度阈值
            iou_threshold: IOU阈值（NMS）
            helmet_inference_conf: 头盔模型推理时的最低置信度过滤
            helmet_accept_conf: 判定"有头盔"所需的最低置信度
            model_name: 已弃用，保留向后兼容性（等同于 person_model）
        """
        # Support legacy model_name parameter
        if model_name is not None:
            person_model = model_name

        print(f"🔧 初始化YOLO检测器...")
        print(f"   行人检测模型: {person_model}")
        print(f"   置信度: {confidence_threshold}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   设备: {self.device}")

        self.person_model = YOLO(person_model)
        self.person_model.to(self.device)

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.use_trained_helmet = use_trained_helmet
        self.helmet_inference_conf = helmet_inference_conf
        self.helmet_accept_conf = helmet_accept_conf

        self.class_names = {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorbike",
        }

        if use_trained_helmet:
            print(f"   头盔检测模型: {helmet_model}")
            self._load_helmet_model(helmet_model)
        else:
            self.helmet_model = None
            print(f"   头盔检测: 启发式（HSV颜色分析）")

        print(f"✅ YOLO模型加载完成!")

    def _load_helmet_model(self, helmet_model: str):
        """从HuggingFace或本地加载头盔检测模型"""
        try:
            from huggingface_hub import hf_hub_download
            print(f"   从HuggingFace下载头盔检测模型: {helmet_model}")
            model_path = hf_hub_download(
                repo_id=helmet_model,
                filename="best.pt"
            )
            self.helmet_model = YOLO(model_path)
            self.helmet_model.to(self.device)
            print(f"   ✅ 头盔检测模型加载成功!")
        except Exception as e:
            print(f"   ⚠️ 头盔检测模型加载失败: {e}")
            print(f"   ⚠️ 降级为启发式检测")
            self.helmet_model = None
            self.use_trained_helmet = False

    # ---------------------------------------------------------------------------
    # Public API – detect() keeps the same signature as before so that
    # VideoProcessor and other callers are unaffected.
    # ---------------------------------------------------------------------------

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

    def detect_helmet_trained(self, frame: np.ndarray,
                               person_bbox: List[int]) -> bool:
        """
        使用已训练模型检测指定行人框内是否佩戴头盔

        Args:
            frame: 完整视频帧（BGR格式）
            person_bbox: 行人边界框 [x1, y1, x2, y2]

        Returns:
            True  – 检测到头盔
            False – 未检测到头盔（或模型不可用）
        """
        if self.helmet_model is None:
            return False

        x1, y1, x2, y2 = [int(c) for c in person_bbox]

        # 裁剪行人区域
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, frame.shape[1])
        y2 = min(y2, frame.shape[0])

        person_roi = frame[y1:y2, x1:x2]
        if person_roi.size == 0:
            return False

        try:
            results = self.helmet_model(person_roi, conf=self.helmet_inference_conf, verbose=False)

            if len(results) > 0:
                result = results[0]
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    # 类别 0 = helmet（keremberke模型）
                    if class_id == 0:
                        confidence = float(box.conf[0])
                        if confidence > self.helmet_accept_conf:
                            return True
        except Exception:
            pass

        return False
