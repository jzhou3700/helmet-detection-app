"""共享视频检测器。"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import warnings

import numpy as np
import torch
from ultralytics import YOLO

warnings.filterwarnings("ignore")


def _infer_helmet_classes(class_names: Dict[int, str]) -> Tuple[set[int], set[int]]:
    helmet_ids: set[int] = set()
    no_helmet_ids: set[int] = set()

    for class_id, name in class_names.items():
        normalized = str(name).strip().lower().replace("_", " ")
        if "without" in normalized or "no helmet" in normalized or "nohelmet" in normalized:
            no_helmet_ids.add(int(class_id))
        elif "with helmet" in normalized or normalized == "helmet" or (
            "helmet" in normalized and "without" not in normalized and "no " not in normalized
        ):
            helmet_ids.add(int(class_id))

    if not helmet_ids and 0 in class_names:
        helmet_ids.add(0)
    if not no_helmet_ids and 1 in class_names:
        no_helmet_ids.add(1)

    return helmet_ids, no_helmet_ids


class YOLODetector:
    """YOLO 检测器，优先使用头盔模型直接输出戴帽/未戴帽目标。"""

    def __init__(
        self,
        person_model: str = "yolov8n.pt",
        helmet_model: str = "models/best.pt",
        use_trained_helmet: bool = True,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.person_model_path = person_model
        self.helmet_model_path = helmet_model
        self.use_trained_helmet = use_trained_helmet

        self.person_model: Optional[YOLO] = None
        self.helmet_model: Optional[YOLO] = None
        self.helmet_class_ids: set[int] = set()
        self.no_helmet_class_ids: set[int] = set()
        self.class_names: Dict[int, str] = {}

        if self.use_trained_helmet:
            self._load_helmet_model()

        if self.helmet_model is None:
            self.use_trained_helmet = False
            self._load_person_model()

    def _resolve_model_target(self, model_ref: str) -> str:
        candidate = Path(model_ref)
        if candidate.exists():
            return str(candidate)
        if model_ref.startswith("huggingface://"):
            return model_ref
        return f"huggingface://{model_ref}"

    def _load_helmet_model(self) -> None:
        target = self._resolve_model_target(self.helmet_model_path)
        self.helmet_model = YOLO(target)
        self.helmet_model.to(self.device)
        self.class_names = {
            int(class_id): str(name) for class_id, name in self.helmet_model.names.items()
        }
        self.helmet_class_ids, self.no_helmet_class_ids = _infer_helmet_classes(self.class_names)

    def _load_person_model(self) -> None:
        self.person_model = YOLO(self.person_model_path)
        self.person_model.to(self.device)

    def detect(self, frame: np.ndarray) -> Dict:
        if self.use_trained_helmet and self.helmet_model is not None:
            return self._detect_with_helmet_model(frame)
        return self._detect_persons_only(frame)

    def _detect_with_helmet_model(self, frame: np.ndarray) -> Dict:
        if self.helmet_model is None:
            raise RuntimeError("helmet model is not loaded")
        model = cast(YOLO, self.helmet_model)

        results = model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        persons: List[Dict] = []
        helmet_count = 0
        no_helmet_count = 0

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = [int(x) for x in box.xyxy[0].cpu().numpy()]
                class_name = self.class_names.get(class_id, str(class_id))

                has_helmet: Optional[bool]
                if class_id in self.helmet_class_ids:
                    has_helmet = True
                    helmet_count += 1
                elif class_id in self.no_helmet_class_ids:
                    has_helmet = False
                    no_helmet_count += 1
                else:
                    has_helmet = None

                persons.append(
                    {
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": confidence,
                        "bbox": bbox,
                        "has_helmet": has_helmet,
                    }
                )

        return {
            "persons": persons,
            "helmet_count": helmet_count,
            "no_helmet_count": no_helmet_count,
            "raw_results": results,
            "using_trained_helmet": True,
        }

    def _detect_persons_only(self, frame: np.ndarray) -> Dict:
        if self.person_model is None:
            raise RuntimeError("person model is not loaded")
        model = cast(YOLO, self.person_model)

        results = model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        persons: List[Dict] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                if class_id != 0:
                    continue
                confidence = float(box.conf[0])
                bbox = [int(x) for x in box.xyxy[0].cpu().numpy()]
                persons.append(
                    {
                        "class_id": class_id,
                        "class_name": "person",
                        "confidence": confidence,
                        "bbox": bbox,
                        "has_helmet": None,
                    }
                )

        return {
            "persons": persons,
            "helmet_count": 0,
            "no_helmet_count": 0,
            "raw_results": results,
            "using_trained_helmet": False,
        }

