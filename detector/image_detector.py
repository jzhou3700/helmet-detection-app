"""图片头盔检测。"""

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO


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


class ImageDetector:
	"""使用头盔模型直接检测图片中的戴帽/未戴帽人员。"""

	def __init__(
		self,
		model_path: str = "models/best.pt",
		confidence_threshold: float = 0.5,
		iou_threshold: float = 0.45,
	):
		self.model_path = str(Path(model_path))
		self.confidence_threshold = confidence_threshold
		self.iou_threshold = iou_threshold
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.model = YOLO(self.model_path)
		self.model.to(self.device)

		self.class_names = {
			int(class_id): str(name) for class_id, name in self.model.names.items()
		}
		self.helmet_class_ids, self.no_helmet_class_ids = _infer_helmet_classes(self.class_names)

		self.colors = {
			"helmet": (0, 255, 0),
			"no_helmet": (0, 0, 255),
			"unknown": (255, 165, 0),
		}

	def detect(self, image_bgr: np.ndarray) -> Dict:
		results = self.model(
			image_bgr,
			conf=self.confidence_threshold,
			iou=self.iou_threshold,
			verbose=False,
		)

		annotated_image = image_bgr.copy()
		detections: List[Dict] = []
		helmet_count = 0
		no_helmet_count = 0

		if results and results[0].boxes is not None:
			for box in results[0].boxes:
				class_id = int(box.cls[0])
				confidence = float(box.conf[0])
				bbox = [int(x) for x in box.xyxy[0].cpu().numpy()]
				class_name = self.class_names.get(class_id, str(class_id))

				if class_id in self.helmet_class_ids:
					has_helmet = True
					helmet_count += 1
					color = self.colors["helmet"]
				elif class_id in self.no_helmet_class_ids:
					has_helmet = False
					no_helmet_count += 1
					color = self.colors["no_helmet"]
				else:
					has_helmet = None
					color = self.colors["unknown"]

				detections.append(
					{
						"class_id": class_id,
						"class_name": class_name,
						"confidence": confidence,
						"bbox": bbox,
						"has_helmet": has_helmet,
					}
				)
				self._draw_detection(annotated_image, bbox, class_name, confidence, color)

		summary = f"Helmet: {helmet_count} | No Helmet: {no_helmet_count}"
		cv2.rectangle(annotated_image, (10, 10), (310, 42), (0, 0, 0), -1)
		cv2.putText(
			annotated_image,
			summary,
			(16, 32),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.7,
			(255, 255, 255),
			2,
		)

		return {
			"annotated_image": annotated_image,
			"helmet_count": helmet_count,
			"no_helmet_count": no_helmet_count,
			"persons": detections,
			"raw_results": results,
			"using_trained_helmet": True,
		}

	def _draw_detection(
		self,
		image: np.ndarray,
		bbox: List[int],
		class_name: str,
		confidence: float,
		color: Tuple[int, int, int],
	) -> None:
		x1, y1, x2, y2 = bbox
		x1 = max(0, x1)
		y1 = max(0, y1)
		x2 = min(image.shape[1] - 1, x2)
		y2 = min(image.shape[0] - 1, y2)

		cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
		label = f"{class_name} {confidence:.2f}"
		(label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
		top = max(0, y1 - label_h - baseline - 6)
		cv2.rectangle(image, (x1, top), (x1 + label_w + 8, y1), color, -1)
		cv2.putText(
			image,
			label,
			(x1 + 4, y1 - baseline - 3),
			cv2.FONT_HERSHEY_SIMPLEX,
			0.5,
			(255, 255, 255),
			1,
		)

