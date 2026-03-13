import torch
import numpy as np
import cv2
from typing import Dict, Tuple
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import warnings

warnings.filterwarnings('ignore')


class ImageDetector:
    """单图像头盔检测器，使用已训练的 helmet-detection 模型直接在全图上检测。"""

    # 颜色：绿色=佩戴头盔，红色=未佩戴头盔
    COLOR_HELMET = (0, 200, 0)
    COLOR_NO_HELMET = (0, 0, 220)
    TEXT_COLOR = (255, 255, 255)

    def __init__(
        self,
        model_name: str = "tdcdpd/Helmet_Detection",
        confidence_threshold: float = 0.4,
        iou_threshold: float = 0.45,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold

        model_path = hf_hub_download(
            repo_id=model_name,
            filename="best.pt",
            repo_type="space",
        )
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def detect(self, image_bgr: np.ndarray) -> Dict:
        """
        对 BGR 格式的图像进行头盔检测。

        Returns:
            {
                "annotated_image": np.ndarray,  # 绘制好标注框的图像
                "helmet_count": int,            # 佩戴头盔人数
                "no_helmet_count": int,         # 未佩戴头盔人数
                "detections": list              # 每个检测结果的详细信息
            }
        """
        results = self.model(
            image_bgr,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        annotated = image_bgr.copy()
        helmet_count = 0
        no_helmet_count = 0
        detections = []

        if results and results[0].boxes is not None:
            result = results[0]
            # Build a set of class ids that represent "wearing a helmet".
            # The tdcdpd/Helmet_Detection model uses class names such as
            # "helmet" / "no helmet" / "no-helmet".
            # We match case-insensitively: treat any class containing "helmet"
            # but NOT preceded by "no" as a positive (wearing) detection.
            helmet_class_ids = {
                cid
                for cid, name in result.names.items()
                if "helmet" in name.lower() and not name.lower().startswith("no")
            }

            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].cpu().numpy()]

                has_helmet = class_id in helmet_class_ids

                if has_helmet:
                    helmet_count += 1
                else:
                    no_helmet_count += 1

                self._draw_box(annotated, x1, y1, x2, y2, has_helmet, confidence)
                detections.append(
                    {
                        "class_name": class_name,
                        "has_helmet": has_helmet,
                        "confidence": confidence,
                        "bbox": [x1, y1, x2, y2],
                    }
                )

        self._draw_summary(annotated, helmet_count, no_helmet_count)

        return {
            "annotated_image": annotated,
            "helmet_count": helmet_count,
            "no_helmet_count": no_helmet_count,
            "detections": detections,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_box(
        self,
        frame: np.ndarray,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        has_helmet: bool,
        confidence: float,
    ):
        color = self.COLOR_HELMET if has_helmet else self.COLOR_NO_HELMET
        label = f"{'Helmet' if has_helmet else 'No Helmet'} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.TEXT_COLOR,
            1,
        )

    def _draw_summary(self, frame: np.ndarray, helmet_count: int, no_helmet_count: int):
        total = helmet_count + no_helmet_count
        lines = [
            f"Total : {total}",
            f"Helmet : {helmet_count}",
            f"No Helmet: {no_helmet_count}",
        ]
        padding = 8
        line_h = 26
        bg_h = padding * 2 + line_h * len(lines)
        max_w = max(
            cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)[0][0] for t in lines
        )
        cv2.rectangle(frame, (5, 5), (5 + max_w + padding * 2, 5 + bg_h), (0, 0, 0), -1)
        for i, text in enumerate(lines):
            color = (0, 220, 0) if "No Helmet" not in text else (0, 80, 220)
            cv2.putText(
                frame,
                text,
                (5 + padding, 5 + padding + line_h * (i + 1) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
            )
