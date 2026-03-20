"""共享视频处理器，支持跨帧去重统计。"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple, cast

import cv2
import numpy as np


class VideoProcessor:
    """逐帧检测并输出带标注视频，同时对跨帧同一人做去重统计。"""

    COLORS = {
        "person_helmet": (0, 255, 0),
        "person_no_helmet": (0, 0, 255),
        "unknown": (255, 165, 0),
    }

    def __init__(
        self,
        detector,
        match_iou_threshold: float = 0.25,
        max_missed_frames: int = 12,
    ):
        self.detector = detector
        self.match_iou_threshold = match_iou_threshold
        self.max_missed_frames = max_missed_frames

    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: int = 0,
        progress_callback: Callable | None = None,
        frame_callback: Callable | None = None,
    ) -> Dict:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {input_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps < 1.0 or fps > 120.0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames_raw = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None
        if max_frames > 0:
            total_frames = min(total_frames, max_frames) if total_frames is not None else max_frames

        video_writer_fourcc = getattr(cv2, "VideoWriter_fourcc", cv2.VideoWriter.fourcc)
        fourcc = video_writer_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            raise ValueError(f"无法创建输出视频: {output_path}")

        stats = {
            "total_frames": total_frames or 0,
            "total_persons": 0,
            "no_helmet_count": 0,
            "helmet_count": 0,
            "total_detection_instances": 0,
            "frames_with_violations": {},
            "detections_per_frame": [],
            "track_summaries": [],
        }

        tracks: Dict[int, Dict] = {}
        next_track_id = 1
        frame_idx = 0

        try:
            while True:
                if total_frames is not None and frame_idx >= total_frames:
                    break
                ret, frame = cap.read()
                if not ret:
                    break

                detections = self.detector.detect(frame)
                frame_persons = [self._normalize_detection(frame, person) for person in detections.get("persons", [])]
                stats["total_detection_instances"] += len(frame_persons)

                matches, unmatched_track_ids, unmatched_detection_ids = self._match_tracks(tracks, frame_persons)

                for track_id, det_idx in matches:
                    detection = frame_persons[det_idx]
                    self._update_track(tracks[track_id], detection, frame_idx)

                for det_idx in unmatched_detection_ids:
                    detection = frame_persons[det_idx]
                    tracks[next_track_id] = self._create_track(next_track_id, detection, frame_idx)
                    stats["total_persons"] += 1
                    if detection["has_helmet"] is False:
                        stats["no_helmet_count"] += 1
                    next_track_id += 1

                for track_id in unmatched_track_ids:
                    track = tracks[track_id]
                    track["missed_frames"] += 1

                expired_track_ids = [
                    track_id
                    for track_id, track in tracks.items()
                    if track["missed_frames"] > self.max_missed_frames
                ]
                for track_id in expired_track_ids:
                    track = tracks.pop(track_id)
                    stats["track_summaries"].append(self._finalize_track(track))

                for track in tracks.values():
                    if track["counted_no_helmet"]:
                        continue
                    if track["has_helmet"] is False:
                        track["counted_no_helmet"] = True
                        stats["no_helmet_count"] += 1

                frame_no_helmet = sum(1 for person in frame_persons if person["has_helmet"] is False)
                frame_helmet = sum(1 for person in frame_persons if person["has_helmet"] is True)
                if frame_no_helmet > 0:
                    stats["frames_with_violations"][frame_idx] = frame_no_helmet

                matched_ids_by_detection = {det_idx: track_id for track_id, det_idx in matches}
                new_track_start = next_track_id - len(unmatched_detection_ids)
                for offset, det_idx in enumerate(unmatched_detection_ids):
                    matched_ids_by_detection[det_idx] = new_track_start + offset

                for det_idx, detection in enumerate(frame_persons):
                    track_id = matched_ids_by_detection.get(det_idx)
                    self._draw_bbox(
                        frame,
                        detection["bbox"],
                        detection["has_helmet"],
                        detection["confidence"],
                        track_id,
                    )

                stats["detections_per_frame"].append(
                    {
                        "frame": frame_idx,
                        "persons": len(frame_persons),
                        "no_helmet": frame_no_helmet,
                        "unique_persons_so_far": stats["total_persons"],
                    }
                )

                self._add_stats_text(
                    frame,
                    frame_idx,
                    len(frame_persons),
                    frame_helmet,
                    frame_no_helmet,
                    total_frames or max(frame_idx + 1, 1),
                    stats["total_persons"],
                    stats["no_helmet_count"],
                )

                if frame_callback is not None:
                    frame_callback(frame, frame_idx + 1, total_frames)

                out.write(frame)

                if progress_callback is not None:
                    progress_callback(frame_idx + 1, total_frames or 0)

                frame_idx += 1
        finally:
            cap.release()
            out.release()

        for track in tracks.values():
            stats["track_summaries"].append(self._finalize_track(track))

        stats["total_frames"] = frame_idx
        stats["helmet_count"] = max(stats["total_persons"] - stats["no_helmet_count"], 0)
        return stats

    def _normalize_detection(self, frame: np.ndarray, detection: Dict) -> Dict:
        bbox = cast(Tuple[int, int, int, int], tuple(int(x) for x in detection["bbox"]))
        has_helmet = detection.get("has_helmet")
        if has_helmet is None:
            has_helmet = self._check_helmet_heuristic(frame, bbox)

        center = self._bbox_center(bbox)
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1], 1)
        normalized = dict(detection)
        normalized.update(
            {
                "bbox": bbox,
                "has_helmet": has_helmet,
                "center": center,
                "size": size,
            }
        )
        return normalized

    def _create_track(self, track_id: int, detection: Dict, frame_idx: int) -> Dict:
        return {
            "track_id": track_id,
            "bbox": detection["bbox"],
            "center": detection["center"],
            "size": detection["size"],
            "last_frame": frame_idx,
            "missed_frames": 0,
            "confidence": detection["confidence"],
            "has_helmet": detection["has_helmet"],
            "counted_no_helmet": detection["has_helmet"] is False,
            "seen_frames": [frame_idx],
        }

    def _update_track(self, track: Dict, detection: Dict, frame_idx: int) -> None:
        track["bbox"] = detection["bbox"]
        track["center"] = detection["center"]
        track["size"] = detection["size"]
        track["last_frame"] = frame_idx
        track["missed_frames"] = 0
        track["confidence"] = max(track["confidence"], detection["confidence"])
        track["seen_frames"].append(frame_idx)
        if detection["has_helmet"] is False:
            track["has_helmet"] = False
        elif track["has_helmet"] is None:
            track["has_helmet"] = detection["has_helmet"]

    def _finalize_track(self, track: Dict) -> Dict:
        return {
            "track_id": track["track_id"],
            "first_frame": min(track["seen_frames"]),
            "last_frame": max(track["seen_frames"]),
            "frames_seen": len(track["seen_frames"]),
            "has_helmet": track["has_helmet"],
            "counted_no_helmet": track["counted_no_helmet"],
            "confidence": track["confidence"],
        }

    def _match_tracks(self, tracks: Dict[int, Dict], detections: List[Dict]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if not tracks or not detections:
            return [], list(tracks.keys()), list(range(len(detections)))

        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in tracks.items():
            if track["missed_frames"] > self.max_missed_frames:
                continue

            for det_idx, detection in enumerate(detections):
                iou = self._bbox_iou(track["bbox"], detection["bbox"])
                center_distance = self._center_distance(track["center"], detection["center"])
                max_distance = max(track["size"], detection["size"], 30) * 0.9 + 30
                if iou < self.match_iou_threshold and center_distance > max_distance:
                    continue

                distance_score = max(0.0, 1.0 - center_distance / max_distance)
                score = iou + distance_score * 0.5
                candidates.append((score, track_id, det_idx))

        matches: List[Tuple[int, int]] = []
        used_tracks = set()
        used_detections = set()
        for _, track_id, det_idx in sorted(candidates, key=lambda item: item[0], reverse=True):
            if track_id in used_tracks or det_idx in used_detections:
                continue
            matches.append((track_id, det_idx))
            used_tracks.add(track_id)
            used_detections.add(det_idx)

        unmatched_track_ids = [track_id for track_id in tracks.keys() if track_id not in used_tracks]
        unmatched_detection_ids = [det_idx for det_idx in range(len(detections)) if det_idx not in used_detections]
        return matches, unmatched_track_ids, unmatched_detection_ids

    def _check_helmet_heuristic(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
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
            dark_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 100]))
            dark_ratio = float(np.sum(dark_mask > 0)) / float(dark_mask.size)
            bright_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([180, 50, 255]))
            bright_ratio = float(np.sum(bright_mask > 0)) / float(bright_mask.size)
            return (dark_ratio > 0.25) or (bright_ratio > 0.2)
        except Exception:
            return False

    def _draw_bbox(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        has_helmet: bool | None,
        confidence: float,
        track_id: int | None,
    ) -> None:
        x1, y1, x2, y2 = [int(x) for x in bbox]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if has_helmet is True:
            color = self.COLORS["person_helmet"]
            state_text = "Helmet"
        elif has_helmet is False:
            color = self.COLORS["person_no_helmet"]
            state_text = "NO Helmet"
        else:
            color = self.COLORS["unknown"]
            state_text = "Unknown"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        prefix = f"ID {track_id}" if track_id is not None else "ID ?"
        label = f"{prefix} | {state_text} {confidence:.2f}"
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            frame,
            (x1, max(0, y1 - label_size[1] - baseline - 5)),
            (x1 + label_size[0] + 5, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    def _add_stats_text(
        self,
        frame: np.ndarray,
        frame_idx: int,
        frame_persons: int,
        frame_helmet: int,
        frame_no_helmet: int,
        total_frames: int,
        unique_persons: int,
        unique_no_helmet: int,
    ) -> None:
        stats_text = [
            f"Frame: {frame_idx + 1}/{max(total_frames, 1)}",
            f"Current Riders: {frame_persons}",
            f"Current Helmet Riders: {frame_helmet}",
            f"Current No Helmet Riders: {frame_no_helmet}",
            f"Unique Riders: {unique_persons}",
            f"Unique No Helmet Riders: {unique_no_helmet}",
        ]

        y_offset = 30
        for idx, text in enumerate(stats_text):
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                frame,
                (5, y_offset + idx * 25 - text_size[1] - 5),
                (10 + text_size[0], y_offset + idx * 25 + 5),
                (0, 0, 0),
                -1,
            )
            color = (0, 0, 255) if "No Helmet" in text and (frame_no_helmet > 0 or unique_no_helmet > 0) else (0, 255, 0)
            cv2.putText(
                frame,
                text,
                (10, y_offset + idx * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    @staticmethod
    def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox
        return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)

    @staticmethod
    def _center_distance(center_a: Tuple[float, float], center_b: Tuple[float, float]) -> float:
        return float(np.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1]))

    @staticmethod
    def _bbox_iou(bbox_a: Tuple[int, int, int, int], bbox_b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area == 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0
        return float(inter_area) / float(union_area)

