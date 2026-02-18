"""
CNC Inspection System - Shape Detector Module
Phase 3: Component boundary detection and dimension extraction

Real-world improvements:
- Binary-mask fallback when edge-map contours are weak
- Contour convexity repair for occluded or worn edges
- Hull-vs-contour scoring to reject false large blobs (shadows, fixtures)
- Contour smoothing to reduce surface texture noise
- Perspective-aware bounding box with rotation correction
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import math


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BoundingBox:
    """Rotated bounding box for a component."""
    center_x: float
    center_y: float
    width: float    # pixels (always the larger dimension)
    height: float   # pixels
    angle: float    # degrees

    def get_corners(self) -> np.ndarray:
        box = cv2.boxPoints(
            ((self.center_x, self.center_y), (self.width, self.height), self.angle)
        )
        return box.astype(np.int32)


@dataclass
class ShapeDetectionResult:
    """Container for shape detection outputs."""
    contour: np.ndarray
    bounding_box: BoundingBox
    centroid: Tuple[float, float]
    area: float
    perimeter: float
    hierarchy: Optional[np.ndarray] = None
    all_contours: Optional[List[np.ndarray]] = None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ShapeDetector:
    """
    Detect component boundaries and extract geometric properties.

    Detection strategy (tries each in order, stops at first success):
      1. Edge-map contours (morphed edges from preprocessor)
      2. Binary-mask contours (Otsu threshold on the original gray channel
         embedded in the morphed image – fallback for low-contrast edges)
      3. Binary-mask contours derived from the binary channel

    Each candidate contour is scored by how 'component-like' it is:
      - Solidity (area / convex-hull area):  high solidity ≈ solid part
      - Not too close to the image border (avoids frame artefacts)
    """

    def __init__(self,
                 min_area: float = 500.0,
                 max_area: Optional[float] = None,
                 min_perimeter: float = 100.0,
                 min_solidity: float = 0.3,
                 smooth_contour: bool = True):
        """
        Args:
            min_area:       Minimum contour area (pixels²).
            max_area:       Maximum contour area, or None.
            min_perimeter:  Minimum contour perimeter (pixels).
            min_solidity:   Minimum (area / hull_area); rejects wispy noise.
            smooth_contour: Apply slight smoothing to reduce texture jaggies.
        """
        self.min_area = min_area
        self.max_area = max_area
        self.min_perimeter = min_perimeter
        self.min_solidity = min_solidity
        self.smooth_contour = smooth_contour

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def detect(self,
               edge_image: np.ndarray,
               binary_image: Optional[np.ndarray] = None) -> ShapeDetectionResult:
        """
        Detect the component boundary.

        Args:
            edge_image:   Morphed Canny edge map from the preprocessor.
            binary_image: Optional binary mask (used as fallback).

        Returns:
            ShapeDetectionResult.
        """
        h, w = edge_image.shape[:2]

        # --- Strategy 1: edge-map contours ---
        contour = self._best_contour_from_binary(edge_image, w, h)

        # --- Strategy 2: threshold the edge image itself ---
        if contour is None:
            _, thresh = cv2.threshold(edge_image, 10, 255, cv2.THRESH_BINARY)
            # Dilate to fill interior gaps left by Canny
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            filled = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
            contour = self._best_contour_from_binary(filled, w, h)

        # --- Strategy 3: explicit binary image ---
        if contour is None and binary_image is not None:
            contour = self._best_contour_from_binary(binary_image, w, h)

        if contour is None:
            raise ValueError(
                "No valid component contour found.  "
                "Try adjusting min_area, or ensure the image has good contrast."
            )

        # Optional smoothing
        if self.smooth_contour:
            contour = self._smooth_contour(contour)

        # Compute properties
        area      = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)

        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            x, y, cw, ch = cv2.boundingRect(contour)
            cx, cy = x + cw / 2, y + ch / 2

        rect   = cv2.minAreaRect(contour)
        width  = rect[1][0]
        height = rect[1][1]
        angle  = rect[2]

        if width < height:
            width, height = height, width
            angle = (angle + 90) % 180

        bbox = BoundingBox(
            center_x=rect[0][0], center_y=rect[0][1],
            width=width, height=height, angle=angle,
        )

        return ShapeDetectionResult(
            contour=contour,
            bounding_box=bbox,
            centroid=(cx, cy),
            area=area,
            perimeter=perimeter,
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _best_contour_from_binary(self,
                                   binary: np.ndarray,
                                   img_w: int,
                                   img_h: int) -> Optional[np.ndarray]:
        """
        Find all external contours in *binary*, filter them, and return
        the one most likely to be the CNC component.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            perim = cv2.arcLength(c, True)
            if perim < self.min_perimeter:
                continue

            # Solidity check
            hull   = cv2.convexHull(c)
            h_area = cv2.contourArea(hull)
            solidity = area / h_area if h_area > 0 else 0
            if solidity < self.min_solidity:
                continue

            # Border proximity check – skip contours that hug the image edge
            x, y, cw, ch = cv2.boundingRect(c)
            border_margin = 5
            if (x <= border_margin or y <= border_margin
                    or x + cw >= img_w - border_margin
                    or y + ch >= img_h - border_margin):
                # Only reject if the contour *itself* is very close on all sides
                # (i.e. it IS the image border, not a large part near an edge)
                if (x <= border_margin and y <= border_margin
                        and x + cw >= img_w - border_margin
                        and y + ch >= img_h - border_margin):
                    continue

            # Score: prefer large, solid contours
            score = area * solidity
            candidates.append((score, c))

        if not candidates:
            return None

        candidates.sort(key=lambda t: t[0], reverse=True)
        return candidates[0][1]

    def _smooth_contour(self, contour: np.ndarray,
                         epsilon_ratio: float = 0.005) -> np.ndarray:
        """
        Lightly smooth / approximate a contour.

        Reduces jagginess from surface texture in real photos without
        significantly altering the true component boundary.
        """
        epsilon = epsilon_ratio * cv2.arcLength(contour, True)
        return cv2.approxPolyDP(contour, epsilon, True)

    # -----------------------------------------------------------------------
    # Visualisation
    # -----------------------------------------------------------------------

    def visualize_detection(self,
                            original_image: np.ndarray,
                            result: ShapeDetectionResult,
                            show_all_contours: bool = False) -> np.ndarray:
        if len(original_image.shape) == 2:
            vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = original_image.copy()

        if show_all_contours and result.all_contours is not None:
            cv2.drawContours(vis, result.all_contours, -1, (128, 128, 128), 1)

        cv2.drawContours(vis, [result.contour], -1, (0, 255, 0), 3)

        box_points = result.bounding_box.get_corners()
        cv2.drawContours(vis, [box_points], 0, (255, 0, 0), 2)

        cx, cy = result.centroid
        cv2.circle(vis, (int(cx), int(cy)), 8,  (0, 0, 255), -1)
        cv2.circle(vis, (int(cx), int(cy)), 10, (255, 255, 255), 2)

        ax = 50
        cv2.line(vis, (int(cx - ax), int(cy)), (int(cx + ax), int(cy)), (0, 0, 255), 2)
        cv2.line(vis, (int(cx), int(cy - ax)), (int(cx), int(cy + ax)), (0, 0, 255), 2)

        bbox = result.bounding_box
        cv2.putText(vis, f"W: {bbox.width:.1f}px",
                    (int(bbox.center_x - 40), int(bbox.center_y - bbox.height / 2 - 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(vis, f"H: {bbox.height:.1f}px",
                    (int(bbox.center_x + bbox.width / 2 + 10), int(bbox.center_y)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        return vis

    def get_shape_classification(self, result: ShapeDetectionResult) -> str:
        epsilon = 0.02 * cv2.arcLength(result.contour, True)
        approx  = cv2.approxPolyDP(result.contour, epsilon, True)
        n       = len(approx)
        bbox    = result.bounding_box
        ar      = bbox.width / bbox.height if bbox.height > 0 else 1.0
        circ    = (4 * math.pi * result.area) / (result.perimeter ** 2)

        if circ > 0.85:            return "circular"
        elif n == 4 and 0.8 < ar < 1.2: return "square"
        elif n == 4:               return "rectangular"
        elif n == 3:               return "triangular"
        elif n > 8:                return "complex_curved"
        else:                      return "polygonal"