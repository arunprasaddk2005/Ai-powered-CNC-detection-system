"""
CNC Inspection System - Hole Detector Module
Phase 4: Bolt hole detection using Hough Circle Transform

This module detects circular holes in CNC components and extracts:
- Hole center positions (x, y)
- Hole radii/diameters
- Filtering within component boundary
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math


@dataclass
class DetectedHole:
    """Single detected hole"""
    center_x: float
    center_y: float
    radius: float
    
    @property
    def diameter(self) -> float:
        return self.radius * 2
    
    def distance_to(self, other: 'DetectedHole') -> float:
        """Compute distance to another hole"""
        dx = self.center_x - other.center_x
        dy = self.center_y - other.center_y
        return math.sqrt(dx*dx + dy*dy)


@dataclass
class HoleDetectionResult:
    """Container for hole detection outputs"""
    holes: List[DetectedHole]
    
    def get_hole_count(self) -> int:
        return len(self.holes)
    
    def get_spacings(self) -> List[Tuple[int, int, float]]:
        """
        Get all pairwise hole spacings
        
        Returns:
            List of (hole_i, hole_j, distance) tuples
        """
        spacings = []
        for i in range(len(self.holes)):
            for j in range(i + 1, len(self.holes)):
                dist = self.holes[i].distance_to(self.holes[j])
                spacings.append((i, j, dist))
        return spacings


class HoleDetector:
    """
    Detect circular holes using Hough Circle Transform
    """
    
    def __init__(self,
                 min_radius: int = 10,
                 max_radius: int = 100,
                 param1: int = 50,
                 param2: int = 30,
                 min_dist: int = 20):
        """
        Initialize hole detector
        
        Args:
            min_radius: Minimum hole radius in pixels
            max_radius: Maximum hole radius in pixels
            param1: Canny edge threshold (higher = fewer circles)
            param2: Accumulator threshold (lower = more circles)
            min_dist: Minimum distance between circle centers
        """
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.param1 = param1
        self.param2 = param2
        self.min_dist = min_dist
    
    def detect(self,
               blurred_image: np.ndarray,
               component_contour: Optional[np.ndarray] = None,
               binary_mask: Optional[np.ndarray] = None,
               bg_is_light: Optional[bool] = None) -> HoleDetectionResult:
        """
        Detect holes using a strategy appropriate for the image type.

        For real photos, uses a two-pass approach:
          Pass 1 — Connected-component analysis on a strict dark threshold (catches
                   clean, uniformly dark holes).
          Pass 2 — HoughCircles on a CLAHE-enhanced inverted binary (catches dim
                   holes that are dirty/rusted and not uniformly dark).

        For synthetic images, uses the original HoughCircles on the blurred grayscale.

        Args:
            blurred_image: Preprocessed grayscale image (after Gaussian blur)
            component_contour: Optional contour to filter holes within boundary
            binary_mask: Optional clean binary part mask (signals real-photo mode)
            bg_is_light: Optional pre-computed background type flag. When None,
                         auto-detected from the blurred image.  Pass the value
                         from ImagePreprocessor to get the most reliable result.
        
        Returns:
            HoleDetectionResult with all detected holes
        """
        if binary_mask is not None:
            return self._detect_real_photo(
                blurred_image, component_contour, binary_mask, bg_is_light)
        else:
            return self._detect_synthetic(blurred_image, component_contour)

    def _detect_real_photo(self,
                           gray: np.ndarray,
                           part_contour: Optional[np.ndarray],
                           part_mask: np.ndarray,
                           bg_is_light: Optional[bool] = None) -> 'HoleDetectionResult':
        """
        Background-adaptive hole detection for real photos.

        Light background (e.g. studio white): holes appear as white gaps in
        the inverted-Otsu part mask. Uses a hull-gap method: computes the
        convex hull of the part, then finds interior gaps as candidate holes.

        Dark/complex background (e.g. wooden table): holes are dark regions
        within the part. Uses a two-pass CC + HoughCircles approach.
        """
        import cv2 as _cv2
        h, w = gray.shape

        # --- Determine background type ---
        if bg_is_light is None:
            # Fall back to analysing the (blurred) grayscale corners
            corners = [gray[0:40, 0:40], gray[0:40, w-40:],
                       gray[h-40:, 0:40], gray[h-40:, w-40:]]
            corner_mean = float(np.mean([c.mean() for c in corners]))
            centre_mean = float(gray[h//4:3*h//4, w//4:3*w//4].mean())
            bg_is_light = corner_mean > (centre_mean + 40)  # require clear separation

        if bg_is_light:
            return self._detect_holes_hull_gap(gray, part_contour, h, w)
        else:
            return self._detect_holes_cc_hough(gray, part_contour, part_mask, h, w)

    def _detect_holes_hull_gap(self, gray, part_contour, h, w):
        """
        Light-background method: holes = gaps inside the part's convex hull.
        Works when holes appear as bright (background shows through) voids.
        """
        import cv2 as _cv2

        # Raw binary without morphological fill (so holes aren't filled in)
        blur = _cv2.bilateralFilter(gray, 9, 75, 75)
        _, binary_raw = _cv2.threshold(
            blur, 0, 255, _cv2.THRESH_BINARY_INV + _cv2.THRESH_OTSU)

        # Convex hull of the part
        hull = _cv2.convexHull(part_contour)
        hull_mask = np.zeros((h, w), np.uint8)
        _cv2.drawContours(hull_mask, [hull], -1, 255, -1)

        # Remove the background (flood-fill from corners)
        flood = binary_raw.copy()
        for pt in [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]:
            _cv2.floodFill(flood, None, pt, 128)
        bg_mask = np.where(flood == 128, 255, 0).astype(np.uint8)

        # Holes = inside hull & not part & not background
        hole_mask = _cv2.bitwise_and(
            hull_mask, _cv2.bitwise_not(binary_raw))
        hole_mask = _cv2.bitwise_and(
            hole_mask, _cv2.bitwise_not(bg_mask))

        # Small morphological open to remove noise
        kernel = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE, (5, 5))
        hole_mask = _cv2.morphologyEx(
            hole_mask, _cv2.MORPH_OPEN, kernel, iterations=1)

        # HoughCircles on the clean hole mask
        blurred_hm = _cv2.GaussianBlur(hole_mask, (9, 9), 2)
        circles = _cv2.HoughCircles(
            blurred_hm, _cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=15, minRadius=10, maxRadius=120)

        holes = []
        if circles is not None:
            for c in np.round(circles[0, :]).astype(int):
                cx, cy, cr = c
                if part_contour is not None:
                    inside = _cv2.pointPolygonTest(
                        part_contour, (float(cx), float(cy)), False)
                    if inside < -(cr * 0.5):
                        continue
                holes.append(DetectedHole(float(cx), float(cy), float(cr)))
        return HoleDetectionResult(holes=holes)

    def _detect_holes_cc_hough(self, gray, part_contour, part_mask, h, w):
        """
        Dark/complex background method: two-pass connected-component + HoughCircles.
        Pass 1 — strict threshold catches uniformly dark holes.
        Pass 2 — CLAHE + loose threshold catches dim/rusted holes.
        """
        import cv2 as _cv2
        all_holes: List[DetectedHole] = []

        # Pass 1: connected-component analysis on strict threshold
        _, dark_strict = _cv2.threshold(gray, 50, 255, _cv2.THRESH_BINARY_INV)
        dark_strict = _cv2.bitwise_and(dark_strict, part_mask)
        num_l, lbls, sts, cents = _cv2.connectedComponentsWithStats(
            dark_strict, connectivity=8)

        for i in range(1, num_l):
            area = sts[i][4]
            if area < 200 or area > 10000:
                continue
            cx_c, cy_c = int(cents[i][0]), int(cents[i][1])
            comp = (lbls == i).astype(np.uint8) * 255
            cnts2, _ = _cv2.findContours(
                comp, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
            if not cnts2:
                continue
            perim = _cv2.arcLength(cnts2[0], True)
            if perim == 0:
                continue
            circ = 4 * math.pi * area / (perim ** 2)
            if circ > 0.70:
                r = int(math.sqrt(area / math.pi))
                all_holes.append(DetectedHole(float(cx_c), float(cy_c), float(r)))

        # Pass 2: CLAHE + HoughCircles for dim/dirty holes
        clahe = _cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, dark_loose = _cv2.threshold(enhanced, 80, 255, _cv2.THRESH_BINARY_INV)
        dark_loose = _cv2.bitwise_and(dark_loose, part_mask)
        blurred_loose = _cv2.GaussianBlur(dark_loose, (9, 9), 2)
        circles = _cv2.HoughCircles(
            blurred_loose, _cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=50, param2=20, minRadius=8, maxRadius=22)

        if circles is not None:
            for c in np.round(circles[0, :]).astype(int):
                cx, cy, cr = c
                if part_contour is not None:
                    inside = _cv2.pointPolygonTest(
                        part_contour, (float(cx), float(cy)), False)
                    if inside < 0:
                        continue
                sample = np.zeros((h, w), np.uint8)
                _cv2.circle(sample, (cx, cy), max(1, cr - 2), 255, -1)
                pixels = gray[sample > 0]
                if len(pixels) == 0 or pixels.min() > 55:
                    continue
                tiny = np.zeros((h, w), np.uint8)
                _cv2.circle(tiny, (cx, cy), max(1, cr - 3), 255, -1)
                local_dark = _cv2.bitwise_and(dark_loose, tiny)
                denom = math.pi * max(1, cr - 3) ** 2
                dark_frac = float(local_dark.sum() // 255) / denom
                if dark_frac < 0.5:
                    continue
                too_close = any(
                    math.sqrt((cx - h2.center_x) ** 2 + (cy - h2.center_y) ** 2)
                    < max(cr, h2.radius) * 1.2
                    for h2 in all_holes)
                if not too_close:
                    all_holes.append(DetectedHole(float(cx), float(cy), float(cr)))

        return HoleDetectionResult(holes=all_holes)

    def _detect_synthetic(self,
                          blurred_image: np.ndarray,
                          component_contour: Optional[np.ndarray]) -> 'HoleDetectionResult':
        """Original HoughCircles approach for synthetic images."""
        circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=self.min_dist,
            param1=self.param1,
            param2=self.param2,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        detected_holes = []
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                if component_contour is not None:
                    dist = cv2.pointPolygonTest(
                        component_contour, (float(x), float(y)), False)
                    if dist < 0:
                        continue
                detected_holes.append(DetectedHole(
                    center_x=float(x),
                    center_y=float(y),
                    radius=float(r)
                ))
        return HoleDetectionResult(holes=detected_holes)
    
    def visualize_detection(self,
                           original_image: np.ndarray,
                           result: HoleDetectionResult,
                           show_labels: bool = True) -> np.ndarray:
        """
        Create visualization of hole detection
        
        Args:
            original_image: Original image
            result: HoleDetectionResult
            show_labels: Show hole IDs and dimensions
        
        Returns:
            Annotated image
        """
        if len(original_image.shape) == 2:
            vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = original_image.copy()
        
        for i, hole in enumerate(result.holes):
            x = int(hole.center_x)
            y = int(hole.center_y)
            r = int(hole.radius)
            
            # Draw circle outline
            cv2.circle(vis, (x, y), r, (0, 255, 255), 2)  # Yellow
            
            # Draw center point
            cv2.circle(vis, (x, y), 3, (0, 0, 255), -1)  # Red
            cv2.circle(vis, (x, y), 5, (255, 255, 255), 1)  # White border
            
            if show_labels:
                # Hole ID
                cv2.putText(vis, f"#{i}",
                           (x - 15, y - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(vis, f"#{i}",
                           (x - 15, y - r - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Diameter
                diameter_text = f"D:{hole.diameter:.1f}px"
                cv2.putText(vis, diameter_text,
                           (x - 30, y + r + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                cv2.putText(vis, diameter_text,
                           (x - 30, y + r + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Summary text
        summary = f"Holes detected: {result.get_hole_count()}"
        cv2.putText(vis, summary, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
        cv2.putText(vis, summary, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        return vis


def demo_hole_detection():
    """Demo: Detect holes in all sample images"""
    import os
    from preprocessor import ImagePreprocessor
    from shape_detector import ShapeDetector
    
    print("=" * 70)
    print("PHASE 4: HOLE DETECTOR DEMO")
    print("=" * 70)
    
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir):
        print(f"\n❌ Error: {sample_dir}/ not found!")
        return
    
    output_dir = "hole_detection_output"
    os.makedirs(output_dir, exist_ok=True)
    
    sample_files = [
        "sample_rectangle.png",
        "sample_flange.png",
        "sample_l_bracket.png",
        "sample_custom_polygon.png"
    ]
    
    preprocessor = ImagePreprocessor()
    shape_detector = ShapeDetector(min_area=1000)
    hole_detector = HoleDetector(min_radius=15, max_radius=80, param2=25)
    
    for sample_file in sample_files:
        filepath = os.path.join(sample_dir, sample_file)
        
        if not os.path.exists(filepath):
            continue
        
        print(f"\n{'─' * 70}")
        print(f"Processing: {sample_file}")
        print(f"{'─' * 70}")
        
        # Load and preprocess
        image = cv2.imread(filepath)
        preprocess_result = preprocessor.process(image, is_synthetic=True)
        
        # Detect shape
        shape_result = shape_detector.detect(preprocess_result.morphed)
        
        # Detect holes
        hole_result = hole_detector.detect(
            preprocess_result.blurred,
            component_contour=shape_result.contour
        )
        
        print(f"✓ Holes detected: {hole_result.get_hole_count()}")
        
        for i, hole in enumerate(hole_result.holes):
            print(f"  Hole #{i}: Center=({hole.center_x:.1f}, {hole.center_y:.1f}), "
                  f"Diameter={hole.diameter:.1f}px")
        
        # Compute spacings
        spacings = hole_result.get_spacings()
        if spacings:
            print(f"  Hole spacings:")
            for i, j, dist in spacings[:5]:  # Show first 5
                print(f"    Hole {i} ↔ Hole {j}: {dist:.1f}px")
            if len(spacings) > 5:
                print(f"    ... and {len(spacings) - 5} more")
        
        # Visualize
        vis = hole_detector.visualize_detection(image, hole_result)
        
        base_name = os.path.splitext(sample_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}_holes_detected.png")
        cv2.imwrite(output_path, vis)
        print(f"✓ Visualization saved: {output_path}")
    
    print(f"\n{'=' * 70}")
    print("✅ HOLE DETECTION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    demo_hole_detection()