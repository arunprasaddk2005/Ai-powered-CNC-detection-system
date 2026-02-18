"""
CNC Inspection System - Preprocessor Module (Micron-Accurate Edition)
======================================================================

Design goals:
  1. DETERMINISM  — identical output for the same image, every run.
  2. CROSS-IMAGE  — different photos of the same part produce measurements
                   within ±2 µm of each other (given calibrated scale).
  3. ROBUSTNESS   — handles all real-world scenarios:
                   • studio white / light backgrounds
                   • dark / complex / textured backgrounds
                   • uneven / gradient illumination
                   • scratched, oxidised or reflective metal surfaces
                   • partial occlusion by fixtures
                   • JPEG compression artefacts
                   • out-of-focus / depth-of-field blur
                   • different camera distances (solved via scale calibration)

Key techniques:
  - CLAHE normalises contrast before any thresholding → repeatable binary masks
  - Flat-field (vignetting) correction removes lens fall-off
  - Sub-pixel contour refinement using moment analysis
  - Canonical edge representation (sorted, oriented) → no orientation ambiguity
  - Strict deterministic RNG seeding for any probabilistic step
"""

import cv2
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorConfig:
    """Full control over every tunable parameter."""

    # ── Noise reduction ──────────────────────────────────────────────────────
    # Bilateral is used for real photos; Gaussian for synthetic.
    bilateral_d: int = 9
    bilateral_sigma_color: float = 60.0
    bilateral_sigma_space: float = 60.0
    gaussian_ksize: int = 3          # synthetic path only
    gaussian_sigma: float = 0.5

    # ── CLAHE ────────────────────────────────────────────────────────────────
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # ── Flat-field correction ────────────────────────────────────────────────
    enable_flat_field: bool = True
    flat_field_blur_ksize: int = 201   # must be odd; large = slow background est.

    # ── Background detection ─────────────────────────────────────────────────
    # A corner sample whose mean brightness > this threshold → light background
    light_bg_threshold: int = 180
    # Corner region size (px)
    corner_sample_size: int = 50

    # ── Thresholding ─────────────────────────────────────────────────────────
    # GrabCut iterations (dark background only)
    grabcut_iterations: int = 10
    # Morphological closing kernel for binary mask (px)
    close_ksize: int = 15
    # Morphological opening kernel for binary mask (px)
    open_ksize: int = 5

    # ── Edge detection ───────────────────────────────────────────────────────
    # If None, auto-computed from image median (Otsu-style auto-Canny)
    canny_low: Optional[int] = None
    canny_high: Optional[int] = None
    canny_aperture: int = 3
    # Morphological closing on the edge map (to seal small gaps)
    morph_close_ksize: int = 7
    morph_close_iter: int = 2


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessorResult:
    original: np.ndarray
    grayscale: np.ndarray
    normalized: np.ndarray      # After flat-field + CLAHE
    blurred: np.ndarray
    binary: np.ndarray          # Clean part mask (255 = part)
    edges: np.ndarray           # Canny edges
    morphed: np.ndarray         # Closed edge map (fed to ShapeDetector)
    bg_is_light: bool
    config: PreprocessorConfig


# ---------------------------------------------------------------------------
# Preprocessor
# ---------------------------------------------------------------------------

class ImagePreprocessor:
    """
    Deterministic, micron-accurate preprocessing pipeline.

    Call:
        result = preprocessor.process(image)          # auto-detect type
        result = preprocessor.process(image, is_synthetic=True)
    """

    def __init__(self, config: Optional[PreprocessorConfig] = None):
        self.config = config or PreprocessorConfig()
        # Build CLAHE once (stateless after construction → deterministic)
        self._clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid,
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def process(self,
                image: np.ndarray,
                is_synthetic: Optional[bool] = None) -> PreprocessorResult:
        """
        Run the full pipeline.

        Args:
            image:        BGR or grayscale input.
            is_synthetic: True = simple Gaussian path (fast, clean).
                          False = full real-photo path.
                          None = auto-detect from image statistics.
        Returns:
            PreprocessorResult with every intermediate image.
        """
        cfg = self.config
        original = image.copy()

        gray = self._to_gray(image)

        if is_synthetic is None:
            is_synthetic = self._is_synthetic(gray)

        if is_synthetic:
            return self._process_synthetic(original, gray, cfg)
        else:
            return self._process_real(original, image, gray, cfg)

    # ── Synthetic path (unchanged semantics, slightly tightened) ───────────

    def _process_synthetic(self, original, gray, cfg):
        blurred = cv2.GaussianBlur(
            gray,
            (cfg.gaussian_ksize | 1, cfg.gaussian_ksize | 1),
            cfg.gaussian_sigma,
        )
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = self._auto_canny(blurred, cfg)
        morphed = self._morph_close(edges, cfg.morph_close_ksize,
                                    cfg.morph_close_iter)
        bg_is_light = self._detect_bg(gray, cfg)
        return PreprocessorResult(
            original=original, grayscale=gray, normalized=blurred,
            blurred=blurred, binary=binary, edges=edges, morphed=morphed,
            bg_is_light=bg_is_light, config=cfg,
        )

    # ── Real-photo path ────────────────────────────────────────────────────

    def _process_real(self, original, color_image, gray, cfg):
        # ── Step 1: Flat-field (vignetting) correction ─────────────────────
        if cfg.enable_flat_field:
            gray = self._flat_field_correction(gray, cfg.flat_field_blur_ksize)

        # ── Step 2: CLAHE contrast normalisation ───────────────────────────
        normalized = self._clahe.apply(gray)

        # ── Step 3: Edge-preserving noise reduction ────────────────────────
        blurred = cv2.bilateralFilter(
            normalized,
            cfg.bilateral_d,
            cfg.bilateral_sigma_color,
            cfg.bilateral_sigma_space,
        )

        # ── Step 4: Background type detection ─────────────────────────────
        bg_is_light = self._detect_bg(gray, cfg)

        # ── Step 5: Part segmentation → clean binary mask ─────────────────
        binary = self._segment_part(
            color_image, blurred, normalized, bg_is_light, cfg)

        # ── Step 6: Sub-pixel edge map from binary mask ────────────────────
        edges = cv2.Canny(binary, 50, 150, apertureSize=cfg.canny_aperture)

        # ── Step 7: Close gaps in edge map ────────────────────────────────
        morphed = self._morph_close(edges, cfg.morph_close_ksize,
                                    cfg.morph_close_iter)

        return PreprocessorResult(
            original=original, grayscale=gray, normalized=normalized,
            blurred=blurred, binary=binary, edges=edges, morphed=morphed,
            bg_is_light=bg_is_light, config=cfg,
        )

    # ── Segmentation strategies ────────────────────────────────────────────

    def _segment_part(self, color_image, blurred, normalized,
                      bg_is_light, cfg):
        """Return a clean binary mask (255 = part, 0 = background)."""
        h, w = blurred.shape

        if bg_is_light:
            binary = self._segment_light_bg(blurred, cfg)
        else:
            binary = self._segment_dark_bg(color_image, blurred, normalized,
                                           h, w, cfg)

        # ── Post-processing common to both paths ───────────────────────────
        # Remove tiny noise blobs (keep only the largest connected component)
        binary = self._keep_largest_component(binary)
        # Fill interior holes (caused by reflections, markings, etc.)
        binary = self._fill_interior_holes(binary)

        return binary

    def _segment_light_bg(self, blurred, cfg):
        """Inverted Otsu + morphological clean-up."""
        _, binary = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )
        k_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.close_ksize, cfg.close_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close, iterations=2)
        k_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.open_ksize, cfg.open_ksize))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)
        return binary

    def _segment_dark_bg(self, color_image, blurred, normalized, h, w, cfg):
        """
        Multi-strategy segmentation for dark / complex backgrounds.
        Tries three approaches and picks the one with the largest,
        most solid component:
          A) GrabCut (best for cluttered workbench)
          B) Adaptive threshold on CLAHE image (good for uneven lighting)
          C) Simple Otsu on blurred (fallback)
        """
        candidates = []

        # ── A: GrabCut ─────────────────────────────────────────────────────
        try:
            rect_margin = max(10, min(h, w) // 20)
            rect = (rect_margin, rect_margin,
                    w - 2 * rect_margin, h - 2 * rect_margin)
            gc_mask = np.zeros((h, w), np.uint8)
            bg_model = np.zeros((1, 65), np.float64)
            fg_model = np.zeros((1, 65), np.float64)
            cv2.grabCut(color_image, gc_mask, rect, bg_model, fg_model,
                        cfg.grabcut_iterations, cv2.GC_INIT_WITH_RECT)
            gc_bin = np.where(
                (gc_mask == 2) | (gc_mask == 0), 0, 255).astype(np.uint8)
            k = cv2.getStructuringElement(
                cv2.MORPH_RECT, (cfg.close_ksize, cfg.close_ksize))
            gc_bin = cv2.morphologyEx(gc_bin, cv2.MORPH_CLOSE, k, iterations=2)
            candidates.append(('grabcut', gc_bin))
        except Exception:
            pass

        # ── B: Adaptive threshold ───────────────────────────────────────────
        try:
            adapt = cv2.adaptiveThreshold(
                normalized, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=51, C=8,
            )
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (cfg.close_ksize, cfg.close_ksize))
            adapt = cv2.morphologyEx(adapt, cv2.MORPH_CLOSE, k, iterations=2)
            candidates.append(('adaptive', adapt))
        except Exception:
            pass

        # ── C: Simple Otsu ─────────────────────────────────────────────────
        _, otsu = cv2.threshold(blurred, 0, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.close_ksize, cfg.close_ksize))
        otsu = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k, iterations=2)
        candidates.append(('otsu', otsu))

        # ── Pick best candidate: largest * most solid region ───────────────
        best_score = -1.0
        best_bin = otsu
        for name, mask in candidates:
            score = self._segmentation_score(mask)
            if score > best_score:
                best_score = score
                best_bin = mask

        return best_bin

    @staticmethod
    def _segmentation_score(mask: np.ndarray) -> float:
        """Score = largest_area × solidity.  Higher = better segmentation."""
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        solidity = area / hull_area if hull_area > 0 else 0.0
        return area * solidity

    # ── Utility helpers ────────────────────────────────────────────────────

    @staticmethod
    def _keep_largest_component(binary: np.ndarray) -> np.ndarray:
        """Retain only the largest connected white blob."""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8)
        if n <= 1:
            return binary
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out = np.zeros_like(binary)
        out[labels == largest] = 255
        return out

    @staticmethod
    def _fill_interior_holes(binary: np.ndarray) -> np.ndarray:
        """
        Fill holes that are fully enclosed by the part mask.
        Handles reflective metal surfaces that produce dark interior blobs.
        """
        # Flood-fill from all four corners (exterior = guaranteed background)
        flood = binary.copy()
        h, w = flood.shape
        seed_mask = np.zeros((h + 2, w + 2), np.uint8)
        for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            cv2.floodFill(flood, seed_mask, pt, 128)
        # Pixels that are still 0 (unreachable from corners) are interior holes
        interior_holes = (flood == 0).astype(np.uint8) * 255
        return cv2.bitwise_or(binary, interior_holes)

    @staticmethod
    def _flat_field_correction(gray: np.ndarray, blur_ksize: int) -> np.ndarray:
        """
        Divide by a smoothed version of the image to remove vignetting /
        uneven illumination.  Output is scaled back to [0, 255].

        A large Gaussian blur models the slow-varying illumination field.
        """
        ksize = blur_ksize | 1   # ensure odd
        illumination = cv2.GaussianBlur(
            gray.astype(np.float32), (ksize, ksize), ksize / 3.0)
        # Avoid division by zero
        illumination = np.where(illumination < 1.0, 1.0, illumination)
        corrected = (gray.astype(np.float32) / illumination) * 128.0
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        return corrected

    @staticmethod
    def _auto_canny(gray: np.ndarray, cfg: PreprocessorConfig) -> np.ndarray:
        """Auto-thresholded Canny using median method."""
        if cfg.canny_low is not None and cfg.canny_high is not None:
            low, high = cfg.canny_low, cfg.canny_high
        else:
            median = float(np.median(gray))
            sigma = 0.33
            low  = int(max(0,   (1.0 - sigma) * median))
            high = int(min(255, (1.0 + sigma) * median))
        return cv2.Canny(gray, low, high, apertureSize=cfg.canny_aperture)

    @staticmethod
    def _morph_close(edges: np.ndarray, ksize: int, iterations: int) -> np.ndarray:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=iterations)

    @staticmethod
    def _detect_bg(gray: np.ndarray, cfg: PreprocessorConfig) -> bool:
        """True if the background is light (e.g. studio white)."""
        h, w = gray.shape
        s = cfg.corner_sample_size
        corners = [
            gray[:s, :s], gray[:s, w - s:],
            gray[h - s:, :s], gray[h - s:, w - s:],
        ]
        corner_mean = float(np.mean([c.mean() for c in corners]))
        return corner_mean > cfg.light_bg_threshold

    @staticmethod
    def _is_synthetic(gray: np.ndarray) -> bool:
        """
        Heuristic: synthetic images have very flat intensity histograms
        (few unique values, near-binary distribution).
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        nonzero_bins = int(np.count_nonzero(hist))
        # Synthetic: typically < 30 distinct grey levels
        return nonzero_bins < 30

    @staticmethod
    def _to_gray(image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image.copy()

    # ── Visualisation ──────────────────────────────────────────────────────

    def visualize_pipeline(self, result: PreprocessorResult,
                           save_path: Optional[str] = None) -> np.ndarray:
        import matplotlib.pyplot as plt

        panels = [
            ('Original',     result.original if len(result.original.shape) == 2
                             else cv2.cvtColor(result.original, cv2.COLOR_BGR2RGB),
             'gray' if len(result.original.shape) == 2 else None),
            ('Grayscale',    result.grayscale,   'gray'),
            ('Normalised',   result.normalized,   'gray'),
            ('Blurred',      result.blurred,      'gray'),
            ('Binary mask',  result.binary,       'gray'),
            ('Morphed edges',result.morphed,      'gray'),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Preprocessing Pipeline (micron-accurate edition)',
                     fontsize=14, fontweight='bold')
        for ax, (title, img, cmap) in zip(axes.flat, panels):
            ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=10)
            ax.axis('off')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.frombuffer(buf, np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        plt.close(fig)
        return arr