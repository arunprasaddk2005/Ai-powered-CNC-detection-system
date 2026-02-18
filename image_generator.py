"""
CNC Inspection System - Synthetic Image Generator
Phase 1: Generate test images for all component shapes

This module creates synthetic top-down images of CNC-machined components
with configurable dimensions, hole patterns, and scale references.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math


@dataclass
class ComponentSpec:
    """Specification for a synthetic component image"""
    shape_type: str  # 'rectangle', 'flange', 'l_bracket', 'custom'
    width_mm: float
    height_mm: float
    holes: List[Tuple[float, float, float]]  # [(x_mm, y_mm, diameter_mm), ...]
    scale_px_per_mm: float = 5.0  # Default: 5 pixels per millimetre
    image_size: Tuple[int, int] = (1920, 1080)  # Width x Height in pixels
    background_color: int = 255  # White background
    component_color: int = 180  # Light gray for metal
    hole_color: int = 0  # Black for holes
    add_noise: bool = False
    noise_level: float = 10.0


class SyntheticImageGenerator:
    """Generate synthetic CNC component images"""
    
    def __init__(self):
        self.reference_bar_height_mm = 10
        self.reference_bar_length_mm = 50  # 50mm scale bar
        
    def generate_rectangle(self, 
                          width_mm: float = 200, 
                          height_mm: float = 150,
                          hole_diameter_mm: float = 12,
                          hole_margin_mm: float = 20,
                          scale_px_per_mm: float = 5.0,
                          add_noise: bool = False) -> Tuple[np.ndarray, ComponentSpec]:
        """
        Generate a rectangular plate with 4 corner holes
        
        Args:
            width_mm: Plate width in millimetres
            height_mm: Plate height in millimetres
            hole_diameter_mm: Diameter of bolt holes
            hole_margin_mm: Distance from hole center to edge
            scale_px_per_mm: Pixel scaling factor
            add_noise: Add synthetic noise to simulate real photos
            
        Returns:
            Tuple of (image array, ComponentSpec with ground truth)
        """
        # Calculate hole positions (4 corners)
        holes = [
            (hole_margin_mm, hole_margin_mm, hole_diameter_mm),  # Top-left
            (width_mm - hole_margin_mm, hole_margin_mm, hole_diameter_mm),  # Top-right
            (hole_margin_mm, height_mm - hole_margin_mm, hole_diameter_mm),  # Bottom-left
            (width_mm - hole_margin_mm, height_mm - hole_margin_mm, hole_diameter_mm)  # Bottom-right
        ]
        
        spec = ComponentSpec(
            shape_type='rectangle',
            width_mm=width_mm,
            height_mm=height_mm,
            holes=holes,
            scale_px_per_mm=scale_px_per_mm,
            add_noise=add_noise
        )
        
        image = self._render_rectangle(spec)
        return image, spec
    
    def generate_flange(self,
                       outer_diameter_mm: float = 180,
                       central_bore_mm: float = 40,
                       bolt_circle_diameter_mm: float = 120,
                       num_holes: int = 6,
                       hole_diameter_mm: float = 10,
                       scale_px_per_mm: float = 5.0,
                       add_noise: bool = False) -> Tuple[np.ndarray, ComponentSpec]:
        """
        Generate a circular flange with central bore and evenly-spaced bolt holes
        
        Args:
            outer_diameter_mm: Outer diameter of flange
            central_bore_mm: Central bore diameter
            bolt_circle_diameter_mm: Diameter of circle on which bolt holes are placed
            num_holes: Number of bolt holes
            hole_diameter_mm: Diameter of each bolt hole
            scale_px_per_mm: Pixel scaling factor
            add_noise: Add synthetic noise
            
        Returns:
            Tuple of (image array, ComponentSpec with ground truth)
        """
        # Calculate hole positions on bolt circle
        holes = []
        angle_increment = 360 / num_holes
        radius = bolt_circle_diameter_mm / 2
        
        for i in range(num_holes):
            angle_rad = math.radians(i * angle_increment)
            x = (outer_diameter_mm / 2) + radius * math.cos(angle_rad)
            y = (outer_diameter_mm / 2) + radius * math.sin(angle_rad)
            holes.append((x, y, hole_diameter_mm))
        
        spec = ComponentSpec(
            shape_type='flange',
            width_mm=outer_diameter_mm,
            height_mm=outer_diameter_mm,
            holes=holes,
            scale_px_per_mm=scale_px_per_mm,
            add_noise=add_noise
        )
        
        # Store flange-specific parameters
        spec.outer_diameter_mm = outer_diameter_mm
        spec.central_bore_mm = central_bore_mm
        spec.bolt_circle_diameter_mm = bolt_circle_diameter_mm
        
        image = self._render_flange(spec)
        return image, spec
    
    def generate_l_bracket(self,
                          vertical_length_mm: float = 200,
                          horizontal_length_mm: float = 150,
                          thickness_mm: float = 50,
                          hole_positions: Optional[List[Tuple[float, float, float]]] = None,
                          scale_px_per_mm: float = 5.0,
                          add_noise: bool = False) -> Tuple[np.ndarray, ComponentSpec]:
        """
        Generate an L-bracket
        
        Args:
            vertical_length_mm: Length of vertical arm
            horizontal_length_mm: Length of horizontal arm
            thickness_mm: Thickness of the bracket arms
            hole_positions: List of (x, y, diameter) for holes, or None for default
            scale_px_per_mm: Pixel scaling factor
            add_noise: Add synthetic noise
            
        Returns:
            Tuple of (image array, ComponentSpec with ground truth)
        """
        if hole_positions is None:
            # Default: 3 holes at structural positions
            hole_positions = [
                (thickness_mm / 2, 30, 12),  # Vertical arm top
                (thickness_mm / 2, vertical_length_mm - 30, 12),  # Vertical arm bottom
                (horizontal_length_mm - 30, vertical_length_mm - thickness_mm / 2, 12)  # Horizontal arm end
            ]
        
        spec = ComponentSpec(
            shape_type='l_bracket',
            width_mm=horizontal_length_mm,
            height_mm=vertical_length_mm,
            holes=hole_positions,
            scale_px_per_mm=scale_px_per_mm,
            add_noise=add_noise
        )
        
        spec.thickness_mm = thickness_mm
        
        image = self._render_l_bracket(spec)
        return image, spec
    
    def generate_custom_polygon(self,
                               vertices_mm: List[Tuple[float, float]],
                               holes: List[Tuple[float, float, float]],
                               scale_px_per_mm: float = 5.0,
                               add_noise: bool = False) -> Tuple[np.ndarray, ComponentSpec]:
        """
        Generate a custom polygon shape
        
        Args:
            vertices_mm: List of (x, y) coordinates defining polygon vertices
            holes: List of (x, y, diameter) for holes
            scale_px_per_mm: Pixel scaling factor
            add_noise: Add synthetic noise
            
        Returns:
            Tuple of (image array, ComponentSpec with ground truth)
        """
        # Calculate bounding box
        xs = [v[0] for v in vertices_mm]
        ys = [v[1] for v in vertices_mm]
        width_mm = max(xs) - min(xs)
        height_mm = max(ys) - min(ys)
        
        spec = ComponentSpec(
            shape_type='custom',
            width_mm=width_mm,
            height_mm=height_mm,
            holes=holes,
            scale_px_per_mm=scale_px_per_mm,
            add_noise=add_noise
        )
        
        spec.vertices_mm = vertices_mm
        
        image = self._render_custom_polygon(spec)
        return image, spec
    
    def _render_rectangle(self, spec: ComponentSpec) -> np.ndarray:
        """Render a rectangular plate"""
        # Create canvas
        canvas = np.ones(spec.image_size[::-1], dtype=np.uint8) * spec.background_color
        
        # Calculate component position (centered)
        comp_width_px = int(spec.width_mm * spec.scale_px_per_mm)
        comp_height_px = int(spec.height_mm * spec.scale_px_per_mm)
        
        center_x = spec.image_size[0] // 2
        center_y = spec.image_size[1] // 2
        
        x1 = center_x - comp_width_px // 2
        y1 = center_y - comp_height_px // 2
        x2 = x1 + comp_width_px
        y2 = y1 + comp_height_px
        
        # Draw rectangle
        cv2.rectangle(canvas, (x1, y1), (x2, y2), spec.component_color, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 100, 2)  # Border
        
        # Draw holes
        for hole_x_mm, hole_y_mm, hole_d_mm in spec.holes:
            hole_x_px = x1 + int(hole_x_mm * spec.scale_px_per_mm)
            hole_y_px = y1 + int(hole_y_mm * spec.scale_px_per_mm)
            hole_radius_px = int((hole_d_mm / 2) * spec.scale_px_per_mm)
            
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, spec.hole_color, -1)
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, 50, 1)  # Border
        
        # Add scale reference
        canvas = self._add_scale_reference(canvas, spec)
        
        # Add noise if requested
        if spec.add_noise:
            canvas = self._add_noise(canvas, spec.noise_level)
        
        return canvas
    
    def _render_flange(self, spec: ComponentSpec) -> np.ndarray:
        """Render a circular flange"""
        canvas = np.ones(spec.image_size[::-1], dtype=np.uint8) * spec.background_color
        
        # Calculate position (centered)
        outer_radius_px = int((spec.outer_diameter_mm / 2) * spec.scale_px_per_mm)
        central_bore_radius_px = int((spec.central_bore_mm / 2) * spec.scale_px_per_mm)
        
        center_x = spec.image_size[0] // 2
        center_y = spec.image_size[1] // 2
        
        # Draw outer circle
        cv2.circle(canvas, (center_x, center_y), outer_radius_px, spec.component_color, -1)
        cv2.circle(canvas, (center_x, center_y), outer_radius_px, 100, 2)  # Border
        
        # Draw central bore
        cv2.circle(canvas, (center_x, center_y), central_bore_radius_px, spec.hole_color, -1)
        cv2.circle(canvas, (center_x, center_y), central_bore_radius_px, 50, 1)  # Border
        
        # Draw bolt holes
        offset_x = center_x - int((spec.width_mm / 2) * spec.scale_px_per_mm)
        offset_y = center_y - int((spec.height_mm / 2) * spec.scale_px_per_mm)
        
        for hole_x_mm, hole_y_mm, hole_d_mm in spec.holes:
            hole_x_px = offset_x + int(hole_x_mm * spec.scale_px_per_mm)
            hole_y_px = offset_y + int(hole_y_mm * spec.scale_px_per_mm)
            hole_radius_px = int((hole_d_mm / 2) * spec.scale_px_per_mm)
            
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, spec.hole_color, -1)
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, 50, 1)  # Border
        
        # Add scale reference
        canvas = self._add_scale_reference(canvas, spec)
        
        if spec.add_noise:
            canvas = self._add_noise(canvas, spec.noise_level)
        
        return canvas
    
    def _render_l_bracket(self, spec: ComponentSpec) -> np.ndarray:
        """Render an L-bracket"""
        canvas = np.ones(spec.image_size[::-1], dtype=np.uint8) * spec.background_color
        
        # Calculate position (centered)
        comp_width_px = int(spec.width_mm * spec.scale_px_per_mm)
        comp_height_px = int(spec.height_mm * spec.scale_px_per_mm)
        thickness_px = int(spec.thickness_mm * spec.scale_px_per_mm)
        
        center_x = spec.image_size[0] // 2
        center_y = spec.image_size[1] // 2
        
        x1 = center_x - comp_width_px // 2
        y1 = center_y - comp_height_px // 2
        
        # Define L-shape as two rectangles
        # Vertical arm
        pts_vertical = np.array([
            [x1, y1],
            [x1 + thickness_px, y1],
            [x1 + thickness_px, y1 + comp_height_px],
            [x1, y1 + comp_height_px]
        ], np.int32)
        
        # Horizontal arm
        pts_horizontal = np.array([
            [x1, y1 + comp_height_px - thickness_px],
            [x1 + comp_width_px, y1 + comp_height_px - thickness_px],
            [x1 + comp_width_px, y1 + comp_height_px],
            [x1, y1 + comp_height_px]
        ], np.int32)
        
        # Draw L-bracket
        cv2.fillPoly(canvas, [pts_vertical], spec.component_color)
        cv2.fillPoly(canvas, [pts_horizontal], spec.component_color)
        cv2.polylines(canvas, [pts_vertical], True, 100, 2)
        cv2.polylines(canvas, [pts_horizontal], True, 100, 2)
        
        # Draw holes
        for hole_x_mm, hole_y_mm, hole_d_mm in spec.holes:
            hole_x_px = x1 + int(hole_x_mm * spec.scale_px_per_mm)
            hole_y_px = y1 + int(hole_y_mm * spec.scale_px_per_mm)
            hole_radius_px = int((hole_d_mm / 2) * spec.scale_px_per_mm)
            
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, spec.hole_color, -1)
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, 50, 1)
        
        # Add scale reference
        canvas = self._add_scale_reference(canvas, spec)
        
        if spec.add_noise:
            canvas = self._add_noise(canvas, spec.noise_level)
        
        return canvas
    
    def _render_custom_polygon(self, spec: ComponentSpec) -> np.ndarray:
        """Render a custom polygon"""
        canvas = np.ones(spec.image_size[::-1], dtype=np.uint8) * spec.background_color
        
        # Calculate position (centered)
        vertices = np.array(spec.vertices_mm)
        min_x, min_y = vertices.min(axis=0)
        max_x, max_y = vertices.max(axis=0)
        
        comp_width_px = int((max_x - min_x) * spec.scale_px_per_mm)
        comp_height_px = int((max_y - min_y) * spec.scale_px_per_mm)
        
        center_x = spec.image_size[0] // 2
        center_y = spec.image_size[1] // 2
        
        offset_x = center_x - comp_width_px // 2 - int(min_x * spec.scale_px_per_mm)
        offset_y = center_y - comp_height_px // 2 - int(min_y * spec.scale_px_per_mm)
        
        # Convert vertices to pixel coordinates
        pts = []
        for vx, vy in spec.vertices_mm:
            px = offset_x + int(vx * spec.scale_px_per_mm)
            py = offset_y + int(vy * spec.scale_px_per_mm)
            pts.append([px, py])
        
        pts = np.array(pts, np.int32)
        
        # Draw polygon
        cv2.fillPoly(canvas, [pts], spec.component_color)
        cv2.polylines(canvas, [pts], True, 100, 2)
        
        # Draw holes
        for hole_x_mm, hole_y_mm, hole_d_mm in spec.holes:
            hole_x_px = offset_x + int(hole_x_mm * spec.scale_px_per_mm)
            hole_y_px = offset_y + int(hole_y_mm * spec.scale_px_per_mm)
            hole_radius_px = int((hole_d_mm / 2) * spec.scale_px_per_mm)
            
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, spec.hole_color, -1)
            cv2.circle(canvas, (hole_x_px, hole_y_px), hole_radius_px, 50, 1)
        
        # Add scale reference
        canvas = self._add_scale_reference(canvas, spec)
        
        if spec.add_noise:
            canvas = self._add_noise(canvas, spec.noise_level)
        
        return canvas
    
    def _add_scale_reference(self, canvas: np.ndarray, spec: ComponentSpec) -> np.ndarray:
        """Add a scale reference bar to the image"""
        bar_length_px = int(self.reference_bar_length_mm * spec.scale_px_per_mm)
        bar_height_px = int(self.reference_bar_height_mm * spec.scale_px_per_mm)
        
        # Position in bottom-right corner
        margin = 30
        x1 = canvas.shape[1] - bar_length_px - margin
        y1 = canvas.shape[0] - bar_height_px - margin - 40
        x2 = x1 + bar_length_px
        y2 = y1 + bar_height_px
        
        # Draw scale bar
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 0, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), 100, 2)
        
        # Add text label
        text = f"{self.reference_bar_length_mm} mm"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        
        text_x = x1 + (bar_length_px - text_size[0]) // 2
        text_y = y2 + 30
        
        cv2.putText(canvas, text, (text_x, text_y), font, font_scale, 0, font_thickness)
        
        return canvas
    
    def _add_noise(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to simulate real photography"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.int16)
        noisy_image = image.astype(np.int16) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    def save_image(self, image: np.ndarray, filepath: str):
        """Save image to file"""
        cv2.imwrite(filepath, image)
        print(f"Image saved: {filepath}")
    
    def get_ground_truth(self, spec: ComponentSpec) -> Dict:
        """
        Extract ground truth measurements from specification
        
        Returns:
            Dictionary with all ground truth values
        """
        ground_truth = {
            'shape_type': spec.shape_type,
            'width_mm': spec.width_mm,
            'height_mm': spec.height_mm,
            'scale_px_per_mm': spec.scale_px_per_mm,
            'num_holes': len(spec.holes),
            'holes': []
        }
        
        # Component center (origin for hole coordinates)
        center_x_mm = spec.width_mm / 2
        center_y_mm = spec.height_mm / 2
        
        for i, (x_mm, y_mm, d_mm) in enumerate(spec.holes):
            # Compute position relative to component center
            rel_x = x_mm - center_x_mm
            rel_y = y_mm - center_y_mm
            
            ground_truth['holes'].append({
                'hole_id': i,
                'absolute_x_mm': x_mm,
                'absolute_y_mm': y_mm,
                'relative_x_mm': rel_x,
                'relative_y_mm': rel_y,
                'diameter_mm': d_mm
            })
        
        # Calculate hole spacings
        spacings = []
        holes_list = spec.holes
        for i in range(len(holes_list)):
            for j in range(i + 1, len(holes_list)):
                x1, y1, _ = holes_list[i]
                x2, y2, _ = holes_list[j]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                spacings.append({
                    'hole_pair': f"{i}-{j}",
                    'spacing_mm': distance
                })
        
        ground_truth['hole_spacings'] = spacings
        
        return ground_truth


def demo_all_shapes():
    """Generate sample images for all four shapes"""
    generator = SyntheticImageGenerator()
    
    print("Generating synthetic CNC component images...")
    print("=" * 60)
    
    # 1. Rectangle
    print("\n1. Generating rectangular plate...")
    rect_img, rect_spec = generator.generate_rectangle(
        width_mm=200,
        height_mm=150,
        hole_diameter_mm=12,
        hole_margin_mm=20
    )
    generator.save_image(rect_img, "sample_rectangle.png")
    rect_gt = generator.get_ground_truth(rect_spec)
    print(f"   Ground truth: {rect_gt['width_mm']}mm × {rect_gt['height_mm']}mm, {rect_gt['num_holes']} holes")
    
    # 2. Flange
    print("\n2. Generating circular flange...")
    flange_img, flange_spec = generator.generate_flange(
        outer_diameter_mm=180,
        central_bore_mm=40,
        bolt_circle_diameter_mm=120,
        num_holes=6,
        hole_diameter_mm=10
    )
    generator.save_image(flange_img, "sample_flange.png")
    flange_gt = generator.get_ground_truth(flange_spec)
    print(f"   Ground truth: {flange_gt['width_mm']}mm diameter, {flange_gt['num_holes']} holes")
    
    # 3. L-Bracket
    print("\n3. Generating L-bracket...")
    lbracket_img, lbracket_spec = generator.generate_l_bracket(
        vertical_length_mm=200,
        horizontal_length_mm=150,
        thickness_mm=50
    )
    generator.save_image(lbracket_img, "sample_l_bracket.png")
    lbracket_gt = generator.get_ground_truth(lbracket_spec)
    print(f"   Ground truth: {lbracket_gt['width_mm']}mm × {lbracket_gt['height_mm']}mm, {lbracket_gt['num_holes']} holes")
    
    # 4. Custom polygon (hexagon)
    print("\n4. Generating custom polygon (hexagon)...")
    hex_vertices = []
    for i in range(6):
        angle = math.radians(60 * i)
        x = 100 + 80 * math.cos(angle)
        y = 100 + 80 * math.sin(angle)
        hex_vertices.append((x, y))
    
    hex_holes = [
        (100, 100, 15),  # Center hole
        (100 + 50, 100, 10),  # Right
        (100 - 50, 100, 10),  # Left
    ]
    
    custom_img, custom_spec = generator.generate_custom_polygon(
        vertices_mm=hex_vertices,
        holes=hex_holes
    )
    generator.save_image(custom_img, "sample_custom_polygon.png")
    custom_gt = generator.get_ground_truth(custom_spec)
    print(f"   Ground truth: {custom_gt['width_mm']:.1f}mm × {custom_gt['height_mm']:.1f}mm, {custom_gt['num_holes']} holes")
    
    print("\n" + "=" * 60)
    print("✓ All sample images generated successfully!")
    print("\nFiles created:")
    print("  - sample_rectangle.png")
    print("  - sample_flange.png")
    print("  - sample_l_bracket.png")
    print("  - sample_custom_polygon.png")


if __name__ == "__main__":
    demo_all_shapes()
