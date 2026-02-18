"""
CNC Inspection System - Reporter Module
Phase 7: Generate annotated images and structured reports

This module creates:
- Annotated images with all measurements and verdicts
- JSON inspection reports
- CSV result tables
"""

import cv2
import numpy as np
import json
import csv
from typing import Dict, List
from dataclasses import dataclass

from shape_detector import ShapeDetectionResult
from hole_detector import HoleDetectionResult
from measurement import MeasurementResult
from tolerance import InspectionReport


class InspectionReporter:
    """
    Generate comprehensive inspection reports
    """
    
    def __init__(self):
        pass
    
    def create_annotated_image(self,
                              original_image: np.ndarray,
                              shape_result: ShapeDetectionResult,
                              hole_result: HoleDetectionResult,
                              measurements: MeasurementResult,
                              inspection_report: InspectionReport) -> np.ndarray:
        """
        Create fully annotated inspection image
        
        Args:
            original_image: Original input image
            shape_result: Shape detection results
            hole_result: Hole detection results
            measurements: Measurements in mm
            inspection_report: PASS/FAIL verdicts
        
        Returns:
            Annotated RGB image
        """
        if len(original_image.shape) == 2:
            vis = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        else:
            vis = original_image.copy()
        
        # Draw component boundary (green)
        cv2.drawContours(vis, [shape_result.contour], -1, (0, 255, 0), 3)
        
        # Draw bounding box (blue)
        box_points = shape_result.bounding_box.get_corners()
        cv2.drawContours(vis, [box_points], 0, (255, 0, 0), 2)
        
        # Draw centroid (red)
        cx, cy = shape_result.centroid
        cv2.circle(vis, (int(cx), int(cy)), 10, (0, 0, 255), -1)
        cv2.circle(vis, (int(cx), int(cy)), 12, (255, 255, 255), 2)
        
        # Component dimensions
        dim_text = f"{measurements.component_width_mm:.1f}mm x {measurements.component_height_mm:.1f}mm"
        self._draw_text_with_background(vis, dim_text,
                                       (30, 50), 
                                       font_scale=0.8,
                                       color=(255, 255, 255),
                                       bg_color=(0, 100, 0))
        
        # Draw holes
        for i, hole in enumerate(hole_result.holes):
            x = int(hole.center_x)
            y = int(hole.center_y)
            r = int(hole.radius)
            
            # Circle (yellow)
            cv2.circle(vis, (x, y), r, (0, 255, 255), 3)
            cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)
            
            # Hole label
            hole_data = measurements.hole_measurements[i]
            label = f"#{i}: {hole_data['diameter_mm']}mm"
            self._draw_text_with_background(vis, label,
                                           (x - 40, y - r - 15),
                                           font_scale=0.6,
                                           color=(0, 255, 255),
                                           bg_color=(0, 0, 0))
        
        # Overall verdict (top-right corner)
        verdict_text = f"VERDICT: {inspection_report.overall_verdict}"
        verdict_color = (0, 255, 0) if inspection_report.overall_verdict == "PASS" else (0, 0, 255)
        self._draw_text_with_background(vis, verdict_text,
                                       (vis.shape[1] - 300, 50),
                                       font_scale=1.0,
                                       color=verdict_color,
                                       bg_color=(0, 0, 0))
        
        return vis
    
    def _draw_text_with_background(self, image, text, position, 
                                   font_scale=0.7, color=(255, 255, 255),
                                   bg_color=(0, 0, 0), thickness=2):
        """Draw text with background rectangle"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        x, y = position
        padding = 5
        
        # Background rectangle
        cv2.rectangle(image,
                     (x - padding, y - text_size[1] - padding),
                     (x + text_size[0] + padding, y + padding),
                     bg_color, -1)
        
        # Text
        cv2.putText(image, text, (x, y), font, font_scale, color, thickness)
    
    def generate_json_report(self,
                            measurements: MeasurementResult,
                            inspection_report: InspectionReport,
                            metadata: Dict = None) -> Dict:
        """
        Generate comprehensive JSON report
        
        Args:
            measurements: All measurements
            inspection_report: Inspection verdicts
            metadata: Optional metadata (filename, date, etc.)
        
        Returns:
            Complete report as dictionary
        """
        report = {
            'metadata': metadata or {},
            'measurements': measurements.to_dict(),
            'inspection': inspection_report.to_dict()
        }
        return report
    
    def save_json_report(self, report: Dict, filepath: str):
        """Save JSON report to file"""
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
    
    def generate_csv_table(self, inspection_report: InspectionReport) -> List[List]:
        """
        Generate CSV table of results
        
        Args:
            inspection_report: Inspection report
        
        Returns:
            List of rows for CSV
        """
        rows = [
            ['Measurement', 'Measured Value', 'Nominal', 'Tolerance', 'Deviation', 'Deviation %', 'Verdict']
        ]
        
        for verdict in inspection_report.verdicts:
            row = [
                verdict.measurement_name,
                f"{verdict.measured_value:.2f}",
                f"{verdict.nominal_value:.2f}" if verdict.nominal_value else "N/A",
                f"±{verdict.tolerance_plus:.2f}" if verdict.tolerance_plus else "N/A",
                f"{verdict.deviation:+.2f}" if verdict.deviation else "N/A",
                f"{verdict.deviation_percent:.2f}%" if verdict.deviation_percent else "N/A",
                verdict.verdict
            ]
            rows.append(row)
        
        return rows
    
    def save_csv_table(self, rows: List[List], filepath: str):
        """Save CSV table to file"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)


def demo_reporter():
    """Demo: Generate complete inspection report"""
    import os
    from datetime import datetime
    from preprocessor import ImagePreprocessor
    from shape_detector import ShapeDetector
    from hole_detector import HoleDetector
    from measurement import MeasurementEngine
    from tolerance import ToleranceChecker, ToleranceConfig
    
    print("=" * 70)
    print("PHASE 7: REPORTER DEMO")
    print("=" * 70)
    
    sample_file = "sample_flange.png"  # Use flange as it has holes
    filepath = os.path.join("sample_images", sample_file)
    
    if not os.path.exists(filepath):
        print("Sample file not found!")
        return
    
    print(f"\nGenerating complete inspection report for: {sample_file}")
    
    # Full pipeline
    image = cv2.imread(filepath)
    preprocessor = ImagePreprocessor()
    shape_detector = ShapeDetector(min_area=1000)
    hole_detector = HoleDetector(min_radius=15, max_radius=80, param2=25)
    measurement_engine = MeasurementEngine(scale_px_per_mm=5.0)
    
    preprocess_result = preprocessor.process(image, is_synthetic=True)
    shape_result = shape_detector.detect(preprocess_result.morphed)
    hole_result = hole_detector.detect(preprocess_result.blurred, shape_result.contour)
    measurements = measurement_engine.compute_measurements(shape_result, hole_result)
    
    # Tolerance checking
    tolerance_config = ToleranceConfig.create_default(
        width_nominal=180.0,
        height_nominal=180.0,
        hole_diameter_nominal=10.0,
        width_tolerance=2.0,
        height_tolerance=2.0,
        hole_diameter_tolerance=1.0
    )
    
    checker = ToleranceChecker(tolerance_config)
    inspection_report = checker.check(measurements)
    
    # Generate reports
    output_dir = "final_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    reporter = InspectionReporter()
    
    # 1. Annotated image
    print("\n1. Creating annotated image...")
    annotated = reporter.create_annotated_image(
        image, shape_result, hole_result, measurements, inspection_report
    )
    annotated_path = os.path.join(output_dir, "flange_inspection_annotated.png")
    cv2.imwrite(annotated_path, annotated)
    print(f"   ✓ Saved: {annotated_path}")
    
    # 2. JSON report
    print("\n2. Generating JSON report...")
    metadata = {
        'filename': sample_file,
        'timestamp': datetime.now().isoformat(),
        'scale_px_per_mm': 5.0
    }
    json_report = reporter.generate_json_report(measurements, inspection_report, metadata)
    json_path = os.path.join(output_dir, "flange_inspection_report.json")
    reporter.save_json_report(json_report, json_path)
    print(f"   ✓ Saved: {json_path}")
    
    # 3. CSV table
    print("\n3. Generating CSV table...")
    csv_rows = reporter.generate_csv_table(inspection_report)
    csv_path = os.path.join(output_dir, "flange_inspection_results.csv")
    reporter.save_csv_table(csv_rows, csv_path)
    print(f"   ✓ Saved: {csv_path}")
    
    print(f"\n{'=' * 70}")
    print("✅ REPORTING COMPLETE")
    print(f"{'=' * 70}")
    print(f"\nOverall Verdict: {inspection_report.overall_verdict}")
    print(f"  ✓ PASS: {inspection_report.pass_count}")
    print(f"  ✗ FAIL: {inspection_report.fail_count}")


if __name__ == "__main__":
    demo_reporter()
