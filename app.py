"""
CNC Inspection System - Streamlit GUI Application
Phase 8: Interactive web interface for component inspection

Usage: streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import json
from datetime import datetime

# Import all modules
from image_generator import SyntheticImageGenerator
from preprocessor import ImagePreprocessor
from shape_detector import ShapeDetector
from hole_detector import HoleDetector
from measurement import MeasurementEngine
from tolerance import ToleranceChecker, ToleranceConfig
from reporter import InspectionReporter


# Page configuration
st.set_page_config(
    page_title="CNC Inspection System",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔧 AI-Based CNC Inspection System")
st.markdown("**Automated dimensional inspection of CNC-machined components**")

# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Image source
image_source = st.sidebar.radio(
    "Image Source",
    ["Upload Image", "Generate Synthetic"]
)

uploaded_file = None
generated_image = None
spec = None

is_real_photo = False
if image_source == "Upload Image":
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg']
    )
    is_real_photo = st.sidebar.toggle(
        "Real photo (not synthetic)",
        value=True,
        help="Enable for actual photos of machined parts. Uses robust background removal and hole detection tuned for real metal surfaces."
    )
elif image_source == "Generate Synthetic":
    st.sidebar.subheader("Synthetic Image Generator")
    
    shape_type = st.sidebar.selectbox(
        "Component Shape",
        ["Rectangle", "Flange", "L-Bracket"]
    )
    
    generator = SyntheticImageGenerator()
    
    if shape_type == "Rectangle":
        width = st.sidebar.slider("Width (mm)", 100, 300, 200)
        height = st.sidebar.slider("Height (mm)", 100, 250, 150)
        hole_dia = st.sidebar.slider("Hole Diameter (mm)", 8, 20, 12)
        
        if st.sidebar.button("Generate Rectangle"):
            generated_image, spec = generator.generate_rectangle(
                width_mm=width,
                height_mm=height,
                hole_diameter_mm=hole_dia
            )
            st.session_state.generated_image = generated_image
            st.session_state.spec = spec
    
    elif shape_type == "Flange":
        diameter = st.sidebar.slider("Outer Diameter (mm)", 100, 250, 180)
        num_holes = st.sidebar.slider("Number of Holes", 4, 8, 6)
        hole_dia = st.sidebar.slider("Hole Diameter (mm)", 6, 15, 10)
        
        if st.sidebar.button("Generate Flange"):
            generated_image, spec = generator.generate_flange(
                outer_diameter_mm=diameter,
                num_holes=num_holes,
                hole_diameter_mm=hole_dia
            )
            st.session_state.generated_image = generated_image
            st.session_state.spec = spec
    
    elif shape_type == "L-Bracket":
        vert = st.sidebar.slider("Vertical Length (mm)", 100, 300, 200)
        horiz = st.sidebar.slider("Horizontal Length (mm)", 100, 250, 150)
        
        if st.sidebar.button("Generate L-Bracket"):
            generated_image, spec = generator.generate_l_bracket(
                vertical_length_mm=vert,
                horizontal_length_mm=horiz
            )
            st.session_state.generated_image = generated_image
            st.session_state.spec = spec
    
    if 'generated_image' in st.session_state:
        generated_image = st.session_state.generated_image
        spec = st.session_state.spec

# Scale factor
st.sidebar.subheader("📏 Scale Calibration")
scale_px_per_mm = st.sidebar.number_input(
    "Pixels per millimeter",
    min_value=1.0,
    max_value=20.0,
    value=5.0,
    step=0.1
)

# Tolerance settings
st.sidebar.subheader("📊 Tolerance Settings")

with st.sidebar.expander("Component Tolerances"):
    width_nominal = st.number_input("Width Nominal (mm)", value=200.0)
    width_tol = st.number_input("Width Tolerance (±mm)", value=1.0)
    
    height_nominal = st.number_input("Height Nominal (mm)", value=150.0)
    height_tol = st.number_input("Height Tolerance (±mm)", value=1.0)

with st.sidebar.expander("Hole Tolerances"):
    hole_dia_nominal = st.number_input("Hole Diameter Nominal (mm)", value=12.0)
    hole_dia_tol = st.number_input("Hole Diameter Tolerance (±mm)", value=0.5)

# Run inspection button
run_inspection = st.sidebar.button("🔍 Run Inspection", type="primary")

# Main content area
if uploaded_file is not None or generated_image is not None:
    
    # Load image
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_name = uploaded_file.name
    else:
        input_image = generated_image
        image_name = "Synthetic Generated Image"
    
    # Display original image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Original Image")
        display_img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        st.image(display_img, use_column_width=True)
    
    # Run inspection when button clicked
    if run_inspection:
        with st.spinner("Running inspection pipeline..."):
            
            # Initialize pipeline
            preprocessor = ImagePreprocessor()
            shape_detector = ShapeDetector(min_area=1000)
            
            # Tune hole detector for real photos vs synthetic
            if is_real_photo:
                hole_detector = HoleDetector(
                    min_radius=10, max_radius=200,
                    param1=50, param2=20, min_dist=30
                )
            else:
                hole_detector = HoleDetector(min_radius=15, max_radius=80, param2=25)
            
            measurement_engine = MeasurementEngine(scale_px_per_mm=scale_px_per_mm)
            
            # Step 1: Preprocess
            preprocess_result = preprocessor.process(input_image, is_synthetic=not is_real_photo)
            
            # Step 2: Detect shape — use binary directly for real photos (more robust)
            if is_real_photo:
                shape_result = shape_detector.detect(preprocess_result.binary)
            else:
                shape_result = shape_detector.detect(preprocess_result.morphed)
            
            # Step 3: Detect holes — pass binary mask and bg_is_light for real photos
            if is_real_photo:
                img_h, img_w = input_image.shape[:2]
                corners_bgr = [
                    input_image[0:40, 0:40], input_image[0:40, img_w-40:],
                    input_image[img_h-40:, 0:40], input_image[img_h-40:, img_w-40:]
                ]
                bg_is_light = float(np.mean([c.mean() for c in corners_bgr])) > 200
                binary_mask = preprocess_result.binary
            else:
                bg_is_light = None
                binary_mask = None

            hole_result = hole_detector.detect(
                preprocess_result.blurred,
                component_contour=shape_result.contour,
                binary_mask=binary_mask,
                bg_is_light=bg_is_light
            )
            
            # Step 4: Compute measurements
            measurements = measurement_engine.compute_measurements(
                shape_result,
                hole_result
            )
            
            # Step 5: Check tolerances
            tolerance_config = ToleranceConfig.create_default(
                width_nominal=width_nominal,
                height_nominal=height_nominal,
                hole_diameter_nominal=hole_dia_nominal,
                width_tolerance=width_tol,
                height_tolerance=height_tol,
                hole_diameter_tolerance=hole_dia_tol
            )
            
            checker = ToleranceChecker(tolerance_config)
            inspection_report = checker.check(measurements)
            
            # Step 6: Generate report
            reporter = InspectionReporter()
            annotated_image = reporter.create_annotated_image(
                input_image,
                shape_result,
                hole_result,
                measurements,
                inspection_report
            )
            
            # Store results in session state
            st.session_state.annotated_image = annotated_image
            st.session_state.measurements = measurements
            st.session_state.inspection_report = inspection_report
            st.session_state.reporter = reporter
            st.session_state.processed = True
    
    # Display results if processed
    if st.session_state.processed:
        with col2:
            st.subheader("✅ Inspection Results")
            annotated_rgb = cv2.cvtColor(st.session_state.annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_rgb, use_column_width=True)
        
        # Verdict banner
        verdict = st.session_state.inspection_report.overall_verdict
        if verdict == "PASS":
            st.success(f"### ✅ OVERALL VERDICT: {verdict}")
        else:
            st.error(f"### ❌ OVERALL VERDICT: {verdict}")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Component Width",
                f"{st.session_state.measurements.component_width_mm:.1f} mm"
            )
        
        with col2:
            st.metric(
                "Component Height",
                f"{st.session_state.measurements.component_height_mm:.1f} mm"
            )
        
        with col3:
            st.metric(
                "Holes Detected",
                st.session_state.measurements.num_holes
            )
        
        with col4:
            st.metric(
                "PASS Rate",
                f"{st.session_state.inspection_report.pass_count}/{len(st.session_state.inspection_report.verdicts)}"
            )
        
        # Detailed results table
        st.subheader("📋 Detailed Measurements")
        
        results_data = []
        for verdict in st.session_state.inspection_report.verdicts:
            results_data.append({
                'Measurement': verdict.measurement_name,
                'Measured': f"{verdict.measured_value:.2f} mm",
                'Nominal': f"{verdict.nominal_value:.2f} mm" if verdict.nominal_value else "N/A",
                'Tolerance': f"±{verdict.tolerance_plus:.2f} mm" if verdict.tolerance_plus else "N/A",
                'Deviation': f"{verdict.deviation:+.2f} mm" if verdict.deviation else "N/A",
                'Verdict': verdict.verdict
            })
        
        st.dataframe(results_data, use_container_width=True)
        
        # Download section
        st.subheader("💾 Download Reports")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Annotated image
            is_success, buffer = cv2.imencode(".png", st.session_state.annotated_image)
            if is_success:
                st.download_button(
                    label="📥 Download Annotated Image",
                    data=buffer.tobytes(),
                    file_name=f"inspection_annotated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png"
                )
        
        with col2:
            # JSON report
            json_report = st.session_state.reporter.generate_json_report(
                st.session_state.measurements,
                st.session_state.inspection_report,
                {'filename': image_name, 'timestamp': datetime.now().isoformat()}
            )
            st.download_button(
                label="📥 Download JSON Report",
                data=json.dumps(json_report, indent=2),
                file_name=f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col3:
            # CSV table
            csv_rows = st.session_state.reporter.generate_csv_table(
                st.session_state.inspection_report
            )
            csv_text = "\\n".join([",".join(row) for row in csv_rows])
            st.download_button(
                label="📥 Download CSV Results",
                data=csv_text,
                file_name=f"inspection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

else:
    # Welcome screen
    st.info("👈 Please upload an image or generate a synthetic component to begin inspection.")
    
    st.markdown("""
    ### 🚀 Features
    - **Automated Inspection**: Detects component boundaries and bolt holes
    - **Precise Measurements**: Dimensional analysis with sub-millimeter accuracy
    - **Tolerance Checking**: PASS/FAIL verdicts based on your specifications
    - **Multiple Shapes**: Supports rectangles, flanges, L-brackets, and custom geometries
    - **Synthetic Image Generation**: Create test images with known ground truth
    - **Comprehensive Reports**: Download annotated images, JSON reports, and CSV tables
    
    ### 📖 How to Use
    1. **Choose Image Source**: Upload a photo or generate a synthetic component
    2. **Configure Scale**: Set the pixels-per-millimeter calibration factor
    3. **Set Tolerances**: Define nominal values and acceptable deviations
    4. **Run Inspection**: Click the "Run Inspection" button
    5. **Review Results**: View measurements, verdicts, and download reports
    """)

# Footer
st.markdown("---")
st.markdown("**CNC Inspection System** | Phase 8: Streamlit GUI | Built with OpenCV, NumPy, and Streamlit")