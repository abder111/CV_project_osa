import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
import math

# Configure page layout
st.set_page_config(
    page_title="Retail Shelf Analysis",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-container {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .stock-level-good {
        color: #28a745;
        font-weight: bold;
    }
    .stock-level-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .stock-level-danger {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'current_image_path' not in st.session_state:
    st.session_state.current_image_path = None

# Import your EnhancedRetailPipeline class here
from pipeline import EnhancedRetailPipeline

def capture_matplotlib_figure():
    """Capture the current matplotlib figure and return it as a PIL Image"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close()  # Close the figure to free memory
    return img

def display_visualization_results(results, image_path):
    """Display the complete visualization results from the pipeline"""
    st.header("🎯 Detection Visualization")
    
    # Create a temporary matplotlib figure to capture
    try:
        # Call the pipeline's visualization method
        # This will create matplotlib figures but not display them
        st.session_state.pipeline.visualize_complete_results(results, save_path=None)
        
        # Capture the matplotlib figure
        viz_image = capture_matplotlib_figure()
        
        # Display the visualization in Streamlit
        st.image(viz_image, caption="Complete Analysis Visualization", use_column_width=True)
        
        # Add download button for the visualization
        buf = io.BytesIO()
        viz_image.save(buf, format='PNG')
        buf.seek(0)
        
        st.download_button(
            label="📥 Download Visualization",
            data=buf.getvalue(),
            file_name=f"shelf_analysis_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png"
        )
        
    except Exception as e:
        st.error(f"Error generating visualization: {str(e)}")
        st.info("Falling back to basic results display...")

def extract_and_process_frames(video_file, interval, max_frames):
    """Extract frames from video and process them with improved file handling"""
    frames_results = []
    tmp_video_path = None
    
    try:
        # Create temporary directory for better organization
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            tmp_video_path = os.path.join(temp_dir, "temp_video.mp4")
            with open(tmp_video_path, 'wb') as f:
                f.write(video_file.read())
            
            # Open video with OpenCV
            cap = cv2.VideoCapture(tmp_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval_frames = int(interval * fps)
            
            frame_count = 0
            processed_frames = 0
            
            while cap.isOpened() and processed_frames < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame at specified interval
                if frame_count % frame_interval_frames == 0:
                    # Save frame in the same temporary directory
                    frame_path = os.path.join(temp_dir, f"frame_{processed_frames}.jpg")
                    cv2.imwrite(frame_path, frame)
                    
                    # Process frame through pipeline
                    frame_result = st.session_state.pipeline.detect_and_classify_complete(frame_path)
                    frame_result['frame_number'] = processed_frames + 1
                    frame_result['timestamp'] = frame_count / fps
                    frame_result['frame_path'] = frame_path  # Store frame path for visualization
                    frames_results.append(frame_result)
                    
                    processed_frames += 1
                
                frame_count += 1
            
            cap.release()
            # Temporary directory and all its contents are automatically cleaned up
            
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return []
    
    return frames_results

def display_trend_analysis(frames_results):
    """Display trend analysis for multiple frames"""
    st.header("📈 Trend Analysis")
    
    # Extract trend data
    timestamps = [frame['timestamp'] for frame in frames_results]
    stock_levels = [frame['summary']['overall_stock_percentage'] for frame in frames_results]
    
    # Create trend charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig_stock_trend = px.line(
            x=timestamps,
            y=stock_levels,
            title="Stock Level Trend Over Time",
            labels={'x': 'Time (seconds)', 'y': 'Stock Level (%)'}
        )
        st.plotly_chart(fig_stock_trend, use_container_width=True)
    
    # Product-specific trends
    st.subheader("Product-Specific Trends")
    
    product_trends = {}
    for frame in frames_results:
        for product, data in frame['summary']['stock_levels'].items():
            if product not in product_trends:
                product_trends[product] = []
            product_trends[product].append(data['stock_percentage'])
    
    for product, trend_data in product_trends.items():
        fig_product_trend = px.line(
            x=timestamps,
            y=trend_data,
            title=f"{product.title()} Stock Trend",
            labels={'x': 'Time (seconds)', 'y': 'Stock Level (%)'}
        )
        st.plotly_chart(fig_product_trend, use_container_width=True)
    
    # Frame-by-frame visualization option
    st.subheader("🎯 Frame-by-Frame Detection Visualization")
    
    if len(frames_results) > 0:
        frame_selector = st.selectbox(
            "Select Frame to Visualize",
            options=range(len(frames_results)),
            format_func=lambda x: f"Frame {x+1} (t={frames_results[x]['timestamp']:.1f}s)"
        )
        
        selected_frame = frames_results[frame_selector]
        if 'frame_path' in selected_frame and os.path.exists(selected_frame['frame_path']):
            display_visualization_results(selected_frame, selected_frame['frame_path'])

def export_json(results):
    """Export results as JSON"""
    json_str = json.dumps(results, indent=2, default=str)
    st.download_button(
        label="Download JSON",
        data=json_str,
        file_name=f"shelf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

def export_csv(results):
    """Export results as CSV"""
    # Create CSV data
    csv_data = []
    for product, data in results['summary']['stock_levels'].items():
        csv_data.append({
            'Product': product,
            'Current_Count': data.get('current_count', 0),
            'Missing_Count': data.get('missing_count', 0),
            'Full_Capacity': data.get('full_capacity', data.get('current_count', 0) + data.get('missing_count', 0)),
            'Stock_Percentage': data.get('stock_percentage', 0)
        })
    
    df = pd.DataFrame(csv_data)
    csv_str = df.to_csv(index=False)
    
    st.download_button(
        label="Download CSV",
        data=csv_str,
        file_name=f"shelf_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def export_report(results):
    """Export detailed text report"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    summary = results['summary']
    
    report = f"""INTELLIGENT SHELF ANALYSIS REPORT
{'='*50}
Generated: {timestamp}

OVERVIEW:
Total Products: {summary.get('total_products_detected', 0)}
Overall Stock: {summary.get('overall_stock_percentage', 0):.1f}%

PRODUCT INVENTORY:
"""
    
    for product, data in summary.get('stock_levels', {}).items():
        stock_pct = data.get('stock_percentage', 0)
        status = "GOOD" if stock_pct >= 90 else "MODERATE" if stock_pct >= 70 else "LOW"
        current_count = data.get('current_count', 0)
        report += f"• {product}: {current_count} items ({stock_pct:.1f}% - {status})\n"
    
    st.download_button(
        label="Download Report",
        data=report,
        file_name=f"shelf_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

def display_results(results):
    """Display analysis results with visualizations"""
    st.header("📊 Analysis Results")
    
    # Show the complete detection visualization first
    if st.session_state.current_image_path and os.path.exists(st.session_state.current_image_path):
        display_visualization_results(results, st.session_state.current_image_path)
    
    # Overview metrics
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    
    summary = results.get('summary', {})
    
    with col1:
        st.metric("Total Products", summary.get('total_products_detected', 0))
    
    with col2:
        st.metric("Missing Products", summary.get('estimated_missing_products', 0))
    
    with col3:
        st.metric("Overall Stock", f"{summary.get('overall_stock_percentage', 0):.1f}%")
    
    # Stock level analysis
    st.subheader("📈 Stock Level Analysis")
    
    # Create stock level visualization
    stock_data = []
    for product, data in summary.get('stock_levels', {}).items():
        stock_data.append({
            'Product': product,
            'Current': data.get('current_count', 0),
            'Missing': data.get('missing_count', 0),
            'Stock Percentage': data.get('stock_percentage', 0)
        })
    
    df_stock = pd.DataFrame(stock_data)
    
    # Stock level bar chart
    fig_stock = px.bar(
        df_stock,
        x='Product',
        y=['Current', 'Missing'],
        title="Product Inventory Status",
        color_discrete_map={'Current': '#28a745', 'Missing': '#dc3545'}
    )
    st.plotly_chart(fig_stock, use_container_width=True)
    
    # FIXED: Dynamic stock percentage gauge charts
    st.subheader("📊 Stock Level Gauges")
    
    # Get the number of products
    num_products = len(summary.get('stock_levels', {}))
    
    if num_products == 0:
        st.warning("No products detected in the analysis.")
        return
    
    # Determine optimal number of columns (max 3 per row)
    cols_per_row = min(3, num_products)
    
    # Create rows of gauge charts
    products_list = list(summary.get('stock_levels', {}).items())
    
    for row_start in range(0, num_products, cols_per_row):
        # Create columns for this row
        cols = st.columns(cols_per_row)
        
        # Fill the columns with gauge charts
        for i in range(cols_per_row):
            product_index = row_start + i
            if product_index < num_products:
                product, data = products_list[product_index]
                
                with cols[i]:
                    stock_percentage = data.get('stock_percentage', 0)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=stock_percentage,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': f"{product.title()} Stock Level"},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 50], 'color': "lightgray"},
                                {'range': [50, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 90
                            }
                        }
                    ))
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # FIXED: Show detailed information for ALL products
    st.subheader("📦 Detailed Product Information")
    
    # Create expandable sections for each product
    for product, data in summary.get('stock_levels', {}).items():
        with st.expander(f"📦 {product.title()} Details", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            # Get values with fallbacks
            current_count = data.get('current_count', 0)
            missing_count = data.get('missing_count', 0)
            stock_percentage = data.get('stock_percentage', 0)
            
            # Calculate full_capacity if not provided
            full_capacity = data.get('full_capacity', current_count + missing_count)
            
            with col1:
                st.metric("Current Count", current_count)
            with col2:
                st.metric("Missing Count", missing_count)
            with col3:
                st.metric("Full Capacity", full_capacity)
            with col4:
                # Color code based on stock level
                if stock_percentage >= 90:
                    st.markdown(f"<span class='stock-level-good'>{stock_percentage:.1f}% GOOD</span>", unsafe_allow_html=True)
                elif stock_percentage >= 70:
                    st.markdown(f"<span class='stock-level-warning'>{stock_percentage:.1f}% MODERATE</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span class='stock-level-danger'>{stock_percentage:.1f}% LOW</span>", unsafe_allow_html=True)
    
    # Alternative: Show all products in a table format
    st.subheader("📋 Product Summary Table")
    
    # Create a comprehensive table with all product information
    table_data = []
    for product, data in summary.get('stock_levels', {}).items():
        stock_percentage = data.get('stock_percentage', 0)
        current_count = data.get('current_count', 0)
        missing_count = data.get('missing_count', 0)
        full_capacity = data.get('full_capacity', current_count + missing_count)
        
        status = "🟢 GOOD" if stock_percentage >= 90 else "🟡 MODERATE" if stock_percentage >= 70 else "🔴 LOW"
        table_data.append({
            'Product': product.title(),
            'Current Count': current_count,
            'Missing Count': missing_count,
            'Full Capacity': full_capacity,
            'Stock %': f"{stock_percentage:.1f}%",
            'Status': status
        })
    
    df_summary = pd.DataFrame(table_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Export options
    st.subheader("📤 Export Results")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        export_json(results)
    
    with col2:
        export_csv(results)
    
    with col3:
        export_report(results)

def process_image(uploaded_image):
    """Process uploaded image through the pipeline"""
    st.header("🖼️ Image Analysis")
    
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Original Image")
        image = Image.open(uploaded_image)
        st.image(image, use_column_width=True)
    
    # Process button
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Processing image... This may take a moment."):
            tmp_file_path = None
            try:
                # Method 1: Use delete=False and manual cleanup with retry
                with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                    tmp_file_path = tmp_file.name
                    image.save(tmp_file_path)
                
                # Store the image path for visualization
                st.session_state.current_image_path = tmp_file_path
                
                # Process image through pipeline
                results = st.session_state.pipeline.detect_and_classify_complete(tmp_file_path)
                st.session_state.results = results
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return
            
            finally:
                # Don't delete the temp file immediately as we need it for visualization
                # It will be cleaned up when the session ends or a new image is processed
                pass
            
            st.success("✅ Analysis completed!")
    
    # Display results if available
    if st.session_state.results:
        display_results(st.session_state.results)

def process_video(uploaded_video):
    """Process uploaded video through the pipeline"""
    st.header("🎥 Video Analysis")
    
    # Video player
    st.video(uploaded_video)
    
    # Frame extraction options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        frame_interval = st.number_input(
            "Frame Interval (seconds)",
            min_value=1,
            max_value=60,
            value=5,
            help="Extract frames every N seconds"
        )
    
    with col2:
        max_frames = st.number_input(
            "Max Frames to Process",
            min_value=1,
            max_value=100,
            value=10,
            help="Maximum number of frames to analyze"
        )
    
    with col3:
        analyze_mode = st.selectbox(
            "Analysis Mode",
            ["Single Frame", "Multiple Frames", "Trend Analysis"],
            help="Choose how to analyze the video"
        )
    
    if st.button("🎬 Analyze Video", type="primary"):
        with st.spinner("Processing video frames..."):
            # Extract frames and process them
            frames_results = extract_and_process_frames(uploaded_video, frame_interval, max_frames)
            st.session_state.results = frames_results
            st.success("✅ Video analysis completed!")
    
    # Display results if available
    if st.session_state.results:
        if analyze_mode == "Trend Analysis":
            display_trend_analysis(st.session_state.results)
        else:
            display_results(st.session_state.results if isinstance(st.session_state.results, dict) else st.session_state.results[0])

# Header
st.markdown("""
<div class="main-header">
    <h1>🛒 Intelligent Retail Shelf Analysis</h1>
    <p>Upload an image or video to analyze product inventory and get restocking recommendations</p>
</div>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model paths configuration
    st.subheader("Model Paths")
    yolo_model_path = st.text_input(
        "YOLO Model Path",
        value="models/individual_products.pt",
        help="Path to your YOLO model for product detection"
    )
    
    cnn_model_path = st.text_input(
        "CNN Model Path",
        value="models/classifier/best_lightweight_cnn.pth",
        help="Path to your CNN model for product classification"
    )
    
    void_model_path = st.text_input(
        "Void Model Path",
        value="models/void.pt",
        help="Path to your void detection model"
    )
    
    # Class names
    st.subheader("Product Classes")

    # Function to load class names from JSON
    @st.cache_data
    def load_class_names_from_json(json_path):
        """Load class names from saved model info JSON file"""
        try:
            import json
            with open(json_path, 'r') as f:
                model_info = json.load(f)
            return model_info.get('class_names', [])
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return []

    # Try to load class names from JSON file
    json_path = 'models/classifier/model_info.json'  # Update this path
    loaded_class_names = load_class_names_from_json(json_path)

    # Use loaded class names as default, or fallback to manual input
    if loaded_class_names:
        default_value = ','.join(loaded_class_names)
        st.info(f"Loaded {len(loaded_class_names)} class names from model info file")
    else:
        default_value = "cocacola,oil,water"
        st.warning("Could not load class names from JSON. Using default values.")

    class_names_input = st.text_input(
        "Class Names (comma-separated)",
        value=default_value,
        help="Enter product class names separated by commas. These were automatically loaded from your saved model info."
    )

    class_names = [name.strip() for name in class_names_input.split(',') if name.strip()]

    # Show current class names
    st.write(f"Current classes ({len(class_names)}): {class_names}")
    # Confidence thresholds
    st.subheader("Detection Thresholds")
    confidence_threshold = st.slider(
        "Product Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        
        step=0.05,
        help="Minimum confidence for product detection"
    )
    
    void_confidence_threshold = st.slider(
        "Void Detection Confidence",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence for void detection"
    )
    
    # Visualization options
    st.subheader("Visualization Options")
    show_visualization = st.checkbox(
        "Show Detection Visualization",
        value=True,
        help="Display the image with detection boxes and analysis charts"
    )
    
    # Initialize pipeline button
    if st.button("🚀 Initialize Pipeline", type="primary"):
        try:
            pipeline = EnhancedRetailPipeline(
                yolo_model_path=yolo_model_path,
                cnn_model_path=cnn_model_path,
                void_model_path=void_model_path,
                class_names=class_names,
                confidence_threshold=confidence_threshold,
                void_confidence_threshold=void_confidence_threshold
            )
            st.session_state.pipeline = pipeline
            st.success("✅ Pipeline initialized successfully!")
        except Exception as e:
            st.error(f"❌ Error initializing pipeline: {str(e)}")

# Main content area
if st.session_state.pipeline is None:
    st.warning("⚠️ Please configure and initialize the pipeline in the sidebar first.")
else:
    # File upload section
    st.header("📁 Upload Media")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_image = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a shelf image for analysis"
        )
    
    with col2:
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video for frame-by-frame analysis"
        )
    
    # Process uploaded content
    if uploaded_image is not None:
        process_image(uploaded_image)
    elif uploaded_video is not None:
        process_video(uploaded_video)

# Footer
st.markdown("---")
st.markdown("🛒 **Intelligent Retail Shelf Analysis** - Powered by Computer Vision & AI")

# Cleanup function for session end
def cleanup_temp_files():
    """Clean up temporary files when session ends"""
    if st.session_state.current_image_path and os.path.exists(st.session_state.current_image_path):
        try:
            os.unlink(st.session_state.current_image_path)
        except:
            pass  # Ignore cleanup errors

# Register cleanup function
import atexit
atexit.register(cleanup_temp_files)