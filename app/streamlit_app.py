"""
EdgeSight – Streamlit Web Application
======================================
Task 1: Vehicle Detection
Run: streamlit run app/streamlit_app.py
"""

import sys
import time
import json
import base64
from pathlib import Path
from io import BytesIO

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Make src importable from app/
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.detector import VehicleDetector
from src.metrics import MetricsEngine
from src.utils import get_logger

logger = get_logger("EdgeSight-UI")
CONFIG = Path(__file__).parent.parent / "configs" / "vehicle_detection.yaml"

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EdgeSight – Vehicle Detection",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0a0e1a; }

/* Gradient header */
.es-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.es-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(139,92,246,0.3) 0%, transparent 70%);
    border-radius: 50%;
}
.es-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
}
.es-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 0.3rem;
}
.es-badge {
    display: inline-block;
    background: rgba(139,92,246,0.2);
    border: 1px solid rgba(139,92,246,0.4);
    color: #a78bfa;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(100,116,139,0.3);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: rgba(139,92,246,0.5); }
.metric-label { color: #64748b; font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase; }
.metric-value { color: #e2e8f0; font-size: 1.8rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; }
.metric-unit  { color: #94a3b8; font-size: 0.8rem; }

/* Detection row */
.det-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.6rem 0.8rem;
    margin: 0.3rem 0;
    background: rgba(30,41,59,0.6);
    border-radius: 8px;
    border-left: 3px solid #a78bfa;
}
.det-class  { font-weight: 600; color: #e2e8f0; font-size: 0.9rem; }
.det-conf   { font-family: 'JetBrains Mono', monospace; color: #34d399; font-size: 0.85rem; }
.det-bbox   { font-family: 'JetBrains Mono', monospace; color: #94a3b8; font-size: 0.78rem; }

.class-car        { border-left-color: #fbbf24 !important; }
.class-motorcycle { border-left-color: #34d399 !important; }
.class-bus        { border-left-color: #60a5fa !important; }
.class-truck      { border-left-color: #c084fc !important; }

/* JSON block */
.json-pre {
    background: #0d1117;
    border: 1px solid rgba(100,116,139,0.25);
    border-radius: 10px;
    padding: 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    color: #7dd3fc;
    overflow-x: auto;
    max-height: 340px;
    overflow-y: auto;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #0d1117 !important; }
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }

/* ── Compact upload widget ─────────────────────────────────── */
[data-testid="stFileUploader"] {
    max-width: 420px !important;
    background: rgba(15,23,42,0.8);
    border: 1.5px dashed rgba(139,92,246,0.4);
    border-radius: 12px;
    padding: 0.4rem 0.8rem !important;
}
[data-testid="stFileUploader"] section {
    padding: 0.4rem !important;
    min-height: unset !important;
}
[data-testid="stFileUploader"] section > div {
    min-height: 60px !important;
    max-height: 80px !important;
}
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] {
    padding: 0.25rem 0.75rem !important;
    font-size: 0.8rem !important;
}
/* Cap image display */
[data-testid="stImage"] img {
    max-height: 400px !important;
    object-fit: contain;
    border-radius: 10px;
}
/* Cap video player */
[data-testid="stVideo"] video {
    max-height: 300px !important;
    border-radius: 10px;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading YOLOv8n model…")
def load_detector(conf: float, device: str) -> VehicleDetector:
    det = VehicleDetector.from_config(CONFIG)
    det.conf = conf
    det.device = device
    return det


# ─── Helper: BGR numpy → PIL ──────────────────────────────────────────────────
def bgr_to_pil(img: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="es-header">
  <div class="es-badge">TASK 1 · VEHICLE DETECTION</div>
  <h1 class="es-title">◈ EdgeSight</h1>
  <p class="es-sub">Edge-Optimised Vehicle Intelligence System &nbsp;·&nbsp; YOLOv8n · OpenCV · EasyOCR · ONNX</p>
</div>
""", unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Detection Config")

    conf_thresh = st.slider(
        "Confidence Threshold", 0.10, 0.95, 0.45, 0.05,
        help="Minimum confidence to display a detection."
    )
    iou_thresh = st.slider(
        "IoU Threshold (NMS)", 0.10, 0.95, 0.45, 0.05,
        help="Non-max suppression overlap threshold."
    )
    device = st.selectbox("Inference Device", ["cpu", "0 (GPU)"], index=0)
    device_str = "cpu" if device.startswith("cpu") else "0"

    st.markdown("---")
    st.markdown("## 🎯 Target Classes")
    use_car  = st.checkbox("🚗 Car",        value=True)
    use_moto = st.checkbox("🏍️ Motorcycle", value=True)
    use_bus  = st.checkbox("🚌 Bus",        value=True)
    use_truck= st.checkbox("🚛 Truck",      value=True)

    target_map = {0: "car", 1: "motorcycle", 2: "bus", 3: "truck"}
    enabled = {0: use_car, 1: use_moto, 2: use_bus, 3: use_truck}
    active_classes = {k: v for k, v in target_map.items() if enabled[k]}

    st.markdown("---")
    st.markdown("## 📊 Model Info")
    st.info("**Model**: Custom YOLOv8n (Vehicle + Plate OCR)  \n**Size**: ~6 MB  \n**GFLOPs**: ~3.2  \n**Framework**: PyTorch")

    run_benchmark = st.button("⚡ Run Speed Benchmark")
    
    st.markdown("---")
    plate_log_container = st.empty()


# ─── Main tabs ────────────────────────────────────────────────────────────────
tab_img, tab_video, tab_metrics = st.tabs(
    ["🖼️  Image Detection", "🎥  Video / Webcam", "📈  Metrics & Benchmark"]
)


# ════════════════════════════════════════════════════════════════════════
# TAB 1: IMAGE DETECTION
# ════════════════════════════════════════════════════════════════════════
with tab_img:
    # ── Compact uploader: 1/3 width ──────────────────────────────────
    up_col, _ = st.columns([1, 2])
    with up_col:
        uploaded = st.file_uploader(
            "Upload an image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="visible",
        )

    if uploaded:
        # Decode
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("Running YOLOv8n inference…"):
            detector = load_detector(conf_thresh, device_str)
            detector.conf = conf_thresh
            detector.iou  = iou_thresh
            detector.target_classes = active_classes

            t0 = time.perf_counter()
            result = detector.detect_frame(img_bgr)
            wall_ms = (time.perf_counter() - t0) * 1000

        annotated = detector.draw_detections(img_bgr, result)

        # ── Side-by-side: original | annotated — fixed 480 px each ───
        col_orig, col_result = st.columns(2, gap="medium")
        with col_orig:
            st.markdown("**📷 Original**")
            st.image(bgr_to_pil(img_bgr), width=480)
        with col_result:
            st.markdown("**◉ Detections**")
            st.image(bgr_to_pil(annotated), width=480)
            
        # Update Plate Log
        plate_logs = [d.plate_text for d in result.detections if getattr(d, 'plate_text', None)]
        if plate_logs:
            with plate_log_container.container():
                st.markdown("### 🏷️ Detected Plates")
                for p in plate_logs:
                    st.markdown(f"`{p}`")

        # ── Metric strip ──────────────────────────────────────────────────
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Vehicles Found</div>
                <div class="metric-value">{result.count}</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Inference Time</div>
                <div class="metric-value">{result.inference_time_ms:.1f}</div>
                <div class="metric-unit">ms</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Est. FPS</div>
                <div class="metric-value">{1000/max(result.inference_time_ms,1):.1f}</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-label">Model Size</div>
                <div class="metric-value">6.2</div>
                <div class="metric-unit">MB</div>
            </div>""", unsafe_allow_html=True)
            
        if plate_logs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.success(f"🚙 **License Plate(s) Detected:** {', '.join(plate_logs)}")

        # ── Detection list ────────────────────────────────────────────────
        st.markdown("#### 🔍 Detection Details")
        color_map = {
            "car": "class-car", "motorcycle": "class-motorcycle",
            "bus": "class-bus", "truck": "class-truck",
        }
        if result.detections:
            for i, det in enumerate(result.detections, 1):
                cls = color_map.get(det.class_name, "")
                x1,y1,x2,y2 = det.bbox
                plate_span = f'<span style="background:rgba(167,139,250,0.2);color:#a78bfa;padding:2px 8px;border-radius:12px;font-size:0.8rem;margin-left:auto;">{det.plate_text}</span>' if det.plate_text else ""
                st.markdown(f"""
                <div class="det-row {cls}">
                  <span style="color:#64748b;font-size:0.8rem">#{i}</span>
                  <span class="det-class">{det.class_name.upper()}</span>
                  <span class="det-conf">{det.confidence:.1%}</span>
                  <span class="det-bbox">[{x1}, {y1}, {x2}, {y2}]</span>
                  {plate_span}
                </div>""", unsafe_allow_html=True)
        else:
            st.info("No vehicles detected. Try lowering the confidence threshold.")

        # ── JSON output ───────────────────────────────────────────────────
        with st.expander("📄 Raw JSON Output"):
            json_str = json.dumps(result.to_dict(), indent=2)
            st.markdown(f'<pre class="json-pre">{json_str}</pre>', unsafe_allow_html=True)
            st.download_button(
                "⬇ Download JSON", json_str,
                file_name="detection_result.json", mime="application/json"
            )

        # ── Download annotated image ──────────────────────────────────────
        buf = BytesIO()
        bgr_to_pil(annotated).save(buf, format="JPEG", quality=92)
        st.download_button(
            "⬇ Download Annotated Image",
            buf.getvalue(),
            file_name="edgesight_result.jpg",
            mime="image/jpeg",
        )
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#475569;">
          <div style="font-size:3rem;margin-bottom:1rem;">🚗</div>
          <p style="font-size:1.1rem;">Upload an image above to start vehicle detection</p>
          <p style="font-size:0.85rem;">Supports: JPG, PNG, BMP, WebP</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 2: VIDEO
# ════════════════════════════════════════════════════════════════════════
with tab_video:
    # ── Compact uploader (1/3 width) + slider side by side ──────────
    vup_col, vslider_col, _ = st.columns([1, 1, 1])
    with vup_col:
        video_file = st.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="visible",
        )
    with vslider_col:
        st.markdown("<br>", unsafe_allow_html=True)
        skip_n = st.slider("Skip N frames", 0, 5, 0)

    if video_file:
        # Save uploaded video to a temp file
        tmp_dir = Path("/tmp/edgesight_upload")
        tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / video_file.name
        out_path = tmp_dir / (Path(video_file.name).stem + "_annotated.mp4")
        tmp_path.write_bytes(video_file.read())

        # ── Show original video in a half-width column ────────────────
        vcol, _ = st.columns([1, 1])
        with vcol:
            st.markdown("**📹 Original**")
            st.video(str(tmp_path))

        if st.button("▶ Run Detection on Video", type="primary"):
            detector = load_detector(conf_thresh, device_str)
            detector.conf = conf_thresh
            detector.iou  = iou_thresh
            detector.target_classes = active_classes

            # ── Read video properties ─────────────────────────────────────
            cap   = cv2.VideoCapture(str(tmp_path))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps   = cap.get(cv2.CAP_PROP_FPS) or 25
            w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # ── Video writer for annotated output ─────────────────────────
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            progress    = st.progress(0, text="Processing frames…")
            live_frame  = st.empty()   # live annotated preview
            main_plate_display = st.empty() # live plate banner
            status_bar  = st.empty()   # live stats

            all_results = []
            frame_id    = 0
            total_vehicles = 0
            inf_times   = []
            seen_plates = set()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_id % (skip_n + 1) == 0:
                    # ── Detect ────────────────────────────────────────────
                    result = detector.detect_frame(frame, frame_id=frame_id)
                    annotated = detector.draw_detections(frame, result)
                    all_results.append(result.to_dict())
                    total_vehicles = max(total_vehicles, result.count)
                    inf_times.append(result.inference_time_ms)
                    
                    # Update Plate Log
                    new_plates = [d.plate_text for d in result.detections if getattr(d, 'plate_text', None) and d.plate_text not in seen_plates]
                    if new_plates:
                        seen_plates.update(new_plates)
                        main_plate_display.success(f"🚙 **Latest License Plate Detected:** {new_plates[-1]}")
                        with plate_log_container.container():
                            st.markdown("### 🏷️ Detected Plates")
                            for p in reversed(list(seen_plates)):
                                st.markdown(f"`{p}`")

                    # ── Write annotated frame to output video ──────────────
                    writer.write(annotated)

                    # ── Live preview every 5th processed frame ────────────
                    if (frame_id // (skip_n + 1)) % 5 == 0:
                        live_frame.image(
                            bgr_to_pil(annotated),
                            caption=f"Frame {frame_id} · {result.count} vehicle(s) · {result.inference_time_ms:.0f} ms",
                            width=520,
                        )
                else:
                    # Write original frame for skipped frames
                    writer.write(frame)

                pct = min(int((frame_id / total) * 100), 100)
                progress.progress(pct, text=f"Frame {frame_id}/{total} — {pct}%")
                status_bar.markdown(
                    f"🚗 **Peak vehicles**: `{total_vehicles}` &nbsp;·&nbsp; "
                    f"⚡ **Avg inference**: `{sum(inf_times)/max(len(inf_times),1):.1f} ms`"
                )
                frame_id += 1

            cap.release()
            writer.release()
            live_frame.empty()

            progress.progress(100, text="Done ✓")

            # ── Summary metrics ───────────────────────────────────────────
            avg_ms  = sum(inf_times) / max(len(inf_times), 1)
            avg_fps = 1000 / max(avg_ms, 1)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Frames Processed</div><div class="metric-value">{len(inf_times)}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Peak Vehicles</div><div class="metric-value">{total_vehicles}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg Inference</div><div class="metric-value">{avg_ms:.0f}</div><div class="metric-unit">ms</div></div>', unsafe_allow_html=True)
            with m4:
                st.markdown(f'<div class="metric-card"><div class="metric-label">Avg FPS</div><div class="metric-value">{avg_fps:.1f}</div></div>', unsafe_allow_html=True)

            # ── Show annotated video in half-width column ───────────────
            if out_path.exists():
                res_col, _ = st.columns([1, 1])
                with res_col:
                    st.markdown("**◉ Annotated Video (with Bounding Boxes)**")
                    st.video(str(out_path))

            # ── Downloads ─────────────────────────────────────────────────
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                json_str = json.dumps(all_results, indent=2)
                st.download_button(
                    "⬇ Download Detection JSON", json_str,
                    file_name="video_detections.json", mime="application/json"
                )
            with col_dl2:
                if out_path.exists():
                    with open(out_path, "rb") as vf:
                        st.download_button(
                            "⬇ Download Annotated Video", vf.read(),
                            file_name="edgesight_annotated.mp4", mime="video/mp4"
                        )
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem;color:#475569;">
          <div style="font-size:3rem;margin-bottom:1rem;">🎥</div>
          <p>Upload a video file to run frame-by-frame detection</p>
          <p style="font-size:0.85rem;">Supports: MP4, AVI, MOV, MKV</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════
# TAB 3: METRICS
# ════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("### 📈 Performance Metrics – Task 1")

    col_info, col_model = st.columns([2, 1])
    with col_info:
        st.markdown("""
        **Evaluation Deliverables (Task 1)**
        | Metric | Description |
        |--------|-------------|
        | **mAP@50** | Mean Average Precision at IoU=0.50 |
        | **Inference Time (ms)** | Per-frame model runtime |
        | **FPS** | Frames processed per second |
        | **Model Size (MB)** | On-disk weight file size |
        """)
    with col_model:
        st.markdown("""
        **YOLOv8n Specs**
        - GFLOPs: ~3.2
        - Params: ~3.2M
        - Size: ~6 MB
        - Classes: car, motorcycle, bus, truck
        """)

    if run_benchmark:
        st.info("📋 Benchmark requires a sample image. Upload one in the **Image Detection** tab first, then click here.")

    metrics_path = Path("outputs/metrics.json")
    if metrics_path.exists():
        with open(metrics_path) as f:
            saved = json.load(f)
        st.markdown("#### 📄 Saved Metrics Report")
        st.json(saved)
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem;color:#475569;">
          <p>No saved metrics yet.</p>
          <p style="font-size:0.85rem;">Run: <code>python main.py --source &lt;image&gt; --metrics</code></p>
        </div>
        """, unsafe_allow_html=True)
