# app.py

import streamlit as st
import pandas as pd
import cv2
import numpy as np
import tempfile
import matplotlib.pyplot as plt
from keras.models import load_model
import os
import io
from datetime import datetime

# Try to import reportlab for PDF creation; if not available we'll fallback to CSV/text downloads
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---------------- Page config & CSS (dark theme approx) ----------------
st.set_page_config(page_title="AegisView — Crack Monitoring", layout="wide", initial_sidebar_state="expanded")

# Minimal dark theme CSS to make app look closer to your reference.
st.markdown(
    """
    <style>
    :root { color-scheme: dark; }
    .stApp {
        background: radial-gradient(circle at top left, #0f2027, #203a43, #2c5364);
        color: #e6eef2;
    }
    .report-card {
        background-color: rgba(15,20,25,0.45);
        padding: 12px;
        border-radius: 10px;
    }
    .label-red { color: #ff6b6b; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Load model (single model for both modes) ----------------
@st.cache_resource(show_spinner=False)
def load_crack_model(path="best_crack_detector.keras"):
    return load_model(path)

with st.spinner("Loading model..."):
    model = load_crack_model("best_crack_detector.keras")
    CONFIDENCE_THRESHOLD = 0.002

# ---------------- Sidebar controls (calibration + mode) ----------------
st.sidebar.title("Controls")
PIXEL_TO_MM = st.sidebar.slider("Pixel → mm (calibration)", 0.01, 0.2, 0.05, step=0.005)
DEPTH_FACTOR = st.sidebar.slider("Depth factor (empirical)", 0.1, 2.0, 0.8, step=0.05)
#CONFIDENCE_THRESHOLD = st.sidebar.slider("Detection confidence threshold", 0.1, 0.9, 0.02, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("App mode:")
mode = st.sidebar.radio("Select input type", ("Video - Structural", "Image - Land"))

# Optional: debug toggle
DEBUG = st.sidebar.checkbox("Show debug frames", value=False)

# ---------------- Utility functions ----------------
def classify_seriousness(depth_mm):
    if depth_mm < 1:
        return "Minor"
    elif depth_mm < 2.2:
        return "Moderate"
    else:
        return "Severe"

def predict_frame_crack(frame, model, px_to_mm, depth_factor, conf_thresh):
    target_h, target_w = model.input_shape[1], model.input_shape[2]
    resized = cv2.resize(frame, (target_w, target_h))

    input_img = resized / 255.0
    input_img = np.expand_dims(input_img, axis=0)

    # Get actual model confidence
    pred = float(model.predict(input_img, verbose=0)[0][0])
    #st.write("DEBUG CONFIDENCE:", pred)

    if pred >= conf_thresh:
     return None    # No crack detected



    # Edge detection
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    crack_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(crack_contour)

    crack_width_px = w
    crack_width_mm = crack_width_px * px_to_mm
    crack_depth_mm = crack_width_mm * depth_factor

    seriousness = classify_seriousness(crack_depth_mm)

    # IMPORTANT: return exactly 5 things
    return crack_depth_mm, seriousness, pred, (x, y, w, h), edges



# ---------------- Video processing with progress ----------------
def process_video(video_path, output_path, model, px_to_mm, depth_factor, conf_thresh, progress_callback=None, debug=False):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    # H.264 codec for compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    report_rows = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        result = predict_frame_crack(frame, model, px_to_mm, depth_factor, conf_thresh)

        if result is not None:
            crack_depth_mm, seriousness, confidence, bbox, edges = result
            text = f"Depth: {crack_depth_mm:.2f} mm | {seriousness}"
            cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,0,255), 2)
            x,y,w,h = bbox
            # draw bbox scaled back up to frame size? (resized was 227x170 - but bbox is on resized)
            # We'll draw a small marker to indicate detection (approx)
            cv2.rectangle(frame, (50,80), (50+200,120), (0,0,255), 2)

            report_rows.append({
                "Frame": frame_count,
                "Depth (mm)": round(crack_depth_mm, 2),
                "Seriousness": seriousness,
                "Confidence": round(confidence, 3)
            })

        # write annotated frame
        out.write(frame)

        # update progress
        if progress_callback and total_frames > 0:
            progress_callback(frame_count / total_frames)

    cap.release()
    out.release()
    #cv2.destroyAllWindows()
    df = pd.DataFrame(report_rows)
    return df

# ---------------- Image processing ----------------
def process_image_file(image_bytes, model, px_to_mm, depth_factor, conf_thresh):
    # Read image bytes into OpenCV image
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None, "Could not read image"

    result = predict_frame_crack(img, model, px_to_mm, depth_factor, conf_thresh)
    if result is None:
        return None, None

    crack_depth_mm, seriousness, confidence, bbox, edges = result
    # annotate a copy for display
    disp = img.copy()
    text = f"Depth: {crack_depth_mm:.2f} mm | {seriousness}"
    cv2.putText(disp, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # convert BGR->RGB for display in Streamlit
    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
    return {
        "annotated_image": disp,
        "depth": round(crack_depth_mm, 2),
        "seriousness": seriousness,
        "confidence": round(confidence, 3)
    }, None

# ---------------- Report creation helpers ----------------

def save_depth_graph(df):
    plt.figure(figsize=(6, 3))
    plt.plot(df["Frame"], df["Depth (mm)"])
    plt.title("Depth vs Frame")
    plt.xlabel("Frame")
    plt.ylabel("Depth (mm)")
    graph_path = "depth_graph.png"
    plt.savefig(graph_path, dpi=150, bbox_inches="tight")
    plt.close()
    return graph_path


def make_csv_bytes(df, meta=None):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    if meta:
        buf_meta = io.StringIO()
        for k, v in meta.items():
            buf_meta.write(f"# {k}: {v}\n")
        buf = io.StringIO(buf_meta.getvalue() + buf.getvalue())
    return buf.getvalue().encode("utf-8")

def make_pdf_bytes(df, meta=None, graph_path=None):
    """
    Create a PDF including table + optional graph image.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    width_a4, height_a4 = A4
    x_margin = 40
    y = height_a4 - 50

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x_margin, y, "AegisView — Crack Monitoring Report")

    # Date
    c.setFont("Helvetica", 10)
    y -= 20
    c.drawString(x_margin, y, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Metadata
    y -= 20
    if meta:
        for k, v in meta.items():
            c.drawString(x_margin, y, f"{k}: {v}")
            y -= 14

    # 🔥 Add Graph Here
    if graph_path:
        y -= 20
        c.setFont("Helvetica-Bold", 12)
        c.drawString(x_margin, y, "Depth vs Frame Graph:")
        y -= 10

        try:
            c.drawImage(graph_path, x_margin, y - 260, width=450, height=260, preserveAspectRatio=True)
            y -= 280
        except Exception as e:
            c.drawString(x_margin, y, f"[Graph could not be inserted: {e}]")
            y -= 20

    # Section header
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x_margin, y, "Detections")
    y -= 18

    # Table entries
    c.setFont("Helvetica", 10)
    for idx, row in df.iterrows():
        line = (
            f"Frame/Image: {row.get('Frame', '-')}  | "
            f"Depth(mm): {row.get('Depth (mm)', '-')}  | "
            f"Seriousness: {row.get('Seriousness', '-')}  | "
            f"Confidence: {row.get('Confidence', '-')}"
        )
        c.drawString(x_margin, y, line[:110])

        y -= 12

        # New page if space is low
        if y < 80:
            c.showPage()
            y = height_a4 - 50
            c.setFont("Helvetica", 10)

    c.save()
    buffer.seek(0)
    return buffer.read()

# ---------------- Main UI ----------------
st.title("🛠️ Crack Monitoring Dashboard")
st.markdown("Analyze structural cracks (video) **or** land cracks (image). Use the sidebar to set calibration and confidence.")

# Use container for main section to allow clearing/replacing
main_container = st.container()

with main_container:
    # Accordion-like sections using st.expander
    video_exp = st.expander("🔴 Video Analysis — Structural (Video Upload)", expanded=(mode=="Video - Structural"))
    image_exp = st.expander("🟢 Image Analysis — Land (Image Upload)", expanded=(mode=="Image - Land"))
    summary_exp = st.expander("📊 Summary & Downloads", expanded=True)

    # VIDEO branch
    with video_exp:
        if mode == "Video - Structural":
            uploaded_video = st.file_uploader("Upload structural inspection video", type=["mp4","mov","avi"], key="video_uploader")
            if uploaded_video:
                # Save to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_video.read())
                input_video_path = tfile.name
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

                st.info("Ready to process the uploaded video. Press Start when ready.")
                if st.button("Start Video Analysis"):
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()

                    def progress_cb(progress_val):
                        progress_bar.progress(min(1.0, progress_val))
                        status_text.text(f"Processing... {int(progress_val*100)}%")

                    df_video = process_video(
                        input_video_path,
                        output_video_path,
                        model=model,
                        px_to_mm=PIXEL_TO_MM,
                        depth_factor=DEPTH_FACTOR,
                        conf_thresh=CONFIDENCE_THRESHOLD,
                        progress_callback=progress_cb,
                        debug=DEBUG
                    )
                    progress_bar.progress(1.0)
                    status_text.success("Processing finished.")

                    if not df_video.empty:
                        st.success("✅ Crack analysis complete for video.")
                        st.write("Video saved. Size:", os.path.getsize(output_video_path), "bytes")
                        # Show video
                        st.subheader("🎥 Annotated Video")
                        with open(output_video_path, "rb") as f:
                            st.video(f.read(), format="video/mp4")

                        st.subheader("📄 Crack Report (Video)")
                        st.dataframe(df_video)

                        # plot
                        st.subheader("📈 Depth vs Frame")
                        fig, ax = plt.subplots(figsize=(8,3))
                        ax.plot(df_video["Frame"], df_video["Depth (mm)"], marker="o", linestyle="-")
                        ax.set_xlabel("Frame")
                        ax.set_ylabel("Depth (mm)")
                        ax.grid(True)
                        st.pyplot(fig)

                        # Provide downloads in summary expander by saving to session state
                        st.session_state["last_report_df"] = df_video
                        st.session_state["last_report_meta"] = {
                            "Input": "Video",
                            "FileName": os.path.basename(uploaded_video.name),
                            "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

                    else:
                        st.warning("⚠️ No cracks detected in the video.")
        else:
            st.write("Video mode disabled — switch to 'Video - Structural' in the sidebar.")

    # IMAGE branch
    with image_exp:
        if mode == "Image - Land":
            uploaded_img = st.file_uploader("Upload land-crack image (jpg/png)", type=["png","jpg","jpeg"], key="image_uploader")
            if uploaded_img:
                st.info("Image uploaded. Click 'Analyze Image' to run detection.")
                if st.button("Analyze Image"):
                    raw_bytes = uploaded_img.read()
                    result, err = process_image_file(raw_bytes, model, PIXEL_TO_MM, DEPTH_FACTOR, CONFIDENCE_THRESHOLD)
                    if err:
                        st.error(err)
                    elif result is None:
                        st.warning("No crack detected in the image (or below confidence threshold).")
                        st.session_state["last_report_df"] = pd.DataFrame()  # clear
                    else:
                        st.success("✅ Crack detected in image.")
                        st.image(result["annotated_image"], use_column_width=True, caption=f"Depth: {result['depth']} mm | {result['seriousness']}")
                        df_img = pd.DataFrame([{
                            "Frame": "Image",
                            "Depth (mm)": result["depth"],
                            "Seriousness": result["seriousness"],
                            "Confidence": result["confidence"]
                        }])
                        st.subheader("📄 Image Crack Report")
                        st.dataframe(df_img)

                        st.session_state["last_report_df"] = df_img
                        st.session_state["last_report_meta"] = {
                            "Input": "Image",
                            "FileName": os.path.basename(uploaded_img.name),
                            "Generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }

        else:
            st.write("Image mode disabled — switch to 'Image - Land' in the sidebar.")

    # SUMMARY & DOWNLOADS
    with summary_exp:
        last_df = st.session_state.get("last_report_df", pd.DataFrame())
        meta = st.session_state.get("last_report_meta", {})

        if last_df is None or last_df.empty:
            st.info("No recent detections to summarize. Run an analysis on an uploaded video or image first.")
        else:
            st.subheader("📊 Summary")
            try:
                max_depth = last_df["Depth (mm)"].max()
                avg_depth = last_df["Depth (mm)"].mean()
                # severity ranking
                sev_idx = last_df["Seriousness"].apply(lambda x: ["Minor","Moderate","Severe"].index(x)).max()
                most_serious_label = ["Minor","Moderate","Severe"][sev_idx]
            except Exception:
                max_depth = None
                avg_depth = None
                most_serious_label = last_df["Seriousness"].iloc[0] if "Seriousness" in last_df.columns else "-"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Max Depth (mm)", f"{max_depth:.2f}" if pd.notna(max_depth) else "-")
            with col2:
                st.metric("Average Depth (mm)", f"{avg_depth:.2f}" if pd.notna(avg_depth) else "-")
            with col3:
                if most_serious_label == "Severe":
                    st.markdown("<span class='label-red'>🚨 Most Serious: SEVERE</span>", unsafe_allow_html=True)
                else:
                    st.write(f"Most Serious: {most_serious_label}")

            # Offer CSV download always
            csv_bytes = make_csv_bytes(last_df, meta)
            st.download_button(
                label="⬇️ Download report (CSV)",
                data=csv_bytes,
                file_name=f"Crack_Severity_report_{meta.get('Generated', datetime.now().strftime('%Y%m%d_%H%M%S'))}.csv",
                mime="text/csv"
            )

            # Offer PDF if severe and reportlab available (or fallback to CSV)
            if most_serious_label == "Severe":
                st.error("🚨 ALERT: Severe crack detected! Immediate attention required.")
                if REPORTLAB_AVAILABLE:
                    pdf_bytes = make_pdf_bytes(last_df, meta)
                    if pdf_bytes:
                        st.download_button(
                            label="⬇️ Download PDF Report (Severe)",
                            data=pdf_bytes,
                            file_name=f"aegisview_report_{meta.get('Generated', datetime.now().strftime('%Y%m%d_%H%M%S'))}.pdf",
                            mime="application/pdf"
                        )
                    else:
                        st.warning("Could not create PDF; offering CSV instead.")
                else:
                    st.warning("PDF generation library not installed. Install `reportlab` to enable PDF reports.")
                    # still offer CSV (already above)

# ---------------- Footer / Helpful tips ----------------
st.markdown("---")
st.markdown("**Tips:** Use the sidebar sliders to calibrate Pixel→mm and Depth factor for your camera setup. Increase the confidence threshold to reduce false positives.")
