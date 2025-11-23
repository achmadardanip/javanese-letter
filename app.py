import streamlit as st
from streamlit_drawable_canvas import st_canvas
from autogluon.multimodal import MultiModalPredictor
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import os

# ==========================
# 1. CONFIG & LOAD MODEL
# ==========================
st.set_page_config(page_title="Javanese OCR Sketchpad", layout="wide")
st.title("✍️ Javanese Script Real-Time OCR")
st.markdown("Draw a Javanese character or word below and click **Predict**.")

MODEL_PATH = "artifacts"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return MultiModalPredictor.load(MODEL_PATH)
    else:
        return None

predictor = load_model()

if predictor is None:
    st.error(f"Model not found at '{MODEL_PATH}'. Please run the training notebook first.")
    st.stop()

# ==========================
# 2. HELPER FUNCTIONS (Logic from Notebook)
# ==========================

def merge_javanese_chars(chars):
    """Heuristic to fix split characters (e.g., da+ra -> ba)."""
    if not chars: return ""
    merged = []
    i = 0
    while i < len(chars):
        curr = chars[i]
        nxt = chars[i+1] if i+1 < len(chars) else None
        
        # Rule: 'da' + 'ra' -> 'ba'
        if curr == 'da' and nxt == 'ra':
            merged.append('ba'); i += 2; continue
        # Rule: 'nga' + 'nga' -> 'nga'
        if curr == 'nga' and nxt == 'nga':
            merged.append('nga'); i += 2; continue
        # Rule: 'tha' + 'nya' -> 'tha'
        if curr == 'tha' and nxt == 'nya':
            merged.append('tha'); i += 2; continue
        # Rule: 'na' + 'ya' -> 'nya'
        if curr == 'na' and nxt == 'ya':
            merged.append('nya'); i += 2; continue
            
        merged.append(curr)
        i += 1
    return "".join(merged)

def process_sketch(image_data):
    """
    Takes raw canvas data (RGBA), converts to clean crops for the model.
    """
    # 1. Convert to numpy array (RGBA)
    img = np.array(image_data)
    
    # 2. Convert to Grayscale for processing
    # Canvas is white background, black ink usually. 
    # If transparent, we need to handle alpha.
    if img.shape[2] == 4:
        # Create white background
        background = np.ones_like(img) * 255
        alpha = img[:, :, 3] / 255.0
        # Blend
        for c in range(3):
            background[:, :, c] = (1.0 - alpha) * 255 + alpha * img[:, :, c]
        gray = cv2.cvtColor(background.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 3. Threshold (Inverse so ink is white, bg is black for contours)
    _, bin_img = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 4. Find Contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 10 and h > 10: # Filter dots
            boxes.append((x, y, w, h))
    
    boxes.sort(key=lambda b: b[0]) # Left to Right

    crops = []
    pad = 10
    h_img, w_img = gray.shape
    
    # To feed to AutoGluon, we need clean RGB images of the crops
    clean_img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    for (x, y, w, h) in boxes:
        y1 = max(0, y-pad); y2 = min(h_img, y+h+pad)
        x1 = max(0, x-pad); x2 = min(w_img, x+w+pad)
        
        crop = clean_img_rgb[y1:y2, x1:x2]
        pil_img = Image.fromarray(crop)
        crops.append(pil_img)
        
    return crops, boxes

# ==========================
# 3. UI LAYOUT
# ==========================

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Draw Here:")
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 1)",  # Fixed fill color with some opacity
        stroke_width=5,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=600,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction:")
    
    if st.button("Recognize Text", type="primary"):
        if canvas_result.image_data is not None:
            
            # 1. Segmentation
            crops, boxes = process_sketch(canvas_result.image_data)
            
            if len(crops) == 0:
                st.warning("No drawing detected! Please write clearly.")
            else:
                # 2. Predict
                # AutoGluon accepts a dataframe containing PIL images in 'image' column
                batch_df = pd.DataFrame({'image': crops})
                raw_preds = predictor.predict(batch_df).tolist()
                
                # 3. Merge Heuristics
                final_word = merge_javanese_chars(raw_preds)
                
                # 4. Display Result
                st.success(f"## Result: **{final_word}**")
                
                # 5. Debug Info (Show individual detections)
                with st.expander("See character breakdown"):
                    st.write("Raw Detections:", raw_preds)
                    
                    # Display crops
                    st.write("Segmented Characters:")
                    cols = st.columns(len(crops))
                    for i, (crop, pred) in enumerate(zip(crops, raw_preds)):
                        with cols[i]:
                            st.image(crop, caption=pred, use_column_width=True)