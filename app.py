import os
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from autogluon.multimodal import MultiModalPredictor

# =====================================
# 1. CONFIG – POINT TO YOUR AG MODEL
# =====================================
MODEL_PATH = "artifacts"  # same path you used in training script
TMP_DIR = Path("streamlit_tmp")
TMP_DIR.mkdir(exist_ok=True)


def clear_tmp_dir():
    """Remove old PNGs so we don't accumulate junk."""
    for p in TMP_DIR.glob("*.png"):
        try:
            p.unlink()
        except Exception:
            pass


# =====================================
# 2. LOAD AUTOGLOUON PREDICTOR
# =====================================
@st.cache_resource
def load_predictor(model_path: str = MODEL_PATH):
    """
    Load the trained AutoGluon MultiModalPredictor once
    and cache it across Streamlit reruns.
    """
    if not os.path.exists(model_path):
        st.error(
            f"Model folder '{model_path}' not found. "
            "Make sure your trained AutoGluon model is saved here."
        )
        st.stop()

    predictor = MultiModalPredictor.load(model_path)
    return predictor


predictor = load_predictor()

# Optional: show class labels in sidebar
try:
    CLASS_LABELS = predictor.class_labels
except Exception:
    CLASS_LABELS = None

# =====================================
# 3. MERGE HEURISTIC (COPIED FROM YOUR SCRIPT)
# =====================================
def merge_javanese_chars(chars):
    """
    MERGE LOGIC: Corrects over-segmentation based on Javanese script rules.
    Input: list of predicted labels, e.g. ["ha","na","ca","ra","ka"]
    Output: a single Latin string, e.g. "hanacaraka"
    """
    if not chars:
        return ""

    merged = []
    i = 0
    while i < len(chars):
        current = chars[i]
        next_char = chars[i + 1] if i + 1 < len(chars) else None

        # RULE 1: 'da' + 'ra' -> 'ba'
        if current == "da" and next_char == "ra":
            merged.append("ba")
            i += 2
            continue

        # RULE 2: 'nga' + 'nga' -> 'nga'
        if current == "nga" and next_char == "nga":
            merged.append("nga")
            i += 2
            continue

        # RULE 3: 'tha' + 'nya' -> 'tha'
        if current == "tha" and next_char == "nya":
            merged.append("tha")
            i += 2
            continue

        # RULE 4: 'na' + 'ya' -> 'nya'
        if current == "na" and next_char == "ya":
            merged.append("nya")
            i += 2
            continue

        # Default: keep as-is
        merged.append(current)
        i += 1

    # NOTE: this returns a Latin string (romanization) directly.
    # ['ha','na','ca','ra','ka'] -> "hanacaraka"
    return "".join(merged)


# =====================================
# 4. IMAGE HELPERS
# =====================================
def pil_from_canvas(image_data: np.ndarray) -> Image.Image:
    """
    Convert RGBA canvas (H, W, 4) to RGB PIL image.
    """
    if image_data is None:
        return None
    img_uint8 = image_data.astype("uint8")
    pil_rgba = Image.fromarray(img_uint8, mode="RGBA")
    return pil_rgba.convert("RGB")


def segment_word_image(
    pil_img: Image.Image,
    min_w: int = 8,
    min_h: int = 10,
    pad: int = 4,
):
    """
    Segment a word-level image into character crops.
    Logic adapted from your get_crops_from_image, but taking a PIL image.
    Returns:
      - crops: list[PIL.Image]
      - boxes: list[(x, y, w, h)]
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    h_img, w_img = gray.shape
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > min_w and h > min_h:
            boxes.append((x, y, w, h))

    # Sort left-to-right
    boxes.sort(key=lambda b: b[0])

    crops = []
    for (x, y, w, h) in boxes:
        y1 = max(0, y - pad)
        y2 = min(h_img, y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(w_img, x + w + pad)
        crop = img_rgb[y1:y2, x1:x2]
        crops.append(Image.fromarray(crop))

    return crops, boxes


def draw_boxes_with_labels(
    pil_img: Image.Image,
    boxes,
    labels,
    confidences,
):
    """
    Draw bounding boxes and label + confidence on the image.
    """
    img_rgb = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    for (x, y, w, h), lab, conf in zip(boxes, labels, confidences):
        # Box
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Text: label + confidence
        text = f"{lab} {conf * 100:.1f}%"
        text_y = max(10, y - 5)
        cv2.putText(
            img_bgr,
            text,
            (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    img_annot_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_annot_rgb)


# =====================================
# 5. PREDICTION HELPERS
# =====================================
def predict_single_char(pil_img: Image.Image):
    """
    Predict a single character from a PIL image.
    This is the 'character mode' (no segmentation).
    """
    clear_tmp_dir()

    tmp_path = TMP_DIR / "single_char.png"
    pil_img.convert("RGB").save(tmp_path)

    df = pd.DataFrame({"image": [str(tmp_path)]})
    preds = predictor.predict(df)
    proba = predictor.predict_proba(df).iloc[0]

    label = preds.iloc[0] if hasattr(preds, "iloc") else preds[0]
    top3 = proba.sort_values(ascending=False).head(3)

    return label, top3


def predict_word_from_image(pil_img: Image.Image):
    """
    Word mode:
      1. Segment word into multiple character crops
      2. Predict each crop
      3. Use merge_javanese_chars to merge into a final Latin word
      4. Also return confidence per character
    """
    clear_tmp_dir()

    crops, boxes = segment_word_image(pil_img)
    if not crops:
        return [], [], "", boxes

    crop_paths = []
    for i, crop in enumerate(crops):
        path = TMP_DIR / f"crop_{i}.png"
        crop.convert("RGB").save(path)
        crop_paths.append(str(path))

    df = pd.DataFrame({"image": crop_paths})
    preds = predictor.predict(df)
    proba_df = predictor.predict_proba(df)

    raw_labels = []
    confidences = []

    # Align predictions and probabilities
    for i in range(len(df)):
        if hasattr(preds, "iloc"):
            lab = preds.iloc[i]
        else:
            lab = preds[i]
        raw_labels.append(lab)
        # Probability of the predicted label
        conf = float(proba_df.iloc[i][lab])
        confidences.append(conf)

    merged_word = merge_javanese_chars(raw_labels)

    return raw_labels, confidences, merged_word, boxes


# =====================================
# 6. STREAMLIT UI
# =====================================
def main():
    st.set_page_config(
        page_title="Javanese Handwriting Demo",
        layout="wide",
        page_icon="✍️",
    )

    st.title("✍️ Javanese Character & Word Demo")
    st.markdown(
        """
Draw Aksara Jawa on the canvas **or** upload an image.

The model:
- Uses your **AutoGluon MultiModal** classifier  
- Supports **character-level** and **word-level (OCR + merge)** prediction  
- Produces a Latin “transliteration” from the predicted labels  
"""
    )

    with st.sidebar:
        st.header("Settings")
        mode = st.radio(
            "Recognition mode",
            ["Character (single glyph)", "Word (OCR + merge)"],
        )

        input_type = st.radio(
            "Input type",
            ["Sketch (canvas)", "Upload image"],
        )

        st.markdown("---")
        if CLASS_LABELS is not None:
            st.write("**Known classes (labels):**")
            st.write(", ".join(map(str, CLASS_LABELS)))

    col_left, col_right = st.columns([1, 1])

    # === INPUT IMAGE (Canvas or Upload) ===
    with col_left:
        st.subheader("Input")

        pil_img = None

        if input_type == "Sketch (canvas)":
            st.write("Draw your character or word below:")

            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 1)",
                stroke_width=10,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=256,
                width=256,
                drawing_mode="freedraw",
                key="canvas",
            )

            if canvas_result.image_data is not None:
                # Avoid predicting on a completely blank canvas
                if np.mean(canvas_result.image_data[:, :, :3]) < 250:
                    pil_img = pil_from_canvas(canvas_result.image_data)
                    st.image(
                        pil_img,
                        caption="Current sketch",
                        use_column_width=True,
                    )
                else:
                    st.info("Canvas is blank. Draw something to see predictions.")
        else:
            uploaded = st.file_uploader(
                "Upload a Javanese text image (.png, .jpg, .jpeg)",
                type=["png", "jpg", "jpeg"],
            )
            if uploaded is not None:
                pil_img = Image.open(uploaded).convert("RGB")
                st.image(
                    pil_img,
                    caption="Uploaded image",
                    use_column_width=True,
                )

    # === PREDICTIONS ===
    with col_right:
        st.subheader("Prediction")

        if pil_img is None:
            st.info("Waiting for input image...")
            return

        # Character mode
        if mode.startswith("Character"):
            label, top3 = predict_single_char(pil_img)

            st.metric("Predicted character label", label)
            st.caption(
                f"Latin transliteration (approx.): **{label}** "
                "(labels are already Latin-based)."
            )

            st.write("Top-3 probabilities:")
            st.table(
                pd.DataFrame(
                    {
                        "label": top3.index.tolist(),
                        "probability": top3.values.tolist(),
                    }
                )
            )

        # Word mode
        else:
            raw_labels, confidences, merged_word, boxes = predict_word_from_image(
                pil_img
            )

            if not raw_labels:
                st.warning(
                    "No characters detected. Try drawing thicker/larger or "
                    "checking contrast in your uploaded image."
                )
                return

            st.write("Raw predicted sequence (per segment):")
            st.code(" ".join(raw_labels))

            st.metric(
                "Merged word (Latin transliteration)",
                merged_word if merged_word else "(empty)",
            )
            st.caption(
                "Transliteration is produced by concatenating the predicted Latin labels "
                "and applying your heuristic merge rules."
            )

            # Per-character table: index, box, label, confidence
            rows = []
            for idx, ((x, y, w, h), lab, conf) in enumerate(
                zip(boxes, raw_labels, confidences)
            ):
                rows.append(
                    {
                        "idx": idx,
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "label": lab,
                        "confidence": round(conf, 4),
                    }
                )

            st.write("Per-character predictions:")
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
            )

            # Annotated image with bounding boxes + labels + confidence
            annotated_img = draw_boxes_with_labels(
                pil_img, boxes, raw_labels, confidences
            )
            st.image(
                annotated_img,
                caption="Segmented characters with label + confidence",
                use_column_width=True,
            )


if __name__ == "__main__":
    main()
