
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


# ---------------------------------------------------------
# PATHS AND CONFIGURATION
# ---------------------------------------------------------
BASE = Path(__file__).resolve().parent
TRAINING_DIR = BASE / "verified_training_pair"
PREDICTION_DIR = BASE / "test_predictions"
COMMENTS_FILE = BASE / "team_comments.csv"
VIDEO_DIR = BASE / "videos"
IMAGE_UPLOAD_MODEL_PATH = BASE / "trained_model" / "water_chestnut_deeplabv3.pth"

# Use the converted H.264 video for browser-compatible playback.
UNET_VIDEO_NAME = "water_chestnut_test_unet_chrome.mp4"
UNET_VIDEO_PATH = VIDEO_DIR / UNET_VIDEO_NAME

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
}

COMMENT_COLUMNS = [
    "Timestamp",
    "Reviewer",
    "Image",
    "Category",
    "Comment",
]

COMMENT_CATEGORIES = [
    "Missed water chestnut",
    "False positive",
    "Good detection",
    "Image quality or lighting",
    "Suggested improvement",
    "Other",
]


# ---------------------------------------------------------
# COMMENTS: LOAD AND SAVE
# ---------------------------------------------------------
def load_comments():
    """Load saved comments without overwriting an unreadable file."""
    if not COMMENTS_FILE.exists():
        return pd.DataFrame(columns=COMMENT_COLUMNS), None

    try:
        comments = pd.read_csv(
            COMMENTS_FILE,
            dtype={"Image": str},
        )

        if not all(
            column in comments.columns
            for column in COMMENT_COLUMNS
        ):
            return None, (
                "The existing comments file has an unexpected format. "
                "Back it up and inspect it before making changes."
            )

        return comments[COMMENT_COLUMNS].copy(), None

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError,
        UnicodeDecodeError,
    ):
        return None, (
            "The existing comments file could not be read. "
            "Back it up and check its format before making changes."
        )


def save_comment(reviewer, image_id, category, comment_text):
    """Append a comment while preserving previously saved comments."""
    existing_comments, error = load_comments()

    if error:
        return False, error

    new_comment = pd.DataFrame(
        [
            {
                "Timestamp": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "Reviewer": reviewer.strip(),
                "Image": str(image_id),
                "Category": category,
                "Comment": comment_text.strip(),
            }
        ],
        columns=COMMENT_COLUMNS,
    )

    updated_comments = pd.concat(
        [existing_comments, new_comment],
        ignore_index=True,
    )

    try:
        updated_comments.to_csv(
            COMMENTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        return True, "Comment saved successfully."

    except OSError as exc:
        return False, f"Could not save the comment: {exc}"


# ---------------------------------------------------------
# IMAGE DOWNLOAD HELPER
# ---------------------------------------------------------
def _image_to_png_bytes(image):
    """Encode a PIL image as PNG bytes for Streamlit downloads."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------
# REAL-PLANT PHOTO INFERENCE
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading the water-chestnut AI model...")
def load_photo_detection_model():
    """Load the same DeepLabV3 model and checkpoint used by predict_water_chestnut.py."""
    device = torch.device("cpu")

    if not IMAGE_UPLOAD_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The trained model checkpoint was not found at: "
            f"{IMAGE_UPLOAD_MODEL_PATH}"
        )

    model = deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=None,
        aux_loss=True,
    )
    model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

    checkpoint = torch.load(
        IMAGE_UPLOAD_MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    return model, device, mean, std


def predict_uploaded_photo(source_image):
    """Return the original RGB photo, predicted mask, overlay, and predicted area."""
    image = ImageOps.exif_transpose(source_image).convert("RGB")
    original_size = image.size

    # Match the exact input size and normalization in predict_water_chestnut.py.
    resized = image.resize((512, 256), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)

    model, device, mean, std = load_photo_detection_model()
    tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)["out"]
        probabilities = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()

    # Resize the probability map back to the original photo dimensions.
    probability_image = Image.fromarray(
        np.uint8(np.clip(probabilities, 0, 1) * 255)
    ).resize(original_size, Image.Resampling.BILINEAR)

    probability_array = np.asarray(probability_image, dtype=np.uint8)
    mask_array = np.where(probability_array >= 128, 255, 0).astype(np.uint8)
    mask = Image.fromarray(mask_array, mode="L")

    # Red indicates pixels predicted as water chestnut.
    red_layer = Image.new("RGBA", original_size, (255, 0, 0, 0))
    red_layer.putalpha(mask.point(lambda pixel: 100 if pixel else 0))
    overlay = Image.alpha_composite(image.convert("RGBA"), red_layer).convert("RGB")

    predicted_area_percent = (
        100.0 * np.count_nonzero(mask_array) / mask_array.size
    )

    return image, mask, overlay, predicted_area_percent


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="BioGlow | Water Chestnut AI",
    page_icon="🌿",
    layout="wide",
)

st.title("BioGlow: AI Water Chestnut Detection")

st.caption(
    "FLL 2026–2027 | Computer vision, model testing, "
    "and aquatic invasive plant collection"
)

st.info(
    "Project goal: identify water chestnut in camera images "
    "to support a future boat-mounted detection and collection system."
)


# ---------------------------------------------------------
# EVALUATION IMAGE OBSERVATIONS
# ---------------------------------------------------------
# Percentages below are visual estimates, not formally calculated metrics.
results = [
    {
        "Image": "111822",
        "Observation": (
            "Most of the large water chestnut area was detected successfully. "
            "A small triangular area at the far upper-right corner and a small "
            "section along the distant upper-center line were missed. "
            "No false positives were observed."
        ),
        "Status": "98% visually estimated detection",
    },
    {
        "Image": "111829",
        "Observation": (
            "Most of the large water chestnut area was detected successfully. "
            "One or two very small patches at the far end were missed. "
            "A few false positives occurred near the end, covering portions "
            "of large plants adjacent to the water chestnut."
        ),
        "Status": "99% visually estimated detection",
    },
    {
        "Image": "111831",
        "Observation": (
            "Approximately 95% of the large water chestnut area was detected. "
            "The complete line of water chestnut at the distant end, "
            "approximately 5% of the target area, was missed. "
            "No false positives were observed."
        ),
        "Status": "95% visually estimated detection",
    },
    {
        "Image": "112652",
        "Observation": (
            "This is a closer view of the water chestnut. Nearly all leaves "
            "were detected, with approximately five individual leaves missed "
            "at different locations. No false positives were observed."
        ),
        "Status": "99.5% visually estimated coverage",
    },
    {
        "Image": "112810",
        "Observation": (
            "A small triangular area near the top and a medium-sized patch "
            "along the bottom line of water chestnut were missed. Two water "
            "patches within the water chestnut area were incorrectly detected "
            "as water chestnut."
        ),
        "Status": "96% visually estimated detection",
    },
    {
        "Image": "112822",
        "Observation": (
            "Three separate water chestnut bunches were present: two large "
            "bunches and one small bunch. Both large bunches were detected "
            "mostly successfully, although a few leaves near their edges "
            "were missed. The third, smaller bunch was not detected at all. "
            "No false positives were observed."
        ),
        "Status": "Small-bunch detection failure",
    },
]


# ---------------------------------------------------------
# DASHBOARD TABS
# ---------------------------------------------------------
(
    tab_overview,
    tab_gallery,
    tab_photo,
    tab_training,
    tab_analysis,
    tab_comments,
    tab_video,
) = st.tabs(
    [
        "Project Overview",
        "Detection Gallery",
        "AI Photo Detection",
        "Training Progress",
        "Results & Next Steps",
        "Team Comments",
        "AI Detection Video",
    ]
)


# ---------------------------------------------------------
# PROJECT OVERVIEW
# ---------------------------------------------------------
with tab_overview:
    st.header("What we are building")

    c1, c2, c3 = st.columns(3)
    c1.metric("Verified training images", "9")
    c2.metric("Training epochs", "20")
    c3.metric("Evaluation images", str(len(results)))

    st.caption(
        "Training counts reflect the current project configuration "
        "and should be updated when the dataset or training run changes."
    )

    st.subheader("How the system works")

    st.markdown(
        """
        1. **Collect images:** use expert-provided photos of aquatic plants.
        2. **Annotate:** outline water chestnut leaves using Labelme.
        3. **Create masks:** convert the outlines into pixel-level labels.
        4. **Train:** teach DeepLabV3-MobileNetV3-Large to classify pixels
           as background or water chestnut.
        5. **Test:** inspect predictions on separate evaluation images.
        6. **Future integration:** connect detection to the boat's
           controller and collection mechanism.
        """
    )

    st.subheader("Two separate demonstrations")

    left, right = st.columns(2)

    with left:
        st.markdown("**Real-plant AI testing**")
        st.write(
            "Use real aquatic photographs to evaluate plant detection, "
            "missed leaves, reflections, and distant patches."
        )

    with right:
        st.markdown("**Artificial-plant hardware testing**")
        st.write(
            "Use a handmade plastic water chestnut model to test the "
            "camera-to-controller signal and collection mechanism. "
            "This does not establish real-plant detection accuracy."
        )


# ---------------------------------------------------------
# DETECTION GALLERY
# ---------------------------------------------------------
with tab_gallery:
    st.header("Detection Gallery")

    st.write(
        "Red overlay regions indicate the model's predicted water-chestnut "
        "pixels. Predictions are not verified ground-truth annotations."
    )

    available = [
        row["Image"]
        for row in results
        if (
            PREDICTION_DIR
            / f"20240720_{row['Image']}_overlay.png"
        ).exists()
    ]

    if not available:
        st.warning(
            "No matching prediction overlays were found in the "
            "'test_predictions' folder. Check the filenames or run "
            "'python .\\predict_water_chestnut.py'."
        )
    else:
        selected = st.selectbox(
            "Choose an evaluation image",
            available,
            format_func=lambda image_id: f"20240720_{image_id}",
            key="gallery_evaluation_image",
        )

        stem = f"20240720_{selected}"

        original_path = PREDICTION_DIR / f"{stem}_original.png"
        mask_path = PREDICTION_DIR / f"{stem}_predicted_mask.png"
        overlay_path = PREDICTION_DIR / f"{stem}_overlay.png"

        a, b, c = st.columns(3)

        with a:
            st.markdown("**Original image**")
            if original_path.exists():
                st.image(
                    Image.open(original_path),
                    use_container_width=True,
                )
            else:
                st.warning("Original prediction input image not found.")

        with b:
            st.markdown("**Predicted mask**")
            if mask_path.exists():
                st.image(
                    Image.open(mask_path),
                    use_container_width=True,
                )
            else:
                st.warning("Predicted mask not found.")

        with c:
            st.markdown("**Red-overlay prediction**")
            if overlay_path.exists():
                st.image(
                    Image.open(overlay_path),
                    use_container_width=True,
                )
            else:
                st.warning("Overlay not found.")

        selected_result = next(
            item for item in results
            if item["Image"] == selected
        )

        st.subheader("Team Observation")
        st.write(selected_result["Observation"])

        st.subheader("Team Assessment")
        st.write(selected_result["Status"])

        st.caption(
            "Detection percentages are team visual estimates, "
            "not calculated segmentation accuracy scores."
        )

        # -------------------------------------------------
        # COMMENT FORM
        # -------------------------------------------------
        st.divider()
        st.subheader(f"Review Image {selected}")

        st.write(
            "Review the original image and AI predictions above, "
            "then enter your feedback below. Your comment will be "
            "automatically associated with this image."
        )

        with st.form(
            f"gallery_comment_form_{selected}",
            clear_on_submit=True,
        ):
            reviewer = st.text_input(
                "Your name or team member ID",
                placeholder="e.g., Pranav",
                key=f"gallery_reviewer_{selected}",
            )

            category = st.selectbox(
                "Comment category",
                COMMENT_CATEGORIES,
                key=f"gallery_category_{selected}",
            )

            comment_text = st.text_area(
                "What do you observe in this image?",
                placeholder=(
                    "Describe missed leaves, incorrect detections, "
                    "or anything the team should investigate."
                ),
                height=110,
                key=f"gallery_comment_{selected}",
            )

            submitted = st.form_submit_button("Save Comment")

        if submitted:
            if not reviewer.strip() or not comment_text.strip():
                st.warning("Please enter your name and a comment.")
            else:
                success, message = save_comment(
                    reviewer,
                    selected,
                    category,
                    comment_text,
                )

                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)

        # Show saved comments for this image.
        saved_comments, load_error = load_comments()

        st.subheader(f"Previous Comments for Image {selected}")

        if load_error:
            st.error(load_error)
        elif saved_comments.empty:
            st.info("No comments have been submitted for this image yet.")
        else:
            image_comments = saved_comments[
                saved_comments["Image"].astype(str) == str(selected)
            ].copy()

            if image_comments.empty:
                st.info("No comments have been submitted for this image yet.")
            else:
                image_comments = image_comments.sort_values(
                    "Timestamp",
                    ascending=False,
                )

                st.dataframe(
                    image_comments,
                    use_container_width=True,
                    hide_index=True,
                )

    # -----------------------------------------------------
    # VERIFIED TRAINING EXAMPLES
    # -----------------------------------------------------
    st.divider()
    st.subheader("Verified Training Examples")

    training_stems = [
        "20240720_111752",
        "20240720_111756",
        "20240720_111816",
        "20240720_111822",
        "20240720_112642",
        "20240720_112655",
        "20240720_112734",
        "20240720_112737",
        "20240720_112824",
    ]

    selected_train = st.selectbox(
        "Inspect an annotated training example",
        training_stems,
        key="gallery_training_example",
    )

    original_train = TRAINING_DIR / f"{selected_train}_image.png"
    mask_train = TRAINING_DIR / f"{selected_train}_mask.png"
    overlay_train = TRAINING_DIR / f"{selected_train}_overlay.png"

    a, b, c = st.columns(3)

    for col, label, path in [
        (a, "Training image", original_train),
        (b, "Human annotation mask", mask_train),
        (c, "Annotation overlay", overlay_train),
    ]:
        with col:
            st.markdown(f"**{label}**")
            if path.exists():
                st.image(
                    Image.open(path),
                    use_container_width=True,
                )
            else:
                st.warning(f"File not found: {path.name}")


# ---------------------------------------------------------
# AI PHOTO DETECTION
# ---------------------------------------------------------
with tab_photo:
    st.header("AI Water Chestnut Detection — Photo Upload")

    st.write(
        "Upload a real aquatic-plant photo to run the existing "
        "DeepLabV3-MobileNetV3-Large model. Red regions are pixels the "
        "model predicts as water chestnut."
    )

    st.info(
        "This runs the trained model on one uploaded photo. The prediction "
        "is an AI estimate, not a verified identification or a formal "
        "accuracy measurement."
    )

    st.caption(f"Model checkpoint: {IMAGE_UPLOAD_MODEL_PATH}")

    uploaded_photo = st.file_uploader(
        "Choose a photo",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Supported formats: JPG, JPEG, and PNG.",
        key="ai_photo_upload",
    )

    if uploaded_photo is not None:
        try:
            with Image.open(uploaded_photo) as uploaded_image:
                original_photo = ImageOps.exif_transpose(uploaded_image).convert("RGB")

            st.write(
                f"Uploaded file: **{uploaded_photo.name}**  |  "
                f"Dimensions: **{original_photo.width} × {original_photo.height} pixels**"
            )

            if not IMAGE_UPLOAD_MODEL_PATH.exists():
                st.error(
                    "The trained model checkpoint was not found. Confirm that "
                    "this file exists before running photo detection: "
                    f"{IMAGE_UPLOAD_MODEL_PATH}"
                )
            else:
                with st.spinner("Running the AI model on this photo..."):
                    _, predicted_mask, prediction_overlay, area_percent = (
                        predict_uploaded_photo(original_photo)
                    )

                metric_col, _ = st.columns([1, 2])
                metric_col.metric(
                    "Predicted image area",
                    f"{area_percent:.2f}%",
                    help=(
                        "Percentage of image pixels classified as water chestnut. "
                        "This is not the percentage of plants correctly detected."
                    ),
                )

                original_col, mask_col, overlay_col = st.columns(3)

                with original_col:
                    st.markdown("**Original photo**")
                    st.image(original_photo, use_container_width=True)

                with mask_col:
                    st.markdown("**Predicted mask**")
                    st.image(predicted_mask, use_container_width=True)

                with overlay_col:
                    st.markdown("**Red-overlay prediction**")
                    st.image(prediction_overlay, use_container_width=True)

                st.download_button(
                    label="Download prediction overlay",
                    data=_image_to_png_bytes(prediction_overlay),
                    file_name=f"{Path(uploaded_photo.name).stem}_overlay.png",
                    mime="image/png",
                    key="download_uploaded_photo_overlay",
                )

                st.download_button(
                    label="Download predicted mask",
                    data=_image_to_png_bytes(predicted_mask),
                    file_name=f"{Path(uploaded_photo.name).stem}_predicted_mask.png",
                    mime="image/png",
                    key="download_uploaded_photo_mask",
                )

        except Exception as exc:
            st.error(
                "Photo detection failed. Check that the model checkpoint matches "
                "the current DeepLabV3 model and that the required Python packages "
                f"are installed. Details: {exc}"
            )


# ---------------------------------------------------------
# TRAINING PROGRESS
# ---------------------------------------------------------
with tab_training:
    st.header("Training Progress")

    st.write(
        "The displayed training history has two phases: first the new "
        "classifier, then fine-tuning of the full network."
    )

    phase1 = [
        0.7728, 0.4780, 0.4060, 0.3715, 0.3669,
        0.3423, 0.3270, 0.3060, 0.2818, 0.2722,
    ]

    phase2 = [
        0.2604, 0.2572, 0.2535, 0.2499, 0.2449,
        0.2414, 0.2382, 0.2358, 0.2334, 0.2307,
    ]

    loss_data = pd.DataFrame(
        {
            "Epoch": list(range(1, 21)),
            "Training loss": phase1 + phase2,
            "Phase": ["Classifier"] * 10 + ["Fine-tuning"] * 10,
        }
    ).set_index("Epoch")

    st.line_chart(loss_data["Training loss"])

    st.dataframe(
        loss_data.reset_index(),
        use_container_width=True,
        hide_index=True,
    )

    st.metric(
        "Training loss change",
        f"{phase1[0]:.4f} → {phase2[-1]:.4f}",
        delta=f"{phase2[-1] - phase1[0]:.4f}",
        delta_color="inverse",
    )

    st.warning(
        "Training loss measures how well the model fits its training data. "
        "It is not test accuracy, precision, recall, or proof of "
        "real-world performance."
    )


# ---------------------------------------------------------
# RESULTS & NEXT STEPS
# ---------------------------------------------------------
with tab_analysis:
    st.header("Evaluation Results")

    df = pd.DataFrame(results)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Summary of Team Observations")

    st.markdown(
        """
        **Current strengths**

        - Most large water chestnut areas are detected in several images.
        - The model produces pixel-level predictions rather than only
          drawing bounding boxes.
        - Some images show few or no visually observed false positives.

        **Current limitations**

        - Small or distant plant patches and individual leaves can be missed.
        - Nearby vegetation and water patches can be incorrectly classified.
        - Results vary between scenes and plant sizes.
        - Evaluation images do not yet have independently verified
          ground-truth masks, so formal segmentation metrics cannot be
          calculated from these visual observations alone.

        **Next experiments**

        1. Annotate images containing small, distant plants and reflections.
        2. Keep evaluation images separate from training data.
        3. Compare future models against the same held-out images.
        4. Create verified evaluation masks to calculate segmentation metrics.
        5. Test the camera, controller signal, and collection hardware
           separately using the artificial plant.
        """
    )

    st.subheader("Important Scientific Distinction")

    st.write(
        "The reported percentages are visual estimates of detected plant "
        "coverage. They are not calculated accuracy scores. Rigorous "
        "evaluation requires verified ground-truth masks for separate "
        "evaluation images and comparison of predictions against those masks."
    )

    st.subheader("Evaluation Image Details")

    for row in results:
        with st.expander(
            f"Image {row['Image']} — {row['Status']}"
        ):
            st.write(row["Observation"])


# ---------------------------------------------------------
# TEAM COMMENTS
# ---------------------------------------------------------
with tab_comments:
    st.header("Team Comments & Review")

    st.write(
        "All comments submitted from the Detection Gallery appear here. "
        "You can filter the comments by image and download the full record."
    )

    comments_df, comments_error = load_comments()

    if comments_error:
        st.error(comments_error)
    elif comments_df.empty:
        st.info(
            "No comments have been saved yet. "
            "Open Detection Gallery, select an image, and submit a comment "
            "using the form below that image."
        )
    else:
        filter_image = st.selectbox(
            "Filter comments by image",
            ["All images"] + [row["Image"] for row in results],
            format_func=lambda value: (
                value
                if value == "All images"
                else f"20240720_{value}"
            ),
            key="comment_filter_image",
        )

        display_df = comments_df.copy()

        if filter_image != "All images":
            display_df = display_df[
                display_df["Image"].astype(str) == filter_image
            ]

        display_df = display_df.sort_values(
            "Timestamp",
            ascending=False,
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download Team Comments CSV",
            data=comments_df.to_csv(
                index=False,
            ).encode("utf-8-sig"),
            file_name="team_comments.csv",
            mime="text/csv",
        )

    st.caption(
        "Comments are stored on this computer. They are not automatically "
        "synchronized with other team members' computers."
    )


# ---------------------------------------------------------
# AI DETECTION VIDEO
# ---------------------------------------------------------
with tab_video:
    st.header("AI Water Chestnut Detection — Video Demo")

    st.write(
        "U-Net prediction on recorded video. "
        "Highlighted regions show the model's predictions."
    )

    st.info(
        "This is recorded-video testing, not validated live "
        "boat-camera operation."
    )

    video_path = UNET_VIDEO_PATH

    if not video_path.exists():
        st.error(f"Video not found: {video_path}")
        st.write("Available files in the videos folder:")

        if VIDEO_DIR.exists():
            for path in sorted(VIDEO_DIR.iterdir()):
                if path.is_file():
                    st.write(f"- {path.name}")
        else:
            st.error(f"Video folder not found: {VIDEO_DIR}")

    else:
        st.write(f"Selected video: {video_path.name}")
        st.write(
            f"File size: "
            f"{video_path.stat().st_size / (1024 * 1024):.2f} MB"
        )

        # Read the browser-compatible H.264 MP4 once.
        video_bytes = video_path.read_bytes()

        # Display the video using the converted MP4.
        st.video(
            video_bytes,
            format="video/mp4",
        )

        # Keep the download option.
        st.download_button(
            label="Download U-Net AI Detection Video",
            data=video_bytes,
            file_name=video_path.name,
            mime="video/mp4",
            key="download_unet_detection_video",
        )
