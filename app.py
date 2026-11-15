"""Streamlit demo for drone detection and country-origin hypotheses.

The hard-negative binary detector is suitable for an alert. Country attribution
is less reliable and is therefore shown only as a ranked shortlist. All metrics
displayed here come from held-out v4 test data, including unseen bird flocks.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from models.inference import resolve_device
from models.two_stage import TwoStageDroneDetector
from utils.visualization import draw_boxes

LABEL_SCHEME = "binary Faster R-CNN + country Faster R-CNN"
NUM_COUNTRIES = 18

DETECTOR_WEIGHTS = "weights/binary_v5_birds_focused/best_model.pth"
COUNTRY_DETECTOR_WEIGHTS = "weights/country_v3/best_model.pth"
CLASSIFIER_WEIGHTS = "weights/crop_country.pth"  # Legacy fallback only.

# Held-out metrics. See the matching reports/ files.
BINARY_MAP_50 = 0.9503
DETECTION_THRESHOLD = 0.50
NMS_THRESHOLD = 0.40
DETECTION_FPR = 0.0782
DETECTION_RECALL = 1.0
BIRD_FLOCK_FPR = 0.0227
COUNTRY_TOP1 = 0.336
COUNTRY_TOP3 = 0.591

st.set_page_config(page_title="Drone Detection", layout="wide")
st.title("Military Drone Detection and Origin Analysis")
st.caption(
    f"Faster R-CNN (ResNet50-FPN) · {NUM_COUNTRIES} countries · "
    f"label scheme `{LABEL_SCHEME}`"
)


@st.cache_resource
def load_model(detector_path, country_detector_path, classifier_path, device):
    """Cache models per weight-path/device combination."""
    return TwoStageDroneDetector(
        detector_path,
        classifier_path=classifier_path,
        device=device,
        country_detector_path=country_detector_path,
    )


device = resolve_device(None)
score_thresh = DETECTION_THRESHOLD
nms_thresh = NMS_THRESHOLD

with st.sidebar:
    st.header("Country attribution")

    show_country = st.checkbox("Show country attribution", value=True)
    top_k = st.slider(
        "Country candidates to show", 1, 5, 3, 1, disabled=not show_country
    )

    st.markdown("---")
    st.markdown("**Held-out performance**")
    st.markdown(f"- Binary detector: **{BINARY_MAP_50:.3f}** mAP@0.5")
    st.markdown(
        f"- FPR at fixed threshold {DETECTION_THRESHOLD:.2f}: "
        f"**{DETECTION_FPR:.2%}**"
    )
    st.markdown(f"- Recall: **{DETECTION_RECALL:.2%}**")
    st.markdown(f"- Bird-flock FPR: **{BIRD_FLOCK_FPR:.2%}**")
    st.markdown(
        f"- Origin top-1 / top-3: **{COUNTRY_TOP1:.1%} / {COUNTRY_TOP3:.1%}**"
    )

st.info(
    "**Detection is the strong output; attribution is exploratory.** The "
    f"binary detector reached {BINARY_MAP_50:.3f} mAP@0.5. On localized "
    f"objects, origin attribution reached {COUNTRY_TOP1:.1%} top-1 and "
    f"{COUNTRY_TOP3:.1%} top-3. Country labels are ranked hypotheses about "
    "airframe design/manufacturing origin—not verified ownership or operator "
    "identity.",
    icon=":material/info:",
)

uploaded_files = st.file_uploader(
    "Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

country_path = (
    COUNTRY_DETECTOR_WEIGHTS if Path(COUNTRY_DETECTOR_WEIGHTS).exists() else None
)
detector_path = (
    DETECTOR_WEIGHTS if Path(DETECTOR_WEIGHTS).exists() else "weights/best_model.pth"
)
detector = load_model(detector_path, country_path, CLASSIFIER_WEIGHTS, str(device))
detector.score_thresh = score_thresh
detector.nms_thresh = nms_thresh

all_rows = []

for uploaded in uploaded_files:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error(f"Could not decode {uploaded.name}; skipping.")
        continue

    st.markdown("---")
    st.subheader(uploaded.name)

    with st.spinner("Running both Faster R-CNN stages..."):
        results = detector.predict(image, top_k=top_k)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Input")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.caption("Detections")
        if results:
            display_results = []
            for result in results:
                shown = dict(result)
                # The score drawn beside this box is the binary detector score,
                # not the country-stage score. Keep attribution in the ranked
                # panel below so "USA 0.86" cannot be misread as 86% confidence
                # that the airframe is American.
                shown["label"] = "Drone"
                display_results.append(shown)
            annotated = draw_boxes(
                image, display_results, show_score=True, show_label=show_country
            )
            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
        else:
            st.warning(
                "No high-confidence drone detections above the threshold. "
                "This does not prove that the image contains no drone."
            )

    if not results:
        continue

    st.markdown(
        f"**{len(results)} drone{'s' if len(results) != 1 else ''} detected**"
    )

    for i, result in enumerate(results, 1):
        x1, y1, x2, y2 = (int(value) for value in result["bbox"])
        candidates = result.get("candidates") or []
        with st.container(border=True):
            left, right = st.columns([1, 2])
            with left:
                st.metric(
                    f"Drone {i}",
                    f"{result['score']:.0%}",
                    help="Binary confidence that this box contains a drone.",
                )
                st.caption(f"box [{x1}, {y1}, {x2}, {y2}]")
            with right:
                if show_country:
                    st.caption("Possible design origin (uncalibrated ranking)")
                    if not candidates:
                        st.caption("No overlapping origin proposal was returned.")
                    for candidate in candidates[:top_k]:
                        st.progress(
                            min(1.0, candidate["share"]),
                            text=(
                                f"{candidate['label']} — "
                                f"{candidate['share']:.0%} relative score share"
                            ),
                        )
                    if len(candidates) == 1:
                        st.caption(
                            "One returned candidate is not evidence of certainty."
                        )
                else:
                    st.caption("Country attribution hidden.")

        all_rows.append(
            {
                "Image": uploaded.name,
                "Drone #": i,
                "Detection confidence": round(result["score"], 3),
                "Top origin": (
                    candidates[0]["label"]
                    if show_country and candidates
                    else "Unresolved" if show_country else "Hidden"
                ),
                "Top relative score share": (
                    round(candidates[0]["share"], 3) if candidates else None
                ),
                "Runner-up": candidates[1]["label"] if len(candidates) > 1 else "",
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

    display_results = []
    for result in results:
        shown = dict(result)
        shown["label"] = "Drone"
        display_results.append(shown)
    annotated = draw_boxes(image, display_results, show_label=show_country)
    _, buffer = cv2.imencode(".jpg", annotated)
    st.download_button(
        "Download annotated image",
        data=buffer.tobytes(),
        file_name=f"detected_{uploaded.name}",
        mime="image/jpeg",
        key=f"dl_{uploaded.name}",
    )

if all_rows:
    st.markdown("---")
    st.subheader("All detections")
    dataframe = pd.DataFrame(all_rows)
    st.dataframe(dataframe, use_container_width=True)

    counts = dataframe["Top origin"].value_counts()
    columns = st.columns(max(len(counts), 1))
    for column, (country, count) in zip(columns, counts.items()):
        column.metric(label=country, value=int(count))
    st.caption(
        "Counts use the highest-scoring origin candidate. Held-out top-1 "
        f"attribution was {COUNTRY_TOP1:.1%}; do not treat them as confirmed "
        "country intelligence."
    )

    st.download_button(
        "Download detections CSV",
        data=dataframe.to_csv(index=False).encode(),
        file_name="detections.csv",
        mime="text/csv",
    )
