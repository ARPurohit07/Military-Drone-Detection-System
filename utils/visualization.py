"""Box drawing.

Colours are generated from the class name rather than hardcoded. The label space
is no longer four fixed strings—it is whatever the dataset builder produced,
currently 18 countries—so a fixed dictionary would silently fall through to
white for most classes. Hashing the
name also keeps a class the same colour across runs and across the app and any
saved figure, without a registry to maintain.
"""

import colorsys
import hashlib

import cv2

# Kept so existing callers and saved reports still resolve the legacy names to
# the same colours they used to have.
LEGACY_COLORS = {
    "Turkish Drones": (0, 255, 0),
    "Iranian Drones": (0, 0, 255),
    "Israeli Drones": (255, 0, 0),
    "Chinese Drones": (255, 255, 0),
}


def color_for(label):
    """Deterministic, visually distinct BGR colour for a class name."""
    if label in LEGACY_COLORS:
        return LEGACY_COLORS[label]
    # Hash to a hue, holding saturation and value high so every colour stays
    # readable against both bright sky and dark ground.
    digest = hashlib.md5(label.encode("utf-8")).digest()
    hue = digest[0] / 255.0
    sat = 0.65 + (digest[1] / 255.0) * 0.30
    val = 0.75 + (digest[2] / 255.0) * 0.25
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(b * 255), int(g * 255), int(r * 255))


def draw_boxes(image_bgr, results, thickness=2, font_scale=0.6,
               show_score=True, show_label=True):
    annotated = image_bgr.copy()
    height, width = annotated.shape[:2]

    for r in results:
        x1, y1, x2, y2 = map(int, r["bbox"])
        color = color_for(r["label"] if show_label else "Drone")
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # With attribution hidden the caption says "Drone" rather than a country
        # the model is only 33.6% likely to have right at top-1.
        name = r["label"] if show_label else "Drone"
        text = f'{name} {r["score"]:.2f}' if show_score else name
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                      font_scale, 1)

        # Put the label inside the frame. A box touching the top edge previously
        # had its caption drawn at a negative y, so the text was clipped away
        # entirely - and a drone near the top edge is the common case.
        label_bottom = y1 if y1 - th - 10 >= 0 else min(y2 + th + 10, height)
        label_top = label_bottom - th - 10
        label_left = max(0, min(x1, width - tw))

        cv2.rectangle(annotated, (label_left, label_top),
                      (label_left + tw, label_bottom), color, -1)
        cv2.putText(annotated, text, (label_left, label_bottom - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)
    return annotated
