import xml.etree.ElementTree as ET
from pathlib import Path

import cv2


YOLO_CLASSES = {
    0: "Turkish Drones",
    1: "Iranian Drones",
    2: "Israeli Drones",
    3: "Chinese Drones",
}


def yolo_to_voc(yolo_label_path, image_path, output_xml_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return
    img_h, img_w = img.shape[:2]

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = image_path.parent.name
    ET.SubElement(annotation, "filename").text = image_path.name
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_w)
    ET.SubElement(size, "height").text = str(img_h)
    ET.SubElement(size, "depth").text = "3"

    with open(yolo_label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:]]
            xmin = int((cx - w / 2) * img_w)
            ymin = int((cy - h / 2) * img_h)
            xmax = int((cx + w / 2) * img_w)
            ymax = int((cy + h / 2) * img_h)
            xmin, ymin = max(0, xmin), max(0, ymin)
            xmax, ymax = min(img_w, xmax), min(img_h, ymax)

            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = YOLO_CLASSES.get(cls_id, f"class_{cls_id}")
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(xmin)
            ET.SubElement(bndbox, "ymin").text = str(ymin)
            ET.SubElement(bndbox, "xmax").text = str(xmax)
            ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ")
    tree.write(str(output_xml_path), xml_declaration=True, encoding="utf-8")


def convert_split(root, split):
    root = Path(root)
    label_dir = root / "labels" / split
    img_dir = root / "images" / split
    ann_dir = root / "annotations" / split
    ann_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for label_file in sorted(label_dir.glob("*.txt")):
        stem = label_file.stem
        for ext in [".jpg", ".png", ".jpeg"]:
            img_path = img_dir / (stem + ext)
            if img_path.exists():
                xml_path = ann_dir / (stem + ".xml")
                yolo_to_voc(label_file, img_path, xml_path)
                count += 1
                break

    print(f"Converted {count} labels to VOC XML in {split}/")


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "drone_dataset_3_test"
    for split in ["train", "val"]:
        convert_split(root, split)
