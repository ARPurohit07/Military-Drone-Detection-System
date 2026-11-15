from icrawler.builtin import GoogleImageCrawler
from pathlib import Path
import os


CATEGORIES = {
    "birds_in_sky": "birds flying in sky",
    "kites_flying": "kites flying in sky",
    "planes_far": "airplane far away in sky",
    "helicopters_far": "helicopter far away in sky",
    "balloons_in_sky": "balloon floating in sky",
    "insects_flying": "insect flying in air",
    "floating_plastic_bag": "plastic bag floating in wind",
    "sky_clouds": "clouds in sky",
    "landscape": "landscape scenery",
}


def scrape_category(category, query, max_num=150, root_dir="Negative"):
    out_dir = Path(root_dir) / category
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(out_dir.glob("*.jpg")))
    if existing >= max_num:
        print(f"  {category}: already have {existing} images, skipping")
        return

    print(f"  {category}: scraping {max_num - existing} more images...")
    crawler = GoogleImageCrawler(
        storage={"root_dir": str(out_dir)},
        log_level=40,
    )
    crawler.crawl(
        keyword=query,
        max_num=max_num,
        min_size=(200, 200),
    )
    print(f"  {category}: done ({len(list(out_dir.glob('*.jpg')))} total)")


def create_empty_labels(neg_dir, label_dir):
    label_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img_path in Path(neg_dir).glob("*.jpg"):
        lbl_path = label_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            lbl_path.touch()
            count += 1
    print(f"Created {count} empty label files in {label_dir}")


def main():
    root = Path("Negative")
    root.mkdir(exist_ok=True)

    print("Scraping hard negatives...")
    for category, query in CATEGORIES.items():
        scrape_category(category, query, max_num=150, root_dir="Negative")

    label_dir = Path("labels_negative")
    create_empty_labels("Negative", label_dir)

    print("\nDone! Negative images saved to Negative/")


if __name__ == "__main__":
    main()
