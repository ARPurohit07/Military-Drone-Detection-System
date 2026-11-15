"""Run the legacy web-collection rebuild end to end.

    python -m data.build.pipeline --target 80 --negatives 110

For the current binary and country datasets, use build_binary_dataset and
build_country_dataset as documented in README.md.

Every stage is independently resumable and idempotent, which is the point: a
build of this size spans hours and will be interrupted. Rerunning the pipeline
picks up where it stopped rather than starting over - the scraper skips URLs
already in its manifest, the auto-labeler skips images that already have an
annotation, and only the final assemble step rewrites anything.

Stage order matters:

1. scrape     airframe queries -> images + provenance manifest
2. negatives  hard negatives, from image search
3. autolabel  zero-shot boxes -> VOC XML (needs 1)
4. assemble   dedup, quarantine conflicts, split, write dataset (needs 1-3)

Only stage 4 writes the trainable dataset, so stages 1-3 can be rerun and
extended freely; the dataset is regenerated from whatever is on disk at the time.
"""

import argparse
import time


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", type=int, default=80,
                   help="unique images per airframe (120 airframes -> ~9k images)")
    p.add_argument("--negatives", type=int, default=110,
                   help="images per negative category (29 categories -> ~3k)")
    p.add_argument("--scheme", default="country",
                   choices=["country", "airframe", "role"],
                   help="what the labels mean in the assembled dataset")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--min-per-class", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=4,
                   help="auto-labeler batch size (4 fits a 4 GB card at "
                        "autolabel.PROCESSOR_SIZE)")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["scrape", "negatives", "autolabel", "assemble"],
                   help="stages to skip")
    args = p.parse_args()

    started = time.time()

    def banner(n, name):
        print(f"\n{'=' * 70}\n[{n}/4] {name}\n{'=' * 70}", flush=True)

    if "scrape" not in args.skip:
        banner(1, f"scraping drone imagery (target {args.target}/airframe)")
        from data.build.scrape import scrape
        scrape(target_per_airframe=args.target)

    if "negatives" not in args.skip:
        banner(2, f"scraping hard negatives (target {args.negatives}/category)")
        from data.build.negatives import harvest, scrape_more
        # Re-index local hard negatives first. This includes deployment incident
        # frames, so the exact false positives that escaped the last model become
        # training signal in the next one. The operation is deterministic and
        # rewrites only the manifest, not the source images.
        harvest(source_dirs=("Negative", "detections"))
        scrape_more(target_per_category=args.negatives)

    if "autolabel" not in args.skip:
        banner(3, "auto-labelling with Grounding DINO")
        from data.build.autolabel import annotate
        annotate(batch_size=args.batch_size)

    if "assemble" not in args.skip:
        banner(4, f"assembling dataset (scheme={args.scheme})")
        from data.build.assemble import assemble
        assemble(scheme=args.scheme, val_fraction=args.val_fraction,
                 test_fraction=args.test_fraction,
                 min_per_class=args.min_per_class)

    mins = (time.time() - started) / 60
    print(f"\npipeline finished in {mins:.1f} min")
    print("\nnext steps:")
    print("  python -m data.validate_dataset --root drone_dataset_v2")
    print("  python train.py")
    print("  python evaluate.py --weights weights/best_model.pth --adverse")


if __name__ == "__main__":
    main()
