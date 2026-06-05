"""End-to-end fetch: resolve the active dataset and download its raw files.

This is the entry point for getting Amazon Reviews 2023 data onto disk. It reads
the config, builds the review/metadata URLs, downloads both (cached), and returns
the local paths ready for parsing.
"""

from pathlib import Path

from .config import active_dataset_category
from .download import download_file
from .sources import meta_url, raw_paths, review_url


def fetch_dataset(config, *, session=None, force=False, progress=True):
    """Download the active dataset's review + metadata files.

    Returns ``(review_path, meta_path)`` as local ``Path`` objects.
    """
    category = active_dataset_category(config)
    base_url = config["amazon_base_url"]
    review_path, meta_path = raw_paths(config["raw_dir"], category)

    download_file(
        review_url(base_url, category),
        review_path,
        session=session,
        force=force,
        progress=progress,
    )
    download_file(
        meta_url(base_url, category),
        meta_path,
        session=session,
        force=force,
        progress=progress,
    )
    return Path(review_path), Path(meta_path)


def main(argv=None):
    """CLI: download the active dataset defined in config/config.yaml."""
    import argparse

    from .config import load_config

    parser = argparse.ArgumentParser(description="Download an Amazon Reviews 2023 dataset.")
    parser.add_argument("--config", default="config/config.yaml", help="path to config file")
    parser.add_argument("--dataset", help="override active_dataset (e.g. digital_music)")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.dataset:
        config["active_dataset"] = args.dataset
    category = active_dataset_category(config)
    print(f"Fetching '{category}' into {config['raw_dir']}/ ...")
    review_path, meta_path = fetch_dataset(config, force=args.force)
    print(f"  reviews:  {review_path}")
    print(f"  metadata: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
