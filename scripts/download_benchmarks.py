from __future__ import annotations

import argparse
import gzip
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DATASETS = {
    "math500": {
        "target": "math500_test.jsonl",
        "urls": [
            "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/data/test-00000-of-00001.jsonl",
            "https://huggingface.co/datasets/lighteval/MATH/resolve/main/MATH.jsonl",
        ],
        "gzip": False,
    },
    "hotpotqa": {
        "target": "hotpot_dev_distractor_v1.json",
        "urls": [
            "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json",
            "https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_dev_distractor_v1.json",
        ],
        "gzip": False,
    },
    "humaneval": {
        "target": "HumanEval.jsonl",
        "urls": [
            "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
            "https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz",
        ],
        "gzip": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download benchmark datasets into ./data.")
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory to store benchmark files.",
    )
    parser.add_argument(
        "--only",
        choices=["math500", "hotpotqa", "humaneval", "all"],
        default="all",
        help="Download only one dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if they already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Per-request timeout in seconds.",
    )
    return parser.parse_args()


def _download_url(url: str, out_path: Path, timeout: int) -> None:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=timeout) as response, out_path.open("wb") as f:
        shutil.copyfileobj(response, f)


def _try_download(urls: list[str], out_path: Path, timeout: int) -> None:
    last_error: Exception | None = None
    for idx, url in enumerate(urls, start=1):
        try:
            print(f"  [{idx}/{len(urls)}] Downloading from: {url}")
            _download_url(url, out_path, timeout)
            return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            print(f"    Failed: {exc}")
            time.sleep(1)
    raise RuntimeError(f"All download sources failed. Last error: {last_error}")


def _gunzip_file(src_path: Path, dst_path: Path) -> None:
    with gzip.open(src_path, "rb") as src, dst_path.open("wb") as dst:
        shutil.copyfileobj(src, dst)


def _download_dataset(name: str, spec: dict, data_dir: Path, force: bool, timeout: int) -> None:
    target = data_dir / spec["target"]
    if target.exists() and not force:
        print(f"Skipping {name}: {target} already exists")
        return

    temp_path = target.with_suffix(target.suffix + ".download")
    if temp_path.exists():
        temp_path.unlink()

    print(f"Downloading {name} -> {target}")
    _try_download(spec["urls"], temp_path, timeout)

    if spec.get("gzip", False):
        gz_path = temp_path.with_suffix(temp_path.suffix + ".gz")
        temp_path.rename(gz_path)
        try:
            _gunzip_file(gz_path, target)
        finally:
            if gz_path.exists():
                gz_path.unlink()
    else:
        temp_path.rename(target)

    print(f"Done: {target} ({target.stat().st_size} bytes)")


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    selected = list(DATASETS.keys()) if args.only == "all" else [args.only]
    failures = 0

    for name in selected:
        spec = DATASETS[name]
        try:
            _download_dataset(name, spec, data_dir, args.force, args.timeout)
        except Exception as exc:
            failures += 1
            print(f"ERROR downloading {name}: {exc}")

    if failures:
        print(f"Completed with {failures} failure(s).")
        return 1

    print("All requested datasets are ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
