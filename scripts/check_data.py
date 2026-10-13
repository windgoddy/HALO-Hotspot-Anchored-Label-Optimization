"""Check benchmark files against the split lists shipped with HALO.

Run from the repository root: python scripts/check_data.py --root ./data
"""

import argparse
from pathlib import Path


IMAGE_FOLDERS = {
    "NUAA-SIRST": "images",
    "NUDT-SIRST": "images",
    "IRSTD-1k": "IRSTD1k_Img",
}
REPO_ROOT = Path(__file__).resolve().parents[1]


def read_ids(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_dataset(data_root, name):
    split_root = REPO_ROOT / "splits" / name
    train_ids = read_ids(split_root / "train.txt")
    test_ids = read_ids(split_root / "test.txt")
    dataset_root = data_root / name

    problems = []
    for split, ids in (("train", train_ids), ("test", test_ids)):
        if len(ids) != len(set(ids)):
            problems.append(f"{split} split contains duplicate IDs")
    overlap = set(train_ids) & set(test_ids)
    if overlap:
        problems.append(f"train/test overlap: {len(overlap)} IDs")

    all_ids = set(train_ids) | set(test_ids)
    for kind, folder in (("image", IMAGE_FOLDERS[name]), ("mask", "masks")):
        missing = [image_id for image_id in sorted(all_ids)
                   if not (dataset_root / folder / f"{image_id}.png").is_file()]
        if missing:
            examples = ", ".join(missing[:5])
            problems.append(f"{len(missing)} missing {kind} files in {folder}/ (e.g. {examples})")

    print(f"{name}: {len(train_ids)} train, {len(test_ids)} test; "
          f"{'PASS' if not problems else 'FAIL'}")
    for problem in problems:
        print(f"  - {problem}")
    return not problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("./data"),
                        help="Directory containing the three dataset folders")
    parser.add_argument("--dataset", choices=IMAGE_FOLDERS, nargs="+",
                        default=list(IMAGE_FOLDERS), help="Datasets to check")
    args = parser.parse_args()

    passed = True
    for name in args.dataset:
        passed = check_dataset(args.root, name) and passed
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
