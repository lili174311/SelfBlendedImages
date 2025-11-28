"""Export self-blended fake/real pairs without altering the training pipeline.

This script reads the same preprocessed frames and augmentations used during
training, but only writes PNG files to disk; it does not change checkpoints or
any training state.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.sbi import SBI_Dataset


def to_uint8_image(tensor: torch.Tensor) -> Image.Image:
    """Convert CxHxW tensor in [0,1] to a PIL image."""
    array = tensor.detach().cpu().numpy()
    array = np.clip(array.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def save_pair_batch(
    fake_batch: torch.Tensor, real_batch: torch.Tensor, fake_dir: Path, real_dir: Path, start_idx: int
) -> int:
    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for i in range(fake_batch.shape[0]):
        fake_img = to_uint8_image(fake_batch[i])
        real_img = to_uint8_image(real_batch[i])

        fake_path = fake_dir / f"fake_{start_idx + i:06d}.png"
        real_path = real_dir / f"real_{start_idx + i:06d}.png"

        fake_img.save(fake_path)
        real_img.save(real_path)
        saved += 1

    return saved


def validate_existing_pairs(fake_dir: Path, real_dir: Path) -> int:
    """检查已有导出是否成对，并给出续写的下一个编号。

    - 若 fake/real 文件数量不一致，则抛出异常，避免覆盖错误文件。
    - 若名称不匹配（如 fake_000003 缺失），仍会用已存在文件中的最大编号+1 继续。
    """

    def max_idx(directory: Path, prefix: str) -> int:
        pattern = re.compile(rf"{prefix}_(\d+)\.png")
        max_seen = -1
        for file in directory.glob(f"{prefix}_*.png"):
            match = pattern.fullmatch(file.name)
            if match:
                max_seen = max(max_seen, int(match.group(1)))
        return max_seen

    fake_count = len(list(fake_dir.glob("fake_*.png")))
    real_count = len(list(real_dir.glob("real_*.png")))
    if fake_count != real_count:
        raise ValueError(
            f"发现已有导出时 fake/real 数量不一致：fake={fake_count}, real={real_count}，请手动检查后再运行。"
        )

    return max(max_idx(fake_dir, "fake"), max_idx(real_dir, "real")) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export SBI real/fake image pairs to disk.")
    parser.add_argument("--phase", choices=["train", "val", "test"], default="train", help="Dataset split to export.")
    parser.add_argument("--image-size", type=int, default=224, help="Resolution passed to SBI_Dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("export_sbi"),
        help="Directory where real/ and fake/ folders will be created.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=0,
        help="Number of pairs to export (0 means export the entire dataset).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue numbering after existing fake_/real_ PNGs instead of starting from 000000.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing exported images if output directories are non-empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite cannot be used together")

    dataset = SBI_Dataset(phase=args.phase, image_size=args.image_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=dataset.worker_init_fn,
    )
    # Each dataset index corresponds to a single preprocessed frame that already has
    # landmark and RetinaFace metadata; __getitem__ returns a cropped real face and a
    # self-blended fake generated from that same frame. We export both as numbered PNGs
    # to keep fake/real pairs aligned 1:1.

    fake_dir = args.output_dir / "fake"
    real_dir = args.output_dir / "real"

    fake_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    if not args.overwrite:
        existing_files = list(fake_dir.glob("*.png")) + list(real_dir.glob("*.png"))
        if existing_files and not args.resume:
            raise FileExistsError(
                f"Found existing PNGs under {args.output_dir}. Use --resume to append new files "
                "or --overwrite to allow replacing them."
            )

    if args.resume:
        saved = validate_existing_pairs(fake_dir, real_dir)
    else:
        saved = 0

    target_total = len(dataset) if args.num_samples <= 0 else args.num_samples

    for fake_batch, real_batch in tqdm(dataloader, desc="Exporting SBI image pairs"):
        if saved >= target_total:
            break

        if saved + fake_batch.shape[0] > target_total:
            excess = saved + fake_batch.shape[0] - target_total
            keep = fake_batch.shape[0] - excess
            fake_batch = fake_batch[:keep]
            real_batch = real_batch[:keep]

        saved_in_batch = save_pair_batch(fake_batch, real_batch, fake_dir, real_dir, start_idx=saved)
        saved += saved_in_batch

    print(f"Saved {saved} pairs to {args.output_dir}")


if __name__ == "__main__":
    main()
