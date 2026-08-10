"""Command-line interface for brainframe.

Subcommands:

* ``prepare``         -- create the data/checkpoint directory layout.
* ``download-sam``   -- fetch a SAM checkpoint.
* ``download-data``  -- print instructions for acquiring OASIS-1 / BraTS.
* ``classify``       -- train/predict the supervised classification baseline.
* ``segment``        -- run retraining-free segmentation on a volume.
* ``reconstruct``    -- build 3D meshes + metrics from a segmentation.
* ``evaluate``       -- run computational therapy evaluation.
* ``visualize``      -- render 3D meshes / cross-sections.
* ``run``            -- run the full pipeline on a volume.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from brainframe.config import LABELS, load_config
from brainframe.data.loaders import load_volume, save_volume
from brainframe.utils.io import ensure_dir
from brainframe.utils.logging import get_logger
from brainframe.utils.seed import set_seed

log = get_logger("cli")


def _common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/default.yaml", help="Master config YAML")
    p.add_argument("--device", default=None, help="Override device (cpu/cuda/mps/auto)")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--seed", type=int, default=None, help="Random seed")


def _resolve_cfg(args) -> object:
    cfg = load_config(args.config)
    if getattr(args, "device", None):
        cfg.device = args.device
    if getattr(args, "seed", None) is not None:
        cfg.seed = args.seed
    if getattr(args, "output_dir", None):
        cfg.paths.output_dir = args.output_dir
    set_seed(cfg.seed)
    return cfg


def cmd_prepare(args) -> int:
    cfg = _resolve_cfg(args)
    for d in [
        cfg.paths.raw_dir,
        cfg.paths.processed_dir,
        cfg.paths.checkpoint_dir,
        cfg.paths.output_dir,
        cfg.pipeline.cache_dir,
    ]:
        ensure_dir(d)
        print(f"prepared {d}")
    return 0


def cmd_download_sam(args) -> int:
    cfg = _resolve_cfg(args)
    from brainframe.segmentation.sam_wrapper import SAM_FILENAME, _download_checkpoint

    model = args.model_type or cfg.segmentation.sam_model_type
    dest = ensure_dir(cfg.paths.checkpoint_dir) / SAM_FILENAME[model]
    if dest.exists() and not args.force:
        print(f"checkpoint exists: {dest}")
        return 0
    res = _download_checkpoint(model, dest)
    if res:
        print(f"downloaded {model} -> {res}")
        return 0
    print(f"download failed for {model} (set --force to retry)")
    return 1


def cmd_download_data(args) -> int:
    print("Data must be acquired manually:")
    print("  OASIS-1: https://www.oasis-brains.org/  (request access, place under data/raw/oasis)")
    print("  BraTS:   https://www.med.upenn.edu/cbica/brats/  (place under data/raw/brats)")
    print("See docs/reproducibility.md for details.")
    return 0


def cmd_classify(args) -> int:
    cfg = _resolve_cfg(args)
    from brainframe.classification.models import build_classifier

    model = build_classifier(cfg.classification.model)
    if args.image:
        from brainframe.classification.predict import predict_volume
        from brainframe.utils.device import resolve_device

        device = resolve_device(cfg.device)
        out = predict_volume(
            model, args.image, device=device, patch_size=cfg.classification.data.patch_size
        )
        import json

        print(json.dumps(out, indent=2))
    else:
        from brainframe.classification.train import train_classifier

        if not args.data_root:
            print("Either --image for prediction or --data-root for training is required.")
            return 2
        from torch.utils.data import DataLoader

        from brainframe.data.datasets import OasisDataset, split_indices

        records = OasisDataset(args.data_root)
        if len(records) == 0:
            print(f"No subjects found under {args.data_root}")
            return 1
        tr, va, te = split_indices(len(records), seed=cfg.seed)
        train_ds = OasisDataset(
            args.data_root, patch_size=cfg.classification.data.patch_size, augment=True, indices=tr
        )
        val_ds = OasisDataset(
            args.data_root, patch_size=cfg.classification.data.patch_size, augment=False, indices=va
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.classification.train.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_ds, batch_size=cfg.classification.train.batch_size)
        train_classifier(
            model,
            train_loader,
            val_loader,
            cfg.classification,
            device=cfg.device,
            history_path=str(ensure_dir(cfg.paths.output_dir) / "train_history.json"),
        )
    return 0


def cmd_segment(args) -> int:
    cfg = _resolve_cfg(args)
    from brainframe.segmentation.inference import segment_volume
    from brainframe.segmentation.sam_wrapper import build_segmenter

    if not args.input:
        print("--input <volume.nii.gz> is required")
        return 2
    result = load_volume(args.input)
    segmenter = build_segmenter(
        cfg.segmentation, device=cfg.device, allow_download=cfg.sam.auto_download
    )
    seg = segment_volume(
        result.volume, cfg.segmentation, spacing=result.spacing, segmenter=segmenter
    )
    out = Path(args.output or (ensure_dir(cfg.paths.output_dir) / "label_volume.npy"))
    np.save(str(out), seg.label_volume)
    print(f"segmentation written to {out} (shape={seg.label_volume.shape}, labels={LABELS})")
    if args.save_nifti:
        save_volume(seg.label_volume.astype(np.float32), args.save_nifti, spacing=result.spacing)
        print(f"NIfTI written to {args.save_nifti}")
    return 0


def cmd_reconstruct(args) -> int:
    cfg = _resolve_cfg(args)
    if not args.input:
        print("--input <label_volume.npy> is required")
        return 2
    label_volume = np.load(args.input).astype(np.int16)
    out_dir = Path(args.output or (ensure_dir(cfg.paths.output_dir) / "reconstruction"))
    from brainframe.pipeline import run_reconstruction

    spacing = tuple(float(s) for s in (args.spacing.split(",") if args.spacing else [1, 1, 1]))
    recon = run_reconstruction(label_volume, cfg, out_dir, spacing=spacing)
    print(f"meshes: {recon['mesh_paths']}")
    print(f"metrics: {recon['metrics_path']}")
    return 0


def cmd_evaluate(args) -> int:
    cfg = _resolve_cfg(args)
    if not args.input:
        print("--input <label_volume.npy> is required")
        return 2
    label_volume = np.load(args.input).astype(np.int16)
    out_dir = Path(args.output or (ensure_dir(cfg.paths.output_dir) / "evaluation"))
    from brainframe.pipeline import run_evaluation

    spacing = tuple(float(s) for s in (args.spacing.split(",") if args.spacing else [1, 1, 1]))
    ev = run_evaluation(label_volume, cfg, out_dir, spacing=spacing)
    print(f"compatibility score: {ev['score']:.3f}")
    print(f"report: {ev['report'].get('figures', {}).get('html')}")
    return 0


def cmd_visualize(args) -> int:
    cfg = _resolve_cfg(args)
    if not args.input:
        print("--input <label_volume.npy> is required")
        return 2
    label_volume = np.load(args.input).astype(np.int16)
    out_dir = Path(args.output or (ensure_dir(cfg.paths.output_dir) / "visualize"))
    ensure_dir(out_dir)
    from brainframe.reconstruction.marching import extract_meshes
    from brainframe.reconstruction.visualize import render_3d, save_cross_sections

    meshes = extract_meshes(label_volume, cfg=cfg.reconstruction)
    save_cross_sections(label_volume, out_dir / "figures")
    render_3d(
        meshes, cfg=cfg.reconstruction, out_path=out_dir / "figures" / "reconstruction_3d.html"
    )
    print(f"visualizations in {out_dir}")
    return 0


def cmd_run(args) -> int:
    cfg = _resolve_cfg(args)
    if not args.input:
        print("--input <volume.nii.gz> is required")
        return 2
    result = load_volume(args.input)
    out_dir = Path(args.output or (ensure_dir(cfg.paths.output_dir) / "pipeline"))
    from brainframe.pipeline import run_pipeline

    res = run_pipeline(
        result.volume,
        cfg,
        output_dir=out_dir,
        device=cfg.device,
        stages=args.stages or cfg.pipeline.stages,
        spacing=result.spacing,
    )
    print(f"pipeline result: {res.to_dict()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="brainframe", description="brainframe CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="Create data/checkpoint directory layout")
    _common_args(p)
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("download-sam", help="Download a SAM checkpoint")
    _common_args(p)
    p.add_argument("--model-type", default=None, choices=["vit_h", "vit_l", "vit_b"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download_sam)

    p = sub.add_parser("download-data", help="Print data acquisition instructions")
    _common_args(p)
    p.set_defaults(func=cmd_download_data)

    p = sub.add_parser("classify", help="Train/predict the supervised classifier")
    _common_args(p)
    p.add_argument("--image", default=None, help="Single volume to predict")
    p.add_argument("--data-root", default=None, help="Dataset root for training")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("segment", help="Run retraining-free SAM segmentation")
    _common_args(p)
    p.add_argument("--input", default=None, help="Input volume (.nii/.nii.gz)")
    p.add_argument("--output", default=None, help="Output label volume (.npy)")
    p.add_argument("--save-nifti", default=None, help="Also save segmentation as NIfTI")
    p.set_defaults(func=cmd_segment)

    p = sub.add_parser("reconstruct", help="Reconstruct 3D meshes + metrics")
    _common_args(p)
    p.add_argument("--input", default=None, help="Input label volume (.npy)")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--spacing", default="1,1,1", help="Voxel spacing (mm)")
    p.set_defaults(func=cmd_reconstruct)

    p = sub.add_parser("evaluate", help="Run computational therapy evaluation")
    _common_args(p)
    p.add_argument("--input", default=None, help="Input label volume (.npy)")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--spacing", default="1,1,1", help="Voxel spacing (mm)")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("visualize", help="Render 3D meshes / cross-sections")
    _common_args(p)
    p.add_argument("--input", default=None, help="Input label volume (.npy)")
    p.add_argument("--output", default=None, help="Output directory")
    p.set_defaults(func=cmd_visualize)

    p = sub.add_parser("run", help="Run the full pipeline on a volume")
    _common_args(p)
    p.add_argument("--input", default=None, help="Input volume (.nii/.nii.gz)")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument(
        "--stages", nargs="*", default=None, help="Subset of stages (segment reconstruct evaluate)"
    )
    p.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
