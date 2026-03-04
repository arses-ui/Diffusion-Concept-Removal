#!/usr/bin/env python3
"""
One-hour proof benchmark for CURE vs CURE-Sequential.

Outputs:
1) Evidence base CURE changes target concept behavior (single-concept delta)
2) Evidence sequential orthogonalized method is better over many concepts
3) Numeric comparison table (CLIP similarity metrics)

Notes:
- Uses lightweight CLIP-based proxy metrics for fast turnaround.
- Intended for slide evidence, not final paper-quality benchmarking.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import torch

# Ensure repo root is importable when script is run from any working directory.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@dataclass
class EvalCheckpoint:
    step: int
    n_erased: int
    target_clip: float
    retention_clip: float
    retention_drop_vs_base: float


@dataclass
class MethodResult:
    method: str
    single_concept_before: float
    single_concept_after: float
    single_concept_delta: float
    baseline_retention_clip: float
    checkpoints: List[EvalCheckpoint]


SAFE_PROMPTS = [
    "a scenic mountain landscape",
    "a wooden chair in a room",
    "a bowl of fruit on a table",
    "a city street at night",
    "a sailing boat on the ocean",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_concepts(concepts_csv: str) -> List[str]:
    concepts = [c.strip() for c in concepts_csv.split(",") if c.strip()]
    if not concepts:
        raise ValueError("No concepts parsed from --concepts")
    return concepts


def detect_device(requested: Optional[str]) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_clip_scorer(model_id: str, device: str):
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return model, processor


def clip_diag_similarity(
    images,
    prompts: List[str],
    clip_model,
    clip_processor,
    device: str,
) -> float:
    if len(images) != len(prompts):
        raise ValueError("images and prompts must be same length for diagonal CLIP score.")
    inputs = clip_processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image  # [N, N]
        diag = torch.diagonal(logits, 0)
        return float(diag.mean().item())


def make_sd14_pipe(device: str, cache_dir: Path, dtype: torch.dtype):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        torch_dtype=dtype,
        safety_checker=None,
        cache_dir=str(cache_dir),
    )
    return pipe.to(device)


def generate_images(
    eraser,
    prompts: List[str],
    seed: int,
    steps: int,
    guidance: float,
    height: int,
    width: int,
):
    images = []
    for i, prompt in enumerate(prompts):
        gen = torch.Generator(device="cpu").manual_seed(seed + i)
        out = eraser.generate(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=gen,
            height=height,
            width=width,
        )
        images.append(out[0])
    return images


def evaluate_state(
    eraser,
    erased_concepts: List[str],
    clip_model,
    clip_processor,
    device: str,
    seed: int,
    steps: int,
    guidance: float,
    height: int,
    width: int,
):
    target_prompts = [f"a photo of a {c}" for c in erased_concepts]
    target_images = generate_images(
        eraser, target_prompts, seed=seed, steps=steps, guidance=guidance, height=height, width=width
    )
    target_clip = clip_diag_similarity(
        target_images, target_prompts, clip_model=clip_model, clip_processor=clip_processor, device=device
    )

    safe_images = generate_images(
        eraser, SAFE_PROMPTS, seed=seed + 10_000, steps=steps, guidance=guidance, height=height, width=width
    )
    retention_clip = clip_diag_similarity(
        safe_images, SAFE_PROMPTS, clip_model=clip_model, clip_processor=clip_processor, device=device
    )

    return target_clip, retention_clip


def erase_once(
    eraser,
    method: str,
    concept: str,
    alpha: float,
):
    from cure.utils import get_default_forget_prompts

    forget_prompts = get_default_forget_prompts(concept)
    if method == "cure":
        eraser.erase_concept(
            forget_prompts=forget_prompts,
            retain_prompts=None,
            alpha=alpha,
            save_original=False,
        )
        return

    if method == "cure_seq":
        eraser.erase_concept(
            forget_prompts=forget_prompts,
            retain_prompts=None,
            alpha=alpha,
            concept_name=concept,
            save_original=False,
        )
        return

    raise ValueError(f"Unknown method '{method}'")


def build_eraser(
    method: str,
    device: str,
    cache_dir: Path,
    embedding_mode: str,
):
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = make_sd14_pipe(device=device, cache_dir=cache_dir, dtype=dtype)

    if method == "cure":
        from cure import CURE

        return CURE(pipe, device=device, embedding_mode=embedding_mode)

    if method == "cure_seq":
        from cure_seq import SequentialCURE

        return SequentialCURE(pipe, device=device, embedding_mode=embedding_mode)

    raise ValueError(f"Unknown method '{method}'")


def run_method(
    method: str,
    concepts: List[str],
    alpha: float,
    eval_every: int,
    device: str,
    cache_dir: Path,
    embedding_mode: str,
    clip_model,
    clip_processor,
    seed: int,
    steps: int,
    guidance: float,
    height: int,
    width: int,
) -> MethodResult:
    eraser = build_eraser(
        method=method,
        device=device,
        cache_dir=cache_dir,
        embedding_mode=embedding_mode,
    )

    # Baseline scores before any erasure
    first_prompt = [f"a photo of a {concepts[0]}"]
    first_img_before = generate_images(
        eraser, first_prompt, seed=seed, steps=steps, guidance=guidance, height=height, width=width
    )
    first_score_before = clip_diag_similarity(
        first_img_before, first_prompt, clip_model=clip_model, clip_processor=clip_processor, device=device
    )

    safe_imgs_before = generate_images(
        eraser, SAFE_PROMPTS, seed=seed + 10_000, steps=steps, guidance=guidance, height=height, width=width
    )
    baseline_retention = clip_diag_similarity(
        safe_imgs_before, SAFE_PROMPTS, clip_model=clip_model, clip_processor=clip_processor, device=device
    )

    checkpoints: List[EvalCheckpoint] = []
    erased: List[str] = []

    for i, concept in enumerate(concepts, start=1):
        erase_once(eraser, method=method, concept=concept, alpha=alpha)
        erased.append(concept)

        if (i % eval_every == 0) or (i == len(concepts)):
            target_clip, retention_clip = evaluate_state(
                eraser=eraser,
                erased_concepts=erased,
                clip_model=clip_model,
                clip_processor=clip_processor,
                device=device,
                seed=seed,
                steps=steps,
                guidance=guidance,
                height=height,
                width=width,
            )
            checkpoints.append(
                EvalCheckpoint(
                    step=i,
                    n_erased=len(erased),
                    target_clip=target_clip,
                    retention_clip=retention_clip,
                    retention_drop_vs_base=(baseline_retention - retention_clip),
                )
            )
            print(
                f"[{method}] step={i:02d} target_clip={target_clip:.4f} "
                f"retention_clip={retention_clip:.4f} "
                f"drop={baseline_retention - retention_clip:.4f}"
            )

    # Score first concept after full run
    first_img_after = generate_images(
        eraser, first_prompt, seed=seed, steps=steps, guidance=guidance, height=height, width=width
    )
    first_score_after = clip_diag_similarity(
        first_img_after, first_prompt, clip_model=clip_model, clip_processor=clip_processor, device=device
    )

    # free memory
    del eraser
    if device == "cuda":
        torch.cuda.empty_cache()

    return MethodResult(
        method=method,
        single_concept_before=first_score_before,
        single_concept_after=first_score_after,
        single_concept_delta=(first_score_after - first_score_before),
        baseline_retention_clip=baseline_retention,
        checkpoints=checkpoints,
    )


def write_summary_md(out_path: Path, run: Dict):
    results = run["results"]
    cure = results["cure"]
    seq = results["cure_seq"]

    cure_last = cure["checkpoints"][-1]
    seq_last = seq["checkpoints"][-1]

    lines = [
        "# Quick Proof Benchmark Summary",
        "",
        "## Core Claims",
        "",
        "1. SD1.4 CURE implementation changes target concept behavior",
        "2. CURE-Sequential better preserves untargeted quality over multi-concept erasure",
        "3. Numeric comparison included below",
        "",
        "## Single-Concept Delta (First Concept)",
        "",
        "| Method | Before CLIP | After CLIP | Delta (After-Before, lower is better) |",
        "|---|---:|---:|---:|",
        f"| CURE | {cure['single_concept_before']:.4f} | {cure['single_concept_after']:.4f} | {cure['single_concept_delta']:.4f} |",
        f"| CURE-Sequential | {seq['single_concept_before']:.4f} | {seq['single_concept_after']:.4f} | {seq['single_concept_delta']:.4f} |",
        "",
        "## Final Multi-Concept Checkpoint",
        "",
        "| Method | #Erased | Target CLIP (lower better) | Retention CLIP (higher better) | Retention Drop vs Base (lower better) |",
        "|---|---:|---:|---:|---:|",
        f"| CURE | {cure_last['n_erased']} | {cure_last['target_clip']:.4f} | {cure_last['retention_clip']:.4f} | {cure_last['retention_drop_vs_base']:.4f} |",
        f"| CURE-Sequential | {seq_last['n_erased']} | {seq_last['target_clip']:.4f} | {seq_last['retention_clip']:.4f} | {seq_last['retention_drop_vs_base']:.4f} |",
        "",
        "## Notes",
        "",
        "- CLIP metrics are quick proxy metrics for slides (not final benchmark metrics).",
        "- Lower target CLIP indicates stronger suppression of erased concepts.",
        "- Lower retention drop indicates less collateral damage on untargeted prompts.",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="One-hour proof benchmark")
    parser.add_argument(
        "--concepts",
        type=str,
        default="car,dog,cat,french horn,golf ball,chain saw,cassette player,parachute,church,garbage truck",
        help="Comma-separated concept list in erase order",
    )
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--embedding-mode", type=str, default="mean_masked")
    parser.add_argument("--cache-dir", type=str, default="./models")
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output-dir", type=str, default="outputs/quick_proof")
    args = parser.parse_args()

    concepts = parse_concepts(args.concepts)
    device = detect_device(args.device)
    cache_dir = Path(args.cache_dir)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Concepts ({len(concepts)}): {concepts}")
    print(f"Output: {out_dir}")

    clip_model, clip_processor = load_clip_scorer(args.clip_model, device=device)

    cure_result = run_method(
        method="cure",
        concepts=concepts,
        alpha=args.alpha,
        eval_every=args.eval_every,
        device=device,
        cache_dir=cache_dir,
        embedding_mode=args.embedding_mode,
        clip_model=clip_model,
        clip_processor=clip_processor,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
    )

    seq_result = run_method(
        method="cure_seq",
        concepts=concepts,
        alpha=args.alpha,
        eval_every=args.eval_every,
        device=device,
        cache_dir=cache_dir,
        embedding_mode=args.embedding_mode,
        clip_model=clip_model,
        clip_processor=clip_processor,
        seed=args.seed,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
    )

    payload = {
        "created_at_utc": now_utc(),
        "config": {
            "concepts": concepts,
            "alpha": args.alpha,
            "eval_every": args.eval_every,
            "seed": args.seed,
            "steps": args.steps,
            "guidance": args.guidance,
            "height": args.height,
            "width": args.width,
            "device": device,
            "embedding_mode": args.embedding_mode,
            "cache_dir": str(cache_dir),
            "clip_model": args.clip_model,
        },
        "results": {
            "cure": asdict(cure_result),
            "cure_seq": asdict(seq_result),
        },
    }

    json_path = out_dir / "results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_summary_md(out_dir / "summary.md", payload)

    print(f"\nSaved: {json_path}")
    print(f"Saved: {out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
