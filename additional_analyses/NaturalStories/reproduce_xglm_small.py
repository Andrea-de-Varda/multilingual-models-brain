#!/usr/bin/env python3
"""Focused reproduction of the Natural Stories XGLM-small shift-3 benchmark.

This script intentionally leaves the upstream Natural Stories scripts
unchanged. It extracts the same word-level representations as
``get_natstor_embeddings.py``, applies the same two-second binning and
leave-one-story-out RidgeCV procedure as ``fit_encoding_natstor.py``, and
compares every reproduced layer/story correlation with the released result.

Generated embeddings and results must be written outside the Git repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import numpy.ma as ma
import pandas as pd
import scipy
import sklearn
import torch
import transformers
from scipy.stats import pearsonr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from transformers import XGLMForCausalLM, XGLMTokenizer


MODEL_ID = "facebook/xglm-564M"
MODEL_LABEL = "xglm_small"
SHIFT_LABEL = "3shift"
STORIES = ("1", "2", "3", "4", "5", "6", "7", "9", "10")
STORY_NAMES = {
    "1": "boar",
    "2": "aqua",
    "3": "matchstickseller",
    "4": "kingofbirds",
    "5": "elvis",
    "6": "mrsticky",
    "7": "highschool",
    "9": "tulips",
    "10": "tree",
}
RIDGE_ALPHAS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=here)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--model-revision", default="main")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stories", nargs="+", default=list(STORIES), choices=STORIES)
    parser.add_argument(
        "--held-out-stories",
        nargs="+",
        default=list(STORIES),
        choices=STORIES,
        help="Held-out folds to fit. Training still uses all other canonical stories.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Layers to fit after extraction; default is every extracted layer.",
    )
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--reference-tolerance", type=float, default=1e-5)
    args = parser.parse_args()
    if args.extract_only and args.fit_only:
        parser.error("--extract-only and --fit-only are mutually exclusive")
    return args


def save_pickle(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_metadata() -> dict:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def embedding_path(output_dir: Path, story: str) -> Path:
    return output_dir / "embeddings" / f"{MODEL_LABEL}_{story}.pkl"


def transcript_words(data_dir: Path, story: str) -> tuple[pd.DataFrame, List[str]]:
    transcript_path = data_dir / "transcribed" / f"{story}.csv"
    frame = pd.read_csv(transcript_path)
    words = frame["text"].str.cat(sep=" ").split()
    if not words:
        raise ValueError(f"No transcript words found in {transcript_path}")
    return frame, words


def word_token_lengths(words: Sequence[str], tokenizer: XGLMTokenizer) -> List[int]:
    lengths: List[int] = []
    for word in words:
        tokens = tokenizer.tokenize(word)
        if not tokens:
            raise ValueError(f"Tokenizer produced no token for word {word!r}")
        # Upstream removes the XGLM sentence-piece marker from the first token.
        # That operation does not alter the number of tokens used for averaging.
        lengths.append(len(tokens))
    return lengths


@torch.inference_mode()
def extract_story_embeddings(
    story: str,
    data_dir: Path,
    output_dir: Path,
    tokenizer: XGLMTokenizer,
    model: XGLMForCausalLM,
    device: torch.device,
    overwrite: bool,
) -> dict:
    destination = embedding_path(output_dir, story)
    if destination.exists() and not overwrite:
        print(f"[extract] story={story}: using {destination}", flush=True)
        return load_pickle(destination)

    transcript, words = transcript_words(data_dir, story)
    sentence = " ".join(words)
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    print(
        f"[extract] story={story} words={len(words)} model_tokens={input_ids.shape[1]}",
        flush=True,
    )
    outputs = model(input_ids.to(device), output_hidden_states=True, return_dict=True)

    # XGLM prepends one special token. This is emb_start=1 in the upstream code.
    token_embeddings = [state[0, 1:, :].detach().cpu().float().numpy() for state in outputs.hidden_states]
    lengths = word_token_lengths(words, tokenizer)
    expected_tokens = sum(lengths)
    observed_tokens = token_embeddings[0].shape[0]
    if observed_tokens != expected_tokens:
        raise ValueError(
            f"Story {story}: {observed_tokens} non-special embeddings for "
            f"{expected_tokens} transcript subword tokens"
        )

    by_layer: Dict[int, np.ndarray] = {}
    for layer, embeddings in enumerate(token_embeddings):
        word_vectors = []
        cursor = 0
        for length in lengths:
            word_vectors.append(embeddings[cursor : cursor + length].mean(axis=0))
            cursor += length
        by_layer[layer] = np.vstack(word_vectors).astype(np.float32, copy=False)

    payload = {
        "story": story,
        "model": str(getattr(model.config, "_name_or_path", MODEL_ID)),
        "model_commit_hash": getattr(model.config, "_commit_hash", None),
        "transcript_path": str((data_dir / "transcribed" / f"{story}.csv").resolve()),
        "transcript_sha256": sha256(data_dir / "transcribed" / f"{story}.csv"),
        "word_count": len(words),
        "model_token_count_with_special": int(input_ids.shape[1]),
        "layers": by_layer,
    }
    save_pickle(payload, destination)
    print(f"[extract] story={story}: saved {destination}", flush=True)
    return payload


def impute_nan_columns(array: np.ndarray) -> np.ndarray:
    # Exact operation used by upstream fit_encoding_natstor.py.
    masked_means = ma.array(array, mask=np.isnan(array)).mean(axis=0)
    return np.where(np.isnan(array), masked_means, array)


def bin_words_to_two_seconds(
    embeddings: np.ndarray,
    transcript: pd.DataFrame,
    number_of_bins: int,
) -> np.ndarray:
    if embeddings.shape[0] != len(transcript):
        raise ValueError(
            f"Embedding/transcript mismatch: {embeddings.shape[0]} vectors for {len(transcript)} rows"
        )
    time_grid = np.arange(0, number_of_bins * 2, 2)
    word_ends = transcript["end"].to_numpy(dtype=float)
    word_bins = np.zeros(len(word_ends), dtype=int)
    for index, end_time in enumerate(word_ends):
        candidates = np.where(end_time > time_grid)[0]
        if candidates.size == 0:
            raise ValueError(f"Word ending at {end_time} s precedes the first time bin")
        word_bins[index] = int(candidates[-1])

    binned = []
    for bin_index in range(number_of_bins):
        selected = embeddings[word_bins == bin_index]
        if selected.size == 0:
            binned.append(np.full(embeddings.shape[1], np.nan, dtype=np.float32))
        else:
            binned.append(selected.mean(axis=0))
    return impute_nan_columns(np.asarray(binned))


def fit_reproduction(
    data_dir: Path,
    output_dir: Path,
    layers: Sequence[int] | None,
    held_out_stories: Sequence[str],
) -> dict:
    response = load_pickle(data_dir / "response" / "d_shift_3")
    payloads = {story: load_pickle(embedding_path(output_dir, story)) for story in STORIES}
    transcripts = {
        story: pd.read_csv(data_dir / "transcribed" / f"{story}.csv") for story in STORIES
    }
    first_payload = payloads[STORIES[0]]
    available_layers = sorted(int(layer) for layer in first_payload["layers"])
    selected_layers = available_layers if layers is None else list(layers)
    unknown = sorted(set(selected_layers) - set(available_layers))
    if unknown:
        raise ValueError(f"Requested unavailable layers: {unknown}; available={available_layers}")

    layerwise: Dict[int, pd.DataFrame] = {}
    rows: List[dict] = []
    for layer in selected_layers:
        print(f"[fit] layer={layer}", flush=True)
        binned = {
            story: bin_words_to_two_seconds(
                payloads[story]["layers"][layer],
                transcripts[story],
                len(response[STORY_NAMES[story]]),
            )
            for story in STORIES
        }
        layer_rows = []
        for held_out in held_out_stories:
            training_stories = [story for story in STORIES if story != held_out]
            x_train = np.concatenate([binned[story] for story in training_stories])
            y_train = np.concatenate([response[STORY_NAMES[story]] for story in training_stories])
            x_test = binned[held_out]
            y_test = np.asarray(response[STORY_NAMES[held_out]])

            x_scaler = StandardScaler()
            y_scaler = StandardScaler()
            x_train = x_scaler.fit_transform(x_train)
            x_test = x_scaler.transform(x_test)
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

            regressor = RidgeCV(alphas=RIDGE_ALPHAS)
            regressor.fit(x_train, y_train)
            prediction = regressor.predict(x_test)
            correlation = float(pearsonr(y_test, prediction).statistic)
            row = {
                "layer": layer,
                "lang": held_out,
                "story": STORY_NAMES[held_out],
                "r": correlation,
                "alpha": float(regressor.alpha_),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
            }
            rows.append(row)
            layer_rows.append([held_out, correlation])
            print(
                f"[fit] layer={layer} held_out={held_out} r={correlation:.9f} "
                f"alpha={regressor.alpha_:g}",
                flush=True,
            )
        layerwise[layer] = pd.DataFrame(layer_rows, columns=["lang", "r"])

    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(layerwise, result_dir / "cross_story_3shift_xglm_small_reproduced.pkl")
    long_frame = pd.DataFrame(rows)
    long_frame.to_csv(result_dir / "cross_story_3shift_xglm_small_reproduced.csv", index=False)
    means = long_frame.groupby("layer", as_index=False).agg(
        mean=("r", "mean"),
        std=("r", "std"),
        count=("r", "count"),
    )
    means.to_csv(result_dir / "layer_means.csv", index=False)
    return {"layerwise": layerwise, "long_frame": long_frame, "means": means}


def compare_with_reference(
    data_dir: Path,
    output_dir: Path,
    reproduced: Mapping[int, pd.DataFrame],
    tolerance: float,
) -> dict:
    reference_path = data_dir / "results" / "cross_story_3shift_xglm_small"
    reference = load_pickle(reference_path)
    comparison_rows = []
    for layer, reproduced_frame in reproduced.items():
        reference_frame = reference[layer].copy()
        reference_frame["lang"] = reference_frame["lang"].astype(str)
        reproduced_frame = reproduced_frame.copy()
        reproduced_frame["lang"] = reproduced_frame["lang"].astype(str)
        merged = reference_frame.merge(reproduced_frame, on="lang", suffixes=("_reference", "_reproduced"))
        for row in merged.itertuples(index=False):
            difference = float(row.r_reproduced - row.r_reference)
            comparison_rows.append(
                {
                    "layer": int(layer),
                    "lang": str(row.lang),
                    "r_reference": float(row.r_reference),
                    "r_reproduced": float(row.r_reproduced),
                    "difference": difference,
                    "absolute_difference": abs(difference),
                }
            )

    frame = pd.DataFrame(comparison_rows)
    result_dir = output_dir / "results"
    frame.to_csv(result_dir / "reference_comparison.csv", index=False)
    correlation_across_values = (
        float(pearsonr(frame["r_reference"], frame["r_reproduced"]).statistic)
        if len(frame) >= 2
        else None
    )
    summary = {
        "reference_path": str(reference_path.resolve()),
        "reference_sha256": sha256(reference_path),
        "n_compared": int(len(frame)),
        "max_absolute_difference": float(frame["absolute_difference"].max()),
        "mean_absolute_difference": float(frame["absolute_difference"].mean()),
        "correlation_across_values": correlation_across_values,
        "tolerance": float(tolerance),
    }
    summary["passes_tolerance"] = summary["max_absolute_difference"] <= tolerance
    return {"frame": frame, "summary": summary}


def write_report(output_dir: Path, comparison: dict, means: pd.DataFrame) -> None:
    summary = comparison["summary"]
    peak_row = means.loc[means["mean"].idxmax()]
    value_correlation = summary["correlation_across_values"]
    value_correlation_text = (
        f"{value_correlation:.9f}" if value_correlation is not None else "not computed (fewer than two values)"
    )
    report = [
        "# Natural Stories XGLM-small shift-3 reproduction",
        "",
        f"- Compared values: {summary['n_compared']}",
        f"- Maximum absolute difference: {summary['max_absolute_difference']:.9g}",
        f"- Mean absolute difference: {summary['mean_absolute_difference']:.9g}",
        f"- Correlation across released and reproduced values: {value_correlation_text}",
        f"- Requested tolerance: {summary['tolerance']:.9g}",
        f"- Passes tolerance: {summary['passes_tolerance']}",
        f"- Reproduced peak layer: {int(peak_row['layer'])}",
        f"- Reproduced peak mean held-out-story r: {peak_row['mean']:.9f}",
        "",
        "Detailed values are in `results/reference_comparison.csv`.",
    ]
    (output_dir / "REPRODUCTION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.time()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"data_dir={data_dir}", flush=True)
    print(f"output_dir={output_dir}", flush=True)

    if not args.fit_only:
        device = torch.device(args.device)
        tokenizer = XGLMTokenizer.from_pretrained(args.model, revision=args.model_revision)
        model = XGLMForCausalLM.from_pretrained(args.model, revision=args.model_revision)
        model.eval().to(device)
        for story in args.stories:
            extract_story_embeddings(
                story,
                data_dir,
                output_dir,
                tokenizer,
                model,
                device,
                args.overwrite_embeddings,
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    run_metadata = {
        "model": args.model,
        "model_revision": args.model_revision,
        "shift": 3,
        "stories": list(args.stories),
        "canonical_encoding_stories": list(STORIES),
        "held_out_stories": list(args.held_out_stories),
        "layers": args.layers,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "environment": environment_metadata(),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "elapsed_seconds": None,
    }

    if not args.extract_only:
        fitted = fit_reproduction(
            data_dir,
            output_dir,
            args.layers,
            args.held_out_stories,
        )
        comparison = compare_with_reference(
            data_dir,
            output_dir,
            fitted["layerwise"],
            args.reference_tolerance,
        )
        run_metadata["comparison"] = comparison["summary"]
        write_report(output_dir, comparison, fitted["means"])
        print(json.dumps(comparison["summary"], indent=2), flush=True)

    run_metadata["elapsed_seconds"] = time.time() - started
    (output_dir / "run_manifest.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"elapsed_seconds={run_metadata['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
