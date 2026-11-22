"""Split-half reproducibility of the frozen human-derived odor judgments; CPU only."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def evaluate(stimuli, output, repetitions=1000, seed=11212026):
    stimuli = Path(stimuli)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((stimuli / "manifest.json").read_text())
    roster = json.loads((stimuli / "roster.json").read_text())
    scale_map = manifest["quality_sd_in_fitting"]
    qualities = sorted(scale_map)
    scales = np.array([scale_map[q] for q in qualities], dtype=float)
    landmarks = json.loads((stimuli / "landmarks.json").read_text())
    selected = [r for r in roster if r["split"] == "test"]
    lookup = {r["cid"]: r for r in roster}
    test = [a for a in landmarks if a["role"] == "test"]
    panel_path = stimuli / "source/selected-human-panel.csv.gz"
    raw = pd.read_csv(panel_path)
    raw["repeat"] = raw.groupby(["Stimulus", "Subject", "MeasurementValue"]).cumcount()
    wide = raw.pivot(
        index=["Stimulus", "Subject", "repeat"],
        columns="MeasurementValue",
        values="Value",
    )
    detected = (
        wide.loc[wide["CAN OR CAN'T SMELL"] == "I smell something", qualities]
        .apply(pd.to_numeric)
        .fillna(0)
    )
    per_subject = detected.groupby(level=["Stimulus", "Subject"]).mean()
    cids = sorted(
        {r["cid"] for r in selected} | {a[k] for a in test for k in ("left", "right")}
    )
    subjects = sorted(raw.Subject.unique())
    ratings = np.stack(
        [
            per_subject.xs(lookup[c]["stimulus_id"])
            .reindex(subjects)
            .to_numpy(dtype=float)
            for c in cids
        ]
    )
    index = {cid: i for i, cid in enumerate(cids)}
    cells = [
        (r["cid"], p["pair_id"], index[r["cid"]], index[p["left"]], index[p["right"]])
        for r in selected
        for p in test
    ]
    target, left, right = np.array([(r[2], r[3], r[4]) for r in cells]).T

    def margins(means):
        z = means / scales
        return np.linalg.norm(z[target] - z[right], axis=1) - np.linalg.norm(
            z[target] - z[left], axis=1
        )

    full = margins(np.nanmean(ratings, axis=1))
    ref = pd.read_csv(
        stimuli / "human_choices.csv", dtype={"cid": str, "pair_id": str}
    ).set_index(["cid", "pair_id"])
    error = max(
        abs(ref.loc[(r[0], r[1]), "human_margin"] - m)
        for r, m in zip(cells, full, strict=True)
    )
    if error > 1e-8:
        raise ValueError(f"Human benchmark reproduction failed: {error}")
    rates = np.empty((repetitions, len(cells)), dtype=float)
    rng = np.random.default_rng(seed)
    for rep in range(repetitions):
        ix = rng.permutation(len(subjects))
        a, b = np.array_split(ix, 2)
        ma, mb = (
            margins(np.nanmean(ratings[:, a, :], axis=1)),
            margins(np.nanmean(ratings[:, b, :], axis=1)),
        )
        rates[rep] = (np.sign(ma) == np.sign(mb)).astype(float)
        rates[rep, (ma == 0) | (mb == 0)] = 0.5
    rr = rates.mean(axis=0)
    frame = pd.DataFrame(
        {
            "cid": [r[0] for r in cells],
            "pair_id": [r[1] for r in cells],
            "frozen_human_margin": full,
            "split_half_sign_agreement": rr,
        }
    )
    frame.to_csv(output / "human-reliability-pairs.csv", index=False)
    out = {
        "scope": "Human observations only. Repeated balanced random splits of the same rater panel; not independent replications or an interval for a new population.",
        "data_sha256": hashlib.sha256(panel_path.read_bytes()).hexdigest(),
        "seed": seed,
        "repetitions": repetitions,
        "test_compounds": len(selected),
        "reference_pairs": len(test),
        "test_comparisons": len(cells),
        "distinct_raters": len(subjects),
        "reproduced_max_abs_margin_error": float(error),
        "mean_split_agreement": float(rr.mean()),
        "median_split_agreement": float(np.median(rr)),
        "split_agreement_range": [float(rr.min()), float(rr.max())],
        "comparisons_ge_0_8": int(np.sum(rr >= 0.8)),
        "comparisons_below_0_6": int(np.sum(rr < 0.6)),
        "mean_agreement_ge_0_8_subset_threshold_is_descriptive": True,
        "full_numerical_comparisons_retained": len(cells),
    }
    (output / "human-reliability.json").write_text(
        json.dumps(out, indent=2, sort_keys=True) + "\n"
    )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stimuli", type=Path, default=Path(__file__).resolve().parents[1] / "stimuli"
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(evaluate(args.stimuli, args.output), indent=2))
