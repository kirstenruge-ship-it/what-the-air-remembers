"""Verify the frozen input manifest and produce its planned, per-trial welfare ledger."""

import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

FROZEN_TRIAL_SHA256 = "4516192c45ff56feaeef616ef3f798ae5e1c280cc3dc9906a66bc23476499631"
SHAMS = {"sham", "clean", "named", "text", "repeat_sham"}


def main(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["file_sha256"]["trials.jsonl"] == FROZEN_TRIAL_SHA256
    for name, expected in manifest["file_sha256"].items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected, (
            name
        )
    rows = [
        json.loads(line) for line in (folder / "trials.jsonl").read_text().splitlines()
    ]
    by_id = {row["trial_id"]: row for row in rows}
    assert len(by_id) == len(rows) == manifest["trial_count"]
    counts = Counter()
    with (folder / "welfare-classes.csv").open("w", newline="") as file:
        fields = [
            "trial_id",
            "stage",
            "arm",
            "visible_conflict_intensity",
            "latent_change",
            "planned_nonzero_positions",
            "named_low_pleasantness_reference",
            "third_person_mild_affect",
            "human_identifiers_in_prompt",
        ]
        output = csv.DictWriter(file, fieldnames=fields)
        output.writeheader()
        for r in rows:
            stage, arm = r["stage"], r["arm"]
            base = by_id[r["baseline_id"]]
            assert base["seed"] == r["seed"] and base["input_ids"] == r["input_ids"]
            assert r["reasoning_effort"] == "max" and r["temperature"] == 1
            assert not ({"continuation_ids", "max_tokens", "max_new_tokens"} & set(r))
            assert r["input_ids"][-3:] == [163587, 39964, 163589]
            assert r["input_ids"][r["slot"]] == r["slot_id"] == 4665
            assert len(r["input_ids"]) <= 257
            active = arm not in SHAMS
            positions = int(active)
            if stage == "encoding":
                assert len(r["stem_positions"]) == 2
                assert all(
                    r["input_ids"][i] == r["slot_id"] for i in r["stem_positions"]
                )
                positions = sum(v != 0 for v in r["write_signs"])
            assert bool(positions) == active
            mild = stage in {"household", "encoding"}
            assert (
                "mildly frustrated" in r["user"] and "relieved" in r["user"]
            ) == mild
            low_ref = "butyric acid" in r["user"]
            if low_ref:
                assert stage == "profile" and arm == "clean"
            output.writerow(
                {
                    "trial_id": r["trial_id"],
                    "stage": stage,
                    "arm": arm,
                    "visible_conflict_intensity": int(mild),
                    "latent_change": active,
                    "planned_nonzero_positions": positions,
                    "named_low_pleasantness_reference": low_ref,
                    "third_person_mild_affect": mild,
                    "human_identifiers_in_prompt": False,
                }
            )
            counts.update(
                {
                    "requests": 1,
                    "active": int(active),
                    "nonzero_positions": positions,
                    "low_reference_rows": int(low_ref),
                    "mild_affect": int(mild),
                }
            )
    assert counts == {
        "requests": 2852,
        "active": 1432,
        "nonzero_positions": 2200,
        "low_reference_rows": 80,
        "mild_affect": 912,
    }
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent)
