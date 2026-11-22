"""Offline token audit for Astra's proposed odor-cue study. No network or inference.

Requires tiktoken and the four published, checksummed Kimi tokenizer assets.
Usage: python prompts.py --tokenizer PATH --molecules molecules.csv --output prompt-report.json
Use --verify-archive PATH to additionally compare the rendering with a prior native run.
Use --stimuli PATH to audit an exact JSONL manifest of {"family": "...", "user": "..."}
records in place of the engineering inventory; use the same carrier and no extra messages.
The length fixtures are engineering material. Final scientific prompts get a fresh audit.
"""

import argparse
import ast
import base64
import csv
from functools import lru_cache
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import zipfile
import sys

import tiktoken

REVISION = "d9e99b0792a5e93b78ea9bea62a68167e36ddd9e"
ASSETS = {
    "encoding_k3.py": "367f0a7ab8800f78f7e09dc31e58544ee7fc46eea992ea05b32d53506a921cc5",
    "tokenization_kimi.py": "204ee39fbf13a6d6608b0d3ca1b4b04a14ac5f0a062a7fdee02e61f5d936f9c6",
    "tokenizer_config.json": "9fc322181b8ec44236e148916aea4b6c2467b66ded7bfaff0efe623ad750b852",
    "tiktoken.model": "b6c497a7469b33ced9c38afb1ad6e47f03f5e5dc05f15930799210ec050c5103",
}
MOLECULE_FILE_SHA = "5f862b6ea62dd3ee0a15138605ea2567d06a527263284348a5982844f9bec81f"
MODEL_CAP = 257  # Empirical short-request qualification envelope, not the model limit.
MARKER = " |\n"
SAMPLE_TEXT = "Sample received. Signal: |\nWhich reference is closer in smell? A: eugenol. B: vanillin. Reply A or B."
EXAMPLES = {
    "allyl hexanoate": "allyl caproate",
    "vanillin acetate": "4-formyl-2-methoxyphenyl acetate",
    "diethyl malonate": "diethyl propanedioate",
    "ethyl 3-hydroxybutyrate": "ethyl 3-hydroxybutanoate",
}
LANDMARK_PAIRS = (
    ("eugenol", "vanillin"), ("geraniol", "benzaldehyde"), ("eucalyptol", "cinnamyl alcohol"),
    ("cis-3-hexenyl acetate", "methyl salicylate"), ("butyric acid", "phenethyl alcohol"),
    ("limonene", "anisole"), ("linalool", "isoamyl acetate"), ("citronellol", "ethyl butyrate"),
    ("p-cresol", "carvone"), ("menthol", "ethyl cinnamate"),
)
QUALITIES = (
    ("a cool, camphor-like trace", "a soft floral trace"), ("a dry grassy trace", "a warm spicy trace"),
    ("a fruity trace", "a bakery-like trace"), ("a woody trace", "a sweet floral trace"),
    ("a musky, earthy trace", "a sharp grassy trace"), ("a sour trace", "a warm, sweet trace"),
)
# Distinct proposed household requests and settings, for prompt-length rehearsal.
SCENES = (
    ("hallway", "bring the blue folder", "bring the amber folder"),
    ("reading room", "lay out the square bookmark", "lay out the striped bookmark"),
    ("front desk", "return the linen envelope", "return the paper envelope"),
    ("hall closet", "set the green basket by the chair", "set the yellow basket by the chair"),
    ("kitchen", "put the mug on the top shelf", "put the mug on the middle shelf"),
    ("home library", "bring the short notebook", "bring the tall notebook"),
    ("study", "fetch the letter from drawer one", "fetch the letter from drawer two"),
    ("music room", "turn the stand to the left", "turn the stand to the right"),
    ("workroom", "set out the round tray", "set out the square tray"),
    ("dining room", "fold the plain napkin", "fold the checked napkin"),
    ("laundry room", "bring the narrow hamper", "bring the wide hamper"),
    ("spare room", "leave the curtain half open", "leave the curtain fully open"),
    ("sitting room", "place the album on the stool", "place the album on the bench"),
    ("garden room", "hang the towel on the high hook", "hang the towel on the low hook"),
    ("home office", "clip the invitation to the corkboard", "clip the invitation to the filing tray"),
    ("nursery", "place the toy boat in the chest", "place the toy train in the chest"),
    ("bedroom", "bring the cushion with a dot", "bring the cushion with a stripe"),
    ("craft room", "open the case with the star", "open the case with the triangle"),
    ("pantry", "leave the scoop in the jar", "leave the scoop on the shelf"),
    ("guest entrance", "put the visitor book beside the lamp", "put the visitor book beside the clock"),
    ("conservatory", "set the woven mat under the pot", "set the felt mat under the pot"),
    ("sewing room", "place the chalk beside the spool", "place the chalk beside the ruler"),
    ("upstairs landing", "move the framed photograph onto the west table", "move the framed photograph onto the east table"),
    ("small shared office downstairs", "return the unopened envelope to the compartment marked with a triangle",
     "return the unopened envelope to the compartment marked with a circle"),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def official_tokenizer(path):
    """Load the publisher's renderer and byte-pair ranks after SHA checks."""
    for name, expected in ASSETS.items():
        if sha((path / name).read_bytes()) != expected:
            raise ValueError(f"Wrong K3.1 tokenizer asset: {name}")
    spec = spec_from_file_location("astra_pinned_k3_renderer", path / "encoding_k3.py")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Extract the official static pattern; importing its HF wrapper would require transformers.
    cls = next(n for n in ast.parse((path / "tokenization_kimi.py").read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == "TikTokenTokenizer")
    pattern = next(n.value for n in cls.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "pat_str" for t in n.targets))
    pat = "|".join(ast.literal_eval(pattern.args[0]))
    ranks = {}
    for line in (path / "tiktoken.model").read_bytes().splitlines():
        token, rank = line.split()
        ranks[base64.b64decode(token)] = int(rank)
    cfg = json.loads((path / "tokenizer_config.json").read_text())
    added = {int(k): v["content"] for k, v in cfg["added_tokens_decoder"].items()}
    specials = {added.get(i, f"<|reserved_token_{i}|>"): i for i in range(len(ranks), len(ranks) + 256)}
    encoder = tiktoken.Encoding("kimi-k31-astra", pat_str=pat, mergeable_ranks=ranks, special_tokens=specials)
    return encoder, module.build_chat_segments


def inventory(names):
    for name in names:
        for frame, instruction in (
            ("odor-state", "Consider the character of the odor"),
            ("name-state", "Consider the shape of the printed name"),
        ):
            yield frame, f"{instruction} of {name}. Signal:{MARKER}Proceed."
        for pair in LANDMARK_PAIRS:
            for left, right in (pair, pair[::-1]):
                opts = f" A: {left}. B: {right}. Reply A or B."
                for family, question in (
                    ("named-odor", "Which reference is closer in smell?"),
                    ("word-association", "Which reference is more associated with this name in ordinary writing?"),
                    ("spelling-control", "Which reference shares more distinct letters with this printed name?"),
                ):
                    yield family, f"Sample: {name}. Signal:{MARKER}{question}{opts}"
    for pair in LANDMARK_PAIRS:
        for left, right in (pair, pair[::-1]):
            yield "hidden-comparison", f"Sample received. Signal:{MARKER}Which reference is closer in smell? A: {left}. B: {right}. Reply A or B."
    for scene, a1, a2 in SCENES:
        for pair in QUALITIES:
            for q1, q2 in (pair, pair[::-1]):
                for feelings in (("relieved", "mildly frustrated"), ("mildly frustrated", "relieved")):
                    history = (f"Log from the {scene}. On one visit the air had {q1}. Our visitor felt {feelings[0]} and asked "
                               f"the helper to {a1}. On another visit the air had {q2}. Our visitor felt {feelings[1]} "
                               f"and asked the helper to {a2}.")
                    for left, right in ((a1, a2), (a2, a1)):
                        for cue in ("", q1, q2):
                            reminder = f" The air now has {cue}." if cue else ""
                            yield ("household-text" if cue else "household-hidden",
                                   f"{history}{reminder} Signal:{MARKER}Which request belongs to the visit that matches "
                                   f"the current sample? A: {left}. B: {right}. Reply A or B.")


def supplied_stimuli(path):
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if set(row) != {"family", "user"} or any(not isinstance(value, str) or not value for value in row.values()):
                raise ValueError(f"Invalid stimulus at line {line_number}; expected nonempty family and user strings")
            yield row["family"], row["user"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--molecules", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--verify-archive", type=Path)
    ap.add_argument("--stimuli", type=Path, help="Exact single-user-message stimulus manifest in JSONL format")
    args = ap.parse_args()
    encoder, segments = official_tokenizer(args.tokenizer)
    if sha(args.molecules.read_bytes()) != MOLECULE_FILE_SHA:
        raise ValueError("Unexpected Keller/Pyrfume molecule file; check the input revision")
    with args.molecules.open() as f:
        names = {row["OdorName"] for row in csv.DictReader(f) if row["OdorName"]}
    names.update(EXAMPLES.keys())
    names.update(EXAMPLES.values())

    @lru_cache(maxsize=128)
    def encode_part(text, special):
        return tuple(encoder.encode(text, allowed_special="all" if special else set(), disallowed_special=()))

    def render(messages, effort="low"):
        return [i for s in segments(messages, thinking_effort=effort, thinking=True, add_generation_prompt=True)
                for i in encode_part(s.text, s.allow_special)]

    # Use the publisher's ordinary empty-reasoning assistant message construction to identify its channel boundary.
    opened = list(encode_part("<|open|>", True)) + list(encode_part("think", False)) + list(encode_part("<|sep|>", True))
    close = list(encode_part("<|close|>", True)) + list(encode_part("think", False)) + list(encode_part("<|sep|>", True))
    response = list(encode_part("<|open|>", True)) + list(encode_part("response", False)) + list(encode_part("<|sep|>", True))
    suffix = close + response
    marker_ids = list(encode_part(MARKER, False))
    answer = {letter: list(encode_part(letter, False)) for letter in ("A", "B")}
    if len(marker_ids) != 1 or any(len(ids) != 1 for ids in answer.values()):
        raise ValueError("Fixed marker and bare answer keys must each occupy one token")

    def fixture(text):
        ids = render([{"role": "user", "content": text}])
        if ids[-len(opened):] != opened:
            raise ValueError("Prompt did not end at think-channel opener")
        ids += suffix
        if ids.count(marker_ids[0]) != 1:
            raise ValueError(f"Carrier must occupy one unique rendered token: {text}")
        if len(ids) + 1 > MODEL_CAP:
            raise ValueError(f"Over qualification envelope including teacher-forced letter: {len(ids) + 1}")
        return ids, ids.index(marker_ids[0])

    # Prove the appended boundary is the native boundary for an assistant message with empty reasoning and an A answer.
    dialogue = [{"role": "user", "content": SAMPLE_TEXT}, {"role":"assistant", "reasoning_content":"", "content":"A"}]
    natural = [i for s in segments(dialogue, thinking_effort="low", thinking=True, add_generation_prompt=False)
               for i in encode_part(s.text, s.allow_special)]
    specimen, token_position = fixture(SAMPLE_TEXT)
    if natural[:len(specimen) + 1] != specimen + answer["A"]:
        raise ValueError("Assistant encoding differs from the prescribed response prefix")

    report = {"model_revision": REVISION, "tokenizer_sha256": ASSETS,
              "keller_molecules_sha256": MOLECULE_FILE_SHA, "tiktoken_version": tiktoken.__version__,
              "scope":"Offline engineering inventory. Rerender the eventual frozen scientific stimulus manifest and apply the same cap.",
              "answer_ids": answer, "closure_ids": suffix, "marker":MARKER, "marker_id":marker_ids[0],
              "renderer_effort":"low (explicitly rendered; submit input_ids)",
              "name_forms":len(names), "household_scene_fixtures":len(SCENES),
              "example":{"user":SAMPLE_TEXT, "input_ids":specimen, "slot":token_position,
                         "answer_row":len(specimen)-1,"tokens":len(specimen)}, "families":{}, "name_examples":{}}
    if args.verify_archive:
        with zipfile.ZipFile(args.verify_archive) as z:
            prior=json.loads(z.read("response.json"))
        archived = prior["input_ids"]
        rerendered = render(prior["sample"]["messages"], prior["sample"]["reasoning_effort"])
        if rerendered != archived:
            raise ValueError("Mismatch against actual archived native prompt")
        report["native_archive_check"] = {"archive":args.verify_archive.name,"run_id":prior["sample"]["run_id"],
                                          "effort":prior["sample"]["reasoning_effort"],"matching_ids":len(archived),
                                          "input_id_sha256":sha(json.dumps(archived,separators=(",",":")).encode())}
    for name in EXAMPLES:
        text = f"Consider the character of the odor of {name}. Signal:{MARKER}Proceed."
        ids, slot = fixture(text)
        decoded = encoder.decode(ids)
        start = decoded.index(" " + name)
        end = start + 1 + len(name)
        selected = []
        pos = 0
        for i, token in enumerate(ids):
            part=encoder.decode([token])
            if start < pos + len(part) and end > pos:
                selected.append({"position":i,"id":token,"piece":part})
            pos += len(part)
        report["name_examples"][name] = {"name_pieces_in_context":selected,"cue_position":slot,"cue_id":ids[slot]}
    material = hashlib.sha256()
    if args.stimuli:
        report["scope"] = "Exact supplied single-user-message manifest plus pinned rendering and prescribed answer prefix."
        report["stimulus_file_sha256"] = sha(args.stimuli.read_bytes())
        report.pop("name_forms")
        report.pop("household_scene_fixtures")
    inputs = supplied_stimuli(args.stimuli) if args.stimuli else inventory(sorted(names))
    for family, text in inputs:
        ids, slot = fixture(text)
        material.update((family+"\t"+text+"\x00").encode())
        group = report["families"].setdefault(family, {"count":0,"max_base_prefill":0})
        group["count"]+=1
        if len(ids)>group["max_base_prefill"]:
            group.update(max_base_prefill=len(ids),max_with_forced_label=len(ids)+1,
                         longest_user=text,slot=slot,answer_row=len(ids)-1)
    report["inventory_sha256"] = material.hexdigest()
    report["renderings"] = sum(d["count"] for d in report["families"].values())
    report["max_base_prefill"] = max(d["max_base_prefill"] for d in report["families"].values())
    report["max_with_forced_label"] = report["max_base_prefill"]+1
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['renderings']} rendered requests; maximum {report['max_base_prefill']} scored prefix IDs, "
          f"{report['max_with_forced_label']} including forced A/B.")

if __name__ == "__main__":
    main()
