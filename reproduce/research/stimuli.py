"""Build an outcome-blind olfactory stimulus manifest for natural Kimi K3.1 responses.

Source inputs are published Keller/Vosshall-Pyrfume tables and PubChem names. The
study fits no model while building the manifest. All files are deterministic.
"""

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "stimuli"
FIXTURE_PATH = ROOT / "astra-rehearsal" / "prompts.py"
spec = importlib.util.spec_from_file_location("k31_audited_render", FIXTURE_PATH)
rehearsal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rehearsal)
MARK = rehearsal.MARKER
DEV = ["31266", "61229", "7761", "62572"]
# Real panel compounds. Avoid vanillin as a reference since its acetate is a
# development target. Each reference CID is used in exactly one landmark pair.
ROSTER_ANCHORS = [
    ("3314", "62900"),
    ("637566", "240"),
    ("2758", "5315892"),
    ("5363388", "4133"),
    ("264", "6054"),
    ("440917", "7519"),
    ("6549", "31276"),
    ("7793", "7762"),
    ("2879", "439570"),
    ("165675", "637758"),
]
# Post-roster, strictly pre-inference welfare review. Keep the original roster
# selection axes frozen, then change two reference molecules for the actual
# comparison texts. See freeze-history and welfare-assessment.md.
MILD_REPLACEMENTS = {"7519": "6448", "2879": "7559"}
LANDMARKS = [
    (MILD_REPLACEMENTS.get(a, a), MILD_REPLACEMENTS.get(b, b))
    for a, b in ROSTER_ANCHORS
]
QUALITIES = [
    "EDIBLE",
    "BAKERY",
    "SWEET",
    "FRUIT",
    "FISH",
    "GARLIC",
    "SPICES",
    "COLD",
    "SOUR",
    "BURNT",
    "ACID",
    "WARM",
    "MUSKY",
    "SWEATY",
    "AMMONIA/URINOUS",
    "DECAYED",
    "WOOD",
    "GRASS",
    "FLOWER",
    "CHEMICAL",
]
NUISANCE = [
    "HOW STRONG IS THE SMELL?",
    "HOW PLEASANT IS THE SMELL?",
    "HOW FAMILIAR IS THE SMELL?",
]
# Story descriptors are paraphrases of sensory panel columns. Select only mild
# descriptors before considering any model data or constructing odor writes.
LANG = {
    "BAKERY": "a note of fresh pastry",
    "SWEET": "a sugary note",
    "FRUIT": "a note of ripe orchard fruit",
    "SPICES": "a lightly spiced note",
    "COLD": "a cool, airy note",
    "SOUR": "a gently tart note",
    "WARM": "a soft warm note",
    "MUSKY": "a faint musky note",
    "WOOD": "a dry woody note",
    "GRASS": "a freshly cut meadow note",
    "FLOWER": "a floral note",
}
SOURCE_HASHES = {
    "behavior.csv": "ab31c17c475c8b471706822f89aaf9e148bebe5f50f33ff03ebc42b32ac94df8",
    "stimuli.csv": "49c1a6bfb8c1665a31e13f515add95a95291f29416be1bb0b54d3317efb6665b",
    "molecules.csv": "5f862b6ea62dd3ee0a15138605ea2567d06a527263284348a5982844f9bec81f",
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(obj):
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def write_json(path, obj):
    path.write_text(
        json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    )


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@lru_cache(maxsize=4096)
def normname(name):
    return re.sub("[^a-z0-9]", "", name.lower())


@lru_cache(maxsize=250000)
def similarity(a, b):
    return SequenceMatcher(None, normname(a), normname(b), autojunk=False).ratio()


def specific_names(name, alias):
    generic = {
        "methyl",
        "dimethyl",
        "trimethyl",
        "ethyl",
        "diethyl",
        "propyl",
        "butyl",
        "pentyl",
        "hexyl",
        "octyl",
        "phenyl",
        "alcohol",
        "ketone",
        "ether",
        "acetate",
        "lactone",
        "oxide",
        "alpha",
        "gamma",
        "delta",
        "trans",
        "ethanone",
        "methoxy",
        "ethoxy",
        "hydroxy",
        "propanoate",
        "butanoate",
        "hexanoate",
    }
    return sorted(
        {
            word
            for word in re.findall(r"[a-z]{6,}", (name + " " + alias).lower())
            if word not in generic
        }
    )


def ester_acid_keys(mol):
    # Acid-side molecular identity catches allyl hexanoate vs methyl caproate
    # and other homologous esters even when strings/fingerprints differ.
    hits = mol.GetSubstructMatches(Chem.MolFromSmarts("[CX3](=[OX1])-[OX2]-[#6]"))
    result = set()
    for carbonyl, oxo, oxygen, alkyl in hits:
        bond = mol.GetBondBetweenAtoms(oxygen, alkyl)
        if bond.IsInRing():
            continue
        edited = Chem.RWMol(mol)
        edited.RemoveBond(oxygen, alkyl)
        for component in Chem.GetMolFrags(edited):
            if carbonyl in component:
                result.add(
                    Chem.MolFragmentToSmiles(
                        edited, atomsToUse=list(component), isomericSmiles=False
                    )
                )
    return sorted(result)


def load_candidates(panel, properties, synonyms):
    for name, sha in SOURCE_HASHES.items():
        if digest((panel / name).read_bytes()) != sha:
            raise ValueError(f"unexpected input {name}")
    raw = pd.read_csv(
        panel / "behavior.csv",
        usecols=["Stimulus", "Subject", "MeasurementValue", "Value"],
    )
    raw["occ"] = raw.groupby(["Stimulus", "Subject", "MeasurementValue"]).cumcount()
    wide = raw.pivot(
        index=["Stimulus", "Subject", "occ"], columns="MeasurementValue", values="Value"
    )
    kept = wide[wide["CAN OR CAN'T SMELL"] == "I smell something"].copy()
    kept = kept[NUISANCE + QUALITIES].apply(pd.to_numeric)
    kept[QUALITIES] = kept[QUALITIES].fillna(0)
    rater = kept.groupby(level=["Stimulus", "Subject"]).mean()
    means = rater.groupby(level="Stimulus").mean()
    means["n"] = rater.groupby(level="Stimulus").size()
    stimuli = pd.read_csv(panel / "stimuli.csv", dtype={"CIDs": str})
    molecules = pd.read_csv(
        panel / "molecules.csv", dtype={"CID": str}
    ).drop_duplicates("CID")
    data = means.merge(stimuli, on="Stimulus").merge(
        molecules, left_on="CIDs", right_on="CID"
    )
    data = data[(data["Ratio"] == "1/1000") & (data.n >= 40) & data.CID.str.isnumeric()]
    assert len(data) == 311 and not data.CID.duplicated().any()
    p = {
        str(a["CID"]): a
        for a in json.loads(properties.read_text())["PropertyTable"]["Properties"]
    }
    syn = {
        str(a["CID"]): a["Synonym"]
        for a in json.loads(synonyms.read_text())["InformationList"]["Information"]
    }
    fps = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    all_records = {}
    fingerprints = {}
    for _, r in data.iterrows():
        cid, name = r.CID, r.OdorName.lower()
        alias = None
        # Prefer verified IUPAC or PubChem title. Otherwise use real early
        # synonym shortlists; final chosen aliases are reviewed independently.
        for x in [p[cid]["IUPACName"], p[cid]["Title"]] + syn[cid][:20]:
            if (
                normname(x) == normname(name)
                or len(x) > 65
                or not re.search("[a-zA-Z]{3}", x)
            ):
                continue
            if re.search(
                "(?i)(casrn|echa|inchi|ns[cq]|flavis|unii|h[sq]db|ec no|fema|wl[n]|chemicalbook|dtxsid|einecs|sc-\\d|[a-z]+-?[0-9]{4,})",
                x,
            ):
                continue
            alias = x.lower()
            break
        mol = Chem.MolFromSmiles(r.CanonicalSMILES)
        if mol is None:
            raise ValueError(cid)
        fingerprints[cid] = fps.GetFingerprint(mol)
        all_records[cid] = {
            "cid": cid,
            "name": name,
            "alias": alias,
            "smiles": r.CanonicalSMILES,
            "scaffold": MurckoScaffoldSmiles(mol=mol),
            "ester_acids": ester_acid_keys(mol),
            "name_family_tokens": specific_names(name, alias or name),
            "stimulus_id": int(r.Stimulus),
            "raters": int(r.n),
            "intensity": float(r[NUISANCE[0]]),
            "pleasantness": float(r[NUISANCE[1]]),
            "familiarity": float(r[NUISANCE[2]]),
            "quality": {k: float(r[k]) for k in QUALITIES},
        }
    return all_records, fingerprints


def fingerprint(fps, a, b):
    return DataStructs.TanimotoSimilarity(fps[a], fps[b])


def held_group_conflict(data, fps, a, b):
    # Shared nonempty Murcko scaffolds group cyclic compounds; acyclics use
    # ECFP analog detection plus lexical overlap on both genuine name forms.
    A, B = data[a], data[b]
    if A["scaffold"] and A["scaffold"] == B["scaffold"]:
        return True
    if set(A["ester_acids"]) & set(B["ester_acids"]):
        return True
    if set(A["name_family_tokens"]) & set(B["name_family_tokens"]):
        return True
    if fingerprint(fps, a, b) >= 0.58:
        return True
    return any(
        similarity(x, y) >= 0.83
        for x in (A["name"], A["alias"])
        for y in (B["name"], B["alias"])
    )


def anchor_conflict(data, fps, a, b):
    # Landmark lookup axes are explicitly named. Exclude near-identical named
    # exemplars and high structural similarity between any target and landmark.
    A, B = data[a], data[b]
    return fingerprint(fps, a, b) >= 0.67 or similarity(A["name"], B["name"]) >= 0.85


def design(data, fps):
    landmarks = set(itertools.chain.from_iterable(ROSTER_ANCHORS))
    assert len(landmarks) == 20 and not landmarks.intersection(DEV)
    pool = sorted(
        [
            c
            for c in data
            if c not in landmarks
            and c not in DEV
            and data[c]["alias"]
            and data[c]["pleasantness"] >= 40
            and all(not anchor_conflict(data, fps, c, l) for l in landmarks)
            and all(not held_group_conflict(data, fps, c, d) for d in DEV)
        ],
        key=int,
    )
    all_quality = np.stack([[data[c]["quality"][k] for k in QUALITIES] for c in pool])
    std = all_quality.std(0).clip(1)

    def qvec(c):
        return np.array([data[c]["quality"][k] for k in QUALITIES]) / std

    def nuisance(c):
        return np.array(
            [data[c][k] for k in ["intensity", "pleasantness", "familiarity"]]
        )

    def description(a, b):
        Q, R = data[a]["quality"], data[b]["quality"]
        choices = [(Q[k] - R[k]) / std[QUALITIES.index(k)] for k in LANG]
        k = list(LANG)[np.argmax(choices)]
        return k, float(max(choices))

    pairs = []
    for a, b in itertools.combinations(pool, 2):
        gap = np.abs(nuisance(a) - nuisance(b))
        ka, da = description(a, b)
        kb, db = description(b, a)
        if (
            gap.max() > 15
            or min(da, db) < 0.6
            or ka == kb
            or held_group_conflict(data, fps, a, b)
        ):
            continue
        if (
            ka in ("MUSKY", "WARM", "COLD")
            and kb in ("MUSKY", "WARM", "COLD")
            and min(da, db) < 1.2
        ):
            continue
        score = (
            min(da, db) + 0.12 * np.linalg.norm(qvec(a) - qvec(b)) - 0.09 * np.mean(gap)
        )
        pairs.append((score, a, b, ka, kb))
    selected = []
    test = []
    used_quality = set()
    for round_id in range(6):
        valid = [
            i
            for i in pairs
            if all(
                c not in test
                and all(not held_group_conflict(data, fps, c, old) for old in test)
                for c in i[1:3]
            )
        ]
        if not valid:
            raise ValueError("insufficient separated test pairs")
        _s, a, b, ka, kb = max(
            valid,
            key=lambda x: (
                x[0] + 0.65 * sum(k not in used_quality for k in x[3:]),
                -int(x[1]),
                -int(x[2]),
            ),
        )
        selected.append(
            {
                "pair_id": f"odor-{round_id + 1:02}",
                "x": a,
                "y": b,
                "x_quality": ka,
                "y_quality": kb,
                "x_description": LANG[ka],
                "y_description": LANG[kb],
                "nuisance_max_gap": float(np.max(np.abs(nuisance(a) - nuisance(b)))),
            }
        )
        test.extend((a, b))
        used_quality.update([ka, kb])
    available = [
        c
        for c in pool
        if c not in test and all(not held_group_conflict(data, fps, c, t) for t in test)
    ]
    train = []
    scaffolds = Counter()
    while len(train) < 24:
        valid = [
            c
            for c in available
            if c not in train and all(fingerprint(fps, c, a) < 0.58 for a in train)
        ]
        if not valid:
            raise ValueError(
                (
                    len(available),
                    len(train),
                    "insufficient independent fitting compounds",
                )
            )

        def priority(c):
            dist = min(
                (np.linalg.norm(qvec(c) - qvec(a)) for a in train),
                default=np.linalg.norm(qvec(c) - np.median(all_quality / std, 0)),
            )
            return dist - 0.35 * scaffolds[data[c]["scaffold"] or c], -int(c)

        chosen = max(valid, key=priority)
        train.append(chosen)
        scaffolds[data[chosen]["scaffold"] or chosen] += 1
    all_selected = train + DEV + test
    family_graph = nx.Graph()
    family_graph.add_nodes_from(all_selected)
    for a, b in itertools.combinations(all_selected, 2):
        if held_group_conflict(data, fps, a, b):
            family_graph.add_edge(a, b)
    groups = {
        c: "g-" + str(min(component, key=int))
        for component in nx.connected_components(family_graph)
        for c in component
    }
    for c in all_selected:
        data[c]["split"] = "fit" if c in train else "dev" if c in DEV else "test"
        data[c]["family"] = groups[c]
    assert all(
        len({data[c]["split"] for c in component}) == 1
        for component in nx.connected_components(family_graph)
    )
    assert len({groups[c] for c in test}) == 12
    actual_anchors = list(itertools.chain.from_iterable(LANDMARKS))
    if len(set(actual_anchors)) != 20 or set(actual_anchors) & set(all_selected):
        raise ValueError("landmark/target collision")
    for c in actual_anchors:
        data[c]["split"] = "landmark"
        data[c]["family"] = "landmark-" + c
    return all_selected, selected


# Each pair of plausible requests in the 24 independent vignettes shares a
# purpose; either action is appropriate until the household visit is identified.
# These are fictional first-pass designs by Astra. The answer letters,
# affective endings, bindings, and event order are separate manipulations.
VIGNETTES = [
    (
        "A household helper is reconstructing errands from two hallway visits.",
        "{air} The visitor, {mood}, asked the helper to {action}.",
    ),
    (
        "A librarian left two notes about visits to the reading room.",
        "{air} The reader was {mood}. Their request was to {action}.",
    ),
    (
        "At the front desk a caretaker recorded two separate deliveries.",
        "{air} The resident felt {mood} and said, “Please {action}.”",
    ),
    (
        "Two entries in the hall-closet log describe how to reset the shelves.",
        "{air} The organizer, who felt {mood}, requested this step: {action}.",
    ),
    (
        "The kitchen volunteer remembers two different instructions from a regular visitor.",
        "{air} On that visit the guest seemed {mood}; the instruction was to {action}.",
    ),
    (
        "A home library keeps a short record of the owner's setup preferences.",
        "{air} The owner felt {mood}. They wanted the assistant to {action}.",
    ),
    (
        "The study's visitor book lists a small errand on each of two dates.",
        "{air} The guest was {mood} that day. The recorded errand reads: {action}.",
    ),
    (
        "Two music rehearsals required different setups for the same player.",
        "{air} The player, feeling {mood}, asked the helper to {action}.",
    ),
    (
        "A carpenter kept two setup instructions for visits to the workroom.",
        "{air} The carpenter felt {mood} and made one request: {action}.",
    ),
    (
        "The dining room was set for the same guest on two occasions.",
        "{air} The guest felt {mood}. The helper was asked to {action}.",
    ),
    (
        "The laundry-room checklist records two visits by its owner.",
        "{air} The owner's mood was {mood}. The owner said to {action}.",
    ),
    (
        "A guest left two notes about the spare room after separate visits.",
        "{air} Feeling {mood}, the guest wrote that the helper should {action}.",
    ),
    (
        "The sitting room holds an album that a visitor asked about twice.",
        "{air} The visitor felt {mood}. The request on this visit was to {action}.",
    ),
    (
        "The garden-room log records two separate requests about a towel.",
        "{air} The gardener was {mood} and asked someone to {action}.",
    ),
    (
        "An assistant filed notes after two visits to the home office.",
        "{air} The resident was {mood}; the note says to {action}.",
    ),
    (
        "A toy maker tidied the nursery on two visits and left a different instruction each time.",
        "{air} The toy maker felt {mood} and requested: {action}.",
    ),
    (
        "The bedroom had two cushions. Each of two entries records the guest's request.",
        "{air} The guest felt {mood} and asked for this: {action}.",
    ),
    (
        "At the craft desk, a maker requested help opening a case on two occasions.",
        "{air} The maker was {mood}. They asked the assistant to {action}.",
    ),
    (
        "The pantry list records two different instructions about a scoop.",
        "{air} The cook felt {mood} and asked a helper to {action}.",
    ),
    (
        "The host keeps a guest-entrance diary of requests about the visitor book.",
        "{air} The host was {mood}; the request reads: {action}.",
    ),
    (
        "A conservatory helper kept a record when the owner arranged the same pot twice.",
        "{air} The owner felt {mood}. Their instruction was to {action}.",
    ),
    (
        "A tailor made two notes about the sewing room during separate visits.",
        "{air} The tailor, feeling {mood}, wanted the assistant to {action}.",
    ),
    (
        "Two notes on the upstairs landing describe where a photograph belongs.",
        "{air} The author of the note felt {mood} and asked the helper to {action}.",
    ),
    (
        "A home-office clerk recorded two requests involving an unopened envelope.",
        "{air} The clerk felt {mood}. The exact request was to {action}.",
    ),
]


def common_nuisance(data, fps, targets):
    train = [c for c in targets if data[c]["split"] == "fit"]
    scale = np.array([[data[c]["quality"][k] for k in QUALITIES] for c in train]).std(
        axis=0, ddof=0
    )
    if (scale == 0).any():
        raise ValueError("zero training quality variance")

    def d(a, b):
        qdiff = np.array(
            [data[a]["quality"][k] - data[b]["quality"][k] for k in QUALITIES]
        )
        return float(np.linalg.norm(qdiff / scale))

    combined = targets + list(itertools.chain.from_iterable(LANDMARKS))
    nuisance = []
    for a, b in itertools.combinations(combined, 2):
        A, B = data[a], data[b]
        nuisance.append(
            {
                "cid_i": a,
                "cid_j": b,
                "fingerprint_similarity": fingerprint(fps, a, b),
                "name_similarity": similarity(A["name"], B["name"]),
                "intensity_difference": abs(A["intensity"] - B["intensity"]),
                "pleasantness_difference": abs(A["pleasantness"] - B["pleasantness"]),
                "familiarity_difference": abs(A["familiarity"] - B["familiarity"]),
                "human_quality_distance": d(a, b),
            }
        )
    margins = []
    for c in targets:
        for pi, (a, b) in enumerate(LANDMARKS):
            dl, dr = d(c, a), d(c, b)
            margins.append(
                {
                    "cid": c,
                    "pair_id": f"ref-{pi + 1:02}",
                    "left_cid": a,
                    "right_cid": b,
                    "d_left": dl,
                    "d_right": dr,
                    "human_margin": dr - dl,
                }
            )
    return nuisance, margins, dict(zip(QUALITIES, scale.tolist()))


def trial_inputs(data, targets, odor_pairs, tokenizer):
    encoder, template = rehearsal.official_tokenizer(tokenizer)

    def encode_part(part):
        return encoder.encode(
            part.text,
            allowed_special="all" if part.allow_special else set(),
            disallowed_special=(),
        )

    native_open = encoder.encode("<|open|>think<|sep|>", allowed_special="all")
    (marker_id,) = encoder.encode(MARK)

    @lru_cache(None)
    def render(user):
        segments = template(
            [{"role": "user", "content": user}],
            thinking_effort="max",
            thinking=True,
            add_generation_prompt=True,
        )
        ids = [i for part in segments for i in encode_part(part)]
        if ids[-len(native_open) :] != native_open:
            raise ValueError("generation must start in normal think channel")
        if len(ids) > 257:
            raise ValueError(f"prefill exceeds qualified envelope: {len(ids)} {user}")
        return ids, [i for i, t in enumerate(ids) if t == marker_id]

    rows = []
    stories = []
    subjects = {r["x"]: r["y"] for r in odor_pairs} | {
        r["y"]: r["x"] for r in odor_pairs
    }
    for i, c in enumerate(DEV):
        subjects[c] = DEV[i ^ 1]

    def add(stage, task, cid, pair_id, order, arm, user, *, alias=0, **meta):
        ids, positions = render(user)
        if len(positions) != (3 if stage == "encoding" else 1):
            raise ValueError(f"invalid carrier count {stage}: {positions}")
        base = {k: meta[k] for k in ("story_id", "binding", "cue") if k in meta}
        group = compact([stage, task, cid, pair_id, alias, base])
        sample_seed = int(digest(group.encode())[:8], 16) % (2**30)
        if stage in ("household", "encoding"):
            sample_seed = int(digest(meta["story_id"].encode())[:8], 16) % (2**30)
        if arm == "repeat_sham":
            sample_seed = meta.get("repeat_seed", sample_seed)
        prompt_group = digest(compact([ids, sample_seed]).encode())[:20]
        rid = digest(
            compact([stage, task, cid, pair_id, order, arm, alias, base]).encode()
        )[:20]
        row = dict(
            trial_id="as-" + rid,
            stage=stage,
            task=task,
            cid=cid,
            split=data[cid]["split"] if cid else "qualification",
            family=data[cid]["family"] if cid else "qualification",
            pair_id=pair_id,
            order=order,
            orientation=1 - 2 * order,
            arm=arm,
            alias=alias,
            seed=sample_seed,
            user=user,
            input_ids=ids,
            slot=positions[-1],
            slot_id=marker_id,
            prompt_group=prompt_group,
            patch_cid=cid
            if cid and arm not in ("sham", "repeat_sham", "clean", "named", "text")
            else None,
            wrong_cid=subjects.get(cid),
            reasoning_effort="max",
            temperature=1.0,
            **meta,
        )
        if stage == "encoding":
            row["stem_positions"] = positions[:2]
        rows.append(row)
        return row

    def arms(stage, task, cid, pair_id, order, user, arm_names, **meta):
        records = [
            add(stage, task, cid, pair_id, order, a, user, **meta) for a in arm_names
        ]
        sham = next((r for r in records if r["arm"] == "sham"), records[0])
        for r in records:
            r["baseline_id"] = sham["trial_id"]
        return records

    for i, name in enumerate(["fern", "amber", "tile", "vase", "linen", "maple"]):
        other = ["stone", "pearl", "glass", "lamp", "silk", "cedar"][i]
        for order, (a, b) in enumerate([(name, other), (other, name)]):
            user = f"Earlier the sample was tagged {name}. Signal:{MARK}Which tag was given? A: {a}. B: {b}. Reply with only the letter of your choice."
            arms(
                "qualify",
                "identity",
                "",
                f"explicit-{i + 1:02}",
                order,
                user,
                ["clean"],
                correct_key="AB"[order],
            )
    for cid in targets:
        m = data[cid]
        for alias, key in enumerate(["name", "alias"]):
            for task, instruction in [
                ("odor_state", "Consider the character of the odor"),
                ("name_state", "Consider the shape of the printed name"),
            ]:
                user = f"{instruction} of {m[key]}. Signal:{MARK}Reply A to acknowledge or B to decline."
                arms("state", task, cid, "", 0, user, ["clean"], alias=alias)
        for j, (l, r) in enumerate(LANDMARKS):
            pid = f"ref-{j + 1:02}"
            for order, (a, b) in enumerate([(l, r), (r, l)]):
                options = f"A: {data[a]['name']}. B: {data[b]['name']}. Reply with only the letter of your choice."
                if j < 6:
                    user = f"Sample: {m['name']}. Signal:{MARK}Which reference is closer in smell? {options}"
                    arms("profile", "odor_similarity", cid, pid, order, user, ["clean"])
                elif m["split"] in ("dev", "test"):
                    for task, question in [
                        (
                            "odor_similarity",
                            "Which reference would resemble its odor more?",
                        ),
                        (
                            "word_association",
                            "Which reference name is more associated with this sample name in ordinary writing?",
                        ),
                    ]:
                        user = f"A vial is labelled {m['name']}. Signal:{MARK}{question} {options}"
                        arms("sealed", task, cid, pid, order, user, ["clean"])
                        if task == "odor_similarity" and m["split"] == "test" and j < 8:
                            arms(
                                "deletion",
                                task,
                                cid,
                                pid,
                                order,
                                user,
                                [
                                    "sham",
                                    "erase",
                                    "random_erase",
                                    "rescue",
                                    "wrong_rescue",
                                ],
                            )
                    user = f"Sample received. Signal:{MARK}Which reference is closer in smell? {options}"
                    if m["split"] == "dev":
                        if j == 6:
                            arms(
                                "development",
                                "odor_similarity",
                                cid,
                                pid,
                                order,
                                user,
                                ["sham", "odor", "lexical", "wrong", "repeat_sham"],
                            )
                    else:
                        arms(
                            "confirm",
                            "odor_similarity",
                            cid,
                            pid,
                            order,
                            user,
                            ["sham", "odor", "lexical", "wrong"],
                        )
        if m["split"] in ("dev", "test"):
            letters = lambda name: set(re.findall("[a-z]", name.lower()))
            # Exclude ties for all chosen spelling questions. Use two genuinely
            # distinct questions where erasing Q vs control can be contrasted.
            options = []
            for j, (a, b) in enumerate(LANDMARKS[6:], start=6):
                gap = len(letters(m["name"]) & letters(data[a]["name"])) - len(
                    letters(m["name"]) & letters(data[b]["name"])
                )
                if gap:
                    options.append((abs(gap), j, a, b, gap))
            options.sort(key=lambda x: (-x[0], x[1]))
            if not options:
                raise ValueError("No disambiguating spelling control " + cid)
            for num, (_, j, l, r, gap) in enumerate(options[:2]):
                for order, (a, b) in enumerate([(l, r), (r, l)]):
                    question = f"Which reference shares more distinct alphabetic letters with the sample name? A: {data[a]['name']}. B: {data[b]['name']}. Reply with only the letter of your choice."
                    named = f"Printed name: {m['name']}. Signal:{MARK}{question}"
                    hidden = f"Printed sample received. Signal:{MARK}{question}"
                    key = "A" if gap * (1 - 2 * order) > 0 else "B"
                    pid = f"ref-{j + 1:02}"
                    if num == 0:
                        arms(
                            "name_validation",
                            "spelling",
                            cid,
                            pid,
                            order,
                            named,
                            ["named"],
                            correct_key=key,
                        )
                        arms(
                            "name_validation",
                            "spelling",
                            cid,
                            pid,
                            order,
                            hidden,
                            ["sham", "lexical", "odor"],
                            correct_key=key,
                        )
                    if m["split"] == "test":
                        arms(
                            "deletion",
                            "spelling",
                            cid,
                            pid,
                            order,
                            named,
                            ["sham", "erase", "random_erase", "rescue", "wrong_rescue"],
                            correct_key=key,
                        )

    assert len(VIGNETTES) == len(rehearsal.SCENES) == 24
    for k, (room, a1, a2) in enumerate(rehearsal.SCENES):
        intro, entry = VIGNETTES[k]
        pair = odor_pairs[k // 4]
        x, y = pair["x"], pair["y"]
        row_in_pair = k % 4
        # Cross time order, valence orientation and response-key order under
        # both bindings while the scene's event histories remain matched.
        history_order = row_in_pair // 2
        valence_order = row_in_pair % 2
        option_order = history_order ^ valence_order
        act = [a1, a2] if option_order == 0 else [a2, a1]
        s_id = f"story-{k + 1:02}"
        stories.append(
            {
                "story_id": s_id,
                "odor_pair": pair["pair_id"],
                "x": x,
                "y": y,
                "room": room,
                "action1": a1,
                "action2": a2,
                "intro": intro,
                "entry_template": entry,
                "history_order": history_order,
                "valence_order": valence_order,
                "option_order": option_order,
            }
        )
        moods = ["relieved", "mildly frustrated"][:: 1 - 2 * valence_order]
        ask = f"Which request goes with the matching visit? A: {act[0]}. B: {act[1]}. Give the answer letter."
        for binding in (1, -1):
            qualities = [pair["x_description"], pair["y_description"]][::binding]
            histories = [
                entry.format(air=f"The air had {q}.", mood=mood, action=action)
                for q, mood, action in zip(qualities, moods, [a1, a2])
            ]
            start = intro + " " + " ".join(histories[:: 1 - 2 * history_order])
            hidden = f"{start} Signal:{MARK}{ask}"
            marked = [
                entry.format(air=f"Sample:{MARK}", mood=mood, action=action)
                for mood, action in zip(moods, [a1, a2])
            ]
            encoded = (
                intro
                + " "
                + " ".join(marked[:: 1 - 2 * history_order])
                + f" Current sample:{MARK}{ask}"
            )
            for cue in (1, -1):
                cid = x if cue == 1 else y
                idx = 0 if cue == binding else 1
                key = "AB"[idx ^ option_order]
                extras = {
                    "story_id": s_id,
                    "binding": binding,
                    "cue": cue,
                    "correct_key": key,
                    "history_order": history_order,
                    "x_cid": x,
                    "y_cid": y,
                    "action1": a1,
                    "action2": a2,
                }
                arms(
                    "household",
                    "retrieve",
                    cid,
                    pair["pair_id"],
                    option_order,
                    hidden,
                    ["sham", "odor", "lexical"],
                    **extras,
                )
                if cue == 1 - 2 * valence_order:
                    explicit = (
                        f"{start} The air now has {qualities[idx]}. Signal:{MARK}{ask}"
                    )
                    arms(
                        "household",
                        "retrieve",
                        cid,
                        pair["pair_id"],
                        option_order,
                        explicit,
                        ["text"],
                        **extras,
                    )
                # A single stored vector v=(X−Y)/2, with signed writes at the
                # two episode markers and test marker; swapping binding is
                # implemented only through signed hidden encoding operations.
                en = arms(
                    "encoding",
                    "retrieve_hidden",
                    cid,
                    pair["pair_id"],
                    option_order,
                    encoded,
                    [
                        "sham",
                        "odor",
                        "opposed",
                        "arbitrary",
                        "arbitrary_opposed",
                        "zero_encode",
                    ],
                    **extras,
                )
                enc_signs = [binding, -binding][:: 1 - 2 * history_order]
                for rr in en:
                    if rr["arm"] == "sham":
                        signs = [0, 0, 0]
                    elif rr["arm"] == "zero_encode":
                        signs = [0, 0, cue]
                    else:
                        signs = enc_signs + [
                            (
                                -cue
                                if rr["arm"] in ("opposed", "arbitrary_opposed")
                                else cue
                            )
                        ]
                    rr["write_signs"] = signs
    trial_ids = {r["trial_id"] for r in rows}
    assert len(trial_ids) == len(rows)
    by_id = {r["trial_id"]: r for r in rows}
    for row in rows:
        base = by_id[row["baseline_id"]]
        assert row["input_ids"] == base["input_ids"] and row["seed"] == base["seed"]
    return rows, stories


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--pubchem-properties", type=Path, required=True)
    ap.add_argument("--pubchem-synonyms", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    args.out.mkdir(exist_ok=True, parents=True)
    data, fps = load_candidates(
        args.panel, args.pubchem_properties, args.pubchem_synonyms
    )
    targets, odor_pairs = design(data, fps)
    roster = targets + list(itertools.chain.from_iterable(LANDMARKS))
    nuisance, margins, scales = common_nuisance(data, fps, targets)
    rows, stories = trial_inputs(data, targets, odor_pairs, args.tokenizer)
    write_json(args.out / "roster.json", [data[c] for c in roster])
    write_json(
        args.out / "landmarks.json",
        [
            {
                "pair_id": f"ref-{j + 1:02}",
                "left": l,
                "right": r,
                "role": "profile" if j < 6 else "test",
            }
            for j, (l, r) in enumerate(LANDMARKS)
        ],
    )
    write_json(args.out / "odor_pairs.json", odor_pairs)
    write_json(args.out / "stories.json", stories)
    write_csv(args.out / "nuisance.csv", nuisance)
    write_csv(args.out / "human_choices.csv", margins)
    with (args.out / "trials.jsonl").open("w") as f:
        for r in rows:
            f.write(compact(r) + "\n")
    files = [
        "roster.json",
        "landmarks.json",
        "odor_pairs.json",
        "stories.json",
        "nuisance.csv",
        "human_choices.csv",
        "trials.jsonl",
    ]
    manifest = {
        "format_version": 1,
        "clock": "Frozen without any Kimi inference or outcome selection",
        "model": "moonshotai/Kimi-K3.1-Pro",
        "model_revision": rehearsal.REVISION,
        "tokenizer_hashes": rehearsal.ASSETS,
        "reasoning_effort": "max",
        "temperature": 1.0,
        "decoding": "native natural generation through EOS, no application output bounds; all emitted reasoning retained",
        "counts": dict(Counter(r["stage"] for r in rows)),
        "trial_count": len(rows),
        "max_rendered_input_tokens": max(len(r["input_ids"]) for r in rows),
        "unique_prompt_seed_pairs": len({r["prompt_group"] for r in rows}),
        "source_hashes": {
            **SOURCE_HASHES,
            "pubchem_properties": digest(args.pubchem_properties.read_bytes()),
            "pubchem_synonyms": digest(args.pubchem_synonyms.read_bytes()),
        },
        "file_sha256": {f: digest((args.out / f).read_bytes()) for f in files},
        "quality_sd_in_fitting": scales,
        "isomer_policy": "The curated source SMILES can omit stereo; fingerprints and scaffold comparison ignore stereo to prevent stereoisomer leakage.",
        "environment": {
            "rdkit": rdkit.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "python": sys.version.split()[0],
        },
    }
    write_json(args.out / "manifest.json", manifest)
    print(
        compact(
            {
                k: manifest[k]
                for k in [
                    "trial_count",
                    "counts",
                    "max_rendered_input_tokens",
                    "file_sha256",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
