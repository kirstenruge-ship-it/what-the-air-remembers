# Rebuilding the frozen stimulus manifest

The Kimi study generator is in research/stimuli.py. It imports only the pinned tokenizer utility, marker, scene fixtures and model-revision metadata it needs from astra-rehearsal/prompts.py. The earlier bounded-answer software examples inside that utility are not research model calls. The trial generator makes no model request. It renders with maximum reasoning effort. Every model-facing input and seed is also provided directly in the versioned inputs.

1. Obtain behavior.csv, stimuli.csv, and molecules.csv from the exact Pyrfume Git commit in ../stimuli/source/provenance.json. Verify their SHA-256 values.
2. Uncompress the two PubChem JSON snapshots in this directory. The registered hashes apply to the uncompressed bytes.
3. Obtain encoding_k3.py, tokenization_kimi.py, tokenizer_config.json, and tiktoken.model from the revision of moonshotai/Kimi-K3.1-Pro identified in ../stimuli/manifest.json. The loader verifies the asset hashes.
4. Use Python 3.11 with NumPy, pandas, RDKit, networkx, and tiktoken. Environment versions appear in the manifest. Run the generator with --panel, --pubchem-properties, --pubchem-synonyms, --tokenizer, and --out, pointing to those local resources.
5. Compare trials.jsonl with SHA-256 4516192c45ff56feaeef616ef3f798ae5e1c280cc3dc9906a66bc23476499631. Earlier inputs and explanations for the two pre-inference amendments are in ../stimuli/freeze-history.
