# What the air remembers

**Status: study registered; prospective Kimi inference has not begun at this snapshot.**

Kirsten Ruge’s independent Hive Fidelity research asks whether a hidden cue synthesized from Kimi K3.1 Pro’s own odor-similarity judgments can recover decisions about an unnamed sample and retrieve an arbitrary detail from a recent household episode. The fixed checkpoint is `moonshotai/Kimi-K3.1-Pro` at `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`. Study execution uses maximum reasoning effort, temperature 1, and natural generation without an application output-token cap.

The [scientific registration](protocol/natural-response-registration.md) defines the revised natural-response endpoint. The [scientific proposal sections](protocol/design.md) contain the original rationale and contrasts. The natural-generation requirements and frozen trial inventory supersede software-only examples or conditional sizing in that earlier document.

- [Frozen inputs and checksums](stimuli/manifest.json) cover 24 fitting, four development and 12 held-out compounds, 20 candidate reference odors and 24 household narratives. The 2,852 planned records include conditional extensions. [Every planned input](stimuli/trials.jsonl) includes the text and exact rendered token IDs.
- [Human-source provenance](stimuli/source/provenance.json) and the [selected human measurements](stimuli/source/selected-human-panel.csv.gz) document the Keller–Vosshall panel reanalysis.
- [Earlier Costco donor-identification audit](phase0/donor-audit/summary.json) and its [individual rows](phase0/donor-audit/donor-ranks.csv) describe a retrospective test using previously captured states. These are separate from the prospective trial results.

The human odor panel was collected by Andreas Keller and Leslie Vosshall. The present study refers to their published [2016 paper](https://doi.org/10.1186/s12868-016-0287-2) and the [Pyrfume archive](https://github.com/pyrfume/pyrfume-data/tree/main/keller_2016). The MIP protocol is credited to K. Ruge and Claude, version 1.0, under CC BY-SA 4.0. Source-specific provenance and terms remain with the supplied materials.
