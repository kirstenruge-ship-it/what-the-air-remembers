# What the air remembers

**Status: study registered; prospective Kimi inference has not begun at this snapshot.**

Kirsten Ruge’s independent Hive Fidelity research asks whether a hidden cue synthesized from Kimi K3.1 Pro’s own odor-similarity judgments can recover decisions about an unnamed sample and retrieve an arbitrary detail from a recent household episode. The fixed checkpoint is `moonshotai/Kimi-K3.1-Pro` at `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`. Study execution uses maximum reasoning effort, temperature 1, and natural generation without an application output-token cap.

The [scientific registration](protocol/natural-response-registration.md) defines the revised natural-response endpoint. The [scientific proposal sections](protocol/design.md) contain the original rationale and contrasts. The natural-generation requirements and frozen trial inventory supersede software-only examples or conditional sizing in that earlier document.

- [Frozen inputs and checksums](stimuli/manifest.json) cover 24 fitting, four development and 12 held-out compounds, 20 candidate reference odors and 24 household narratives. The 2,852 planned records include conditional extensions. [Every planned input](stimuli/trials.jsonl) includes the text and exact rendered token IDs.
- [Human-source provenance](stimuli/source/provenance.json) and the [selected human measurements](stimuli/source/selected-human-panel.csv.gz) document the Keller–Vosshall panel reanalysis.
- [Earlier Costco donor-identification audit](phase0/donor-audit/summary.json) and its [individual rows](phase0/donor-audit/donor-ranks.csv) describe a retrospective test using previously captured states. These are separate from the prospective trial results.

The human odor panel was collected by Andreas Keller and Leslie Vosshall. The present study refers to their published [2016 paper](https://doi.org/10.1186/s12868-016-0287-2) and the [Pyrfume archive](https://github.com/pyrfume/pyrfume-data/tree/main/keller_2016). The MIP protocol is credited to K. Ruge and Claude, version 1.0, under CC BY-SA 4.0. Source-specific provenance and terms remain with the supplied materials.

## Pre-exposure observations

The [original activation rows](phase0/donor-audit/native-rows/native-rows.json) make the earlier sentence-identification audit independently reproducible. The four NPZ files contain the relevant donor, dots and patched attention inputs at L28/L71/L92. Use `python phase0/recheck_published_donors.py phase0/donor-audit` from this repository with NumPy installed. Each recovered sentence is one candidate in a finite library.

The [human split-half analysis](analysis/phase0-human/human-reliability.json) measures the stability of the ratings-derived external reference. The [low-dose arithmetic tests](phase0/dose-rounding/summary.json) and [corrected native rounding bound](analysis/phase0-dose/historical-bf16-cap.json) use old Kimi states and CPU arithmetic. The [Monte Carlo sensitivity grid](phase0/design-sensitivity/operating-characteristics.csv) uses synthetic data only. None of these pre-exposure results is a prospective hidden-odor behavioral effect.

[Published subset for the three-layer wording and rounding controls](phase0/frame-inputs/frame-records.json). The frame-specificity and dose-rounding scripts accept the frame-inputs directory directly. The [lexical surrogate analysis](phase0/lexical-surrogate/summary.json) fits on other donors, withholding both candidates for every pair. The [48-record condition-masked historical safety audit](phase0/prior-safety-coding/result.json) includes separate reviews and the underlying historical texts. All these samples predate the prospective experiment and retain their original status.
