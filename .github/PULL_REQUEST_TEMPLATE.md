# Summary

<!-- What does this change and why. Link the issue or discussion. -->

## Scientific impact

- **What scientific or data behaviour changes?**
- **Does this change raw model output?** (yes / no — if yes, explain why that is correct)
- **Does this change normalization** (values, units, georeferencing, contract fields)?
- **Is this display-only** (colours, previews, figures) with no effect on published values?
- **Does this introduce or modify a derived metric?** (name, formula, version id)

## Provenance

- **What provenance is attached to any new or changed value?** (model, checkpoint, revision, sourceDataset,
  sourceTile, datasetSplit, processingVersion)
- **Origin level of the affected values:** source observation / reference annotation / model prediction /
  MOONATLAS derived metric

## Tests

- **What tests demonstrate correctness?** (paths + what fails without the change)
- Do the tests run on a clean clone without downloads? If not, how do they skip?

## Reproducibility

- **Does this affect reproducibility or determinism?** (byte-identical rebuild)
- **Does it change an upstream dataset, model or version?** (which, from which revision to which)
- **Does `processingVersion` need a bump?** (no / minor / major — and is it in this PR?)

## Licensing and attribution

- **Are licensing or attribution changes required?** (`NOTICE`, `configs/`, `sources.json`)
- Does this add redistributed third-party material? If yes, which licence permits it, and is the modification
  notice present?

## Wording check

- [ ] Candidate / prediction wording — no *confirmed*, *discovered*, *verified*, *found*
- [ ] Ice prospectivity is not described as a probability or a measurement of ice
- [ ] IMP softmax described as uncalibrated; IMP candidate is not an active volcano
- [ ] Reference annotations are not presented as model output
- [ ] Proximity is not presented as coverage
- [ ] Train/validation results labelled in-sample

## Checklist

- [ ] `pytest` passes
- [ ] `ruff check .` passes
- [ ] Docs updated (methodology / provenance / limitations) where behaviour changed
- [ ] No checkpoints, datasets, builds or artifacts committed
