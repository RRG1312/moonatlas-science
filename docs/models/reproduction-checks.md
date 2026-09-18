# Reproduction checks

MOONATLAS recomputes the upstream test configuration over its own held-out samples and publishes both numbers.
These are consistency checks on a small sample, **not** new benchmarks, and they never replace the model card.

| Task | Samples | MOONATLAS | Model card |
|------|---------|----------|------------|
| WAC crater detection | 100 held-out test tiles | mAP 0.2602 · AP@50 0.6239 | mAP 0.2581 · AP@50 0.6183 |
| IMP segmentation | 10 held-out test tiles | IoU₁ 0.5824 · F1₁ 0.7361 | IoU₁ 0.5709 · F1₁ 0.7268 |
| Polar ice prospectivity | — | RMSE/MAE against the reference map, per patch | The card publishes regression errors, not a per-split table |

Small samples move in both directions; a difference of this size is expected and is not evidence that the model
is better or worse than published.

Reproduce them with `moonatlas-science report crater_batch_report` and `report imp_batch_report` over a build
that contains the corresponding samples.
