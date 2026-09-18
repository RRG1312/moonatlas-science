# Checkpoints

All four artefacts are released by NASA-IBM AI4Science under Apache-2.0 and are downloaded at pinned revisions,
sha256-verified. MOONATLAS runs them unmodified: no retraining, no distillation, no fine-tuning.

| Checkpoint | Head | Input | Official test metrics (model card) |
|------------|------|-------|-------------------------------------|
| Backbone | — | Multimodal, multiresolution | Needed to construct every task model |
| `WAC_ni_lfm_ps8_lora_s46.ckpt` | Faster R-CNN + LoRA | WAC 512 × 512 at ~100 m/px | mAP 0.2581 · AP@50 0.6183 |
| `ni_lfm_ps8_all_modalities_s42.ckpt` | Dense regression (full fine-tune) | 8 layers / 9 channels, 256 × 256 at 240 m/px | RMSE 0.0293 · MAE 0.0197 · R² 0.9884 |
| `ni_lfm_ps8_frozen_s44.ckpt` | Frozen encoder + UNet decoder | NAC 256 × 256 at 1 m/px | IoU₁ 0.5709 · F1₁ 0.7268 |

Loading is strict: an unexpected or missing state-dict key aborts the run and is recorded in the raw artifact's
`run.json`. The exact revisions in use are printed by `moonatlas-science env` and written into every build's
`sources.json`.

Upstream states that these outputs are not validated for operational decision-making. MOONATLAS repeats that and
makes no stronger claim.
