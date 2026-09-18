# Irregular mare patch (IMP) segmentation

**Model:** NASA-IBM LFM IMP segmentation checkpoint (frozen encoder, UNet decoder), binary segmentation of
single-channel LROC NAC tiles at 1 m/px, 256 × 256 px (256 m on a side).

## Inference and postprocessing

The model outputs two-class logits; the prediction is `argmax` over classes. The raw artifact keeps the logits'
argmax mask and the class-1 softmax mean over predicted pixels.

## Published values

| Value | Origin | Meaning |
|-------|--------|---------|
| `pixelCount`, `areaM2` | model prediction | Mask pixels and their area at the tile's resolution |
| `pixelFraction` | model prediction | Share of the tile covered by the mask |
| `meanSoftmaxScore` | model prediction | Mean class-1 softmax over predicted pixels — **uncalibrated**, never a probability or a confidence |
| `iou` vs reference | MOONATLAS derived | Agreement with the Hargitai et al. annotation distributed by SomBench |

## Repeat observations

The same SomBench target may be imaged by several NAC products. Those observations are **grouped for
navigation only** (`imp_groups.py`); their segmentations are never merged, and the grouping is not a claim that
two observations show the identical physical feature.

## Language

An IMP candidate is a **model segmentation of a volcanic feature candidate**. It is not an active volcano, and
the published description says so.
