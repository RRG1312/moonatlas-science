# Data origin hierarchy

Four origins. They are never mixed, and every published value declares one.

| Origin | What it is | Examples |
|--------|------------|----------|
| `source_observation` | Upstream imagery, input layers, tile metadata | A WAC tile, a NAC tile, the ice input layers |
| `reference_annotation` | Published human/expert work, shown for comparison | Robbins (2019) boxes, the Coyan et al. prospectivity map, Hargitai et al. masks |
| `model_prediction` | MOONATLAS's own inference with a published checkpoint | Crater boxes, the prospectivity array, the IMP mask |
| `derived_metric` | A MOONATLAS computation over the above | IoU, RMSE, match flags, interest scores, coverage measurements |

Rules:

1. A reference annotation is never presented as model output, and never the other way round.
2. The default presentation of MOONATLAS data is the model's own prediction.
3. A derived metric is always labelled as derived and carries a versioned formula id.
4. Nothing is published without provenance; a value that cannot carry provenance does not belong in the
   contract.
