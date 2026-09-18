"""Scientific sources and attribution for sources.json.

Shared by the real pipeline (build-dataset.py) and the mock fixture generator so
attribution never drifts. Revisions are filled with the pinned Hugging Face commits
by the real pipeline; they stay None for mock data.
Only verifiable URLs: no DOIs are guessed for references we have not confirmed.
"""

from __future__ import annotations

HF = "https://huggingface.co"
FRACCARO_2026 = (
    "Fraccaro, P. et al. (2026). Multimodal-Multiresolution Foundation Model for Lunar Remote Sensing. "
    "NASA-IBM AI4Science."
)
PATIL_2026 = (
    "Patil, H. et al. (2026). SomBench: Benchmark Dataset for Advancing Machine Learning in Lunar Science. "
    "NASA-IBM AI4Science."
)


def _source(id, kind, name, organization, url, license, citation, modified):
    return {
        "id": id,
        "kind": kind,
        "name": name,
        "organization": organization,
        "url": url,
        "license": license,
        "citation": citation,
        "revision": None,
        "modified": modified,
    }


SOURCES = [
    _source("lfm-backbone", "model", "NASA-IBM Lunar Foundation Model", "NASA-IBM AI4Science",
            f"{HF}/nasa-ibm-ai4science/NASA-IBM-Lunar-Foundation-Model", "Apache-2.0", FRACCARO_2026, False),
    _source("lfm-crater-detection", "model", "NASA-IBM LFM — Crater Detection", "NASA-IBM AI4Science",
            f"{HF}/nasa-ibm-ai4science/Crater-Detection-NASA-IBM-Lunar-Foundation-Model", "Apache-2.0",
            FRACCARO_2026, False),
    _source("lfm-ice-prospectivity", "model", "NASA-IBM LFM — Polar Ice Prospectivity", "NASA-IBM AI4Science",
            f"{HF}/nasa-ibm-ai4science/Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model", "Apache-2.0",
            FRACCARO_2026, False),
    _source("lfm-imp-segmentation", "model", "NASA-IBM LFM — Irregular Mare Patch Segmentation",
            "NASA-IBM AI4Science", f"{HF}/nasa-ibm-ai4science/IMP-Segmentation-NASA-IBM-Lunar-Foundation-Model",
            "Apache-2.0", FRACCARO_2026, False),
    _source("sombench-wac-crater-detection", "dataset", "SomBench: Robbins Crater Detection (WAC)",
            "NASA-IBM AI4Science", f"{HF}/datasets/nasa-ibm-ai4science/Sombench-WAC-Crater-Detection",
            "CC-BY-4.0", PATIL_2026, True),
    _source("sombench-ice-prospectivity-regression", "dataset", "SomBench: Polar Ice Prospectivity Regression",
            "NASA-IBM AI4Science", f"{HF}/datasets/nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression",
            "CC-BY-4.0", PATIL_2026, True),
    _source("sombench-imp-segmentation", "dataset", "SomBench: Irregular Mare Patch Segmentation",
            "NASA-IBM AI4Science", f"{HF}/datasets/nasa-ibm-ai4science/Sombench-IMP-Segmentation",
            "CC-BY-4.0", PATIL_2026, True),
    _source("robbins-2019", "reference", "Robbins (2019) global lunar crater catalog", "Robbins, S. J.",
            "https://doi.org/10.1029/2018JE005592", "Publication",
            "Robbins, S. J. (2019). A New Global Database of Lunar Impact Craters >1–2 km. JGR Planets, 124(4), "
            "871–892.", False),
    _source("coyan-2025", "reference", "Coyan et al. (2025) polar ice prospectivity map", "Coyan et al.",
            f"{HF}/datasets/nasa-ibm-ai4science/Sombench-Ice-Prospectivity-Regression", "Publication",
            "Coyan et al. (2025), knowledge-driven fuzzy-overlay polar ice prospectivity map, as packaged in "
            "SomBench.", False),
    _source("hargitai-2025", "reference", "Hargitai et al. (2025) irregular mare patch annotations",
            "Hargitai et al.", f"{HF}/datasets/nasa-ibm-ai4science/Sombench-IMP-Segmentation", "Publication",
            "Hargitai et al. (2025), irregular mare patch polygon annotations, as packaged in SomBench.", False),
    _source("nasa-svs-cgi-moon-kit", "imagery", "CGI Moon Kit (LROC color map)",
            "NASA Scientific Visualization Studio", "https://svs.gsfc.nasa.gov/4720", "NASA media guidelines",
            "NASA's Scientific Visualization Studio, CGI Moon Kit (LRO LROC and LOLA data).", True),
    _source("lro-lroc", "instrument", "Lunar Reconnaissance Orbiter Camera (LROC)", "NASA / Arizona State University",
            "https://lroc.im-ldi.com", "Public data", "LRO LROC WAC and NAC imagery.", False),
    _source("lro-lola", "instrument", "Lunar Orbiter Laser Altimeter (LOLA)", "NASA Goddard Space Flight Center",
            "https://science.nasa.gov/mission/lro/", "Public data",
            "LRO LOLA topography (slope, aspect, curvature).", False),
    _source("lro-diviner", "instrument", "Diviner Lunar Radiometer Experiment", "NASA / UCLA",
            "https://www.diviner.ucla.edu", "Public data",
            "LRO Diviner thermal observations (maximum temperature, ice stability).", False),
]
