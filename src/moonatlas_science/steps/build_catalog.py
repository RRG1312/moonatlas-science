"""Build imp-observation-groups.json, discoveries.json and global-stats.json from the normalized build.

Rules live in catalog.py and imp_groups.py.

Public discovery ids come from the append-only registry (configs/discovery-registry.json, committed); new
identities are appended to it. Usage: python -m moonatlas_science.steps.build_catalog
"""

from datetime import datetime, timezone

from moonatlas_science import build

from ..catalog import build_discoveries, build_stats
from ..imp_groups import group_observations
from ..registry import DiscoveryRegistry

if __name__ == "__main__":
    regions = build.read_json("crater-regions.json", [])
    craters = build.read_json("craters.json", {})
    sectors = build.read_json("ice-sectors.json", [])
    imp_regions = build.read_json("imp-regions.json", [])
    imp_groups = group_observations(imp_regions)
    build.write_json("imp-observation-groups.json", imp_groups)
    date = datetime.now(timezone.utc).date().isoformat()
    registry = DiscoveryRegistry.load()
    registered = len(registry.entries)
    discoveries = build_discoveries(regions, craters, sectors, imp_regions, registry,
                                    lambda region: build.read_json(region["detailPath"])["prediction"], date,
                                    imp_groups=imp_groups)
    build.write_json("discoveries.json", discoveries)
    registry.save()
    stats = build_stats(regions, craters, sectors, imp_regions, imp_groups, discoveries,
                        lambda region: len(build.read_json(region["detailPath"])["reference"]))
    build.write_json("global-stats.json", stats)
    print(f"{len(discoveries)} discoveries · {stats['craterDetections']} crater model detections · "
          f"{stats['polarPatchesAnalyzed']['north'] + stats['polarPatchesAnalyzed']['south']} ice patches · "
          f"{stats['impTilesAnalyzed']} IMP observations in {stats['impObservationGroups']} groups · registry {registered} -> {len(registry.entries)} ids")
