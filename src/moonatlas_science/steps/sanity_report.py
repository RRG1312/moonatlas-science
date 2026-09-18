"""Generate visual sanity figures and the sanity report for inferred samples.

Usage: python -m moonatlas_science.steps.sanity_report   (the three science smoke-test samples)
"""



from moonatlas_science import config, sanity

if __name__ == "__main__":
    summaries = [
        sanity.crater_figure(config.SMOKE_SAMPLES["crater-detection"])[1],
        sanity.ice_figure(config.SMOKE_SAMPLES["ice-prospectivity"])[1],
        sanity.imp_figure(config.SMOKE_SAMPLES["imp-segmentation"])[1],
    ]
    files = sorted(p.relative_to(config.BUILD_DIR).as_posix() for p in config.BUILD_DIR.rglob("*") if p.is_file())
    report = sanity.write_report(summaries, files)
    print(report.read_text(encoding="utf-8"))
