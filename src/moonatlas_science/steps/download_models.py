"""Download the pinned NASA-IBM LFM checkpoints and configs, verifying every checkpoint's sha256.

About 6.3 GB: backbone 2.4 GB (needed to construct the models), crater 1.1 GB, IMP 1.1 GB, ice 1.7 GB.

Usage: python -m moonatlas_science.steps.download_models
"""



from moonatlas_science import hub

if __name__ == "__main__":
    for path in hub.fetch_models():
        print(f"ok {path}")
