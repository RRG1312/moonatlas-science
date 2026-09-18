"""Compose per-pole ice mosaics from the normalized sectors in their native polar stereographic grids.

Fails on mismatched CRS, misaligned grids or overlapping patches; poles without processed coverage are omitted.

Usage: python -m moonatlas_science.steps.build_ice_mosaic
"""



from moonatlas_science import mosaic

if __name__ == "__main__":
    for item in mosaic.build_mosaics():
        print(f"{item['id']}: {len(item['patches'])} patches, {item['grid']['widthPx']}x{item['grid']['heightPx']} px")
