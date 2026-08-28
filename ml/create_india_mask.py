import json
from pathlib import Path

import numpy as np
import xarray as xr


RAINFALL_FILE = Path("data/processed/rainfall_2020_clean.nc")
INDIA_FILE = Path("data/boundaries/india.geojson")
OUTPUT_FILE = Path("data/processed/india_mask_2020.nc")


def point_in_polygon(x: float, y: float, polygon: list[list[float]]) -> bool:
    """
    Return True if point (x, y) is inside a polygon.

    x = longitude
    y = latitude
    """
    inside = False
    j = len(polygon) - 1

    for i in range(len(polygon)):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-15) + xi
        )

        if intersects:
            inside = not inside

        j = i

    return inside


def point_in_geometry(
    longitude: float,
    latitude: float,
    geometry: dict,
) -> bool:
    """
    Check whether a point falls inside a GeoJSON Polygon or MultiPolygon.
    """
    geometry_type = geometry["type"]
    coordinates = geometry["coordinates"]

    if geometry_type == "Polygon":
        # First ring = outer boundary.
        if point_in_polygon(longitude, latitude, coordinates[0]):
            return True

        # Other rings are holes.
        for hole in coordinates[1:]:
            if point_in_polygon(longitude, latitude, hole):
                return False

        return False

    if geometry_type == "MultiPolygon":
        for polygon in coordinates:
            if point_in_polygon(longitude, latitude, polygon[0]):
                in_hole = any(
                    point_in_polygon(longitude, latitude, hole)
                    for hole in polygon[1:]
                )

                if not in_hole:
                    return True

        return False

    raise ValueError(f"Unsupported GeoJSON geometry: {geometry_type}")


def main() -> None:
    print("Loading rainfall grid...")

    dataset = xr.open_dataset(RAINFALL_FILE)
    rainfall = dataset["rainfall"]

    print("Loading India boundary...")

    with INDIA_FILE.open("r", encoding="utf-8") as file:
        india_geojson = json.load(file)

    india_geometry = india_geojson["features"][0]["geometry"]

    latitudes = rainfall["LATITUDE"].values
    longitudes = rainfall["LONGITUDE"].values

    mask = np.zeros(
        (len(latitudes), len(longitudes)),
        dtype=bool,
    )

    print("Creating India land mask...")

    for lat_index, latitude in enumerate(latitudes):
        for lon_index, longitude in enumerate(longitudes):
            mask[lat_index, lon_index] = point_in_geometry(
                float(longitude),
                float(latitude),
                india_geometry,
            )

    mask_da = xr.DataArray(
        mask,
        coords={
            "LATITUDE": latitudes,
            "LONGITUDE": longitudes,
        },
        dims=("LATITUDE", "LONGITUDE"),
        name="india_land_mask",
    )

    mask_dataset = mask_da.to_dataset()

    mask_dataset["india_land_mask"].attrs["description"] = (
        "Approximate India land mask derived from Natural Earth country boundary"
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    mask_dataset.to_netcdf(OUTPUT_FILE)

    total_cells = mask.size
    india_cells = int(mask.sum())

    print()
    print("===== ASTRA BLEND: INDIA MASK =====")
    print(f"Grid cells:       {total_cells:,}")
    print(f"India cells:      {india_cells:,}")
    print(f"Outside India:    {total_cells - india_cells:,}")
    print(f"India coverage:   {india_cells / total_cells * 100:.2f}%")
    print(f"Saved to:         {OUTPUT_FILE}")


if __name__ == "__main__":
    main()