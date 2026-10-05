import json
from pathlib import Path


INPUT_FILE = Path("data/boundaries/countries.geojson")
OUTPUT_FILE = Path("data/boundaries/india.geojson")


def main() -> None:
    with INPUT_FILE.open("r", encoding="utf-8") as file:
        data = json.load(file)

    india_feature = None

    for feature in data["features"]:
        properties = feature.get("properties", {})

        # Natural Earth uses ADMIN for the country name.
        if properties.get("ADMIN") == "India":
            india_feature = feature
            break

    if india_feature is None:
        raise ValueError("India was not found in the GeoJSON file.")

    india_data = {
        "type": "FeatureCollection",
        "name": "India",
        "features": [india_feature],
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(india_data, file)

    print("===== ASTRA BLEND: INDIA BOUNDARY =====")
    print("India boundary found successfully.")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()