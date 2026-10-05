from pathlib import Path

import cfgrib
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "forecasts"

DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = DATA_DIR / "gfs_live_f006.grib2"


URL = (
    "https://nomads.ncep.noaa.gov/cgi-bin/"
    "filter_gfs_0p25.pl"
)

PARAMS = {
    "file": "gfs.t00z.pgrb2.0p25.f006",
    "var_APCP": "on",
    "lev_surface": "on",
    "subregion": "",
    "leftlon": "68",
    "rightlon": "98",
    "toplat": "37",
    "bottomlat": "6",
    "dir": "/gfs.20260828/00/atmos",
}


def download_gfs() -> None:
    print("===== DOWNLOADING LIVE GFS =====")

    response = requests.get(
        URL,
        params=PARAMS,
        timeout=120,
    )

    response.raise_for_status()

    if not response.content.startswith(b"GRIB"):
        raise RuntimeError(
            "Downloaded content does not look like a GRIB2 file."
        )

    OUTPUT_FILE.write_bytes(
        response.content
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Size: {len(response.content):,} bytes")


def decode_gfs() -> None:
    print()
    print("===== DECODING LIVE GFS =====")

    datasets = cfgrib.open_datasets(
        OUTPUT_FILE
    )

    print(
        f"Datasets found: {len(datasets)}"
    )

    for i, dataset in enumerate(datasets):

        if "tp" in dataset.data_vars:
            variable = dataset["tp"]

        elif "apcp" in dataset.data_vars:
            variable = dataset["apcp"]

        else:
            continue

        print()
        print(f"===== DATASET {i} =====")
        print(dataset)

        print()
        print("Variable:", variable.name)
        print(
            "Units:",
            variable.attrs.get("units"),
        )
        print(
            "Short name:",
            variable.attrs.get(
                "GRIB_shortName"
            ),
        )
        print(
            "Step type:",
            variable.attrs.get(
                "GRIB_stepType"
            ),
        )

        print(
            "Minimum:",
            float(variable.min()),
        )

        print(
            "Maximum:",
            float(variable.max()),
        )

        print(
            "Mean:",
            float(variable.mean()),
        )

        print(
            "Valid time:",
            dataset["valid_time"].values,
        )


def main() -> None:
    download_gfs()
    decode_gfs()


if __name__ == "__main__":
    main()