from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
import ssl

import certifi


# ============================================================
# AstraBlend — GFS Download Test
# ============================================================

# Use a recent known GFS date/cycle for the first test.
DATE = "20260831"
CYCLE = "00"

# +24 hours from the 18 UTC forecast run.
FORECAST_HOUR = 24

# India bounding box.
LEFT_LON = 66.5
RIGHT_LON = 100.0
TOP_LAT = 38.5
BOTTOM_LAT = 6.5

OUTPUT_DIR = Path("data/raw/gfs")


def build_url() -> str:
    """Build the NOAA/NCEP NOMADS GFS subset URL."""

    filename = f"gfs.t{CYCLE}z.pgrb2.0p25.f{FORECAST_HOUR:03d}"

    params = {
        "file": filename,
        "dir": f"/gfs.{DATE}/{CYCLE}/atmos",
        "var_APCP": "on",
        "subregion": "",
        "leftlon": str(LEFT_LON),
        "rightlon": str(RIGHT_LON),
        "toplat": str(TOP_LAT),
        "bottomlat": str(BOTTOM_LAT),
    }

    base_url = (
        "https://nomads.ncep.noaa.gov/cgi-bin/"
        "filter_gfs_0p25.pl"
    )

    return f"{base_url}?{urlencode(params)}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    url = build_url()

    output_file = (
        OUTPUT_DIR
        / f"gfs_{DATE}_{CYCLE}_f{FORECAST_HOUR:03d}.grib2"
    )

    print("===== ASTRA BLEND: GFS DOWNLOAD TEST =====")
    print(f"Date:          {DATE}")
    print(f"Cycle:         {CYCLE} UTC")
    print(f"Forecast hour: +{FORECAST_HOUR} h")
    print(f"Source URL:    {url}")
    print(f"Output file:   {output_file}")

    try:
        ssl_context = ssl.create_default_context(
            cafile=certifi.where()
        )

        with urlopen(
            url,
            timeout=120,
            context=ssl_context,
        ) as response:
            data = response.read()

        if not data:
            raise RuntimeError("NOAA returned an empty response.")

        output_file.write_bytes(data)

    except Exception as error:
        print()
        print("DOWNLOAD FAILED")
        print(f"Reason: {error}")
        return

    file_size_kb = output_file.stat().st_size / 1024

    print()
    print("DOWNLOAD SUCCESSFUL")
    print(f"File size: {file_size_kb:.1f} KB")
    print(f"Saved to:  {output_file}")


if __name__ == "__main__":
    main()