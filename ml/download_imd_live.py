from datetime import date, timedelta
from pathlib import Path

import requests


# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "live_imd"
)

IMD_URL = (
    "https://www.imdpune.gov.in/"
    "cmpg/Realtimedata/Rainfall/rain.php"
)

# IMD 0.25 x 0.25 rainfall grid
NX = 135
NY = 129
BYTES_PER_VALUE = 4

EXPECTED_FILE_SIZE = (
    NX
    * NY
    * BYTES_PER_VALUE
)


# ==========================================================
# Download one date
# ==========================================================

def download_imd_rainfall(
    target_date: date,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = (
        "imd_rainfall_"
        f"{target_date.strftime('%Y%m%d')}"
        ".bin"
    )

    output_path = (
        OUTPUT_DIR
        / filename
    )

    if (
        output_path.exists()
        and
        output_path.stat().st_size
        == EXPECTED_FILE_SIZE
    ):
        print(
            target_date,
            "already exists",
            f"({EXPECTED_FILE_SIZE} bytes)",
        )

        return True

    # IMD form expects ddmmyyyy
    imd_date = (
        target_date.strftime(
            "%d%m%Y"
        )
    )

    print()
    print(
        "Downloading:",
        target_date,
    )

    print(
        "IMD form date:",
        imd_date,
    )

    try:
        response = requests.post(
            IMD_URL,
            data={
                "rain": imd_date,
            },
            timeout=60,
            allow_redirects=True,
        )

        response.raise_for_status()

    except Exception as error:
        print(
            "Request failed:",
            error,
        )

        return False

    data = response.content

    print(
        "Received:",
        len(data),
        "bytes",
    )

    print(
        "Content-Type:",
        response.headers.get(
            "Content-Type"
        ),
    )

    print(
        "Content-Disposition:",
        response.headers.get(
            "Content-Disposition"
        ),
    )

    # ------------------------------------------------------
    # Validate binary
    # ------------------------------------------------------

    if len(data) != EXPECTED_FILE_SIZE:
        print(
            "NOT SAVED:"
        )

        print(
            "Expected",
            EXPECTED_FILE_SIZE,
            "bytes but received",
            len(data),
        )

        preview = data[:300]

        try:
            preview = preview.decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            preview = str(
                preview
            )

        print(
            "Response preview:"
        )

        print(
            preview
        )

        return False

    # ------------------------------------------------------
    # Save
    # ------------------------------------------------------

    output_path.write_bytes(
        data
    )

    print(
        "Saved:",
        output_path,
    )

    print(
        "Valid IMD grid:",
        f"{NX} x {NY}",
    )

    return True


# ==========================================================
# Download date range
# ==========================================================

def download_date_range(
    start_date: date,
    end_date: date,
):
    current = start_date

    successful = []
    unavailable = []

    while current <= end_date:

        success = (
            download_imd_rainfall(
                current
            )
        )

        if success:
            successful.append(
                current
            )
        else:
            unavailable.append(
                current
            )

        current += timedelta(
            days=1
        )

    print()
    print(
        "=" * 60
    )

    print(
        "IMD LIVE DOWNLOAD SUMMARY"
    )

    print(
        "=" * 60
    )

    print(
        "Downloaded / available:",
        len(successful),
    )

    for item in successful:
        print(
            " ✓",
            item,
        )

    if unavailable:

        print()

        print(
            "Not available / failed:",
            len(unavailable),
        )

        for item in unavailable:
            print(
                " ✗",
                item,
            )


# ==========================================================
# Run
# ==========================================================

if __name__ == "__main__":

    download_date_range(

        start_date=date(
            2026,
            9,
            1,
        ),

        end_date=date(
            2026,
            9,
            4,
        ),
    )