from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr


INPUT_FILE = Path("data/processed/india_rainfall_2020.nc")
OUTPUT_FILE = Path("data/processed/india_rainfall_2020_map.png")


def main() -> None:
    dataset = xr.open_dataset(INPUT_FILE)
    rainfall = dataset["rainfall"]

    # Add daily rainfall values across 2020.
    annual_rainfall = rainfall.sum(dim="TIME", skipna=True)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot the real 0.25° IMD grid.
    annual_rainfall.plot(
        ax=ax,
        cmap="viridis",
        cbar_kwargs={
            "label": "Rainfall (mm)"
        },
    )

    ax.set_title(
        "AstraBlend — IMD Annual Rainfall — India — 2020",
        fontsize=16,
        pad=15,
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.grid(
        True,
        alpha=0.25,
        linewidth=0.5,
    )

    fig.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        OUTPUT_FILE,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("===== ASTRA BLEND: RAINFALL MAP =====")
    print(f"Saved map to: {OUTPUT_FILE}")
    print(
        f"Minimum annual rainfall: "
        f"{float(annual_rainfall.min()):.2f} mm"
    )
    print(
        f"Maximum annual rainfall: "
        f"{float(annual_rainfall.max()):.2f} mm"
    )


if __name__ == "__main__":
    main()