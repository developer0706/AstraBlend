from pathlib import Path

import h5py


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GFS_FILE = (
    PROJECT_ROOT
    / "data"
    / "forecasts"
    / "gfs_india_f006.nc"
)


def show_item(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"DATASET: {name}")
        print(f"  Shape: {obj.shape}")
        print(f"  Dtype: {obj.dtype}")

        for key, value in obj.attrs.items():
            print(f"  ATTR {key}: {value}")


def main():
    print("===== ASTRA BLEND: GFS NETCDF4 INSPECTION =====")
    print(f"File: {GFS_FILE}")

    if not GFS_FILE.exists():
        raise FileNotFoundError(
            f"GFS file not found: {GFS_FILE}"
        )

    with h5py.File(GFS_FILE, "r") as file:
        print()
        print("===== FILE STRUCTURE =====")
        file.visititems(show_item)

        print()
        print("===== ROOT ATTRIBUTES =====")

        for key, value in file.attrs.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()