import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATABASE_FILE = PROJECT_ROOT / "database" / "astrablend.db"


def main() -> None:
    connection = sqlite3.connect(DATABASE_FILE)
    cursor = connection.cursor()

    # --------------------------------------------------------
    # Weather models
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            provider TEXT,
            model_type TEXT,
            description TEXT
        )
        """
    )

    # --------------------------------------------------------
    # Geographic locations
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            state TEXT,
            district TEXT
        )
        """
    )

    # --------------------------------------------------------
    # Original model-skill table
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_skill (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id INTEGER NOT NULL,
            location_id INTEGER,
            variable TEXT NOT NULL,
            lead_hours INTEGER,
            season TEXT,
            weather_regime TEXT,
            mae REAL,
            rmse REAL,
            bias REAL,
            sample_count INTEGER,
            FOREIGN KEY (model_id) REFERENCES weather_models(id),
            FOREIGN KEY (location_id) REFERENCES locations(id)
        )
        """
    )

    # --------------------------------------------------------
    # Forecast-skill table
    #
    # This will eventually store the historical performance
    # of each forecast model by location, season and lead time.
    # --------------------------------------------------------
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS forecast_skill (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            season TEXT,
            lead_hours INTEGER NOT NULL,
            mae REAL,
            rmse REAL,
            bias REAL,
            sample_count INTEGER,
            UNIQUE(
                model_name,
                latitude,
                longitude,
                season,
                lead_hours
            )
        )
        """
    )

    connection.commit()
    connection.close()

    print("===== ASTRA BLEND DATABASE =====")
    print(f"Database: {DATABASE_FILE}")
    print("Database initialization complete.")


if __name__ == "__main__":
    main()