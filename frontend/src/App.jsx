import { useEffect, useMemo, useState } from "react";
import {
  Circle,
  MapContainer,
  Popup,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";
const INDIA_CENTER = [22.5, 79];

const CITIES = [
  { name: "Delhi", latitude: 28.61, longitude: 77.21 },
  { name: "Mumbai", latitude: 19.08, longitude: 72.88 },
  { name: "Bengaluru", latitude: 12.97, longitude: 77.59 },
  { name: "Chennai", latitude: 13.08, longitude: 80.27 },
  { name: "Kolkata", latitude: 22.57, longitude: 88.36 },
  { name: "Hyderabad", latitude: 17.39, longitude: 78.49 },
  { name: "Jaipur", latitude: 26.91, longitude: 75.79 },
  { name: "Ahmedabad", latitude: 23.02, longitude: 72.57 },
  { name: "Pune", latitude: 18.52, longitude: 73.86 },
  { name: "Lucknow", latitude: 26.85, longitude: 80.95 },
  { name: "Bhopal", latitude: 23.26, longitude: 77.41 },
  { name: "Guwahati", latitude: 26.14, longitude: 91.74 },
];

const DEMO_SCENARIOS = {
  normal: {
    label: "Low Risk",
    probability: 4.8,
    rainfall: 8.4,
    title: "No heavy-rain warning",
    description:
      "Forecast conditions remain below the heavy-rain alert threshold.",
    severity: "normal",
    icon: "✓",
  },

  heavy: {
    label: "Heavy Rain",
    probability: 72.4,
    rainfall: 89.6,
    title: "Heavy rainfall risk detected",
    description:
      "Elevated rainfall risk detected. Monitor local weather conditions and official advisories.",
    severity: "heavy",
    icon: "!",
  },

  extreme: {
    label: "Extreme",
    probability: 94.7,
    rainfall: 218.4,
    title: "Extreme rainfall alert",
    description:
      "Very high rainfall risk detected. Severe rainfall conditions may cause flooding and disruption.",
    severity: "extreme",
    icon: "!!",
  },
};

/*
  These are date-stamped official IMD reference examples,
  NOT AstraBlend model output.
*/
const IMD_REFERENCE_ALERTS = [
  {
    region: "Odisha",
    level: "Extremely Heavy Rain",
    date: "30 Aug 2026",
    type: "extreme",
  },
  {
    region: "Gangetic West Bengal",
    level: "Very Heavy Rain",
    date: "30 Aug 2026",
    type: "very-heavy",
  },
  {
    region: "Assam & Meghalaya",
    level: "Very Heavy Rain",
    date: "30 Aug 2026",
    type: "very-heavy",
  },
];

function LocationPicker({ onSelect }) {
  useMapEvents({
    click(event) {
      onSelect(
        Number(event.latlng.lat.toFixed(2)),
        Number(event.latlng.lng.toFixed(2)),
        "Custom location"
      );
    },
  });

  return null;
}

function MapRecenter({ latitude, longitude }) {
  const map = useMap();

  useEffect(() => {
    map.flyTo(
      [latitude, longitude],
      Math.max(map.getZoom(), 6),
      { duration: 0.85 }
    );
  }, [latitude, longitude, map]);

  return null;
}

function forecastColor(rainfall = 0) {
  if (rainfall >= 100) return "#8b5cf6";
  if (rainfall >= 50) return "#ef4444";
  if (rainfall >= 25) return "#f97316";
  if (rainfall >= 10) return "#facc15";
  if (rainfall >= 2.5) return "#22c55e";
  return "#38bdf8";
}

function forecastRadius(rainfall = 0) {
  if (rainfall >= 100) return 13;
  if (rainfall >= 50) return 11;
  if (rainfall >= 25) return 9;
  if (rainfall >= 10) return 7;
  if (rainfall >= 2.5) return 5;
  return 3.5;
}

/*
  IMD rainfall classification.
*/
function rainfallSeverity(rainfall) {
  const value = Number(rainfall);

  if (!Number.isFinite(value)) {
    return {
      label: "Awaiting forecast",
      className: "unknown",
    };
  }

  if (value < 0.1) {
    return {
      label: "Dry",
      className: "dry",
    };
  }

  if (value <= 15.5) {
    return {
      label: "Light Rain",
      className: "light",
    };
  }

  if (value <= 64.4) {
    return {
      label: "Moderate Rain",
      className: "moderate-rain",
    };
  }

  if (value <= 115.5) {
    return {
      label: "Heavy Rain",
      className: "heavy-rain",
    };
  }

  if (value <= 204.4) {
    return {
      label: "Very Heavy Rain",
      className: "very-heavy-rain",
    };
  }

  return {
    label: "Extremely Heavy Rain",
    className: "extreme-rain",
  };
}

function formatDate(dateString) {
  if (!dateString) return "--";

  const date = new Date(`${dateString}T00:00:00`);

  if (Number.isNaN(date.getTime())) {
    return dateString;
  }

  return date.toLocaleDateString("en-IN", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });
}

function RiskGauge({ value = 0 }) {
  const safeValue = Math.min(
    Math.max(Number(value) || 0, 0),
    100
  );

  return (
    <div className="risk-gauge">
      <div className="risk-track">
        <div
          className="risk-fill"
          style={{ width: `${safeValue}%` }}
        />
      </div>

      <span>{safeValue.toFixed(1)}%</span>
    </div>
  );
}


function ModelComparisonChart({ forecast, locationName }) {
  const values = useMemo(() => {
    if (!forecast) return null;

    const ml = Number(forecast.ml_forecast_rainfall_mm ?? 0);
    const gfs = Number(forecast.gfs_forecast_rainfall_mm ?? 0);
    const astrablend = Number(
      forecast.astrablend_forecast_rainfall_mm ?? 0
    );

    const maxValue = Math.max(ml, gfs, astrablend, 1);

    return [
      {
        label: "Machine Learning",
        short: "ML",
        value: ml,
        width: (ml / maxValue) * 100,
        gradient:
          "linear-gradient(90deg, #22d3ee 0%, #3b82f6 100%)",
      },
      {
        label: "GFS Model",
        short: "GFS",
        value: gfs,
        width: (gfs / maxValue) * 100,
        gradient:
          "linear-gradient(90deg, #818cf8 0%, #6366f1 100%)",
      },
      {
        label: "AstraBlend",
        short: "AB",
        value: astrablend,
        width: (astrablend / maxValue) * 100,
        gradient:
          "linear-gradient(90deg, #38bdf8 0%, #8b5cf6 100%)",
      },
    ];
  }, [forecast]);

  return (
    <div
      style={{
        marginTop: "28px",
        marginBottom: "28px",
        padding: "28px",
        border: "1px solid rgba(148, 163, 184, 0.16)",
        borderRadius: "24px",
        background:
          "linear-gradient(135deg, rgba(8,20,35,0.92), rgba(20,24,55,0.88))",
        boxShadow: "0 18px 45px rgba(2, 8, 23, 0.22)",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: "18px",
          flexWrap: "wrap",
          marginBottom: "26px",
        }}
      >
        <div>
          <span className="section-kicker">
            MODEL COMPARISON
          </span>
          <h3
            style={{
              margin: "7px 0 0",
              fontSize: "24px",
            }}
          >
            Rainfall forecast comparison
          </h3>
          <p
            style={{
              margin: "8px 0 0",
              color: "#7f93aa",
              fontSize: "14px",
            }}
          >
            Live ML, GFS and 60/40 AstraBlend forecast
            for the selected location.
          </p>
        </div>

        <span
          style={{
            padding: "9px 15px",
            borderRadius: "999px",
            background: "rgba(56, 189, 248, 0.08)",
            border: "1px solid rgba(56, 189, 248, 0.2)",
            color: "#7dd3fc",
            fontWeight: 700,
          }}
        >
          {locationName}
        </span>
      </div>

      {values ? (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: "22px",
          }}
        >
          {values.map((item) => (
            <div
              key={item.short}
              style={{
                display: "grid",
                gridTemplateColumns:
                  "minmax(135px, 0.8fr) minmax(220px, 3fr) 95px",
                alignItems: "center",
                gap: "18px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: "3px",
                }}
              >
                <strong
                  style={{
                    color: "#f8fafc",
                    fontSize: "15px",
                  }}
                >
                  {item.short}
                </strong>
                <span
                  style={{
                    color: "#64748b",
                    fontSize: "12px",
                  }}
                >
                  {item.label}
                </span>
              </div>

              <div
                style={{
                  height: "18px",
                  borderRadius: "999px",
                  background:
                    "rgba(148, 163, 184, 0.10)",
                  overflow: "hidden",
                }}
              >
                <div
                  style={{
                    width: `${item.width}%`,
                    height: "100%",
                    borderRadius: "999px",
                    background: item.gradient,
                    boxShadow:
                      "0 0 18px rgba(56, 189, 248, 0.18)",
                    transition: "width 0.8s ease",
                  }}
                />
              </div>

              <strong
                style={{
                  textAlign: "right",
                  color: "#f8fafc",
                  fontSize: "14px",
                }}
              >
                {item.value.toFixed(2)} mm
              </strong>
            </div>
          ))}

          <div
            style={{
              marginTop: "4px",
              paddingTop: "18px",
              borderTop:
                "1px solid rgba(148, 163, 184, 0.12)",
              color: "#7f93aa",
              fontSize: "13px",
            }}
          >
            AstraBlend = 60% Machine Learning + 40% GFS
          </div>
        </div>
      ) : (
        <div
          style={{
            padding: "28px",
            textAlign: "center",
            color: "#64748b",
            border:
              "1px dashed rgba(148, 163, 184, 0.18)",
            borderRadius: "16px",
          }}
        >
          Generate a forecast to compare ML, GFS and
          AstraBlend for {locationName}.
        </div>
      )}
    </div>
  );
}

function App() {
  const [latitude, setLatitude] = useState(23.26);
  const [longitude, setLongitude] = useState(77.41);

  const [locationName, setLocationName] =
    useState("Bhopal");

  const [selectedCity, setSelectedCity] =
    useState("Bhopal");

  const [forecast, setForecast] = useState(null);

  const [forecastPoints, setForecastPoints] =
    useState([]);

  const [mapMeta, setMapMeta] = useState(null);

  const [hotspotPoints, setHotspotPoints] =
    useState([]);

  const [hotspotMeta, setHotspotMeta] =
    useState(null);

  const [loading, setLoading] = useState(false);

  const [mapLoading, setMapLoading] =
    useState(true);

  const [error, setError] = useState("");

  /*
    null = genuine live AstraBlend state
    normal/heavy/extreme = judge demo preview
  */
  const [demoScenario, setDemoScenario] =
    useState(null);

  useEffect(() => {
    async function loadForecastMap() {
      try {
        setMapLoading(true);
        setError("");

        const response = await fetch(
          `${API_URL}/forecast-map`
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
              "Unable to load AstraBlend forecast map."
          );
        }

        setForecastPoints(data.points || []);
        setMapMeta(data);
      } catch (err) {
        setError(
          err.message ||
            "Unable to load AstraBlend forecast map."
        );
      } finally {
        setMapLoading(false);
      }
    }

    loadForecastMap();
  }, []);

  /*
    Live regional hotspot layer.
    This is intentionally separate from the original
    major-city /forecast workflow so the dashboard
    still works even if hotspot loading fails.
  */
  useEffect(() => {
    async function loadHotspots() {
      try {
        const response = await fetch(
          `${API_URL}/live-risk-regions`
        );

        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail ||
              "Unable to load live hotspot data."
          );
        }

        const points = Object.values(
          data.regions || {}
        ).flatMap((region) =>
          (region.locations || []).filter(
            (item) =>
              item.blend_ready === true &&
              !item.error
          )
        );

        setHotspotPoints(points);
        setHotspotMeta(data);
      } catch (err) {
        console.error(
          "Hotspot layer unavailable:",
          err
        );
      }
    }

    loadHotspots();
  }, []);

  function selectLocation(
    lat,
    lon,
    name = "Custom location"
  ) {
    setLatitude(lat);
    setLongitude(lon);
    setLocationName(name);

    const matchingCity = CITIES.find(
      (city) =>
        city.latitude === lat &&
        city.longitude === lon
    );

    setSelectedCity(
      matchingCity ? matchingCity.name : ""
    );

    setForecast(null);
    setDemoScenario(null);
    setError("");
  }

  function handleCityChange(event) {
    const name = event.target.value;

    setSelectedCity(name);

    const city = CITIES.find(
      (item) => item.name === name
    );

    if (!city) return;

    selectLocation(
      city.latitude,
      city.longitude,
      city.name
    );
  }

  function selectHotspot(point) {
    if (!point) return;

    setLatitude(point.latitude);
    setLongitude(point.longitude);
    setLocationName(point.location);
    setSelectedCity("");
    setDemoScenario(null);
    setError("");

    setForecast({
      ...point,

      ml_forecast_rainfall_mm:
        point.ml_forecast_mm,

      gfs_forecast_rainfall_mm:
        point.gfs_forecast_mm,

      astrablend_forecast_rainfall_mm:
        point.astrablend_mm,

      heavy_rain_probability_percent:
        point.heavy_probability_percent,

      heavy_rain_warning:
        point.heavy_classifier_warning,

      forecast_target_date:
        point.target_date,

      ml_source_date:
        point.source_date,
    });
  }

  async function getForecast() {
    setLoading(true);
    setError("");
    setDemoScenario(null);

    try {
      const response = await fetch(
        `${API_URL}/forecast?latitude=${encodeURIComponent(
          latitude
        )}&longitude=${encodeURIComponent(longitude)}`,
        {
          method: "POST",
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Forecast request failed."
        );
      }

      if (data.blend_ready === false) {
        throw new Error(
          data.status ||
            "AstraBlend is waiting for aligned input data."
        );
      }

      setForecast(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to generate forecast."
      );
    } finally {
      setLoading(false);
    }
  }

  const forecastDate =
    forecast?.forecast_target_date ||
    mapMeta?.forecast_target_date ||
    hotspotMeta?.forecast_target_date;

  const mapSummary = useMemo(() => {
  if (!forecastPoints.length) return {};

  const blendValues = forecastPoints
    .map((p) => Number(p.astrablend_forecast_rainfall_mm))
    .filter(Number.isFinite);

  const mlValues = forecastPoints
    .map((p) => Number(p.ml_forecast_rainfall_mm))
    .filter(Number.isFinite);

  const gfsValues = forecastPoints
    .map((p) => Number(p.gfs_forecast_rainfall_mm))
    .filter(Number.isFinite);

  if (!blendValues.length) return {};

  return {
    mean_astrablend_rainfall_mm: (
      blendValues.reduce((a, b) => a + b, 0) /
      blendValues.length
    ).toFixed(2),

    mean_ml_rainfall_mm: (
      mlValues.reduce((a, b) => a + b, 0) /
      mlValues.length
    ).toFixed(2),

    mean_gfs_rainfall_mm: (
      gfsValues.reduce((a, b) => a + b, 0) /
      gfsValues.length
    ).toFixed(2),

    max_astrablend_rainfall_mm:
      Math.max(...blendValues).toFixed(2),
  };
}, [forecastPoints]);

  const selectedRainfall =
    forecast?.astrablend_forecast_rainfall_mm;

  const selectedColor = useMemo(
    () => forecastColor(selectedRainfall || 0),
    [selectedRainfall]
  );

  const liveWarning =
    forecast?.rainfall_alert ??
    forecast?.heavy_rain_warning ??
    false;

  const liveProbability =
    forecast?.heavy_rain_probability_percent ?? 0;

  const activeDemo = demoScenario
    ? DEMO_SCENARIOS[demoScenario]
    : null;

  const activeWarning =
    activeDemo
      ? activeDemo.severity !== "normal"
      : liveWarning;

  const activeProbability =
    activeDemo
      ? activeDemo.probability
      : liveProbability;

  const activeRainfall =
    activeDemo
      ? activeDemo.rainfall
      : selectedRainfall;

  const activeSeverity =
    rainfallSeverity(activeRainfall);

  const warningTitle =
    activeDemo
      ? activeDemo.title
      : liveWarning
      ? "Elevated rainfall risk detected"
      : "Low heavy-rain risk";

  const warningDescription =
    activeDemo
      ? activeDemo.description
      : forecast?.alert_message
      ? forecast.alert_message
      : liveWarning
      ? "The AstraBlend hazard classifier has detected elevated heavy-rain probability."
      : "No heavy-rain warning is currently triggered for this location.";

  return (
    <div className="app">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />
      <div className="ambient ambient-three" />

      <header className="header">
        <div className="brand">
          <div className="brand-mark">
            <span>✦</span>
          </div>

          <div>
            <h1>AstraBlend</h1>
            <p>AI × NWP Weather Intelligence</p>
          </div>
        </div>

        <div className="header-actions">
          {forecastDate && (
            <div className="date-pill">
              <span>Forecast</span>
              <strong>
                {formatDate(forecastDate)}
              </strong>
            </div>
          )}

          <div className="status">
            <span className="status-dot" />
            LIVE SYSTEM
          </div>
        </div>
      </header>

      <main className="container">

        {/* HERO */}

        <section className="hero">
          <div className="hero-copy">
            <div className="hero-badge">
              <span className="hero-badge-dot" />
              HYBRID WEATHER INTELLIGENCE
            </div>

            <h2>
              Tomorrow’s rain,
              <span> predicted smarter.</span>
            </h2>

            <p className="subtitle">
              AstraBlend combines machine learning
              with GFS numerical weather prediction
              to create smarter rainfall forecasts
              and early-warning intelligence for India.
            </p>

            <div className="hero-tags">
              <span>60% ML</span>
              <span>40% GFS</span>
              <span>0.25° Grid</span>
              <span>Heavy-rain Risk</span>
            </div>
          </div>

          <div className="hero-visual">
            <div className="weather-orb">
              <div className="orb-core">
                <span className="orb-icon">☁</span>

                <strong>
                  {mapSummary.mean_astrablend_rainfall_mm ??
                    "--"}
                </strong>

                <small>mm avg rainfall</small>
              </div>

              <div className="orbit orbit-one" />
              <div className="orbit orbit-two" />
            </div>

            <div className="floating-chip chip-one">
              <span>ML</span>
              {mapSummary.mean_ml_rainfall_mm ??
                "--"}{" "}
              mm
            </div>

            <div className="floating-chip chip-two">
              <span>GFS</span>
              {mapSummary.mean_gfs_rainfall_mm ??
                "--"}{" "}
              mm
            </div>

            <div className="floating-chip chip-three">
              <span>MAX</span>
              {mapSummary.max_astrablend_rainfall_mm ??
                "--"}{" "}
              mm
            </div>
          </div>
        </section>

        {/* CITY CHIPS */}

        <section className="quick-cities">
          <div className="section-heading">
            <div>
              <span className="section-kicker">
                QUICK EXPLORE
              </span>

              <h3>Jump to a city</h3>
            </div>

            <p>
              Or click anywhere on the map.
            </p>
          </div>

          <div className="city-chips">
            {CITIES.slice(0, 8).map((city) => (
              <button
                key={city.name}
                className={`city-chip ${
                  selectedCity === city.name
                    ? "active"
                    : ""
                }`}
                onClick={() =>
                  selectLocation(
                    city.latitude,
                    city.longitude,
                    city.name
                  )
                }
              >
                {city.name}
              </button>
            ))}
          </div>
        </section>

        {/* LIVE HOTSPOTS */}

        <section className="quick-cities">
          <div className="section-heading">
            <div>
              <span className="section-kicker">
                LIVE HOTSPOTS
              </span>

              <h3>
                Regional AstraBlend monitoring
              </h3>
            </div>

            <p>
              Click a hotspot to load its live
              ML + GFS blended forecast instantly.
            </p>
          </div>

          <div className="city-chips">
            {hotspotPoints.map((point) => (
              <button
                key={`${point.state}-${point.city}`}
                className={`city-chip ${
                  locationName === point.location
                    ? "active"
                    : ""
                }`}
                onClick={() =>
                  selectHotspot(point)
                }
              >
                {point.city}
                {point.risk_level !== "LOW"
                  ? ` · ${point.risk_level}`
                  : ""}
              </button>
            ))}

            {!hotspotPoints.length && (
              <span className="coordinates">
                Loading live hotspots...
              </span>
            )}
          </div>
        </section>

        {/* MAP */}

        <section className="map-card">
          <div className="map-header">
            <div>
              <span className="section-kicker">
                LIVE FORECAST MAP
              </span>

              <h3>
                India rainfall intelligence
              </h3>

              <p>
                Hybrid ML + GFS rainfall forecast
                across
                {mapMeta?.point_count
                  ? ` ${mapMeta.point_count.toLocaleString(
                      "en-IN"
                    )} grid points`
                  : " India"}
                .
              </p>
            </div>

            <div className="map-controls">
              <select
                value={selectedCity}
                onChange={handleCityChange}
                className="city-select"
              >
                <option value="">
                  Select a city
                </option>

                {CITIES.map((city) => (
                  <option
                    key={city.name}
                    value={city.name}
                  >
                    {city.name}
                  </option>
                ))}
              </select>

              <button
                className="map-button"
                onClick={getForecast}
                disabled={loading}
              >
                <span>
                  {loading ? "◌" : "✦"}
                </span>

                {loading
                  ? "Analysing..."
                  : "Generate Forecast"}
              </button>
            </div>
          </div>

          <div className="map-stat-strip">
            <div>
              <span>Forecast date</span>
              <strong>
                {formatDate(forecastDate)}
              </strong>
            </div>

            <div>
              <span>Mean blend</span>
              <strong>
                {mapSummary.mean_astrablend_rainfall_mm ??
                  "--"}{" "}
                mm
              </strong>
            </div>

            <div>
              <span>Peak rainfall</span>
              <strong>
                {mapSummary.max_astrablend_rainfall_mm ??
                  "--"}{" "}
                mm
              </strong>
            </div>

            <div>
              <span>Risk zones</span>
              <strong>
                {mapMeta?.heavy_rain_warning_count ??
                  "--"}
              </strong>
            </div>
          </div>

          <div className="map-wrapper">
            {mapLoading && (
              <div className="map-loading-overlay">
                <div className="loader-ring" />

                <strong>
                  Building AstraBlend map
                </strong>

                <span>
                  Combining ML + GFS forecasts...
                </span>
              </div>
            )}

            <MapContainer
              center={INDIA_CENTER}
              zoom={5}
              scrollWheelZoom={true}
              className="india-map"
            >
              <TileLayer
                attribution="&copy; OpenStreetMap contributors"
                url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
              />

              <LocationPicker
                onSelect={selectLocation}
              />

              <MapRecenter
                latitude={latitude}
                longitude={longitude}
              />

              {forecastPoints.map(
                (point, index) => {
                  const rainfall = Number(
                    point.astrablend_forecast_rainfall_mm ??
                      point.rainfall_mm ??
                      0
                  );

                  const color =
                    forecastColor(rainfall);

                  return (
                    <Circle
                      key={`${point.latitude}-${point.longitude}-${index}`}
                      center={[
                        point.latitude,
                        point.longitude,
                      ]}
                      radius={
                        forecastRadius(
                          rainfall
                        ) * 1000
                      }
                      pathOptions={{
                        color,
                        fillColor: color,
                        fillOpacity: 0.58,
                        weight: 0,
                      }}
                    >
                      <Popup>
                        <div className="forecast-popup">
                          <div className="popup-heading">
                            <span>✦</span>

                            <div>
                              <strong>
                                AstraBlend
                                Forecast
                              </strong>

                              <small>
                                {point.latitude.toFixed(
                                  2
                                )}
                                °,{" "}
                                {point.longitude.toFixed(
                                  2
                                )}
                                °
                              </small>
                            </div>
                          </div>

                          <div className="popup-primary">
                            <strong>
                              {rainfall.toFixed(
                                2
                              )}
                            </strong>
                            <span>mm</span>
                          </div>

                          <div className="popup-grid">
                            <div>
                              <span>ML</span>

                              <strong>
                                {Number(
                                  point.ml_forecast_rainfall_mm ??
                                    0
                                ).toFixed(
                                  2
                                )}{" "}
                                mm
                              </strong>
                            </div>

                            <div>
                              <span>GFS</span>

                              <strong>
                                {Number(
                                  point.gfs_forecast_rainfall_mm ??
                                    0
                                ).toFixed(
                                  2
                                )}{" "}
                                mm
                              </strong>
                            </div>
                          </div>

                          <div className="popup-risk">
                            <span>
                              Heavy-rain risk
                            </span>

                            <strong>
                              {Number(
                                point.heavy_rain_probability_percent ??
                                  0
                              ).toFixed(
                                1
                              )}
                              %
                            </strong>
                          </div>
                        </div>
                      </Popup>
                    </Circle>
                  );
                }
              )}

              {hotspotPoints.map((point) => {
                const rainfall = Number(
                  point.astrablend_mm || 0
                );

                const color =
                  forecastColor(rainfall);

                return (
                  <Circle
                    key={`hotspot-${point.state}-${point.city}`}
                    center={[
                      point.latitude,
                      point.longitude,
                    ]}
                    radius={
                      forecastRadius(rainfall) *
                      1300
                    }
                    pathOptions={{
                      color,
                      fillColor: color,
                      fillOpacity: 0.82,
                      weight: 2,
                    }}
                    eventHandlers={{
                      click: () =>
                        selectHotspot(point),
                    }}
                  >
                    <Popup>
                      <div className="forecast-popup">
                        <div className="popup-heading">
                          <span>✦</span>

                          <div>
                            <strong>
                              {point.city},{" "}
                              {point.state}
                            </strong>

                            <small>
                              LIVE ASTRABLEND HOTSPOT
                            </small>
                          </div>
                        </div>

                        <div className="popup-primary">
                          <strong>
                            {Number(
                              point.astrablend_mm
                            ).toFixed(2)}
                          </strong>
                          <span>mm</span>
                        </div>

                        <div className="popup-grid">
                          <div>
                            <span>ML</span>
                            <strong>
                              {Number(
                                point.ml_forecast_mm
                              ).toFixed(2)}{" "}
                              mm
                            </strong>
                          </div>

                          <div>
                            <span>GFS</span>
                            <strong>
                              {Number(
                                point.gfs_forecast_mm
                              ).toFixed(2)}{" "}
                              mm
                            </strong>
                          </div>
                        </div>

                        <div className="popup-risk">
                          <span>
                            {
                              point.rainfall_category
                            }
                          </span>

                          <strong>
                            {point.risk_level}
                          </strong>
                        </div>
                      </div>
                    </Popup>
                  </Circle>
                );
              })}

              <Circle
                center={[
                  latitude,
                  longitude,
                ]}
                radius={13500}
                pathOptions={{
                  color: forecast
                    ? selectedColor
                    : "#ffffff",
                  fillColor: "#ffffff",
                  fillOpacity: 0.92,
                  weight: 4,
                }}
              >
                <Popup>
                  <strong>
                    {locationName}
                  </strong>

                  <br />

                  {latitude.toFixed(2)}°,{" "}
                  {longitude.toFixed(2)}°

                  {forecast && (
                    <>
                      <br />
                      <br />
                      AstraBlend:{" "}
                      <strong>
                        {
                          forecast.astrablend_forecast_rainfall_mm
                        }{" "}
                        mm
                      </strong>
                    </>
                  )}
                </Popup>
              </Circle>
            </MapContainer>
          </div>

          <div className="legend">
            <span className="legend-title">
              Rainfall intensity
            </span>

            <span>
              <i className="legend-box low" />
              &lt; 2.5
            </span>

            <span>
              <i className="legend-box moderate" />
              2.5–10
            </span>

            <span>
              <i className="legend-box medium" />
              10–25
            </span>

            <span>
              <i className="legend-box high" />
              25–50
            </span>

            <span>
              <i className="legend-box very-high" />
              50–100
            </span>

            <span>
              <i className="legend-box extreme" />
              &gt; 100 mm
            </span>
          </div>
        </section>

        {error && (
          <div className="error">
            <span>!</span>

            <div>
              <strong>
                Something needs attention
              </strong>

              <p>{error}</p>
            </div>
          </div>
        )}

        {/* LOCATION FORECAST */}

        <section className="forecast-section">
          <div className="section-heading result-heading">
            <div>
              <span className="section-kicker">
                LOCATION INTELLIGENCE
              </span>

              <h3>{locationName}</h3>
            </div>

            <div className="coordinates">
              {latitude.toFixed(2)}° N ·{" "}
              {longitude.toFixed(2)}° E
            </div>
          </div>

          <div className="forecast-comparison">

            <div className="model-card ml-card">
              <div className="model-card-top">
                <div className="model-icon">
                  AI
                </div>

                <span className="model-badge">
                  MACHINE LEARNING
                </span>
              </div>

              <div className="model-value">
                <strong>
                  {forecast
                    ? forecast.ml_forecast_rainfall_mm
                    : "--"}
                </strong>

                <span>mm</span>
              </div>

              <p>
                Learns rainfall patterns from
                observed IMD data.
              </p>
            </div>

            <div className="plus-symbol">
              +
            </div>

            <div className="model-card gfs-card">
              <div className="model-card-top">
                <div className="model-icon">
                  NWP
                </div>

                <span className="model-badge">
                  GFS MODEL
                </span>
              </div>

              <div className="model-value">
                <strong>
                  {forecast
                    ? forecast.gfs_forecast_rainfall_mm
                    : "--"}
                </strong>

                <span>mm</span>
              </div>

              <p>
                Physics-based numerical weather
                prediction.
              </p>
            </div>

            <div className="equals-symbol">
              =
            </div>

            <div className="model-card blend-card">
              <div className="blend-glow" />

              <div className="model-card-top">
                <div className="model-icon blend-icon">
                  ✦
                </div>

                <span className="model-badge">
                  ASTRABLEND
                </span>
              </div>

              <div className="model-value">
                <strong>
                  {forecast
                    ? forecast.astrablend_forecast_rainfall_mm
                    : "--"}
                </strong>

                <span>mm</span>
              </div>

              <p>
                Final hybrid forecast using
                60% ML + 40% GFS.
              </p>

              <div className="final-tag">
                FINAL FORECAST
              </div>
            </div>
          </div>

          <ModelComparisonChart
            forecast={forecast}
            locationName={locationName}
          />

          {/* RAINFALL SEVERITY */}

          <div className="severity-row">
            <div>
              <span className="severity-label">
                FORECAST CLASSIFICATION
              </span>

              <strong>
                Rainfall severity
              </strong>
            </div>

            <div
              className={`severity-badge ${activeSeverity.className}`}
            >
              {activeSeverity.label}
            </div>
          </div>

          {/* REAL / DEMO WARNING */}

          <div
            className={`risk-panel ${
              activeWarning
                ? activeDemo?.severity ===
                  "extreme"
                  ? "extreme-danger"
                  : "danger"
                : "safe"
            }`}
          >
            <div className="risk-left">
              <div className="risk-icon">
                {activeDemo
                  ? activeDemo.icon
                  : activeWarning
                  ? "!"
                  : "✓"}
              </div>

              <div>
                <div className="warning-title-line">
                  <span className="risk-kicker">
                    HEAVY-RAIN EARLY WARNING
                  </span>

                  {activeDemo && (
                    <span className="demo-label">
                      DEMO SCENARIO
                    </span>
                  )}

                  {!activeDemo && (
                    <span className="live-label">
                      LIVE MODEL
                    </span>
                  )}
                </div>

                <h3>{warningTitle}</h3>

                <p>
                  {warningDescription}
                </p>

                {activeDemo && (
                  <p className="demo-disclaimer">
                    Simulation only — this is not
                    the current AstraBlend forecast.
                  </p>
                )}
              </div>
            </div>

            <div className="risk-score">
              <span>Risk probability</span>

              <strong>
                {activeProbability
                  ? `${Number(
                      activeProbability
                    ).toFixed(1)}%`
                  : "--"}
              </strong>

              <RiskGauge
                value={activeProbability}
              />
            </div>
          </div>
        </section>

        {/* JUDGE DEMO CONTROLS */}

        <section className="demo-section">
          <div className="section-heading">
            <div>
              <span className="section-kicker">
                JUDGE DEMONSTRATION
              </span>

              <h3>
                Preview the warning system
              </h3>
            </div>

            <p>
              Simulate alert conditions without
              changing the actual forecast.
            </p>
          </div>

          <div className="demo-toolbar">
            <button
              className={`demo-button live ${
                !demoScenario ? "active" : ""
              }`}
              onClick={() =>
                setDemoScenario(null)
              }
            >
              <span>●</span>
              Live Forecast
            </button>

            <button
              className={`demo-button normal ${
                demoScenario === "normal"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setDemoScenario("normal")
              }
            >
              ✓ Normal
            </button>

            <button
              className={`demo-button heavy ${
                demoScenario === "heavy"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setDemoScenario("heavy")
              }
            >
              ⚠ Heavy Rain
            </button>

            <button
              className={`demo-button extreme ${
                demoScenario === "extreme"
                  ? "active"
                  : ""
              }`}
              onClick={() =>
                setDemoScenario("extreme")
              }
            >
              !! Extreme Rain
            </button>
          </div>

          <div className="demo-explanation">
            <span>i</span>

            <p>
              Demo mode changes only the warning
              preview above. It does not modify the
              live AstraBlend forecast, map, API
              response or model output.
            </p>
          </div>
        </section>

        {/* REAL IMD REFERENCE EXAMPLES */}

        <section className="reference-section">
          <div className="section-heading">
            <div>
              <span className="section-kicker">
                REAL-WORLD REFERENCE
              </span>

              <h3>
                Official IMD warning examples
              </h3>
            </div>

            <div className="imd-badge">
              IMD · 30 AUG 2026
            </div>
          </div>

          <div className="reference-disclaimer">
            <strong>
              External verification examples
            </strong>

            <span>
              These are official IMD warnings and
              are displayed as real-world reference
              cases. They are not AstraBlend model
              predictions.
            </span>
          </div>

          <div className="reference-grid">
            {IMD_REFERENCE_ALERTS.map(
              (alert) => (
                <div
                  key={alert.region}
                  className={`reference-card ${alert.type}`}
                >
                  <div className="reference-top">
                    <span className="reference-source">
                      IMD
                    </span>

                    <span className="reference-date">
                      {alert.date}
                    </span>
                  </div>

                  <h4>
                    {alert.region}
                  </h4>

                  <div className="reference-level">
                    {alert.level}
                  </div>
                </div>
              )
            )}
          </div>

          <p className="reference-note">
            Reference warnings verified from India
            Meteorological Department publications
            for 30 August 2026.
          </p>
        </section>

        {/* VALIDATION */}

        <section className="validation-section">
          <div className="validation-heading">
            <div>
              <span className="section-kicker">
                2024 MONSOON VALIDATION
              </span>

              <h3>
                Why blend ML with GFS?
              </h3>

              <p>
                AstraBlend was evaluated on common
                IMD-GFS grid points before selecting
                the current 60/40 continuous rainfall
                blend.
              </p>
            </div>

            <div className="validation-formula">
              <span>ASTRABLEND</span>

              <strong>
                0.60 × ML + 0.40 × GFS
              </strong>
            </div>
          </div>

          <div className="validation-grid">

            <div className="validation-card highlight">
              <span>ASTRABLEND MAE</span>

              <strong>
                7.595
              </strong>

              <small>mm</small>

              <p>
                Lower continuous rainfall error.
              </p>
            </div>

            <div className="validation-card">
              <span>VS GFS</span>

              <strong>
                7.51%
              </strong>

              <small>lower MAE</small>

              <p>
                Compared with GFS MAE of
                8.211 mm.
              </p>
            </div>

            <div className="validation-card">
              <span>DAILY WINS</span>

              <strong>
                22/31
              </strong>

              <small>days</small>

              <p>
                AstraBlend achieved the lowest
                daily MAE most often.
              </p>
            </div>

            <div className="validation-card">
              <span>VALIDATION SAMPLE</span>

              <strong>
                137,454
              </strong>

              <small>forecasts</small>

              <p>
                Common grid-point forecasts used
                in the validation sample.
              </p>
            </div>

          </div>

          <div className="validation-note">
            <span>Important</span>

            <p>
              60/40 is a validation-selected blend.
              Heavy-rain warnings are handled by a
              separate hazard classifier rather than
              claiming that the continuous blend
              improves extreme-event detection.
            </p>
          </div>
        </section>

        {/* SYSTEM INFORMATION */}

        <section className="system-grid">
          <div className="system-card">
            <span className="system-icon">
              ◉
            </span>

            <div>
              <small>
                Forecast target
              </small>

              <strong>
                {formatDate(
                  forecast?.forecast_target_date ||
                    mapMeta?.forecast_target_date
                )}
              </strong>
            </div>
          </div>

          <div className="system-card">
            <span className="system-icon">
              ↗
            </span>

            <div>
              <small>
                ML source date
              </small>

              <strong>
                {formatDate(
                  forecast?.ml_source_date ||
                    mapMeta?.ml_source_date
                )}
              </strong>
            </div>
          </div>

          <div className="system-card">
            <span className="system-icon">
              ◫
            </span>

            <div>
              <small>
                IMD grid
              </small>

              <strong>
                {forecast
                  ? `${forecast.imd_grid_latitude}, ${forecast.imd_grid_longitude}`
                  : "0.25° India"}
              </strong>
            </div>
          </div>

          <div className="system-card">
            <span className="system-icon">
              ✦
            </span>

            <div>
              <small>
                Forecast model
              </small>

              <strong>
                {forecast?.model ||
                  mapMeta?.model ||
                  "AstraBlend v1.3"}
              </strong>
            </div>
          </div>
        </section>

        <footer className="footer">
          <div className="footer-brand">
            <span>✦</span>
            AstraBlend
          </div>

          <p>
            Hybrid AI-NWP rainfall intelligence
            for smarter weather decisions.
          </p>

          <div className="footer-status">
            <span />
            System operational
          </div>
        </footer>
      </main>
    </div>
  );
}

export default App;