import requests


url = (
    "https://nomads.ncep.noaa.gov/cgi-bin/"
    "filter_gfs_0p25.pl"
)

params = {
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

response = requests.get(
    url,
    params=params,
    timeout=120,
)

print("===== LIVE GFS TEST =====")
print("HTTP:", response.status_code)
print("URL:")
print(response.url)
print("Content type:", response.headers.get("Content-Type"))
print("Size:", len(response.content), "bytes")
print("First 8 bytes:", response.content[:8])