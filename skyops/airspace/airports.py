"""A small table of major airports under the OpenSky capture footprint (Indian subcontinent).

Used to (a) relax separation minima in terminal areas, (b) suppress low-altitude anomaly alerts
near runways and (c) label the map. Coordinates are approximate reference points.
"""
from __future__ import annotations

import math

AIRPORTS: list[dict] = [
    dict(icao="VIDP", iata="DEL", name="Delhi Indira Gandhi Intl", lat=28.5562, lon=77.1000),
    dict(icao="VABB", iata="BOM", name="Mumbai Chhatrapati Shivaji Intl", lat=19.0896, lon=72.8656),
    dict(icao="VOBL", iata="BLR", name="Bengaluru Kempegowda Intl", lat=13.1989, lon=77.7069),
    dict(icao="VOMM", iata="MAA", name="Chennai Intl", lat=12.9941, lon=80.1709),
    dict(icao="VOHS", iata="HYD", name="Hyderabad Rajiv Gandhi Intl", lat=17.2403, lon=78.4294),
    dict(icao="VECC", iata="CCU", name="Kolkata Netaji Subhas Chandra Bose Intl", lat=22.6547, lon=88.4467),
    dict(icao="VAAH", iata="AMD", name="Ahmedabad Sardar Vallabhbhai Patel Intl", lat=23.0772, lon=72.6347),
    dict(icao="VOCI", iata="COK", name="Kochi Cochin Intl", lat=10.1520, lon=76.4019),
    dict(icao="VAPO", iata="PNQ", name="Pune", lat=18.5821, lon=73.9197),
    dict(icao="VAGO", iata="GOI", name="Goa Dabolim", lat=15.3808, lon=73.8314),
    dict(icao="VOTV", iata="TRV", name="Thiruvananthapuram Intl", lat=8.4821, lon=76.9201),
    dict(icao="VEGT", iata="GAU", name="Guwahati Lokpriya Gopinath Bordoloi Intl", lat=26.1061, lon=91.5859),
    dict(icao="VILK", iata="LKO", name="Lucknow Chaudhary Charan Singh Intl", lat=26.7606, lon=80.8893),
    dict(icao="VIJP", iata="JAI", name="Jaipur Intl", lat=26.8242, lon=75.8122),
    dict(icao="VOCB", iata="CJB", name="Coimbatore Intl", lat=11.0300, lon=77.0434),
    dict(icao="VOVZ", iata="VTZ", name="Visakhapatnam", lat=17.7212, lon=83.2245),
    dict(icao="VABO", iata="BDQ", name="Vadodara", lat=22.3362, lon=73.2263),
    dict(icao="VANP", iata="NAG", name="Nagpur Dr. Babasaheb Ambedkar Intl", lat=21.0922, lon=79.0472),
    dict(icao="VOTR", iata="TRZ", name="Tiruchirappalli Intl", lat=10.7654, lon=78.7097),
    dict(icao="VIAR", iata="ATQ", name="Amritsar Sri Guru Ram Dass Jee Intl", lat=31.7096, lon=74.7973),
    dict(icao="VCBI", iata="CMB", name="Colombo Bandaranaike Intl", lat=7.1808, lon=79.8841),
    dict(icao="VGHS", iata="DAC", name="Dhaka Hazrat Shahjalal Intl", lat=23.8433, lon=90.3978),
    dict(icao="VNKT", iata="KTM", name="Kathmandu Tribhuvan Intl", lat=27.6966, lon=85.3591),
    dict(icao="OPKC", iata="KHI", name="Karachi Jinnah Intl", lat=24.9065, lon=67.1608),
    dict(icao="VYYY", iata="RGN", name="Yangon Intl", lat=16.9073, lon=96.1332),
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0088 * math.asin(math.sqrt(a))


def nearest_airport(lat: float, lon: float) -> tuple[dict, float]:
    """Return (airport, distance_km) of the closest airport in the table."""
    best, best_d = AIRPORTS[0], float("inf")
    for ap in AIRPORTS:
        d = haversine_km(lat, lon, ap["lat"], ap["lon"])
        if d < best_d:
            best, best_d = ap, d
    return best, best_d
