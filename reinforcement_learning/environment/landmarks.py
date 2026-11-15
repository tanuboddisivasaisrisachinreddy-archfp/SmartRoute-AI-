"""Hyderabad locations supported out of the box.

``seed`` coordinates are approximate WGS84 positions.  During preprocessing
``scripts/download_data.py --landmarks`` re-geocodes every name through
OpenStreetMap Nominatim; results are accepted when they fall within 3 km of
the seed (guards against wrong matches) and written, together with the
nearest road-graph intersection, to ``data/processed/landmarks.json``.
"""
from __future__ import annotations

LANDMARKS = [
    # id, display name, Nominatim query, seed lat, seed lon, category
    ("hitech-city", "Hitech City", "HITEC City, Hyderabad", 17.4504, 78.3810, "IT hub"),
    ("madhapur", "Madhapur", "Madhapur, Hyderabad", 17.4483, 78.3915, "Neighbourhood"),
    ("gachibowli", "Gachibowli", "Gachibowli, Hyderabad", 17.4401, 78.3489, "IT hub"),
    ("kondapur", "Kondapur", "Kondapur, Hyderabad", 17.4699, 78.3570, "Neighbourhood"),
    ("jubilee-hills", "Jubilee Hills", "Jubilee Hills, Hyderabad", 17.4300, 78.4090, "Neighbourhood"),
    ("banjara-hills", "Banjara Hills", "Banjara Hills, Hyderabad", 17.4156, 78.4347, "Neighbourhood"),
    ("ameerpet", "Ameerpet", "Ameerpet, Hyderabad", 17.4375, 78.4482, "Commercial"),
    ("kukatpally", "Kukatpally", "Kukatpally, Hyderabad", 17.4948, 78.3996, "Neighbourhood"),
    ("secunderabad", "Secunderabad", "Secunderabad Railway Station", 17.4337, 78.5016, "Transit"),
    ("begumpet", "Begumpet", "Begumpet, Hyderabad", 17.4447, 78.4664, "Neighbourhood"),
    ("lb-nagar", "LB Nagar", "L. B. Nagar, Hyderabad", 17.3457, 78.5522, "Neighbourhood"),
    ("uppal", "Uppal", "Uppal, Hyderabad", 17.4058, 78.5591, "Neighbourhood"),
    ("miyapur", "Miyapur", "Miyapur, Hyderabad", 17.4968, 78.3614, "Neighbourhood"),
    ("charminar", "Charminar", "Charminar, Hyderabad", 17.3616, 78.4747, "Heritage"),
    ("mehdipatnam", "Mehdipatnam", "Mehdipatnam, Hyderabad", 17.3950, 78.4400, "Commercial"),
    ("raidurg", "Raidurg", "Raidurg, Hyderabad", 17.4295, 78.3834, "Neighbourhood"),
    ("financial-district", "Financial District", "Financial District, Nanakramguda, Hyderabad",
     17.4155, 78.3404, "IT hub"),
    ("hitec-city-metro", "HITEC City Metro Station", "HITEC City metro station, Hyderabad",
     17.4474, 78.3827, "Transit"),
    ("rgia-airport", "Hyderabad Airport (RGIA)", "Rajiv Gandhi International Airport, Hyderabad",
     17.2403, 78.4294, "Airport"),
]


def seed_landmarks() -> list[dict]:
    return [
        {"id": i, "name": n, "query": q, "lat": la, "lon": lo, "category": c, "source": "seed"}
        for i, n, q, la, lo, c in LANDMARKS
    ]
