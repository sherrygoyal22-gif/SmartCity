"""City / state -> coordinates (and back) for the admin office location.

Tries OpenStreetMap Nominatim first. If the server has no internet, or
nothing is found, falls back to a built-in table of Indian state/UT capitals
so picking a state always places the pin somewhere sensible.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

USER_AGENT = "SmartCityGarbageDetection/1.0 (admin location lookup)"

# state / UT -> (capital, latitude, longitude)
STATE_CAPITALS = {
    "Andhra Pradesh": ("Amaravati", 16.5730, 80.3575),
    "Arunachal Pradesh": ("Itanagar", 27.0844, 93.6053),
    "Assam": ("Dispur", 26.1433, 91.7898),
    "Bihar": ("Patna", 25.5941, 85.1376),
    "Chhattisgarh": ("Raipur", 21.2514, 81.6296),
    "Goa": ("Panaji", 15.4909, 73.8278),
    "Gujarat": ("Gandhinagar", 23.2156, 72.6369),
    "Haryana": ("Chandigarh", 30.7333, 76.7794),
    "Himachal Pradesh": ("Shimla", 31.1048, 77.1734),
    "Jharkhand": ("Ranchi", 23.3441, 85.3096),
    "Karnataka": ("Bengaluru", 12.9716, 77.5946),
    "Kerala": ("Thiruvananthapuram", 8.5241, 76.9366),
    "Madhya Pradesh": ("Bhopal", 23.2599, 77.4126),
    "Maharashtra": ("Mumbai", 19.0760, 72.8777),
    "Manipur": ("Imphal", 24.8170, 93.9368),
    "Meghalaya": ("Shillong", 25.5788, 91.8933),
    "Mizoram": ("Aizawl", 23.7271, 92.7176),
    "Nagaland": ("Kohima", 25.6751, 94.1086),
    "Odisha": ("Bhubaneswar", 20.2961, 85.8245),
    "Punjab": ("Chandigarh", 30.7333, 76.7794),
    "Rajasthan": ("Jaipur", 26.9124, 75.7873),
    "Sikkim": ("Gangtok", 27.3389, 88.6065),
    "Tamil Nadu": ("Chennai", 13.0827, 80.2707),
    "Telangana": ("Hyderabad", 17.3850, 78.4867),
    "Tripura": ("Agartala", 23.8315, 91.2868),
    "Uttar Pradesh": ("Lucknow", 26.8467, 80.9462),
    "Uttarakhand": ("Dehradun", 30.3165, 78.0322),
    "West Bengal": ("Kolkata", 22.5726, 88.3639),
    "Andaman and Nicobar Islands": ("Port Blair", 11.6234, 92.7265),
    "Chandigarh": ("Chandigarh", 30.7333, 76.7794),
    "Dadra and Nagar Haveli and Daman and Diu": ("Daman", 20.3974, 72.8328),
    "Delhi": ("New Delhi", 28.6139, 77.2090),
    "Jammu and Kashmir": ("Srinagar", 34.0837, 74.7973),
    "Ladakh": ("Leh", 34.1526, 77.5771),
    "Lakshadweep": ("Kavaratti", 10.5626, 72.6369),
    "Puducherry": ("Puducherry", 11.9416, 79.8083),
}
_STATE_LOOKUP = {name.lower(): name for name in STATE_CAPITALS}
_STATE_LOOKUP.update({"nct of delhi": "Delhi", "orissa": "Odisha", "uttaranchal": "Uttarakhand"})


def canonical_state(name: str | None) -> str | None:
    return _STATE_LOOKUP.get((name or "").strip().lower())


def _get_json(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=6) as response:
        return json.loads(response.read().decode("utf-8"))


def forward(city: str = "", state: str = ""):
    """Return {lat, lng, label, source} for a city and/or state, or None."""
    city, state = (city or "").strip(), (state or "").strip()
    if not city and not state:
        return None
    known_state = canonical_state(state)
    parts = [part for part in (city, known_state or state) if part]
    if known_state:
        parts.append("India")
    try:
        data = _get_json(
            "https://nominatim.openstreetmap.org/search?"
            + urllib.parse.urlencode({"format": "jsonv2", "limit": 1, "q": ", ".join(parts)})
        )
        if data:
            return {
                "lat": round(float(data[0]["lat"]), 6),
                "lng": round(float(data[0]["lon"]), 6),
                "label": data[0].get("display_name", ""),
                "source": "openstreetmap",
            }
    except Exception:
        pass
    if known_state and not city:
        capital, lat, lng = STATE_CAPITALS[known_state]
        return {"lat": lat, "lng": lng, "label": f"{capital}, {known_state}", "source": "offline"}
    if known_state and city.lower() == STATE_CAPITALS[known_state][0].lower():
        capital, lat, lng = STATE_CAPITALS[known_state]
        return {"lat": lat, "lng": lng, "label": f"{capital}, {known_state}", "source": "offline"}
    return None


def reverse(latitude: float, longitude: float):
    """Return {city, state} for coordinates, or None when it cannot be found."""
    try:
        data = _get_json(
            "https://nominatim.openstreetmap.org/reverse?"
            + urllib.parse.urlencode(
                {"format": "jsonv2", "lat": latitude, "lon": longitude, "zoom": 10, "addressdetails": 1}
            )
        )
        address = data.get("address", {})
        city = (
            address.get("city") or address.get("town") or address.get("village")
            or address.get("municipality") or address.get("county") or address.get("state_district") or ""
        )
        state = address.get("state") or address.get("union_territory") or ""
        if city or state:
            return {"city": city, "state": canonical_state(state) or state}
    except Exception:
        pass
    return None
