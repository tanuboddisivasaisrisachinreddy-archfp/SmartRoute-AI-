"""Location geocoding (built-in landmarks + Nominatim, bounded to Hyderabad)."""
from backend.app.services.geocoding import Geocoder
from reinforcement_learning.environment.landmarks import seed_landmarks

def test_geocoder_finds_hyderabad_landmarks():
    g = Geocoder(seed_landmarks(), enabled=False)
    r = g.search("charmi")
    assert r["results"][0]["name"] == "Charminar"
    assert g.search("airport")["results"][0]["name"].startswith("Hyderabad Airport")
    assert g.search("hitec")["results"][0]["id"] in ("hitech-city", "hitec-city-metro")


def test_geocoder_nominatim_bounded_to_hyderabad(monkeypatch):
    calls = {}

    class Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return [{"name": "Inorbit Mall", "display_name": "Inorbit Mall, Madhapur, Hyderabad", "lat": "17.4344",
                     "lon": "78.3866", "type": "mall"}]

    def fake_get(url, params, headers, timeout):
        calls.update(params=params, headers=headers)
        return Resp()

    monkeypatch.setattr("backend.app.services.geocoding.requests.get", fake_get)
    g = Geocoder(seed_landmarks(), enabled=True)
    r = g.search("Inorbit Mall")
    assert any(x["source"] == "nominatim" for x in r["results"])
    assert calls["params"]["bounded"] == 1 and calls["params"]["countrycodes"] == "in"
    assert "Hyderabad" in calls["params"]["q"]
    assert "SmartRouteAI" in calls["headers"]["User-Agent"]


