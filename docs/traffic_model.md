# Traffic model (simulation)

All values are **simulated** unless TomTom live data is configured. Implementation: `reinforcement_learning/environment/traffic.py`.

## Congestion of edge e at simulation minute t

```
c_e(t) = clip( L(t) · k_class(e) · h_e  +  0.10 · z_e(t)  +  Σ_incidents I_e(t),  0, 0.95 )
```

| Symbol | Meaning | Values |
|---|---|---|
| `L(t)` | network level = scenario base × time-of-day profile + surge | base: low 0.12, moderate 0.32, heavy 0.58, peak 0.78, random 0.38; "peak" fixes the profile at 1.0 |
| profile | weekday shape, peaks ~10:00 and ~18:45, night ≈ 0.3 | assumption |
| `k_class` | road-class sensitivity | motorway 0.55, trunk 1.0, primary 1.1, secondary 1.0, tertiary 0.9 |
| `h_e` | spatial factor: Gaussian bumps (σ 2.5 km) around assumed hotspots (Ameerpet, HITEC City, Kukatpally, Mehdipatnam, Secunderabad, LB Nagar, Charminar, Panjagutta, Gachibowli, Uppal) × seeded noise U(0.85, 1.15) | assumption |
| `z_e(t)` | Ornstein–Uhlenbeck noise, τ = 30 min | random fluctuation |
| `I_e(t)` | incident intensity × distance weight; full severity for its duration, then linear recovery | user-triggered / random |
| surge | `m · exp(−(t − t_s)/τ)` | sudden increase + recovery |

## Speed and travel time

```
v_e(t)  = max(4 km/h, v_free(e) · (1 − 0.85 · c_e(t)))
tt_e(t) = length_e / v_e(t)
```

Free-flow speeds (km/h): motorway 80, trunk 50, primary 40, secondary 32, tertiary 26 – or 0.8 × OSM maxspeed when tagged.

## Controls (API / Traffic page)
* scenario and intensity multiplier (`POST /api/traffic/simulate`)
* trigger incident at a point or at random (`POST /api/traffic/incident`), clear one/all (`POST /api/traffic/incident/clear`)
* sudden increase (`surge`), advance the clock (`POST /api/traffic/advance`), reset (`POST /api/traffic/reset`)

## Cost model
`cost_INR = Σ len_e[km] / mileage · (1 + 0.6 · c_e) · fuel_price` with mileage 15 km/L and petrol ₹116/L (Hyderabad retail level in autumn 2026). Both are assumptions, configurable through `MILEAGE_KM_PER_L` and `FUEL_PRICE_INR_PER_L`.
