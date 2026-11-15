"""Route evaluation under identical, time-dependent traffic conditions."""
from __future__ import annotations

import os

from ..environment.graph import RoadGraph
from ..environment.route_env import PREFERENCES, RewardConfig, RouteEnv, edge_cost
from ..environment.traffic import TrafficModel

# Documented cost model assumptions (override via env vars)
FUEL_PRICE_INR_PER_L = float(os.getenv("FUEL_PRICE_INR_PER_L", 116.0))
MILEAGE_KM_PER_L = float(os.getenv("MILEAGE_KM_PER_L", 15.0))
CONGESTION_FUEL_PENALTY = 0.6  # +60% consumption at full congestion (stop-and-go)


def congestion_label(c: float) -> str:
    if c < 0.2:
        return "Low"
    if c < 0.4:
        return "Moderate"
    if c < 0.6:
        return "Heavy"
    return "Severe"


def evaluate_path(graph: RoadGraph, traffic: TrafficModel, edges: list[int], source: int, goal: int,
                  t0: float, cfg: RewardConfig | None = None) -> dict:
    """Drive the path through the environment, so travel time, congestion and
    reward are all computed with the same time-dependent traffic model."""
    cfg = cfg or PREFERENCES["fastest"]
    env = RouteEnv(graph, traffic, cfg, max_steps=max(10_000, len(edges) + 5))
    env.reset(source, goal, t0)
    tt_total = dist = cong_time = cost = fuel = 0.0
    revisits = 0
    for e in edges:
        if env.done:
            break
        _, _, _, info = env.step(e)
        if info.get("invalid"):
            raise ValueError(f"edge {e} is not connected to the current node")
        tt, c, ln = float(info["travel_time_s"]), float(info["congestion"]), float(graph.length_m[e])
        tt_total += tt
        dist += ln
        cong_time += c * tt
        cost += edge_cost(cfg, tt, ln, c)
        fuel += ln / 1000.0 / MILEAGE_KM_PER_L * (1 + CONGESTION_FUEL_PENALTY * c)
        revisits += int(info["revisit"])
    reached = env.node == goal
    exposure = cong_time / tt_total if tt_total > 0 else 0.0
    return {
        "reached_destination": bool(reached),
        "distance_km": round(dist / 1000.0, 3),
        "travel_time_min": round(tt_total / 60.0, 2),
        "congestion_exposure": round(exposure, 4),
        "congestion_level": congestion_label(exposure),
        "congested_minutes": round(cong_time / 60.0, 2),
        "segments": len(edges),
        "total_reward": round(float(env.total_reward), 3),
        "generalized_cost": round(cost, 3),
        "fuel_litres": round(fuel, 3),
        "estimated_cost_inr": round(fuel * FUEL_PRICE_INR_PER_L, 2),
        "revisits": revisits,
        "avg_speed_kph": round((dist / 1000.0) / (tt_total / 3600.0), 1) if tt_total > 0 else 0.0,
    }


def cost_model_doc() -> dict:
    return {
        "fuel_price_inr_per_litre": FUEL_PRICE_INR_PER_L,
        "mileage_km_per_litre": MILEAGE_KM_PER_L,
        "congestion_fuel_penalty": CONGESTION_FUEL_PENALTY,
        "formula": "cost_INR = sum_e len_e[km] / mileage * (1 + 0.6 * c_e) * fuel_price",
        "note": "Assumed petrol car; values configurable through FUEL_PRICE_INR_PER_L and MILEAGE_KM_PER_L.",
    }
