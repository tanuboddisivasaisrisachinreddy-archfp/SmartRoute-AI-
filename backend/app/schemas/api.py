"""Pydantic request / response schemas."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from reinforcement_learning.environment.config import HYDERABAD_BBOX

Preference = Literal["fastest", "shortest", "balanced"]
Scenario = Literal["current", "low", "moderate", "heavy", "peak", "random", "live"]
Event = Literal["none", "incident_on_route", "surge"]
Algorithm = Literal["dijkstra", "q_learning", "dqn"]

_M = 0.05  # tolerance (degrees) around the Hyderabad bounding box


class Point(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    name: str | None = None

    @field_validator("lat")
    @classmethod
    def _lat_in_hyd(cls, v):
        if not HYDERABAD_BBOX["south"] - _M <= v <= HYDERABAD_BBOX["north"] + _M:
            raise ValueError("latitude outside the Hyderabad coverage area")
        return v

    @field_validator("lon")
    @classmethod
    def _lon_in_hyd(cls, v):
        if not HYDERABAD_BBOX["west"] - _M <= v <= HYDERABAD_BBOX["east"] + _M:
            raise ValueError("longitude outside the Hyderabad coverage area")
        return v


class CompareRequest(BaseModel):
    source: Point
    destination: Point
    preference: Preference = "fastest"
    scenario: Scenario = "current"
    event: Event = "none"
    algorithms: list[Algorithm] = Field(default_factory=lambda: ["dijkstra", "q_learning", "dqn"], min_length=1)
    include_alternatives: bool = True


class OptimizeRequest(BaseModel):
    source: Point
    destination: Point
    preference: Preference = "fastest"
    scenario: Scenario = "current"
    algorithm: Algorithm | Literal["best"] = "best"


class SimulateRequest(BaseModel):
    scenario: Literal["low", "moderate", "heavy", "peak", "random"] | None = None
    intensity: float | None = Field(None, ge=0, le=2)
    advance_min: float | None = Field(None, ge=0, le=240)
    surge: float | None = Field(None, ge=0, le=0.8, description="sudden network-wide increase")
    clock: str | None = Field(None, pattern=r"^\d{1,2}:\d{2}$")


class IncidentRequest(BaseModel):
    lat: float | None = None
    lon: float | None = None
    radius_m: float = Field(700, ge=100, le=5000)
    severity: float = Field(0.6, ge=0.05, le=0.95)
    duration_min: float = Field(45, ge=1, le=600)
    recovery_min: float = Field(20, ge=0, le=240)
    label: str = "Road incident"
    random: bool = False


class ClearIncidentRequest(BaseModel):
    incident_id: int | None = None


class ResetRequest(BaseModel):
    scenario: Literal["low", "moderate", "heavy", "peak", "random"] = "moderate"
    clock: str = Field("09:30", pattern=r"^\d{1,2}:\d{2}$")


class TrainingRequest(BaseModel):
    algorithm: Literal["q_learning", "dqn"]
    episodes: int = Field(300, ge=10, le=20000)
    learning_rate: float | None = Field(None, gt=0, le=1)
    discount_factor: float | None = Field(None, gt=0, le=1)
    epsilon: float = Field(1.0, ge=0, le=1)
    batch_size: int = Field(128, ge=8, le=1024)
    buffer_size: int = Field(40_000, ge=1000, le=500_000)
    seed: int = 42
    preferences: list[Preference] = Field(default_factory=lambda: ["fastest"])
    goals: str | None = Field(None, description="comma-separated landmark ids, default all")


class ExperimentRequest(BaseModel):
    pairs: int = Field(8, ge=1, le=30)
    scenarios: list[Literal["low", "moderate", "heavy", "peak", "random"]] = Field(
        default_factory=lambda: ["moderate", "peak"])
    preferences: list[Preference] = Field(default_factory=lambda: ["fastest"])
    events: list[Event] = Field(default_factory=lambda: ["none", "incident_on_route"])
    seed: int = 0


class Message(BaseModel):
    detail: str
    data: dict[str, Any] | None = None
