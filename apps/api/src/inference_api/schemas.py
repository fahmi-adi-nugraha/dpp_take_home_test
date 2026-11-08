"""Request and response schemas for the inference API."""

from __future__ import annotations

from typing import Literal

from delivery_time.inference import DeliveryTimeInput
from pydantic import BaseModel, ConfigDict, Field

WeatherCategory = Literal["Cloudy", "Fog", "Sandstorms", "Stormy", "Sunny", "Windy"]
TrafficCategory = Literal["Low", "Medium", "High", "Jam"]
CityCategory = Literal["Metropolitan", "Semi-Urban", "Urban", "Metropolitian"]
FestivalCategory = Literal["Yes", "No"]


class GeoPoint(BaseModel):
    """Latitude and longitude pair."""

    model_config = ConfigDict(extra="forbid")

    latitude: float = Field(ge=-90, le=90, examples=[12.9716])
    longitude: float = Field(ge=-180, le=180, examples=[77.5946])


class DeliveryTimeRequest(BaseModel):
    """Raw order, location, and courier inputs for delivery-time inference."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "restaurant": {"latitude": 12.9716, "longitude": 77.5946},
                "delivery": {"latitude": 12.9352, "longitude": 77.6245},
                "order_time": "12:30",
                "weather": "Sunny",
                "traffic": "Medium",
                "city": "Urban",
                "festival": "No",
                "multiple_deliveries": 1,
                "vehicle_condition": 1,
                "courier_age": 30,
                "courier_rating": 4.7,
            }
        },
    )

    restaurant: GeoPoint | None = Field(
        default=None,
        description="Restaurant location. Missing or invalid coordinates produce a warning.",
    )
    delivery: GeoPoint | None = Field(
        default=None,
        description="Delivery location. Missing or invalid coordinates produce a warning.",
    )
    order_time: str | None = Field(
        default=None,
        pattern=r"^\d{1,2}:\d{2}$",
        description="Order placement time in HH:MM format.",
        examples=["12:30"],
    )
    weather: WeatherCategory | None = Field(
        default=None,
        description="Weather condition at delivery time.",
    )
    traffic: TrafficCategory | None = Field(
        default=None,
        description="Road traffic density.",
    )
    city: CityCategory | None = Field(
        default=None,
        description="City or area type.",
    )
    festival: FestivalCategory | None = Field(
        default=None,
        description="Whether the order coincided with a festival.",
    )
    multiple_deliveries: int | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Number of deliveries in the same trip.",
    )
    vehicle_condition: int | None = Field(
        default=None,
        ge=0,
        le=3,
        description="Vehicle condition score from 0 to 3.",
    )
    courier_age: float | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Courier age in years.",
    )
    courier_rating: float | None = Field(
        default=None,
        ge=0,
        le=10,
        description="Courier rating. Out-of-range values are passed through to the model.",
    )

    def to_inference_input(self) -> DeliveryTimeInput:
        """Convert the API payload into the model package input dataclass."""
        return DeliveryTimeInput(
            restaurant_latitude=(self.restaurant.latitude if self.restaurant is not None else None),
            restaurant_longitude=(
                self.restaurant.longitude if self.restaurant is not None else None
            ),
            delivery_latitude=(self.delivery.latitude if self.delivery is not None else None),
            delivery_longitude=(self.delivery.longitude if self.delivery is not None else None),
            order_time=self.order_time,
            weather=self.weather,
            traffic=self.traffic,
            city=self.city,
            festival=self.festival,
            multiple_deliveries=self.multiple_deliveries,
            vehicle_condition=self.vehicle_condition,
            courier_age=self.courier_age,
            courier_rating=self.courier_rating,
        )


class DeliveryTimeResponse(BaseModel):
    """Delivery-time prediction and model metadata."""

    prediction_minutes: float
    model: str
    model_version: str | None = None
    selected_feature_set: str | None = None
    prediction_point: Literal["order_placement"] = "order_placement"
    warnings: list[str] = Field(default_factory=list)


class ModelStatusResponse(BaseModel):
    """Readiness information for one model family."""

    model: str
    loaded: bool
    path: str
    error: str | None = None
    model_version: str | None = None
    selected_feature_set: str | None = None


class HealthResponse(BaseModel):
    """Service health and model readiness."""

    status: Literal["ok", "degraded"]
    models: list[ModelStatusResponse]


class ModelsResponse(BaseModel):
    """List of model statuses."""

    models: list[ModelStatusResponse]


class SentimentRequest(BaseModel):
    """Review text for sentiment inference."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "text": "The food was great and the delivery was fast.",
            }
        },
    )

    text: str = Field(min_length=1, max_length=10_000)


class SentimentResponse(BaseModel):
    """Five-class sentiment prediction and model metadata."""

    label: int
    label_name: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    model_version: str | None = None
