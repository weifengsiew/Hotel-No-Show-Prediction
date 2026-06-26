"""Feature engineering nodes."""

from __future__ import annotations

import calendar

import numpy as np
import pandas as pd

REFERENCE_YEAR = 2024
MONTH_NUMBER_BY_NAME = {
    month: month_number
    for month_number, month in enumerate(calendar.month_name)
    if month
}


def parse_price_currency_and_amount(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add price_currency and price_amount columns."""
    engineered_data = feature_data.copy()
    parsed_price = engineered_data["price"].astype("string").str.extract(
        r"^\s*(?P<currency>SGD|USD)\$\s*(?P<amount>\d+(?:\.\d+)?)\s*$"
    )
    engineered_data["price_currency"] = parsed_price["currency"].astype("string")
    engineered_data["price_amount"] = pd.to_numeric(
        parsed_price["amount"], errors="coerce"
    )

    return engineered_data


def price_amount_usd_to_sgd(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add price_amount_sgd by converting USD prices to SGD."""
    engineered_data = feature_data.copy()
    is_sgd = engineered_data["price_currency"].eq("SGD").fillna(False)
    is_usd = engineered_data["price_currency"].eq("USD").fillna(False)
    engineered_data["price_amount_sgd"] = np.select(
        [
            is_sgd,
            is_usd,
        ],
        [
            engineered_data["price_amount"],
            engineered_data["price_amount"] * 1.35,
        ],
        default=np.nan,
    )

    return engineered_data


def build_reference_year_dates(month_names: pd.Series, day_number: pd.Series) -> pd.Series:
    """Build reference-year dates from month-name and day columns."""
    date_parts = {
        "year": REFERENCE_YEAR,
        "month": month_names.map(MONTH_NUMBER_BY_NAME).astype("float64"),
        "day": pd.to_numeric(day_number, errors="coerce").astype("float64"),
    }
    return pd.to_datetime(date_parts, errors="coerce")


def add_stay_length_nights(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add stay_length_nights from arrival and checkout dates."""
    engineered_data = feature_data.copy()
    arrival_dates = build_reference_year_dates(
        engineered_data["arrival_month"],
        engineered_data["arrival_day"],
    )
    checkout_dates = build_reference_year_dates(
        engineered_data["checkout_month"],
        engineered_data["checkout_day"],
    )
    stay_length = (checkout_dates - arrival_dates).dt.days
    engineered_data["stay_length_nights"] = stay_length.where(
        stay_length >= 0,
        stay_length + 366,
    )

    return engineered_data


def add_booking_to_arrival_month_gap(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add booking_to_arrival_month_gap from booking and arrival months."""
    engineered_data = feature_data.copy()
    booking_month_number = engineered_data["booking_month"].map(MONTH_NUMBER_BY_NAME)
    arrival_month_number = engineered_data["arrival_month"].map(MONTH_NUMBER_BY_NAME)
    month_gap = arrival_month_number - booking_month_number
    engineered_data["booking_to_arrival_month_gap"] = month_gap.where(
        month_gap >= 0,
        month_gap + 12,
    )

    return engineered_data


def add_total_guests(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add total guest count."""
    engineered_data = feature_data.copy()
    engineered_data["total_guests"] = (
        engineered_data["num_adults"] + engineered_data["num_children"]
    )

    return engineered_data


def add_price_per_guest(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add price per guest."""
    engineered_data = feature_data.copy()
    total_guests = engineered_data["total_guests"].where(
        engineered_data["total_guests"] > 0
    )
    engineered_data["price_per_guest"] = engineered_data["price_amount_sgd"] / total_guests

    return engineered_data


def add_price_per_night(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add price per night."""
    engineered_data = feature_data.copy()
    stay_length = engineered_data["stay_length_nights"].where(
        engineered_data["stay_length_nights"] > 0
    )
    engineered_data["price_per_night"] = engineered_data["price_amount_sgd"] / stay_length

    return engineered_data


def add_price_per_guest_per_night(feature_data: pd.DataFrame) -> pd.DataFrame:
    """Add price per guest per night."""
    engineered_data = feature_data.copy()
    total_guests = engineered_data["total_guests"].where(
        engineered_data["total_guests"] > 0
    )
    stay_length = engineered_data["stay_length_nights"].where(
        engineered_data["stay_length_nights"] > 0
    )
    guest_nights = total_guests * stay_length
    engineered_data["price_per_guest_per_night"] = (
        engineered_data["price_amount_sgd"] / guest_nights
    )

    return engineered_data
