"""Feature engineering pipeline."""

from __future__ import annotations

from kedro.pipeline import Pipeline, node

from hotel_no_show_prediction.pipelines.feature_engineering.nodes import (
    add_booking_to_arrival_month_gap,
    add_price_per_guest,
    add_price_per_guest_per_night,
    add_price_per_night,
    add_stay_length_nights,
    add_total_guests,
    price_amount_usd_to_sgd,
    parse_price_currency_and_amount,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature engineering pipeline."""
    return Pipeline(
        [
            node(
                func=parse_price_currency_and_amount,
                inputs="noshow_cleaned",
                outputs="noshow_price_parsed",
                name="parse_price_currency_and_amount",
            ),
            node(
                func=price_amount_usd_to_sgd,
                inputs="noshow_price_parsed",
                outputs="noshow_price_amount_sgd",
                name="price_amount_usd_to_sgd",
            ),
            node(
                func=add_stay_length_nights,
                inputs="noshow_price_amount_sgd",
                outputs="noshow_stay_length_nights",
                name="add_stay_length_nights",
            ),
            node(
                func=add_booking_to_arrival_month_gap,
                inputs="noshow_stay_length_nights",
                outputs="noshow_stay_timing_features",
                name="add_booking_to_arrival_month_gap",
            ),
            node(
                func=add_total_guests,
                inputs="noshow_stay_timing_features",
                outputs="noshow_total_guest_features",
                name="add_total_guests",
            ),
            node(
                func=add_price_per_guest,
                inputs="noshow_total_guest_features",
                outputs="noshow_price_per_guest_features",
                name="add_price_per_guest",
            ),
            node(
                func=add_price_per_night,
                inputs="noshow_price_per_guest_features",
                outputs="noshow_price_per_night_features",
                name="add_price_per_night",
            ),
            node(
                func=add_price_per_guest_per_night,
                inputs="noshow_price_per_night_features",
                outputs="noshow_features",
                name="add_price_per_guest_per_night",
            ),
        ]
    )
