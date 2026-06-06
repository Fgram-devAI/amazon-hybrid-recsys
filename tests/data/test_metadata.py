"""Tests for item-metadata preparation for content features."""

from src.data.metadata import prepare_metadata


def test_prepare_metadata_builds_text_blob_and_missing_flags():
    records = [
        {
            "parent_asin": "A1",
            "title": "Cool CD",
            "description": ["Great <b>album</b>"],
            "categories": ["Music", "Pop"],
            "price": "$9.99",
            "average_rating": 4.5,
            "rating_number": 120,
            "store": "AcmeRecords",
        },
        {"parent_asin": "A2"},  # everything missing
    ]

    df = prepare_metadata(iter(records))

    a1 = df[df["parent_asin"] == "A1"].iloc[0]
    assert "Cool CD" in a1["text"]
    assert "Great album" in a1["text"]  # HTML stripped
    assert "Music" in a1["text"] and "Pop" in a1["text"]
    assert a1["price"] == 9.99
    assert bool(a1["price_missing"]) is False
    assert a1["store"] == "AcmeRecords"

    a2 = df[df["parent_asin"] == "A2"].iloc[0]
    assert a2["title"] == ""
    assert a2["text"] == ""
    assert a2["store"] == ""
    assert bool(a2["price_missing"]) is True
    assert bool(a2["average_rating_missing"]) is True
    assert bool(a2["rating_number_missing"]) is True


def test_prepare_metadata_empty_keeps_stable_schema():
    # no records -> still a frame with the expected columns (no downstream KeyError)
    df = prepare_metadata(iter([]))
    assert len(df) == 0
    assert "parent_asin" in df.columns
