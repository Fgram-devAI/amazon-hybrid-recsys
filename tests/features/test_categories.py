"""Tests for category normalization and filtering."""

from src.features.categories import filter_categories, normalize_category


def test_normalize_strips_whitespace_and_case():
    assert normalize_category("  comedy  ") == "Comedy"
    assert normalize_category("ACTION & ADVENTURE") == "Action & Adventure"
    assert normalize_category("movies & tv") == "Movies & TV"
    assert normalize_category("rpg") == "RPG"
    assert normalize_category("dvd") == "DVD"


def test_normalize_returns_none_for_empty():
    assert normalize_category("") is None
    assert normalize_category(None) is None
    assert normalize_category("   ") is None


def test_filter_drops_generic_roots_and_empty_keeps_informative():
    raw = ["Movies & TV", "Comedy", "", None, "Drama", "movies & tv"]
    generic = ["Movies & TV", "Video Games", "Digital Music"]
    assert filter_categories(raw, generic_roots=generic) == ["Comedy", "Drama"]


def test_filter_is_order_preserving_and_deduplicated():
    raw = ["Comedy", "Drama", "Comedy", "Drama"]
    assert filter_categories(raw, generic_roots=[]) == ["Comedy", "Drama"]
