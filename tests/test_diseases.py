"""Tests for the 10-disease taxonomy."""

from __future__ import annotations

from brainframe.classification.diseases import (
    DISEASE_TAXONOMY,
    LATERALITIES,
    PATTERNS,
    REGIONS,
    disease_names,
    get_disease,
    num_classes,
)


def test_taxonomy_has_ten_diseases():
    assert num_classes() == 10
    assert len(DISEASE_TAXONOMY) == 10


def test_class_ids_are_contiguous_from_zero():
    ids = sorted(d.class_id for d in DISEASE_TAXONOMY)
    assert ids == list(range(10))


def test_names_match_expected_set():
    expected = {
        "Healthy",
        "AD",
        "PD",
        "MS",
        "Glioma",
        "Stroke",
        "Epilepsy",
        "HD",
        "ALS",
        "TBI",
    }
    assert set(disease_names()) == expected


def test_get_disease_roundtrip():
    for d in DISEASE_TAXONOMY:
        assert get_disease(d.class_id).name == d.name


def test_signatures_use_valid_enums():
    for d in DISEASE_TAXONOMY:
        for r in d.preferred_regions:
            assert r in REGIONS, f"{d.short_name}: unknown region {r}"
        assert d.pattern in PATTERNS
        assert d.laterality in LATERALITIES
        lo, hi = d.size_mm3
        assert 0.0 <= lo <= hi
        nlo, nhi = d.region_count
        assert 0 <= nlo <= nhi


def test_to_dict_keys():
    d = DISEASE_TAXONOMY[3]
    dct = d.to_dict()
    assert dct["class_id"] == 3
    assert "summary" in dct and "references" in dct
    assert isinstance(dct["preferred_regions"], list)


def test_healthy_signature_has_no_regions():
    h = get_disease(0)
    assert h.preferred_regions == ()
    assert h.region_count == (0, 0)
