from __future__ import annotations

from src.round2_v3.build_event_process_features import feature_registry


def test_event_feature_registry_offsets_are_causal() -> None:
    registry = feature_registry(
        [
            "holder_2__level",
            "holder_2__delta_6h",
            "holder_2__relative_delta_12h",
            "holder_2__std_4h",
            "holder_2__level_vs_168h",
        ]
    ).set_index("feature_name")
    assert registry.loc["holder_2__level", "source_offset_min_minutes"] == 0
    assert registry.loc["holder_2__delta_6h", "source_offset_min_minutes"] == -705
    assert (
        registry.loc["holder_2__relative_delta_12h", "source_offset_min_minutes"]
        == -1425
    )
    assert registry.loc["holder_2__std_4h", "source_offset_min_minutes"] == -225
    assert (
        registry.loc["holder_2__level_vs_168h", "source_offset_min_minutes"]
        == -10065
    )
    assert registry["causal"].all()
    assert not registry["target_history"].any()
