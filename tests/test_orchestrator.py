"""Tests for census.orchestrator (argparse command routing)."""

from __future__ import annotations

from pathlib import Path

import pytest

from census import orchestrator


class TestBuildParser:
    def test_parse_prepare_data_command(self) -> None:
        parser = orchestrator.build_parser()
        args = parser.parse_args(["prepare-data"])
        assert args.command == "prepare-data"

    def test_parse_train_command(self) -> None:
        parser = orchestrator.build_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"

    def test_parse_slice_command(self) -> None:
        parser = orchestrator.build_parser()
        args = parser.parse_args(["slice", "--slice-features", "sex", "race"])
        assert args.command == "slice"
        assert args.slice_features == ["sex", "race"]

    def test_parse_all_command(self) -> None:
        parser = orchestrator.build_parser()
        args = parser.parse_args(["all", "--skip-data"])
        assert args.command == "all"
        assert args.skip_data is True


class TestMainRouting:
    def test_main_prepare_data_dispatches(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"prepare": 0}

        monkeypatch.setattr(orchestrator, "configure_logging", lambda *_a, **_kw: None)

        def fake_prepare(**_kwargs: object) -> Path:
            calls["prepare"] += 1
            return Path("data/data_cleaned/census_cleaned.csv")

        monkeypatch.setattr(orchestrator, "prepare_data_pipeline", fake_prepare)
        exit_code = orchestrator.main(["prepare-data"])
        assert exit_code == 0
        assert calls["prepare"] == 1

    def test_main_train_dispatches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, int] = {"train": 0}

        monkeypatch.setattr(orchestrator, "configure_logging", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            orchestrator,
            "run_training_pipeline",
            lambda: calls.__setitem__("train", calls["train"] + 1),
        )
        exit_code = orchestrator.main(["train"])
        assert exit_code == 0
        assert calls["train"] == 1

    def test_main_slice_dispatches(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: dict[str, tuple[str, ...] | None] = {"slice": None}

        monkeypatch.setattr(orchestrator, "configure_logging", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            orchestrator,
            "run_slice_pipeline",
            lambda **kwargs: calls.__setitem__("slice", kwargs["slice_features"]),
        )
        exit_code = orchestrator.main(["slice", "--slice-features", "sex", "race"])
        assert exit_code == 0
        assert calls["slice"] == ("sex", "race")

    def test_main_all_dispatches_full_flow(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: dict[str, int] = {"prepare": 0, "train": 0, "slice": 0}

        monkeypatch.setattr(orchestrator, "configure_logging", lambda *_a, **_kw: None)
        monkeypatch.setattr(
            orchestrator,
            "prepare_data_pipeline",
            lambda **_kwargs: calls.__setitem__("prepare", calls["prepare"] + 1),
        )
        monkeypatch.setattr(
            orchestrator,
            "run_training_pipeline",
            lambda: calls.__setitem__("train", calls["train"] + 1),
        )
        monkeypatch.setattr(
            orchestrator,
            "run_slice_pipeline",
            lambda **_kwargs: calls.__setitem__("slice", calls["slice"] + 1),
        )

        exit_code = orchestrator.main(["all"])
        assert exit_code == 0
        assert calls == {"prepare": 1, "train": 1, "slice": 1}
