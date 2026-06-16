import yaml
import pytest

from dashboard.services.parameter_service import ParameterService
from src.configuration.param_editing import flatten_parameters, read_path, set_param_values, write_path


def write_yaml(path, data):
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


class TestDottedTopLevelParameterPaths:
    def test_read_path_supports_dotted_top_level_mapping(self):
        data = {"external.movement": {"alpha_probability": 0.35}, "logging": {"enabled": True}}

        assert read_path(data, "external.movement.alpha_probability") == 0.35
        assert read_path(data, "logging.enabled") is True

    def test_write_path_supports_dotted_top_level_mapping(self):
        data = {"external.movement": {"alpha_probability": 0.35}, "logging": {"enabled": True}}

        write_path(data, "external.movement.alpha_probability", 0.5, create=False)
        write_path(data, "logging.enabled", False, create=False)

        assert data["external.movement"]["alpha_probability"] == 0.5
        assert data["logging"]["enabled"] is False

    def test_write_path_rejects_missing_leaf_without_create(self):
        data = {"external.movement": {"alpha_probability": 0.35}}

        with pytest.raises(KeyError, match="external.movement.missing_probability"):
            write_path(data, "external.movement.missing_probability", 0.1, create=False)

    def test_set_param_values_updates_system_style_keys(self, tmp_path):
        path = tmp_path / "system.yaml"
        write_yaml(
            path,
            {
                "stop.at": 500,
                "external.movement": {"alpha_probability": 0.35, "aggregate_probability": 0.08},
                "logging": {"enabled": True},
            },
        )

        updated = set_param_values(
            path,
            {
                "stop.at": 250,
                "external.movement.alpha_probability": 0.45,
                "logging.enabled": False,
            },
        )

        assert updated["stop.at"] == 250
        assert updated["external.movement"]["alpha_probability"] == 0.45
        assert updated["logging"]["enabled"] is False

    def test_flatten_parameters_keeps_dashboard_keys_for_dotted_groups(self):
        rows = flatten_parameters({"external.movement": {"alpha_probability": 0.35}})

        assert rows == [{"key": "external.movement.alpha_probability", "type": "float", "value": "0.35"}]


class TestParameterServiceSystemFileEditing:
    def test_parse_and_save_rows_accepts_system_yaml_dotted_groups(self, tmp_path):
        write_yaml(
            tmp_path / "system.yaml",
            {
                "stop.at": 500,
                "external.movement": {"alpha_probability": 0.35, "aggregate_probability": 0.08},
                "logging": {"enabled": True},
            },
        )
        service = ParameterService(param_dir=tmp_path)
        rows = service.editable_rows("system")
        edited_rows = [
            {**row, "value": "0.45"}
            if row["key"] == "external.movement.alpha_probability"
            else row
            for row in rows
        ]

        updates, errors = service.parse_and_validate("system", edited_rows)
        updated = service.save_rows("system", edited_rows, dry_run=True)

        assert errors == []
        assert updates["external.movement.alpha_probability"] == 0.45
        assert updated["external.movement"]["alpha_probability"] == 0.45
