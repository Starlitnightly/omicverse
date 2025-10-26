# config_manager.py

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

class ConfigManager:
    """Utility helpers for reading and writing the Streamlit agent configuration."""

    CONFIG_PATH = Path(__file__).resolve().parent / "config.json"

    DEFAULT_PACKAGES: List[Dict[str, str]] = [
        {
            "name": "cellrank_notebooks",
            "converted_jsons_subdir": "6O_json_files/cellrank_notebooks",
            "annotated_scripts_subdir": "annotated_scripts/cellrank_notebooks",
        },
        {
            "name": "scanpy_tutorials",
            "converted_jsons_subdir": "6O_json_files/scanpy-tutorials",
            "annotated_scripts_subdir": "annotated_scripts/scanpy-tutorials",
        },
        {
            "name": "scvi_tutorials",
            "converted_jsons_subdir": "6O_json_files/scvi-tutorials",
            "annotated_scripts_subdir": "annotated_scripts/scvi-tutorials",
        },
        {
            "name": "spateo_tutorials",
            "converted_jsons_subdir": "6O_json_files/spateo-tutorials",
            "annotated_scripts_subdir": "annotated_scripts/spateo-tutorials",
        },
        {
            "name": "squidpy_notebooks",
            "converted_jsons_subdir": "6O_json_files/squidpy_notebooks",
            "annotated_scripts_subdir": "annotated_scripts/squidpy_notebooks",
        },
        {
            "name": "ov_tut",
            "converted_jsons_subdir": "6O_json_files/ov_tut",
            "annotated_scripts_subdir": "annotated_scripts/ov_tut",
        },
    ]

    @staticmethod
    def default_config() -> Dict[str, object]:
        """Return the default configuration, including package metadata."""

        base_dir = os.environ.get("OV_PACKAGE_BASE_DIR")
        if base_dir is None:
            base_dir = str(Path.home() / "omicverse_packages")

        return {
            "file_selection_model": "qwen2.5-coder:3b",
            "query_processing_model": "qwen2.5-coder:7b",
            "rate_limit": 5,
            "paper_checker_mode": False,
            "computer_use_agent": False,
            "selected_package": "",
            "package_base_dir": base_dir,
            "packages": ConfigManager.DEFAULT_PACKAGES,
        }

    @staticmethod
    def load_config() -> Dict[str, object]:
        """Load the configuration from disk, merging it with defaults when missing values."""

        if ConfigManager.CONFIG_PATH.exists():
            with open(ConfigManager.CONFIG_PATH, "r", encoding="utf-8") as file_handle:
                persisted = json.load(file_handle)
        else:
            persisted = {}

        return ConfigManager._merge_with_defaults(persisted)

    @staticmethod
    def save_config(config: Dict[str, object]) -> None:
        """Persist the configuration to disk with normalised package metadata."""

        ConfigManager.CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        merged = ConfigManager._merge_with_defaults(config)
        merged["package_base_dir"] = str(merged.get("package_base_dir", ""))
        merged["packages"] = ConfigManager._normalise_package_entries(
            merged.get("packages", ConfigManager.DEFAULT_PACKAGES)
        )
        with open(ConfigManager.CONFIG_PATH, "w", encoding="utf-8") as file_handle:
            json.dump(merged, file_handle, indent=2)

    @staticmethod
    def _merge_with_defaults(config: Dict[str, object]) -> Dict[str, object]:
        defaults = ConfigManager.default_config()
        merged = {**defaults, **config}

        if "packages" in config and isinstance(config["packages"], Iterable):
            merged["packages"] = ConfigManager._normalise_package_entries(config["packages"])
        else:
            merged["packages"] = ConfigManager.DEFAULT_PACKAGES

        return merged

    @staticmethod
    def _normalise_package_entries(packages: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
        normalised = []
        for entry in packages:
            if not isinstance(entry, dict):
                continue
            cleaned: Dict[str, str] = {"name": entry.get("name")}
            if entry.get("converted_jsons_directory"):
                cleaned["converted_jsons_directory"] = str(entry["converted_jsons_directory"])
            if entry.get("converted_jsons_subdir"):
                cleaned["converted_jsons_subdir"] = str(entry["converted_jsons_subdir"])
            if entry.get("annotated_scripts_directory"):
                cleaned["annotated_scripts_directory"] = str(entry["annotated_scripts_directory"])
            if entry.get("annotated_scripts_subdir"):
                cleaned["annotated_scripts_subdir"] = str(entry["annotated_scripts_subdir"])
            if cleaned["name"]:
                normalised.append(cleaned)
        return normalised

    @staticmethod
    def resolve_package_directories(config: Dict[str, object]) -> Tuple[List[Dict[str, Path]], List[str]]:
        """Resolve configured package directories into absolute :class:`Path` objects."""

        base_dir_value = config.get("package_base_dir") or ConfigManager.default_config()["package_base_dir"]
        base_dir = Path(str(base_dir_value)).expanduser()
        resolved_packages: List[Dict[str, Path]] = []
        warnings: List[str] = []

        package_entries = config.get("packages") or ConfigManager.DEFAULT_PACKAGES
        for package in package_entries:
            if not isinstance(package, dict):
                warnings.append("Ignoring malformed package entry in configuration.")
                continue

            name = package.get("name")
            if not name:
                warnings.append("Encountered package entry without a name; skipping.")
                continue

            converted_dir = ConfigManager._resolve_package_path(
                package.get("converted_jsons_directory"),
                package.get("converted_jsons_subdir"),
                base_dir,
            )
            annotated_dir = ConfigManager._resolve_package_path(
                package.get("annotated_scripts_directory"),
                package.get("annotated_scripts_subdir"),
                base_dir,
            )

            if converted_dir is None or annotated_dir is None:
                warnings.append(
                    f"Package '{name}' is missing directory information; expected either absolute paths or subdir entries."
                )
                continue

            resolved_packages.append(
                {
                    "name": name,
                    "converted_jsons_directory": converted_dir,
                    "annotated_scripts_directory": annotated_dir,
                }
            )

        return resolved_packages, warnings

    @staticmethod
    def _resolve_package_path(
        absolute_path: str,
        relative_path: str,
        base_dir: Path,
    ) -> Path | None:
        path_value = absolute_path or relative_path
        if not path_value:
            return None

        candidate = Path(path_value).expanduser()
        if not candidate.is_absolute():
            candidate = base_dir / candidate
        return candidate

    @staticmethod
    def get_package_names(config: Dict[str, object]) -> List[str]:
        packages, _ = ConfigManager.resolve_package_directories(config)
        return [pkg["name"] for pkg in packages]
