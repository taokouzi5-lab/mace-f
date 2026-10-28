"""Consistency tests for the central configuration constants.

These guard the cross-references between config tables: MODEL_TIERS vs
EXPERIMENT_MATRIX, CARBON_INTENSITY vs ISO_TO_COUNTRY, PUE value ranges,
calibration factor ordering, dataset metadata, and experiment structure.
"""

from carbon_benchmark import config
from carbon_benchmark.decision_matrix import (
    INFRASTRUCTURE_FACTORS,
    LOCATION_FACTORS,
)


class TestModelTiers:
    def test_all_experiment_models_are_in_model_tiers(self) -> None:
        for entry in config.EXPERIMENT_MATRIX:
            assert entry["model"] in config.MODEL_TIERS, (
                f"{entry['model']} missing from MODEL_TIERS"
            )

    def test_all_model_tiers_used_by_matrix(self) -> None:
        matrix_models = {e["model"] for e in config.EXPERIMENT_MATRIX}
        for model in config.MODEL_TIERS:
            assert model in matrix_models, f"{model} unused in EXPERIMENT_MATRIX"

    def test_tier_labels_are_t1_to_t5(self) -> None:
        assert set(config.MODEL_TIERS.values()) == {"T1", "T2", "T3", "T4", "T5"}

    def test_tier_assignments_match_tier_profiles(self) -> None:
        from carbon_benchmark.decision_matrix import TIER_PROFILES

        def normalise(name: str) -> str:
            return "".join(c for c in name.lower() if c.isalnum())

        for model, tier in config.MODEL_TIERS.items():
            profile_names = {
                normalise(m) for m in TIER_PROFILES[tier]["models"]
            }
            assert normalise(model) in profile_names, f"{model} vs {TIER_PROFILES[tier]['models']}"


class TestExperimentMatrix:
    def test_23_experiments_with_unique_ids(self) -> None:
        ids = [e["id"] for e in config.EXPERIMENT_MATRIX]
        assert len(ids) == 23
        assert len(set(ids)) == 23
        assert ids == [f"E{i:02d}" for i in range(1, 24)]

    def test_valid_methods_and_tasks(self) -> None:
        for entry in config.EXPERIMENT_MATRIX:
            assert entry["method"] in {"A", "B", "C", "D", "E", "F", "G"}
            assert entry["task"] in {"training", "inference", "analysis"}

    def test_method_a_covers_t1_to_t4_training(self) -> None:
        method_a = [e for e in config.EXPERIMENT_MATRIX if e["method"] == "A"]
        assert len(method_a) == 9
        tiers = {config.MODEL_TIERS[e["model"]] for e in method_a}
        assert tiers == {"T1", "T2", "T3", "T4"}
        assert all(e["task"] == "training" for e in method_a)

    def test_method_d_is_inference_only(self) -> None:
        method_d = [e for e in config.EXPERIMENT_MATRIX if e["method"] == "D"]
        assert len(method_d) == 10
        assert all(e["task"] == "inference" for e in method_d)

    def test_methods_f_g_are_analysis(self) -> None:
        fg = [e for e in config.EXPERIMENT_MATRIX if e["method"] in ("F", "G")]
        assert len(fg) == 2
        assert all(e["task"] == "analysis" for e in fg)


class TestCarbonIntensity:
    def test_all_intensities_positive(self) -> None:
        for country, ci in config.CARBON_INTENSITY.items():
            assert ci > 0, f"{country} intensity not positive"

    def test_iso_mapping_values_are_known_countries(self) -> None:
        for iso, country in config.ISO_TO_COUNTRY.items():
            assert len(iso) == 3 and iso.isupper() and iso.isalpha()
            assert country in config.CARBON_INTENSITY, f"{iso} → {country} missing"

    def test_default_country_present(self) -> None:
        assert config.DEFAULT_COUNTRY in config.CARBON_INTENSITY

    def test_uk_baseline_231(self) -> None:
        assert config.CARBON_INTENSITY["UK"] == 231.0

    def test_location_factors_ci_match_config(self) -> None:
        """decision_matrix.LOCATION_FACTORS ci values equal config.CARBON_INTENSITY."""
        name_to_key = {
            "Sweden": "Sweden", "Switzerland": "Switzerland", "Norway": "Norway",
            "Finland": "Finland", "France": "France", "USA (California)": "USA_California",
            "UK": "UK", "Ireland": "Ireland", "Germany": "Germany",
            "Netherlands": "Netherlands", "South Korea": "South_Korea",
            "USA (Texas)": "USA_Texas", "Mexico": "Mexico", "Vietnam": "Vietnam",
            "Japan": "Japan", "China": "China", "Serbia": "Serbia", "India": "India",
            "USA (Iowa)": "USA_Iowa", "South Africa": "South_Africa",
        }
        assert set(name_to_key) == set(LOCATION_FACTORS)
        for name, key in name_to_key.items():
            assert LOCATION_FACTORS[name]["ci"] == config.CARBON_INTENSITY[key], name


class TestPueAndCalibration:
    def test_pue_values_in_expected_range(self) -> None:
        for infra, pue in config.PUE.items():
            assert 1.0 < pue <= 2.5, f"PUE {infra} out of range"

    def test_pue_keys_match_decision_matrix(self) -> None:
        assert set(config.PUE) == set(INFRASTRUCTURE_FACTORS)
        for infra, pue in config.PUE.items():
            assert INFRASTRUCTURE_FACTORS[infra]["pue"] == pue

    def test_calibration_factors_cover_tiers(self) -> None:
        assert set(config.CALIBRATION_FACTORS) == {"T1", "T2", "T3", "T4", "T5"}

    def test_calibration_factors_within_1_to_2(self) -> None:
        for tier, cf in config.CALIBRATION_FACTORS.items():
            assert 1.0 <= cf <= 2.0, f"{tier} factor {cf} out of range"

    def test_calibration_factors_decrease_with_tier(self) -> None:
        """Heavier tiers have smaller relative overhead (Fischer 2025)."""
        factors = config.CALIBRATION_FACTORS
        for lower, higher in [("T1", "T2"), ("T2", "T3"), ("T3", "T4"), ("T4", "T5")]:
            assert factors[lower] > factors[higher], f"{lower} <= {higher}"


class TestDatasets:
    def test_dataset_metadata_complete(self) -> None:
        assert set(config.DATASETS) == {"secom", "nasa_cmapss", "neu_steel", "cnc_tool_wear"}
        for key, meta in config.DATASETS.items():
            for field in ("name", "url", "task"):
                assert meta.get(field), f"{key} missing {field}"

    def test_dataset_tasks_valid(self) -> None:
        for meta in config.DATASETS.values():
            assert meta["task"] in {
                "binary_classification", "regression", "multi_class_classification",
            }


class TestOtherConstants:
    def test_random_seed(self) -> None:
        assert config.RANDOM_SEED == 42

    def test_representative_energy_positive(self) -> None:
        assert config.REPRESENTATIVE_ENERGY_KWH > 0

    def test_hyperparam_grid(self) -> None:
        assert set(config.HYPERPARAM_GRID["batch_sizes"]) == {8, 16, 32, 64}
        assert len(config.HYPERPARAM_GRID["combinations"]) == 8
        for bs, lr in config.HYPERPARAM_GRID["combinations"]:
            assert bs in config.HYPERPARAM_GRID["batch_sizes"]
            assert 0 < lr <= 1e-3
        assert 0 < config.HYPERPARAM_GRID["target_f1"] < 1
        assert config.HYPERPARAM_GRID["max_epochs"] > 0

    def test_deployment_scenarios(self) -> None:
        assert set(config.DEPLOYMENT_SCENARIOS) == {
            "continuous_24x7", "batch_qc", "on_demand",
        }
        for scenario in config.DEPLOYMENT_SCENARIOS.values():
            assert scenario["inferences_per_day"] > 0
            assert 1 <= scenario["days_per_year"] <= 366
            assert scenario["description"]

    def test_manufacturing_prompts(self) -> None:
        assert len(config.MANUFACTURING_DESIGN_PROMPTS) == 10
        for prompt in config.MANUFACTURING_DESIGN_PROMPTS:
            assert isinstance(prompt, str) and prompt.strip()

    def test_codecarbon_config(self) -> None:
        cc = config.CODECARBON_CONFIG
        assert cc["measure_power_secs"] == 15
        assert cc["offline_mode"] is True
        assert cc["save_to_file"] is True

    def test_hardware_keys(self) -> None:
        for key in ("cpu", "gpu", "vram_gb", "ram_gb", "os", "python"):
            assert config.HARDWARE[key]

    def test_paths_under_project_root(self) -> None:
        assert config.DATA_DIR.is_absolute()
        assert config.RESULTS_DIR.is_absolute()
        assert config.LOG_DIR.is_absolute()
        assert config.DATA_DIR.parent.name == "data"
