"""
MACE-F Carbon-Efficiency Decision Framework (M5)

The final deliverable of the MACE-F project: a data-driven, multi-dimensional
lookup system that maps manufacturing AI choices to expected carbon emissions.

All values are derived from the 23-experiment benchmark on real manufacturing
datasets (SECOM, NASA CMAPSS, NEU Steel Defects), measured via CodeCarbon v3.2.7
with Fischer (2025) calibration, on NVIDIA RTX 3070 Ti Laptop GPU (UK grid baseline).

Usage:
    from carbon_benchmark.decision_matrix import MACEFramework

    fw = MACEFramework()
    result = fw.assess(
        model_tier="T3",
        deployment_scenario="batch_qc",
        country="China",
        use_transfer_learning=True,
    )
    print(f"Annual CO2: {result['annual_co2eq_kg']:.2f} kg")
"""

from __future__ import annotations

from dataclasses import dataclass

# ============================================================================
# Core Data Tables — all values from experiment results
# ============================================================================

# Tier definitions with measured training carbon (UK grid).
# training_co2_g 口径: sum of the CodeCarbon `emissions` column (kg → g) per rep,
# median of 3 reps per model, then median of the per-model medians within the tier.
# Source: results/E_methodA_<model>_rep{1,2,3}_emissions.csv
TIER_PROFILES: dict[str, dict] = {
    "T1": {
        "label": "Ultra-Lightweight ML",
        "param_range": "<1K parameters",
        "models": ["Linear Regression", "Decision Tree"],
        "data_modality": ["tabular", "time-series"],
        "training_co2_g": 0.062625671,   # median of per-model medians (3 reps each): lin_reg 0.041478, decision_tree 0.083774
        "training_time_s": 2,
        "accuracy_range": "80–94% (R²)",
        "annual_infer_co2_kg_24x7": 0.50,
        "carbon_class": "NEGLIGIBLE",
    },
    "T2": {
        "label": "Lightweight ML / Compact DL",
        "param_range": "1K–10M parameters",
        "models": ["Random Forest", "XGBoost", "MobileNet-v2"],
        "data_modality": ["tabular", "time-series", "image"],
        "training_co2_g": 0.062602708,   # median of per-model medians (3 reps each): random_forest 0.062603, xgboost 0.056939, mobilenet_v2 2.815484
        "training_time_s": 144,          # median: 2s, 2s, 215s = 2s → but for image, 215s
        "accuracy_range": "93–99.5% (Acc/F1)",
        "annual_infer_co2_kg_24x7": 0.41,
        "carbon_class": "LOW",
    },
    "T3": {
        "label": "Standard Deep Learning",
        "param_range": "10M–50M parameters",
        "models": ["ResNet-50", "LSTM"],
        "data_modality": ["time-series", "image"],
        "training_co2_g": 4.782465659,   # median of per-model medians (3 reps each): resnet_50 8.794553, lstm 0.770378
        "training_time_s": 291,          # median of 524s and 47s
        "accuracy_range": "91–99.6% (R²/Acc)",
        "annual_infer_co2_kg_24x7": 0.40,
        "carbon_class": "MODERATE",
    },
    "T4": {
        "label": "Large Deep Learning",
        "param_range": "50M–500M parameters",
        "models": ["ResNet-152", "ViT-B/16"],
        "data_modality": ["image"],
        "training_co2_g": 38.235954621,  # median of per-model medians (3 reps each): resnet_152 21.207512, vit_b_16 55.264397
        "training_time_s": 1364,         # median of 1244s and 1481s
        "accuracy_range": "99.3–99.9% (Acc/F1)",
        "annual_infer_co2_kg_24x7": 1.09,
        "carbon_class": "HIGH",
    },
    "T5": {
        "label": "Generative / Foundation Models",
        "param_range": ">500M parameters",
        "models": ["Stable Diffusion", "GPT-4o-mini"],
        "data_modality": ["image-generation", "text"],
        "training_co2_g": None,           # not measured (requires cloud/cluster)
        "training_time_s": None,
        "accuracy_range": "N/A (generative)",
        "annual_infer_co2_kg_24x7": 1351, # median: 2635 and 67
        "carbon_class": "CRITICAL",
    },
}

# Deployment scenario annual inference multipliers
# Baseline = single inference energy × annual inferences
DEPLOYMENT_SCENARIOS: dict[str, dict] = {
    "on_demand": {
        "label": "On-Demand / Weekly QC",
        "inferences_per_year": 25_000,
        "manufacturing_example": "Weekly quality audit, occasional design generation",
        "carbon_multiplier": 1.0,         # reference baseline
    },
    "batch_qc": {
        "label": "Batch QC / Daily",
        "inferences_per_year": 2_500_000,
        "manufacturing_example": "Nightly batch inspection, per-shift QC reports",
        "carbon_multiplier": 100,
    },
    "continuous_24x7": {
        "label": "Continuous / Real-Time Production Line",
        "inferences_per_year": 31_536_000,
        "manufacturing_example": "Real-time defect detection on conveyor, 1 infer/sec",
        "carbon_multiplier": 1260,
    },
}

# Geographic location carbon intensity multiplier (relative to UK baseline = 231 gCO2/kWh)
# Source: Method F experimental data
LOCATION_FACTORS: dict[str, dict] = {
    "Sweden":              {"ci": 13,   "factor": 0.06,  "region": "Nordic",        "risk": "MINIMAL"},
    "Switzerland":         {"ci": 16,   "factor": 0.07,  "region": "Central Europe", "risk": "MINIMAL"},
    "Norway":              {"ci": 20,   "factor": 0.09,  "region": "Nordic",        "risk": "MINIMAL"},
    "Finland":             {"ci": 54,   "factor": 0.23,  "region": "Nordic",        "risk": "LOW"},
    "France":              {"ci": 56,   "factor": 0.24,  "region": "Western Europe", "risk": "LOW"},
    "USA (California)":    {"ci": 204,  "factor": 0.88,  "region": "North America",  "risk": "LOW"},
    "UK":                  {"ci": 231,  "factor": 1.00,  "region": "Western Europe", "risk": "BASELINE"},
    "Ireland":             {"ci": 331,  "factor": 1.43,  "region": "Western Europe", "risk": "MODERATE"},
    "Germany":             {"ci": 338,  "factor": 1.46,  "region": "Central Europe", "risk": "MODERATE"},
    "Netherlands":         {"ci": 397,  "factor": 1.72,  "region": "Western Europe", "risk": "MODERATE"},
    "South Korea":         {"ci": 416,  "factor": 1.80,  "region": "East Asia",     "risk": "MODERATE"},
    "USA (Texas)":         {"ci": 427,  "factor": 1.85,  "region": "North America",  "risk": "MODERATE"},
    "Mexico":              {"ci": 431,  "factor": 1.87,  "region": "Latin America",  "risk": "MODERATE"},
    "Vietnam":             {"ci": 432,  "factor": 1.87,  "region": "Southeast Asia", "risk": "MODERATE"},
    "Japan":               {"ci": 472,  "factor": 2.04,  "region": "East Asia",     "risk": "HIGH"},
    "China":               {"ci": 582,  "factor": 2.52,  "region": "East Asia",     "risk": "HIGH"},
    "Serbia":              {"ci": 671,  "factor": 2.90,  "region": "Eastern Europe", "risk": "HIGH"},
    "India":               {"ci": 708,  "factor": 3.06,  "region": "South Asia",    "risk": "HIGH"},
    "USA (Iowa)":          {"ci": 737,  "factor": 3.19,  "region": "North America",  "risk": "HIGH"},
    "South Africa":        {"ci": 1009, "factor": 4.37,  "region": "Africa",        "risk": "CRITICAL"},
}

# Infrastructure PUE factors (relative to edge device PUE=1.02 baseline)
# Source: Method G experimental data
INFRASTRUCTURE_FACTORS: dict[str, dict] = {
    "edge_device":            {"pue": 1.02, "factor": 1.00, "label": "Edge Device"},
    "google_cloud":           {"pue": 1.10, "factor": 1.08, "label": "Google Cloud"},
    "aws":                    {"pue": 1.15, "factor": 1.13, "label": "AWS"},
    "azure":                  {"pue": 1.20, "factor": 1.18, "label": "Azure"},
    "colocation":             {"pue": 1.40, "factor": 1.37, "label": "Colocation DC"},
    "on_premise_datacenter":  {"pue": 1.70, "factor": 1.67, "label": "On-Premise DC"},
    "factory_floor":          {"pue": 2.00, "factor": 1.96, "label": "Factory Floor Server"},
}

# Carbon reduction levers (from Methods B, C)
REDUCTION_LEVERS: dict[str, dict] = {
    "transfer_learning_frozen": {
        "method": "C",
        "saving": 0.606,     # 60.6% energy reduction
        "cost": "None — same hardware, pre-trained weights freely available",
        "applicable_to": ["T2", "T3", "T4"],
        "evidence": "ResNet-50 on NEU Steel: 0.0196→0.0077 kWh",
    },
    "hyperparameter_optimisation": {
        "method": "B",
        "saving": 0.16,      # 16% from worst to best configuration
        "cost": "Minimal — 8 trial runs, ~30 min total on single GPU",
        "applicable_to": ["T2", "T3", "T4"],
        "evidence": "ResNet-50: bs=8,lr=1e-3 → bs=64,lr=1e-3 saves 13.6%",
    },
    "model_downgrade": {
        "method": "A",
        "saving": 0.999,     # 99.9% from ViT-B/16 to Random Forest
        "cost": "Accuracy trade-off — must verify task requirements",
        "applicable_to": ["T1", "T2", "T3", "T4"],
        "evidence": "ViT-B/16 (33.37g, 99.4%) → Random Forest (0.01g, 93.3%)",
    },
    "cloud_migration": {
        "method": "G",
        "saving": 0.89,      # 89% factory floor China → Google Cloud Finland
        "cost": "Cloud subscription + data egress; latency may increase",
        "applicable_to": ["T1", "T2", "T3", "T4", "T5"],
        "evidence": "40.57g (Factory+China) → 2.07g (GCP+Finland)",
    },
}

# ============================================================================
# Decision logic
# ============================================================================

@dataclass
class AssessmentResult:
    """Structured output of a MACE-F carbon assessment."""
    model_tier: str
    tier_label: str
    data_modality: str
    deployment_scenario: str
    country: str
    infrastructure: str

    # Carbon estimates
    training_co2_g: float
    single_inference_co2_g: float
    annual_inferences: int
    annual_co2eq_kg: float

    # Multipliers applied
    location_factor: float
    infrastructure_factor: float
    transfer_learning_saving: float

    # Decision guidance
    carbon_class: str
    recommendations: list[str]
    reduction_potential: str


class MACEFramework:
    """MACE-F Carbon-Efficiency Decision Framework.

    Provides a multi-dimensional lookup that answers:
      1. What is the expected carbon footprint of my AI deployment?
      2. Which factors dominate the carbon cost?
      3. What is the most impactful reduction strategy?
    """

    def __init__(self, baseline_country: str = "UK"):
        self.baseline_country = baseline_country

    # ------------------------------------------------------------------
    # Tier lookup
    # ------------------------------------------------------------------
    def get_tier_profile(self, tier: str) -> dict:
        """Return the carbon profile for a given tier."""
        if tier not in TIER_PROFILES:
            raise KeyError(f"Unknown tier '{tier}'. Valid: {list(TIER_PROFILES)}")
        return dict(TIER_PROFILES[tier])

    def tier_for_model(self, model_name: str) -> str:
        """Map a model name to its tier."""
        for tier, profile in TIER_PROFILES.items():
            if model_name in profile["models"]:
                return tier
        raise KeyError(f"Unknown model '{model_name}'")

    # ------------------------------------------------------------------
    # Location & infrastructure multipliers
    # ------------------------------------------------------------------
    def get_location_factor(self, country: str) -> float:
        """Return carbon intensity multiplier relative to UK baseline."""
        if country not in LOCATION_FACTORS:
            raise KeyError(f"Unknown country '{country}'. Valid: {list(LOCATION_FACTORS)}")
        return float(LOCATION_FACTORS[country]["factor"])

    def get_infrastructure_factor(self, infra: str) -> float:
        """Return PUE multiplier relative to edge device baseline."""
        if infra not in INFRASTRUCTURE_FACTORS:
            raise KeyError(f"Unknown infrastructure '{infra}'. Valid: {list(INFRASTRUCTURE_FACTORS)}")
        return float(INFRASTRUCTURE_FACTORS[infra]["factor"])

    # ------------------------------------------------------------------
    # Full assessment
    # ------------------------------------------------------------------
    def assess(
        self,
        model_tier: str,
        deployment_scenario: str = "batch_qc",
        country: str = "UK",
        infrastructure: str = "on_premise_datacenter",
        use_transfer_learning: bool = False,
        data_modality: str | None = None,
    ) -> AssessmentResult:
        """Run a complete carbon assessment for a manufacturing AI deployment.

        Args:
            model_tier: One of T1, T2, T3, T4, T5.
            deployment_scenario: on_demand, batch_qc, or continuous_24x7.
            country: Deployment country (see LOCATION_FACTORS keys).
            infrastructure: Deployment infrastructure type.
            use_transfer_learning: Whether frozen-backbone transfer learning is used.
            data_modality: Optional — tabular, time-series, image, image-generation, text.

        Returns:
            AssessmentResult with all carbon estimates and guidance.
        """
        tier = self.get_tier_profile(model_tier)
        scenario = DEPLOYMENT_SCENARIOS[deployment_scenario]
        loc_factor = self.get_location_factor(country)
        infra_factor = self.get_infrastructure_factor(infrastructure)

        # ---- Training carbon (one-off) ----
        training_co2 = tier["training_co2_g"] or 0.0
        training_co2_adj = training_co2 * loc_factor * infra_factor

        # ---- Single inference carbon ----
        # Estimated from annual 24/7 data: annual / 31,536,000 inferences
        if tier["annual_infer_co2_kg_24x7"] and tier["annual_infer_co2_kg_24x7"] > 0:
            single_infer_co2_g = (tier["annual_infer_co2_kg_24x7"] * 1000) / 31_536_000
        else:
            single_infer_co2_g = 0.0

        # ---- Annual inference carbon ----
        annual_n = scenario["inferences_per_year"]
        annual_infer_co2_kg = single_infer_co2_g * annual_n / 1000 * loc_factor * infra_factor

        # ---- Apply transfer learning saving (training only) ----
        tl_saving = 0.0
        if use_transfer_learning and model_tier in ("T2", "T3", "T4"):
            tl_saving = REDUCTION_LEVERS["transfer_learning_frozen"]["saving"]
            training_co2_adj *= (1 - tl_saving)

        # ---- Carbon class & recommendations ----
        total_annual_kg = annual_infer_co2_kg + training_co2_adj / 1000
        carbon_class, recs = self._classify_and_recommend(
            total_annual_kg, model_tier, country, infrastructure, use_transfer_learning
        )

        return AssessmentResult(
            model_tier=model_tier,
            tier_label=tier["label"],
            data_modality=data_modality or ", ".join(tier["data_modality"]),
            deployment_scenario=deployment_scenario,
            country=country,
            infrastructure=infrastructure,
            training_co2_g=round(training_co2_adj, 3),
            single_inference_co2_g=round(single_infer_co2_g, 8),
            annual_inferences=annual_n,
            annual_co2eq_kg=round(annual_infer_co2_kg, 3),
            location_factor=round(loc_factor, 2),
            infrastructure_factor=round(infra_factor, 2),
            transfer_learning_saving=round(tl_saving, 3),
            carbon_class=carbon_class,
            recommendations=recs,
            reduction_potential=self._reduction_summary(model_tier, country, infrastructure),
        )

    # ------------------------------------------------------------------
    # Carbon classification
    # ------------------------------------------------------------------
    @staticmethod
    def _classify_and_recommend(
        total_annual_kg: float,
        tier: str,
        country: str,
        infra: str,
        use_tl: bool,
    ) -> tuple[str, list[str]]:
        """Classify carbon impact and generate tier-specific recommendations.

        Carbon class bands follow report Table 4 (kg CO₂e/year):
          A < 0.1 | B 0.1–1 | C 1–100 | D 100–1000 | E ≥ 1000
        """
        recs = []

        if total_annual_kg < 0.1:
            carbon_class = "A — NEGLIGIBLE — below reporting threshold"
            recs.append("No action required; carbon impact is negligible.")
        elif total_annual_kg < 1.0:
            carbon_class = "B — LOW — comparable to a few hours of office PC use"
            recs.append("Monitor annually; no urgent action needed.")
        elif total_annual_kg < 100:
            carbon_class = "C — MODERATE — comparable to a short-haul flight (50–100 kg CO₂)"
            recs.append("Consider model downgrade or cloud migration for meaningful reduction.")
        elif total_annual_kg < 1000:
            carbon_class = "D — HIGH — comparable to a long-haul flight (500–1000 kg CO₂)"
            recs.append("Immediate action recommended: model downgrade, cloud migration, or edge deployment.")
        else:
            carbon_class = "E — CRITICAL — comparable to annual car emissions (>1000 kg CO₂)"
            recs.append("URGENT: Deploy only if essential; prioritise edge + low-carbon grid + lightweight model.")

        # Tier-specific guidance
        if tier == "T5":
            recs.append("T5 generative models dominate carbon — verify that traditional CV/ML cannot solve the task.")
        if tier in ("T3", "T4") and not use_tl:
            recs.append("Apply transfer learning (frozen backbone) — saves ~60% training energy with minimal accuracy loss.")
        if LOCATION_FACTORS.get(country, {}).get("risk") in ("HIGH", "CRITICAL"):
            recs.append(f"High-carbon grid ({country}): consider cloud deployment in Nordic region (Sweden/Finland) for 10–50× reduction.")

        return carbon_class, recs

    @staticmethod
    def _reduction_summary(tier: str, country: str, infra: str) -> str:
        """Summarise the most impactful reduction lever."""
        loc_risk = LOCATION_FACTORS.get(country, {}).get("risk", "BASELINE")
        infra_pue = INFRASTRUCTURE_FACTORS.get(infra, {}).get("pue", 1.0)

        if tier == "T5":
            return "Largest lever: Replace generative AI with traditional model (100–1000× reduction)."
        if loc_risk in ("HIGH", "CRITICAL") and infra_pue >= 1.7:
            return "Largest lever: Migrate to cloud (Nordic region) — location + PUE compound to ~50× reduction."
        if loc_risk in ("HIGH", "CRITICAL"):
            return "Largest lever: Deploy in low-carbon region — location alone gives 10–50× reduction."
        if tier in ("T3", "T4"):
            return "Largest lever: Transfer learning (60%) or model downgrade to T2 (90–99% reduction)."
        return "Carbon impact is already low; focus on maintaining current efficiency."


# ============================================================================
# Quick-reference lookup table generator
# ============================================================================

def generate_decision_table() -> list[dict]:
    """Generate the full carbon-efficiency decision matrix as a list of dicts.

    Crosses: 5 tiers × 3 scenarios × 3 representative locations.
    """
    fw = MACEFramework()
    rows = []
    for tier in ["T1", "T2", "T3", "T4", "T5"]:
        for scenario in ["on_demand", "batch_qc", "continuous_24x7"]:
            for country in ["Sweden", "UK", "China"]:
                for infra in ["google_cloud", "on_premise_datacenter"]:
                    r = fw.assess(
                        model_tier=tier,
                        deployment_scenario=scenario,
                        country=country,
                        infrastructure=infra,
                        use_transfer_learning=(tier in ("T2", "T3", "T4")),
                    )
                    rows.append({
                        "tier": tier,
                        "scenario": scenario,
                        "country": country,
                        "infra": infra,
                        "training_co2_g": r.training_co2_g,
                        "annual_co2eq_kg": r.annual_co2eq_kg,
                        "carbon_class": r.carbon_class,
                    })
    return rows


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":

    fw = MACEFramework()

    # Demo: three representative assessments
    cases = [
        {
            "name": "Semiconductor yield prediction — conservative",
            "tier": "T1",
            "scenario": "batch_qc",
            "country": "UK",
            "infra": "on_premise_datacenter",
            "tl": False,
        },
        {
            "name": "Steel defect detection — standard",
            "tier": "T3",
            "scenario": "continuous_24x7",
            "country": "China",
            "infra": "factory_floor",
            "tl": True,
        },
        {
            "name": "Generative design tool — worst case",
            "tier": "T5",
            "scenario": "continuous_24x7",
            "country": "India",
            "infra": "factory_floor",
            "tl": False,
        },
        {
            "name": "Aerospace predictive maintenance — optimal",
            "tier": "T3",
            "scenario": "batch_qc",
            "country": "Sweden",
            "infra": "google_cloud",
            "tl": True,
        },
    ]

    print("=" * 72)
    print("MACE-F Carbon-Efficiency Decision Framework")
    print("Manufacturing AI Carbon Assessment — Baseline: UK grid, RTX 3070 Ti")
    print("=" * 72)

    for case in cases:
        kwargs = {k: v for k, v in case.items() if k != "name"}
        kwargs["model_tier"] = kwargs.pop("tier")
        kwargs["deployment_scenario"] = kwargs.pop("scenario")
        kwargs["use_transfer_learning"] = kwargs.pop("tl")
        kwargs["infrastructure"] = kwargs.pop("infra")
        r = fw.assess(**kwargs)
        print(f"\n{'─' * 72}")
        print(f"Case: {case['name']}")
        print(f"  Tier:              {r.model_tier} — {r.tier_label}")
        print(f"  Scenario:          {r.deployment_scenario} ({r.annual_inferences:,} infer/year)")
        print(f"  Location:          {r.country} (×{r.location_factor})")
        print(f"  Infrastructure:    {r.infrastructure} (×{r.infrastructure_factor})")
        print(f"  Transfer learning: {r.transfer_learning_saving:.0%} saving")
        print("  ─────────────────────────────────────────")
        print(f"  Training CO₂:      {r.training_co2_g:.3f} g (one-off)")
        print(f"  Single inference:  {r.single_inference_co2_g:.6f} g")
        print(f"  Annual CO₂:        {r.annual_co2eq_kg:.3f} kg/year")
        print(f"  Carbon class:      {r.carbon_class}")
        print(f"  Reduction:         {r.reduction_potential}")
        for rec in r.recommendations:
            print(f"  → {rec}")

    print(f"\n{'=' * 72}")
    print("Framework loaded. Use MACEFramework().assess() for custom assessments.")
