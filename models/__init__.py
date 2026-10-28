"""Model factory functions for manufacturing AI benchmarks (T1–T5)."""

from carbon_benchmark.models.traditional_ml import (
    create_linear_regression,
    create_decision_tree,
    create_random_forest,
    create_xgboost,
)
from carbon_benchmark.models.deep_learning import (
    create_mobilenet_v2,
    create_resnet50,
    create_resnet152,
    create_vit_b16,
    create_lstm_rul,
    compare_training_modes,
)
from carbon_benchmark.models.generative import (
    create_stable_diffusion_pipeline,
    create_gpt_manufacturing_queries,
)

__all__ = [
    "create_linear_regression",
    "create_decision_tree",
    "create_random_forest",
    "create_xgboost",
    "create_mobilenet_v2",
    "create_resnet50",
    "create_resnet152",
    "create_vit_b16",
    "create_lstm_rul",
    "compare_training_modes",
    "create_stable_diffusion_pipeline",
    "create_gpt_manufacturing_queries",
]
