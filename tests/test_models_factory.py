"""Factory smoke tests for traditional ML, deep learning, and generative models.

Traditional ML factories build real sklearn/xgboost objects.  Deep learning
factories use pretrained=False so NO network weight downloads occur; torch and
torchvision are guarded with importorskip.  Generative factories are tested
with fakes so no HuggingFace/OpenAI calls happen.
"""

from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

from carbon_benchmark.models import traditional_ml as tml
from carbon_benchmark.models.generative import (
    _default_manufacturing_qa_pairs,
    create_gpt_manufacturing_queries,
    create_stable_diffusion_pipeline,
)

def _load_torch() -> Any:
    """Skip when torch/torchvision are unavailable (CI minimal env)."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch

    return torch


def _load_dl() -> Any:
    _load_torch()
    from carbon_benchmark.models import deep_learning as dl_module

    return dl_module


class _LazyModule:
    """Resolve module attributes on first access.

    ``torch`` / ``dl`` are module-global proxies: torch-free tests never
    touch them, while the first DL test access performs the importorskips
    (skipping that test) or loads the real modules.
    """

    def __init__(self, loader: Callable[[], Any]) -> None:
        object.__setattr__(self, "_loader", loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        # Forward writes so monkeypatch.setattr patches the real module.
        setattr(self._loader(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(self._loader(), name)


torch = _LazyModule(_load_torch)
dl = _LazyModule(_load_dl)


# ---------------------------------------------------------------------------
# Traditional ML factories
# ---------------------------------------------------------------------------

class TestLinearRegression:
    def test_factory_returns_working_model(self) -> None:
        model, train_fn, inference_fn = tml.create_linear_regression()
        rng = np.random.default_rng(0)
        X = rng.normal(size=(30, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + 0.1
        trained = train_fn(X, y)
        assert trained is model
        preds = inference_fn(X[:5])
        assert preds.shape == (5,)
        assert preds.dtype == np.float64


class TestDecisionTree:
    def test_factory_and_default_depth(self) -> None:
        model, train_fn, inference_fn = tml.create_decision_tree()
        assert model.max_depth == 10
        assert model.class_weight == "balanced"
        rng = np.random.default_rng(1)
        X = rng.normal(size=(40, 4))
        y = rng.integers(0, 2, 40)
        assert train_fn(X, y) is model
        assert inference_fn(X[:6]).shape == (6,)


class TestRandomForest:
    def test_factory_parameters(self) -> None:
        model, train_fn, inference_fn = tml.create_random_forest(n_estimators=10, max_depth=5)
        assert model.n_estimators == 10
        assert model.max_depth == 5
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 4))
        y = rng.integers(0, 2, 60)
        train_fn(X, y)
        assert inference_fn(X[:8]).shape == (8,)


class TestXgboost:
    def test_factory_parameters(self) -> None:
        pytest.importorskip("xgboost")
        model, train_fn, inference_fn = tml.create_xgboost(
            n_estimators=10, max_depth=3, learning_rate=0.05,
        )
        assert model.n_estimators == 10
        assert model.max_depth == 3
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 3))
        y = rng.normal(size=50)
        train_fn(X, y)
        assert inference_fn(X[:7]).shape == (7,)


class TestComputeMetrics:
    def test_binary_classification(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 1, 1])
        metrics = tml.compute_metrics(y_true, y_pred, "classification")
        assert set(metrics) == {"accuracy", "f1"}
        assert metrics["accuracy"] == pytest.approx(4 / 6)
        assert 0.0 <= metrics["f1"] <= 1.0

    def test_multiclass_uses_weighted_f1(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        metrics = tml.compute_metrics(y_true, y_pred, "classification")
        assert "f1" in metrics

    def test_f1_valueerror_falls_back_to_micro(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import sklearn.metrics as sm

        real_f1 = sm.f1_score
        calls: list = []

        def flaky_f1(*args: Any, **kwargs: Any) -> float:
            calls.append(kwargs.get("average"))
            if len(calls) == 1:
                raise ValueError("binary f1 on multiclass")
            return real_f1(*args, **kwargs)

        monkeypatch.setattr(sm, "f1_score", flaky_f1)
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 1, 2])
        metrics = tml.compute_metrics(y_true, y_pred, "classification")
        assert calls == ["weighted", "micro"]
        assert "f1" in metrics

    def test_regression(self) -> None:
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.5, 2.5, 3.5, 4.5])
        metrics = tml.compute_metrics(y_true, y_pred, "regression")
        assert metrics["mae"] == pytest.approx(0.5)
        assert metrics["r2"] == pytest.approx(1.0 - 0.25 / 1.25)


# ---------------------------------------------------------------------------
# Deep learning shared utilities
# ---------------------------------------------------------------------------

class TestGetDataloaders:
    def test_classification_casts_labels_to_long(self) -> None:
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(32, 3)).astype(np.float32)
        y_train = rng.integers(0, 2, 32)
        X_test = rng.normal(size=(8, 3)).astype(np.float32)
        y_test = rng.integers(0, 2, 8)
        train_loader, val_loader = dl._get_dataloaders(X_train, y_train, X_test, y_test)
        assert len(train_loader) == 1
        assert len(val_loader) == 1
        bx, by = next(iter(train_loader))
        assert by.dtype == torch.long

    def test_regression_keeps_float_labels(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.normal(size=(16, 2)).astype(np.float32)
        y = rng.normal(size=16).astype(np.float32)
        train_loader, _ = dl._get_dataloaders(X, y, X, y, batch_size=8, classification=False)
        _, by = next(iter(train_loader))
        assert by.dtype == torch.float32


class TestDefaultTrainingLoop:
    def test_trains_one_epoch(self) -> None:
        model = torch.nn.Linear(4, 2)
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(16, 4)).astype(np.float32)
        y_train = rng.integers(0, 2, 16)
        X_test = rng.normal(size=(8, 4)).astype(np.float32)
        y_test = rng.integers(0, 2, 8)
        train_loader, val_loader = dl._get_dataloaders(X_train, y_train, X_test, y_test, batch_size=8)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        result = dl._default_training_loop(
            model, train_loader, val_loader, criterion, optimizer, epochs=1, device="cpu",
        )
        assert result is model
        # Loop ends with model.eval() after validation
        assert model.training is False


# ---------------------------------------------------------------------------
# Deep learning factories (no pretrained downloads)
# ---------------------------------------------------------------------------

def _image_batch(n: int, c: int, h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, c, h, w)).astype(np.float32)


class TestCreateMobilenetV2:
    def test_factory_without_pretrained(self) -> None:
        model, train_fn, inference_fn = dl.create_mobilenet_v2(num_classes=6, pretrained=False)
        assert isinstance(model, torch.nn.Module)
        assert model.classifier[1].out_features == 6
        # Frozen backbone: classifier trainable, backbone frozen
        assert all(p.requires_grad for p in model.classifier.parameters())
        backbone_trainable = [
            p.requires_grad for name, p in model.named_parameters()
            if not name.startswith("classifier")
        ]
        assert not any(backbone_trainable)

    def test_train_and_inference_smoke(self) -> None:
        model, train_fn, inference_fn = dl.create_mobilenet_v2(
            num_classes=6, freeze_backbone=True, pretrained=False,
        )
        X_train = _image_batch(8, 3, 32, 32, seed=1)
        y_train = np.random.default_rng(2).integers(0, 6, 8)
        X_test = _image_batch(4, 3, 32, 32, seed=3)
        y_test = np.random.default_rng(4).integers(0, 6, 4)
        train_fn(X_train, y_train, X_test, y_test, epochs=1, batch_size=4)
        preds = inference_fn(X_test, batch_size=2)
        assert preds.shape == (4,)
        assert np.all((preds >= 0) & (preds < 6))


class TestCreateResnet50:
    def test_factory_without_pretrained(self) -> None:
        model, train_fn, inference_fn = dl.create_resnet50(num_classes=6, pretrained=False)
        assert model.fc.out_features == 6
        assert not any(
            p.requires_grad for name, p in model.named_parameters()
            if not name.startswith("fc")
        )

    def test_train_and_inference_smoke(self) -> None:
        model, train_fn, inference_fn = dl.create_resnet50(
            num_classes=6, freeze_backbone=True, pretrained=False,
        )
        X_train = _image_batch(4, 3, 32, 32, seed=5)
        y_train = np.random.default_rng(6).integers(0, 6, 4)
        X_test = _image_batch(2, 3, 32, 32, seed=7)
        y_test = np.random.default_rng(8).integers(0, 6, 2)
        train_fn(X_train, y_train, X_test, y_test, epochs=1, batch_size=2)
        assert inference_fn(X_test, batch_size=2).shape == (2,)


class TestCreateResnet152:
    def test_factory_without_pretrained(self) -> None:
        model, train_fn, inference_fn = dl.create_resnet152(num_classes=6, pretrained=False)
        assert model.fc.out_features == 6

    def test_train_and_inference_smoke(self) -> None:
        model, train_fn, inference_fn = dl.create_resnet152(
            num_classes=6, freeze_backbone=True, pretrained=False,
        )
        # Batch size 2 required: BatchNorm needs >1 value per channel in train mode
        X_train = _image_batch(4, 3, 32, 32, seed=9)
        y_train = np.random.default_rng(10).integers(0, 6, 4)
        X_test = _image_batch(1, 3, 32, 32, seed=11)
        y_test = np.random.default_rng(12).integers(0, 6, 1)
        train_fn(X_train, y_train, X_test, y_test, epochs=1, batch_size=2)
        assert inference_fn(X_test, batch_size=1).shape == (1,)


class TestCreateVitB16:
    def test_factory_without_pretrained(self) -> None:
        model, train_fn, inference_fn = dl.create_vit_b16(num_classes=6, pretrained=False)
        assert model.heads.head.out_features == 6

    def test_inference_smoke_224(self) -> None:
        model, train_fn, inference_fn = dl.create_vit_b16(
            num_classes=6, freeze_backbone=True, pretrained=False,
        )
        X = _image_batch(1, 3, 224, 224, seed=13)
        preds = inference_fn(X, batch_size=1)
        assert preds.shape == (1,)
        assert 0 <= preds[0] < 6

    def test_train_smoke_224(self) -> None:
        """ViT uses LayerNorm (no BatchNorm) so batch_size=1 is trainable."""
        model, train_fn, inference_fn = dl.create_vit_b16(
            num_classes=6, freeze_backbone=True, pretrained=False,
        )
        X_train = _image_batch(2, 3, 224, 224, seed=14)
        y_train = np.random.default_rng(15).integers(0, 6, 2)
        X_test = _image_batch(1, 3, 224, 224, seed=16)
        y_test = np.random.default_rng(17).integers(0, 6, 1)
        train_fn(X_train, y_train, X_test, y_test, epochs=1, batch_size=1)
        assert inference_fn(X_test, batch_size=1).shape == (1,)


class TestCreateLstmRul:
    def test_factory(self) -> None:
        model, train_fn, inference_fn = dl.create_lstm_rul(input_features=26)
        assert isinstance(model, torch.nn.Module)
        total_params = sum(p.numel() for p in model.parameters())
        # LSTM(26→128, 1 layer): 4 gates × (W_ih 26·128 + W_hh 128·128 + 2 biases·128)
        # plus Linear(128→1): 128 weights + 1 bias
        assert total_params == 4 * (26 * 128 + 128 * 128 + 2 * 128) + 128 + 1

    def test_train_and_inference_smoke(self) -> None:
        model, train_fn, inference_fn = dl.create_lstm_rul(input_features=5, hidden_size=8)
        rng = np.random.default_rng(0)
        X_train = rng.normal(size=(16, 30, 5)).astype(np.float32)
        y_train = rng.normal(size=16).astype(np.float32)
        X_test = rng.normal(size=(8, 30, 5)).astype(np.float32)
        y_test = rng.normal(size=8).astype(np.float32)
        train_fn(X_train, y_train, X_test, y_test, epochs=1, batch_size=8)
        preds = inference_fn(X_test, batch_size=4)
        assert preds.shape == (8,)


class TestCompareTrainingModes:
    def test_resnet_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []

        def fake_factory(num_classes, freeze_backbone, pretrained):
            calls.append((freeze_backbone, pretrained))
            return None, None, None

        monkeypatch.setattr(dl, "create_resnet50", fake_factory)
        results = dl.compare_training_modes(
            "resnet_50", np.zeros((1, 3, 32, 32), np.float32),
            np.zeros(1, np.int64), np.zeros((1, 3, 32, 32), np.float32), np.zeros(1, np.int64),
        )
        assert set(results) == {"from_scratch", "frozen_backbone", "full_finetune"}
        assert calls == [(False, False), (True, True), (False, True)]
        assert "description" in results["frozen_backbone"]

    def test_vit_modes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list = []

        def fake_factory(num_classes, freeze_backbone, pretrained):
            calls.append((freeze_backbone, pretrained))
            return None, None, None

        monkeypatch.setattr(dl, "create_vit_b16", fake_factory)
        results = dl.compare_training_modes(
            "vit_b_16", np.zeros((1, 3, 224, 224), np.float32),
            np.zeros(1, np.int64), np.zeros((1, 3, 224, 224), np.float32), np.zeros(1, np.int64),
        )
        assert set(results) == {"from_scratch", "frozen_backbone", "full_finetune"}
        assert calls == [(False, False), (True, True), (False, True)]


# ---------------------------------------------------------------------------
# Generative factories (fully faked — no network)
# ---------------------------------------------------------------------------

class TestGptManufacturingQueries:
    def test_no_api_key_returns_placeholder_answers(self, caplog: pytest.LogCaptureFixture) -> None:
        pytest.importorskip("openai")
        import logging

        caplog.set_level(logging.WARNING)
        client, query_fn, _ = create_gpt_manufacturing_queries(api_key="")
        assert client is None
        results = query_fn()
        assert len(results) == len(_default_manufacturing_qa_pairs())
        assert all(r["answer"] == "[NO_API_KEY]" for r in results)
        assert "No OpenAI API key" in caplog.text

    def test_fake_client_answers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("openai")
        import openai

        class FakeCompletions:
            def __init__(self, raise_error: bool = False) -> None:
                self.calls: list = []
                self.raise_error = raise_error

            def create(self, **kwargs: Any) -> SimpleNamespace:
                self.calls.append(kwargs)
                if self.raise_error:
                    raise RuntimeError("API down")
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="answer-42"))]
                )

        class FakeChat:
            def __init__(self) -> None:
                self.completions = FakeCompletions()

        class FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                self.kwargs = kwargs
                self.chat = FakeChat()

        monkeypatch.setattr(openai, "OpenAI", FakeClient)
        client, query_fn, _ = create_gpt_manufacturing_queries(api_key="sk-fake")
        assert isinstance(client, FakeClient)
        assert client.kwargs["api_key"] == "sk-fake"

        qa = [{"question": "Q?", "context": "C."}]
        results = query_fn(qa_pairs=qa, max_tokens=100)
        assert results[0]["answer"] == "answer-42"
        assert client.chat.completions.calls[0]["model"] == "gpt-4o-mini"
        assert client.chat.completions.calls[0]["max_tokens"] == 100

    def test_query_error_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("openai")
        import openai

        class FakeCompletions:
            def create(self, **kwargs: Any) -> SimpleNamespace:
                raise ConnectionError("offline")

        class FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                self.chat = SimpleNamespace(completions=FakeCompletions())

        monkeypatch.setattr(openai, "OpenAI", FakeClient)
        _, query_fn, _ = create_gpt_manufacturing_queries(api_key="sk-fake")
        results = query_fn(qa_pairs=[{"question": "Q?", "context": "C."}])
        assert results[0]["answer"].startswith("[ERROR:")

    def test_rate_limit_sleep_after_10(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("openai")
        import openai

        sleeps: list = []
        monkeypatch.setattr(
            "carbon_benchmark.models.generative.time.sleep", lambda s: sleeps.append(s),
        )

        class FakeCompletions:
            def create(self, **kwargs: Any) -> SimpleNamespace:
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
                )

        class FakeClient:
            def __init__(self, **kwargs: Any) -> None:
                self.chat = SimpleNamespace(completions=FakeCompletions())

        monkeypatch.setattr(openai, "OpenAI", FakeClient)
        _, query_fn, _ = create_gpt_manufacturing_queries(api_key="sk-fake")
        qa_pairs = [{"question": f"Q{i}?", "context": "C"} for i in range(11)]
        results = query_fn(qa_pairs=qa_pairs)
        assert len(results) == 11
        assert sleeps == [0.5]


class _FakePipe:
    """Stand-in for diffusers.StableDiffusionPipeline."""

    def __init__(self) -> None:
        self.calls: list = []
        self.device: Any = None

    def to(self, device: str) -> "_FakePipe":
        self.device = device
        return self

    def __call__(self, prompt: str, **kwargs: Any) -> SimpleNamespace:
        self.calls.append((prompt, kwargs))
        return SimpleNamespace(images=[np.zeros((8, 8, 3), dtype=np.uint8)])


class TestStableDiffusionPipeline:
    def test_factory_with_fake_pipe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("diffusers")
        import diffusers

        pipe = _FakePipe()
        from_pretrained_calls: list = []

        @classmethod
        def fake_from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakePipe:
            from_pretrained_calls.append((args, kwargs))
            return pipe

        monkeypatch.setattr(
            diffusers.StableDiffusionPipeline, "from_pretrained", fake_from_pretrained,
        )
        result_pipe, generate_fn, inference_fn = create_stable_diffusion_pipeline()
        assert result_pipe is pipe
        assert inference_fn is None
        assert from_pretrained_calls[0][0][0] == "runwayml/stable-diffusion-v1-5"
        assert pipe.device in ("cuda", "cpu")

    def test_generate_fn_explicit_prompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("diffusers")
        import diffusers

        pipe = _FakePipe()
        monkeypatch.setattr(
            diffusers.StableDiffusionPipeline, "from_pretrained",
            classmethod(lambda cls, *a, **k: pipe),
        )
        _, generate_fn, _ = create_stable_diffusion_pipeline()
        images = generate_fn(
            prompts=["bracket design"], n_per_prompt=5, num_inference_steps=3,
            height=64, width=64,
        )
        assert len(images) == 5
        assert all(isinstance(img, np.ndarray) for img in images)
        # The % 5 progress branch exercised (j=4)
        assert all(kw["num_inference_steps"] == 3 for _, kw in pipe.calls)
        assert all(kw["height"] == 64 for _, kw in pipe.calls)

    def test_generate_fn_default_prompts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("diffusers")
        import diffusers

        from carbon_benchmark.config import MANUFACTURING_DESIGN_PROMPTS

        pipe = _FakePipe()
        monkeypatch.setattr(
            diffusers.StableDiffusionPipeline, "from_pretrained",
            classmethod(lambda cls, *a, **k: pipe),
        )
        _, generate_fn, _ = create_stable_diffusion_pipeline()
        images = generate_fn()
        assert len(images) == len(MANUFACTURING_DESIGN_PROMPTS) * 10
        prompts_used = [p for p, _ in pipe.calls]
        assert prompts_used == [
            p for p in MANUFACTURING_DESIGN_PROMPTS for _ in range(10)
        ]


# ---------------------------------------------------------------------------
# Package re-exports
# ---------------------------------------------------------------------------

class TestPackageExports:
    def test_models_init_reexports(self) -> None:
        from carbon_benchmark import models

        for name in (
            "create_linear_regression", "create_decision_tree", "create_random_forest",
            "create_xgboost", "create_mobilenet_v2", "create_resnet50",
            "create_resnet152", "create_vit_b16", "create_lstm_rul",
            "compare_training_modes", "create_stable_diffusion_pipeline",
            "create_gpt_manufacturing_queries",
        ):
            assert callable(getattr(models, name)), name

