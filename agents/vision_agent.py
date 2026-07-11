"""Vision agent interface and implementations.

Defines ONE interface that every vision backend must satisfy. Everything
downstream depends ONLY on this contract — never on ResNet18 specifically.

Two implementations:
  - EnsembleVisionAgent: loads N checkpoints, aggregates softmax distributions
  - MockVisionAgent: synthetic but realistic probability distributions
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: torchvision is imported lazily inside _ResNet18Wrapper so that the
# MockVisionAgent path (and the whole agentic pipeline) works even when
# torchvision is not installed. Only the real EnsembleVisionAgent needs it.


# ── Contract ─────────────────────────────────────────────────────────────────


@dataclass
class VisionPrediction:
    """Contract every vision backend must return."""

    label: str
    confidence: float
    probs: dict[str, float]
    agreement_ratio: float
    per_model_preds: list[str]


class VisionAgent(ABC):
    """Abstract base — every backend implements this."""

    @abstractmethod
    def predict(self, image: Any) -> VisionPrediction:
        ...


# ── Helpers ──────────────────────────────────────────────────────────────────

_CLASS_NAMES_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "class_names.json"
)


def _load_class_names() -> list[str]:
    with open(_CLASS_NAMES_PATH) as f:
        return json.load(f)


def _clean_class_name(raw: str) -> str:
    """Turn raw PlantVillage class name into a readable query string."""
    # Remove triple-underscore disease separator
    parts = raw.split("___")
    crop = parts[0].replace("_", " ").replace("(", "").replace(")", "").strip()
    if len(parts) > 1 and "healthy" in parts[1].lower():
        return f"{crop} healthy"
    disease = (
        parts[1]
        .replace("_", " ")
        .replace("(", "")
        .replace(")", "")
        .strip()
    )
    return f"{crop} {disease}"


# ── EnsembleVisionAgent ─────────────────────────────────────────────────────


class _ResNet18Wrapper(nn.Module):
    """Load a ResNet18 checkpoint as a clean classifier."""

    def __init__(self, checkpoint_path: str | Path, num_classes: int = 38):
        super().__init__()
        from torchvision import models  # lazy: only the real backend needs it

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(512, num_classes)
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class EnsembleVisionAgent(VisionAgent):
    """Loads N model checkpoints and aggregates predictions.

    Works correctly with N=1 today (our ResNet18), scales to N=4 later
    with no changes elsewhere. Adding a new model means only appending
    a checkpoint path to `checkpoint_paths` — nothing else changes.
    """

    def __init__(
        self,
        checkpoint_paths: list[str | Path],
        class_names: list[str] | None = None,
        device: str = "cpu",
    ):
        self.class_names = class_names or _load_class_names()
        self.device = device
        self.num_classes = len(self.class_names)

        self.models: list[_ResNet18Wrapper] = []
        for path in checkpoint_paths:
            model = _ResNet18Wrapper(path, self.num_classes)
            model.to(self.device)
            model.eval()
            self.models.append(model)

        # ImageNet normalization for PlantVillage
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def _preprocess(self, image: Any) -> torch.Tensor:
        """Convert input image to normalized tensor."""
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            return image.to(self.device)

        from PIL import Image

        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
            image = image.unsqueeze(0)
            return image.to(self.device)

        raise TypeError(f"Unsupported image type: {type(image)}")

    @torch.no_grad()
    def predict(self, image: Any) -> VisionPrediction:
        tensor = self._preprocess(image)

        all_probs: list[np.ndarray] = []
        individual_labels: list[str] = []

        for model in self.models:
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            individual_labels.append(self.class_names[int(np.argmax(probs))])

        # Average softmax distributions across all models
        avg_probs = np.mean(all_probs, axis=0)
        final_idx = int(np.argmax(avg_probs))
        final_label = self.class_names[final_idx]
        confidence = float(avg_probs[final_idx])

        # Margin = top1 - top2
        sorted_probs = np.sort(avg_probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        # Agreement ratio
        agreement_count = sum(
            1 for lbl in individual_labels if lbl == final_label
        )
        agreement_ratio = agreement_count / len(self.models)

        # Full probs dict
        probs_dict = {
            self.class_names[i]: float(avg_probs[i])
            for i in range(self.num_classes)
        }

        return VisionPrediction(
            label=final_label,
            confidence=confidence,
            probs=probs_dict,
            agreement_ratio=agreement_ratio,
            per_model_preds=individual_labels,
        )


# ── MockVisionAgent ─────────────────────────────────────────────────────────


class MockVisionAgent(VisionAgent):
    """Returns synthetic but realistic probability distributions.

    Enables the entire agentic pipeline to be built, run, and evaluated
    end-to-end even if the real checkpoint is broken or missing.
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        seed: int | None = None,
        accuracy: float = 0.82,
    ):
        self.class_names = class_names or _load_class_names()
        self._class_index = {cn: i for i, cn in enumerate(self.class_names)}
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        # Simulated top-1 accuracy when running in oracle mode (see predict()).
        self.accuracy = accuracy

    def predict(self, image: Any) -> VisionPrediction:
        num_classes = len(self.class_names)

        # ── Oracle hint ──────────────────────────────────────────────────
        # For a meaningful evaluation, main.py passes the ground-truth class
        # name as `image`. The mock then behaves like a realistic classifier:
        # correct ~`self.accuracy` of the time, with confidence correlated to
        # correctness (right answers skew confident, wrong answers skew
        # uncertain). Real backends receive real images, so a bare string is
        # unambiguously a mock oracle hint. When no hint is given (image is
        # None), predictions are random but still span all decision tiers.
        oracle_idx: int | None = None
        if isinstance(image, str) and image in self._class_index:
            oracle_idx = self._class_index[image]

        if oracle_idx is not None:
            is_correct = self.rng.random() < self.accuracy
            if is_correct:
                true_idx = oracle_idx
                # A competent classifier is usually confident when right.
                scenario = self.rng.choices(
                    ["direct", "low_margin", "moderate"],
                    weights=[0.70, 0.15, 0.15],
                )[0]
            else:
                # Wrong answer: anchor on some other class, and skew toward
                # low-confidence / confused so the agentic layer can catch it.
                true_idx = self.rng.randrange(num_classes)
                while true_idx == oracle_idx:
                    true_idx = self.rng.randrange(num_classes)
                scenario = self.rng.choices(
                    ["moderate", "low_conf", "confused", "direct"],
                    weights=[0.30, 0.30, 0.30, 0.10],
                )[0]
        else:
            # No ground-truth hint: random anchor, uniform tier coverage.
            true_idx = self.rng.randint(0, num_classes - 1)
            scenario = self.rng.choice(
                ["direct", "moderate", "low_margin", "low_conf", "confused"]
            )

        # Each scenario targets a specific (top1, top2) probability band, then
        # we synthesize a realistic-looking softmax hitting those targets:
        #   direct        -> high conf, wide margin        -> Tier 1
        #   moderate      -> mid conf                      -> Tier 2 (retrieve)
        #   low_margin    -> high conf but top1~top2 close -> Tier 2 (retrieve)
        #   low_conf      -> below escalation threshold    -> Tier 3
        #   confused      -> low conf + ensemble disagree  -> Tier 3

        if scenario == "direct":
            top1 = self.rng.uniform(0.88, 0.98)
            margin_target = self.rng.uniform(0.20, top1 - 0.02)
        elif scenario == "moderate":
            top1 = self.rng.uniform(0.60, 0.82)
            margin_target = self.rng.uniform(0.18, min(0.35, top1 - 0.05))
        elif scenario == "low_margin":
            top1 = self.rng.uniform(0.86, 0.95)
            margin_target = self.rng.uniform(0.02, 0.12)  # < 0.15 -> retrieve
        elif scenario == "low_conf":
            top1 = self.rng.uniform(0.35, 0.52)
            margin_target = self.rng.uniform(0.03, 0.10)
        else:  # confused
            top1 = self.rng.uniform(0.20, 0.45)
            margin_target = self.rng.uniform(0.01, 0.06)

        top2 = max(0.0, top1 - margin_target)

        # Distribute the remaining mass across the other classes as a
        # gently-decaying tail so the vector looks like a real softmax.
        remaining = max(0.0, 1.0 - top1 - top2)
        tail = self.np_rng.dirichlet(np.ones(num_classes - 2) * 0.4)
        tail = tail * remaining

        raw_probs = np.empty(num_classes, dtype=np.float64)
        # Place top1 on the anchored true class, top2 on a distinct runner-up.
        runner_idx = (true_idx + 1) % num_classes
        raw_probs[true_idx] = top1
        raw_probs[runner_idx] = top2
        tail_iter = iter(tail)
        for i in range(num_classes):
            if i not in (true_idx, runner_idx):
                raw_probs[i] = next(tail_iter)
        raw_probs = raw_probs / raw_probs.sum()

        final_idx = int(np.argmax(raw_probs))
        final_label = self.class_names[final_idx]
        confidence = float(raw_probs[final_idx])

        sorted_probs = np.sort(raw_probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1])

        # Simulate ensemble disagreement (N=1 mock pretends to be a voter set).
        if scenario == "confused":
            agreement_ratio = self.rng.choice([0.0, 0.33, 0.5])
            fake_pred_idx = self.rng.randint(0, num_classes - 1)
            while fake_pred_idx == final_idx:
                fake_pred_idx = self.rng.randint(0, num_classes - 1)
            per_model = [final_label, self.class_names[fake_pred_idx]]
        else:
            agreement_ratio = 1.0
            per_model = [final_label]

        probs_dict = {
            self.class_names[i]: float(raw_probs[i])
            for i in range(num_classes)
        }

        return VisionPrediction(
            label=final_label,
            confidence=confidence,
            probs=probs_dict,
            agreement_ratio=agreement_ratio,
            per_model_preds=per_model,
        )


# ── Factory ──────────────────────────────────────────────────────────────────


def create_vision_agent(
    mode: str = "mock",
    checkpoint_paths: list[str | Path] | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> VisionAgent:
    """Create a vision agent by config flag.

    Args:
        mode: "ensemble" for real models, "mock" for synthetic.
        checkpoint_paths: Paths to .pth files (required for ensemble).
        device: Torch device.
        seed: Random seed for MockVisionAgent reproducibility.

    Returns:
        VisionAgent implementation.
    """
    if mode == "ensemble":
        if not checkpoint_paths:
            raise ValueError("checkpoint_paths required for ensemble mode")
        return EnsembleVisionAgent(
            checkpoint_paths=checkpoint_paths, device=device
        )
    elif mode == "mock":
        return MockVisionAgent(seed=seed)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'ensemble' or 'mock'.")
