"""Confidence and image quality assessment agent.

Evaluates classifier output quality and image fitness for diagnosis.
Returns: {confidence, margin, agreement_ratio, quality_ok}

Quality checks:
  - Laplacian-variance blur detection
  - Brightness range check
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


@dataclass
class QualityAssessment:
    """Result of confidence and quality evaluation."""

    confidence: float
    margin: float
    agreement_ratio: float
    quality_ok: bool
    blur_score: float
    brightness_ok: bool
    quality_reasons: list[str]


class ConfidenceQualityAgent:
    """Evaluates prediction confidence and image quality.

    Args:
        blur_threshold: Minimum Laplacian variance to consider image sharp.
        brightness_min: Minimum mean pixel brightness (0-255 scale).
        brightness_max: Maximum mean pixel brightness (0-255 scale).
    """

    def __init__(
        self,
        blur_threshold: float = 50.0,
        brightness_min: float = 40.0,
        brightness_max: float = 220.0,
    ):
        self.blur_threshold = blur_threshold
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max

    def assess(
        self,
        confidence: float,
        margin: float,
        agreement_ratio: float,
        image: np.ndarray | None = None,
    ) -> QualityAssessment:
        """Run all quality checks and return combined assessment.

        Args:
            confidence: Classifier confidence (max softmax).
            margin: Top1 - top2 probability gap.
            agreement_ratio: Fraction of ensemble models agreeing on winner.
            image: Optional numpy image array (H, W, C) or (H, W) for
                   quality checks. If None, quality_ok defaults to True.

        Returns:
            QualityAssessment with all metrics.
        """
        reasons: list[str] = []

        # ── Blur check ───────────────────────────────────────────────────
        blur_score = 0.0
        if image is not None:
            blur_score = self._laplacian_variance(image)
            if blur_score < self.blur_threshold:
                reasons.append(
                    f"Image is blurry (Laplacian variance={blur_score:.1f}, "
                    f"threshold={self.blur_threshold})"
                )

        # ── Brightness check ─────────────────────────────────────────────
        brightness_ok = True
        if image is not None:
            brightness_ok = self._check_brightness(image)
            if not brightness_ok:
                mean_brightness = float(np.mean(image))
                reasons.append(
                    f"Image brightness out of range "
                    f"(mean={mean_brightness:.1f}, "
                    f"range=[{self.brightness_min}, {self.brightness_max}])"
                )

        # ── Determine overall quality ────────────────────────────────────
        quality_ok = blur_score >= self.blur_threshold and brightness_ok
        # If no image provided, quality is assumed OK (e.g., mock mode)
        if image is None:
            quality_ok = True

        return QualityAssessment(
            confidence=confidence,
            margin=margin,
            agreement_ratio=agreement_ratio,
            quality_ok=quality_ok,
            blur_score=blur_score,
            brightness_ok=brightness_ok,
            quality_reasons=reasons,
        )

    def _laplacian_variance(self, image: np.ndarray) -> float:
        """Compute Laplacian variance as a blur metric.

        Low values indicate blurry images. Typical sharp images have
        variance > 100; very blurry images are < 20.
        Falls back to 0.0 if OpenCV is not installed.
        """
        if not _HAS_CV2:
            return 0.0
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        gray = gray.astype(np.float64)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def _check_brightness(self, image: np.ndarray) -> bool:
        """Check if image brightness is within acceptable range."""
        mean_brightness = float(np.mean(image))
        return self.brightness_min <= mean_brightness <= self.brightness_max
