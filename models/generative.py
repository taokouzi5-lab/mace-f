"""
Generative AI model factories for manufacturing AI benchmarks (T5).

Models:
  T5: Stable Diffusion v1.5 — generative component design (inference only)
  T5: GPT-4o-mini — manufacturing process document Q&A (API inference only)

Both are inference-only measurements because:
  1. Training T5 models from scratch is prohibitively expensive (>100 kWh).
  2. Manufacturing use of generative AI is almost always via pre-trained
     models or APIs, not training from scratch.
  3. Per Luccioni et al. (2024), inference emissions for generative models
     are the dominant lifecycle contributor.

References:
  Luccioni et al. (2024) — Stable Diffusion XL: ~4.1 miles driving per 1000 images
  Patterson et al. (2022) — generative AI is orders of magnitude beyond T4
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from carbon_benchmark.config import MANUFACTURING_DESIGN_PROMPTS

logger = logging.getLogger(__name__)


# ======================================================================
# T5: Stable Diffusion v1.5 (inference only)
# ======================================================================

def create_stable_diffusion_pipeline(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    n_images_per_prompt: int = 10,
) -> Tuple[Any, Callable[..., Any], Callable[..., List[np.ndarray]]]:
    """Create a Stable Diffusion pipeline for manufacturing design generation.

    Use case: Generate concept images of mechanical components from text prompts.
    Inference only — model weights downloaded from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier.
        n_images_per_prompt: Number of images to generate per prompt (10 default).

    Returns:
        (pipeline, generate_fn, None) tuple.  inference_fn is None because
        generation IS the inference — use generate_fn directly.

    Reference:
      Luccioni et al. (2024): 1000 SDXL images ≈ 4.1 miles driving equivalent.
    """
    import torch

    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    logger.info(f"Loading Stable Diffusion pipeline ({model_id}) on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,  # Disable for manufacturing technical drawings
    )
    pipe = pipe.to(device)
    logger.info("Stable Diffusion pipeline loaded.")

    def generate_fn(
        prompts: List[str] = None,
        n_per_prompt: int = n_images_per_prompt,
        num_inference_steps: int = 50,
        height: int = 512,
        width: int = 512,
    ) -> List[np.ndarray]:
        """Generate manufacturing design images from text prompts.

        Args:
            prompts: List of text prompts.  Defaults to manufacturing prompts.
            n_per_prompt: Images per prompt (total = len(prompts) × n_per_prompt).
            num_inference_steps: Denoising steps (50 = quality; 20 = speed).
            height, width: Output image dimensions.

        Returns:
            List of PIL Images (len = len(prompts) × n_per_prompt).
        """
        if prompts is None:
            prompts = MANUFACTURING_DESIGN_PROMPTS

        all_images: list = []
        total = len(prompts) * n_per_prompt
        logger.info(
            f"Generating {total} images ({len(prompts)} prompts × "
            f"{n_per_prompt} each, {num_inference_steps} steps)"
        )

        for i, prompt in enumerate(prompts):
            logger.info(f"  Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
            for j in range(n_per_prompt):
                with torch.no_grad():
                    result = pipe(
                        prompt,
                        num_inference_steps=num_inference_steps,
                        height=height,
                        width=width,
                    )
                all_images.append(result.images[0])
                if (j + 1) % 5 == 0:
                    logger.info(f"    {j+1}/{n_per_prompt} images")

        logger.info(f"Generated {len(all_images)} images total.")
        return all_images

    logger.info("Created Stable Diffusion pipeline (T5, inference only)")
    return pipe, generate_fn, None  # No separate inference_fn


# ======================================================================
# T5: GPT-4o-mini (API inference only)
# ======================================================================

def create_gpt_manufacturing_queries(
    api_key: str = "",
    model_name: str = "gpt-4o-mini",
) -> Tuple[Any, Callable[..., Any], Callable[..., List[Dict[str, str]]]]:
    """Create a GPT-4o-mini API interface for manufacturing document Q&A.

    Use case: Answer manufacturing process queries from technical documentation.
    Inference only via OpenAI API — local energy tracked by CodeCarbon.

    IMPORTANT: CodeCarbon only measures LOCAL energy (CPU, network, RAM).
    Server-side emissions are estimated separately using published data.

    Args:
        api_key: OpenAI API key (reads from env var OPENAI_API_KEY if empty).
        model_name: OpenAI model identifier.

    Returns:
        (client, query_fn, None) tuple.
    """
    import os

    from openai import OpenAI

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        logger.warning(
            "No OpenAI API key found. Set OPENAI_API_KEY env var or pass api_key. "
            "GPT-4o-mini queries will be skipped."
        )

    client = OpenAI(api_key=api_key) if api_key else None

    def query_fn(
        qa_pairs: List[Dict[str, str]] = None,
        model: str = model_name,
        max_tokens: int = 300,
    ) -> List[Dict[str, str]]:
        """Query GPT-4o-mini with manufacturing QA pairs.

        Args:
            qa_pairs: List of {question, context} dicts.
            model: Model name.
            max_tokens: Maximum response tokens.

        Returns:
            List of {question, context, answer} dicts.
        """
        if qa_pairs is None:
            qa_pairs = _default_manufacturing_qa_pairs()

        if client is None:
            logger.warning("No OpenAI client — returning empty answers.")
            return [
                {**qa, "answer": "[NO_API_KEY]"}
                for qa in qa_pairs
            ]

        results: List[Dict[str, str]] = []
        logger.info(f"Querying {model} with {len(qa_pairs)} QA pairs...")

        for i, qa in enumerate(qa_pairs):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a manufacturing engineering assistant. "
                                "Answer concisely based on the provided context."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Context: {qa['context']}\n\n"
                                f"Question: {qa['question']}"
                            ),
                        },
                    ],
                    max_tokens=max_tokens,
                    temperature=0.1,  # Low temperature for factual answers
                )
                answer = response.choices[0].message.content
            except Exception as exc:
                logger.error(f"  Query {i+1}/{len(qa_pairs)} failed: {exc}")
                answer = f"[ERROR: {exc}]"

            results.append({**qa, "answer": answer or ""})

            if (i + 1) % 10 == 0:
                logger.info(f"  {i+1}/{len(qa_pairs)} queries complete")
                time.sleep(0.5)  # Rate limiting courtesy

        logger.info(f"Completed {len(results)} GPT queries.")
        return results

    logger.info(f"Created GPT-4o-mini interface (T5, API inference, model={model_name})")
    return client, query_fn, None


def _default_manufacturing_qa_pairs() -> List[Dict[str, str]]:
    """Return a minimal set of manufacturing QA pairs for benchmarking.

    In production, replace these with a full set of 50 pairs from real
    manufacturing process documentation.
    """
    return [
        {
            "question": "What is the recommended cutting speed for AISI 4140 steel?",
            "context": (
                "AISI 4140 is a chromium-molybdenum alloy steel with good "
                "hardenability. Typical cutting speeds: roughing 60-90 m/min, "
                "finishing 90-120 m/min with carbide tooling."
            ),
        },
        {
            "question": "What is the maximum operating temperature for this bearing?",
            "context": (
                "Deep groove ball bearing 6205-2RS. C3 internal clearance. "
                "Limiting speed 14,000 rpm. Operating temperature range "
                "-20°C to +120°C with standard grease."
            ),
        },
        {
            "question": "How should I store this adhesive before use?",
            "context": (
                "Two-part epoxy adhesive. Part A: resin, Part B: hardener. "
                "Shelf life 12 months. Store at 15-25°C in original sealed "
                "containers away from direct sunlight and moisture."
            ),
        },
    ]
