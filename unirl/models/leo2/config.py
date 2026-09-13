"""Leo2 pipeline config -- shared between bundle / pipeline / stages; see README.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

LEO2_VAE_LATENT_CHANNELS = 48
LEO2_VAE_SPATIAL = 16
LEO2_VAE_TEMPORAL = 4
LEO2_TIMESTEP_SCALE = 1000.0

# The hymm yaml the released iter_0063300 weights were trained under; rope-theta and
# fixed-space differ between stage yamls and a mismatch biases the prediction.
LEO2_CONFIG_RELPATH = "hymm/configs/leo2/leo2_moe_v1_1_a12b_muon_wzd_256p_stage2_part2.yaml"


@dataclass
class Leo2PipelineConfig:
    # --- code + weights (all site-specific; set them in the recipe or the environment) ---
    hymm_repo_path: Optional[str] = None
    config_yaml: Optional[str] = None
    ckpt_path: str = ""
    generation_config_path: str = ""
    assets_base: Optional[str] = None
    # Node-local mirrors of the text encoder / video VAE, used only when the directory
    # exists on this node; otherwise hymm reads the copies under ASSETS_BASE.
    text_encoder_base: Optional[str] = None
    vae_base: Optional[str] = None
    # Appended after the yaml; EP must stay 1 because UniRL's FSDP hosts the experts locally.
    extra_hymm_args: List[str] = field(
        default_factory=lambda: [
            "--bot-task",
            "video",
            "--use-system-prompt",
            "li-dit-encode-visual-qwen-3.5",
            "--gate-impl",
            "deepseek",
            "--vae-type",
            "16x16x4-48c-hy-v3_3-release2",
        ]
    )

    # --- precisions (H3-shaped knobs) ---
    model_precision: str = "bf16"
    autocast_precision: str = "bf16"
    trajectory_precision: str = "bf16"
    logprob_precision: str = "fp32"

    # --- schedule ---
    video_shift: float = 3.0

    # --- placement ---
    text_encoder_gpu_transient: bool = True  # park the 18 GB Qwen3.5-9B on CPU between encodes
    vae_on_gpu: bool = True

    # loading
    skip_load_ckpt: bool = False  # debug only: random weights
    # Build on meta and let the backend load post-wrap; see README.md '## Precision & loading'.
    meta_init_transformer: bool = False
    uniform_bf16: bool = True  # cast the fp32 MoE router too; see README.md '## Precision & loading'

    device: Optional[str] = None


__all__ = [
    "Leo2PipelineConfig",
    "LEO2_CONFIG_RELPATH",
    "LEO2_VAE_LATENT_CHANNELS",
    "LEO2_VAE_SPATIAL",
    "LEO2_VAE_TEMPORAL",
    "LEO2_TIMESTEP_SCALE",
]
