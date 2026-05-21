"""Model plugin examples.

Empty placeholder. All first-party models live under
``diffusionrl.models.*`` and register their Hydra config names
(``sd3``, ``wan21``, ``wan22``, ``qwen_image``,
``hunyuan_video15``, ``hunyuan_image3``) directly.

For a third-party model bundle, define your own subclass of
``diffusionrl.models.<base-Pipeline>`` and reference it via
``_target_:`` in an experiment YAML.
"""

__all__: list[str] = []
