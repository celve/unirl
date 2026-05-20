"""Model plugin examples.

Empty placeholder. All first-party models live under
``diffusionrl.models_new.*`` and register their Hydra config names
(``sd3_v2``, ``wan21_v2``, ``wan22_v2``, ``qwen_image_v2``,
``hunyuan_video15_v2``, ``hunyuan_image3_v2``) directly.

For a third-party model bundle, define your own subclass of
``diffusionrl.models_new.<base-Pipeline>`` and reference it via
``_target_:`` in an experiment YAML.
"""

__all__: list[str] = []
