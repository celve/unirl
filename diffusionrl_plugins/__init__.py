"""
diffusionrl_plugins — external / third-party extension namespace.

This package is a plain Python namespace.  There is NO auto-discovery or
auto-registration magic.  To use a plugin, pass its full dotpath via
the corresponding CLI argument:

    --model.model-type your_module.your_model.YourModel
    --sampler.sampler-type diffusionrl_plugins.samplers.minimal_sampler.MinimalSampler
    --algorithm.algorithm-type diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm
    --reward.reward-type diffusionrl_plugins.rewards.minimal_reward.MinimalRewardScorer
    --rollout-buffer-plugin-paths your_module.your_buffer_plugin

Plugin paths are validated by importability only.

Algorithm is the extension point for rollout requirements, advantage logic,
and gradient objective ownership.

Model short-name resolution (`--model-type your_model`) works when the model
class declares:
    - declared_model_type()
    - default_sampler_path()
    - default_sampler_engine()

For new plugin algorithms, import shared data types from:

    diffusionrl.types
"""
