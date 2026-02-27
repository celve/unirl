"""
diffusionrl_plugins — external / third-party extension namespace.

This package is a plain Python namespace.  There is NO auto-discovery or
auto-registration magic.  To use a plugin, pass its full dotpath via
the corresponding CLI argument:

    --model-path   diffusionrl_plugins.models.wan21.Wan21ModelBundle
    --sampler-path diffusionrl_plugins.samplers.minimal_sampler.MinimalSampler
    --algorithm-path diffusionrl_plugins.algorithms.minimal_algorithm.MinimalAlgorithm
    --loss-type custom --loss-path diffusionrl_plugins.losses.minimal_loss.MinimalBackwardLoss
    --reward-path diffusionrl_plugins.rewards.minimal_reward.MinimalRewardWorker
    --rollout-pipeline-path diffusionrl_plugins.rollout_fns.minimal_pipeline.minimal_pipeline
    --rollout-buffer-plugin-paths your_module.your_buffer_plugin

Plugin paths are validated by importability only.

Algorithm and loss are separate extension points:
- algorithm: rollout requirements + advantage logic + timestep filtering
- loss: gradient objective used by TrainingActor/TrainExecutor

Model short-name resolution (`--model-type your_model`) works when the model
class declares:
    - declared_model_type()
    - default_sampler_path()
    - default_sampler_engine()

For new plugin algorithms/losses, import shared data types from:

    diffusionrl.types
"""
