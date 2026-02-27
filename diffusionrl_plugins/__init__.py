"""
diffusionrl_plugins — external / third-party extension namespace.

This package is a plain Python namespace.  There is NO auto-discovery or
auto-registration magic.  To use a plugin, pass its full dotpath via
the corresponding CLI argument:

    --model-path   diffusionrl_plugins.models.wan21.Wan21ModelBundle
    --sampler-path diffusionrl_plugins.samplers.my_sampler.MySampler
    --algorithm-path diffusionrl_plugins.algorithms.my_algo.MyAlgorithm
    --loss-path diffusionrl_plugins.losses.my_loss.MyLoss

Plugin paths are validated by importability only.

Algorithm and loss are separate extension points:
- algorithm: rollout requirements + advantage logic + timestep filtering
- loss: gradient objective used by TrainingActor/TrainExecutor

For new plugin algorithms/losses, import shared data types from:

    diffusionrl.types
"""
