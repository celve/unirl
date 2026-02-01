# Optional Geneval MMCV/MMDetection Setup

This setup is optional. Only install these packages if your workflow needs the
Geneval/OpenMMLab stack.

## Why separate?

`mmcv`/`mmdet` have CUDA build constraints and are not required for core
DiffusionRL training pipelines.

## Recommended versions

- `mmcv` tag: `v1.7.2`
- `mmdetection` branch: `2.x` (currently `2.28.2` in our validated env)

These versions match the validated local setup used in this repository.

## Install steps

Run from any working directory (examples use `~/mmgrpo`):

```bash
pip install -U openmim
mim install mmengine

cd ~/mmgrpo
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv
git checkout tags/v1.7.2
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -e . -v
cd ..

git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout 2.x
pip install -e . -v
cd ..
```

## Sanity check

```bash
python -c "import mmcv, mmdet; print('mmcv', mmcv.__version__); print('mmdet', mmdet.__version__); from mmcv.ops import nms; print('mmcv.ops ok')"
```

Expected:

- `mmcv` reports `1.7.2`
- `mmdet` reports a `2.x` version (e.g., `2.28.2`)
- `mmcv.ops ok` prints without import errors
