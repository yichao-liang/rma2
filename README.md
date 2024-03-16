Repo for Rapid Motor Adaptation for Robotic Manipulator Arms ([link]([www.google.com](https://arxiv.org/abs/2312.04670))). It's still under active construction.

# Installation
1. Clone the repo:
```
git clone --recurse-submodules https://github.com/yichao-liang/rma2
```

2. Create a conda environment:
```
conda env create -f environment_copy.yml
```

3. Train a model by, for example:
```
python main.py -n 50 -bs 5000 -rs 2000 \
            --randomized_training --ext_disturbance --obs_noise \
            -e PickSingleYCB-v1 
```
