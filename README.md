# Introduction

Unofficialed PyTorch implementation of [Unsupervised Coherent Video Cartoonization with Perceptual Motion Consistency](https://arxiv.org/abs/2204.00795) (CVPR 2022)

The code is refer to the original [implementation](https://github.com/Victarry/PMC-Video-Animation/) by the author and it's base project: [White-box-Cartoonization](https://github.com/SystemErrorWang/White-box-Cartoonization/) in Tensorflow.

The main difference is I use original PyTorch 2.0.0, and the pretrained network in torchvision istead of self-pretrained network. I also adjust the hyperparameters to fit the images into $[0,1]$.

# Dependencies

- PyTorch 2.0.0 (Install it using the command given in official website)
- scikit-image >= 0.20.0rc8
- tqdm
- joblib

# Usage
- Change the parameters in `config.py` if needed.
- Change the path of the dataset in `pataset.py`.
- makedir `model` and `result` in the root folder.
- Run `python main.py` to start training.

The middle results will be put in the root folder with name `tt.jpg`.

After training, run `eval.py` to generate results to the result folder.

You may use the pytorch-fid to evaluate the results.

# Citing

Consider using following bibtex to cite this work:

```
@misc{liu2022unsupervised,
      title={Unsupervised Coherent Video Cartoonization with Perceptual Motion Consistency}, 
      author={Zhenhuan Liu and Liang Li and Huajie Jiang and Xin Jin and Dandan Tu and Shuhui Wang and Zheng-Jun Zha},
      year={2022},
      eprint={2204.00795},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```
@misc{LW_PyTorchCartoonSSA,
      title={Cartoonization SSA PyTorch}, 
      author={Ling Wang},
      year={2023},
      howpublish={https://github.com/wychlw/Cartoonization-PyTorch-SSA}
}
```