# CAMix
(TCSVT 2022) Context-Aware Mixup for Domain Adaptive Semantic Segmentation

## Citing CAMix
If you find CAMix useful in your research, please consider citing:
```bibtex
@ARTICLE{zhou2021context,
  author={Zhou, Qianyu and Feng, Zhengyang and Gu, Qiqi and Pang, Jiangmiao and Cheng, Guangliang and Lu, Xuequan and Shi, Jianping and Ma, Lizhuang},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Context-Aware Mixup for Domain Adaptive Semantic Segmentation}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TCSVT.2022.3206476}
}
```

### Requirements
*  CUDA/CUDNN 
*  Python3
*  Pytorch
*  Scipy==1.2.0
*  Other requirements
    ```bash
    pip install -r requirements.txt
    ```

# Run training and testing

### Example of training a model with unsupervised domain adaptation on GTA5->CityScapes on a single gpu

python3 train_ucspa_gtav.py --config ./configs/configUDA_ucspa_gtav.json --name UDA

### Example of testing a model with domain adaptation with CityScapes as target domain

python3 evaluateUDA.py --model-path *checkpoint.pth*


The key code of CAMix could be referred in L137-224 of camix_transforms.py

## Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [DACS](https://github.com/vikolss/DACS)


## License

This project is released under the [Apache License 2.0](LICENSE), while some 
specific features in this repository are with other licenses. Please refer to 
[LICENSES.md](LICENSES.md) for the careful check, if you are using our code for 
commercial matters.
