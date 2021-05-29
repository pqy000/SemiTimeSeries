# Semi Time series classification
> The main idea is to combine the MeanTeacher with the series saliency module. While improving the accuracy of the model, it can also improves the interpretability quantitatively and qualitatitvely. Compared with the above work that only improves accuracy, it may provide more insights.

![image](http://i2.tiimg.com/695850/76c3f37c8527973c.png)

### Requirement
The package includes ```sklearn```, ```numpy,``` ```pytorch```..etc. If any packages are missiing, just use conda install.

### Structure
The structure of the software is redundant. ```mainOurs.py``` includes some options. The important parameters are in followings:

#### option
* `--dataset`
    * The experiments includes six datasets. ([Download link](https://cloud.tsinghua.edu.cn/d/b5e6a34ec6f74eb2a3bc/)) The previous papers mainly design experiments on the six datasets. Until now, for each dataset, I ran for 5 times (random seed 0,1,2) and recorded the mean and variance. As shown in the experiments, compared with the previous  **SOTA**  results, there is a significant improvement, and I almostly didn't tune the parameters.

* `--model_name`
    * It includes three opinions, of course, the code mainly focuses on the our method
        * `SupCE`: The supervised training procedure
        * `SemiTime`: The previous  **SOTA**  baselines
        * `SemiTeacher`: Our method ( **MeanTeacher**  +  **Series Saliency**).

* `--label_ratio`
    * The option is used to limit the proportion of labeled data.
* `--Saliency`
    * The option is to indicate whehter the use the series saliency module in the MeanTeacher training.
* Other parameters are some detailed parameters.

#### Directory

* `optim/` 
    * Under the `optim/` directory, there are some main optimization method
        * `generalWay.py` includes our implement method
        * `pretrain.py` includes the baseline
* `model/` 
    * The mainly architecture is Temporal Convolution neural network
* `Dataloader/`
    * The directory is important, including some dataloaders that read the UCR time series classification data. In the implementation, the consistency loss is also randomly selected from the labeled and unlabel data.

#### Usage example
After introducing the results of previous code, some examples for running commands.

```
python mainOurs.py --model_name SemiTeacher --dataset=CricketX --gpu=2 --label_ratio 0.4
```
```
python mainOurs.py --model_name SemiTime --dataset=CricketX --gpu=2 --label_ratio 0.4
```
```
python mainOurs.py --model_name SupCE --dataset=CricketX --gpu=2 --label_ratio 0.4
```

### Architecture

The model architecture is intuitive, which migrating the commonly used mean teacher method in the semi supervised learning of time series. We combine it with the previously designed series saliency module. As shown in the Figure, we will probably know the specific implementation method. The detail implementation in code. At present, the accuracy has been significantly improved. This is a good news! On the other hand, we proves the series saliency module is helpful in semi-supervised learning.

The second part is to used the series saliency for interpretation in time series semi-supervised learning. I have implemented the code before, also I'll migrate from time series forecasting to time series classification. We'll provide more quantitative and qualitative analysis. The motivation is to observe how the deep models learn the information with increasing label size. This may require more domain knowledge and cherry pick some visualization. 

Finally, I think combine with the improved accuracy with the interpretation results in the semi-supervised learning. I think there is some important contribution in the semi-supervised learning in time series classification.

## Experiments results

We mainly compare the lates two papers on time series supervised learning. [Second paper](https://haoyfan.github.io/papers/SemiTime_ICASSP2021.pdf) reproduces the results of [first paper](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_39).
Therefore, we will compare with their methods. The experiment results show that the series saliency is also a useful augmentation. Now the more visualization results will be added (like t-sne).


---

| Label Ratio       | 10%                           | 20%                    | 40%                    | 100%                  |
| ----------------- | ----------------------------- | ---------------------- | ---------------------- | --------------------- |
| **Dataset** | **CricketX**            |                        |                        |                       |
| SemiTime          | 44.88 (3.13)                  | 51.61 (1.22)           | 58.71 (2.78)           | 65.66 (1.58)          |
| MeanTeacher       | 39.54 (1.16)                  | 51.59 (1.98)           | 62.87 (1.69)           |                       |
| MT w/ SS          |                               |                        | **63.45** (1.28) |                       |
|                   |                               |                        |                        |                       |
| **Dataset** | **InsectWingbeatSound** |                        |                        |                       |
| SemiTime          | 54.96  (1.61)                 | 59.01 (1.56)           | 62.38 (0.76)           | 66.57 (0.67)          |
| MeanTeacher       | 56.33 (2.1)                   | 61.21 (2.17)           | 63.37(0.92)            | 67.53(1.98)           |
| MT w/ SS          | **57.24** (2.27)        | **61.47** (1.91) | **64.9** (2.1)   | **68.99**(1.98) |
|                   |                               |                        |                        |                       |
| **Dataset** | MFPT                          |                        |                        |                       |
| SemiTime          | 64.16(0.85)                   | 69.84(0.94)            | 76.49 (0.54)           | 84.33(0.50)           |
| MeanTeacher       |                               |                        |                        |                       |
| MT w/ SS          |                               |                        |                        |                       |
|                   |                               |                        |                        |                       |
| **Dataset** | Uwave                         |                        |                        |                       |
| SemiTime          | 81.46(0.60)                   | 84.57(0.49)            | 86.91(0.47)            | 90.29(0.32)           |
| MeanTeacher       | **92.28** (0.51)        | **94.94**(0.68)  | **96.36**(0.7)   |                       |
| MT w/ SS          |                               |                        |                        |                       |
|                   |                               |                        |                        |                       |
| **Dataset** | Epilep                        |                        |                        |                       |
| SemiTime          | 74.86(0.42)                   | 75.54(0.63)            | 77.01(0.79)            | 79.26(1.20)           |
| MeanTeacher       |                               |                        |                        |                       |
| MT w/ SS          |                               |                        |                        |                       |
|                   |                               |                        |                        |                       |




## Reference
The two papers were not presented at the top-tier conference. I think the main reason is lack of further analysis.

[1][SEMI-SUPERVISED TIME SERIES CLASSIFICATION BY TEMPORAL RELATION PREDICTION](https://haoyfan.github.io/papers/SemiTime_ICASSP2021.pdf)

[2][Self-Supervised Time Series Representation Learning by Inter-Intra Relational Reasoning](https://openreview.net/pdf?id=qFQTP00Q0kp)

[3][Self-supervised Learning for Semi-supervised Time Series Classification](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_39)



