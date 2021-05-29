# Semi Time series classification
> The main idea is to combine the MeanTeacher with the series saliency module. While improving the accuracy of the model, it can also enhance the interpretability quantitatively and qualitatively. Compared with the above work that only improves accuracy, it may provide more insights.


![image](http://i2.tiimg.com/695850/76c3f37c8527973c.png)

## Environment
The package ```sklearn```, ```numpy,``` ```pytorch```..etc.

## Dataset
The data composed of 6 publicly available datasets downloadable from ([Download link](https://cloud.tsinghua.edu.cn/d/b5e6a34ec6f74eb2a3bc/)).The following are the detailed parameters of the three data sets I have completed the experiment.

| Dataset                | Train | Test | Dimension | Class |
| ---------------------- | ----- | ---- | --------- | ----- |
| UWaveGestureLibraryAll | 2688  | 894  | 945       | 8     |
| CricketX               | 458   | 156  | 300       | 12    |
| InsectWingbeatSound    | 1320  | 440  | 256       | 11    |


## Structure
 ```mainOurs.py``` includes some options. The following are the important options, the python script takes along with their description:
### option
* `--dataset`
    * The experiments include six datasets. The previous papers mainly design experiments on the six datasets. Up until now, for each dataset, I ran for 5 times (random seed 0,1,2) and recorded the mean and variance. As shown in the experiments, there is a significant improvement compared with the previous  **SOTA**  results.

* `--model_name`
    * It includes three opinions.
        * `SupCE`: The supervised training procedure
        * `SemiTime`: The previous  **SOTA**  baselines
        * `SemiTeacher`: Our method ( **MeanTeacher**  +  **Series Saliency**).

* `--label_ratio`
    * The option is used to limit the proportion of labeled data.
* `--Saliency`
    * The option is to indicate whether using series saliency module in the MeanTeacher.
* Other parameters are some detailed parameters.

### Directory

* `optim/` 
    * Under the `optim/` directory, there are several semi supervised learning method.
        * `generalWay.py` includes our implement method
        * `pretrain.py` includes the baseline
* `model/` 
    * The mainly DL architecture is Temporal Convolution neural network
* `Dataloader/`
    * The directory is necessary, including some data loaders that read the UCR time series classification data. The data used to calculate the consistency loss sample from both labelled and unlabeled data in our implementation.

### Usage example
After introducing the results of previous code, some examples for running commands.

```
## MeanTeacher
python mainOurs.py --model_name SemiTeacher --dataset=CricketX --gpu=2 --label_ratio 0.4
```
```
## Semi time Method
python mainOurs.py --model_name SemiTime --dataset=CricketX --gpu=2 --label_ratio 0.4
```
```
## Supervised method
python mainOurs.py --model_name SupCE --dataset=CricketX --gpu=2 --label_ratio 0.4
```

## Architecture

The model architecture is intuitive, which migrates the commonly used **Mean Teacher** method to the semi-supervised learning of time series. We combine it with the previously proposed **series saliency** module. As shown in the figure, we can guess the design idea of the model. The implementation details are in code. At present, the algorithm significantly improves accuracy. **This is good news!** 🎉 🎉 😄 On the other hand, we validated the series saliency module is helpful in semi-supervised learning.

The second part is to use the series saliency for interpretation in time series semi-supervised learning. I will implement the codes, and migrate from time series forecasting to time series classification. We'll provide more quantitative and qualitative analysis. The motivation is to observe learning procedure with increasing label size. The phenomenon may require more domain knowledge and cherry-pick some visualization.

Finally, I think that easy-to-implementation series saliency can significantly improve prediction accuracy and interpretability, contributing to the time series semi-supervised learning!

## Experiments results

We mainly compare the latest two papers on time series supervised learning. The [second paper](https://haoyfan.github.io/papers/SemiTime_ICASSP2021.pdf) reproduces the results of the [first paper](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_39). Therefore, we will compare their methods. The experiment results show that the series saliency is also an effective augmentation. Now the more visualization results will be added (like t-sne).

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
| **Dataset** | Uwave                         |                        |                        |                       |
| SemiTime          | 81.46(0.60)                   | 84.57(0.49)            | 86.91(0.47)            | 90.29(0.32)           |
| MeanTeacher       | **92.28** (0.51)              | **94.94**(0.68)        | **96.36**(0.7)         |                       |
| MT w/ SS          |                               |                        |                        |                       |
|                   |                               |                        |                        |                       |
| **Dataset** | MFPT                          |                        |                        |                       |
| SemiTime          | 64.16(0.85)                   | 69.84(0.94)            | 76.49 (0.54)           | 84.33(0.50)           |
| MeanTeacher       |                               |                        |                        |                       |
| MT w/ SS          |                               |                        |                        |                       |
|                   |                               |                        |                        |                       |
| **Dataset** | Epilep                        |                        |                        |                       |
| SemiTime          | 74.86(0.42)                   | 75.54(0.63)            | 77.01(0.79)            | 79.26(1.20)           |
| MeanTeacher       |                               |                        |                        |                       |
| MT w/ SS          |                               |                        |                        |                       |
|                   |                               |                        |                        |                       |




## Reference
The two papers were not presented at the top-tier conference. I think the main reason is the lack of further analysis for semi-supervised learning.

[1][SEMI-SUPERVISED TIME SERIES CLASSIFICATION BY TEMPORAL RELATION PREDICTION](https://haoyfan.github.io/papers/SemiTime_ICASSP2021.pdf)

[2][Self-Supervised Time Series Representation Learning by Inter-Intra Relational Reasoning](https://openreview.net/pdf?id=qFQTP00Q0kp)

[3][Self-supervised Learning for Semi-supervised Time Series Classification](https://link.springer.com/chapter/10.1007/978-3-030-47426-3_39)



