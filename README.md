# Semi Time series classification
> The main idea is to combine the MeanTeacher with the series saliency module. While improving the accuracy of the model, it can also improves the interpretability quantitatively and qualitatitvely. Compared with the above work that only improves accuracy, it may provide more insights.

[Google](http://www.google.com/)

### Requirement
The package includes ```sklearn```, ```numpy,``` ```pytorch```..etc. If any packages are missiing, just use conda install.

### Structure
The structure of the software is redundant. The important file is ```mainOurs.py```, which also includes some options. The important parameters are followings:

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress


OS X & Linux:

```sh
sh run.sh
```

## Usage example

A few motivating and useful examples of how your product can be used. Spice this up with code blocks and potentially more screenshots.

_For more examples and usage, please refer to the [Wiki][wiki]._

## Development setup

Describe how to install all development dependencies and how to run an automated test-suite of some kind. Potentially do this for multiple platforms.

```sh
make install
npm test
```

## Release History

* 0.2.1
    * CHANGE: Update docs (module code remains unchanged)
* 0.2.0
    * CHANGE: Remove `setDefaultXYZ()`
    * ADD: Add `init()`
* 0.1.1
    * FIX: Crash when calling `baz()` (Thanks @GenerousContributorName!)
* 0.1.0
    * The first proper release
    * CHANGE: Rename `foo()` to `bar()`
* 0.0.1
    * Work in progress

## Experiment results

Semi Time represent the sota results in semi supervised time series classification 

– [@YourTwitter](https://twitter.com/dbader_org) – YourEmail@example.com

| Label Ratio       | 10%                           | 20%                    | 40%                   | 100%         |
| ----------------- | ----------------------------- | ---------------------- | --------------------- | ------------ |
| **Dataset** | **CricketX**            |                        |                       |              |
| SemiTime          | 44.88 (3.13)                  | 51.61 (1.22)           | 58.71 (2.78)          | 65.66 (1.58) |
| MeanTeacher       | 39.54 (1.16)                  | 51.59 (1.98)           | 62.87 (1.69)          |              |
| MT w/ SS          |                               |                        | **63.45** (1.28)     |              |
|                   |                               |                        |                       |              |
| **Dataset** | **InsectWingbeatSound** |                        |                       |              |
| SemiTime          | 54.96  (1.61)                 | 59.01 (1.56)           | 62.38 (0.76)          | 66.57 (0.67) |
| MeanTeacher       | **57.95** (1.64)        | **61.47** (1.58) | **64.29**(1.18) |              |
| MT w/ SS          |                               |                        |                       |              |
|                   |                               |                        |                       |              |
| **Dataset** | MFPT                          |                        |                       |              |
| SemiTime          | 64.16(0.85)                   | 69.84(0.94)            | 76.49 (0.54)          | 84.33(0.50)  |
| MeanTeacher       |                               |                        |                       |              |
| MT w/ SS          |                               |                        |                       |              |
|                   |                               |                        |                       |              |
| **Dataset** | Uwave                         |                        |                       |              |
| SemiTime          | 81.46(0.60)                   | 84.57(0.49)            | 86.91(0.47)           | 90.29(0.32)  |
| MeanTeacher       | **92.28** (0.51)        | **94.94**(0.68)  | **96.36**(0.7)  |              |
| MT w/ SS          |                               |                        |                       |              |
|                   |                               |                        |                       |              |
| **Dataset** | Epilep                        |                        |                       |              |
| SemiTime          | 74.86(0.42)                   | 75.54(0.63)            | 77.01(0.79)           | 79.26(1.20)  |
| MeanTeacher       |                               |                        |                       |              |
| MT w/ SS          |                               |                        |                       |              |
|                   |                               |                        |                       |              |


[https://github.com/yourname/github-link](https://github.com/dbader/)

## Contributing



