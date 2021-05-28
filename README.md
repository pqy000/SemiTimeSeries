# SemiTimeSeries

Describe briefly what makes your project stand out.

## Requirements

- List of required tools for this project.

## Installation

1. Steps to install your project.
1. Include commands if possible.

   ```sh
   echo "Hello World"
   ```

## Usage

- `incognito` - Open an incognito window with [Google](https://www.google.com/).

### experiment results
          
| Label Ratio       | 10%                    | 20%                    | 40%                   | 100%         |
| ----------------- | ---------------------- | ---------------------- | --------------------- | ------------ |
| **Dataset** | **CricketX**     |                        |                       |              |
| SemiTime          | 44.88 (3.13)           | 51.61 (1.22)           | 58.71 (2.78)          | 65.66 (1.58) |
| MeanTeacher       | 39.54 (1.16)           | 51.59 (1.98)           | **62.87** (1.69)          |              |
| **Dataset** | InsectWingbeatSound    |                        |                       |              |
| SemiTime          | 54.96  (1.61)          | 59.01 (1.56)           | 62.38 (0.76)          | 66.57 (0.67) |
| MeanTeacher       | **57.95** (1.64) | **61.47** (1.58) | **64.29**(1.18) |              |
| **Dataset** | MFPT                   |                        |                       |              |
| SemiTime          | 64.16(0.85)            | 69.84(0.94)            | 76.49 (0.54)          | 84.33(0.50)  |
| MeanTeacher       |                        |                        |                       |              |
| **Dataset** | Uwave                  |                        |                       |              |
| SemiTime          | 81.46(0.60)            | 84.57(0.49)            | 86.91(0.47)           | 90.29(0.32)  |
| MeanTeacher       |                        |                        |                       |              |
| **Dataset** | Epilep                 |                        |                       |              |
| SemiTime          | 74.86(0.42)            | 75.54(0.63)            | 77.01(0.79)           | 79.26(1.20)  |
| MeanTeacher       |                        |                        |                       |              |

