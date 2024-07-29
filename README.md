# Sign Language Sense Disambiguation
Submission of Group 1 for 'Erweiterungsmodul Computerlinguistik' @ LMU Munich 2024

## Description
This term project on Sign Language Translation is a submission for the 'Erweiterungsmodul Computerlinguistik' course at the Ludwig-Maximilians-Universtät in Munich. The course is taught by Özge Alacam and Beiduo Chen.
The authors perform Sign Language Sense Disambiguation on newly created and augmented datasets.

This project researches the improvement possibilities of disambiguation using a transformer-based model by enhancing the available sign language data with cut-outs of specific bodyparts.
It is based on the prior research by [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830).


## Installation
<!-- The data is stored in the ```data``` folder and ready-to-use for the transformer. -->
The data can be downloaded with the links provided below.
It was created in the course of the experiment development.
The original data by [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830) can be downloaded with the script provided in the ```data``` folder.

All used package versions are provided in the ```requirements.txt``` file.

## Usage
TODO

The transformer can be trained with
    ```python -m signjoey train configs/sign.yaml```

The following configuration files are available in ```configs``` for experiments:
- ```sign.yaml```: Standard configuration by the authors of the original repository.

## Datasets

### GPT Generated Texts
| Dataset                   | Train | Test | Dev |
|---------------------------|-------|------|-----|
| Baseline            | https://drive.google.com/uc?id=1wRwKE7A4eyXWvcI6Qb5xrfJmltTMjSBm | https://drive.google.com/uc?id=1l9J-ojZ1LaM70pOYS7YMqqEA7252AxxN | https://drive.google.com/uc?id=1DPAe3807kDr1kAui5S8ZK8x2yx85lIL4 |
| HandsAndWholeDataGPT      | https://drive.google.com/uc?id=1-H7WGFtiPwb0U7lPa6oaFcnOLjMeqEOH | https://drive.google.com/uc?id=1-PiHfNcaWho6wVY3x6kFEEviurvzoUVd | https://drive.google.com/uc?id=1-tze5w6w-UTLU2QKazmTRJW9LCS95uj9 |
| MouthAndWholeDataGPT      | https://drive.google.com/uc?id=1-PQAdsovmgwK_oVobU4wNccB6p1y0OIe | https://drive.google.com/uc?id=1-_g2I1wnxHyah35-Wi5mK4qds4ck5_R_ | https://drive.google.com/uc?id=1-n6DY8v03I65zomyBIkWC6xPn-CpgnHS |
| HandsMouthAndWholeDataGPT | https://drive.google.com/uc?id=1-RjXkc2r_YRyQUFswFij28YRQF9158jg | https://drive.google.com/uc?id=1-fwanvnOyebxKSDQ9UqpHGccR2cIxcN5 | https://drive.google.com/uc?id=1055gu-eQI8CpwJJWKqL7TmNKvFn8Rxrp |
| HandsOnlyDataGPT          | https://drive.google.com/uc?id=1-N6uxpmU-RhctMPCwXYBFwEVDhKIJYo2 | https://drive.google.com/uc?id=1-ZJ4Sxuh264_cLDb0dIeEuKUo2emJ3Ad | https://drive.google.com/uc?id=1-kNidBToKRBp64YS9flgQlOfyxksdf4L |
| MouthOnlyDataGPT          | https://drive.google.com/uc?id=1-VI2yZ6v5uQ_y_vTNHqr3YImqGB1pG1I | https://drive.google.com/uc?id=1-tV27pi069Jq-X0epnha-gadXy8yZC96 | https://drive.google.com/uc?id=1-vbGtlBMtHzAzrXEo1vFp9xlUGWpg92j |

### GPT Generated Texts with Data Augmentation
| Dataset                   | Train | Test | Dev |
|---------------------------|-------|------|-----|
| Baseline            | https://drive.google.com/uc?id=1-Okmgv4s5F0e572JuPkVFL5sh90cRxwi | https://drive.google.com/uc?id=1-Xs3cHrqSp-l3YnTxCXbpNkJqe4fiHm- | https://drive.google.com/uc?id=1-cTA9XMxaIUuDctm5mhEZpocB1f40J46 |
| HandsAndWholeDataGPT      | https://drive.google.com/uc?id=1-q_0fLtoWh6ZApy3U6_WtMcP_3Eqa4Q8 | https://drive.google.com/uc?id=1-tZu0kSDdS-NY4yZIUU4wwT1QKaQIhox | https://drive.google.com/uc?id=1-u97dtFlyIEzloNVtcVCTajJvOz2tVZi |
| MouthAndWholeDataGPT      | https://drive.google.com/uc?id=107G7bqeqQrUwZrOfTXQOtGY1aHyD5dHS | https://drive.google.com/uc?id=10DPaDt6efFjCQFY03KGBlBxIaNTyVpIW | https://drive.google.com/uc?id=10Cs4ps7ZzwMC_D5KQno7jyi8-psc53bO |
| HandsMouthAndWholeDataGPT | https://drive.google.com/uc?id=10FH1uJkbhcnEbNTV5nOl9LhvRO-JHgrH | https://drive.google.com/uc?id=10HVjUsJcZz-3rIX4gqpnXbMc-hrm4JpF | https://drive.google.com/uc?id=10Ge1t7-xwSsX85mrYFJpanV-nakCWX_Z |
| HandsOnlyDataGPT          | https://drive.google.com/uc?id=189OcEcYrntX_L3w-Cg6MjHDE37dduxKe | https://drive.google.com/uc?id=1LtB0Gyefb4mnair_CIsZ1wmU4LRtZmX2 | https://drive.google.com/uc?id=1hGPjmXXHePidX0ceGba8Y6c0n05P9QyE |
| MouthOnlyDataGPT          | https://drive.google.com/uc?id=1-En8z16WiJNhi81nnT5IYFnxLkUNT73e | https://drive.google.com/uc?id=1-Ev9WznGivXHOfy7QaSdDhLjtAasV69e | https://drive.google.com/uc?id=1-I5_uG9VWUtjL9CoPa3dkxw3s2MvQ5tU |

### Wordset Matching with GPT Substitutions
| Dataset                 | Train | Test | Dev |
|-------------------------|-------|------|-----|
| Baseline                | https://drive.google.com/uc?id=1-h082qaKIP4ydS1AxK80ICz6a2rBm0I_ | https://drive.google.com/uc?id=1-hsvA37hc28O9up5YYKg5gVwECx8WLGG | https://drive.google.com/uc?id=1-ik7_OdpgtiGumMgkYeEBfBgByT4-z-5 |
| HandsAndWholeData       | https://drive.google.com/uc?id=1-8VEREqCO3nPyaqUdk_epu25qUgfPU0r | https://drive.google.com/uc?id=1-8_-UhNLZxcaxijXyRGOsK8DkfxpSdG8 | https://drive.google.com/uc?id=1-7PnIAhSzhc7EJWj4bZNYd46iaNM06Z_ |
| MouthAndWholeData       | https://drive.google.com/uc?id=1cqD9-VTqo5i5IS-1Kc-PBiG9TM-irDB7 | https://drive.google.com/uc?id=1vXW0XP2bbTIsMaIeQNqhxXqCSy0Cw4xN | https://drive.google.com/uc?id=1os9V8IxXUHKhBop8VciTO0NZQKVY5JPB |
| HandsMouthAndWholeData  | https://drive.google.com/uc?id=1-0AvlM6wKtmW1RLvimNWF5yfDzCW-eYc | https://drive.google.com/uc?id=1E8_GH8J-Fv06N6BcjQH-ekId_TYjWP8M | https://drive.google.com/uc?id=10rC0L_RAtuIPcDxuvzo7nde6hm5Nf8gZ |
| HandsOnlyData           | https://drive.google.com/uc?id=1-1dByBwk1UjUPkL4-hQXdIclvqzP2zGv | https://drive.google.com/uc?id=1-26T3zqfojt4QFUHLyYosZd1HRHLuCIF | https://drive.google.com/uc?id=1-1ZPFe098yDqPK8q5OsF9A6arA7imDuA |
| MouthOnlyData           | https://drive.google.com/uc?id=1-3L2z45oFlA8VXAPpGDJeDHAS_fwzJ-x | https://drive.google.com/uc?id=1-5lks2H8Yra3v5AXzM2Ke2fnRMfs0bcm | https://drive.google.com/uc?id=1-2PC8CvozTMaaVQU_xhP7uk90hNdnafa |


## Results
TODO

## Acknowledgment
We want to thank our supervisors Özge Alacam and Beiduo Chen for their support and advice in the developing stages of the project.

## Project status
<!-- This project was finished on August 2nd, 2024. -->
This project is a work-in-progress.
