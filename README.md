# Sign Language Sense Disambiguation
Submission of Group 1 for 'Erweiterungsmodul Computerlinguistik' @ LMU Munich 2024

## Description
This term project on Sign Language Translation is a submission for the 'Erweiterungsmodul Computerlinguistik' course at the Ludwig-Maximilians-Universtät in Munich. The course is taught by Özge Alacam and Beiduo Chen.
The authors perform Sign Language Sense Disambiguation on newly created and augmented datasets.

This project researches the improvement possibilities of disambiguation using a transformer-based model by enhancing the available sign language data with cut-outs of specific bodyparts.
It is based on the prior research by [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830).


## Installation
All dependencies required for the project can be installed using ``pip install -r .\requirements.txt``. To download the raw data for our project cd to the ``data`` folder and run ``python ./download_raw_data.py``. This will download and unpack all required files under ``data/raw_data``. You can then run the preprocessing using ``python ./run_preprocessing.py``. You can configure the preprocessing steps using the ``preprocess_config.yaml`` file inside the data folder:
- gpt_subs: set this option to True for GPT substitutions for placeholders in the natural language translations (this option requires access to the OpenAI API. Therefore, you will need to set an OpenAI API key in your env variables)
- gpt_full: set this option to True for GPT generated replacements of the natural language translations (this option requires access to the OpenAI API. Therefore, you will need to set an OpenAI API key in your env variables)
- augmented: set this option to True for creating extra training data using data augmentation (image flipping, greyscaling)
- bleu: set this option to True to align the preprocessed gloss with the original gloss using bleu score (instead of worset overlaps)

If you do not want to run the preprocessing from scratch, you can download our best performing datasets (bleu alignment, with gpt substitutions and augmented data) using ``python ./download_preprocessed_datasets.py``

Furthermore, we tried out finetuning the original transformer (as reported in [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830)) with our custom created datasets. For this, we pretrained the original transformer for 70 epochs. You can find checkpoints under: https://drive.google.com/file/d/11YX0lTdkRF09xdT9UzuZ42zTvMyldR1I/view?usp=sharing

## Usage
TODO

The transformer can be trained with
    ```python -m signjoey train configs/sign.yaml```

The following configuration files are available in ```configs``` for experiments:
- ```sign.yaml```: Standard configuration by the authors of the original repository.

## Datasets


### Bleu Matching + GPT Subs + Augmentation
| Dataset                   | Train | Test | Dev |
|---------------------------|-------|------|-----|
| Baseline            | https://drive.google.com/file/d/10OVYAfXhXa-aFCnTbHczkWxNohmuCQM9 | https://drive.google.com/file/d/1aR4ybwTi6DzMsHhh4rQZVnTeaURfOcdN | https://drive.google.com/file/d/10OVYAfXhXa-aFCnTbHczkWxNohmuCQM9 |
| HandsAndWholeData     | https://drive.google.com/file/d/13SjJ4QyKABtupA0rmx8vuxQwyJPclSbf |  https://drive.google.com/file/d/1INpTE0vUKj2DxIJEZFUh4UCuiDp2exYC | https://drive.google.com/file/d/1iP7ba0KQQ31--PNu3ZVZSjjPke1wZbao |
| MouthAndWholeData    | https://drive.google.com/file/d/11cl9kjcW3IPw84gLk7-SwsBQYMEcwiw8 | https://drive.google.com/file/d/1zDkeL1C5t9uQWr0qYF8_QsNLPCTZUwlF | https://drive.google.com/file/d/19T_qqVD2ft1sZzsKhfuAq3dHBhNS6AKo |
| HandsMouthAndWholeData | https://drive.google.com/file/d/1qLSluNUMywQNRNqRueImH59spbYByw2n | https://drive.google.com/file/d/1XzqZ7UX7UNyFNgbD6kcooLbVv0OCDJhQ | https://drive.google.com/file/d/11nann53Ntd3ly9LzlyomGCUGJxQ16KEI |
| RightHandAndWholeData       | https://drive.google.com/file/d/1Jx5W6f3xD0c5vHpTKwD-96bRSNbtuEgZ | https://drive.google.com/file/d/1gnxnpi6a5ntAviabFE32vCUIqY3VXAcL | https://drive.google.com/file/d/1GamKKx2s4aJs1Xa-O9SF1MBykHMRuww_ |
| LeftHandAndWholeData        | https://drive.google.com/file/d/1nUe49pdCuV7MJZBKn4Ovqo0hkLAsognB | https://drive.google.com/file/d/1PVfnWVx1mmZNqphOjM3wvx3NEHA_1W5m | https://drive.google.com/file/d/10LYy9Gx3Ictz59gPRHAK_NOcnWalNJRO |


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

### EfficientNet B7 - Wordset Matching with GPT Substitutions
| Dataset                 | Train | Test | Dev |
|-------------------------|-------|------|-----|
| Baseline                | [https://drive.google.com/uc?id=1gUSL44-8C8ZNzBN83giwNuzbJRFAAPym] | https://drive.google.com/uc?id=1vcIqV0JFZlsjzur4bUb_JXFT0fhhn9_u | https://drive.google.com/uc?id=1bptuKXd-vHpTyhfTGhV2CNJwv5oG8Y9c |
| HandsAndWholeData       | https://drive.google.com/uc?id=1-8VEREqCO3nPyaqUdk_epu25qUgfPU0r | https://drive.google.com/uc?id=1-8_-UhNLZxcaxijXyRGOsK8DkfxpSdG8 | https://drive.google.com/uc?id=1-7PnIAhSzhc7EJWj4bZNYd46iaNM06Z_ |
| MouthAndWholeData       | https://drive.google.com/uc?id=1--XPOEAv8q_tefXt-hs4lsaXUtjlWh9C | https://drive.google.com/uc?id=1--e2fMnV8KRiyRmVnN0ul3V8chvT0UbX | https://drive.google.com/uc?id=1-0dcE9pDgRQozJV0VcCvf_71SOHwR_9G |
| HandsMouthAndWholeData  | https://drive.google.com/uc?id=1ZYe07sJkPgA-nn-efEDNfsrVZMjNWCqB | https://drive.google.com/uc?id=1--HhISm4dkxZJufavzSqto3hotbagM9l | https://drive.google.com/uc?id=1IHlN1ov2IAu9AaT9a4mcYRJXd_N-uGuM |
| HandsMouthAndWholeData Augmented    | https://drive.google.com/uc?id=1-JoMed-i3JihHRuOVaQ_aU1OO5FLDRfl | https://drive.google.com/uc?id=1-Kk51IemietdS4haf5nS90iA8MxMoini | https://drive.google.com/uc?id=1-M_Eq-glEhNnatNl_piJhKPJzVGvCfb- |
| MouthAndWholeData Augmented   | https://drive.google.com/uc?id=1-AnQ2uDidyLLB8GX9YypM4o3cnezZO6x | https://drive.google.com/uc?id=1-CkTBtPRZmOG3nV6DSFfB6pdWJmJssSk | https://drive.google.com/uc?id=1-IQWAhMfFIMilyGNlQD__MeEkGaaPzVS |

## Results
TODO

## Acknowledgment
We want to thank our supervisors Özge Alacam and Beiduo Chen for their support and advice in the developing stages of the project.

## Project status
<!-- This project was finished on August 2nd, 2024. -->
This project is a work-in-progress.
