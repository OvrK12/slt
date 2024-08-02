# Sign Language Sense Disambiguation
Submission of Group 1 for 'Erweiterungsmodul Computerlinguistik' @ LMU Munich 2024

## Description
This term project on Sign Language Translation is a submission for the 'Erweiterungsmodul Computerlinguistik' course at the Ludwig-Maximilians-Universtät in Munich. The course is taught by Özge Alacam and Beiduo Chen.
The authors perform Sign Language Sense Disambiguation on newly created and augmented datasets.

This project researches the improvement possibilities of disambiguation using a transformer-based model by enhancing the available sign language data with cut-outs of specific bodyparts.
It is based on the prior research by [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830).


## Installation
All dependencies required for the project can be installed using ``pip install -r .\requirements.txt``. To download the raw data for our project cd to the ``data`` folder and run ``python ./download_raw_data.py``. This will download the RWTH-PHOENIX-Weather Database and unpack all files under ``data/raw_data``. Furthermore, the extracted bodyparts are part of the downloaded files. If you want to rerun the bodypart extraction manually on the dataset, you can do so by running ``python ./bodypart_extraction.py``.

To create the datasets for our experiments, you can then run the preprocessing using ``python ./run_preprocessing.py``. This will create datasets which combine the original images of the RWTH-PHOENIX-Weather Database with the extracted bodyparts. You can configure the preprocessing steps using the ``preprocess_config.yaml`` file inside the data folder:
- gpt_subs: set this option to True for GPT substitutions for placeholders in the natural language translations (this option requires access to the OpenAI API. Therefore, you will need to set an OpenAI API key in your env variables)
- gpt_full: set this option to True for GPT generated replacements of the natural language translations (this option requires access to the OpenAI API. Therefore, you will need to set an OpenAI API key in your env variables)
- augmented: set this option to True for creating extra training data using data augmentation (image flipping, greyscaling)
- bleu: set this option to True to align the preprocessed gloss with the original gloss using bleu score (instead of worset overlaps)

If you do not want to run the preprocessing from scratch, you can download our best performing datasets (bleu alignment, with gpt substitutions and augmented data) using ``python ./download_preprocessed_datasets.py``

Furthermore, we tried out finetuning the original transformer (as reported in [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830)) with our custom created datasets. For this, we pretrained the original transformer for 70 epochs. You can find checkpoints under: https://drive.google.com/file/d/11YX0lTdkRF09xdT9UzuZ42zTvMyldR1I/view?usp=sharing

## Usage
The transformer can be trained with ```python -m signjoey train CONFIG```

As configuration files, we provide two options:
- ```configs/generate_config.py```: Script to create a config file for the desired dataset by editing the parameters within the script. 
- ```configs/baseline.yaml.example```: Standard configuration by [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830).

## Dataset
The links to all of the datasets can be found in the file ```data/links.md```. We list only the best performing dataset here:

### Bleu Matching + GPT Subs + Augmentation
| Dataset                   | Train | Test | Dev |
|---------------------------|-------|------|-----|
| Baseline            | https://drive.google.com/file/d/1u6EHk9fVclyAqBtC2Qyxdu4g8_EpJeJe | https://drive.google.com/file/d/1aR4ybwTi6DzMsHhh4rQZVnTeaURfOcdN | https://drive.google.com/file/d/10OVYAfXhXa-aFCnTbHczkWxNohmuCQM9 |
| HandsAndWholeData     | https://drive.google.com/file/d/13SjJ4QyKABtupA0rmx8vuxQwyJPclSbf |  https://drive.google.com/file/d/1INpTE0vUKj2DxIJEZFUh4UCuiDp2exYC | https://drive.google.com/file/d/1iP7ba0KQQ31--PNu3ZVZSjjPke1wZbao |
| MouthAndWholeData    | https://drive.google.com/file/d/11cl9kjcW3IPw84gLk7-SwsBQYMEcwiw8 | https://drive.google.com/file/d/1zDkeL1C5t9uQWr0qYF8_QsNLPCTZUwlF | https://drive.google.com/file/d/19T_qqVD2ft1sZzsKhfuAq3dHBhNS6AKo |
| HandsMouthAndWholeData | https://drive.google.com/file/d/1qLSluNUMywQNRNqRueImH59spbYByw2n | https://drive.google.com/file/d/1XzqZ7UX7UNyFNgbD6kcooLbVv0OCDJhQ | https://drive.google.com/file/d/11nann53Ntd3ly9LzlyomGCUGJxQ16KEI |
| RightHandAndWholeData       | https://drive.google.com/file/d/1Jx5W6f3xD0c5vHpTKwD-96bRSNbtuEgZ | https://drive.google.com/file/d/1gnxnpi6a5ntAviabFE32vCUIqY3VXAcL | https://drive.google.com/file/d/1GamKKx2s4aJs1Xa-O9SF1MBykHMRuww_ |
| LeftHandAndWholeData        | https://drive.google.com/file/d/1nUe49pdCuV7MJZBKn4Ovqo0hkLAsognB | https://drive.google.com/file/d/1PVfnWVx1mmZNqphOjM3wvx3NEHA_1W5m | https://drive.google.com/file/d/10LYy9Gx3Ictz59gPRHAK_NOcnWalNJRO |

## Results
| Body Part                   | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE | CHRF |
|---------------------------|---------------|---------|--------|--------|--------|--------|
| Baseline            | 68.60 | 66.37 | 65.48 | 64.93 | 70.60 | 71.98 |
| HandsAndWholeData     | 76.38 | 74.85 | 74.20 | 73.81 | 77.25 | 80.10 |
| MouthAndWholeData    | 68.65 | 66.79 | 66.12 | 65.80 | 68.58 | 72.24 |
| HandsMouthAndWholeData | 73.27 | 71.14 | 70.25 | 69.76 | 74.06 | 75.73 |

## Acknowledgment
We want to thank our supervisors Özge Alacam and Beiduo Chen for their support and advice in the developing stages of the project.

## Project status
This project was finished on August 2nd, 2024.
<!-- This project is a work-in-progress. -->
