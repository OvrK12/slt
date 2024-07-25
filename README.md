# Sign Language Sense Disambiguation
Submission of Group 1 for 'Erweiterungsmodul Computerlinguistik' @ LMU Munich 2024

## Description
This term project on Information Retrieval (IR) with Vector Space Models (VSM) is a submission for the 'Erweiterungsmodul Computerlinguistik' course at the Ludwig-Maximilians-Universtät in Munich. The course is taught by Özge Alacam and Beiduo Chen.
The authors perform Sign Language Sense Disambiguation on newly created and augmented datasets.

This project researches the improvement possibilities of disambiguation using a transformer-based model by enhancing the available sign language data with cut-outs of specific bodyparts.
It is based on the prior research by [Camgöz et al. (2020)](https://arxiv.org/abs/2003.13830).


## Installation
The data is stored in the ```data``` folder and ready-to-use for the transformer.
It was created in the course of the experiment development.

All used package versions are provided in the ```requirements.txt``` file.

## Usage
TODO

The transformer can be trained with
    ````python -m signjoey train configs/sign.yaml```

The following configuration files are available in ```configs``` for experiments:
- ```sign.yaml```: Standard configuration by the authors of the original repository.

## Results
TODO

## Acknowledgment
We want to thank our supervisors Özge Alacam and Beiduo Chen for their support and advice in the developing stages of the project.

## Project status
<!-- This project was finished on August 2nd, 2024. -->
This project is a work-in-progress.