<div align="center">
      
# VITRO: Vocabulary Inversion for Time-series Representation Optimization

<a href='https://arxiv.org/abs/2412.17921'><img src='https://img.shields.io/badge/ArXiv-2311.12886-red'></a> 
<a href='https://fil-mp.github.io/project_page/'><img src='https://img.shields.io/badge/Project-Page-Blue'></a>
</div>

## 🛠 Prerequisites

Install the necessary dependencies by first creating a conda environment:

```
conda create -n "myenv" python=3.11.0
conda activate myenv
```
To install all dependencies:
```
pip install -r requirements.txt
```

## 📊 Prepare Datasets

Begin by downloading the required datasets. All datasets are available at [Autoformer](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy). Create a separate folder named `./data` under `src`. Then, place the downloaded contents under the new folder.

## 💻 Training

For training we provide scripts under `./scripts`. Example usage:
```shell
cd src
```
```shell
sh scripts/stage1.sh
```
```shell
sh scripts/stage2.sh
```


## 📚 Citation
If you find this repo useful, please consider citing our paper as follows:
```
@inproceedings{bellos-2025-vitro,
      title={VITRO: Vocabulary Inversion for Time-series Representation Optimization}, 
      author={Filippos Bellos and Nam H. Nguyen and Jason J. Corso},
      year={2025},
      booktitle={2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
}
```

## Acknowledgement
Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [Time-LLM](https://github.com/KimMeen/Time-LLM/) as the code base and we have extensively modified it for our purposes. We thank the authors for sharing their code.
