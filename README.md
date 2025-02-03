# VITRO

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
@misc{bellos2024vitrovocabularyinversiontimeseries,
      title={VITRO: Vocabulary Inversion for Time-series Representation Optimization}, 
      author={Filippos Bellos and Nam H. Nguyen and Jason J. Corso},
      year={2024},
      eprint={2412.17921},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.17921}, 
}
```

## Acknowledgement
Our implementation adapts [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and [Time-LLM](https://github.com/KimMeen/Time-LLM/) as the code base and we have extensively modified it for our purposes. We thank the authors for sharing their code.
