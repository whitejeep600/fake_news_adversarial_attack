(This experiment is run as part of my Master's degree and is in active development)

## General info

An experiment about adversarial attacks on fake news detection models and the influence
of a knowledge graph on the base accuracy of such models and on their robustness to
the attacks.

## Environment

To set up a Python environment called `adversarial` with all the required packages and 
GPU support, run `setup.sh`.

## Data

# Knowledge graph
To get the knowledge graph, run

```
$ wget https://github.com/BunsenFeng/news_stance_detection/blob/main/KG/USPoliticalKG.data

$ mv USPoliticalKG.data data/knowledge_graph.txt  # let's not pretend this isn't a txt file
```

# Baseline model

For the baseline model, the files *train.csv* and *test.csv* from 
[this Kaggle competition](https://www.kaggle.com/competitions/fake-news/data) have been used as the
train and test set, respectively. They have  the following columns: id, title, author, text, label, 
with label 0 meaning  reliable and 1 meaning unreliable information. The code can be run without 
additional adjustments if they are saved under the main directory as 
`data/kaggle/train.csv` and `data/kaggle/test.csv`.  Then, run

`$ python -m models.baseline.train_eval_split`

to complete the data setup for the baseline model.


## Attacked models

## Attack algorithms

### TextFooler

This attack is based on [this](https://github.com/jind11/TextFooler?fbclid=IwAR1PyCLr8kNDfQi8MKGhujfxG2iYCQKbs6NleA8vfkx5ATosiAI0VABHw28)
repository and related paper. My own implementation can be found under 
`attacks/text_fooler.py`
