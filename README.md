As the train and test set, the files *train.csv* and *test.csv* from
[this Kaggle competition](https://www.kaggle.com/competitions/fake-news/data) have been used,
respectively.

To get the knowledge graph, run

```
$ wget https://github.com/BunsenFeng/news_stance_detection/blob/main/KG/USPoliticalKG.data

$ mv USPoliticalKG.data data/knowledge_graph.txt  # let's not pretend this isn't a txt file
```
