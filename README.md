(This experiment is run as part of my Master's degree and is in active
development)

As the train and test set, the files *train.csv* and *test.csv* from
[this Kaggle competition](https://www.kaggle.com/competitions/fake-news/data) have been used, respectively. They have
the following columns: id, title, author, text, label, with label 0 meaning
reliable and 1 meaning unreliable information.


To get the knowledge graph, run

```
$ wget https://github.com/BunsenFeng/news_stance_detection/blob/main/KG/USPoliticalKG.data

$ mv USPoliticalKG.data data/knowledge_graph.txt  # let's not pretend this isn't a txt file
```
