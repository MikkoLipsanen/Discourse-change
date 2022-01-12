# Contrastive pretraining for discourse change detection

The repository contains the code used for running the experiments that are presented in the Data Science Master's thesis *Contrastive pretraining in discourse change detection* (2022, University of Helsinki). The following guidelines provide instructions on how to use the code for replicating different parts of the experiments. Most of them require access to a high performance computing environment, unless only a very small subset of the data is used. 


## Data

The data used in the experiments is a corpus of Finnish news articles by the Finnish Broadcasting Company YLE, containing in total over 700 000 articles from the years 2011-2018. The dataset is available for acedemic use at https://korp.csc.fi/download/YLE/fi/2011-2018-src/.

## Data preprocessing and embedding

Data preprocessing and embedding functions are located in the `preprocess_data.py` file. We assume here that the downloaded YLE news corpus is located under to the filepath `data/yle/ylenews-fi-2011-2018-src/data/fi`, where the folder `/fi` has subfolder for each year covered by the corpus. The code collects the individual `.json` files containing the article data from the subfolders of `/fi`, extracts the needed data and saves it as a Pandas dataframe using the pickle library. It then performs preprocessing to the article texts as described in the thesis, and then uses Sentence-BERT to create the article embeddings. The embeddings are saved as a dict, `embeddings_dict.pkl`, where the key is the article id, and the value is the embedding tensor for the article in question. A trimmed version of the dataframe, containing only the id and topics covered by each article, is saved as `df_light.pkl`.

To run the code, change the paths in the following snippet according to your needs:
```
python3 preprocess_data.py --data_path 'data/yle/ylenews-fi-2011-2018-src/data/fi' --save_path 'data/'
```

## Synthetic dataset generation



## Contrastive pretraining



## Unsupervised classification



## Unsupervised pivot point detection



## Supervised classification



## Supervised pivot point detection
