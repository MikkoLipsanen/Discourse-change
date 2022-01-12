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

Synthetic datasets can be created using the code in the `get_samples.py` file. The functions used for data generation take a number of arguments that are described individually below.

`--data_path`: Path where `df_light.pkl` file was saved during preprocessing. Default path is `data/df_light.pkl`.

`--emb_path`: Path where `embeddings_dict.pkl` file was saved during preprocessing. Default path is `data/embeddings_dict.pkl`.

`--save_path`: Folder where the generated samples are saved. Default path is `data/samples`. This folder needs to have two subfolders, `train` and `test`.

`--tr_samples`: Number of samples datasets that are needed for training and validation sets combined. Default number is 5000.

`--ts_samples`: Number of samples datasets that are needed for test data. Default number is 2000.

`--num_timepoints`: Number of timepoints in a single sample dataset. Default number is 100.

`--min_docs`: Minimum number of documents in a single sample dataset. Default number is 4000.

`--max_docs`: Maximum number of documents in a single sample dataset. Default number is 5000.

`--test_categories`: Article categories used for the test data. Default categories are 'autot', 'musiikki', 'luonto', 'vaalit' and 'taudit'.

`--tr_val_categories`: Article categories used for the train and validation data. Default categories are 'työllisyys','jääkiekko', 'kulttuuri', 'rikokset', 'koulut', 'tulipalot' and 'ruoat'.

`--n_topics_tr`: Number of topics used in the training and validation samples. Default number is 7.

`--n_topics_ts`: Number of topics used in the test samples. Default number is 5.

`--n_unstable`: Number of unstable categories used in the samples. Default number is 1.

`--n_stable`: Number of stable categories used in the samples. Default number is 1.

`--add_noise`: Defines whether random noise is added to the discourse patterns. Default is 'no'.

`--noise_std`: If noise is added, defines the standard deviation of the gaussian random noise. Default is 0.0001.

`--rand_docs`: Defines whether random documents are used for the stable categories. If this option is chosen, it impacts the number of topics in the data. Default is 'no'.

To run the code, add the required arguments to the following code:
```
python3 get_samples.py
```

## Contrastive pretraining



## Unsupervised classification



## Unsupervised pivot point detection



## Supervised classification



## Supervised pivot point detection
