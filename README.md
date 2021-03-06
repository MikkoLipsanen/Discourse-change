# Contrastive pretraining in discourse change detection

The repository contains the code used for running the experiments that are presented in the Data Science Master's thesis *Contrastive pretraining in discourse change detection* (2022, University of Helsinki). The following guidelines provide instructions on how to use the code for replicating different parts of the experiments. Most of the operations require access to a high performance computing environment, unless only a very small subset of the data is used. 


## Data

The data used in the experiments is a corpus of Finnish news articles by the Finnish Broadcasting Company YLE, containing in total over 700 000 articles from the years 2011-2018. The dataset is available for acedemic use at https://korp.csc.fi/download/YLE/fi/2011-2018-src/.

## Data preprocessing and embedding

Data preprocessing and embedding functions are located in the `preprocess_data.py` file. It is assumed here that the downloaded YLE news corpus is located under to the filepath `data/yle/ylenews-fi-2011-2018-src/data/fi`, where the folder `/fi` has subfolder for each year covered by the corpus. The code collects the individual `.json` files containing the article data from the subfolders of `/fi`, extracts the needed data and saves it as a Pandas dataframe using the pickle library. It then performs preprocessing to the article texts as described in the thesis, and then uses Sentence-BERT to create the article embeddings. The embeddings are saved as a dict, `embeddings_dict.pkl`, where the key is the article id, and the value is the embedding tensor for the article in question. A trimmed version of the dataframe, containing only the id and topics covered by each article, is saved as `df_light.pkl`.

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

`--tr_val_categories`: Article categories used for the train and validation data. Default categories are 'ty??llisyys','j????kiekko', 'kulttuuri', 'rikokset', 'koulut', 'tulipalot' and 'ruoat'.

`--n_topics_tr`: Number of topics used in the training and validation samples. Default number is 7.

`--n_topics_ts`: Number of topics used in the test samples. Default number is 5.

`--n_unstable`: Number of unstable categories used in the samples. Default number is 1.

`--n_stable`: Number of stable categories used in the samples. Default number is 1.

`--add_noise`: Defines whether random noise is added to the discourse patterns. Default is 'no'.

`--noise_std`: If noise is added, defines the standard deviation of the gaussian random noise. Default is 0.0001.

`--rand_docs`: Defines whether random documents are used for the stable categories. If this option is chosen, it impacts the number of topics in the data. Default is 'no'.

To run the code, add the required arguments to the following code snippet:
```
python3 get_samples.py
```

## Contrastive pretraining

Contrastive model can be trained using the code in the `contrastive_training.py` file. The code takes a number of arguments, and the ones that have not appeared before are described individually below.

`--data_path`: Path where samples were saved during data generation. Default path is `data/samples`.

`--res_path`: Path where a dictionary containing the training results are saved. Default path is `results/`.

`--model_path`: Path for saving the PyTorch model. Default path is `model/contrastive_model.pt`.

`--continue_tr`: Defines whether training is continued using a saved model. Default is 'no'.

`--batch_size`: Size of the batch defined as number of timepoints in one batch. Default size is 3000.

`--tr_size`: Defines the number of sample datasets in the training dataset. Default number is 3000.

`--val_size`: Defines the number of sample datasets in the validation dataset. Default number is 2000.

`--ts_size`: Defines the number of sample datasets in the test dataset. Default number is 1000.

`--lr`: Defines the learning rate. Default is 0.01.

`--epochs`: Defines the number of training epochs. Default is 50.

`--hidden_size`: Defines the hidden state size of the bidirectional LSTM layers. Default is 10.

`--lstm_layers`: Number of LSTM layers used in the model. Default number is 2.

`--emb_size`: Size of the document embedding vector. Default size is 300.

`--te_output_size`: Size of the vector that is used as input to the LSTM layers. Default size is 300.

`--rnn_output_size`: Size of the vector that is the output of the model. Default size is 300.

`--lstm_dropout`: Defines the dropout probability for the bidirectional LSTM layers. Default probability is 0.2.

`--attn_dropout`: Defines the dropout probability for the transformer encoder layers. Default probability is 0.2.

`--n_heads`: Number of attention heads used in the transformer encoder layers. Default is 2.

`--attn_layers'`: Number of transformer encoder layers used in the model. Default number is 4.

`--tau'`: Temperature parameter for the SupCon loss function. Default value is 0.6.

`--smoothing_type'`: Type of smoothing calculation used by the model. Options are 'mean', 'std', 'double' and 'no', when smothing is not applied. Default value is 'no'.

`--window_size'`: Size of the smoothing window. Default size is 9.


To run the code, add the required arguments to the following code snippet:
```
python3 contrastive_training.py
```

## Saving timepoint and document vectors 

In order to save the timepoint vectors, which are an intermediate product of the contrastive model, for further use, the code in `save_timepoint_vecs.py` file can be used.

`--data_path`: Path where samples were saved during data generation. Default path is `data/samples`.

`--save_path`: Path where the timepoint vectors are saved. Default path is `results/timepoint_vecs/`.

`--model_path`: Path where the PyTorch model was saved. Default path is `model/contrastive_model.pt`.

It is important that the other arguments defining the model parameters are the same that were used for the contrastive model.

To run the code, add the required arguments to the following code snippet:
```
python3 save_timepoint_vecs.py
```

In order to save the document vectors, which are also an intermediate product of the contrastive model, for further use, the code in `save_doc_vecs.py` file can be used.


`--save_path`: Path where the document vectors are saved. Default path is `results/doc_vecs/`.

`--n_docs`: Number of document vectors sampled from each dataset. Default number is 2000.

`--n_samples`: Number of datasets sampled. Default number is 5.

It is important that the other arguments defining the model parameters are the same that were used for the contrastive model.

To run the code, add the required arguments to the following code snippet:
```
python3 save_doc_vecs.py
```

## Unsupervised timepoint classification and pivot point detection

Unsupervised timepoint classification and pivot point prediction can both be performed with the code in the file `unsup_tp_class.py`.

`--data_path`: Path where the timepoint vectors were saved. Default path is `results/timepoint_vecs/timepoint_vecs.pkl`.

`--n_perm`: Number of permutations used for each dataset. Default number is 1000.

`--win_size`: Size of the smoothing window used when computing the averaged timelines. Default size is 10.

`--p_threshold`: P-value threshold. Default threshold is 0.05.

`--pad_mode`: Defines what values are used for padding. Default mode is 'mean'. Other padding options can be found at https://numpy.org/doc/stable/reference/generated/numpy.pad.html.

`--dist`: Distance measure used for calculating timepoint distances. Default measure is 'cosine'. Other option is 'euclidean'.

`--vis_samples`: Diefines the number of random samples that are visualized. Default number is 0.

`--pr_margin`: Defines the margin parameter for ruptures precision_recall-function. Default margin is 5.

To run the code, add the required arguments to the following code snippet:
```
python3 unsup_tp_class.py
```

## Supervised timepoint classification

Supervised timepoint classification can both be performed with the code in the file `supervised_tp_class.py`.

`--save_model_path`: Path where the classification model is saved. Default path is `model/supervised_tp_model.pt`.

`--pos_w`: Weight parameter for the loss function. Default weight is 3.

`--epochs`: Defines the number of training epochs. Default is 20.

It is important that the other arguments defining the model parameters are the same that were used for the contrastive model.

To run the code, add the required arguments to the following code snippet:
```
python3 supervised_tp_class.py
```

## Supervised pattern based classification

Supervised pattern based classification can be performed with the code in the file `supervised_pattern_class.py`.

`--save_model_path`: Path where the classification model is saved. Default path is `model/supervised_pattern_model.pt`.

`--n_cats`: Number of different discourse patterns in the data. Default number is 7.

It is important that the other arguments defining the model parameters are the same that were used for the contrastive model.

To run the code, add the required arguments to the following code snippet:
```
python3 supervised_pattern_class.py
```

## Supervised pivot point detection

Supervised pivot point detection can be performed with the code in the file `supervised_pivot_point_pred.py`.

`--model_path`: Path where the timepoint classification model was saved. Default path is `model/supervised_tp_model.pt`.

`--ts_size`: Number of datasets used for the task. Default number is 1000.

`--pr_margin`: Defines the margin parameter for ruptures precision_recall-function. Default margin is 5.

It is important that the other arguments defining the model parameters are the same that were used for the timepoint classification model.

To run the code, add the required arguments to the following code snippet:
```
python3 supervised_pivot_point_pred.py
```

## Evaluation methods

This group of methods are used for evaluating the performance of contrastive pretraining.

### t-SNE

t-SNE visualization for timepoint vectors can be performed with the code in the file `t_SNE.py`.

`--data_path'`: Path where the timepoint timepoint vectors were saved. Default path is `results/timepoint_vecs/timepoint_vecs.pkl`. 

`--perplexity`: Perplexity parameter for the t-SNE operation. Default perplexity is 30.

`--n_iter`: Number of iterations used in the t-SNE operation. Default number is 300.

`--n_vecs`: Number of timepoint vectors used in the t-SNE operation. Default number is 50000.

This code can be easily modified to perform the t-SNE operation also to model output vectors and document vectors.

To run the code, add the required arguments to the following code snippet:
```
python3 t_SNE.py
```

### K-means clustering

K-means clustering of the timepoint vectors can be performed with the code in the file `k_means.py`.

`--data_path'`: Path where the timepoint timepoint vectors were saved. Default path is `results/timepoint_vecs/timepoint_vecs.pkl`. 

`--n_samples`: Number of sample sets used in the t-SNE operation. Default number is 500.

This code can be easily modified to perform K-means clustering also to model output vectors and document vectors.

To run the code, add the required arguments to the following code snippet:
```
python3 k_means.py
```

### Permutation testing

Permutation testing can be performed for the timepoint vectors with the code in the file `permutations.py`.

`--data_path'`: Path where the timepoint timepoint vectors were saved. Default path is `results/timepoint_vecs/timepoint_vecs.pkl`. 

`--n_perm`: Number of permutations per dataset used in the tests. Default number is 1000.

`--n_sets`: Number of sample dataset used in the tests. Default number is 400.

This code can be easily modified to perform permutation testing also to model output vectors and document vectors.

To run the code, add the required arguments to the following code snippet:
```
permutations.py
```
