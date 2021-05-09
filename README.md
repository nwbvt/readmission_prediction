# readmission_prediction


## Word2Vec Embedding

The Word2Vec poriton of this project consists of 2 Python source files and an embedding subdirectory containing
a csv file of cleaned text data to be used for training embeddings as well as 2 pre-trained embeddings. 

readmission_prediction/
   - embedding_main.py
   - embedding_utils.py
   - embedding/
      - cleaned_text.csv
      - word2vec.model
      - word2vec.wordvectors
      - word2vec_250d_1win.model
      - word2vec_250d_1win.wordvectors

### Import Embedding Files

```python
import embedding_utils
import embedding_main.py
```

### Load the Baseline Embedding

```python
embedding = load_embedding('embedding/word2vec.wordvectors')
```
### Load the Top Performing Embedding

```python
embedding = load_embedding('embedding/word2vec_250d_1win.wordvectors')
```
### Train Attention Model with a Specific Pre-trained Embedding

The attention model can be trained for a specified number of epochs using one of the above embeddings.

```python
 num_epochs = 5
 trainModel(num_epochs, embedding)
```

### Creating and Training Multiple Embeddings

The embedding_main.py file contains functions to create multiple embeddings with different word vector dimensions
and window sizes. The createEmbeddings() and trainModelWithEmbeddings() functions take in list parameters for the
word vector dimensions and wondow sizes. These are lists of strings and will result in one embedding being created
for each dimension / window size combination. The model and word vector files with be saved in the embedding directory
for later usage. (dimensions=DIMENSIONS, windows=WINDOWS)

Example:

```python
dimensions = ['100', '200']
windows = ['5', '10']

createEmbeddings(dimensions, windows)
```

This will create 4 embeddings: 
 1. 100-dimension word vector with a window size of 5
 2. 100-dimension word vector with a window size of 10
 3. 200-dimension word vector with a window size of 5
 4. 200-dimension word vector with a window size of 10

The names for the ebedding and word vector files will reflect the parameter combinations to make identification easy.
The embeddings and models can be loaded using funtions in the embedding_utils.py file. 

The embedding exploration can be reproduced by calling embeddin_main.py from the command line. All of the print 
statements in model.py were comment out except for the results print statement. Stdout was redirected to a file
to capture the results of training the 12 embeddings for 5 epochs each. Output from within embedding_main.py are
directed to stderr so updates can be observed in the terminal during training while the results are written to a
file.

```python
$ python3 embedding_main.py 1> results.txt
```
