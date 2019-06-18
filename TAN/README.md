# networks.py 
## `class LSTM_TAN(nn.Module):`
```
def __init__(self,version,embedding_dim, hidden_dim, vocab_size, n_targets,embedding_matrix,dropout = 0.5):
```
__Args__
1. version - str - one of ["lstm,"tan-","tan"]
2. embedding_dim - int - dimension of word_embedding
3. hidden_dim - int - dimension of LSTM hidden state
4. vocab_size - int - number of words in the vocabulary
5. n_targets - int - number of target classes
6. embedding_matrix - numpy array dtype=float - word embedding matrix
6. dropout - The dropout to be applied before on the final hidden state(lstm)/attention-weighted hidden state (tan-,tan)

### Running the code
To run the code you have to run *get_training_plots.py* or *early_stopping_training.py*<br>
To change models, call the classes with specific versions as mentioned above<br>
