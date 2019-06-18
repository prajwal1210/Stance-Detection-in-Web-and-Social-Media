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
5. n_targets - int - number of dataset classes
6. embedding_matrix - numpy array dtype=float - word embedding matrix
6. dropout - The dropout to be applied before on the final hidden state(lstm)/attention-weighted hidden state (tan-,tan)

__Returns__
1. A torch.nn.module object for the specified version

```
def forward(self, sentence, target,verbose=False)

```
__Args__
1. sentence - a numpy array of shape [1xN] and dtype int, where N is the length of the input sentence and each entry is the corresponding index of the word in the `embedding_matrix`
2. target -  a numpy array of shape [1xM] and dtype int, where M is the length of the target and each entry is the corresponding index of the word in the `embedding_matrix

__Returns__
1. target_scores - a torch float Tensor of shape [1xn_targets], where N is the number of dataset classes. This is the log likelihood probabilities of all the classes



# Running the code
```
python early_stopping_training.py <dataset> <version>
```
1. dataset is one of `['VC', 'HC', 'HRT', 'LA', 'CC', 'SC', 'EC', 'MMR', 'AT', 'FM']`
```
if dataset == 'EC':
        topic = 'E-ciggarettes are safer than normal ciggarettes'
        folder = "Data_MPCHI_P"
    elif dataset == 'SC':
        topic = 'Sun exposure can lead to skin cancer'
        folder = "Data_MPCHI_P"
    elif dataset == 'VC':
        topic = 'Vitamin C prevents common cold'
        folder = "Data_MPCHI_P"
    elif dataset == 'HRT':
        topic = 'Women should take HRT post menopause'
        folder = "Data_MPCHI_P"
    elif dataset == 'MMR':
        topic = 'MMR vaccine can cause autism'
        folder = "Data_MPCHI_P"
    elif dataset == 'AT' :
        topic = "atheism"
    elif dataset == 'HC' :
        topic = "hillary clinton"
    elif dataset == 'LA' :
        topic = "legalization of abortion"
    elif dataset == 'CC' :
        topic = "climate change is a real concern"
    elif dataset == 'FM' :
        topic = "feminist movement"

```
2. version is one of ["lstm","tan-","tan+"]

__Returns__
Trains the model according to the voting scheme discussed in the paper. Prints the final F-Score on the test set.
