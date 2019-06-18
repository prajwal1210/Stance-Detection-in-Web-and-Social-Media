import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import sys




class LSTM_TAN(nn.Module):
    def __init__(self,version,embedding_dim, hidden_dim, vocab_size, n_targets,embedding_matrix,dropout = 0.5):
        super(LSTM_TAN, self).__init__()
        if version not in ["tan-","tan","lstm"]:
            print("Version is tan-,tan,lstm")
            sys.exit(-1)

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeddings.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float))
        self.word_embeddings.weight.requires_grad=True
        self.version = version

        if version == "tan-":
            self.attention = nn.Linear(embedding_dim,1)
        elif version == "tan":
            self.attention = nn.Linear(2*embedding_dim,1)


        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=(version!="lstm"))

        self.dropout = nn.Dropout(dropout)

        if version !="lstm":
            self.hidden2target = nn.Linear(2*self.hidden_dim, n_targets)
        else:
            self.hidden2target = nn.Linear(self.hidden_dim, n_targets)

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))




    def forward(self, sentence, target,verbose=False):
        x_emb = self.word_embeddings(sentence)
        version = self.version

        if version != "tan-":
            t_emb = self.word_embeddings(target)
            t_emb = torch.mean(t_emb,dim=0,keepdim=True)
            xt_emb = torch.cat((x_emb,t_emb.expand(len(sentence),-1)),dim=1)

        if version == "tan-":
            lstm_out, _ = self.lstm(
                x_emb.view(len(sentence), 1 , self.embedding_dim))

            a = self.attention(x_emb)

            final_hidden_state = torch.mm(F.softmax(a.view(1,-1),dim=1),lstm_out.view(len(sentence),-1))

        elif version == "tan":
            a = self.attention(xt_emb)

            lstm_out, _ = self.lstm(x_emb.view(len(sentence), 1 , self.embedding_dim))

            final_hidden_state = torch.mm(F.softmax(a.view(1,-1),dim=1),lstm_out.view(len(sentence),-1))

        elif version == "lstm":
            _, hidden_state = self.lstm(
                    x_emb.view(len(sentence), 1 , self.embedding_dim))

            final_hidden_state = hidden_state[0].view(-1,self.hidden_dim)


        target_space = self.hidden2target(self.dropout(final_hidden_state))
        target_scores = F.log_softmax(target_space, dim=1)


        return target_scores

