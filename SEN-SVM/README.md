## Requirements
Python version - 3.6/3.7 with NLTK installed

## Usage
  To run the model for SemEval dataset : python SVM.py

  To run the model for MPCHI dataset : python SVM_mpchi.py

The folder final_feature_set contains all extracted features. Note that medical feature has not been extracted for SemEval dataset.

The features can be extracted by running the 3 codes STA_features.py, te_f.py and Sentiment_API_2.py on individual datasets. 
While extracting the sentiment feature the StanfordCoreNLP server is needed to be run.
