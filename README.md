This is the code repository for the paper "Stance Detection in Web and Social Media: A Comparative Study" published in the proceedings of CLEF 2019 conference (http://clef2019.clef-initiative.eu/).

## Overview of Code
- **Preprocessing**:
	This folder contains the preprocessing code as mentioned in our paper and the datasets. The folder contains all instructions to run the code.
- **CNN**: 
	This is a simple Kim's CNN based model applied for stance detection. The respective folder has the instructions on how to run the codes
- **KDEY**: 
		 This folder contains our attempt at implementing the approach given in the paper : Twitter Stance Detection — A Subjectivity and Sentiment Polarity Inspired Two-Phase Approach [link](https://ieeexplore.ieee.org/document/8215685 ).The folder contains all instructions to run the code
- **TAN**:
	This folder contains the codes for TAN and LSTM, the details of which are mentioned in our paper. The folder contains all instructions to run the code. 
- **SEN-SVM**:
	This folder contains the codes for the SEN-SVM method, the details of which are mentioned in our paper. The folder contains all instructions to run the code. 
- **Bert.ipynb**:
	This ipython notebook contains the code used for running BERT on the dataset. This needs to be be run on [Google Colab](https://colab.research.google.com) with TPU support. 

## Publication

If you use these codes, please cite our paper:

    @inproceedings{
	title={{Stance Detection in Web and Social Media: A Comparative Study}},
	author={Ghosh, Shalmoli and Singhania, Prajwal and Singh, Siddharth and Rudra, Koustav and Ghosh, Saptarshi},
	booktitle={{Proc. Conference and Labs of the Evaluation Forum (CLEF)}},
	year={2019}}
