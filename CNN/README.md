## Requirements
Python version - 2.7

You need to download Google news vectors 300 dimensional bin file and place 
it in the main directory to run the code
You will need to have Theano=0.7 to run the code. Please refer : https://github.com/nestle1993/SE16-Task6-Stance-Detection

## Usage 
#### Training and Testing    
	 python run.py <Path to the Data Directory>
#### Printing the scores for each dataset within data directory    
	 python score.py <Path to the Data Directory>
#### Printing the tuned hyperparameters for each dataset within data directory    
	 python get_params.py <Path to the Data Directory>

**Note:** The Data Directory be one of the preprocessed folders created by the preprocessing script.
Example directory structure of the data directory for SemEval Data:
			
	Data_SemE_P
	├── AT
	|   ├── test_clean.txt
	|   ├── test_preprocessed.csv
	|   ├── train_clean.txt
	|   └── train_preprocessed.csv
	├── CC
	|   ├── test_clean.txt
	|   ├── test_preprocessed.csv
	|   ├── train_clean.txt
	|   └── train_preprocessed.csv
	├── FM
	|   ├── test_clean.txt
	|   ├── test_preprocessed.csv
	|   ├── train_clean.txt
	|   └── train_preprocessed.csv
	├── HC
	|   ├── test_clean.txt
	|   ├── test_preprocessed.csv
	|   ├── train_clean.txt
	|   └── train_preprocessed.csv
	└── LA
	    ├── test_clean.txt
	    ├── test_preprocessed.csv
	    ├── train_clean.txt
	    └── train_preprocessed.csv
