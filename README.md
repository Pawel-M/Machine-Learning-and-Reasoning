# Machine Learning and Reasoning

Code for the MRP project 'Machine Learning and Reasoning' at Maastricht University.

## Dataset
To create the datasets of a one variable or mental models (single or multiple) as a conclusion from a logic sentence 
first run `dataset/conclusion_generation.py` with parameters set in the `__main__` seciton of the script.
The script will create a `.csv` file in `data` folder.

Next, the `.csv` file must be encoded by running the `dataset/encoding.py` script with the same parameters as used in 
`dataset/conclusion_generation.py`. This script will encode the logic sentences and conclusions into numpy arrays
and save them as a `.pkl` file in `data` folder.

The dataset can be loaded and used with the functions in `dataset.common.py` python file. The dataset is loaded as 
an object `DeductionDataset` object (defined in `dataset.common.py`).

## Models
The tested existing architectures can be found in `models/rnn_example.py` for recurrent models 
and in `models/transformer.py` for Transformer. In `models/encoder_decoder.py` encoder-decoder architectures based on
LSTM cam be found.

In `models/single_mm_net.py` the implementation of Single-mmNet model can be found with the implementations
of different versions of Mental Model Inference Layer.

Multi-mmNet models are implemented in two files: `models/multi_mm_net_direct.py` for direct output version, and in 
`modeld/multi_mm_net_combination.py` for combination version.

## Training and Analysis
For training and parts of the analysis process we used Jupyter Notebooks located in `notebooks` folder.
Additionally, `analysis` and `visualization` folders contain scripts used for analysis and visualization.

The script `models/training.py` contain utility functions for training the models with specifying the varying 
parameters as lists. 

## Results
In the 'Experiment results' folder the Excel file can be found with the results of all the experiments performed 
during the project.