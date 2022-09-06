# Models comparison for survival analysis

Author: Anaïs Chossegros

## Project Description

This project compares three models predictions to perform survival analysis. The use case is to predict survival time from patients with cancer. The accuracy is measured with the C-index metric.


## Installation

With a Python 3.7.13 environment, install the necessary packages with:

`pip install -r requirements.txt`

Export the data files from their raw format to `.tsv`



#### Download the data

The data for the project can be found with the TCGA portal. The gene expressions are collected from the rna_seq.augmented_star_gene_counts files. 
The associated clinical file and metadata file can be downloaded. 

## Code Description
For the pre-processing of the data:
-	Data extraction pancreas and Data extraction lung. These files read the data from the TCGA dataset, process them for being suitable to the DESeq2 normalisation, and read the data after the DESeq2 normalisation
-	Data Loader: processes the normalised data for the survival models. The processing includes converting the vital status to Boolean, sorting the data by days to death order, converting all the data to float format and then to tensor.
-	Data visualisation: Visualises the data

For the Cox-nnet model and the transfer learning: 
-	Model: Implements the Cox-nnet model, and its associated functions for the training
-	Survival_CosFunc_CIndex: computes the loss and the C-index
-	Train: loops over the batches, the epoch and the k folder to update the model parameters
-	Cox-nnet: runs the model and plots the results. Performs the Transfer Learning.

For Cox-PH and Random Forest:
-	Cox-PH: Runs the Cox-PH model
-	Random Forest: Runs the Random Survival Forest model

## Support

[Ask the author](mailto:amc21@ic.ac.uk)



## Acknowledgment

Project managed with the help of Dr M. Papathanasiou and Dr K.Polizzi.


## Project status

Ended 31st August 2022.
