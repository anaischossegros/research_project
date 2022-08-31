# Unsupervised Learning of Clinical Scales

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
https://portal.gdc.cancer.gov/exploration?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22cases.project.program.name%22%2C%22value%22%3A%5B%22TCGA%22%5D%7D%7D%2C%7B%22content%22%3A%7B%22field%22%3A%22genes.is_cancer_gene_census%22%2C%22value%22%3A%5B%22true%22%5D%7D%2C%22op%22%3A%22in%22%7D%5D%7D


. ## Support

[Ask the author](mailto:amc21@ic.ac.uk)



## Acknowledgment

Project managed with the help of Dr M. Papathanasiou and Dr K.Polizzi.


## Project status

Ended 31st August 2022.
