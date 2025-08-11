# Explaining Drug Toxicity


When it comes to determining the safety profile/ toxicity of a drug molecule, most standard computational models focus on predictive accuracy. However, when the patient's health is on the line, we cannot rely on a black box model. Our main aim through this project is to shift the focus on interpretability of the model using explainable AI and find out what kind of chemical substructures help determine the toxicity of an input molecule.

## Authors

[Maximilian Ruhdorfer](https://github.com/maxruhdorfer/xai-drug-discovery)
[Shaswata Roy](https://github.com/ShaswataRoy)
[Sunil Philip](https://github.com/sunilphil)

## Dataset

In this project we explore the TOX21 dataset (https://moleculenet.org/datasets-1) and focus on NR-AhR (Nuclear Receptor - Aryl hydrocarbon Receptor) activity. The TOX21 dataset contains 7831 samples of chemical compounds with molecular SMILES with 12 binary labels (activ/inactive) which represent the outcome of toxicological experiments. 

We perform simple data-preprocessing which includes
- Ensuringing that the ID corresponds to an actual molecule and the entry is not `NaN`
- Sanitizing the molecule: check the chemical representation of the molecule
- Removing possible duplicates
- Extracting meaningful properties both at the molecular and at the atomic level
- Transforming the molecular data into `pytorch_geometric` graph data using the `rdkit` package

See `data_exploration.ipynb`.

## Models

We study three different model architectures:
1. Graph Isomorphism Network (GIN)
2. Graph Attention Network (GAT)
3. Message Passing Neural Network (MPNN)

We train these networks to generate graph representations that we pass to a linear classifier to classify graphs into non-toxic and toxic.

We perform a hyperparameter search to identify the best model and evaluate the performances according to accuracy, balanced accuracy and the area under the ROC (Receiver Operating Characteristic) curve. The performances of the final models on the test set are visualized in the confusion matrix and the ROC curve below

## Explainability

## Directory Structure

```
xai-drug-discovery/
├── README.md
├── __pycache__/
├── data_1/
│   ├── raw/
│   │   └── tox21.csv
│   └── processed/
│       ├── pre_filter.pt
│       ├── pre_transform.pt
│       └── tox21.pt
├── data_exploration.ipynb
├── dataset.py
├── explain_results/
│   ├── GAT_Att.csv
│   ├── GAT_Dummy.csv
│   ├── GAT_GNN.csv
│   ├── GIN_Dummy.csv
│   ├── GIN_GNN.csv
│   ├── GIN_GNN_0.csv
│   ├── GIN_GNN_1.csv
│   ├── MPNN_Dummy.csv
│   └── MPNN_GNN.csv
├── explanation.ipynb
├── model_comparison.ipynb
├── model_training/
│   ├── Train_GAT.ipynb
│   ├── Train_GIN.ipynb
│   ├── Train_MPNN.ipynb
│   ├── data_1/
│   ├── dataset.py
│   ├── models.py
│   └── trainer.py
├── models/
│   ├── GAT/
│   │   └── cross-val/
│   │       ├── GAT_0_best.pth ... GAT_9_latest.pth
│   ├── GIN/
│   │   └── cross-val/
│   │       ├── GIN_0_best.pth ... GIN_9_latest.pth
│   ├── MPNN/
│   │   └── cross-val/
│   │       ├── MPNN_0_best.pth ... MPNN_9_latest.pth
│   └── final/
│       ├── GAT_best.pth
│       ├── GIN_best.pth
│       └── MPNN_best.pth
├── models.py
├── trainer.py
```

## Running the Code

`data_exploration.ipynb` For exploring the dataset and basic preprocessing. This includes

- Ensuring that the ID corresponds to an actual molecule and the entry is not *None*
- Sanitize the molecule: check and correct the chemical representation of the molecule
- Remove possible duplicates
- Extract meaningful properties both at the molecular and at the atomic level

`model_training` Contains the notebook files for training the 3 models and performing a hyperparameter search
`model_comparison.ipynb` Comparing the predictive accuracy of the 3 models
`explanation.ipynb` Applying GNNexplainer to all the 3 models and benchmarking the explanability



