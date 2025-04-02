# Drug-Efficacy-Prediction-Model

- Developed machine learning models to predict HIV drug efficacy using a pre-processed dataset (CA, CM, CI classes) 
from Moleculenet.ai, performing stratified random splitting (80% train, 20% test). 
- Extracted 1D/2D/3D molecular descriptors (e.g., logP, Morgan Fingerprints, MORSE) from graph-structured molecules, 
training base models and stacking them with cross-validation for enhanced predictions. 
- Implemented a Graph Convolutional Network to learn neural fingerprints from molecular structures, and a CNN 
(InceptionV3 backbone) on SMILES strings for representation learning, optimizing drug efficacy classification.  
- Evaluated models using accuracy, F1-score, and Cohen’s kappa, aligning predictive insights with clinical research 
objectives. 
