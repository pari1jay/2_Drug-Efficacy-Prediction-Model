# Efficacy-Prediction-Model

### 1. Efficacy Prediction Model 
- Check Efficacy using a pre-processed dataset (CA, CM, CI classes) from [Moleculenet.ai](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/158204006/AIDS+Antiviral+Screen+Data)
- Merge Data: Link NSC across files to combine screening results, EC50/IC50, and structures.
- Filter Compounds: Focus on CA/CM for active candidates.
- Calculate Selectivity Index (SI): SI = IC50/EC50 to identify compounds with high efficacy and low toxicity.
  
- ML model: performing random splitting (80% train, 20% test). 
- Extracted 1D/2D/3D molecular descriptors (e.g., logP, Morgan Fingerprints, MORSE) from graph-structured molecules, 
training base models and stacking them with cross-validation for enhanced predictions. 
- Implemented a Graph Convolutional Network to learn neural fingerprints from molecular structures, and a CNN 
(InceptionV3 backbone) on SMILES strings for representation learning, optimizing drug efficacy classification.  
- Evaluated models using accuracy, F1-score, and Cohen’s kappa, aligning predictive insights with clinical research 
objectives. 
