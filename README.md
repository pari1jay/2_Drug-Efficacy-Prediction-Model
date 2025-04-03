# Efficacy-Prediction-Model

### 1. Efficacy Prediction Model 

- Check Efficacy using a pre-processed dataset (CA, CM, CI classes) from [Moleculenet.ai](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/158204006/AIDS+Antiviral+Screen+Data)
- Merge Data: Link NSC across files to combine screening results, EC50/IC50, and structures.
- Filter Compounds: Focus on CA/CM for active candidates.
- Calculate Selectivity Index (SI): SI = IC50/EC50 to identify compounds with high efficacy and low toxicity.
  
- Data preprocessing :
  - Manage duplicate entries,
  - Mismatched screening conclusions,
  - flag interpretation sign to values and
  - Handle missing data.
    
- ML model: performing random splitting (80% train, 20% test). 
- Extracted molecular descriptors (e.g., logP, Morgan Fingerprints, MORSE) from data, 
- training base models, check with test data/.
- Evaluated models using accuracy, F1-score, and Cohen’s kappa, aligning predictive insights with clinical research 
objectives. 
