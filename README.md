# 🧪 HIV Drug Effectiveness Prediction using Machine Learning
### 🚀 Project Goal
Build a machine learning model to predict which chemical compounds can fight HIV effectively, helping researchers focus on the most promising candidates and skip compounds that likely won't work.


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

### F1-score : 
A balance between precision and recall — especially useful when the classes (e.g., active vs. inactive compounds) are imbalanced. It ensures the model isn't just accurate, but also reliably identifies true positives without too many false alarms.

### Cohen’s Kappa : 
Measures how well your model agrees with actual labels beyond chance. It's a more robust metric than accuracy when classes are uneven or when guessing could yield misleading results.

## 💡 Why This Matters
This tool helps scientists focus lab efforts only on promising compounds, accelerating the HIV drug discovery process — just like a spam filter helps skip useless emails.
