### Project Goal
####  End-to-End ML Pipeline with Model Deployment
Project scope:

- Build complete QSAR prediction system (EC50/IC50 prediction)
- Include data preprocessing, feature engineering, model training, and deployment
- Create REST API for molecular property predictions
- Add model monitoring and retraining capabilities
- Implement A/B testing framework for model comparison

skills: Shows full-stack ML engineering skills, not just modeling
Technical stack: Python, FastAPI/Flask, Docker, AWS/GCP, MLflow, PostgreSQL

### This Dataset allows for Industry-Relevant Skills Demonstration
- Check Efficacy using a pre-processed dataset (CA, CM, CI classes) from [Moleculenet.ai](https://wiki.nci.nih.gov/spaces/NCIDTPdata/pages/158204006/AIDS+Antiviral+Screen+Data)


This dataset is particularly demonstrates:
- Real pharmaceutical data experience
- Regulatory-compliant dataset (publicly available, no IP issues)
- Large-scale data handling (40K+ compounds)
- Multi-modal learning (structure + activity data)
- Biomedical domain knowledge

### Procedure: 
Data Sources(raw csv files) → Data Pipeline(preprocessing and validation) → Feature Store(molecular descriptors) → Model Training(multiple models) → Model Registry(MLflow registry) → Deployment(FastAPI, REST API) → Monitoring (Grafana Dashboard)


- Merge Data: Link NSC across files to combine screening results, EC50/IC50, and structures.
- Filter Compounds: Focus on CA/CM for active candidates.
- Calculate Selectivity Index (SI): SI = IC50/EC50 to identify compounds with high efficacy and low toxicity.
  
- Data preprocessing :
  - Manage duplicate entries,
  - Mismatched screening conclusions,
  - flag interpretation sign to values and
  - Handle missing data.
    
- ML model:
 - Performing random splitting (80% train, 20% test). 
 - Extracted molecular descriptors (e.g., logP, Morgan Fingerprints, MORSE) from data. 
 - Training base models, check with test data.
 - Evaluated models using accuracy, F1-score, and Cohen’s kappa, aligning predictive insights with clinical research. 


