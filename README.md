Workflow Instructions:

Here is the step-by-step guide to reproducing our results:

Data Preprocessing: First, download the PDBbind database (specifically the refined set) 
and the scPDB database. Run the data preprocessing script using these datasets to obtain 
the processed protein pocket data.

GPS-VAE Training: Run the GPS model training script using the processed protein data 
from the previous step.

Transformer-SELFIES Training: Prepare the preprocessed SMILES strings from 
the ChEMBL database, and run the Transformer-SELFIES model training script.

Molecular Generation: Take the best-performing checkpoints from both the GPS 
and Transformer-SELFIES models,and run the molecular generator script to produce 
the initial molecular candidates.

Evolutionary Optimization: Feed the generated models/molecules into 
the STONED algorithm script to perform evolutionary optimization 
and obtain the final molecular structures.

Evaluation: Download and install AutoDock Vina. Finally, run the evaluation scripts 
to perform molecular docking, QED calculations, and other relevant property assessments 
on the final generated results.