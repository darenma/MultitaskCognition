# MultitaskCognition

This is the Repository for the manucsript "Predicting categorical and continuous outcomes of subjects on the Alzheimer’s Disease spectrum using a single MRI without PET, cognitive or fluid biomarkers".

## Currently cleaning, documentating, and formatting the code. The full model structure will be uploaded by 5pm PST Sep.30th (Monday).

## Data Availability:
Please download the data accordingly:
The structural MRI data used in this work are available to researchers via the Human Connectome Project - Young Adults (HCP-YA) data access procedure described at https://www.humanconnectome.org/study/hcp-young-adult/document/extensively-processed-fmri-data-documentation and via the Alzheimer’s Disease Neuroimaging Initiative (ADNI) data access procedure described at http://adni.loni.usc.edu/data-samples/accessdata/.

We will shortly share 2 example MRI scans for testing the script.

## Step 1: Co-registration and Segmentation using [FSL-FAST](https://web.mit.edu/fsl_v5.0.10/fsl/doc/wiki/FAST.html) on MRI files.

For example: `fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o output_name input_image.nii.gz`

Here `-t 1` means we are executing on T1-weighted images, and `-n 3` tells the model to segment for 3 tissues (GM, WM, CSF)

The `.nii` NIFTI files were obtained by using the dcm to nii tool in Python, or, some websites have provided them directly.

## Step 2: Train the UNet and MedicalNet Models.

## Step 3: Analyze the results and conduct feature importance.

## Step 4: Finetune the model on stage and seeding results from EBM (Paper 2: WIP).



##
If you have any question please reach out via daren.ma@ucsf.edu
