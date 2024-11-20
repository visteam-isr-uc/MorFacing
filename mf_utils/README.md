# Generate Predictions for MMPMR Estimation

This repository contains a Python script designed to generate random test embeddings to estimate MMPMR (Mean Morph Pair Match Rate). The script creates synthetic embeddings for intra-class (enrollment and reference data) and interpolated embeddings for morphs, with the level of similarity controlled by various coefficients.

## Overview

The script generates embeddings for three types of data:
1. **Enrollment Data**: Creates synthetic embeddings representing intra-class variations.
2. **Reference Data**: Generates reference embeddings by adding random perturbations to enrollment data.
3. **Morph Data**: Creates synthetic embeddings for morph images by combining features from two different enrolled classes.


## Usage

The script can be executed from the command line using Python. The primary parameters are specified via command-line arguments.

### Arguments

Below are the available arguments that can be passed to the script:

| Argument                 | Description                                            | Default Value                            |
|--------------------------|--------------------------------------------------------|------------------------------------------|
| `--path_to_data`         | Path to the root directory containing benchmark data.  | `../data/data_aligned/alignment_default/`|
| `--path_to_predictions`  | Path to the directory where predictions will be saved. | `../models/test_model/predictions`       |
| `--dataset_enrollment`   | Relative path to the enrollment dataset.               | `/FACING2/Originals/Enrollment`          |
| `--dataset_reference`    | Relative path to the reference dataset.                | `/FACING2/Originals/Reference`           |
| `--datasets_morph`       | Relative path to the morph dataset.                    | `/FACING2/Morphs/`                       |
| `--feature_size`         | Dimensionality of the feature vector.                  | `128`                                    |
| `--extention`            | File extension for the images.                         | `.jpg`                                   |
| `--similarity_enrollemnt`| Similarity coefficient for enrollment data.            | `0.8`                                    |
| `--similarity_reference` | Similarity coefficient for reference data.             | `0.8`                                    |
| `--similarity_morph`     | Similarity coefficient for morph data.                 | `0.9`                                    |

### Example

To run the script, use the following command:
```bash
python generate_predictions.py \
    --path_to_data "../data/data_aligned/alignment_default/" \
    --path_to_predictions "../models/test_model/predictions" \
    --dataset_enrollment "/FACING2/Originals/Enrollment" \
    --dataset_reference "/FACING2/Originals/Reference" \
    --datasets_morph "/FACING2/Morphs/" \
    --feature_size 128 \
    --extention ".jpg" \
    --similarity_enrollemnt 0.8 \
    --similarity_reference 0.8 \
    --similarity_morph 0.9
```




