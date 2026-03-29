# AI Project — UGent

## Project Structure

```
data/
├── 1_source/          # Given data for the project
├── 2_downsampled/     # Downsampled datasets (244×244 → 128×128)
└── 3_augmented/       # Augmented training sets based on downsampled data
    ├── v1/            # All images vertically rotated
    ├── v2/            # v1 + small rotation
    ├── v.../
    └── vbest/         # Final set of augmented examples

models/
├── baseline/
│   ├── doc/           # Documentation, references
│   └── weights/       # Saved weights of the final model
└── resnet50/
    ├── doc/           # Documentation, references
    └── weights/       # Saved weights of pretrained model + updated final model

notebooks/
├── 0_generalcode/     # General Python functions to include where needed
└── <task subfolders>/ # One subfolder per notebook

report/                # Folder for the report
assignment_details/    # PDF and slides for the assignment
```

---

## Project packages
| Allowed | Used | Package | Purpose |
|---------|------|---------|---------|
| Y | N | numpy | Numerical computing — arrays, matrix operations, math functions |
| Y | N | pandas | Data manipulation — DataFrames, loading/cleaning tabular data (CSV, Excel) |
| Y | N | matplotlib | Low-level plotting library — line charts, histograms, custom visualizations |
| Y | N | seaborn | High-level statistical plots built on matplotlib — heatmaps, distributions, correlation plots |
| Y | N | scikit-learn | Classical ML — classifiers, regressors, preprocessing, cross-validation, metrics |
| Y | Y | torch | Deep learning framework by Meta (PyTorch) — flexible neural network research & training |
| Y | Y | scikeras | Scikit-learn wrapper for Keras/TensorFlow — use Keras models with sklearn APIs (GridSearchCV, pipelines) |
| Y | N | scipy | Scientific computing — statistical tests, signal processing, optimization, linear algebra |
| Y | N | imbalanced-learn | Handling class imbalance — SMOTE oversampling, undersampling, resampling strategies |
| Y | Y | Pillow | Image processing — loading, resizing, converting image files (likely for X-ray/MRI data) |
| Y | N | requests | HTTP requests — downloading data or calling REST APIs |
| Y | N | tensorboard | Visualization tool for TensorFlow/PyTorch training — loss curves, metrics dashboards |
| N | Y | os | interacts with the operating system |
| N | Y | sys | interacts with the Python interpreter itself |
| N | Y | pathlib | interacts with the filesystem (different operating systems) |

## Workflow & Task Dependencies

| Task | Description | Owner | Notebook |
|------|-------------|-------|----------|
| 1 | Data Exploration, Pre-Processing and Augmentation | Karel | `preprocessing.ipynb` |
| 2 | Building a Simple Baseline Model | Carlo | `baseline.ipynb` *(depends on Task 1)* |
| 3 | Transfer Learning | Michiel | `transferlearning.ipynb` *(depends on Task 1)* |
| 4 | Latent Space Analysis and Error Inspection | TBD | `latentspace.ipynb` *(depends on best model from Task 3)* |
| 5 | Model Explainability with Grad-CAM | TBD | `GradCAM.ipynb` *(depends on Task 2 or 3)* |
| 6 | Report *(max 6 pages)* | TBD | — |

---

## Deadline

**30 April 2026 — 23:59**
