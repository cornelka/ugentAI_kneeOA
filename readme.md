# AI Project — UGent

## Project Structure

```
data/
└── 1_source/          # Given data for the project

notebooks/
├── 0_generalcode/     # General Python functions to include where needed
└── <task subfolders>/ # One subfolder per notebook, with weights and other info where applicable

results/               # Model compare results

report/                # Folder for the report
assignment_details/    # PDF and slides for the assignment
```

---

## Project packages
| Allowed | Used | Package | Purpose |
|---------|------|---------|---------|
| N | Y | os | Standard library — environment variables, file system paths |
| N | Y | sys | Standard library — Python interpreter interaction, sys.path manipulation |
| N | Y | pathlib | Standard library — object-oriented filesystem paths |
| N | Y | itertools | Standard library — iterator utilities (confusion matrix plotting) |
| N | Y | datetime | Standard library — timestamps for experiment logging |
| Y | Y | numpy | Numerical computing — arrays, matrix operations, math functions |
| Y | Y | pandas | Data manipulation — DataFrames, loading/cleaning tabular data (CSV, Excel) |
| Y | Y | matplotlib | Low-level plotting library — line charts, histograms, custom visualizations |
| Y | N | seaborn | High-level statistical plots built on matplotlib — heatmaps, distributions, correlation plots |
| Y | Y | scikit-learn | Classical ML — metrics, PCA, t-SNE, silhouette score, cross-validation |
| Y | Y | torch | Deep learning framework by Meta (PyTorch) — flexible neural network research & training |
| Y | N | scikeras | Scikit-learn wrapper for Keras/TensorFlow — use Keras models with sklearn APIs (GridSearchCV, pipelines) |
| Y | N | scipy | Scientific computing — statistical tests, signal processing, optimization, linear algebra |
| Y | N | imbalanced-learn | Handling class imbalance — SMOTE oversampling, undersampling, resampling strategies |
| Y | Y | Pillow | Image processing — loading, resizing, converting image files |
| Y | N | requests | HTTP requests — downloading data or calling REST APIs |
| Y | N | tensorboard | Visualization tool for TensorFlow/PyTorch training — loss curves, metrics dashboards |
| N | Y | torchvision | PyTorch image utilities — dataset loaders, transforms, pretrained models |
| N | Y | grad-cam | Gradient/score-based saliency maps for CNNs — GradCAM and ScoreCAM |
| N | Y | keras | High-level deep learning API (v3, backend-agnostic) — used for GPU verification |
| N | Y | tensorflow | Keras backend used in this environment |

## Workflow & Task Dependencies

| Task | Description | Owner | Notebook |
|------|-------------|-------|----------|
| 1 | Data Exploration, Pre-Processing and Augmentation | Karel | `preprocessing.ipynb` |
| 2 | Building a Simple Baseline Model | Carlo | `Final baseline.ipynb` *(depends on Task 1)* |
| 3 | Transfer Learning | Michiel | `transferlearning2 1.ipynb` *(depends on Task 1)* |
| 4 | Latent Space Analysis and Error Inspection | Karel | `latentspace2.ipynb` *(depends on best model from Task 2/3)* |
| 5 | Model Explainability with Grad-CAM | Karel | `GradCAM2.ipynb` *(depends on best model Task 2/3)* |
| 6 | Report *(max 6 pages)* | Michiel/Carlo | — |

---

## Deadline

**30 April 2026 — 23:59** > Extended to 7 May 2026 23:59
