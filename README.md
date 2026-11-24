# ArtFace

The repository for the paper: "Deep-learning Based Identity Verification of Sitters in Historical Portraiture: a Pilot Study on the RKD Collection" (GCH 2026, [doi:10.2312/gch.20261001](https://doi.org/10.2312/gch.20261001)). See [Citation](#citation).

The goal of this project is to investigate whether standard face recognition networks — trained on photographs — can match faces of the same sitters (people depicted) across historical paintings. The original painting data can be found in `data/paintings/`, which is a subset of paintings from the RKD dataset selected for this experiment.

As a **reference dataset**, the same pipeline is applied to a curated set of historical painted portraits in `data2/` (764 portraits of 210 sitters, mostly oil paintings from museum collections). This dataset is based on Huber et al., "Verification of Sitter Identity Across Historical Portrait Paintings by Confidence-aware Face Recognition", ICPR 2022 ([doi:10.1109/ICPR56361.2022.9956452](https://doi.org/10.1109/ICPR56361.2022.9956452)). Each portrait has structured labels (sitter, age, sex, pose, artist, year, location, technique and source; see `data2/HP_labels_v8_source.csv`). The results on the RKD collection are compared with the results on this reference dataset in `compare_features.ipynb`.

The detection and recognition are primarily based on: https://github.com/deepinsight/insightface

---

## Data Availability

The image data is **not public** and is therefore not included in this repository. The `data/` folder in this repository only contains a single example image per sub-folder to show the expected structure. Please contact the authors to request access to the RKD data set.

The RKD metadata is shared as `rkd_metadata.zip` (names of private owners removed). Unpack it into `data/`; the code expects the file at `data/rkd_metadata.xml`, so rename the XML if it has a different name.

The reference dataset in `data2/` is third-party data as well: it comes from Huber et al. (ICPR 2022) and must be requested from the authors of that paper.

The pre-trained face detection and face recognition models are **third-party models** from InsightFace and are not distributed with this repository either — see [Pre-trained Models](#pre-trained-models).

---

## Repository Structure

```
ArtFace/
├── notebooks/          Jupyter notebooks (data preparation and evaluation)
├── results/            Extracted features and the lists of faces used in the paper
├── figures/            Figures produced by the notebooks
├── face_detection/     Face detection code (RetinaFace)
├── face_recognition/   Face recognition code (ArcFace)
├── gender_labeler/     GUI tool for labelling the gender of each sitter
├── data/               RKD data (not included, available on request)
└── data2/              Reference dataset (not included, request from Huber et al.)
```

The notebooks use paths relative to the `notebooks/` folder (e.g. `../data/`, `../results/`), so start Jupyter from the repository root or from `notebooks/`.

| Path                                        | Description                                                                                                                                              |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/paintings/`                         | Original paintings (subset of the RKD dataset)                                                                                                           |
| `data/detections/`                        | Paintings with the detected bounding boxes drawn in (output of face detection, for visual inspection)                                                    |
| `data/faces/`                             | Cropped and aligned face images (output of face detection)                                                                                               |
| `data/sitters/`                           | Faces organised in one folder per identity (RKD sitter ID); output of the manual labelling                                                               |
| `data/sitters_multi_face/`                | A **copy/subset** of `data/sitters/` containing only the identities with two or more faces. Do not add these to `data/sitters/` when counting. |
| `data/rkd_metadata.xml`                   | RKD metadata of the paintings and sitters (from `rkd_metadata.zip`)                                                                                               |
| `data2/`                                  | Reference dataset — historical painted portraits with structured labels (`HP_labels_v8_source.csv`)                                                   |
| `results/features.pkl`                    | Pandas DataFrame with extracted feature vectors per face (RKD dataset, before the manual removal of faces)                                               |
| `results/face_list_all.csv`               | The 1,884 faces in `features.pkl` (`folder`, `filename`), without the feature vectors                                                              |
| `results/face_list_paper.csv`             | The 1,859 faces used in the paper (after the manual removal of faces)                                                                                    |
| `results/face_list_multi.csv`             | Evaluation subset: the 205 faces of the 71 identities with two or more faces                                                                             |
| `results/face_list_multi_excl_111078.csv` | Evaluation subset without sitter `111078`: 194 faces of 70 identities                                                                                  |
| `notebooks/explore_metadata.ipynb`        | Exploration of the RKD metadata                                                                                                                          |
| `notebooks/data_preparation.ipynb`        | Data preparation for the RKD dataset (builds `data/sitters/`)                                                                                          |
| `notebooks/data_preparation2.ipynb`       | Data preparation for the reference (historical portraits) dataset                                                                                        |
| `notebooks/examine_features.ipynb`        | Feature evaluation for the RKD dataset                                                                                                                   |
| `notebooks/examine_features2.ipynb`       | Feature evaluation for the historical portraits dataset                                                                                                  |
| `notebooks/compare_features.ipynb`        | Side-by-side comparison of recognition performance across both datasets                                                                                  |
| `gender_labeler/gender_labeler.py`        | GUI tool to manually assign a gender label to each sitter                                                                                                |
| `gender_labeler/gender_labels.csv`        | Output of the gender labeller — one row per painting/sitter                                                                                             |

> **Note:** The Jupyter Notebook files do not require a specific Python version. The face detection and face recognition code do have environment requirements — see the sections below.

---

## Pipeline Overview

### 1. Data Preparation / Face Detection

To perform face recognition, the faces first need to be extracted and aligned:

1. **Detect** all faces in each painting.
2. **Align** each face using an affine transformation so that the eyes, nose, and mouth are at consistent locations across all images (`norm_crop` was added to the original `test_all.py` for this step).
3. **Organise** the cropped faces into per-identity folders so the folder name can serve as a label.

The detection code lives in `face_detection/`. It is based on RetinaFace from the InsightFace repository. The relevant original code is under `insightface/detection/retinaface/`; the files in this repo were copied into that folder structure.

**Environment:** MXNet with GPU support is required. This is no longer supported for the latest Python versions — Python 3.7 is recommended. Running in a Linux/WSL environment is strongly advised.

**Output naming:** Each cropped face is saved as `<original_filename>-<face_index>.jpg` (e.g. a painting with three faces produces `-0`, `-1`, `-2`). The resulting files are stored in `data/faces/`; the paintings with the detections drawn in are stored in `data/detections/`.

#### Manual identity labelling

After detection, faces were linked to identities through a semi-manual process:

- Art pieces containing exactly one sitter and one detected face were matched automatically, with a sanity check to catch errors.
- For paintings with multiple faces, the identities from the RKD metadata were used to match each face. The metadata often included the location of the sitter in the composition; in ambiguous cases the identity was deduced manually, or the face was discarded.
- Single-person paintings for which no identity could be found in the metadata were rejected.
- Afterwards, the identity folders were reviewed manually. Faces that ended up under two identities, faces of too low quality for reliable recognition, and faces matched to the wrong person were removed **by hand** from `data/sitters/` (see [Dataset Statistics](#dataset-statistics)).

The curation notes (which paintings were matched by hand, rejected, or removed as duplicates) are available on request.

If the full pipeline is followed correctly, the result should be reproducible as the `data/sitters/` folder.

#### Dataset Statistics

The numbers reported in the paper are **1,673 paintings, 1,859 faces and 1,725 identities**. They are obtained as follows:

| Step                                                                |      Identities |           Faces |
| ------------------------------------------------------------------- | --------------: | --------------: |
| Paintings in `data/paintings/`                                    |              – | 1,673 paintings |
| Faces detected in `data/faces/`                                   |              – |           2,440 |
| Identity folders in `data/sitters/` (incl. 156 empty folders)     |           1,882 |              – |
| Features extracted (`results/features.pkl`)                       |           1,750 |           1,884 |
| After manually removing duplicate, low-quality and mismatched faces | **1,725** | **1,859** |
| Evaluation subset: identities with ≥ 2 faces                       |              71 |             205 |
| Evaluation subset without sitter `111078`                         |              70 |             194 |

Notes:

- **Empty folders**: `notebooks/data_preparation.ipynb` creates a folder for every sitter in the metadata. If no face could be matched to a sitter, the folder stays empty. Empty folders are ignored.
- **`data/sitters_multi_face/`** is a copy of the 71 identities with two or more faces. It is a subset of `data/sitters/`; do not add both folders together.
- **Removed faces**: the features in `results/features.pkl` were extracted *before* the manually removed faces were deleted from `data/sitters/`. `examine_features.ipynb` and `compare_features.ipynb` therefore keep only the faces listed in `results/face_list_paper.csv` (1,884 → 1,859). This does not depend on the contents of `data/sitters/`; if you extract new features from the cleaned folder, nothing is dropped.
- **Sitter `111078`** is excluded from the second evaluation subset because all 11 of their images come from a single photo session, which would inflate the genuine scores.
- The exact faces used in the analysis are listed in `results/face_list_paper.csv`; the evaluation subsets in `results/face_list_multi.csv` and `results/face_list_multi_excl_111078.csv`.

---

### 2. Face Recognition

The face recognition network (ArcFace) takes an aligned face image as input and produces a **512-dimensional feature vector**.

Similarity between vectors is measured using **cosine similarity**:

- A cosine similarity of **1** means the vectors are identical.
- A cosine similarity of **-1** means the vectors point in opposite directions.

Pairs of faces from the **same identity** are called *Genuine Matches*; pairs from **different identities** are called *Impostor Matches*. By sweeping over similarity thresholds, performance is evaluated as the **True Acceptance Rate (TAR) at a fixed False Acceptance Rate (FAR)** — for example, TAR at FAR = 0.1% tells us what fraction of genuine matches are correctly accepted when only 0.1% of impostor matches are incorrectly accepted.

The recognition code lives in `face_recognition/` and is based on `insightface/recognition/arcface_torch/`. The environment requirements are less strict than for face detection, but running in a separate environment is still recommended.

**Output:** `features.pkl` (stored in `results/`) — a Pandas DataFrame with columns:

- `folder` — identity label (folder name)
- `filename` — source file name
- `feature` — 512-element feature vector

This file is used by the Jupyter Notebooks for further evaluation and analysis.

---

### 3. Gender Labelling

`gender_labeler/gender_labeler.py` is a standalone Tkinter GUI for manually assigning a gender label (Male / Female / Unknown) to each identity in `data/sitters/`. One label is assigned per person — all face crops for that person are shown alongside thumbnails of the corresponding paintings so the labeller has full context.

**Inputs:**

- `data/sitters/` — identity folders produced by the pipeline
- `data/detections/` — painting images with bounding boxes (for context thumbnails)
- RKD metadata XML — used to display the sitter's full name

**Output:** `gender_labeler/gender_labels.csv` — one row per painting/sitter combination with columns `painting_stem`, `person_id`, `full_name`, `gender`. The tool resumes automatically from where it left off.

**Keyboard shortcuts:** `m` = Male, `f` = Female, `u` = Unknown, `←`/`→` = navigate. All folder paths are configurable through a settings dialog at startup and are persisted in `gender_labeler/gender_labeler_config.json` (a local file, not tracked in git). By default the tool looks for the repository's `data/` folder. The tool can also be packaged as a standalone executable using PyInstaller (see `gender_labeler/GenderLabeler.spec` and `gender_labeler/build_exe.bat`).

---

## Jupyter Notebooks

### explore_metadata.ipynb

Explores the RKD metadata in `data/rkd_metadata.xml`. Builds two mappings:

- paintings → detected face files in `data/faces/`
- paintings → their RKD priref record

Provides a per-painting inspection tool: given a filename or index, it prints the painting title, artist attribution, and full sitter metadata (person ID, full name, description, identification status and certainty, birth/death dates). Also contains a cell that dumps the entire XML tree structure for reference. This notebook was used to support the manual identity labelling step.

### data_preparation.ipynb

Data preparation for the RKD dataset. Reads the XML metadata and the contents of `data/paintings/` and `data/faces/`, then builds:

- `prirefs_beschikbaar` — prirefs (RKD artwork records) for which a painting file is available
- `zitters` — sitter ID → list of painting prirefs that depict that sitter
- A list of sitters appearing in more than one painting

Then copies the detected face crops from `data/faces/` into per-identity folders under `data/sitters/{sitter_id}/`. Also includes a snippet that copies only the identities with two or more face images into a separate folder `data/sitters_multi_face/` for evaluation (a subset of `data/sitters/`, not additional data).

### data_preparation2.ipynb

Data preparation for the reference dataset — historical painted portraits with structured ground-truth labels in `data2/HP_labels_v8_source.csv`. The CSV contains per-image fields: sitter ID (`pid`), full name, age, sex, pose, artist, year, location, and source URL. The notebook:

1. Builds a `zitters` dictionary mapping each sitter ID (`pid`) to the list of available face image files.
2. Lists sitters that appear in more than one painting.
3. Copies the face images into per-sitter folders under `data2/historical_sitters/{pid}/` — the same structure as `data/sitters/` — so the same recognition pipeline applies.

### examine_features.ipynb

Evaluates face recognition performance on the RKD dataset. Loads `results/features.pkl` (1,884 rows → 1,859 after dropping the faces that were removed by hand, see [Dataset Statistics](#dataset-statistics)) covering 1,725 unique identities. Restricts evaluation to sitters with more than one face image, and excludes sitter `111078` whose images come from a single photo session (which would bias the genuine scores).

Uses **leave-one-out** validation to compute genuine and impostor cosine similarity scores, then produces:

- Genuine vs. Impostor score distribution histogram
- ROC curve (linear and log-scale) with EER marked
- TAR at standard FAR operating points (0.001 %, 0.01 %, 0.1 %, 1 %, 10 %)
- T-SNE visualisation of the 512-dimensional embedding space (for multi-face identities only)
- Binary similarity heatmap at a fixed threshold
- Image pair export: saves side-by-side face crop images for True Matches, False Matches, and False Rejects into separate folders for qualitative inspection

### examine_features2.ipynb

Same evaluation framework as `examine_features.ipynb`, applied to the historical portraits dataset (`data2/features_historical.pkl`). Contains 764 rows across 210 unique identities. Uses leave-one-out to compute:

- Genuine vs. Impostor score distribution histogram
- ROC curve (linear and log-scale) with EER (~14.6%) marked
- TAR at standard FAR operating points
- (Commented-out) T-SNE visualisation
- Similarity matrix heatmap

### compare_features.ipynb

Loads both `results/features.pkl` (RKD paintings) and `data2/features_historical.pkl` (historical portraits) and computes leave-one-out metrics for three conditions:

- **Historical portraits** — the structured `data2` dataset
- **RKD (all identities)** — the full RKD feature set
- **RKD (multi-face only)** — restricted to identities with more than one face image

Produces overlaid score distribution plots and ROC curves (linear and log-scale) for a direct side-by-side comparison of recognition performance across both datasets.

---

## Setup

Clone the InsightFace repository alongside this one:

```bash
git clone https://github.com/deepinsight/insightface
```

Copy the files from `face_detection/` into `insightface/detection/retinaface/` and the files from `face_recognition/` into `insightface/recognition/arcface_torch/`.

### Pre-trained Models

The models are **third-party models published by InsightFace** and are not part of this repository; they are available for non-commercial research purposes only. Download them from the original InsightFace pages and place them at the paths the scripts expect:

| Step             | Model                                                          | Download from                                                                                                          | Expected location (default argument)                                                                                |
| ---------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| Face detection   | RetinaFace-R50                                                 | [InsightFace RetinaFace](https://github.com/deepinsight/insightface/tree/master/detection/retinaface)                   | `face_detection/model/R50/R50-0000.params` + `R50-symbol.json` (`--model ./model/R50/R50` in `test_all.py`) |
| Face recognition | ArcFace R100 trained on Glint360K (`glint360k_cosface_r100`) | [InsightFace arcface_torch model zoo](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) | `face_recognition/glint360k_cosface_r100/backbone.pth` (`--weight` in `inference.py`)                         |

See also `face_detection/model/R50.txt` and `face_recognition/glint360k_cosface_r100.txt` for the expected folder layout.

---

## Citation

If you use this code or data, please cite:

```bibtex
@inproceedings{10.2312:gch.20261001,
booktitle = {Eurographics Workshop on Graphics and Cultural Heritage (GCH 2026)},
editor = {Pintus, Ruggero and Bosch, Carles and Schroer, Carla and Charalambous, Panagiotis and Ciortan, Irina and Kotzinos, Dimitris and Andujar, Carlos and Dininno, Domenica},
title = {{Deep-learning Based Identity Verification of Sitters in Historical Portraiture: a Pilot Study on the RKD Collection}},
author = {Bunda, Sebastian T. and Klink, Matthies and Bunda, Lisa M. and Talavera, Estefanía and Costiner, Lisandra S. and SERO, DZEMILA},
year = {2026},
publisher = {The Eurographics Association},
ISSN = {2312-6124},
ISBN = {978-3-03868-329-2},
DOI = {10.2312/gch.20261001}
}
```

If you use the reference dataset in `data2/`, please also cite:

```bibtex
@INPROCEEDINGS{9956452,
  author={Huber, Marco and Terhörst, Philipp and Luu, Anh Thi and Kirchbuchner, Florian and Damer, Naser},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  title={Verification of Sitter Identity Across Historical Portrait Paintings by Confidence-aware Face Recognition},
  year={2022},
  pages={938-944},
  doi={10.1109/ICPR56361.2022.9956452}}
```
