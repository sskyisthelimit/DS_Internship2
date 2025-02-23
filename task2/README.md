## Main notebooks

1. **Pipeline inference demo: [task2/notebooks/demo.ipynb](task2/notebooks/demo.ipynb)**
1. **Evaluation/demo for NER: [task2/notebooks/nlp/demo_and_test_reports.ipynb](task2/notebooks/nlp/demo_and_test_reports.ipynb)**
1. **Train/eval for CV classifier: [task2/notebooks/cv/animal_classification_train&eval.ipynb](task2/notebooks/cv/animal_classification_train&eval.ipynb)**

---

## Steps that was made while working on task2:

### 0. collect animal names:
- After looking on scientific databases - the problem was that we need more 'common' animal names, not latin scientific, so
- so i used all possible online llms, generated raw dataset, preprocessed with: [task2/notebooks/nlp/names_preprocessing.ipynb](task2/notebooks/nlp/names_preprocessing.ipynb)

### 1. Generate synthetic data using Llama 3.1
- Using names from step above:
  - Notebook 1: [task2/notebooks/nlp/animals_900_synthetic.ipynb](task2/notebooks/nlp/animals_900_synthetic.ipynb) (result in `task2/dataset/raw/generated.zip`)
- Because of temperature (model param) some outputs were invalid so:
  - Notebook 2: [task2/notebooks/nlp/failed_animals_synthetic.ipynb](task2/notebooks/nlp/failed_animals_synthetic.ipynb) (`task2/dataset/raw/generated1.zip`)
- using llms was generated: `task2/dataset/raw/general_animal_names.json`, and used in notebook below.
  - Notebook 3: [task2/notebooks/nlp/general_names_synthetic.ipynb](task2/notebooks/nlp/general_names_synthetic.ipynb) (`task2/dataset/raw/generated2.zip`)

### 2. Preprocess data to create training, validation datasets (not test, test dataset is small but from real world distribution):
- Dataset creation notebook: [task2/notebooks/nlp/dataset_creation.ipynb](task2/notebooks/nlp/dataset_creation.ipynb)
- Resulting train, val datasets in: `task2/dataset/processed`

### 3. Manually collect samples from reddit, annotate using [https://arunmozhi.in/ner-annotator/](https://arunmozhi.in/ner-annotator/)
- Resulting dataset: `task2/dataset/processed/test_dataset.json`

### 4. Train bert-base-cased: [task2/notebooks/nlp/ner_train.ipynb](task2/notebooks/nlp/ner_train.ipynb)

### 5. Test eval/demo: [task2/notebooks/nlp/demo_and_test_reports.ipynb](task2/notebooks/nlp/demo_and_test_reports.ipynb)

---

## Computer vision part:

- **10 animal classes dataset:** [https://www.kaggle.com/datasets/alessiocorrado99/animals10](https://www.kaggle.com/datasets/alessiocorrado99/animals10)

### 0. EDA:
- [task2/notebooks/cv/EDA.ipynb](task2/notebooks/cv/EDA.ipynb)

### 1. Train & eval:
- [task2/notebooks/cv/animal_classification_train&eval.ipynb](task2/notebooks/cv/animal_classification_train&eval.ipynb)

---

## Final result (inference script):

`task2/notebooks/demo.ipynb`

## Usage

### Prerequisites
- **Python 3.9+**

### Setup Instructions
1. **Create a virtual environment:**
```bash
   python3 -m venv virt
   source virt/bin/activate
```

2. **Install the required dependencies, project as executable:**
```bash
  python3 -m pip install -r requirements.txt
  python3 -m pip install -e .
```

## Project Structure

- **`src/`**:
  - Contains training, inferencing, eval scripts for cv, nlp

- **`notebooks/`**:
  - Includes main demonstration notebook, training & evaluation notebooks for cv & nlp.

- **`datasets/`**:
  - Stores processed NER datasets, raw NER datasets, cv partial dataset.


## Scripts Overview

## main inference: task2/src/inference.py command-line arguments
| Argument             | Type   | Default       | Description                                               |
|----------------------|--------|---------------|-----------------------------------------------------------|
| `--image_path`       | string | Required      | Path to the image.                                        |
| `--sentence`         | string | Required      | Sentence that will be used        .                       |
| `--device`           | string | Required      | Device to use (`cuda:0` for GPU or `cpu`).                |
--	string	./	 (e.g., CNN_report.log, FCNN_report.log, RF_report.log).
