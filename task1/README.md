## Usage

### Prerequisites
- **Python 3.9+**

### Setup Instructions
1. **Create a virtual environment:**
```bash
   python3 -m venv virt
   source virt/bin/activate
```

2. Install a version of [PyTorch](https://pytorch.org/get-started/locally/) that supports GPU if required.
3. **Install the required dependencies, project as executable:**
```bash
  python3 -m pip install -r requirements.txt
  python3 -m pip install -e .
```

## Project Structure

- **`src/`**:
  - Contains classes definitions, utility functions and scripts for training and inference.

- **`notebooks/`**:
  - Includes demonstration notebook, training & evaluation notebook.

- **`datasets/`**:
  - Stores raw MNIST dataset.

- **`weights/`**:
  - Stores trained weights for each model.

- **`logs/`**:
  - Stores classification reports.

## Scripts Overview

## train.py command-line arguments
| Argument             | Type   | Default       | Description                                               |
|----------------------|--------|---------------|-----------------------------------------------------------|
| `--datapath`         | string | Required      | Path to the folder containing the dataset files           |
| `--batch_size`       | int    | 64            | Batch size for training the models.                       |
| `--weights_save_dir` | string | Required      | Directory where the trained model weights will be saved.  |
| `--device`           | string | Required      | Device to use (`cuda:0` for GPU or `cpu`).                |
--	string	./	 (e.g., CNN_report.log, FCNN_report.log, RF_report.log).

## eval.py command-line arguments
| Argument             | Type   | Default       | Description                                               |
|----------------------|--------|---------------|-----------------------------------------------------------|
| `--datapath`         | string | Required      | Path to the folder containing the dataset files           |
| `--batch_size`       | int    | 64            | Batch size for training the models.                       |
| `--weights_save_dir` | string | Required      | Directory where the saved model weights are located.      |
| `--device`           | string | Required      | Device to use (`cuda:0` for GPU or `cpu`).                |
| `--reports_dir`      | string | ./            | Directory to save the evaluation report log files.        |
