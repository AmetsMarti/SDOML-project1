

## Spoken Digit Classification
*Amets Martiarena, Teva Philippe, Alvaro Crespo*

[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://ametsmarti.github.io/SDOML-project1/)

### Project description
This project uses the [Audio MNIST Dataset](https://huggingface.co/datasets/gilkeyio/AudioMNIST) to train a model that predicts numbers from 0 to 9 based on audio input. The neural network is a multilayer perceptron with an input of 26 neurons, two hidden layers of 64 and 32 neurons, and a final output layer of 10 neurons. It uses the Adam optimizer and the cross-entropy loss function.

### Running the code
In order to run the jupyter notebook, you can use `uv sync` to create an environment and synchronize the dependencies. Depending on your code editor, you might need to manually select the created environment as a Python kernel. For a more detailed description of all the project's requirements, please check `pyproject.toml`. The notebooks are divided into two: The 00 notebook downloads and visualizes the data. The 01 notebooks performs the feature extracion and model training.

### Dependency Management with uv
This project uses [uv](https://docs.astral.sh/uv/) for dependency management and virtual environment handling.

**Setup on a fresh system:**
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync
```

### Documentation

The published documentation is available at:

**[https://ametsmarti.github.io/SDOML-project1/](https://ametsmarti.github.io/SDOML-project1/)**

To regenerate it locally:
```bash
# Using Make
make docs

# Or directly with pdoc
pdoc sdoml_task1 -o docs --docformat numpy
```

The generated documentation will be available in the `docs/` directory. Open `docs/index.html` in a browser to view the documentation.

### Project Organization
The project is organized by making use of the [cookie cutter template](https://cookiecutter-data-science.drivendata.org/)
<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- Contain documentation of the project
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         sdoml_task1 and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── sdoml_task1   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes sdoml_task1 a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------
