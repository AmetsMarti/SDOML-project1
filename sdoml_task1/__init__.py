"""
<!-- -->

 Project Description
-------------------
This package trains a neural network model to predict numbers from 0 to 9 based on audio input using the 
[Audio MNIST Dataset](https://huggingface.co/datasets/gilkeyio/AudioMNIST). The model is a multilayer perceptron 
with an input of 26 neurons, two hidden layers of 64 and 32 neurons, and a final output layer of 10 neurons. 
It uses the Adam optimizer and the cross-entropy loss function.

The code is organized into the following modules:

* Dataset downloading and loading (`sdoml_task1.dataset`)
* Audio feature extraction and preprocessing (`sdoml_task1.features`)
* Model training and inference (`sdoml_task1.modeling`)
* Visualization and result analysis (`sdoml_task1.plots`)
* Project configuration (`sdoml_task1.config`)

 Usage
-----
This project uses [uv](https://docs.astral.sh/uv/) for dependency management and virtual environment
handling. If you want to run the notebooks yourself, make sure to generate the environment with `uv sync`.

Depending on your code editor, you might need to manually select the created environment as a Python kernel.
For a more detailed description of all the project's requirements, please check `pyproject.toml`.

There are four jupyter notebooks:

1. `00_download_dataset.ipynb`: Download the dataset, feature extraction and dataset visualization.
2. `01_audio_training.ipynb`: Model training.
3. `02_analysis_part.ipynb`: Result analysis and visualization.
4. `03_gradio_interface.ipynb`: The gradio app.

The notebooks are there for simplicity, but the detailed functions can be found in `sdoml_task1/`.

Additionally, there is a Docker image available to test the gradio app. The image is rather large (around 7GB), so keep that in mind when downloading it:

```bash
# Download the image
docker pull ametslortek/sdoml-demo:latest
```

Then to run the project, we also need to expose the containers network, so may vary between OSes.

```bash
docker run --network=host ametslortek/sdoml-demo:latest
```

With this, the interactive UI will run in localhost:7861

 Project Organization
--------------------
The project is organized using the [cookie cutter template](https://cookiecutter-data-science.drivendata.org/).

    ├── LICENSE
    ├── Makefile
    ├── README.md
    ├── data/
    ├── docs/
    ├── models/
    ├── notebooks/
    ├── pyproject.toml
    ├── references/
    ├── reports/
    ├── requirements.txt
    ├── setup.cfg
    └── sdoml_task1/
        ├── __init__.py
        ├── config.py
        ├── dataset.py
        ├── features.py
        ├── modeling/
        └── plots.py

*Made by: Amets Martiarena, Teva Philippe, Alvaro Crespo (GPL-3.0 license)*
"""

__all__ = ["config", "dataset", "features", "plots", "modeling"]