"""
Spoken digit classification (SDOML Task 1).

 Project Description
-------------------
This project uses the [Audio MNIST Dataset](https://huggingface.co/datasets/gilkeyio/AudioMNIST)
to train a model that predicts numbers from 0 to 9 based on audio input. The neural network is a
multilayer perceptron with an input of 26 neurons, two hidden layers of 64 and 32 neurons, and a
final output layer of 10 neurons. It uses the Adam optimizer and the cross-entropy loss function.

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

Additionally, there is a Docker image available to test the gradio app (around 7GB):

```bash
# Download the image
docker pull alvarocrespo02/sdoml-demo

# Run the app
docker run -p 7860:7860 alvarocrespo02/sdoml-demo
```

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