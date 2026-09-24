# Score-Based Diffusion Generative Model

The model uses a variance-preserving SDE (VP-SDE) to slowly turn the target distribution into Gaussian noise, and its time-reversal to run that process backwards — from noise back to the target distribution. To run the reverse process, we need a directional field that points back toward the data at every noise level; this is called the score. Since the score isn't known in closed form, we train a neural network to learn it. For this small project, we employ a toy dataset called "Two Moons."

## Contents

- `src/sde.py` — the noise schedule (how the noise grows over time) and the forward process from the VP-SDE
- `src/toy_data.py` — generates the toy "two moons" datasets
- `src/layers.py`, `src/network.py` — a small neural network, written from scratch
- `src/loss.py`, `src/optimizer.py` — the training loss and optimizers (SGD, Adam)
- `src/train.py` — trains a model
- `src/sample.py` — generates new data by running the learned process in reverse
- `tests/evaluation.py` — evaluates how close the learned score is to the true score, and how close generated samples are to real data
- `notebooks/note.ipynb` — the write-up: the math, the experiments, and the plots

## Setup

Take a look at `notebooks/note.ipynb` to see how it works: the noise schedules, the training runs, and the generated samples.

To install: 
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

To train a model:
```bash
python -m src.train
```

To generate a sample from a trained model:
```bash
python -m src.sample
```
