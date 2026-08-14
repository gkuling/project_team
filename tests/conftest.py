'''
Shared fixtures for the project_team test suite. Everything here is
synthetic and CPU-only: no downloads, no MNIST, no network.
'''
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import matplotlib
matplotlib.use('Agg')


@pytest.fixture(autouse=True)
def _seeded():
    np.random.seed(0)
    torch.manual_seed(0)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope='session')
def tiny_image_dataset(tmp_path_factory):
    '''
    24 deterministic 8x8 grayscale PNGs on disk plus a DataFrame with
    img_data (path), label (3 balanced classes) and patient (12 patients x 2
    images) columns. A stand-in for MNIST that needs no download.
    '''
    folder = tmp_path_factory.mktemp('tiny_images')
    rows = []
    rng = np.random.default_rng(7)
    for i in range(24):
        label = i % 3
        # class-dependent brightness so a tiny model can actually learn
        img = (rng.integers(0, 60, size=(8, 8)) + 60 * label
               ).astype(np.uint8)
        path = str(folder / f'img_{i}.png')
        Image.fromarray(img, mode='L').save(path)
        rows.append({'img_data': path,
                     'label': label,
                     'patient': 'patient_' + str(i // 2)})
    return pd.DataFrame(rows)


@pytest.fixture()
def tiny_df():
    '''30-row tabular DataFrame for io-manager split tests.'''
    return pd.DataFrame({
        'feat': np.arange(30, dtype=float),
        'target': [i % 2 for i in range(30)],
        'group': ['g' + str(i) for i in range(30)],
        'strat': [i % 3 for i in range(30)],
    })
