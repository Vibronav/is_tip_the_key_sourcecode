from contextlib import contextmanager
import numpy as np
import torch

@contextmanager
def random_state_context_torch(seed=None):
    state = torch.get_rng_state()
    if seed is None:
        torch.seed()
    else:
        torch.manual_seed(seed)

    yield
    torch.set_rng_state(state)

@contextmanager
def random_state_context_numpy(seed=None):
    state = np.random.get_state()
    np.random.seed(seed)
    
    yield
    np.random.set_state(state)