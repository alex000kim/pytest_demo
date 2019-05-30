import pytest
import numpy as np
from my_funcs import greeting, my_stdev

@pytest.mark.parametrize("name", ["Jane", "John"])
def test_greeting(name):
    assert greeting(name) == f"Hello, {name}"

def test_my_stdev():
    x = np.random.normal(size=1000)
    assert my_stdev(x) == np.std(x, ddof=1)


def add_three(x,y,z):
    return x + y + z
