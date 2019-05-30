import pytest
import numpy as np
from my_funcs import greeting, my_stdev

def test_greeting():
	name = 'Jane'
	assert greeting(name) == 'Hello, Jane'

@pytest.mark.parametrize("name", ['Jane', 'John'])
def test_greeting_multiple(name):
	assert greeting(name) == f'Hello, {name}'


def test_my_stdev_1():
	x = np.random.normal(size=100)
	assert my_stdev(x) > 0


def test_my_stdev_2():
	x = np.random.normal(size=100)
	assert my_stdev(x) == np.std(x, ddof=1) 

def test_my_stdev_3():
	x = np.random.normal(size=100)
	assert my_stdev(x) > np.std(x, ddof=0)


def test_my_stdev_raises():
	x = np.random.normal(size=1)
	with pytest.raises(ZeroDivisionError):
		my_stdev(x)


@pytest.fixture()
def setup_data():
	x = np.random.normal(size=10000)
	yield x


def test_my_stdev_1_v2(setup_data):
	x = setup_data
	assert my_stdev(x) > 0


def test_my_stdev_2_v2(setup_data):
	x = setup_data
	assert my_stdev(x) == np.std(x, ddof=1) 

def test_my_stdev_3_v2(setup_data):
	x = setup_data
	assert my_stdev(x) > np.std(x, ddof=0)


def test_my_stdev_3_v2(setup_data):
	x = setup_data
	assert my_stdev(x) > np.std(x, ddof=0)



def test_my_stdev_approx(setup_data):
	x = setup_data
	delta = 0.2
	assert my_stdev(x) == pytest.approx(1.0, delta)
	assert 1 - delta < my_stdev(x) < 1 + delta