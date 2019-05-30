import numpy as np

def greeting(name):
	return f'Hello, {name}'

def my_stdev(x):
  mu = x.mean()
  diff = x - mu
  diff_sq = diff**2
  if len(x) == 1:
  	raise ZeroDivisionError
  stdev = np.sqrt(diff_sq.sum()/(len(x)-1))
  return stdev