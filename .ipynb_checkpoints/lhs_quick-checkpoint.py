import numpy as np

np.random.seed(13)

wv = np.random.rand(10000) * 5
e = np.random.rand(10000) * 6
sza = np.random.rand(10000) * 60


bins = 10
wv_range = np.linspace(np.min(wv),np.max(wv), bins)
e_range = np.linspace(np.min(e),np.max(e), bins)
sza_range = np.linspace(np.min(sza),np.max(sza), bins)