import numpy as np

def LatinHypercubeSample(df):
    
    np.random.seed(13)
    
    wv = df['wv range']
    elev = df['elev med']
    sza = df['zen']
    
    bins = 11
    # building intervals
    wv_range = np.linspace(np.min(wv),np.max(wv), bins)
    e_range = np.linspace(np.min(elev),np.max(elev), bins)
    sza_range = np.linspace(np.min(sza),np.max(sza), bins)
    
    
    samples_per_space = 1
    print(f'num samples to build: {(bins-1)**3*samples_per_space}')
    
    sample_set = []
    for _w in range(len(wv_range) - 1):
        for _e in range(len(e_range) - 1):
            for _sza in range(len(sza_range) - 1):
                subset = (wv > wv_range[_w]) & (wv <= wv_range[_w + 1]) & (elev > e_range[_e]) & (elev <= e_range[_e + 1]) & \
                (sza > sza_range[_sza]) & (sza <= sza_range[_sza + 1])
                perm = np.random.permutation(np.sum(subset))
    
                samples = np.where(subset)[0][perm[:samples_per_space]]
                sample_set.extend(samples.tolist())

    return sample_set

def LatinHypercubeSample2(df):
    
    np.random.seed(13)
    
    wv = df['wv range']
    sza = df['zen']
    
    bins = 35
    # building intervals
    wv_range = np.linspace(np.min(wv),np.max(wv), bins)
    sza_range = np.linspace(np.min(sza),np.max(sza), bins)
    
    
    samples_per_space = 1
    print(f'num samples to build: {(bins-1)**2*samples_per_space}')
    
    sample_set = []
    for _w in range(len(wv_range) - 1):
        for _sza in range(len(sza_range) - 1):
            subset = (wv > wv_range[_w]) & (wv <= wv_range[_w + 1]) & \
            (sza > sza_range[_sza]) & (sza <= sza_range[_sza + 1])
            perm = np.random.permutation(np.sum(subset))
    
            samples = np.where(subset)[0][perm[:samples_per_space]]
            sample_set.extend(samples.tolist())

    return sample_set

def LatinHypercubeSample3(df):
    
    np.random.seed(13)
    
    elev = df['elev med']
    sza = df['zen']
    
    bins = 31
    # building intervals
    e_range = np.linspace(np.min(elev),np.max(elev), bins)
    sza_range = np.linspace(np.min(sza),np.max(sza), bins)
    
    
    samples_per_space = 1
    print(f'num samples to build: {(bins-1)**2*samples_per_space}')
    
    sample_set = []
    for _e in range(len(e_range) - 1):
        for _sza in range(len(sza_range) - 1):
            subset = (elev > e_range[_e]) & (elev <= e_range[_e + 1]) & (sza > sza_range[_sza]) & (sza <= sza_range[_sza + 1])
            perm = np.random.permutation(np.sum(subset))
    
            samples = np.where(subset)[0][perm[:samples_per_space]]
            sample_set.extend(samples.tolist())

    return sample_set