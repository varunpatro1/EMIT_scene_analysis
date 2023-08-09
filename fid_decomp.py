import numpy as np

def decompose(fid_column):
    date_col = []
    time_col = []
    orbit_col = []
    scene_col = []
    for fid in fid_column:
        date = fid[4:12]
        time = fid[13:19]
        orbit = fid[21:26]
        scene = fid[28:31]
        date_col.append(date)
        time_col.append(time)
        orbit_col.append(orbit)
        scene_col.append(scene)
    return date_col, time_col, orbit_col, scene_col

def recompose(date_col, time_col, orbit_col, scene_col):
    fid_col = []
    for i in range(len(date_col)):
        fid = 'emit' + date_col[i] + 't' + time_col[i] + '_o' + orbit_col[i] + '_s' + scene_col[i]
        fid_col.append(fid)
    return fid_col
