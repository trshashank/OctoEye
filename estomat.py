import loris
import numpy as np
import matplotlib.pyplot as plt 
import os
import fnmatch
from tqdm import tqdm
import scipy.io as sio

wavelength = 600
FILENAME = f"2025-02-21T10-49-52Z_dvs_without_hot_pixels_crop"
path = f"/media/samiarja/USB/gen4-windows/recordings/{wavelength}/"
myPath = path + FILENAME + ".es"
events_stream = []
my_file = loris.read_file(myPath)
events = my_file['events']
for idx in tqdm(range(len(events))):
    event = -1*np.ones(4,dtype=np.uint16)
    event[0] = events[idx][0]
    event[1] = events[idx][1]
    event[2] = events[idx][2]
    event[3] = events[idx][3]
    events_stream.append(event)

sio.savemat(path + FILENAME + ".mat",{'events':np.asarray(events_stream)})
print(FILENAME+" file's saved")
