import argparse
import os
from pathlib import Path
import urllib
import warnings
import numpy as np
from tqdm import tqdm
import octoeye
import loris

parent_path = "/media/samiarja/USB/OctoEye_Dataset/pattern9"
events_list = []

folder_count = len(os.listdir(f"{parent_path}/txt_per_frame"))

for i in tqdm(range(folder_count)):
    reco = i
    
    with open(f"{parent_path}/txt_per_frame/{reco}.txt", 'r') as file:
        for line in file:
            t = int(float(line.split()[0]))
            x = int(line.split()[1])
            y = int(line.split()[2])
            on = bool(int(line.split()[3]))

            events_list.append((t, x, y, on))

    dtype = [('t', '<u8'), ('x', '<u2'), ('y', '<u2'), ('on', '?')]
    events_array = np.array(events_list, dtype=dtype)
    width  = events_array['x'].max() + 1
    height = events_array['y'].max() + 1

    events_array["t"] = (events_array["t"] - events_array["t"][0])


    nEvents = events_array["x"].shape[0]
    x  = events_array["x"].reshape((nEvents, 1))
    y  = events_array["y"].reshape((nEvents, 1))
    p  = events_array["on"].reshape((nEvents, 1))
    ts = events_array["t"].reshape((nEvents, 1))

    events = np.zeros((nEvents,4))
    events = np.concatenate((ts,x, y, p),axis=1).reshape((nEvents,4))
    finalArray = np.asarray(events)
    loris.write_events_to_file(finalArray, f"{parent_path}/event_stream.es", "txyp")
print(f"File: events converted to .es")