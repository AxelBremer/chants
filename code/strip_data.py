import pandas as pd
from pycantus import Cantus
from os import getcwd
from os import path
import json
from tqdm import tqdm

ids = []
modes = []
vps = []

def get_id_mode_volp(row):
    if pd.notna(row['volpiano']):
        ids.append(row.name)
        vps.append(row['volpiano'])
        modes.append(row['mode'])

tqdm.pandas()

Cantus.load()
# chant = Cantus.get_chant('97603bf6175683c41f278f3891fb002cc6152379c0d6699afdbe0a48fe0331e0')
df = Cantus.chants
df.progress_apply(get_id_mode_volp, axis=1)

directory = path.join(getcwd(), 'data')

d = {'ids':ids, 'modes':modes, 'vps':vps}

with open(directory+'/stripped_data.json', 'w') as fp:
    json.dump(d, fp)