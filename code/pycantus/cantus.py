import pandas as pd
import os.path as path
from os import getcwd
import glob

from .chant import Chant

class CANTUS(object):
    def __init__(self, directory=None):
        if directory is None:
            cur_dirname = path.dirname(path.realpath(__file__))
            self.directory = path.join(cur_dirname, 'data')
        else:
            self.directory = directory

        self.status = 'no data loaded'

    def __repr__(self):
        return f'Cantus({self.status})'
    
    def load(self, what='all'):
        self.sources = self.load_dataset('sources')
        self.centuries = self.load_dataset('centuries')
        self.feasts = self.load_dataset('feasts')
        self.genres = self.load_dataset('genres')
        self.indexers = self.load_dataset('indexers')
        self.notations = self.load_dataset('notations')
        self.offices = self.load_dataset('offices')
        self.provenances = self.load_dataset('provenances')
        self.siglum = self.load_dataset('siglum')
        
        if what is 'all':
            self.chants = self.load_chants_from_chunks()
            self.status = 'all data loaded'
        elif what is 'demo':
            self.chants = self.load_dataset('chants-demo')
            self.status = 'demo data loaded'
        
    def load_dataset(self, name):
        resource = path.join(self.directory, f'{name}.csv')
        df = pd.read_csv(resource, index_col='id')
        return df

    def load_chants_from_chunks(self, file_pattern='chants-chunk-*.csv'):
        """Load chants from multiple CSV files."""
        files = glob.glob(path.join(self.directory, file_pattern))
        dataframes = []
        for file in files:
            df = pd.read_csv(file, low_memory=False)
            dataframes.append(df)

        chants = pd.concat(dataframes)
        chants.set_index('id', inplace=True, verify_integrity=True)
        return chants


    def get_chant(self, chant_id):
        return Chant(chant_id, self)

# Instantiate
directory = path.join(getcwd(), 'data')
Cantus = CANTUS(directory)