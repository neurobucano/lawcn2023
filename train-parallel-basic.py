import os, sys, h5py, time, multiprocessing
import numpy as np
import pandas as pd
import psutil
#import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from itertools import product
from functools import partial
import warnings
import logging

#warnings.simplefilter(action='ignore', category=FutureWarning)


class Training():

    def __init__(self, session_ids, area, model, data_path, logger):
        self.session_ids = session_ids
        self.areas = area
        self.data_path = data_path
        self.logger = logger
        self.stimulus_presentations = pd.read_hdf(f"{self.data_path}/{self.session_ids}-stim.h5", key='stim')
        self.model = model
        self.num_trials = 20

        
        
    def get_neuronal_pattern (self, t0, spikes, binSize=0.01, windowDur=0.25):
        t1=t0+windowDur
        unit_list=list(spikes.unique())
        nneurons=len(unit_list)

        nbins = int(np.floor(windowDur/binSize))
        pattern = np.zeros((nneurons,nbins),dtype=np.int8)
        bins = np.arange(0,windowDur+binSize,binSize)
        i=0
        for unit_id in unit_list:
            #self.logger.info ('get pattern %d ' % i)
            mask = (spikes==unit_id)
            s= spikes[mask].index.astype(float)
            spk=s[(s>=t0) & (s<t1)]
            pattern[i]=np.histogram(spk-t0, bins)[0]
            i+=1

        return (pattern)
  

    
    def get_spikes(self):
        file_path = f"{self.data_path}/{self.session_ids}-{self.areas}-spk.h5"

        if not os.path.exists(file_path):
            return None, None

        spikes = pd.read_hdf(file_path, key='spk')
        unit_list=list(set(spikes.unique()))

        return unit_list, spikes

    def get_stimulus(self, ):
        t_positive = self.stimulus_presentations.query("stimulus_name=='Natural_Images_Lum_Matched_set_ophys_G_2019'")['start_time'].values
        estimulo = 'Natural-G x Gabors'
        if len(t_positive) == 0:
            t_positive = self.stimulus_presentations.query("stimulus_name=='Natural_Images_Lum_Matched_set_ophys_H_2019'")['start_time'].values
            estimulo = 'Natural-H x Gabors'
        t_negative = self.stimulus_presentations.query("stimulus_name=='gabor_20_deg_250ms'")['start_time'].values
        n_positive_max = min(len(t_positive), len(t_negative),50)

        return t_positive, t_negative, estimulo, n_positive_max

    def train_model(self, X, y, n_positive):

        auc_scores = []
        num_folds = 10

        
        skf = StratifiedKFold(n_splits=num_folds)
        model = self.model
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]

            fold_auc = roc_auc_score(y_test, y_pred)
            auc_scores.append(fold_auc)

        return np.mean(auc_scores)

    def run_single_analysis (self, result_folder, ntrials=20):
        print(self.session_ids, self.areas)
        unit_list, spikes = self.get_spikes()
        nneurons = len(unit_list)

        if unit_list is None or spikes is None:
            return None

        t_positive, t_negative, estimulo, n_positive_max = self.get_stimulus()

        n_samples = min([len(t_positive),len(t_negative),250])

        results = []
        df = pd.DataFrame()
        n=len(unit_list)
        for trial in range(0,ntrials):
            start = time.time()
            self.logger.info ('%s-%s: n=%d starting trial=%d \t %s' %  (self.session_ids, self.areas, n, trial, type(self.model)))

            # Seleciona n_samples de tempos positivos e tempos negativos para compor a base
            selected_positive = np.random.permutation(t_positive)[0:n_samples]
            selected_negative = np.random.permutation(t_negative)[0:n_samples]            

            patterns_positive =[]
            patterns_negative = []
            selected_neurons = np.random.permutation(unit_list)[0:n]

            spk= spikes[spikes.isin(selected_neurons)]
            for t0 in selected_positive:
                print (t0)
                patterns_positive.append(self.get_neuronal_pattern(t0=t0, spikes=spk))
            for t0 in selected_negative:
                patterns_negative.append(self.get_neuronal_pattern(t0=t0, spikes=spk))


            X = np.concatenate([patterns_positive, patterns_negative])
            X = np.array([pattern.flatten() for pattern in X])
            y = np.array(np.concatenate([np.ones(len(patterns_positive)), np.zeros(len(patterns_negative))]))

            trial_mean_auc = self.train_model(X, y, n)
            end = time.time()
            trial_results = {
                'sessionid': self.session_ids,
                'stim': estimulo,
                'area': self.areas,
                'AUC': trial_mean_auc,
                'model': type(self.model),
                'cpuid': psutil.Process(os.getpid()).cpu_num(),
                'duration': (end - start)
            }
            results.append(trial_results)
            self.logger.info ('%s-%s: n=%d done trial=%d (AC=%2.2f)' %  (self.session_ids, self.areas, n, trial,trial_mean_auc))
            if (len(results)>0): 
                current = pd.DataFrame(results)
                results = []
                result_filename='%s/%s-%s-res-basic.h5' % (result_folder,self.session_ids,self.areas)
                if os.path.exists(result_filename):
                    tmp = pd.read_hdf(result_filename, key='basic')
                    df = pd.concat([tmp, current])
                else:
                    df = current
                df.to_hdf(result_filename, key='basic')
        return None


class Informations():
    def __init__(self, data_path):
        self.data_path = data_path
        self.model = DecisionTreeClassifier()

    def list_ids(self):
        
        files_list = os.listdir(data_path)

        id_list = []
        for filename in files_list:
            if filename.endswith('-spk.h5'):
                id_list.append(filename)
        return id_list

    def infos(self, areas):
        infos = []
        for area in areas:
            area = str(area)
            for session_id in self.list_ids():
                session_id = int(session_id)
                if os.path.exists(os.path.join(self.data_path, f"{session_id}-{area}-spk.h5")):
                    info = {
                        'session_id': session_id,
                        'area': area,
                        'model': type(self.model),
                        'data_path': self.data_path
                    }
                    infos.append(info)
                else:
                    continue
        return infos

def analyze_session(info):
    session_id = info['session_id']
    area = info['area']
    model = info['model']
    data_path = info['data_path']

    training = Training(session_id, area, model, data_path)
    session_results = training.run_analysis()
    return session_results

def analyze_file_single(filename, result_folder, logger, model_name='DecisionTreeClassifier', data_path='../data/allen-icvb'):
    

    models = {'DecisionTreeClassifier': DecisionTreeClassifier(), 
                'GaussianNB': GaussianNB(), 
                'GaussianProcess': GaussianProcessClassifier(1.0 * RBF(1.0)),
                'SVMLinear': svm.SVC(kernel='linear')}
    
    model_names = list(models.keys())
    if not (model_name in model_names):
        raise KeyError("Invalid model name: '{}'.".format(model_name))

    model=models[model_name]


    area = filename.split('-')[-2]
    session_id=filename.split('-')[-3].split('/')[-1]
    logger.info('Starting training session ...');
    training = Training(session_id, area, model, data_path, logger)
    session_results = training.run_single_analysis(result_folder)
    return session_results



import time
def get_logger(log_filename=None, log_path = '../log'):
    
    if (log_filename is None):
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        log_filename='%s/%s-basic.log' % (log_path,timestr)
        
    else:
        log_filename='%s/%s' % (log_path,log_filename)
     
    # Create a logger
    logger = logging.getLogger('ndcbasic')
    logger.setLevel(logging.DEBUG)

    # Create a file handler
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)

    # Create a console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return (logger)

def get_filenames(data_folder='../data', result_folder='../results', acronyms=None):
    files_list = os.listdir(data_folder)

    item_list=[]
    acronym_condition = True
    endswith_condition = False
    
    conds = [acronym_condition, endswith_condition]
  
    for filename in files_list:
        endswith_condition = filename.endswith('spk.h5')
        if not(acronyms is None) and endswith_condition:  
            acronym = filename.split('-')[-2]
            acronym_condition = (acronym in acronyms)
        conds = [acronym_condition, endswith_condition]
        if not(False in conds):
            item={'session_id':filename.split('-')[-3].split('/')[-1],
                'area':filename.split('-')[-2],
                'filename':filename}
            item_list.append(item)
        
    df_filename=pd.DataFrame(item_list)
    filenames=df_filename.sort_values(by=['area','session_id'])['filename'].values
    return (filenames)

def pre_analyze_file_single(args):
    filename=args['filename']
    data_folder = args['data_folder']
    result_folder = args['result_folder']
    model_name=args['model_name']
    n0=args['n0']

    area = filename.split('-')[-2]
    session_id=filename.split('-')[-3].split('/')[-1]
    result_filename='%s/%s-%s-res-basic.h5' % (result_folder,session_id,area)
    if (os.path.exists(result_filename)):
        return
    
    timestr = time.strftime("%Y%m%d%H%M%S")
    logger=get_logger(log_filename='%s-%s-%s-basic.log' % (session_id,area,timestr))
    logger.info ('Session id: %s   Area: %s ' % (session_id, area))
    logger.info ('Input spiking acitivty: %s   ' % filename)
    
    logger.info ('Result folder: %s' % result_folder)
    
    logger.info ('Result filename: %s ' % result_filename)
    df=analyze_file_single(filename, result_folder, logger, data_path=data_folder, model_name=model_name)
    logger.info ('[DONE]')
    

def main():
    
    if len(sys.argv) < 3:
        print("Usage: python script.py data_path result_folder model_name")
        return
    
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    data_folder=sys.argv[1]
    result_folder=sys.argv[2]
    model_name = sys.argv[3]
    area_name = sys.argv[4]
    TH= ['LP','MGv','MGm','TH','PIL','SGN','MGd','POL','PoT','LGd','PP','SPFp','Eth','LGv']
    HP= ['CA1','DG','CA3','SUB','HPF','ProS']
    V1=['VISp','VISpm','VISal','VISl','VISrl','VISam']


    
    acronyms={"V1": V1, "HP": HP, "TH": TH}
    if not (area_name in acronyms.keys()):
        raise KeyError("Invalid area name: '{}'.".format(area_name))

    filenames=get_filenames(data_folder, result_folder, acronyms=acronyms[area_name])
    
    args=[]
    for filename in filenames:
        item={'filename':filename,
              'data_folder':  data_folder,
              'result_folder': result_folder,
              'n0': 1,
              'model_name': model_name}
        args.append(item)

    nitems = len(args)
    nprocs = min(len(args),1)
    with multiprocessing.Pool(processes=nprocs) as pool:
        all_results = pool.map(pre_analyze_file_single, args)
    

if __name__ == '__main__':
    
    

    main()
#with multiprocessing.Pool(processes=num_processes) as pool:
#   all_results = pool.map(analyze_file, spk_filenames[23:])
#inal_results = pd.concat(all_results, ignore_index=True)
#inal_results.to_csv(result_path, index=False)