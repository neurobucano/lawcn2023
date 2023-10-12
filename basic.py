
import os
import pandas as pd
import numpy as np

class SessionHnd:
  def __init__(self, data_path):
    self.data_path = data_path
    self.sessions=pd.read_hdf('%s/ecephys_sessions.h5' % data_path)
    self._set_session_ids()

  def _set_session_ids(self):
    session_ids=[]
    for filename in os.listdir(self.data_path):
      if filename.startswith('1'):
        session_ids.append(int(filename[0:10]))
    session_ids=list(set(session_ids))
    self.session_ids = session_ids

  def get_acronyms_for_session(self, session_id):
    if session_id not in self.session_ids:
        raise KeyError("A session_id '{}' não foi encontrada.".format(session_id))

    acronyms=self.sessions.loc[session_id].structure_acronyms
    acronyms=acronyms.split("'")[1:-1]
    while (', ' in acronyms):
      acronyms.remove(', ')
    acronyms.remove('root') # Remove root
    acronyms = [item for item in acronyms if not item.startswith('DG-')]
    acronyms.append('DG')
    acronyms.sort()
    return (acronyms)

  def get_sessions_for (self, acronym):

    sessions = []
    for session_id in self.session_ids:
      if (self.exists_session_for(session_id, acronym)):
        sessions.append(session_id)
    return (sessions)

  def exists_session_for(self, session_id, acronym):
    filename = '%s/%d-%s-spk.h5' % (self.data_path, session_id, acronym)
    return (os.path.exists(filename))

  def get_spikes (self, session_id, acronym):
    spikes=None

    filename_spk='%s/%s-%s-spk.h5' % (self.data_path,session_id,acronym)
    if not os.path.exists(filename_spk):
        return (spikes)

    spikes = pd.read_hdf(filename_spk, key='spk')
    spikes.name = '%s-%s' % (session_id,acronym)
    spikes=spikes.sort_index()
    return (spikes)

  def get_speed_run (self, session_id):
    filename = '%s/behavior/%d-comp-vel.h5' % (self.data_path,session_id)
    if not os.path.exists(filename):
        raise FileNotFoundError("File not found '{}'.".format(filename))
    v=pd.read_hdf (filename)
    return (v)

  def get_reward (self, session_id):
    filename = '%s/%d-rew.h5' % (self.data_path,session_id)
    reward = pd.read_hdf(filename)
    return (reward)

  def get_stim (self, session_id):
    filename = '%s/%d-stim.h5' % (self.data_path,session_id)
    stim = pd.read_hdf(filename)
    return (stim)

  def get_stim_times (self, stim_name, session_id):
    stim = self.get_stim (session_id)
    times = None
    stim_names = {'natural': 'Nat', 'spont': 'spont', 'gabor': 'gabor', 'flash': 'flash'}
    if (stim_name in stim_names.keys()):
      N=stim.stimulus_name.str.startswith(stim_names[stim_name])
      times=stim[N]['start_time'].dropna().values
    return (times)
  



class NeuroRuler:


  def crop (self, spikes, a, b):
    mask = ((spikes.index>=a) & (spikes.index<=b))
    return (spikes[mask])

  def inst_firing_rate (self, spikes, bin_size, a=0, b=None,duration=0.25):
    ''' Retorna a taxa instantanea de disparos de uma dada população dentro um
    intervalo dado, em uma dada resolução temporal. Retorna a informação
    dentro de um Panda Series.
    '''
    if(len(spikes==0)) and not(duration is None):
      b=a+duration
      bins=np.arange(a,b,bin_size)
      zeros=np.zeros_like(bins[0:-1])
      fr = pd.Series (zeros, index=bins[0:-1])
    else:
      nneurons=len(spikes.unique())
      t=self.crop(spikes, a=a,b=b).index
      bins=np.arange(a,b,bin_size)
      count, bins=np.histogram(t,bins)
      fr = pd.Series (count/(bin_size*nneurons), index=bins[0:-1])
    return (fr)

  def get_max_on_spont(self, session_id, acronym, data_path, bin_size):
    shnd = SessionHnd(data_path=data_path)
    spikes = shnd.get_spikes(session_id, acronym)
    stim = shnd.get_stim (session_id)
    a,b=stim[stim.stimulus_block==3][['start_time','end_time']].values[0]
    ifr=ruler.inst_firing_rate(spikes, bin_size=bin_size, a=a, b=b)
    sorted_ifr=ifr.sort_values(ascending=False)
    t0=sorted_ifr.index[0]
    t1=sorted_ifr.index[0]+bin_size
    return (t0)

  def get_min_on_spont(self, session_id, acronym, data_path, bin_size):
    shnd = SessionHnd(data_path=data_path)
    spikes = shnd.get_spikes(session_id, acronym)
    stim = shnd.get_stim (session_id)
    a,b=stim[stim.stimulus_block==3][['start_time','end_time']].values[0]
    ifr=ruler.inst_firing_rate(spikes, bin_size=bin_size, a=a, b=b)
    sorted_ifr=ifr.sort_values(ascending=True)
    t0=sorted_ifr.index[0]
    t1=sorted_ifr.index[0]+bin_size
    return (t0)

  def calc_psth (self, spikes, t0, d_pre=.1, d_pos = .35, bin_size = 1e-3):
    labels = spikes.unique()
    nneurons = len(labels)
    window_width = d_pre + d_pos
    bins=np.arange(0,window_width,bin_size)
    psth ={}
    for label in labels:
        psth[label] = np.zeros_like (bins[0:-1])
    for T in t0:
      a=T-d_pre
      b=a+window_width
      mask = ((spikes.index>=a) & (spikes.index<=b))
      spk = spikes[mask]
      for label in labels:
        mask = (spk==label)
        t=spk[mask].index - a
        count,b = np.histogram(t,bins)
        psth[label]+=count

    return (pd.DataFrame(psth, index=bins[0:-1])/(window_width*nneurons))


class NeuroView:

  def _get_neuron_spikes (self, spikes, neuron_id, a, b):
    spk_neuron=spikes[spikes==neuron_id]
    X=np.array(spk_neuron.loc[a:b].index)
    return (X)

  def raster_plot (self, spikes, ax=None, a=0, b=None, color='black', bar_duration = 1, bar_width = 2, y0=None, order=None, filename=None):

    if (ax is None):
      ax=plt.subplot (1,1,1)

    if b is None:
      b=spikes.index[-1]

    if order is None:
      neuron_ids=spikes.unique()
    else:
      neuron_ids = order

    if y0 is None:
      y=len(neuron_ids)
    else:
      y=y0

    for neuron_id in neuron_ids:
      X=self._get_neuron_spikes (spikes=spikes, neuron_id=neuron_id, a=a, b=b)
      Y=[y]*len(X)
      ax.plot (X,Y,'.', ms=1, color=color)
      y-=1

    # Plot a barra horizontal equivalente a 1 segundo de duração.

    ax.plot ([a,a+bar_duration],[y-bar_width,y-bar_width],'-k', ms=0.2,lw=2)

    #plt.title ('Raster plot de toda população no intervalo [%d,%d]s' % (a,b))
    ax.axis('off')
    if not (filename is None):
      plt.savefig(filename)

