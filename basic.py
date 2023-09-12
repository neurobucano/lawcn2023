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

    acronyms=[]
    filenames=os.listdir(self.data_path)
    for filename in filenames:
      parts = filename.split('-')
      if (parts[0].isnumeric()):
        if int(parts[0])==session_id  and parts[-1]=='spk.h5':
          acronyms.append(parts[1])

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