import numpy as np
import pickle

# Load your SMPL-H model from the .npz file
npz_file = '/move/u/takaraet/UniversalHumanoidControl/data/smpl/SMPLH_FEMALE.npz'
import pickle

# Creating a dictionary from the npz file
data = {key: npz_file[key] for key in npz_file_keys}

# Pickle file path
pickle_file_path = '/mnt/data/SMPLH_FEMALE.pkl'

# Saving as a pickle file
with open(pickle_file_path, 'wb') as f:
    pickle.dump(data, f)

pickle_file_path