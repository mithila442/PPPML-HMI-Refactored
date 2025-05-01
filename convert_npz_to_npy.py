# ~/Downloads/PPPML-HMI/convert_npz_to_npy.py
import os
import numpy as np

npz_folder = 'raw_npz'
output_folder = 'my_rescaled'

os.makedirs(output_folder, exist_ok=True)

for fname in os.listdir(npz_folder):
    if fname.endswith('.npz'):
        npz_path = os.path.join(npz_folder, fname)
        data = np.load(npz_path)['ct']
        npy_path = os.path.join(output_folder, fname.replace('.npz', '.npy'))
        np.save(npy_path, data)
        print(f"Saved {npy_path}")
