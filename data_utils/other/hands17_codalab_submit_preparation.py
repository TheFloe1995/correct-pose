"""
Prepare predictions for submission to the HANDS 2017 challenge leader board at codalab.
"""

import os
import torch
import numpy as np

# Config
prediction_file = 'results/predictions/HANDS17_V2V_uncorrected_test.pt'
########################################################################

predictions = torch.load(prediction_file, map_location='cpu').detach().numpy()
names = np.load(os.path.join('data', 'predictor_results', 'HANDS17', 'testset_names.npy'))

file_name = os.path.splitext(os.path.split(prediction_file)[-1])[0]
txt_file_path = os.path.join('results', 'predictions', file_name + '.txt')

print('Writing txt file...')
with open(txt_file_path, 'w') as f:
    for i, (name, pred) in enumerate(zip(names, predictions.reshape(-1, 63))):
        line = '{}\t{}\n'.format(name, '\t'.join(['{:.4f}'.format(v) for v in pred]))
        f.write(line)
        if i % 10000 == 0:
            print(i)
