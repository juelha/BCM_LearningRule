import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

import numpy as np

def draw_weights(weights, epoch):
    #if fig is None:
    fig = plt.figure(figsize=(12.9,2))
    pxl_x = pxl_y = int((weights.shape[1])**(1/2))
    n_units, _ = weights.shape

    all_weights = np.zeros((pxl_y,pxl_x*n_units))

    # iterate over units
    for unit in range(n_units):
        all_weights[:,unit*pxl_y:(unit+1)*pxl_y] = weights[unit,:].reshape(pxl_y,pxl_x)

    # color bar
    abs_max = np.amax(np.absolute(all_weights))
    im = plt.imshow(all_weights, cmap='bwr', vmin=-abs_max, vmax=abs_max)
    fig.colorbar(im, ticks=[np.amin(all_weights), 0, np.amax(all_weights)])

    # fig costumization 
   # plt.axis('off')
    plt.title(f"Weights at epoch: {epoch+1}", pad=20)
    # set x-ticks at center of each unit's image
    centers = [unit * pxl_x + pxl_x / 2 for unit in range(n_units)]
    labels = [f"Neuron {unit + 1}" for unit in range(n_units)]
    plt.xticks(centers, labels, fontsize=8, rotation=0)
    plt.yticks([])  # Hide the Y-axis

    # show 
    fig.canvas.draw()   
    display(fig)
    clear_output(wait=True)
    fig.clear()
    plt.close(fig)


def selectivity_metric(activity):
    print(activity.shape)
    print(torch.mean(activity, dim=1))
    s =  1 - torch.mean(activity, dim=1)/torch.max(activity, dim=1).values # calc selectivity per neuron -> go row wise over batch
    # print(s)
    # print(s.shape)
    # s = torch.mean(s, dim=1) # take avg of selectivity over entire batch
    return s