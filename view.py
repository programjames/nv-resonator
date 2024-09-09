import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

def visualize_mode(filename, mode_number):
    with h5py.File(filename, 'r') as f:
        # Read mesh data
        coordinates = f['coordinates'][:]
        topology = f['topology'][:]
        
        # Read mode data
        mode_group = f[f'modes/mode_{mode_number}']
        mode_data = mode_group['data'][:]
        frequency = mode_group.attrs['frequency']
        
    # Create triangulation
    print(coordinates.shape)
    print(topology.shape)
    tri = Triangulation(coordinates[:, 0], coordinates[:, 1], topology.reshape(-1, 3))
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    contour = ax.tricontourf(tri, mode_data, cmap='RdBu_r', levels=20)
    ax.set_aspect('equal')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('r (m)')
    ax.set_title(f'Mode {mode_number}: {frequency/1e9:.4f} GHz')
    fig.colorbar(contour, label='Mode amplitude')
    
    plt.tight_layout()
    plt.show()

# Example usage
visualize_mode('resonator_modes.h5', 1)  # Visualize the first mode