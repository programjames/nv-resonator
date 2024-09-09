import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_mode_fast(filename, mode_number):
    with h5py.File(filename, 'r') as f:
        # Read mesh data
        coordinates = f['coordinates'][:]
        topology = f['topology'][:]
        
        # Read mode data
        mode_group = f[f'modes/mode_{mode_number}']
        mode_data = mode_group['data'][:]
        frequency = mode_group.attrs['frequency']
    
    # Get unique z and r coordinates
    z = np.unique(coordinates[:, 0])
    r = np.unique(coordinates[:, 1])
    
    # Reshape mode data into a 2D grid
    Z, R = np.meshgrid(z, r)
    mode_grid = np.zeros_like(Z)
    for i, (zi, ri) in enumerate(coordinates[:, :2]):
        iz = np.argmin(np.abs(z - zi))
        ir = np.argmin(np.abs(r - ri))
        mode_grid[ir, iz] = mode_data[i]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(Z, R, mode_grid, cmap='RdBu_r', shading='auto')
    ax.set_aspect('equal')
    ax.set_xlabel('z (m)')
    ax.set_ylabel('r (m)')
    ax.set_title(f'Mode {mode_number}: {frequency/1e9:.4f} GHz')
    fig.colorbar(mesh, label='Mode amplitude')
    
    plt.tight_layout()
    plt.show()

# Example usage
visualize_mode_fast('resonator_modes.h5', 8)  # Visualize the first mode