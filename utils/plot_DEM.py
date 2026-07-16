import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

# 1. Load the DEM
Z = np.load('generated_data/DEM/Maranhão/DEM.npy')

# 3. Create X and Y coordinate grids based on image dimensions
shape = Z.shape
X = np.arange(0, shape[1])
Y = np.arange(0, shape[0])
X, Y = np.meshgrid(X, Y)

# 4. Set up the 3D plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(12, 8))

# 5. Plot the surface
# We use the 'terrain' or 'viridis' colormap, shaded by the Z (altitude) values
surf = ax.plot_surface(X, Y, Z, cmap=cm.terrain,
                       linewidth=0, antialiased=True)

# 6. Add a color bar to show the altitude scale
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Altitude')

# 7. Adjust view angle and labels
ax.set_title("3D Terrain Elevation Map")
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Elevation")

# Tweak the viewing angle (elevation, azimuth) for a better perspective
ax.view_init(elev=45, azim=-45)

plt.show()