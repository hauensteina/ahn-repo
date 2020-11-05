
from pdb import set_trace as BP
from PIL import Image
import numpy as np

resolution = 300 # DPI
cols, rows = ( int(8.5 * resolution), int(11.0 * resolution)) # letter

grid_spacing = 60

pixels = np.zeros((rows, cols, 3), dtype=np.uint8)
pixels[:,:] = (255,255,255)

lc = 50
line_color = (lc,lc,lc)

for row in range( 0, rows, grid_spacing):
    pixels[row, :] = line_color

for col in range( 0, cols, grid_spacing):
    pixels[:, col] = line_color


img = Image.fromarray(pixels)
img.save( 'grid.pdf', resolution=resolution)
