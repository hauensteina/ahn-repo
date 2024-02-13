import os,argparse
import hashlib

import uuid
import numpy as np
from PIL import Image
import random


def usage():
    name = os.path.basename( __file__)
    msg = f'''
    Description:
      {name}:  Generate random icons to use instead of avatars or pictures.

    Synopsis:
      {name} --mode [random|sierpinsky|mandelbrot|uuid] --size <n> --output <fname> --iterations <n>

    Example:
      {name} --mode random --size 16 --output icon

    Output goes to icon.png

''' 
    msg += '\n '
    return msg

def main():
    parser = argparse.ArgumentParser( usage=usage())
    parser.add_argument('--mode', required=True)
    parser.add_argument('--size', default=16, help='Size of the image')
    parser.add_argument('--outfname', default='gen_sigil_out', help='Output file name')
    parser.add_argument('--iterations', default=2, help='Number of iterations')
    args = parser.parse_args()
    
    if args.mode == 'random':
        random_icon(args.outfname, args.size)
    elif args.mode == 'sierpinsky':
        sierpinsky_icon(args.outfname, int(args.iterations))
    elif args.mode == 'mandelbrot':
        mandelbrot_icon(args.outfname, int(args.size), int(args.iterations))
    elif args.mode == 'uuid':
        uuid_icon(args.outfname, int(args.size))
    elif args.mode == 'sample':
        for i in range(0, 10):
            uuid_icon(f"{args.outfname}_{i}", int(args.size))
    else:
        print(f"Unknown mode {args.mode}")
        exit(1)
        
def random_icon(outfname, size):
    """ Generate a random 16x16 image using a 16-color palette. """
    palette = [
        (0, 0, 0),       # Black
        (255, 255, 255), # White
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (0, 255, 255),   # Cyan
        (255, 0, 255),   # Magenta
        (192, 192, 192), # Light Gray
        (128, 128, 128), # Gray
        (128, 0, 0),     # Maroon
        (128, 128, 0),   # Olive
        (0, 128, 0),     # Dark Green
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Teal
        (0, 0, 128),     # Navy
    ]
    

    # Generate a random 16x16 image using the 16-color palette
    image_data = np.random.choice(len(palette), (size,size))

    # Convert palette indices to RGB values
    rgb_image = np.array([palette[index] for index in image_data.ravel()])  # Flatten, map, and reshape
    rgb_image = rgb_image.reshape((16, 16, 3))

    # Create and save the image
    image = Image.fromarray(np.uint8(rgb_image), 'RGB')
    image.save(f'{outfname}.png')

def sierpinsky_icon(outfname, iterations):
    """ Generate a Sierpinski carpet icon. """
    palette = [
        (255, 255, 255),  # White (for 0)
        (0, 0, 0),        # Black (for 1)
        # Add more colors to expand the palette
    ]
    size = 3 ** iterations  
    grid = np.ones((size, size))

    def create_carpet(x, y, size):
        if size == 1:
            return
        print(f"Size {size} at {x},{y}")
        new_size = size // 3
        grid[y+new_size:y+2*new_size, x+new_size:x+2*new_size] = 0  # Middle square

        for dy in range(0, size, new_size):
            for dx in range(0, size, new_size):
                if not (dx == new_size and dy == new_size):  # Skip the middle square
                    create_carpet(x+dx, y+dy, new_size)


    def map_to_palette(grid, palette):
        # Map the binary grid to a color palette
        rgb_image = np.array([palette[int(value)] for value in grid.ravel()])
        rgb_image = rgb_image.reshape((grid.shape[0], grid.shape[1], 3))
        return rgb_image

    create_carpet(0, 0, size)

    # Map the grid to the palette
    rgb_image = map_to_palette(grid, palette)

    # Create and save the image
    image = Image.fromarray(np.uint8(rgb_image), 'RGB')
    image.save(f'{outfname}.png')

def mandelbrot_icon(outfname, size, iterations):
    """ Generate a Mandelbrot set icon. """
    def mandelbrot(c, iterations):
        z = 0
        n = 0
        while abs(z) <= 2 and n < iterations:
            z = z*z + c
            n += 1
        return n

    def mandelbrot_set(xmin, xmax, ymin, ymax, imgx, imgy, iterations, cx=0.0, cy=0.0, zoom=1):
        ix, iy = np.meshgrid(np.linspace(xmin, xmax, imgx), np.linspace(ymin, ymax, imgy))
        c = (ix + cx) / zoom + 1j * (iy + cy) / zoom
        mandel = np.frompyfunc(mandelbrot, 2, 1)
        iters = mandel(c, iterations).astype(np.float64)
        return iters / iterations

    def map_to_palette(grid, palette):
        # Normalize the grid to fit the palette size
        norm_grid = np.floor(grid * (len(palette) - 1))
        # Map the normalized grid to the palette
        rgb_image = np.array([palette[int(value)] for value in norm_grid.ravel()])
        rgb_image = rgb_image.reshape((grid.shape[0], grid.shape[1], 3))
        return rgb_image

    # Random parameters for variety
    cx, cy = random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)  # Center point
    zoom = random.uniform(0.5, 2)  # Zoom level

    # 16-color palette
    palette = [
        (66, 30, 15), (25, 7, 26), (9, 1, 47), (4, 4, 73),
        (0, 7, 100), (12, 44, 138), (24, 82, 177), (57, 125, 209),
        (134, 181, 229), (211, 236, 248), (241, 233, 191), (248, 201, 95),
        (255, 170, 0), (204, 128, 0), (153, 87, 0), (106, 52, 3)
    ]

    # Generate the Mandelbrot set
    mandelbrot_grid = mandelbrot_set(-2.0, 1.0, -1.5, 1.5, size, size, iterations, cx, cy, zoom)

    # Map the Mandelbrot set to the palette
    rgb_image = map_to_palette(mandelbrot_grid, palette)

    # Create and save the image
    image = Image.fromarray(np.uint8(rgb_image), 'RGB')
    image.save(f'{outfname}.png')
    
def uuid_icon(outfname, size):
    uuidstr = str(uuid.uuid4()).replace('-','')
    grid = grid_from_uuid(uuidstr, size)
    # Create and save the image
    image = Image.fromarray(np.uint8(grid), 'RGB')
    image.save(f'{outfname}.png')
    
def grid_from_uuid(uuidstr, size):
    """ 
    Generate multicolor sigil from a uuid4 string. 
    The uuid is hashed to get a random distribution of individual bits.
    Backround is white. Foreground is two different colors from a palette of 16 colors.
    The first two digits of the hash are used to select color1 (color for upper rows).
    The second two digits of the hash are used to select color2 (color for lower rows).
    The bits of the hash are traversed left to right to determine whether color gets applied (bit is 1) or
    the white background remains visible (bit is 0).
    The right half of the image is a mirror of the left half.
    If there is too much background or too many rows with no color, we start bit traversal at a different offset
    until we get a reasonable image. 
    The process is deterministic and will always give the same image for the same uuid and size.    
    """
    
    palette = [
        (0xff, 0xff, 0xff), 
        (0x00, 0x6c, 0x84), 
        (0x95,0x1c,0x42),
        (48,52,109),
        (0x9d,0xe0,0xad),
        (133,76,48),
        (52,101,36),
        (208,70,72),
        (0xd3,0x80,0xb9),
        (89,125,206),
        (210,125,44),
        (133,149,161),
        (109,170,44),
        (210,170,153),
        (109,194,202),
        (218,212,94),
        (0x72,0x4d,0xbd)
    ]

    palettegrid = np.zeros((len(palette),2,3))
    for i in range(0, len(palette)):
        palettegrid[i] = palette[i]
    image = Image.fromarray(np.uint8(palettegrid), 'RGB')
    image.save(f'gen_sigil_palette.png')

    print(f"UUID: {uuidstr}")
    hashstr = hashlib.sha256(uuidstr.encode('utf-8')).hexdigest()
    nbits = len(hashstr) * 4
    hashnum = int(hashstr, 16)
    idx0 = 0
    idx1 = int(hashstr[0:2],16) % len(palette)
    if idx1 == idx0: idx1 = (idx1 + 1) % len(palette)
    idx2 = int(hashstr[2:4],16) % len(palette)
    while idx2 == idx0 or idx2 == idx1:
        idx2 = (idx2 + 1) % len(palette) 
    bgcol = palette[ idx0 ]
    color1 = palette[ idx1 ]
    color2 = palette[ idx2 ]
    
    offset = 0
    halfsize = size // 2 + 1
    for _ in range(0, 20): # dont loop forever
        bitsum = 0
        n_zero_rows = 0
        grid = np.zeros((size, size, 3))
        for row in range(0, size):
            rowsum = 0
            if row < size // 2:
                color = color1
            else:
                color = color2
            for col in range(halfsize):
            #for col in range(0,size):
                idx = (row * size + col)
                val = (hashnum >> ((idx+offset) % nbits)) & 1
                if val == 1:
                    bitsum += 1
                    rowsum += 1
                    grid[row, col] = color
                    grid[row,size - 1 - col] = color # mirror left/right
                else:
                    grid[row, col] = bgcol
                    grid[row,size - 1 - col] = bgcol
            if rowsum == 0:
                n_zero_rows += 1        
        # Terminate if the image looks reasonable        
        if bitsum > size * halfsize * 0.3 and n_zero_rows < 2: 
            break
        else:
            print(f"Weird image, retrying")
            offset += 1

    return grid
    
main()