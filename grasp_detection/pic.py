from PIL import Image
import numpy as np

# Create a Numpy array
array = np.zeros([100, 200, 3], dtype=np.uint8)
array[:,:100] = [255, 128, 0] # Orange left side
array[:,100:] = [0, 0, 255]   # Blue right side

# Convert array to image
img = Image.fromarray(array)
img.save('output.png')
