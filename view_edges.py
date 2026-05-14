import numpy as np
from PIL import Image

# 1. Read the raw 256x256 binary file
raw_data = np.fromfile("test_output.raw", dtype=np.uint8)

# 2. Reshape it back into a 2D image grid
img_array = raw_data.reshape((256, 256))

# 3. Save it as a standard PNG
img = Image.fromarray(img_array)
img.save("edges.png")
print("Success: Converted raw output to edges.png!")
