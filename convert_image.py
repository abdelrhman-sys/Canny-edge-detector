from PIL import Image
import numpy as np

# 1. Open the photo, convert to grayscale ('L'), and resize to 256x256
img = Image.open("test.png").convert("L").resize((256, 256))

# 2. Convert it to a numpy array of 8-bit unsigned integers
img_array = np.array(img, dtype=np.uint8)

# 3. Save it as the raw binary input file
img_array.tofile("test_input.raw")

print("Success: cameraman.png converted to 256x256 test_input.raw!")