import numpy as np

# Set the dimensions
W, H = 256, 256

# Create a pure black background (0 = black)
img = np.zeros((H, W), dtype=np.uint8)

# Draw a white box in the middle (255 = white)
# We make it start at row 50 and end at row 200, same for columns.
img[50:200, 50:200] = 255 

# Save it as a raw binary file
img.tofile("test_input.raw")
print("Success: Saved 256x256 test_input.raw!")
