from PIL import Image
import numpy as np


img = np.array(Image.open("f:\\2.png"))
print(img[0][0])
print(img[92][239])