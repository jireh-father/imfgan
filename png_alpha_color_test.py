import glob, os
from PIL import Image
import numpy as np

path = "E:\data\poster\\title"
files = glob.glob(os.path.join(path, "*"))

for file in files:
    if os.path.basename(file) != "transparent.png":
        continue
    img = Image.open(file)

    img = np.array(img)
    print(file)
    print(img.shape)
    print(img[0][0])
    print(img[5][5])
    for i in range(20):
        for j in range(20):
            img[i][j] = [255, 255, 255, 0]
            img[i][j + 25] = [0, 0, 0, 255]
            img[i][j + 50] = [255, 255, 255, 255]
    Image.fromarray(img).save(os.path.join(path, "test3.png"))

    # print(img[197][332])
