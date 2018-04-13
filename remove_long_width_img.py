from PIL import Image
import glob, os

path = "F:\data\imfgan\\test\movie\poster"
files = glob.glob(os.path.join(path, "*"))
cnt = 0
for file in files:
    f = open(file, "rb")
    try:
        img = Image.open(f)
    except:
        f.close()
        os.remove(file)
        continue

    f.close()
    if img.size[0] > img.size[1]:
        print(file)
        cnt +=1
        os.remove(file)

print(cnt)
