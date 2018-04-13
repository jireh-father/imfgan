import os, sys, glob
from PIL import Image
import random

cnt = 1
bg_path = "F:\data\imfgan\\background"
logo_path = "F:\data\imfgan\\title"
credit_path = "F:\data\imfgan\downloads\movie credit png"

output_path = "F:\data\imfgan\\dataset\\fake"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# use width and height 50%
bg_imgs = glob.glob(os.path.join(bg_path, "*.jpg"))
if len(bg_imgs) < 1:
    sys.exit()
total = 0
stop = False
logo_files = glob.glob(os.path.join(logo_path, "*.png"))
credit_files = glob.glob(os.path.join(credit_path, "*.png"))
i = 0
random.shuffle(bg_imgs)
while True:
    for bg_img in bg_imgs:

        img_f = open(bg_img, 'rb')
        try:
            img_obj = Image.open(img_f)
        except:
            img_f.close()
            continue
        h = img_obj.size[1]
        w = img_obj.size[0]

        left_start = 0
        left_end = w // 4
        right_start = w - left_end
        right_end = w
        top_start = 0
        top_end = h // 4
        bot_start = h - top_end
        bot_end = h

        left = random.randrange(left_start, left_end)
        right = random.randrange(right_start, right_end)
        top = random.randrange(top_start, top_end)
        bot = random.randrange(bot_start, bot_end)
        img_obj = img_obj.crop((left, top, right, bot))
        bg = Image.new('RGBA', img_obj.size, (0, 0, 0, 0))
        bg.paste(img_obj, (0, 0))
        # bg.save("f:\\2.png")

        logo_file = random.choice(logo_files)
        credit_file = random.choice(credit_files)
        logo_f = open(logo_file, 'rb')
        credit_f = open(credit_file, 'rb')
        try:
            logo_obj = Image.open(logo_f).convert("RGBA")
            credit_obj = Image.open(credit_f).convert("RGBA")
        except:
            logo_f.close()
            credit_f.close()
            img_f.close()
            continue
        if logo_obj.size[0] > bg.size[0]:
            logo_obj.thumbnail((bg.size[0], bg.size[0]), Image.ANTIALIAS)
        if credit_obj.size[0] > bg.size[0]:
            credit_obj.thumbnail((bg.size[0], bg.size[0]), Image.ANTIALIAS)

        logo_size_ratio = random.randrange(30, 100) / 100
        credit_size_ratio = random.randrange(30, 100) / 100
        logo_obj = logo_obj.resize((int(logo_obj.size[0] * logo_size_ratio), int(logo_obj.size[1] * logo_size_ratio)),
                                   Image.ANTIALIAS)
        credit_obj = credit_obj.resize(
            (int(credit_obj.size[0] * credit_size_ratio), int(credit_obj.size[1] * credit_size_ratio)), Image.ANTIALIAS)
        w = bg.size[0]
        h = bg.size[1]
        left_start = 0
        left_end = int(w - (w // 3))
        top_start = 0
        top_end = int(h - (h // 5))

        credit_left = random.randrange(left_start, left_end)
        credit_top = random.randrange(top_start, top_end)

        bg.paste(credit_obj, (credit_left, credit_top), credit_obj)

        logo_left = random.randrange(left_start, left_end)
        logo_top = random.randrange(top_start, top_end)

        bg.paste(logo_obj, (logo_left, logo_top), logo_obj)
        bg = bg.convert('RGB')
        bg.save(os.path.join(output_path, str(i)+"_"+os.path.basename(bg_img) + ".jpg"))
        total += 1
        print(total)
        logo_f.close()
        credit_f.close()
        img_f.close()
        if total >= cnt:
            stop = True
            break
    i+=1
    if stop:
        break
