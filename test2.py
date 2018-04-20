from PIL import Image
def open_and_resize_with_padding(image_file):
    input_width = 1000
    input_height = 2500
    img = Image.open(image_file)
    w, h = img.size
    w = float(w)
    h = float(h)
    img_ratio = w / h
    output_ratio = float(input_width) / input_height
    if w > input_width or h > input_height:
        if img_ratio >= output_ratio:
            # resize by width
            resize_ratio = input_width / w
            t_w = input_width
            t_h = input_height * resize_ratio
        else:
            resize_ratio = input_height / h
            t_w = input_width * resize_ratio
            t_h = input_height
        img = img.resize((t_w, t_h), Image.ANTIALIAS)

    bg = Image.new('RGB', (input_width, input_height), (0, 0, 0))
    bg.paste(img, (0, 0))
    bg.save("haha.jpg")

    # img = np.array(bg)
    # img /= 255
    # return (img - 0.5) * 2


open_and_resize_with_padding("f:/5.jpg")