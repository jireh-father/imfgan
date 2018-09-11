import glob, os, random
import numpy as np
from PIL import Image


class Dataset:
    def __init__(self, config):
        self.config = config
        self.poster_files = glob.glob(os.path.join(config.poster_dir, "*.jpg"))
        self.bg_files = glob.glob(os.path.join(config.bg_dir, "*.jpg"))
        self.title_files = glob.glob(os.path.join(config.title_dir, "*.png"))
        self.credit_files = glob.glob(os.path.join(config.credit_dir, "*.png"))
        return
        if not self.poster_files or not self.bg_files or not self.title_files or not self.credit_files:
            raise Exception("some dataset are empty.")
        if self.config.dataset_preload:
            self.posters = []
            self.bgs = []
            self.credits = []
            self.titles = []
            self._preload_dataset()

    def batch(self):
        input_batch = []
        real_batch = []
        cnt = 0
        while self.config.batch_size > cnt:
            try:
                if self.config.dataset_preload:
                    poster = random.choice(self.posters)
                    bg = random.choice(self.bgs)
                    title = random.choice(self.titles)
                    credit = random.choice(self.credits)
                else:
                    poster = self._open_and_resize(random.choice(self.poster_files), self.config.real_width,
                                                   self.config.real_height)
                    bg = self._open_and_resize(random.choice(self.bg_files), self.config.input_width,
                                               self.config.input_height)
                    title = self._open_and_resize_with_padding(random.choice(self.title_files))
                    credit = self._open_and_resize_with_padding(random.choice(self.credit_files))
                if len(bg.shape) != 3 or bg.shape[2] != 3:
                    continue
                if len(title.shape) != 3 or title.shape[2] != 4:
                    continue
                if len(credit.shape) != 3 or credit.shape[2] != 4:
                    continue
                if len(poster.shape) != 3 or poster.shape[2] != 3:
                    continue
                image_input = np.concatenate((bg, title, credit), axis=2)

                input_batch.append(image_input)
                real_batch.append(poster)
                cnt += 1
            except:
                import traceback
                traceback.print_exc()
                continue
        if cnt < 1:
            raise Exception("failed to get batch data.")
        return np.array(input_batch), np.array(real_batch)

    def _open_and_resize(self, image_file, width, height):
        img = Image.open(image_file)
        img = np.array(img.resize((width, height), Image.ANTIALIAS))
        img = img / 255.
        return (img - 0.5) * 2

    def _open_and_resize_with_padding(self, image_file):
        img = Image.open(image_file)
        w, h = img.size
        w = float(w)
        h = float(h)
        img_ratio = w / h
        output_ratio = float(self.config.input_width) / self.config.input_height
        if w > self.config.input_width or h > self.config.input_height:
            if img_ratio >= output_ratio:
                # resize by width
                resize_ratio = self.config.input_width / w
                t_w = self.config.input_width
                t_h = self.config.input_height * resize_ratio
            else:
                resize_ratio = self.config.input_height / h
                t_w = self.config.input_width * resize_ratio
                t_h = self.config.input_height
            img = img.resize((int(round(t_w)), int(round(t_h))), Image.ANTIALIAS)

        bg = Image.new('RGBA', (self.config.input_width, self.config.input_height), (0, 0, 0, 0))
        bg.paste(img, (0, 0))

        img = np.array(bg)
        img = img / 255.
        return (img - 0.5) * 2

    def _preload(self, files, dataset, is_text=False):
        for file in files:
            try:
                if is_text:
                    dataset.append(self._open_and_resize_with_padding(file))
                else:
                    dataset.append(self._open_and_resize(file))
            except:
                continue

    def _preload_dataset(self):
        self._preload(self.poster_files, self.posters)
        self._preload(self.bg_files, self.bgs)
        self._preload(self.credit_files, self.credits, True)
        self._preload(self.title_files, self.titles, True)
