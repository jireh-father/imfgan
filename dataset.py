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
        while self.config.batch_size >= cnt:
            try:
                if self.config.dataset_preload:
                    poster = random.choice(self.posters)
                    bg = random.choice(self.bgs)
                    title = random.choice(self.titles)
                    credit = random.choice(self.credits)
                else:
                    poster = self._open_and_resize(random.choice(self.poster_files))
                    bg = self._open_and_resize(random.choice(self.bg_files))
                    title = self._open_and_resize(random.choice(self.title_files))
                    credit = self._open_and_resize(random.choice(self.credit_files))
                image_input = np.concatenate((bg, title, credit), axis=2)

                input_batch.append(image_input)
                real_batch.append(poster)
                cnt += 1
            except:
                continue
        if cnt < 1:
            raise Exception("failed to get batch data.")
        return np.array(input_batch), np.array(real_batch)

    def _open_and_resize(self, image_file):
        img = Image.open(image_file)
        img = np.array(img.resize((self.config.input_width, self.config.input_height), Image.ANTIALIAS))
        img /= 255
        return (img - 0.5) * 2

    def _preload(self, files, dataset):
        for file in files:
            try:
                dataset.append(self._open_and_resize(file))
            except:
                continue

    def _preload_dataset(self):
        self._preload(self.poster_files, self.posters)
        self._preload(self.bg_files, self.bgs)
        self._preload(self.credit_files, self.credits)
        self._preload(self.title_files, self.titles)
