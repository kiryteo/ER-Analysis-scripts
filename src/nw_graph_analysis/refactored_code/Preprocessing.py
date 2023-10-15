# preprocessing module

import imageio
import skimage
from skimage.filters import threshold_local
import mclahe as mc


class DataPreprocessor:
    def __init__(self, modality):
        self.modality = modality

    def normalize(self, img):
        return (img - img.min()) / (img.max() - img.min())
    
    def area_opening(self, img, area_threshold=2):
        return skimage.morphology.area_opening(img, area_threshold=area_threshold)
    
    def erosion(self, img):
        return skimage.morphology.erosion(img)
    
    def area_closing(self, img, area_threshold=32):
        return skimage.morphology.area_closing(img, area_threshold=area_threshold)
    
    def threshold_local(self, img, block_size=3):
        return threshold_local(img, block_size)
    
    def multidim_clahe(self, img):
        img = self.normalize(img)
        return mc.mclahe(img)

    def preprocess_samples(self, img):
        if self.modality == 'confocal':
            img = self.normalize(img)
        else:
            img = self.multidim_clahe(img)
        img = self.area_opening(img)
        img = self.erosion(img)
        img = self.area_opening(img)
        img = self.area_closing(img)
        img = self.area_opening(img)
        img = self.threshold_local(img)
        return img

    # TODO: add preprocessing loop for groups
