import os
import imageio
import numpy as np
import itertools
import matplotlib.pyplot as plt

class FourierAnalysis:
    def __init__(self):
        self.home = os.path.expanduser('~')
        self.confocal_data_path = '/MIAL/data/confocal_movies/'

    def get_std_img(self, path):
        """Read and standardize an image."""
        img = imageio.imread(path)
        return (img - img.min()) / (img.max() - img.min())

    def construct_path(self, group, num, i):
        """Construct the file path based on the group and indices."""
        if group == 'Control':
            return f'{self.home}{self.confocal_data_path}{group}/files/img_{num}_decon_t0{i:02d}.tif'
        else:
            return f'{self.home}{self.confocal_data_path}{group}/files/{group[0]}{num}_decon_t0{i:02d}_ch00.tif'

    def fourier_analysis(self, l):
        """Perform Fourier analysis on the list of images."""
        l = np.reshape(l, (-1, 128, 128))
        f = np.fft.fftn(l - np.mean(l))
        fabs = np.abs(f)
        fviz = np.fft.fftshift(fabs)
        return [np.sum(each) for each in fviz]

    def seq_fourier_analysis(self, group, num_series):
        """Perform Fourier analysis on a sequence of images."""
        l = [self.get_std_img(self.construct_path(group, num_series, i)) for i in range(100)]
        l = np.concatenate(l)
        return self.fourier_analysis(l)

    def seq_movie_fourier_analysis(self, group, num_series):
        """Perform Fourier analysis on a movie sequence of images."""
        l = [self.get_std_img(self.construct_path(group, num, i)) for num, i in itertools.product(range(1, num_series + 1), range(100))]
        l = np.concatenate(l)
        return self.fourier_analysis(l)

if __name__ == "__main__":
    # Example usage
    fa = FourierAnalysis()
    group = 'Control'
    num_series = 5
    result = fa.seq_movie_fourier_analysis(group, num_series)
    print(result)