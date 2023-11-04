import numpy as np

from PIL import Image
import io
import numpy as np

# image_data = list(uploader.value.values())[0]['content']
# im = np.array(Image.open(io.BytesIO(image_data)))

im = np.array(Image.open('/localhome/asa420/MIAL/data/confocal-data/ATL/std/A1_decon_t000_ch00_std.png'))
# im = (im - im.min()) / (im.max() - im.min())


csum = lambda z: np.cumsum(z)[:-1]
dsum = lambda z: np.cumsum(z[::-1])[-2::-1]
argmax = lambda x, f: np.mean(x[:-1][f == np.max(f)])  # Use the mean for ties.
clip = lambda z: np.maximum(1e-30, z)

def preliminaries(n, x):
  """Some math that is shared across multiple algorithms."""
  assert np.all(n >= 0)
  x = np.arange(len(n), dtype=n.dtype) if x is None else x
  assert np.all(x[1:] >= x[:-1])
  w0 = clip(csum(n))
  w1 = clip(dsum(n))
  p0 = w0 / (w0 + w1)
  p1 = w1 / (w0 + w1)
  mu0 = csum(n * x) / w0
  mu1 = dsum(n * x) / w1
  d0 = csum(n * x**2) - w0 * mu0**2
  d1 = dsum(n * x**2) - w1 * mu1**2
  return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def GHT(n, x=None, nu=0, tau=0, kappa=0, omega=0.5):
  assert nu >= 0
  assert tau >= 0
  assert kappa >= 0
  assert omega >= 0 and omega <= 1
  x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
  v0 = clip((p0 * nu * tau**2 + d0) / (p0 * nu + w0))
  v1 = clip((p1 * nu * tau**2 + d1) / (p1 * nu + w1))
  f0 = -d0 / v0 - w0 * np.log(v0) + 2 * (w0 + kappa *      omega)  * np.log(w0)
  f1 = -d1 / v1 - w1 * np.log(v1) + 2 * (w1 + kappa * (1 - omega)) * np.log(w1)
  return argmax(x, f0 + f1), f0 + f1


def im2hist(im, zero_extents=False):
  # Convert an image to grayscale, bin it, and optionally zero out the first and last bins.
  max_val = np.iinfo(im.dtype).max
  x = np.arange(max_val+1)
  e = np.arange(-0.5, max_val+1.5)
  assert len(im.shape) in [2, 3]
  im_bw = np.amax(im[...,:3], -1) if len(im.shape) == 3 else im
  n = np.histogram(im_bw, e)[0]
  if zero_extents:
    n[0] = 0
    n[-1] = 0
  return n, x, im_bw

# Precompute a histogram and some integrals.
n, x, im_bw = im2hist(im)
# prelim = preliminaries(n, x)

# Compute the GHT.
ght = GHT(n, x, nu=0, tau=0, kappa=0, omega=0.5)

print(n.shape)