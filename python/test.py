from import_data import import_csv
import scipy
from matplotlib import pyplot as plt

x,y = import_csv('../resampled-to-8bit.csv')

im = x[0]
plt.imshow(im)

"""
scipy.misc.imshow(im)
scipy.misc.imshow(misc.imrotate(im, 45))
scipy.misc.imshow(misc.imrotate(im, 15))
scipy.misc.imshow(misc.imrotate(im, 90))
scipy.misc.imshow(misc.imrotate(im, 100))
"""

