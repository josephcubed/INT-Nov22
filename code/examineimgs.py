import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
from astropy.visualization import ZScaleInterval, ImageNormalize
from photutils.aperture import CircularAperture, CircularAnnulus
from photutils.centroids import centroid_com, centroid_sources
from photutils import aperture_photometry
from astropy.stats import sigma_clipped_stats
import csv
import re

night = '22'
fitses = []
for name in sorted(glob.glob(f'/Users/josephmurtagh/Documents/INTReducedData/202211{night}/'+'*.fit')):
	fitses.append(name)

positions = pd.read_csv('matchedpositions.csv')

# simply plot the stars over all frames
# for file in fitses:
#     hdu = fits.open(file)
#     img = hdu[0].data
#     hdr = hdu[0].header
#     irafname = hdr['IRAFNAME']

#     xpos = []
#     ypos = []
#     for row in positions[irafname]:
#         xpos.append(float(re.search(r"(?<=\().+?(?=\,)", row).group()))
#         ypos.append(float(re.search(r"(?<=\ ).+?(?=\))", row).group()))

#     fig = plt.figure(figsize=(10,10))
#     plt.imshow(img, cmap='gray', norm=ImageNormalize(img, interval=ZScaleInterval()))
#     plt.scatter(xpos, ypos, s=10., color='blue', marker='x')
#     plt.title(irafname)
#     plt.gca().invert_yaxis()
#     plt.show()

# calculate the average drift inbetween each frame and overall
drifts = []

xpos = []
ypos = []
for row in positions['r1642676']:
    xpos.append(float(re.search(r"(?<=\().+?(?=\,)", row).group()))
    ypos.append(float(re.search(r"(?<=\ ).+?(?=\))", row).group()))

d = {'x':xpos, 'y':ypos}
old_df = pd.DataFrame(d)

fig = plt.figure()

for i, file in enumerate(fitses[1:]):
    hdu = fits.open(file)
    hdr = hdu[0].header
    irafname = hdr['IRAFNAME']

    xpos = []
    ypos = []
    for row in positions[irafname]:
        xpos.append(float(re.search(r"(?<=\().+?(?=\,)", row).group()))
        ypos.append(float(re.search(r"(?<=\ ).+?(?=\))", row).group()))
    
    d = {'x':xpos, 'y':ypos}
    new_df = pd.DataFrame(d)

    drift = np.sqrt((old_df['x'] - new_df['x'])**2 + (old_df['y'] - new_df['y'])**2)
    avg_drift = np.mean(drift)
    avg_drift_arcs = avg_drift * 0.33
    drifts.append(avg_drift_arcs)

    plt.scatter(i, avg_drift_arcs)

    old_df = new_df

drifts = np.asarray(drifts)
print(f'Average drift is: {np.mean(drifts[drifts < 10])}')
plt.xlabel('frame')
plt.ylabel('drift /"')
# plt.ylim(0, 3)
plt.show()