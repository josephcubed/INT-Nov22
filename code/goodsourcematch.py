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

def merge(l1, l2):
	return [(l1[i], l2[i]) for i in range(0, len(l1))]

def filter_data(df, initial=False):
	if initial:
		df = df[df['flux'] < 40000]
		df = df[df['flux'] > 1000]
	# if not initial:
	# 	df = df[df['flux'] < 40000]
	# 	df = df[df['flux'] > 5000]

	mask = []
	for row in df.itertuples(index=False):
		if (60 <= row.x <= 1750) and (60 <= row.y <= 1750):
			mask.append(True)
		else:
			mask.append(False)
	df = df[mask]

	mask = []
	for row in df.itertuples(index=False):
		if (1510 <= row.x <= 1620) and (420 <= row.y <= 1821):
			mask.append(False)
		else:
			mask.append(True)
	df = df[mask]

	return df[['x','y']].copy()

# ccd properties here
fwhm_pix = 1.5 / 0.33
gain = 1.6
rdnoise = 2.5

# small aperture and big aperture and annulus radii
aps = [fwhm_pix, 4.0 * fwhm_pix]
ann_aps = [4.0 * fwhm_pix + 1.0, 5.0 * fwhm_pix]

# read in all fits files
night = '22'
fitses = []
for name in sorted(glob.glob(f'/Users/josephmurtagh/Documents/INTReducedData/202211{night}/'+'*.fit')):
	fitses.append(name)

# read in the reference catalogue 
refcatalogue0 = np.genfromtxt('/Users/josephmurtagh/Documents/INTReducedData/20221122/sextractor positions/position_r1642676.cat')
refcatalogue0 = pd.DataFrame(refcatalogue0, columns=['m','flux','x','y'])
refcatalogue0 = filter_data(refcatalogue0, initial=True).head(15)

# df = refcatalogue0[['x','y']].copy()
# record = df.to_records(index=False)
# result = list(record)
# dic = {'r1642676':result}
# df = pd.DataFrame(dic)
# print(df)
# df.to_csv('matchedpositions.csv', mode='a', index=False)

midrefcatalogue = pd.read_csv('mid_refcat.csv')
postrefcatalogue = pd.read_csv('post_refcat.csv')

# decide which frames to use
# frames = fitses[1:19] # <- pre standards taking
frames = fitses[19:] # <- post standards taking

# set the reference catalogue as the old position dataframe for looping
# old_df = refcatalogue0
# old_df = midrefcatalogue
old_df = postrefcatalogue


# hdu = fits.open(fitses[0])
# img = hdu[0].data
# fig = plt.figure(figsize=(10,10))
# plt.imshow(img, cmap='gray', norm=ImageNormalize(img, interval=ZScaleInterval()))
# plt.scatter(old_df['x'], old_df['y'], marker='x', s=10, color='blue')
# plt.gca().invert_yaxis()
# plt.title(hdu[0].header['IRAFNAME'])
# plt.savefig(f'./example imgs/{hdu[0].header["IRAFNAME"]}.png', overwrite=True)


# source match each frame by comparing each sextractor catalogue 
for i, frame in enumerate(frames):
	hdu = fits.open(frame)
	img = hdu[0].data
	hdr = hdu[0].header
	irafname = hdr['IRAFNAME']

	df = pd.DataFrame(data=np.genfromtxt(f'./sextractor positions/position_{irafname}.cat'), columns=['mag','flux','x','y'])
	df = df[['x','y']].copy()

	new_pos = []

	for a in range(len(refcatalogue0)):
		x = old_df['x'].iloc[a]
		y = old_df['y'].iloc[a]
		distances = []

		for row in df.itertuples(index=False, name=None):
			d = np.sqrt((x - row[0])**2 + (y - row[1])**2)
			distances.append(d)

		distances = [abs(d) for d in distances]
		idx = np.argmin(distances)
		new_pos.append((df.iloc[idx][0], df.iloc[idx][1]))

	new_df = pd.DataFrame(new_pos, columns=['x','y'])
	old_df = new_df

	# print(irafname)
	# append to the wider position csv file
	dic = {irafname:new_pos}
	df = pd.DataFrame(dic)
	csv_input = pd.read_csv('matchedpositions.csv')
	csv_input[irafname] = df
	csv_input.to_csv('matchedpositions.csv', index=False)

# 	# fig = plt.figure(figsize=(10,10))
# 	# plt.imshow(img, cmap='gray', norm=ImageNormalize(img, interval=ZScaleInterval()))
# 	# plt.scatter(old_df['x'], old_df['y'], marker='x', s=10, color='blue')
# 	# plt.gca().invert_yaxis()
# 	# plt.title(irafname)
# 	# plt.savefig(f'./example imgs/{irafname}.png', overwrite=True)
	