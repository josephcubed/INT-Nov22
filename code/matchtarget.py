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

from astroquery.jplhorizons import Horizons
from datetime import timedelta
from astropy.time import Time

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

night = '22'
fitses = []
for name in sorted(glob.glob(f'/Users/josephmurtagh/Documents/INTReducedData/202211{night}/'+'*.fit')):
	fitses.append(name)

for file in fitses:
    hdu = fits.open(file)
    img = hdu[0].data
    hdr = hdu[0].header

    dtime = Time(f'{hdr["MJD-OBS"]}', format = 'mjd')
    dtime.format = 'iso'
    dtime2 = dtime + timedelta(days=1)

    obj = Horizons(id=objid, location='I41', epochs={'start':dtime.iso, 'stop':dtime2.iso, 'step':'1d'})
    eph = obj.ephemerides(quantities='1,19,20,24,39') 
    # 1 = RA + DEC (/decimal degrees, icrf), 19(r) = heliocentric comet dist (/AU), 
    # 20(delta) = observer-comet dist (/AU), 24(alpha) = phase angle (/degrees)sad 
        