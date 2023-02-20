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


def merge(l1, l2):
	return [(l1[i], l2[i]) for i in range(0, len(l1))]

def error_array(image, gain, readnoise):
	bkg_array = image*0.0 + np.sqrt(image * gain + readnoise**2)
	return bkg_array/gain

def photometry(pos, img, gain, rdnoise, exptime): #errors aren't working for some reason - investigate?
	apertures = [CircularAperture(pos, r=r) for r in aps]
	annulus = CircularAnnulus(pos, r_in=sky_ap[0], r_out=sky_ap[1])
	annulus_masks = annulus.to_mask(method='center')
	err_pixels = error_array(img, gain, rdnoise)

	bkg_median = []
	for mask in annulus_masks:
		annulus_data = mask.multiply(img)
		annulus_data_1d = annulus_data[mask.data > 0]
		_, median_sigclip, _ = sigma_clipped_stats(annulus_data_1d)
		bkg_median.append(median_sigclip)
	bkg_median = np.array(bkg_median)

	phot = aperture_photometry(img, apertures)

	phot['annulus_median'] = bkg_median

	bkg = phot['annulus_median'] * apertures[0].area
	phot['aper_sum_bkgsub_0'] = phot['aperture_sum_0'] - bkg
	bkg = phot['annulus_median'] * apertures[1].area
	phot['aper_sum_bkgsub_1'] = phot['aperture_sum_1'] - bkg

	phot['inst_mag_0'] = -2.5 * np.log10(phot['aper_sum_bkgsub_0'] / exptime)
	# phot['inst_mag_err_0'] = 1.086 * phot['aperture_sum_err_0'] / phot['aper_sum_bkhsub_0']
	phot['inst_mag_1'] = -2.5 * np.log10(phot['aper_sum_bkgsub_1'] / exptime)
	# phot['inst_mag_err_1'] = 1.086 * phot['aperture_sum_err_1'] / phot['aper_sum_bkhsub_1']

	phot['ap_corr'] = phot['inst_mag_0'] - phot['inst_mag_1']

	print(f'Number of objects with photometry is {len(phot)}')

	return phot['inst_mag_0'], phot['inst_mag_1'], phot['ap_corr']

def plot(x, y, xlabel, ylabel, title=None, annotate=False):
    fig = plt.figure()

    # if not error:
    plt.scatter(x, y)
    # elif error:
    # plt.errorbar(x, y, yerr=yerr)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if title:
        plt.title(title)

    if annotate:
        txt = [row for row in positions[irafname]]
        for i, t in enumerate(txt):
            plt.annotate(t, (photsmall[i], l[i]))

    return fig

night = '22'
fitses = []
for name in sorted(glob.glob(f'/Users/josephmurtagh/Documents/INTReducedData/202211{night}/'+'*.fit')):
	fitses.append(name)

positions = pd.read_csv('matchedpositions.csv')

gain = 1.6
rdnoise = 2.5
fwhm_pix = 1.0/0.33
aps = [fwhm_pix, 4.0*fwhm_pix]
sky_ap = [4.0*fwhm_pix+1.0, 5.0*fwhm_pix]

inst_mag = pd.DataFrame(columns = list(positions.columns))
apcorr_errs = []

for file in fitses:
    hdu = fits.open(file)
    img = hdu[0].data
    hdr = hdu[0].header

    irafname = hdr['IRAFNAME']
    exptime = hdr['EXPTIME']
    
    xpos = []
    ypos = []
    for row in positions[irafname]:
        xpos.append(float(re.search(r"(?<=\().+?(?=\,)", row).group()))
        ypos.append(float(re.search(r"(?<=\ ).+?(?=\))", row).group()))

    d = {'x':xpos, 'y':ypos}
    pos = pd.DataFrame(d)

    photsmall, photbig, apcorr = photometry(pos, img, gain, rdnoise, exptime)
    _, apcorrgood, apcorr_err = sigma_clipped_stats(apcorr, sigma=3.0)
    print(f'aperture correction is {apcorrgood:5.3f}+/-{apcorr_err:5.3f}')
    inst_mag[irafname] = photsmall #- apcorrgood
    # apcorr_errs.append(apcorr_err)

    # plot instrumental magnitudes
    # l = np.arange(0, len(photsmall))
    # plot(photsmall, l, 'instrumental magnitude', 'star', title=irafname, annotate=False)
    # plt.show()

    # plot aperture corrections 
    # plot(apcorr, photsmall, 'aperture correction /mags', 'instrumental magnitude /mags', title=irafname)
    # plt.gca().invert_yaxis()
    # plt.show()

# plot individual star instrumental mags
l = np.arange(0, len(fitses))
# apcorr_errs = np.asarray(apcorr_errs)
# yerr = np.sqrt((apcorr_errs * )**2 + )
for i in range(len(inst_mag)):
    plot(l, inst_mag.loc[i], 'frame', 'instrumental mag (aperture corrected)')
    plt.show()

