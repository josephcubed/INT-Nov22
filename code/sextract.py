import sys
import subprocess
import glob
from astropy.io import fits

def runSex(sexFile, imageName, options=None, verbose=False):
	'''
	Inputs
	------
	sexFile: string


	Outputs
	-------

	'''
	sexName = 'sex'

	comm = sexName + ' ' + imageName + ' -c ' + sexFile

	if options:
		for ii in options:
			comm += ' -' + ii + ' ' + options[ii]
	
	process = subprocess.Popen(comm.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding='utf8')
	
	junk = process.communicate()
	if verbose:
		print(comm)
		for i in range(len(junk)):
			print(junk[i])

	return junk[1]

night = sys.argv[1]
fitses = []

for name in sorted(glob.glob(f'/Users/josephmurtagh/Documents/INTReducedData/202211{night}/'+'*.fit')):
	fitses.append(name)

for j in range(len(fitses)):
	hdu = fits.open(fitses[j])
	hdr = hdu[0].header
	irafname = hdr['IRAFNAME']

	runSex('default.sex', f'/Users/josephmurtagh/Documents/INTReducedData/202211{night}/'+irafname+'.fit', verbose=True)
	process = subprocess.Popen(f'sort -k 1n test.cat > position_{irafname}.cat', stdin=subprocess.PIPE, shell=True)
	# junk = process.communicate(); print(junk)

process = subprocess.Popen(f'mv *.cat "sextractor positions"', stdin=subprocess.PIPE, shell=True)