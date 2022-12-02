"""
Script to compute and save NIRCAM and MIRI PSF
"""

import numpy as np
import os
import webbpsf
import matplotlib.pyplot as plt

# set environment path
os.environ['WEBBPSF_PATH'] = "/home/benutzer/data/webbpsf-data-1.1.0/webbpsf-data"

nircam_filter_list = ['F200W', 'F300M', 'F335M', 'F360M']
miri_filter_list = ['F770W', 'F1000W', 'F1130W', 'F2100W']

psf_file_path = '../data/jwst_psf/'
if not os.path.isdir(psf_file_path):
    os.makedirs(psf_file_path)


for nircam_filter in nircam_filter_list:
    nircam = webbpsf.NIRCam()
    nircam.filter = nircam_filter
    psf_native = nircam.calc_psf(oversample=1, normalize='last', outfile=psf_file_path + 'native_psf_' + nircam_filter +
                                                                         '.fits')

for miri_filter in miri_filter_list:
    miri = webbpsf.MIRI()
    miri.filter = miri_filter
    psf_native = miri.calc_psf(oversample=1, normalize='last', outfile=psf_file_path + 'native_psf_' + miri_filter +
                                                                         '.fits')


exit()


print(psf_native.info())
print(np.sum(psf_native[0].data))
print(np.sum(psf_native[1].data))
print(np.sum(psf_native[2].data))
print(np.sum(psf_native[3].data))


fig, ax = plt.subplots(nrows=2, ncols=4)

ax[1, 0].imshow(psf_native[0].data, origin='lower')
ax[1, 1].imshow(psf_native[1].data, origin='lower')
ax[1, 2].imshow(psf_native[2].data, origin='lower')
ax[1, 3].imshow(psf_native[3].data, origin='lower')


plt.show()

exit()

