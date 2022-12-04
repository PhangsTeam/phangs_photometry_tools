"""
Script to compute and save NIRCAM and MIRI PSF
"""

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
    webbpsf.display_psf(psf_native)
    plt.savefig('plot_output/psf_nircam_%s.png' % nircam_filter)
    plt.clf()

for miri_filter in miri_filter_list:
    miri = webbpsf.MIRI()
    miri.filter = miri_filter
    psf_native = miri.calc_psf(oversample=1, normalize='last', outfile=psf_file_path + 'native_psf_' + miri_filter +
                                                                         '.fits')
    webbpsf.display_psf(psf_native)
    plt.savefig('plot_output/psf_miri_%s.png' % miri_filter)
    plt.clf()
