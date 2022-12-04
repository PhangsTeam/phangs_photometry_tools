"""
Script to compute and save HST PSFs for the PHANGS HST collaboration
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import interpolate
from astropy.io import fits

from photometry_tools.helper_func import download_file

from matplotlib.colors import LogNorm
from matplotlib.colorbar import ColorbarBase

# Pixel scales of different cameras
#
# taken from Wide Field Camera 3 Instrument Handbook for Cycle 30 page 20
# https://hst-docs.stsci.edu/wfc3ihb/files/60242283/109773105/1/1641931722986/wfc3_ihb.pdf
acs_wfc_pix_scale = 202 / 4096  # = 0.04931640625

# taken from ACS Data Handbook page 8
# https://www.stsci.edu/files/live/sites/www/files/home/hst/documentation/_documents/acs/acs_dhb_v6.pdf
wfc3_pix_scale = 162 / 4096  # = 0.03955078125

# pixel scale after re-drizzling
# taken from F275W header of NGC0628 PHANGS data product
# D001ISCL= 0.03962 / Drizzle, default IDCTAB pixel size(arcsec)
drizzled_pix_scale = 0.03962


def rebin_2d_array(a, shape):
    sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
    return a.reshape(sh).mean(-1).mean(1)


def plot_hst_wfc3_psf(oversampled_psf, detector_psf, pix_scale, camera, band):

    length = (detector_psf.shape[0] / 2) * pix_scale

    fontsize = 17
    figure = plt.figure(figsize=(11, 5))


    ax_oversampled = figure.add_axes([-0.12, 0.115, 0.80, 0.80])
    ax_detector = figure.add_axes([0.25, 0.115, 0.80, 0.80])
    ax_cbar = figure.add_axes([0.89, 0.12, 0.02, 0.78])

    cmap = 'viridis'
    ticks = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]
    vmin=1e-7
    vmax=1e-1
    norm = LogNorm(vmin=vmin, vmax=vmax)
    oversampled_psf[oversampled_psf <= 0] = vmin
    detector_psf[detector_psf <= 0] = vmin
    ax_oversampled.imshow(oversampled_psf, extent=(-length, length, -length, length), origin='lower', norm=norm,)
    ax_detector.imshow(detector_psf, extent=(-length, length, -length, length), origin='lower', norm=norm,)

    ColorbarBase(ax_cbar, orientation='vertical', cmap=cmap, norm=norm, extend='neither', ticks=ticks)
    ax_cbar.set_ylabel(r'Fractional Intensity per pixel', labelpad=0, fontsize=fontsize)
    ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)

    ax_oversampled.set_title('Oversampled %s %s' % (camera, band), fontsize=fontsize)
    ax_detector.set_title('Native resolution %s %s' % (camera, band), fontsize=fontsize)

    ax_oversampled.set_ylabel('y [arcsec]', fontsize=fontsize)
    ax_oversampled.set_xlabel('x [arcsec]', fontsize=fontsize)
    ax_detector.set_xlabel('x [arcsec]', fontsize=fontsize)
    ax_detector.set_yticklabels([])
    ax_oversampled.tick_params(axis='both', which='both', width=2, color='white', top=True, right=True, direction='in', labelsize=fontsize)
    ax_detector.tick_params(axis='both', which='both', width=2, color='white', top=True, right=True, direction='in', labelsize=fontsize)

    if not os.path.isdir('plot_output'):
        os.makedirs('plot_output')

    plt.tight_layout()
    plt.savefig('plot_output/psf_%s_%s.png' % (camera, band))
    figure.clf()


def plot_hst_acs_psf(original_psf, model_psf, oversampled_psf, drizzle_psf,
                     length_original, length_model, length_oversampled, length_drizzle, band):

    fontsize = 17
    figure = plt.figure(figsize=(20, 5))

    ax_original = figure.add_axes([-0.25, 0.115, 0.8, 0.8])
    ax_model = figure.add_axes([-0.045, 0.115, 0.8, 0.8])
    ax_oversampled = figure.add_axes([0.2, 0.115, 0.8, 0.8])
    ax_drizzle = figure.add_axes([0.405, 0.115, 0.8, 0.8])
    ax_cbar = figure.add_axes([0.93, 0.12, 0.02, 0.78])


    vmin=1e-7
    vmax=1e-1
    norm = LogNorm(vmin=vmin, vmax=vmax)
    original_psf[original_psf <= 0] = vmin
    model_psf[model_psf <= 0] = vmin
    oversampled_psf[oversampled_psf <= 0] = vmin
    drizzle_psf[drizzle_psf <= 0] = vmin

    ticks = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1]

    cmap = 'viridis'
    ax_original.imshow(original_psf, extent=(-length_original, length_original, -length_original, length_original), origin='lower', norm=norm)
    ax_model.imshow(model_psf, extent=(-length_model, length_model, -length_model, length_model), origin='lower', norm=norm)
    ax_oversampled.imshow(oversampled_psf, extent=(-length_oversampled, length_oversampled, -length_oversampled, length_oversampled), origin='lower', norm=norm)
    ax_drizzle.imshow(drizzle_psf, extent=(-length_drizzle, length_drizzle, -length_drizzle, length_drizzle), origin='lower', norm=norm)

    ColorbarBase(ax_cbar, orientation='vertical', cmap=cmap, norm=norm, extend='neither', ticks=ticks)
    ax_cbar.set_ylabel(r'Fractional Intensity per pixel', labelpad=0, fontsize=fontsize)
    ax_cbar.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)

    ax_original.set_title('Original oversampled ACS res %s' % band, fontsize=fontsize)
    ax_model.set_title('Interpolated model %s' % band, fontsize=fontsize)
    ax_oversampled.set_title('Oversampled WFC3 res %s' % band, fontsize=fontsize)
    ax_drizzle.set_title('Native WFC3 res %s' % band, fontsize=fontsize)

    ax_original.set_ylabel('y [arcsec]', fontsize=fontsize)
    # ax_oversampled.set_ylabel('y [arcsec]', fontsize=fontsize)
    ax_original.set_xlabel('x [arcsec]', fontsize=fontsize)
    ax_model.set_xlabel('x [arcsec]', fontsize=fontsize)
    ax_oversampled.set_xlabel('x [arcsec]', fontsize=fontsize)
    ax_drizzle.set_xlabel('x [arcsec]', fontsize=fontsize)
    ax_model.set_yticklabels([])
    # ax_oversampled.set_yticklabels([])
    ax_drizzle.set_yticklabels([])
    ax_original.tick_params(axis='both', which='both', width=2, color='white', top=True, right=True, direction='in', labelsize=fontsize)
    ax_model.tick_params(axis='both', which='both', width=2, color='white', top=True, right=True, direction='in', labelsize=fontsize)
    ax_oversampled.tick_params(axis='both', which='both', width=2, color='white', top=True, right=True, direction='in', labelsize=fontsize)
    ax_drizzle.tick_params(axis='both', which='both', width=2, color='white', top=True, right=True, direction='in', labelsize=fontsize)

    if not os.path.isdir('plot_output'):
        os.makedirs('plot_output')

    plt.tight_layout()
    plt.savefig('plot_output/psf_acs_%s.png' % band)
    figure.clf()


filter_list_wfc3 = ['F225W', 'F275W', 'F336W', 'F390W', 'F438W', 'F467M', 'F555W', 'F606W', 'F621M', 'F775W', 'F814W',
                    'F850L']
filter_list_acs = ['F435W', 'F475W', 'F606W', 'F625W_SM3', 'F658N_SM3', 'F775W', 'F814W', 'F850L_SM3']

# getting psf function 4 times oversampled for the entire detector as described in
# for wfc3 https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/wfc3/documentation/
# instrument-science-reports-isrs/_documents/2018/WFC3-2018-14.pdf
# for ACS https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/acs/documentation/
# instrument-science-reports-isrs/_documents/isr1808.pdf
# data are available at FTP server:
ftp_url = 'https://www.stsci.edu/~jayander/HST1PASS/LIB/PSFs/STDPSFs/'

# downloaded data will be saved at:
psf_file_path = 'data_output/hst_psf/'
# create folder for WFC and ACS
if not os.path.isdir(psf_file_path + 'wfc3/'):
    os.makedirs(psf_file_path + 'wfc3/')
if not os.path.isdir(psf_file_path + 'acs/'):
    os.makedirs(psf_file_path + 'acs/')


for band in filter_list_wfc3:
    print(band)
    psf_file_name = 'STDPSF_WFC3UV_%s.fits' % band

    # download file
    download_file(file_path=psf_file_path + 'wfc3/' + psf_file_name, url=ftp_url + 'WFC3UV/' + psf_file_name)

    # open hdu
    psf_hdu = fits.open(psf_file_path + 'wfc3/' + psf_file_name)
    # sum up all PSFs
    oversampled_4_psf = np.sum(psf_hdu[0].data[:, 4:96, 4:96], axis=0)
    # normalize psf
    oversampled_4_psf /= np.sum(oversampled_4_psf)
    # get native binning therefore get rif of the 4x4 oversampling
    native_psf = rebin_2d_array(oversampled_4_psf, (23, 23))
    # normalize
    native_psf /= np.sum(native_psf)

    # save psf function
    if not os.path.isdir('../data/hst_psf/wfc3/'):
        os.makedirs('../data/hst_psf/wfc3/')

    native_hdu = fits.PrimaryHDU(native_psf)
    hdul_native = fits.HDUList([native_hdu])
    hdul_native.writeto('../data/hst_psf/wfc3/native_psf_wfc3_%s.fits' % band, overwrite=True)

    oversampled_hdu = fits.PrimaryHDU(oversampled_4_psf)
    hdul_oversampled = fits.HDUList([oversampled_hdu])
    hdul_oversampled.writeto('../data/hst_psf/wfc3/oversampled_psf_wfc3_%s.fits' % band, overwrite=True)

    # now plotiplot
    plot_hst_wfc3_psf(oversampled_psf=oversampled_4_psf, detector_psf=native_psf, pix_scale=wfc3_pix_scale,
                      camera='wfc3', band=band)


for band in filter_list_acs:
    print(band)
    psf_file_name = 'STDPSF_ACSWFC_%s.fits' % band
    # download file
    download_file(file_path=psf_file_path + 'acs/' + psf_file_name, url=ftp_url + 'ACSWFC/' + psf_file_name)
    # open hdu
    psf_hdu = fits.open(psf_file_path + 'acs/' + psf_file_name)
    # get normalized psf
    oversampled_acs_4_psf = np.sum(psf_hdu[0].data, axis=0)
    oversampled_acs_4_psf /= np.sum(oversampled_acs_4_psf)

    # interpolate psf
    length_acs = oversampled_acs_4_psf.shape[0] * acs_wfc_pix_scale / 4
    X = np.linspace(-length_acs/2, length_acs/2, oversampled_acs_4_psf.shape[0])
    Y = np.linspace(-length_acs/2, length_acs/2, oversampled_acs_4_psf.shape[1])
    x, y = np.meshgrid(X, Y, sparse=True)
    f = interpolate.interp2d(x, y, oversampled_acs_4_psf, kind='cubic')
    # produce high resolution model
    X_high_res = np.linspace(-length_acs/2, length_acs/2, oversampled_acs_4_psf.shape[0] * 4)
    Y_high_res = np.linspace(-length_acs/2, length_acs/2, oversampled_acs_4_psf.shape[1] * 4)
    psf_high_res = f(X_high_res, Y_high_res)

    # produce same psf as for the WFC3 filters
    drizzled_pixel_numbers = 23
    length_drizzle = drizzled_pixel_numbers * drizzled_pix_scale
    X_native = np.linspace(-(drizzled_pixel_numbers-1)/2*drizzled_pix_scale,
                           (drizzled_pixel_numbers-1)/2*drizzled_pix_scale, drizzled_pixel_numbers)
    Y_native = np.linspace(-(drizzled_pixel_numbers-1)/2*drizzled_pix_scale,
                           (drizzled_pixel_numbers-1)/2*drizzled_pix_scale, drizzled_pixel_numbers)
    X_oversampled = np.linspace(-(drizzled_pixel_numbers-1)/2*drizzled_pix_scale,
                                (drizzled_pixel_numbers-1)/2*drizzled_pix_scale, drizzled_pixel_numbers*4)
    Y_oversampled = np.linspace(-(drizzled_pixel_numbers-1)/2*drizzled_pix_scale,
                                (drizzled_pixel_numbers-1)/2*drizzled_pix_scale, drizzled_pixel_numbers*4)
    native_psf = f(X_native, Y_native)
    oversampled_4_psf = f(X_oversampled, Y_oversampled)
    # normalize psf
    native_psf /= np.sum(native_psf)
    oversampled_4_psf /= np.sum(oversampled_4_psf)

    # plot that everything
    plot_hst_acs_psf(original_psf=oversampled_acs_4_psf, model_psf=psf_high_res, oversampled_psf=oversampled_4_psf,
                     drizzle_psf=native_psf, length_original=length_acs, length_model=length_acs,
                     length_oversampled=length_drizzle, length_drizzle=length_drizzle, band=band)

    # save psf function
    if not os.path.isdir('../data/hst_psf/acs/'):
        os.makedirs('../data/hst_psf/acs/')

    native_hdu = fits.PrimaryHDU(native_psf)
    hdul_native = fits.HDUList([native_hdu])
    hdul_native.writeto('../data/hst_psf/acs/native_psf_acs_%s.fits' % band, overwrite=True)

    oversampled_hdu = fits.PrimaryHDU(oversampled_4_psf)
    hdul_oversampled = fits.HDUList([oversampled_hdu])
    hdul_oversampled.writeto('../data/hst_psf/acs/oversampled_psf_acs_%s.fits' % band, overwrite=True)

