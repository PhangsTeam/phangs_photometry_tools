"""
Example for a stand alone SEd plot
"""

from astropy.coordinates import SkyCoord
import astropy.units as u
from photometry_tools import helper_func
from photometry_tools.plotting_tools import plot_coord_circle
from scipy.constants import c as speed_of_light
import numpy as np
import matplotlib.pyplot as plt

# some constants / variables
sr_per_square_deg = 0.00030461741978671
nircam_zero_point_flux_corr = {'F200W': 0.854, 'F300M': 0.997, 'F335M': 1.000, 'F360M': 1.009}


def load_hst_band(img_file_name, flux_unit='Jy'):

    # load the band observations
    img_data, img_header, img_wcs = helper_func.load_img(file_name=img_file_name)

    # convert the flux unit
    if 'PHOTFNU' in img_header:
        conversion_factor = img_header['PHOTFNU']
    elif 'PHOTFLAM' in img_header:
        # wavelength in angstrom
        pivot_wavelength = img_header['PHOTPLAM']
        # inverse sensitivity, ergs/cm2/Ang/electron
        sensitivity = img_header['PHOTFLAM']
        # speed of light in Angstrom/s
        c = speed_of_light * 1e10
        # change the conversion facto to get erg s−1 cm−2 Hz−1
        f_nu = sensitivity * pivot_wavelength ** 2 / c
        # change to get Jy
        conversion_factor = f_nu * 1e23
    else:
        raise KeyError('there is no PHOTFNU or PHOTFLAM in the header')

    pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * sr_per_square_deg
    # rescale data image
    if flux_unit == 'Jy':
        # rescale to Jy
        conversion_factor = conversion_factor
    elif flux_unit == 'mJy':
        # rescale to Jy
        conversion_factor *= 1e3
    elif flux_unit == 'MJy/sr':
        # get the size of one pixel in sr with the factor 1e6 for the conversion of Jy to MJy later
        # change to MJy/sr
        conversion_factor /= (pixel_area_size_sr * 1e6)
    else:
        raise KeyError('flux_unit ', flux_unit, ' not understand!')

    img_data *= conversion_factor

    return img_data, img_wcs


def load_jwst_band(img_file_name, flux_unit='Jy', hdu_number=0):

    # load the band observations
    img_data, img_header, img_wcs = helper_func.load_img(file_name=img_file_name, hdu_number=hdu_number)
    pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * sr_per_square_deg
    # rescale data image
    if flux_unit == 'Jy':
        # rescale to Jy
        conversion_factor = pixel_area_size_sr * 1e6

    elif flux_unit == 'mJy':
        # rescale to Jy
        conversion_factor = pixel_area_size_sr * 1e9
    elif flux_unit == 'MJy/sr':
        conversion_factor = 1
    else:
        raise KeyError('flux_unit ', flux_unit, ' not understand')

    # add flux zero-point correction

    img_data *= conversion_factor

    return img_data, img_wcs


def erase_wcs_axis(ax):
    ax.coords['ra'].set_ticklabel_visible(False)
    ax.coords['ra'].set_axislabel(' ')
    ax.coords['dec'].set_ticklabel_visible(False)
    ax.coords['dec'].set_axislabel(' ')
    ax.tick_params(axis='both', which='both', width=0.00001, direction='in', color='k')


def plot_postage_stamps(ax, cutout, filter_color, fontsize=15, show_ax_label=False, title=None):

    m, s = np.nanmean(cutout.data), np.nanstd(cutout.data)
    ax.imshow(cutout.data, cmap='Greys', vmin=m-s, vmax=m+5*s)
    ax.set_title(title, fontsize=fontsize, color=filter_color)
    if show_ax_label:
        ax.tick_params(axis='both', which='both', width=3, length=7, direction='in',
                                              color='k', labelsize=fontsize-11)
        ax.coords['dec'].set_ticklabel(rotation=90)
        ax.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.3, fontsize=fontsize-11)
        ax.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize-11)
        ax.coords['ra'].set_ticks(number=2)
        ax.coords['ra'].display_minor_ticks(True)
        ax.coords['dec'].set_ticks(number=2)
        ax.coords['dec'].display_minor_ticks(True)
    else:
        erase_wcs_axis(ax)


# declare the file names
f275w_file_name = '/home/benutzer/data/PHANGS-HST/v1.0/ngc628mosaic/hlsp_phangs-hst_hst_wfc3-uvis_ngc628mosaic_f275w_v1_exp-drc-sci.fits'
f336w_file_name = '/home/benutzer/data/PHANGS-HST/v1.0/ngc628mosaic/hlsp_phangs-hst_hst_wfc3-uvis_ngc628mosaic_f336w_v1_exp-drc-sci.fits'
f435w_file_name = '/home/benutzer/data/PHANGS-HST/v1.0/ngc628mosaic/hlsp_phangs-hst_hst_acs-wfc_ngc628mosaic_f435w_v1_exp-drc-sci.fits'
f555w_file_name = '/home/benutzer/data/PHANGS-HST/v1.0/ngc628mosaic/hlsp_phangs-hst_hst_wfc3-uvis_ngc628mosaic_f555w_v1_exp-drc-sci.fits'
f814w_file_name = '/home/benutzer/data/PHANGS-HST/v1.0/ngc628mosaic/hlsp_phangs-hst_hst_acs-wfc_ngc628mosaic_f814w_v1_exp-drc-sci.fits'
f200w_file_name = '/home/benutzer/data/PHANGS-JWST/v0p4p2/ngc0628/ngc0628_nircam_lv3_f200w_i2d_align.fits'
f300m_file_name = '/home/benutzer/data/PHANGS-JWST/v0p4p2/ngc0628/ngc0628_nircam_lv3_f300m_i2d_align.fits'
f335m_file_name = '/home/benutzer/data/PHANGS-JWST/v0p4p2/ngc0628/ngc0628_nircam_lv3_f335m_i2d_align.fits'
f360m_file_name = '/home/benutzer/data/PHANGS-JWST/v0p4p2/ngc0628/ngc0628_nircam_lv3_f360m_i2d_align.fits'
f770w_file_name = '/home/benutzer/data/PHANGS-JWST/v0p5_miri/ngc0628_miri_f770w_anchored.fits'
f1000w_file_name = '/home/benutzer/data/PHANGS-JWST/v0p5_miri/ngc0628_miri_f1000w_anchored.fits'
f1130w_file_name = '/home/benutzer/data/PHANGS-JWST/v0p5_miri/ngc0628_miri_f1130w_anchored.fits'
f2100w_file_name = '/home/benutzer/data/PHANGS-JWST/v0p5_miri/ngc0628_miri_f2100w_anchored.fits'


f275w_img, f275w_wcs = load_hst_band(img_file_name=f275w_file_name, flux_unit='MJy/sr')
f336w_img, f336w_wcs = load_hst_band(img_file_name=f336w_file_name, flux_unit='MJy/sr')
f435w_img, f435w_wcs = load_hst_band(img_file_name=f435w_file_name, flux_unit='MJy/sr')
f555w_img, f555w_wcs = load_hst_band(img_file_name=f555w_file_name, flux_unit='MJy/sr')
f814w_img, f814w_wcs = load_hst_band(img_file_name=f814w_file_name, flux_unit='MJy/sr')
f200w_img, f200w_wcs = load_jwst_band(img_file_name=f200w_file_name, flux_unit='MJy/sr', hdu_number='SCI')
f300m_img, f300m_wcs = load_jwst_band(img_file_name=f300m_file_name, flux_unit='MJy/sr', hdu_number='SCI')
f335m_img, f335m_wcs = load_jwst_band(img_file_name=f335m_file_name, flux_unit='MJy/sr', hdu_number='SCI')
f360m_img, f360m_wcs = load_jwst_band(img_file_name=f360m_file_name, flux_unit='MJy/sr', hdu_number='SCI')
f770w_img, f770w_wcs = load_jwst_band(img_file_name=f770w_file_name, flux_unit='MJy/sr')
f1000w_img, f1000w_wcs = load_jwst_band(img_file_name=f1000w_file_name, flux_unit='MJy/sr')
f1130w_img, f1130w_wcs = load_jwst_band(img_file_name=f1130w_file_name, flux_unit='MJy/sr')
f2100w_img, f2100w_wcs = load_jwst_band(img_file_name=f2100w_file_name, flux_unit='MJy/sr')

# add zero point correction to NIRCAM data
f200w_img *= nircam_zero_point_flux_corr['F200W']
f300m_img *= nircam_zero_point_flux_corr['F300M']
f335m_img *= nircam_zero_point_flux_corr['F335M']
f360m_img *= nircam_zero_point_flux_corr['F360M']

# get cutouts
ra_cutout = 24.173946 - 33.5 / 3600
dec_cutout = 15.783662 - 27.6 / 3600
# size of image
size_of_cutout = (3, 3)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_pos = SkyCoord(ra=ra_cutout, dec=dec_cutout, unit=(u.degree, u.degree), frame='fk5')

f275w_cutout = helper_func.get_img_cutout(img=f275w_img, wcs=f275w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f336w_cutout = helper_func.get_img_cutout(img=f336w_img, wcs=f336w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f435w_cutout = helper_func.get_img_cutout(img=f435w_img, wcs=f435w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f555w_cutout = helper_func.get_img_cutout(img=f555w_img, wcs=f555w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f814w_cutout = helper_func.get_img_cutout(img=f814w_img, wcs=f814w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f200w_cutout = helper_func.get_img_cutout(img=f200w_img, wcs=f200w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f300m_cutout = helper_func.get_img_cutout(img=f300m_img, wcs=f300m_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f335m_cutout = helper_func.get_img_cutout(img=f335m_img, wcs=f335m_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f360m_cutout = helper_func.get_img_cutout(img=f360m_img, wcs=f360m_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f770w_cutout = helper_func.get_img_cutout(img=f770w_img, wcs=f770w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f1000w_cutout = helper_func.get_img_cutout(img=f1000w_img, wcs=f1000w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f1130w_cutout = helper_func.get_img_cutout(img=f1130w_img, wcs=f1130w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)
f2100w_cutout = helper_func.get_img_cutout(img=f2100w_img, wcs=f2100w_wcs, coord=cutout_pos, cutout_size=size_of_cutout)


figure = plt.figure(figsize=(30, 10))
fontsize = 33
# sed axis
ax_sed = figure.add_axes([0.06, 0.09, 0.935, 0.905])

# postage stamps axis
ax_f275w_ps = figure.add_axes([0.02 + 0 * 0.065, 0.71, 0.19, 0.19], projection=f275w_cutout.wcs)
ax_f336w_ps = figure.add_axes([0.02 + 1 * 0.065, 0.71, 0.19, 0.19], projection=f336w_cutout.wcs)
ax_f435w_ps = figure.add_axes([0.02 + 2 * 0.065, 0.71, 0.19, 0.19], projection=f435w_cutout.wcs)
ax_f555w_ps = figure.add_axes([0.02 + 3 * 0.065, 0.71, 0.19, 0.19], projection=f555w_cutout.wcs)
ax_f814w_ps = figure.add_axes([0.02 + 4 * 0.065, 0.71, 0.19, 0.19], projection=f814w_cutout.wcs)

ax_f200w_ps = figure.add_axes([0.37 + 0 * 0.065, 0.71, 0.19, 0.19], projection=f200w_cutout.wcs)
ax_f300m_ps = figure.add_axes([0.37 + 1 * 0.065, 0.71, 0.19, 0.19], projection=f300m_cutout.wcs)
ax_f335m_ps = figure.add_axes([0.37 + 2 * 0.065, 0.71, 0.19, 0.19], projection=f335m_cutout.wcs)
ax_f360m_ps = figure.add_axes([0.37 + 3 * 0.065, 0.71, 0.19, 0.19], projection=f360m_cutout.wcs)

ax_f770w_ps = figure.add_axes([0.65 + 0 * 0.065, 0.71, 0.19, 0.19], projection=f770w_cutout.wcs)
ax_f1000w_ps = figure.add_axes([0.65 + 1 * 0.065, 0.71, 0.19, 0.19], projection=f1000w_cutout.wcs)
ax_f1130w_ps = figure.add_axes([0.65 + 2 * 0.065, 0.71, 0.19, 0.19], projection=f1130w_cutout.wcs)
ax_f2100w_ps = figure.add_axes([0.65 + 3 * 0.065, 0.71, 0.19, 0.19], projection=f2100w_cutout.wcs)

# plot the cutouts
plot_postage_stamps(ax=ax_f275w_ps, cutout=f275w_cutout, title='F275W', filter_color='k', fontsize=fontsize, show_ax_label=True)
plot_postage_stamps(ax=ax_f336w_ps, cutout=f336w_cutout, title='F336W', filter_color='k', fontsize=fontsize)
plot_postage_stamps(ax=ax_f435w_ps, cutout=f435w_cutout, title='F435W', filter_color='k', fontsize=fontsize)
plot_postage_stamps(ax=ax_f555w_ps, cutout=f555w_cutout, title='F555W', filter_color='k', fontsize=fontsize)
plot_postage_stamps(ax=ax_f814w_ps, cutout=f814w_cutout, title='F814W', filter_color='k', fontsize=fontsize)

plot_postage_stamps(ax=ax_f200w_ps, cutout=f200w_cutout, title='F200W', filter_color='tab:blue', fontsize=fontsize)
plot_postage_stamps(ax=ax_f300m_ps, cutout=f300m_cutout, title='F300M', filter_color='tab:orange', fontsize=fontsize)
plot_postage_stamps(ax=ax_f335m_ps, cutout=f335m_cutout, title='F335M', filter_color='tab:green', fontsize=fontsize)
plot_postage_stamps(ax=ax_f360m_ps, cutout=f360m_cutout, title='F360M', filter_color='tab:red', fontsize=fontsize)

plot_postage_stamps(ax=ax_f770w_ps, cutout=f770w_cutout, title='F770W', filter_color='tab:purple', fontsize=fontsize)
plot_postage_stamps(ax=ax_f1000w_ps, cutout=f1000w_cutout, title='F1000W', filter_color='tab:brown', fontsize=fontsize)
plot_postage_stamps(ax=ax_f1130w_ps, cutout=f1130w_cutout, title='F1130W', filter_color='tab:pink', fontsize=fontsize)
plot_postage_stamps(ax=ax_f2100w_ps, cutout=f2100w_cutout, title='F2100W', filter_color='tab:gray', fontsize=fontsize)

# plot the circles
plot_coord_circle(ax=ax_f275w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f336w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f435w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f555w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f814w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)

plot_coord_circle(ax=ax_f200w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f300m_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f335m_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f360m_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)

plot_coord_circle(ax=ax_f770w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f1000w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f1130w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)
plot_coord_circle(ax=ax_f2100w_ps, pos=cutout_pos, rad=0.1, color='r', linewidth=2)


plt.savefig('plot_output/test_sed_plot.png')
