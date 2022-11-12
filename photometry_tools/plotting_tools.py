"""
Plotting tools for photometry and SED fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, LogNorm

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle


class PlotPhotometry:
    """
    class to plot all kinds of photometric presentations
    """

    @staticmethod
    def plot_cutout_panel_hst_nircam_miri(hst_band_list, nircam_band_list, miri_band_list, cutout_dict,
                                          circ_pos=None, circ_rad=None, circ_color=None,
                                          fontsize=18, figsize=(18, 13),
                                          vmax_vmin_hst=None, vmax_vmin_nircam=None, vmax_vmin_miri=None,
                                          ticks_hst=None, ticks_nircam=None, ticks_miri=None,
                                          cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
                                          log_scale=False, axis_length=None,
                                          ra_tick_num=3, dec_tick_num=3):


        norm_hst = compute_cbar_norm(vmax_vmin=vmax_vmin_hst,
                                     cutout_list=[cutout_dict['%s_img_cutout' % band].data for band in hst_band_list],
                                     log_scale=log_scale)
        norm_nircam = compute_cbar_norm(vmax_vmin=vmax_vmin_nircam,
                                        cutout_list=[cutout_dict['%s_img_cutout' % band].data
                                                     for band in nircam_band_list], log_scale=log_scale)
        norm_miri = compute_cbar_norm(vmax_vmin=vmax_vmin_miri,
                                      cutout_list=[cutout_dict['%s_img_cutout' % band].data for band in miri_band_list],
                                      log_scale=log_scale)
        if log_scale:
            cbar_label = r'log(S /[MJy / sr])'
        else:
            cbar_label = r'S [MJy / sr]'

        # build up a figure
        figure = plt.figure(figsize=figsize)

        ax_color_bar_hst = figure.add_axes([0.925, 0.68, 0.015, 0.28])
        ax_color_bar_nircam = figure.add_axes([0.925, 0.375, 0.015, 0.28])
        ax_color_bar_miri = figure.add_axes([0.925, 0.07, 0.015, 0.28])

        for hst_band, index in zip(hst_band_list, range(len(hst_band_list))):
            ax = figure.add_axes([0.01 + index * 0.176, 0.70, 0.24, 0.24],
                                 projection=cutout_dict['%s_img_cutout' % hst_band].wcs)

            ax.imshow(cutout_dict['%s_img_cutout' % hst_band].data, norm=norm_hst, cmap=cmap_hst)
            set_lim2cutout(ax=ax, cutout=cutout_dict['%s_img_cutout' % hst_band], cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            if circ_pos:
                if isinstance(circ_pos, list):
                    for pos, rad, color in zip(circ_pos, circ_rad, circ_color):
                        plot_coord_circle(ax, position=pos, radius=rad*u.arcsec, color=color)
                else:
                    plot_coord_circle(ax, position=circ_pos, radius=circ_rad*u.arcsec, color=circ_color)
            data_shape = cutout_dict['%s_img_cutout' % hst_band].data.shape
            ax.text(data_shape[0] * 0.1, data_shape[1] * 0.85, hst_band, color='k', fontsize=fontsize+2)
            ax.tick_params(axis='both', which='both', width=1.5, length=7, direction='in', color='k', labelsize=fontsize-1)
            if index == 0:
                ax.coords['dec'].set_ticklabel(rotation=90)
                ax.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)
            else:
                ax.coords['dec'].set_ticklabel_visible(False)
                ax.coords['dec'].set_axislabel(' ')
            ax.coords['ra'].set_ticklabel_visible(False)
            ax.coords['ra'].set_axislabel(' ')
            ax.coords['ra'].set_ticks(number=ra_tick_num)
            ax.coords['ra'].display_minor_ticks(True)
            ax.coords['dec'].set_ticks(number=dec_tick_num)
            ax.coords['dec'].display_minor_ticks(True)

        create_cbar(ax_cbar=ax_color_bar_hst, cmap=cmap_hst, norm=norm_hst, cbar_label=cbar_label, fontsize=fontsize,
                    ticks=ticks_hst)

        for nircam_band, index in zip(nircam_band_list, range(len(nircam_band_list))):
            ax = figure.add_axes([0.0 + index * 0.22, 0.365, 0.3, 0.3],
                                 projection=cutout_dict['%s_img_cutout' % nircam_band].wcs)
            ax.imshow(cutout_dict['%s_img_cutout' % nircam_band].data, norm=norm_nircam, cmap=cmap_nircam)
            if circ_pos:
                if isinstance(circ_pos, list):
                    for pos, rad, color in zip(circ_pos, circ_rad, circ_color):
                        plot_coord_circle(ax, position=pos, radius=rad*u.arcsec, color=color)
                else:
                    plot_coord_circle(ax, position=circ_pos, radius=circ_rad*u.arcsec, color=circ_color)
            data_shape = cutout_dict['%s_img_cutout' % nircam_band].data.shape
            ax.text(data_shape[0] * 0.1, data_shape[1] * 0.85, nircam_band, color='k', fontsize=fontsize+2)
            ax.tick_params(axis='both', which='both', width=1.5, length=7, direction='in', color='k', labelsize=fontsize-1)
            set_lim2cutout(ax=ax, cutout=cutout_dict['%s_img_cutout' % nircam_band], cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            if index == 0:
                ax.coords['dec'].set_ticklabel(rotation=90)
                ax.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)
            else:
                ax.coords['dec'].set_ticklabel_visible(False)
                ax.coords['dec'].set_axislabel(' ')
            ax.coords['ra'].set_ticklabel_visible(False)
            ax.coords['ra'].set_axislabel(' ')
            ax.coords['ra'].set_ticks(number=ra_tick_num)
            ax.coords['ra'].display_minor_ticks(True)
            ax.coords['dec'].set_ticks(number=dec_tick_num)
            ax.coords['dec'].display_minor_ticks(True)
        create_cbar(ax_cbar=ax_color_bar_nircam, cmap=cmap_nircam, norm=norm_nircam, cbar_label=cbar_label, fontsize=fontsize,
                    ticks=ticks_nircam, labelpad=2, tick_width=2, orientation='vertical', extend='neither')

        for miri_band, index in zip(miri_band_list, range(len(miri_band_list))):
            ax = figure.add_axes([0.0 + index * 0.22, 0.06, 0.3, 0.3],
                                 projection=cutout_dict['%s_img_cutout' % miri_band].wcs)
            ax.imshow(cutout_dict['%s_img_cutout' % miri_band].data, norm=norm_miri, cmap=cmap_miri)
            if circ_pos:
                if isinstance(circ_pos, list):
                    for pos, rad, color in zip(circ_pos, circ_rad, circ_color):
                        plot_coord_circle(ax, position=pos, radius=rad*u.arcsec, color=color)
                else:
                    plot_coord_circle(ax, position=circ_pos, radius=circ_rad*u.arcsec, color=circ_color)
            data_shape = cutout_dict['%s_img_cutout' % miri_band].data.shape
            ax.text(data_shape[0] * 0.1, data_shape[1] * 0.85, miri_band, color='k', fontsize=fontsize+2)
            ax.tick_params(axis='both', which='both', width=1.5, length=7, direction='in', color='k', labelsize=fontsize-1)
            set_lim2cutout(ax=ax, cutout=cutout_dict['%s_img_cutout' % miri_band], cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            if index == 0:
                ax.coords['dec'].set_ticklabel(rotation=90)
                ax.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)
            else:
                ax.coords['dec'].set_ticklabel_visible(False)
                ax.coords['dec'].set_axislabel(' ')
            ax.coords['ra'].set_ticklabel(rotation=0)
            ax.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)
            ax.coords['ra'].set_ticks(number=ra_tick_num)
            ax.coords['ra'].display_minor_ticks(True)
            ax.coords['dec'].set_ticks(number=dec_tick_num)
            ax.coords['dec'].display_minor_ticks(True)

        create_cbar(ax_cbar=ax_color_bar_miri, cmap=cmap_miri, norm=norm_miri, cbar_label=cbar_label, fontsize=fontsize,
                    ticks=ticks_miri, labelpad=2, tick_width=1.5, orientation='vertical', extend='neither')

        return figure


def compute_cbar_norm(vmax_vmin=None, cutout_list=None, log_scale=False):
    """
    Computing the color bar scale for a single or multiple cutouts.

    Parameters
    ----------
    vmax_vmin : tuple
    cutout_list : list
        This list should include all cutouts
    log_scale : bool

    Returns
    -------
    norm : ``matplotlib.colors.Normalize``  or ``matplotlib.colors.LogNorm``
    """
    if (vmax_vmin is None) & (cutout_list is None):
        raise KeyError('either vmax_vmin or cutout_list must be not None')

    # get maximal value
    if vmax_vmin is None:
        list_of_means = [np.mean(cutout) for cutout in cutout_list]
        list_of_stds = [np.std(cutout) for cutout in cutout_list]
        mean, std = (np.mean(list_of_means), np.std(list_of_stds))

        vmin = mean - 1 * std
        vmax = mean + 10 * std
    else:
        vmin, vmax = vmax_vmin[0], vmax_vmin[1]
    if log_scale:
        if vmin < 0:
            vmin = vmax / 100
        norm = LogNorm(vmin, vmax)
    else:
        norm = Normalize(vmin, vmax)
    return norm


def create_cbar(ax_cbar, cmap, norm, cbar_label, fontsize, ticks=None, labelpad=2, tick_width=2, orientation='vertical',
                extend='neither'):
    """

    Parameters
    ----------
    ax_cbar : ``matplotlib.pylab.axis``
    cmap : str
        same as name parameter of ``matplotlib.colors.Colormap.name``
    norm : ``matplotlib.colors.Normalize``  or ``matplotlib.colors.LogNorm``
    cbar_label : str
    fontsize : int or float
    ticks : list
    labelpad : int or float
    tick_width : int or float
    orientation : str
        default is `vertical`
    extend : str
        default is 'neither'
        can be 'neither', 'min' , 'max' or 'both'
    """
    ColorbarBase(ax_cbar, orientation=orientation, cmap=cmap, norm=norm,   extend=extend, ticks=ticks)
    ax_cbar.set_ylabel(cbar_label, labelpad=labelpad, fontsize=fontsize)
    ax_cbar.tick_params(axis='both', which='both', width=tick_width, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)


def set_lim2cutout(ax, cutout, cutout_pos, ra_length, dec_length):

    lim_top_left = cutout.wcs.world_to_pixel(SkyCoord(cutout_pos.ra + ra_length/2*u.arcsec,
                                                      cutout_pos.dec + dec_length/2*u.arcsec))
    lim_bottom_right = cutout.wcs.world_to_pixel(SkyCoord(cutout_pos.ra - ra_length/2*u.arcsec,
                                                          cutout_pos.dec - dec_length/2*u.arcsec))
    ax.set_xlim(lim_top_left[0], lim_bottom_right[0])
    ax.set_ylim(lim_bottom_right[1], lim_top_left[1])


def plot_coord_circle(ax, position, radius, color='g', linestyle='-', linewidth=3, alpha=1., fill=False):
    if fill:
        facecolor = color
    else:
        facecolor = 'none'
    circle = SphericalCircle(position, radius,
                             edgecolor=color, facecolor=facecolor, linewidth=linewidth, linestyle=linestyle,
                             alpha=alpha, transform=ax.get_transform('icrs'))
    ax.add_patch(circle)


def draw_box(ax, wcs, coord, box_size, color='k', linewidth=2):

    if isinstance(box_size, tuple):
        box_size = box_size * u.arcsec
    elif isinstance(box_size, float) | isinstance(box_size, int):
        box_size = (box_size, box_size) * u.arcsec
    else:
        raise KeyError('cutout_size must be float or tuple')

    top_left_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra + box_size[0] / 2, dec=coord.dec + box_size[1] / 2))
    top_right_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra + box_size[0] / 2, dec=coord.dec - box_size[1] / 2))
    bottom_left_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra - box_size[0] / 2, dec=coord.dec + box_size[1] / 2))
    bottom_right_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra - box_size[0] / 2, dec=coord.dec - box_size[1] / 2))

    ax.plot([top_left_pix[0], top_right_pix[0]], [top_left_pix[1], top_right_pix[1]], color=color, linewidth=linewidth)
    ax.plot([bottom_left_pix[0], bottom_right_pix[0]], [bottom_left_pix[1], bottom_right_pix[1]], color=color,
            linewidth=linewidth)
    ax.plot([top_left_pix[0], bottom_left_pix[0]], [top_left_pix[1], bottom_left_pix[1]], color=color,
            linewidth=linewidth)
    ax.plot([top_right_pix[0], bottom_right_pix[0]], [top_right_pix[1], bottom_right_pix[1]], color=color,
            linewidth=linewidth)

