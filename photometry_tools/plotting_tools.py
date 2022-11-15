"""
Plotting tools for photometry and SED fitting
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Ellipse

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

        # get figure
        figure = plt.figure(figsize=figsize)
        # arrange axis
        axis_dict = add_axis_hst_nircam_miri(figure=figure, hst_band_list=hst_band_list,
                                             nircam_band_list=nircam_band_list, miri_band_list=miri_band_list,
                                             cutout_dict=cutout_dict)

        for hst_band, index in zip(hst_band_list, range(len(hst_band_list))):
            # plot data
            axis_dict['ax_%s' % hst_band].imshow(cutout_dict['%s_img_cutout' % hst_band].data,
                                                 norm=norm_hst, cmap=cmap_hst)
            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % hst_band], cutout=cutout_dict['%s_img_cutout' % hst_band], cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            # plot circles
            if circ_pos is not None:
                plot_coord_circle(ax=axis_dict['ax_%s' % hst_band], pos=circ_pos, rad=circ_rad,
                                  color=circ_color)
            # add text
            arrange_text(ax=axis_dict['ax_%s' % hst_band],
                         data_shape=cutout_dict['%s_img_cutout' % hst_band].data.shape, text=hst_band,
                         fontsize=fontsize-2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % hst_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize-1, ra_tick_num=ra_tick_num, dec_tick_num=dec_tick_num)
        # arrange color bar
        create_cbar(ax_cbar=axis_dict['ax_color_bar_hst'], cmap=cmap_hst, norm=norm_hst, cbar_label=cbar_label,
                    fontsize=fontsize, ticks=ticks_hst)

        for nircam_band, index in zip(nircam_band_list, range(len(nircam_band_list))):
            # plot data
            axis_dict['ax_%s' % nircam_band].imshow(cutout_dict['%s_img_cutout' % nircam_band].data,
                                                 norm=norm_nircam, cmap=cmap_nircam)
            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % nircam_band], cutout=cutout_dict['%s_img_cutout' % nircam_band], cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            # plot circles
            if circ_pos is not None:
                plot_coord_circle(ax=axis_dict['ax_%s' % nircam_band], pos=circ_pos, rad=circ_rad,
                                  color=circ_color)
            # add text
            arrange_text(ax=axis_dict['ax_%s' % nircam_band],
                         data_shape=cutout_dict['%s_img_cutout' % nircam_band].data.shape, text=nircam_band,
                         fontsize=fontsize-2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % nircam_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize-1, ra_tick_num=ra_tick_num, dec_tick_num=dec_tick_num)
        # arrange color bar
        create_cbar(ax_cbar=axis_dict['ax_color_bar_nircam'], cmap=cmap_nircam, norm=norm_nircam, cbar_label=cbar_label,
                    fontsize=fontsize, ticks=ticks_nircam)

        for miri_band, index in zip(miri_band_list, range(len(miri_band_list))):
            # plot data
            axis_dict['ax_%s' % miri_band].imshow(cutout_dict['%s_img_cutout' % miri_band].data,
                                                 norm=norm_miri, cmap=cmap_miri)
            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % miri_band], cutout=cutout_dict['%s_img_cutout' % miri_band], cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            # plot circles
            if circ_pos is not None:
                plot_coord_circle(ax=axis_dict['ax_%s' % miri_band], pos=circ_pos, rad=circ_rad,
                                  color=circ_color)
            # add text
            arrange_text(ax=axis_dict['ax_%s' % miri_band],
                         data_shape=cutout_dict['%s_img_cutout' % miri_band].data.shape, text=miri_band,
                         fontsize=fontsize-2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % miri_band], dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize-1, ra_tick_num=ra_tick_num, dec_tick_num=dec_tick_num)
        # arrange color bar
        create_cbar(ax_cbar=axis_dict['ax_color_bar_miri'], cmap=cmap_miri, norm=norm_miri, cbar_label=cbar_label,
                    fontsize=fontsize, ticks=ticks_miri)

        return figure

    @staticmethod
    def plot_circ_flux_extraction(hst_band_list, nircam_band_list, miri_band_list, cutout_dict, aperture_dict,
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

        # get figure
        figure = plt.figure(figsize=figsize)
        # arrange axis
        axis_dict = add_axis_hst_nircam_miri(figure=figure, hst_band_list=hst_band_list,
                                             nircam_band_list=nircam_band_list, miri_band_list=miri_band_list,
                                             cutout_dict=cutout_dict)

        for hst_band, index in zip(hst_band_list, range(len(hst_band_list))):
            # plot data
            axis_dict['ax_%s' % hst_band].imshow(cutout_dict['%s_img_cutout' % hst_band].data,
                                                 norm=norm_hst, cmap=cmap_hst)

            # detected sources
            for i in range(len(aperture_dict['aperture_dict_%s' % hst_band]['source_table'])):
                e = Ellipse(xy=(aperture_dict['aperture_dict_%s' % hst_band]['source_table']['x'][i],
                                aperture_dict['aperture_dict_%s' % hst_band]['source_table']['y'][i]),
                            width=3*aperture_dict['aperture_dict_%s' % hst_band]['source_table']['a'][i],
                            height=3*aperture_dict['aperture_dict_%s' % hst_band]['source_table']['b'][i],
                            angle=
                            aperture_dict['aperture_dict_%s' % hst_band]['source_table']['theta'][i]*180./np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('green')
                axis_dict['ax_%s' % hst_band].add_artist(e)

            # plot aperture search
            if aperture_dict['recenter']:
                # radius to search peak
                plot_coord_circle(ax=axis_dict['ax_%s' % hst_band], pos=aperture_dict['init_pos'],
                                  rad=aperture_dict['recenter_rad'], color='b', linestyle='--', linewidth=1, alpha=0.6)
            plot_coord_circle(ax=axis_dict['ax_%s' % hst_band],
                              pos=aperture_dict['aperture_dict_%s' % hst_band]['new_pos'],
                              rad=aperture_dict['aperture_rad_dict']['aperture_%s' % hst_band],
                              color='r', linewidth=2, alpha=1)

            axis_dict['ax_%s' % hst_band].scatter(aperture_dict['aperture_dict_%s' % hst_band]['new_pos'].ra,
                                                  aperture_dict['aperture_dict_%s' % hst_band]['new_pos'].dec,
                                                  transform=axis_dict['ax_%s' % hst_band].get_transform('fk5'),
                                                  marker='x', color='r')
            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % hst_band], cutout=cutout_dict['%s_img_cutout' % hst_band],
                           cutout_pos=cutout_dict['cutout_pos'], ra_length=axis_length[0], dec_length=axis_length[1])

            # add text
            arrange_text(ax=axis_dict['ax_%s' % hst_band],
                         data_shape=cutout_dict['%s_img_cutout' % hst_band].data.shape, text=hst_band,
                         fontsize=fontsize-2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % hst_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize-1, ra_tick_num=ra_tick_num, dec_tick_num=dec_tick_num)
        # arrange color bar
        create_cbar(ax_cbar=axis_dict['ax_color_bar_hst'], cmap=cmap_hst, norm=norm_hst, cbar_label=cbar_label,
                    fontsize=fontsize, ticks=ticks_hst)

        for nircam_band, index in zip(nircam_band_list, range(len(nircam_band_list))):
            # plot data
            axis_dict['ax_%s' % nircam_band].imshow(cutout_dict['%s_img_cutout' % nircam_band].data,
                                                 norm=norm_nircam, cmap=cmap_nircam)

            # detected sources
            for i in range(len(aperture_dict['aperture_dict_%s' % nircam_band]['source_table'])):
                e = Ellipse(xy=(aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['x'][i],
                                aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['y'][i]),
                            width=3*aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['a'][i],
                            height=3*aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['b'][i],
                            angle=
                            aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['theta'][i]*180./np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('green')
                axis_dict['ax_%s' % nircam_band].add_artist(e)

            # plot aperture search
            if aperture_dict['recenter']:
                # radius to search peak
                plot_coord_circle(ax=axis_dict['ax_%s' % nircam_band], pos=aperture_dict['init_pos'],
                                  rad=aperture_dict['recenter_rad'], color='b', linestyle='--', linewidth=1, alpha=0.6)
            plot_coord_circle(ax=axis_dict['ax_%s' % nircam_band],
                              pos=aperture_dict['aperture_dict_%s' % nircam_band]['new_pos'],
                              rad=aperture_dict['aperture_rad_dict']['aperture_%s' % nircam_band],
                              color='r', linewidth=2, alpha=1)

            axis_dict['ax_%s' % nircam_band].scatter(aperture_dict['aperture_dict_%s' % nircam_band]['new_pos'].ra,
                                                     aperture_dict['aperture_dict_%s' % nircam_band]['new_pos'].dec,
                                                     transform=axis_dict['ax_%s' % nircam_band].get_transform('fk5'),
                                                     marker='x', color='r')

            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % nircam_band], cutout=cutout_dict['%s_img_cutout' % nircam_band],
                           cutout_pos=cutout_dict['cutout_pos'], ra_length=axis_length[0], dec_length=axis_length[1])

            # add text
            arrange_text(ax=axis_dict['ax_%s' % nircam_band],
                         data_shape=cutout_dict['%s_img_cutout' % nircam_band].data.shape, text=nircam_band,
                         fontsize=fontsize-2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % nircam_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize-1, ra_tick_num=ra_tick_num, dec_tick_num=dec_tick_num)
        # arrange color bar
        create_cbar(ax_cbar=axis_dict['ax_color_bar_nircam'], cmap=cmap_nircam, norm=norm_nircam, cbar_label=cbar_label,
                    fontsize=fontsize, ticks=ticks_nircam)

        for miri_band, index in zip(miri_band_list, range(len(miri_band_list))):
            # plot data
            axis_dict['ax_%s' % miri_band].imshow(cutout_dict['%s_img_cutout' % miri_band].data,
                                                  norm=norm_miri, cmap=cmap_miri)

            # detected sources
            for i in range(len(aperture_dict['aperture_dict_%s' % miri_band]['source_table'])):
                e = Ellipse(xy=(aperture_dict['aperture_dict_%s' % miri_band]['source_table']['x'][i],
                                aperture_dict['aperture_dict_%s' % miri_band]['source_table']['y'][i]),
                            width=3*aperture_dict['aperture_dict_%s' % miri_band]['source_table']['a'][i],
                            height=3*aperture_dict['aperture_dict_%s' % miri_band]['source_table']['b'][i],
                            angle=
                            aperture_dict['aperture_dict_%s' % miri_band]['source_table']['theta'][i]*180./np.pi)
                e.set_facecolor('none')
                e.set_edgecolor('green')
                axis_dict['ax_%s' % miri_band].add_artist(e)

            # plot aperture search
            if aperture_dict['recenter']:
                # radius to search peak
                plot_coord_circle(ax=axis_dict['ax_%s' % miri_band], pos=aperture_dict['init_pos'],
                                  rad=aperture_dict['recenter_rad'], color='b', linestyle='--', linewidth=1, alpha=0.6)
            plot_coord_circle(ax=axis_dict['ax_%s' % miri_band],
                              pos=aperture_dict['aperture_dict_%s' % miri_band]['new_pos'],
                              rad=aperture_dict['aperture_rad_dict']['aperture_%s' % miri_band],
                              color='r', linewidth=2, alpha=1)

            axis_dict['ax_%s' % miri_band].scatter(aperture_dict['aperture_dict_%s' % miri_band]['new_pos'].ra,
                                                   aperture_dict['aperture_dict_%s' % miri_band]['new_pos'].dec,
                                                   transform=axis_dict['ax_%s' % miri_band].get_transform('fk5'),
                                                   marker='x', color='r')

            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % miri_band], cutout=cutout_dict['%s_img_cutout' % miri_band],
                           cutout_pos=cutout_dict['cutout_pos'], ra_length=axis_length[0], dec_length=axis_length[1])
            # plot circles
            if circ_pos is not None:
                plot_coord_circle(ax=axis_dict['ax_%s' % miri_band], pos=circ_pos, rad=circ_rad,
                                  color=circ_color)
            # add text
            arrange_text(ax=axis_dict['ax_%s' % miri_band],
                         data_shape=cutout_dict['%s_img_cutout' % miri_band].data.shape, text=miri_band,
                         fontsize=fontsize-2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % miri_band], dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize-1, ra_tick_num=ra_tick_num, dec_tick_num=dec_tick_num)
        # arrange color bar
        create_cbar(ax_cbar=axis_dict['ax_color_bar_miri'], cmap=cmap_miri, norm=norm_miri, cbar_label=cbar_label,
                    fontsize=fontsize, ticks=ticks_miri)

        return figure


def add_axis_hst_nircam_miri(figure, hst_band_list, nircam_band_list, miri_band_list, cutout_dict, cbar=True):

    axis_dict = {}
    if cbar:
        ax_color_bar_hst = figure.add_axes([0.925, 0.68, 0.015, 0.28])
        ax_color_bar_nircam = figure.add_axes([0.925, 0.375, 0.015, 0.28])
        ax_color_bar_miri = figure.add_axes([0.925, 0.07, 0.015, 0.28])
        axis_dict.update({'ax_color_bar_hst': ax_color_bar_hst})
        axis_dict.update({'ax_color_bar_nircam': ax_color_bar_nircam})
        axis_dict.update({'ax_color_bar_miri': ax_color_bar_miri})

    for hst_band, index in zip(hst_band_list, range(len(hst_band_list))):
        ax = figure.add_axes([0.01 + index * 0.176, 0.70, 0.24, 0.24],
                             projection=cutout_dict['%s_img_cutout' % hst_band].wcs)
        axis_dict.update({'ax_%s' % hst_band: ax})

    for nircam_band, index in zip(nircam_band_list, range(len(nircam_band_list))):
        ax = figure.add_axes([0.0 + index * 0.22, 0.365, 0.3, 0.3],
                             projection=cutout_dict['%s_img_cutout' % nircam_band].wcs)
        axis_dict.update({'ax_%s' % nircam_band: ax})

    for miri_band, index in zip(miri_band_list, range(len(miri_band_list))):
        ax = figure.add_axes([0.0 + index * 0.22, 0.06, 0.3, 0.3],
                             projection=cutout_dict['%s_img_cutout' % miri_band].wcs)
        axis_dict.update({'ax_%s' % miri_band: ax})

    return axis_dict


def arrange_text(ax, data_shape, text, axis_offset_x=0.1, axis_offset_y=0.85, fontsize=15, color='k'):
    ax.text(data_shape[0] * axis_offset_x, data_shape[1] * axis_offset_y, text, color=color, fontsize=fontsize+2)


def arr_axis_params(ax, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                    ra_minpad=0.8, dec_minpad=0.8,
                    fontsize=15, labelsize=14, ra_tick_num=3, dec_tick_num=3):

    ax.tick_params(axis='both', which='both', width=1.5, length=7, direction='in', color='k', labelsize=labelsize)

    if not ra_tick_label:
        ax.coords['ra'].set_ticklabel_visible(False)
        ax.coords['ra'].set_axislabel(' ')
    else:
        ax.coords['ra'].set_axislabel(ra_axis_label, minpad=ra_minpad, fontsize=fontsize)

    if not dec_tick_label:
        ax.coords['dec'].set_ticklabel_visible(False)
        ax.coords['dec'].set_axislabel(' ')
    else:
        ax.coords['dec'].set_ticklabel(rotation=90)
        ax.coords['dec'].set_axislabel(dec_axis_label, minpad=dec_minpad, fontsize=fontsize)

    ax.coords['ra'].set_ticks(number=ra_tick_num)
    ax.coords['ra'].display_minor_ticks(True)
    ax.coords['dec'].set_ticks(number=dec_tick_num)
    ax.coords['dec'].display_minor_ticks(True)


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


def plot_coord_circle(ax, pos, rad, color, linestyle='-', linewidth=3, alpha=1., fill=False):

    if fill:
        facecolor = color
    else:
        facecolor = 'none'

    if isinstance(pos, list):
        if not isinstance(rad, list):
            rad = [rad] * len(pos)
        if not isinstance(color, list):
            color = [color] * len(pos)
        if not isinstance(linestyle, list):
            linestyle = [linestyle] * len(pos)
        if not isinstance(linewidth, list):
            linewidth = [linewidth] * len(pos)
        if not isinstance(alpha, list):
            alpha = [alpha] * len(pos)
        for pos_i, rad_i, color_i, linestyle_i, linewidth_i, alpha_i in zip(pos, rad, color, linestyle, linewidth,
                                                                            alpha):
            circle = SphericalCircle(pos_i, rad_i*u.arcsec, edgecolor=color_i, facecolor=facecolor, linewidth=linewidth_i,
                                     linestyle=linestyle_i, alpha=alpha_i, transform=ax.get_transform('fk5'))
            ax.add_patch(circle)
    else:
        circle = SphericalCircle(pos, rad*u.arcsec, edgecolor=color, facecolor=facecolor, linewidth=linewidth,
                                 linestyle=linestyle, alpha=alpha, transform=ax.get_transform('fk5'))
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

