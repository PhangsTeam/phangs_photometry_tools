"""
Plotting tools for photometry and SED fitting
"""

import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize, LogNorm
from matplotlib.patches import Ellipse
from matplotlib import transforms
import matplotlib.image as mpimg

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization.wcsaxes import SphericalCircle

from astropy.visualization import make_lupton_rgb
from reproject import reproject_interp
from astropy.io import fits
from photometry_tools import helper_func


class PlotPhotometry:
    """
    class to plot all kinds of photometric presentations
    """

    @staticmethod
    def plot_cutout_panel_hst_nircam_miri(hst_band_list, nircam_band_list, miri_band_list, cutout_dict,
                                          circ_pos=None, circ_rad=None, circ_color=None,
                                          fontsize=18, figsize=(18, 13),
                                          vmin_vmax_hst=None, vmin_vmax_nircam=None, vmin_vmax_miri=None,
                                          ticks_hst=None, ticks_nircam=None, ticks_miri=None,
                                          cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
                                          log_scale=False, axis_length=None,
                                          ra_tick_num=3, dec_tick_num=3):

        norm_hst = compute_cbar_norm(vmin_vmax=vmin_vmax_hst,
                                     cutout_list=[cutout_dict['%s_img_cutout' % band].data for band in hst_band_list],
                                     log_scale=log_scale)
        norm_nircam = compute_cbar_norm(vmin_vmax=vmin_vmax_nircam,
                                        cutout_list=[cutout_dict['%s_img_cutout' % band].data
                                                     for band in nircam_band_list], log_scale=log_scale)
        norm_miri = compute_cbar_norm(vmin_vmax=vmin_vmax_miri,
                                      cutout_list=[cutout_dict['%s_img_cutout' % band].data for band in miri_band_list],
                                      log_scale=log_scale)
        if log_scale:
            cbar_label = r'log(S /[MJy / sr])'
        else:
            cbar_label = r'S [MJy / sr]'

        # get figure
        figure = plt.figure(figsize=figsize)
        # arrange axis
        axis_dict = add_axis_hst_nircam_miri_panel(figure=figure, hst_band_list=hst_band_list,
                                                   nircam_band_list=nircam_band_list, miri_band_list=miri_band_list,
                                                   cutout_dict=cutout_dict)

        for hst_band, index in zip(hst_band_list, range(len(hst_band_list))):
            # plot data
            if ((not np.isnan(norm_hst.vmin)) and (not np.isnan(norm_hst.vmax)) and
                    (np.sum(cutout_dict['%s_img_cutout' % hst_band].data) != 0)):
                axis_dict['ax_%s' % hst_band].imshow(cutout_dict['%s_img_cutout' % hst_band].data,
                                                     norm=norm_hst, cmap=cmap_hst)
            else:
                axis_dict['ax_%s' % hst_band].cla()
                continue

            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % hst_band], cutout=cutout_dict['%s_img_cutout' % hst_band],
                           cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])

            # plot circles
            if circ_pos is not None:
                plot_coord_circle(ax=axis_dict['ax_%s' % hst_band], pos=circ_pos, rad=circ_rad,
                                  color=circ_color)
            # add text
            arrange_text(ax=axis_dict['ax_%s' % hst_band],
                         data_shape=cutout_dict['%s_img_cutout' % hst_band].data.shape, text=hst_band,
                         fontsize=fontsize - 2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % hst_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize - 1, ra_tick_num=ra_tick_num,
                            dec_tick_num=dec_tick_num)
        # arrange color bar
        if ((not np.isnan(norm_hst.vmin) and not np.isnan(norm_hst.vmax)) and
                (norm_hst.vmin != norm_hst.vmax)):
            create_cbar(ax_cbar=axis_dict['ax_color_bar_hst'], cmap=cmap_hst, norm=norm_hst, cbar_label=cbar_label,
                        fontsize=fontsize, ticks=ticks_hst)
        else:
            axis_dict['ax_color_bar_hst'].cla()
        for nircam_band, index in zip(nircam_band_list, range(len(nircam_band_list))):
            # plot data
            if ((not np.isnan(norm_nircam.vmin)) and (not np.isnan(norm_nircam.vmax)) and
                    (np.sum(cutout_dict['%s_img_cutout' % nircam_band].data) != 0)):
                axis_dict['ax_%s' % nircam_band].imshow(cutout_dict['%s_img_cutout' % nircam_band].data,
                                                     norm=norm_nircam, cmap=cmap_nircam)
            else:
                axis_dict['ax_%s' % nircam_band].cla()
                continue
            # # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % nircam_band], cutout=cutout_dict['%s_img_cutout' % nircam_band],
                           cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            # plot circles
            if circ_pos is not None:
                plot_coord_circle(ax=axis_dict['ax_%s' % nircam_band], pos=circ_pos, rad=circ_rad,
                                  color=circ_color)
            # add text
            arrange_text(ax=axis_dict['ax_%s' % nircam_band],
                         data_shape=cutout_dict['%s_img_cutout' % nircam_band].data.shape, text=nircam_band,
                         fontsize=fontsize - 2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % nircam_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize - 1, ra_tick_num=ra_tick_num,
                            dec_tick_num=dec_tick_num)
        # arrange color bar
        if ((not np.isnan(norm_nircam.vmin) and not np.isnan(norm_nircam.vmax)) and
                (norm_nircam.vmin != norm_nircam.vmax)):
            create_cbar(ax_cbar=axis_dict['ax_color_bar_nircam'], cmap=cmap_nircam, norm=norm_nircam, cbar_label=cbar_label,
                        fontsize=fontsize, ticks=ticks_nircam)
        else:
            axis_dict['ax_color_bar_nircam'].cla()

        for miri_band, index in zip(miri_band_list, range(len(miri_band_list))):

            # plot data
            if ((not np.isnan(norm_miri.vmin)) and (not np.isnan(norm_miri.vmax)) and
                    (np.sum(cutout_dict['%s_img_cutout' % miri_band].data) != 0)):
                axis_dict['ax_%s' % miri_band].imshow(cutout_dict['%s_img_cutout' % miri_band].data,
                                                      norm=norm_miri, cmap=cmap_miri)
            else:
                axis_dict['ax_%s' % miri_band].cla()
                continue

            # set limits
            set_lim2cutout(ax=axis_dict['ax_%s' % miri_band], cutout=cutout_dict['%s_img_cutout' % miri_band],
                           cutout_pos=cutout_dict['cutout_pos'],
                           ra_length=axis_length[0], dec_length=axis_length[1])
            # plot circles
            if circ_pos is not None:
                plot_coord_circle(ax=axis_dict['ax_%s' % miri_band], pos=circ_pos, rad=circ_rad,
                                  color=circ_color)
            # add text
            arrange_text(ax=axis_dict['ax_%s' % miri_band],
                         data_shape=cutout_dict['%s_img_cutout' % miri_band].data.shape, text=miri_band,
                         fontsize=fontsize - 2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % miri_band], dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize - 1, ra_tick_num=ra_tick_num,
                            dec_tick_num=dec_tick_num)
        # arrange color bar
        if ((not np.isnan(norm_miri.vmin) and not np.isnan(norm_miri.vmax)) and
                (norm_miri.vmin != norm_miri.vmax)):
            create_cbar(ax_cbar=axis_dict['ax_color_bar_miri'], cmap=cmap_miri, norm=norm_miri, cbar_label=cbar_label,
                        fontsize=fontsize, ticks=ticks_miri)
        else:
            axis_dict['ax_color_bar_miri'].cla()

        return figure

    @staticmethod
    def plot_circ_flux_extraction(hst_band_list, nircam_band_list, miri_band_list, cutout_dict, aperture_dict,
                                  circ_pos=None, circ_rad=None, circ_color=None,
                                  fontsize=18, figsize=(18, 13),
                                  vmin_vmax_hst=None, vmin_vmax_nircam=None, vmin_vmax_miri=None,
                                  ticks_hst=None, ticks_nircam=None, ticks_miri=None,
                                  cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
                                  log_scale=False, axis_length=None,
                                  ra_tick_num=3, dec_tick_num=3):

        norm_hst = compute_cbar_norm(vmin_vmax=vmin_vmax_hst,
                                     cutout_list=[cutout_dict['%s_img_cutout' % band].data for band in hst_band_list],
                                     log_scale=log_scale)
        norm_nircam = compute_cbar_norm(vmin_vmax=vmin_vmax_nircam,
                                        cutout_list=[cutout_dict['%s_img_cutout' % band].data
                                                     for band in nircam_band_list], log_scale=log_scale)
        norm_miri = compute_cbar_norm(vmin_vmax=vmin_vmax_miri,
                                      cutout_list=[cutout_dict['%s_img_cutout' % band].data for band in miri_band_list],
                                      log_scale=log_scale)
        if log_scale:
            cbar_label = r'log(S /[MJy / sr])'
        else:
            cbar_label = r'S [MJy / sr]'

        # get figure
        figure = plt.figure(figsize=figsize)
        # arrange axis
        axis_dict = add_axis_hst_nircam_miri_panel(figure=figure, hst_band_list=hst_band_list,
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
                            width=3 * aperture_dict['aperture_dict_%s' % hst_band]['source_table']['a'][i],
                            height=3 * aperture_dict['aperture_dict_%s' % hst_band]['source_table']['b'][i],
                            angle=
                            aperture_dict['aperture_dict_%s' % hst_band]['source_table']['theta'][i] * 180. / np.pi)
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
                         fontsize=fontsize - 2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % hst_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize - 1, ra_tick_num=ra_tick_num,
                            dec_tick_num=dec_tick_num)
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
                            width=3 * aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['a'][i],
                            height=3 * aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['b'][i],
                            angle=
                            aperture_dict['aperture_dict_%s' % nircam_band]['source_table']['theta'][i] * 180. / np.pi)
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
                         fontsize=fontsize - 2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % nircam_band], ra_tick_label=False, dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize - 1, ra_tick_num=ra_tick_num,
                            dec_tick_num=dec_tick_num)
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
                            width=3 * aperture_dict['aperture_dict_%s' % miri_band]['source_table']['a'][i],
                            height=3 * aperture_dict['aperture_dict_%s' % miri_band]['source_table']['b'][i],
                            angle=
                            aperture_dict['aperture_dict_%s' % miri_band]['source_table']['theta'][i] * 180. / np.pi)
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
                         fontsize=fontsize - 2)
            # arrange axis
            dec_tick_label = False
            if index == 0:
                dec_tick_label = True
            arr_axis_params(ax=axis_dict['ax_%s' % miri_band], dec_tick_label=dec_tick_label,
                            fontsize=fontsize, labelsize=fontsize - 1, ra_tick_num=ra_tick_num,
                            dec_tick_num=dec_tick_num)
        # arrange color bar
        create_cbar(ax_cbar=axis_dict['ax_color_bar_miri'], cmap=cmap_miri, norm=norm_miri, cbar_label=cbar_label,
                    fontsize=fontsize, ticks=ticks_miri)

        return figure

    @staticmethod
    def plot_rgb_with_sed(cutout_dict, band_name_r, band_name_g, band_name_b,
                          aperture_dict_list, axis_length,
                          amp_fact_r=1, amp_fact_g=1, amp_fact_b=1, reproject_to='highest',
                          fontsize=20, figsize=(25, 8)):

        rgb_image, reproject_band = compute_rgb_image(cutout_dict=cutout_dict, band_name_r=band_name_r,
                                                      band_name_g=band_name_g,
                                                      band_name_b=band_name_b, amp_fact_r=amp_fact_r,
                                                      amp_fact_g=amp_fact_g,
                                                      amp_fact_b=amp_fact_b, reproject_to=reproject_to)

        figure = plt.figure(figsize=figsize)
        # arrange axis
        ax_rgb = figure.add_axes([0.04, 0.05, 0.25, 0.9], projection=cutout_dict['%s_img_cutout' % reproject_band].wcs)
        ax_sed = figure.add_axes([0.36, 0.08, 0.63, 0.9])

        ax_rgb.imshow(rgb_image)

        text_pos = SkyCoord(cutout_dict['cutout_pos'].ra + (axis_length[0] / 2) * 0.9 * u.arcsec,
                            cutout_dict['cutout_pos'].dec + (axis_length[1] / 2) * 0.85 * u.arcsec)
        # plot bands as text
        mulit_color_text_annotation(figure=figure, ax=ax_rgb, list_strings=[band_name_r, band_name_g, band_name_b],
                                    list_color=['red', 'lime', 'dodgerblue'],
                                    pos=text_pos, fontsize=fontsize-2)
        # set limits
        set_lim2cutout(ax=ax_rgb, cutout=cutout_dict['%s_img_cutout' % reproject_band],
                       cutout_pos=cutout_dict['cutout_pos'], ra_length=axis_length[0], dec_length=axis_length[1])

        arr_axis_params(ax=ax_rgb, tick_color='white', fontsize=fontsize, labelsize=fontsize - 1,
                        ra_tick_num=3, dec_tick_num=4)

        colors = ['tab:blue', 'tab:orange', 'tab:green']
        # plot all apertures

        if isinstance(aperture_dict_list, dict):
            aperture_dict_list = [aperture_dict_list]

        for aperture_dict, aperture_index in zip(aperture_dict_list, range(len(aperture_dict_list))):
            plot_coord_circle(ax=ax_rgb, pos=aperture_dict['init_pos'], rad=aperture_dict['recenter_rad'], linewidth=1,
                              color='white')
            pos_annotation = SkyCoord(aperture_dict['init_pos'].ra + aperture_dict['recenter_rad'] * u.arcsec,
                                      aperture_dict['init_pos'].dec + aperture_dict['recenter_rad'] * u.arcsec)
            text_annotation(ax=ax_rgb, pos=pos_annotation, text=aperture_index + 1, color='white')
            plot_sed_data_points(ax=ax_sed, bands=cutout_dict['band_list'], aperture_dict=aperture_dict,
                                 color=colors[aperture_index], annotation=aperture_index + 1)

        ax_sed.set_xscale('log')
        ax_sed.set_yscale('log')
        ax_sed.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
        ax_sed.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-8, fontsize=fontsize)

        ax_sed.legend(frameon=False, fontsize=fontsize)
        ax_sed.tick_params(axis='both', which='both', width=2, length=5, direction='in', pad=10, labelsize=fontsize)
        # ax_sed.set_xlim(200 * 1e-3, 3e1)
        # ax_sed.set_ylim(0.0000009, 3e4)

        return figure


    @staticmethod
    def plot_cigale_sed_panel(hst_band_list, nircam_band_list, miri_band_list, cutout_dict, aperture_dict,
                              cigale_logo_file_name=None, filter_colors=None, fontsize=33, x_axis=True,
                              title=None, show_ax_label=None):

        if show_ax_label is None:
            show_ax_label = ['F275W']

        if filter_colors is None:
            filter_colors = np.array(['k', 'k', 'k', 'k', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red',
                                      'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray'])

        # plotting
        figure = plt.figure(figsize=(30, 10))

        ax_sed = figure.add_axes([0.06, 0.09, 0.935, 0.905])

        if cigale_logo_file_name is not None:
            ax_cigale_logo = figure.add_axes([0.885, 0.31, 0.15, 0.15])
            ax_cigale_logo.imshow(mpimg.imread(cigale_logo_file_name))
            ax_cigale_logo.axis('off')

        # get axis for postage stamps
        axis_dict = add_axis_hst_nircam_miri_postage(figure=figure, hst_band_list=hst_band_list,
                                                     nircam_band_list=nircam_band_list, miri_band_list=miri_band_list,
                                                     cutout_dict=cutout_dict)

        # plot the postage_staps
        plot_postage_stamps_circ(axis_dict=axis_dict, cutout_dict=cutout_dict, aperture_dict=aperture_dict,
                            filter_colors=filter_colors, fontsize=fontsize, show_ax_label=show_ax_label)

        plot_sed_data_points(ax=ax_sed, band_list=cutout_dict['band_list'], aperture_dict=aperture_dict,
                             color=filter_colors, annotation=None, line_color='k')

        if title is not None:
            if isinstance(title, str):
                ax_sed.text(0.25, 0.1, title, fontsize=fontsize-4)
            elif isinstance(title, list):
                for title_string, title_index in zip(title[::-1], range(len(title))):
                    ax_sed.text(0.25, 10**(-len(title) + 1 + title_index*0.6), title_string, fontsize=fontsize-4)

        ax_sed.set_ylabel(r'S$_{\nu}$ (mJy)', fontsize=fontsize)
        ax_sed.tick_params(axis='both', which='both', width=4, length=5, direction='in', pad=10, labelsize=fontsize)
        ax_sed.set_xlim(200 * 1e-3, 3e1)
        ax_sed.set_ylim(0.0000009, 3e4)
        ax_sed.set_xscale('log')
        ax_sed.set_yscale('log')

        if x_axis:
            ax_sed.set_xlabel(r'Observed ${\lambda}$ ($\mu$m)', labelpad=-8, fontsize=fontsize)
        else:
            ax_sed.set_xticklabels([])

        return figure, ax_sed

    @staticmethod
    def plot_iter_source_detection(fit_result_dict, band, fontsize=18):

        data_sub = fit_result_dict['data_sub']
        wcs = fit_result_dict['wcs']
        object_table_n1 = fit_result_dict['object_table_n1']
        fit_result_n1 = fit_result_dict['fit_result_n1']
        model_data_n1 = fit_result_dict['model_data_n1']
        residuals_n1 = fit_result_dict['residuals_n1']
        object_table_residuals_n1 = fit_result_dict['object_table_residuals_n1']
        object_table_n2 = fit_result_dict['object_table_n2']
        fit_result_n2 = fit_result_dict['fit_result_n2']
        model_data_n2 = fit_result_dict['model_data_n2']
        residuals_n2 = fit_result_dict['residuals_n2']
        object_table_residuals_n2 = fit_result_dict['object_table_residuals_n2']

        img_mean, img_std, img_max = np.mean(data_sub), np.std(data_sub), np.max(data_sub)

        # plot results
        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(18, 7))
        # box = (x.min(), x.max(), y.min(), y.max())          # left, right, bottom, top
        im0 = ax[0, 0].imshow(data_sub, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean + 10*img_std, cmap='inferno')
        cbar1 = fig.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)
        im1 = ax[0, 1].imshow(model_data_n1, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean + 10*img_std,  cmap='inferno')
        cbar2 = fig.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)
        im2 = ax[0, 2].imshow(fit_result_n1.best_fit, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean + 10*img_std,  cmap='inferno')
        cbar3 = fig.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)
        im3 = ax[0, 3].imshow(residuals_n1, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean +img_std,  cmap='inferno')
        cbar4 = fig.colorbar(im3, ax=ax[0, 3], fraction=0.046, pad=0.04)

        ax[0, 0].set_title('Data % s' % band, fontsize=fontsize)
        ax[0, 1].set_title('Model', fontsize=fontsize)
        ax[0, 2].set_title('Model convolved', fontsize=fontsize)
        ax[0, 3].set_title('Residuals', fontsize=fontsize)

        x_n1, y_n1, a_n1, b_n1 = helper_func.transform_ellips_world2pix(param_table=object_table_n1, wcs=wcs)
        x_res_n1, y_res_n1, a_res_n1, b_res_n1 = helper_func.transform_ellips_world2pix(param_table=object_table_residuals_n1, wcs=wcs)

        # plot an ellipse for each object
        for i in range(len(object_table_n1)):
            e = Ellipse(xy=(x_n1[i], y_n1[i]),
                        width=3*a_n1[i],
                        height=3*b_n1[i],
                        angle=object_table_n1['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            e.set_linewidth(1)
            ax[0, 0].add_artist(e)
            ax[0, 0].text(x_n1[i], y_n1[i], i, horizontalalignment='center', color='white')
        # plot an ellipse for each object
        for i in range(len(object_table_residuals_n1)):
            e = Ellipse(xy=(x_res_n1[i], y_res_n1[i]),
                        width=3*a_res_n1[i],
                        height=3*b_res_n1[i],
                        angle=object_table_residuals_n1['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            e.set_linewidth(3)
            ax[0, 3].add_artist(e)

        # box = (x.min(), x.max(), y.min(), y.max())          # left, right, bottom, top
        im0 = ax[1, 0].imshow(data_sub, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean + 10*img_std, cmap='inferno')
        cbar1 = fig.colorbar(im0, ax=ax[1, 0], fraction=0.046, pad=0.04)
        im1 = ax[1, 1].imshow(model_data_n2, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean + 10*img_std,  cmap='inferno')
        cbar2 = fig.colorbar(im1, ax=ax[1, 1], fraction=0.046, pad=0.04)
        im2 = ax[1, 2].imshow(fit_result_n2.best_fit, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean + 10*img_std,  cmap='inferno')
        cbar3 = fig.colorbar(im2, ax=ax[1, 2], fraction=0.046, pad=0.04)
        im3 = ax[1, 3].imshow(residuals_n2, interpolation="none", origin='lower', vmin=img_mean-img_std, vmax=img_mean +img_std,  cmap='inferno')
        cbar4 = fig.colorbar(im3, ax=ax[1, 3], fraction=0.046, pad=0.04)
        ax[1, 0].set_title('Data %s' % band, fontsize=fontsize)
        ax[1, 1].set_title('NEW Model', fontsize=fontsize)
        ax[1, 2].set_title('Model convolved', fontsize=fontsize)
        ax[1, 3].set_title('Residuals', fontsize=fontsize)

        x_n2, y_n2, a_n2, b_n2 = helper_func.transform_ellips_world2pix(param_table=object_table_n2, wcs=wcs)
        x_res_n2, y_res_n2, a_res_n2, b_res_n2 = helper_func.transform_ellips_world2pix(param_table=object_table_residuals_n2, wcs=wcs)

        # plot an ellipse for each object
        for i in range(len(object_table_n2)):
            e = Ellipse(xy=(x_n2[i], y_n2[i]),
                        width=3*a_n2[i],
                        height=3*b_n2[i],
                        angle=object_table_n2['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            e.set_linewidth(1)
            ax[1, 0].add_artist(e)
            ax[1, 0].text(x_n2[i], y_n2[i], i, horizontalalignment='center', color='white')
        # plt.show()
        # plot an ellipse for each object
        for i in range(len(object_table_residuals_n2)):
            e = Ellipse(xy=(x_res_n2[i], y_res_n2[i]),
                        width=3*a_res_n2[i],
                        height=3*b_res_n2[i],
                        angle=object_table_residuals_n2['theta'][i] * 180. / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            e.set_linewidth(3)
            ax[1, 3].add_artist(e)

        return fig


def add_axis_hst_nircam_miri_postage(figure, hst_band_list, nircam_band_list, miri_band_list, cutout_dict):

    axis_dict = {}

    for hst_band, index in zip(hst_band_list, range(len(hst_band_list))):
        ax = figure.add_axes([0.02 + index * 0.065, 0.71, 0.19, 0.19],
                             projection=cutout_dict['%s_img_cutout' % hst_band].wcs)
        axis_dict.update({'ax_%s' % hst_band: ax})

    for nircam_band, index in zip(nircam_band_list, range(len(nircam_band_list))):
        ax = figure.add_axes([0.37 + index * 0.065, 0.71, 0.19, 0.19],
                             projection=cutout_dict['%s_img_cutout' % nircam_band].wcs)
        axis_dict.update({'ax_%s' % nircam_band: ax})

    for miri_band, index in zip(miri_band_list, range(len(miri_band_list))):
        ax = figure.add_axes([0.65 + index * 0.065, 0.71, 0.19, 0.19],
                             projection=cutout_dict['%s_img_cutout' % miri_band].wcs)
        axis_dict.update({'ax_%s' % miri_band: ax})

    return axis_dict


def plot_postage_stamps_circ(axis_dict, cutout_dict, aperture_dict, filter_colors, fontsize=15, show_ax_label=None):

    band_list = cutout_dict['band_list']

    if show_ax_label is None:
        show_ax_label = ['None']
    elif isinstance(show_ax_label, str):
        show_ax_label = [show_ax_label]

    for band, band_index in zip(band_list, range(len(band_list))):
        m, s = np.nanmean(cutout_dict['%s_img_cutout' % band].data), np.nanstd(cutout_dict['%s_img_cutout' % band].data)
        axis_dict['ax_%s' % band].imshow(cutout_dict['%s_img_cutout' % band].data, cmap='Greys', vmin=m-s, vmax=m+5*s)
        plot_coord_circle(ax=axis_dict['ax_%s' % band], pos=aperture_dict['aperture_dict_%s' % band]['new_pos'],
                          rad=aperture_dict['aperture_rad_dict']['aperture_%s' % band], color='r',
                          linewidth=2)
        axis_dict['ax_%s' % band].set_title(band.upper(), fontsize=fontsize, color=filter_colors[band_index])
        if band in show_ax_label:
            axis_dict['ax_%s' % band].tick_params(axis='both', which='both', width=3, length=7, direction='in',
                                                  color='k', labelsize=fontsize-11)
            axis_dict['ax_%s' % band].coords['dec'].set_ticklabel(rotation=90)
            axis_dict['ax_%s' % band].coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.3, fontsize=fontsize-11)
            axis_dict['ax_%s' % band].coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize-11)
            axis_dict['ax_%s' % band].coords['ra'].set_ticks(number=2)
            axis_dict['ax_%s' % band].coords['ra'].display_minor_ticks(True)
            axis_dict['ax_%s' % band].coords['dec'].set_ticks(number=2)
            axis_dict['ax_%s' % band].coords['dec'].display_minor_ticks(True)
        else:
            erase_wcs_axis(axis_dict['ax_%s' % band])


def plot_postage_stamps(ax, cutout, filter_color='k', fontsize=15, show_ax_label=False, title=''):

    if show_ax_label is None:
        show_ax_label = ['None']
    elif isinstance(show_ax_label, str):
        show_ax_label = [show_ax_label]


    m, s = np.nanmean(cutout.data), np.nanstd(cutout.data)
    ax.imshow(cutout.data, cmap='Greys', vmin=m-s, vmax=m+5*s)
    ax.set_title(title.upper(), fontsize=fontsize, color=filter_color)
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


def erase_wcs_axis(ax):
    ax.coords['ra'].set_ticklabel_visible(False)
    ax.coords['ra'].set_axislabel(' ')
    ax.coords['dec'].set_ticklabel_visible(False)
    ax.coords['dec'].set_axislabel(' ')
    ax.tick_params(axis='both', which='both', width=0.00001, direction='in', color='k')


def plot_sed_data_points(ax, band_list, aperture_dict, color, annotation=None, line_color=None, snr_plot=3, snr_detect=5):

    if isinstance(color, str):
        color = [color] * len(band_list)

    flux_list = []
    flux_err_list = []
    wave_list = []
    for band, band_index in zip(band_list, range(len(band_list))):
        flux = aperture_dict['aperture_dict_%s' % band]['flux']
        flux_err = aperture_dict['aperture_dict_%s' % band]['flux_err']
        wave = aperture_dict['aperture_dict_%s' % band]['wave']

        flux_list.append(flux)
        flux_err_list.append(flux_err)
        wave_list.append(wave)

        if (flux < 0) | (flux < snr_detect * flux_err):
            ax.errorbar(wave, snr_plot*flux_err,  yerr=flux_err, ecolor=color[band_index],
                        elinewidth=5, capsize=10, uplims=True, xlolims=False)
        else:
            ax.errorbar(wave, flux, yerr=flux_err, ms=15, ecolor='k', fmt='o', color=color[band_index])

    wave_list = np.array(wave_list)
    flux_list = np.array(flux_list)
    flux_err_list = np.array(flux_err_list)

    upper_limit_mask = (flux_list < 0) | (flux_list < snr_detect * flux_err_list)
    if line_color is not None:
        ax.plot(wave_list[~upper_limit_mask], flux_list[~upper_limit_mask], color=line_color, label=annotation)


def compute_rgb_image(cutout_dict, band_name_r, band_name_g, band_name_b,
                      amp_fact_r=1, amp_fact_g=1, amp_fact_b=1, reproject_to='highest'):
    # get rgb image
    shape_values = [np.sum(cutout_dict['%s_img_cutout' % band_name_r].data.shape),
                    np.sum(cutout_dict['%s_img_cutout' % band_name_g].data.shape),
                    np.sum(cutout_dict['%s_img_cutout' % band_name_b].data.shape)]
    band_names = [band_name_r, band_name_g, band_name_b]

    if reproject_to == 'highest':
        reproject_band = band_names[np.where(shape_values == np.max(shape_values))[0][0]]
        print('reproject_band ', reproject_band)
    elif reproject_to == 'lowest':
        reproject_band = band_names[np.where(shape_values == np.min(shape_values))[0][0]]
        print('reproject_band ', reproject_band)
    elif reproject_to in band_names:
        reproject_band = reproject_to
    else:
        raise KeyError('reproject_to must be either highest, lowest or one of the b g r bands')

    # get reprojection_data
    new_wcs = cutout_dict['%s_img_cutout' % reproject_band].wcs
    new_shape = cutout_dict['%s_img_cutout' % reproject_band].data.shape

    reprojected_data = [None, None, None]
    for band, band_index in zip(band_names, range(len(band_names))):
        if band == reproject_band:
            reprojected_data[band_index] = cutout_dict['%s_img_cutout' % band].data
        else:
            reprojected_data[band_index] = reproject_image(data=cutout_dict['%s_img_cutout' % band].data,
                                                           wcs=cutout_dict['%s_img_cutout' % band].wcs,
                                                           new_wcs=new_wcs, new_shape=new_shape)

    rgb_image = make_lupton_rgb(reprojected_data[0] * amp_fact_r, reprojected_data[1] * amp_fact_g,
                                reprojected_data[2] * amp_fact_b, Q=15, stretch=7)

    return rgb_image, reproject_band


def reproject_image(data, wcs, new_wcs, new_shape):
    hdu = fits.PrimaryHDU(data=data, header=wcs.to_header())
    return reproject_interp(hdu, new_wcs, shape_out=new_shape, return_footprint=False)


def add_axis_hst_nircam_miri_panel(figure, hst_band_list, nircam_band_list, miri_band_list, cutout_dict, cbar=True):
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
    ax.text(data_shape[0] * axis_offset_x, data_shape[1] * axis_offset_y, text, color=color, fontsize=fontsize + 2)


def arr_axis_params(ax, ra_tick_label=True, dec_tick_label=True,
                    ra_axis_label='R.A. (2000.0)', dec_axis_label='DEC. (2000.0)',
                    ra_minpad=0.8, dec_minpad=0.8, tick_color='k', label_color='k',
                    fontsize=15, labelsize=14, ra_tick_num=3, dec_tick_num=3):

    ax.tick_params(axis='both', which='both', width=1.5, length=7, direction='in', color=tick_color,
                   labelsize=labelsize)

    if not ra_tick_label:
        ax.coords['ra'].set_ticklabel_visible(False)
        ax.coords['ra'].set_axislabel(' ', color=label_color)
    else:
        ax.coords['ra'].set_ticklabel(rotation=0, color=label_color)
        ax.coords['ra'].set_axislabel(ra_axis_label, minpad=ra_minpad, color=label_color, fontsize=fontsize)

    if not dec_tick_label:
        ax.coords['dec'].set_ticklabel_visible(False)
        ax.coords['dec'].set_axislabel(' ', color=label_color)
    else:
        ax.coords['dec'].set_ticklabel(rotation=90, color=label_color)
        ax.coords['dec'].set_axislabel(dec_axis_label, minpad=dec_minpad, color=label_color, fontsize=fontsize)

    ax.coords['ra'].set_ticks(number=ra_tick_num)
    ax.coords['ra'].display_minor_ticks(True)
    ax.coords['dec'].set_ticks(number=dec_tick_num)
    ax.coords['dec'].display_minor_ticks(True)


def compute_cbar_norm(vmin_vmax=None, cutout_list=None, log_scale=False):
    """
    Computing the color bar scale for a single or multiple cutouts.

    Parameters
    ----------
    vmin_vmax : tuple
    cutout_list : list
        This list should include all cutouts
    log_scale : bool

    Returns
    -------
    norm : ``matplotlib.colors.Normalize``  or ``matplotlib.colors.LogNorm``
    """
    if (vmin_vmax is None) & (cutout_list is None):
        raise KeyError('either vmin_vmax or cutout_list must be not None')

    # get maximal value
    if vmin_vmax is None:
        list_of_means = [np.nanmean(cutout) for cutout in cutout_list]
        list_of_stds = [np.nanstd(cutout) for cutout in cutout_list]
        mean, std = (np.nanmean(list_of_means), np.nanstd(list_of_stds))

        vmin = mean - 5 * std
        vmax = mean + 20 * std
    else:
        vmin, vmax = vmin_vmax[0], vmin_vmax[1]
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
    ColorbarBase(ax_cbar, orientation=orientation, cmap=cmap, norm=norm, extend=extend, ticks=ticks)
    ax_cbar.set_ylabel(cbar_label, labelpad=labelpad, fontsize=fontsize)
    ax_cbar.tick_params(axis='both', which='both', width=tick_width, direction='in', top=True, labelbottom=False,
                        labeltop=True, labelsize=fontsize)


def set_lim2cutout(ax, cutout, cutout_pos, ra_length, dec_length):
    # lim_top_left = cutout.wcs.world_to_pixel(SkyCoord(cutout_pos.ra + ra_length / 2 * u.arcsec,
    #                                                   cutout_pos.dec + dec_length / 2 * u.arcsec))
    # lim_bottom_right = cutout.wcs.world_to_pixel(SkyCoord(cutout_pos.ra - ra_length / 2 * u.arcsec,
    #                                                       cutout_pos.dec - dec_length / 2 * u.arcsec))
    # # ax.set_xlim(lim_top_left[0], lim_bottom_right[0])
    # # ax.set_ylim(lim_bottom_right[1], lim_top_left[1])

    ra_length_pix = helper_func.transform_world2pix_scale(length_in_arcsec=ra_length, wcs=cutout.wcs, dim=0)
    dec_length_pix = helper_func.transform_world2pix_scale(length_in_arcsec=dec_length, wcs=cutout.wcs, dim=1)

    ra_pix_length = cutout.data.shape[0]
    dec_pix_length = cutout.data.shape[1]

    border_ra = (ra_pix_length - ra_length_pix)/2
    border_dec = (dec_pix_length - dec_length_pix)/2

    ax.set_xlim(border_ra, ra_pix_length-border_ra)
    ax.set_ylim(border_dec, dec_pix_length-border_dec)


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
            circle = SphericalCircle(pos_i, rad_i * u.arcsec, edgecolor=color_i, facecolor=facecolor,
                                     linewidth=linewidth_i,
                                     linestyle=linestyle_i, alpha=alpha_i, transform=ax.get_transform('fk5'))
            ax.add_patch(circle)
    else:
        circle = SphericalCircle(pos, rad * u.arcsec, edgecolor=color, facecolor=facecolor, linewidth=linewidth,
                                 linestyle=linestyle, alpha=alpha, transform=ax.get_transform('fk5'))
        ax.add_patch(circle)


def text_annotation(ax, pos, text, color='k', rotation=0, horizontalalignment=None, verticalalignment=None,
                    fontsize=15):
    # horizontalalignment or ha {'left', 'center', 'right'}
    # verticalalignment or va {'bottom', 'baseline', 'center', 'center_baseline', 'top'}
    ax.text(pos.ra.degree, pos.dec.degree, text, color=color, fontsize=fontsize, rotation=rotation,
            horizontalalignment='right', transform=ax.get_transform('fk5'))


def mulit_color_text_annotation(figure, ax, list_strings, list_color, pos, fontsize=15):
    t = ax.get_transform('fk5')
    for s, c in zip(list_strings, list_color):
        text = ax.text(pos.ra.degree, pos.dec.degree, s+" ", color=c, fontsize=fontsize, transform=t)
        text.draw(figure.canvas.get_renderer())
        ex = text.get_window_extent()
        t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


def draw_box(ax, wcs, coord, box_size, color='k', linewidth=2, linestyle='-'):
    if isinstance(box_size, tuple):
        box_size = box_size * u.arcsec
    elif isinstance(box_size, float) | isinstance(box_size, int):
        box_size = (box_size, box_size) * u.arcsec
    else:
        raise KeyError('cutout_size must be float or tuple')

    top_left_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra + (box_size[1] / 2)/np.cos(coord.dec.degree*np.pi/180),
                                               dec=coord.dec + (box_size[0] / 2)))
    top_right_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra - (box_size[1] / 2)/np.cos(coord.dec.degree*np.pi/180),
                                                dec=coord.dec + (box_size[0] / 2)))
    bottom_left_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra + (box_size[1] / 2)/np.cos(coord.dec.degree*np.pi/180),
                                                  dec=coord.dec - (box_size[0] / 2)))
    bottom_right_pix = wcs.world_to_pixel(SkyCoord(ra=coord.ra - (box_size[1] / 2)/np.cos(coord.dec.degree*np.pi/180),
                                                   dec=coord.dec - (box_size[0] / 2)))

    ax.plot([top_left_pix[0], top_right_pix[0]], [top_left_pix[1], top_right_pix[1]], color=color, linewidth=linewidth, linestyle=linestyle)
    ax.plot([bottom_left_pix[0], bottom_right_pix[0]], [bottom_left_pix[1], bottom_right_pix[1]], color=color,
            linewidth=linewidth, linestyle=linestyle)
    ax.plot([top_left_pix[0], bottom_left_pix[0]], [top_left_pix[1], bottom_left_pix[1]], color=color,
            linewidth=linewidth, linestyle=linestyle)
    ax.plot([top_right_pix[0], bottom_right_pix[0]], [top_right_pix[1], bottom_right_pix[1]], color=color,
            linewidth=linewidth, linestyle=linestyle)


class DensityContours:

    def __init__(self):
        r"""
        Class to produce contour lines

        """

    @staticmethod
    def compute_contour_counts(x_data, y_data):

        good = np.invert(((np.isnan(x_data) | np.isnan(y_data)) | (np.isinf(x_data) | np.isinf(y_data))))

        k = stats.gaussian_kde(np.vstack([x_data[good], y_data[good]]))
        xi, yi = np.mgrid[x_data[good].min():x_data[good].max():x_data[good].size**0.5*1j,
                              y_data[good].min():y_data[good].max():y_data[good].size**0.5*1j]

        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # set zi_1 to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi = zi.reshape(xi.shape)
        return xi, yi, zi

    @staticmethod
    def plot_contours_percentage(ax, xi, yi, zi, color='black', percent=True, **kwargs):

        if 'linewidth' in kwargs:
            linewidth = kwargs.get('linewidth')
        else:
            linewidth = 1.5

        if 'fontsize' in kwargs:
            fontsize = kwargs.get('fontsize')
        else:
            fontsize = 8

        if 'alpha' in kwargs:
            alpha = kwargs.get('alpha')
        else:
            alpha = 1

        if 'contour_levels' in kwargs:
            contour_levels = kwargs.get('contour_levels')
        else:
            contour_levels = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]

        cs = ax.contour(xi, yi, zi, levels=contour_levels, colors=color, linewidths=linewidth, origin='lower',
                        alpha=alpha)
        if percent:
            ax.clabel(cs, fmt='%.2f', colors=color, fontsize=fontsize)

    @staticmethod
    def get_contours_percentage(ax, x_data, y_data,
                                color='black', percent=True, **kwargs):

        if 'linewidth' in kwargs:
            linewidth = kwargs.get('linewidth')
        else:
            linewidth = 1.5

        if 'fontsize' in kwargs:
            fontsize = kwargs.get('fontsize')
        else:
            fontsize = 8

        if 'alpha' in kwargs:
            alpha = kwargs.get('alpha')
        else:
            alpha = 1

        if 'contour_levels' in kwargs:
            contour_levels = kwargs.get('contour_levels')
        else:
            contour_levels = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]
            # contour_levels = [0.68, 0.95]

        good = np.invert(((np.isnan(x_data) | np.isnan(y_data)) | (np.isinf(x_data) | np.isinf(y_data))))

        k = stats.gaussian_kde(np.vstack([x_data[good], y_data[good]]))
        xi, yi = np.mgrid[x_data[good].min():x_data[good].max():x_data[good].size**0.5*1j,
                              y_data[good].min():y_data[good].max():y_data[good].size**0.5*1j]

        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        # set zi_1 to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi = zi.reshape(xi.shape)

        cs = ax.contour(xi, yi, zi, levels=contour_levels, colors=color, linewidths=linewidth, origin='lower',
                        alpha=alpha)
        if percent:
            ax.clabel(cs, fmt='%.2f', colors=color, fontsize=fontsize)

    @staticmethod
    def get_two_contours_percentage(ax, x_data_1, y_data_1, x_data_2, y_data_2,
                                    color_1='black', color_2='blue', percent_1=True, percent_2=True, **kwargs):

        if 'linewidth' in kwargs:
            linewidth = kwargs.get('linewidth')
        else:
            linewidth = 1.5

        if 'fontsize' in kwargs:
            fontsize = kwargs.get('fontsize')
        else:
            fontsize = 8

        if 'alpha' in kwargs:
            alpha = kwargs.get('alpha')
        else:
            alpha = 1

        if 'contour_levels' in kwargs:
            contour_levels = kwargs.get('contour_levels')
        else:
            # contour_levels = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]
            contour_levels = [0, 0.2, 0.5, 0.68, 0.95, 0.99]
            # contour_levels = [0.68, 0.95]

        good_1 = np.invert(np.isnan(x_data_1) | np.isnan(y_data_1))
        good_2 = np.invert(np.isnan(x_data_2) | np.isnan(y_data_2))

        k_1 = stats.gaussian_kde(np.vstack([x_data_1[good_1], y_data_1[good_1]]))

        if ('x_lim_1' in kwargs) & ('y_lim_1' in kwargs):
            x_lim_1 = kwargs.get('x_lim_1')
            y_lim_1 = kwargs.get('y_lim_1')
        else:
            x_lim_1 = (x_data_1[good_1].min(), x_data_1[good_1].max())
            y_lim_1 = (y_data_1[good_1].min(), y_data_1[good_1].max())
        if 'size_1' in kwargs:
            size_1 = kwargs.get('size_1')
        else:
            size_1 = x_data_1[good_1].size
        xi_1, yi_1 = np.mgrid[x_lim_1[0]:x_lim_1[1]:size_1**0.5*1j,
                              y_lim_1[0]:y_lim_1[1]:size_1**0.5*1j]

        # xi_1, yi_1 = np.mgrid[x_data_1[good_1].min():x_data_1[good_1].max():x_data_1[good_1].size**0.5*1j,
        #                       y_data_1[good_1].min():y_data_1[good_1].max():y_data_1[good_1].size**0.5*1j]

        zi_1 = k_1(np.vstack([xi_1.flatten(), yi_1.flatten()]))

        # set zi_1 to 0-1 scale
        zi_1 = (zi_1-zi_1.min())/(zi_1.max() - zi_1.min())
        zi_1 = zi_1.reshape(xi_1.shape)

        cs_1 = ax.contour(xi_1, yi_1, zi_1, levels=contour_levels, colors=color_1, linewidths=linewidth, origin='lower',
                          alpha=alpha)

        if percent_1:
            ax.clabel(cs_1, fmt='%.2f', colors=color_1, fontsize=fontsize)

        k_2 = stats.gaussian_kde(np.vstack([x_data_2[good_2], y_data_2[good_2]]))

        if ('x_lim_2' in kwargs) & ('y_lim_2' in kwargs):
            x_lim_2 = kwargs.get('x_lim_2')
            y_lim_2 = kwargs.get('y_lim_2')
        else:
            x_lim_2 = (x_data_2[good_2].min(), x_data_2[good_2].max())
            y_lim_2 = (y_data_2[good_2].min(), y_data_2[good_2].max())
        if 'size_2' in kwargs:
            size_2 = kwargs.get('size_2')
        else:
            size_2 = x_data_2[good_2].size
        xi_2, yi_2 = np.mgrid[x_lim_2[0]:x_lim_2[1]:size_2**0.5*1j,
                              y_lim_2[0]:y_lim_2[1]:size_2**0.5*1j]
        # xi_2, yi_2 = np.mgrid[x_data_2[good_2].min():x_data_2[good_2].max():x_data_2[good_2].size**0.5*1j,
        #                       y_data_2[good_2].min():y_data_2[good_2].max():y_data_2[good_2].size**0.5*1j]

        zi_2 = k_2(np.vstack([xi_2.flatten(), yi_2.flatten()]))

        # set zi_2 to 0-1 scale
        zi_2 = (zi_2-zi_2.min())/(zi_2.max() - zi_2.min())
        zi_2 =zi_2.reshape(xi_2.shape)

        cs_2 = ax.contour(xi_2, yi_2, zi_2, levels=contour_levels, colors=color_2, linewidths=linewidth, origin='lower',
                          alpha=alpha)

        if percent_2:
            ax.clabel(cs_2, fmt='%.2f', colors=color_2, fontsize=fontsize)

    @staticmethod
    def get_three_contours_percentage(ax, x_data_1, y_data_1, x_data_2, y_data_2, x_data_3, y_data_3,
                                      color_1='black', color_2='blue', color_3='red',
                                      percent_1=True, percent_2=True, percent_3=True, **kwargs):

        if 'linewidth' in kwargs:
            linewidth = kwargs.get('linewidth')
        else:
            linewidth = 1.5

        if 'fontsize' in kwargs:
            fontsize = kwargs.get('fontsize')
        else:
            fontsize = 8

        if 'alpha_1' in kwargs:
            alpha_1 = kwargs.get('alpha_1')
        else:
            alpha_1 = 1

        if 'alpha_2' in kwargs:
            alpha_2 = kwargs.get('alpha_2')
        else:
            alpha_2 = 1

        if 'alpha_3' in kwargs:
            alpha_3 = kwargs.get('alpha_3')
        else:
            alpha_3 = 1

        if 'contour_levels_1' in kwargs:
            contour_levels_1 = kwargs.get('contour_levels_1')
        else:
            contour_levels_1 = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]

        if 'contour_levels_2' in kwargs:
            contour_levels_2 = kwargs.get('contour_levels_2')
        else:
            contour_levels_2 = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]

        if 'contour_levels_3' in kwargs:
            contour_levels_3 = kwargs.get('contour_levels_3')
        else:
            contour_levels_3 = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]


        k_1 = stats.gaussian_kde(np.vstack([x_data_1, y_data_1]))
        xi_1, yi_1 = np.mgrid[x_data_1.min():x_data_1.max():x_data_1.size**0.5*1j,
                              y_data_1.min():y_data_1.max():y_data_1.size**0.5*1j]

        zi_1 = k_1(np.vstack([xi_1.flatten(), yi_1.flatten()]))

        # set zi_1 to 0-1 scale
        zi_1 = (zi_1-zi_1.min())/(zi_1.max() - zi_1.min())
        zi_1 = zi_1.reshape(xi_1.shape)

        cs_1 = ax.contour(xi_1, yi_1, zi_1, levels=contour_levels_1, colors=color_1, linewidths=linewidth, origin='lower',
                          alpha=alpha_1)

        if percent_1:
            ax.clabel(cs_1, fmt='%.2f', colors=color_1, fontsize=fontsize)

        k_2 = stats.gaussian_kde(np.vstack([x_data_2, y_data_2]))
        xi_2, yi_2 = np.mgrid[x_data_2.min():x_data_2.max():x_data_2.size**0.5*1j,
                              y_data_2.min():y_data_2.max():y_data_2.size**0.5*1j]

        zi_2 = k_2(np.vstack([xi_2.flatten(), yi_2.flatten()]))

        # set zi_2 to 0-1 scale
        zi_2 = (zi_2-zi_2.min())/(zi_2.max() - zi_2.min())
        zi_2 = zi_2.reshape(xi_2.shape)

        cs_2 = ax.contour(xi_2, yi_2, zi_2, levels=contour_levels_2, colors=color_2, linewidths=linewidth, origin='lower',
                          alpha=alpha_2)

        if percent_2:
            ax.clabel(cs_2, fmt='%.2f', colors=color_2, fontsize=fontsize)

        k_3 = stats.gaussian_kde(np.vstack([x_data_3, y_data_3]))
        xi_3, yi_3 = np.mgrid[x_data_3.min():x_data_3.max():x_data_3.size**0.5*1j,
                              y_data_3.min():y_data_3.max():y_data_3.size**0.5*1j]

        zi_3 = k_3(np.vstack([xi_3.flatten(), yi_3.flatten()]))

        # set zi_3 to 0-1 scale
        zi_3 = (zi_3-zi_3.min())/(zi_3.max() - zi_3.min())
        zi_3 = zi_3.reshape(xi_3.shape)

        cs_3 = ax.contour(xi_3, yi_3, zi_3, levels=contour_levels_3, colors=color_3, linewidths=linewidth, origin='lower',
                          alpha=alpha_3)

        if percent_3:
            ax.clabel(cs_3, fmt='%.2f', colors=color_3, fontsize=fontsize)

    @staticmethod
    def get_four_contours_percentage(ax, x_data_1, y_data_1, x_data_2, y_data_2, x_data_3, y_data_3, x_data_4, y_data_4,
                                      color_1='black', color_2='blue', color_3='m', color_4='c',
                                      percent_1=True, percent_2=True, percent_3=True, percent_4=True, **kwargs):

        if 'linewidth' in kwargs:
            linewidth = kwargs.get('linewidth')
        else:
            linewidth = 1.5

        if 'fontsize' in kwargs:
            fontsize = kwargs.get('fontsize')
        else:
            fontsize = 8

        if 'alpha_1' in kwargs:
            alpha_1 = kwargs.get('alpha_1')
        else:
            alpha_1 = 1

        if 'alpha_2' in kwargs:
            alpha_2 = kwargs.get('alpha_2')
        else:
            alpha_2 = 1

        if 'alpha_3' in kwargs:
            alpha_3 = kwargs.get('alpha_3')
        else:
            alpha_3 = 1

        if 'alpha_4' in kwargs:
            alpha_4 = kwargs.get('alpha_4')
        else:
            alpha_4 = 1

        if 'contour_levels_1' in kwargs:
            contour_levels_1 = kwargs.get('contour_levels_1')
        else:
            contour_levels_1 = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]

        if 'contour_levels_2' in kwargs:
            contour_levels_2 = kwargs.get('contour_levels_2')
        else:
            contour_levels_2 = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]

        if 'contour_levels_3' in kwargs:
            contour_levels_3 = kwargs.get('contour_levels_3')
        else:
            contour_levels_3 = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]


        if 'contour_levels_4' in kwargs:
            contour_levels_4 = kwargs.get('contour_levels_4')
        else:
            contour_levels_4 = [0, 0.1, 0.2, 0.5, 0.68, 0.95, 0.99]


        k_1 = stats.gaussian_kde(np.vstack([x_data_1, y_data_1]))
        xi_1, yi_1 = np.mgrid[x_data_1.min():x_data_1.max():x_data_1.size**0.5*1j,
                              y_data_1.min():y_data_1.max():y_data_1.size**0.5*1j]

        zi_1 = k_1(np.vstack([xi_1.flatten(), yi_1.flatten()]))

        # set zi_1 to 0-1 scale
        zi_1 = (zi_1-zi_1.min())/(zi_1.max() - zi_1.min())
        zi_1 = zi_1.reshape(xi_1.shape)

        cs_1 = ax.contour(xi_1, yi_1, zi_1, levels=contour_levels_1, colors=color_1, linewidths=linewidth, origin='lower',
                          alpha=alpha_1)

        if percent_1:
            ax.clabel(cs_1, fmt='%.2f', colors=color_1, fontsize=fontsize)

        k_2 = stats.gaussian_kde(np.vstack([x_data_2, y_data_2]))
        xi_2, yi_2 = np.mgrid[x_data_2.min():x_data_2.max():x_data_2.size**0.5*1j,
                              y_data_2.min():y_data_2.max():y_data_2.size**0.5*1j]

        zi_2 = k_2(np.vstack([xi_2.flatten(), yi_2.flatten()]))

        # set zi_2 to 0-1 scale
        zi_2 = (zi_2-zi_2.min())/(zi_2.max() - zi_2.min())
        zi_2 = zi_2.reshape(xi_2.shape)

        cs_2 = ax.contour(xi_2, yi_2, zi_2, levels=contour_levels_2, colors=color_2, linewidths=linewidth, origin='lower',
                          alpha=alpha_2)

        if percent_2:
            ax.clabel(cs_2, fmt='%.2f', colors=color_2, fontsize=fontsize)

        k_3 = stats.gaussian_kde(np.vstack([x_data_3, y_data_3]))
        xi_3, yi_3 = np.mgrid[x_data_3.min():x_data_3.max():x_data_3.size**0.5*1j,
                              y_data_3.min():y_data_3.max():y_data_3.size**0.5*1j]

        zi_3 = k_3(np.vstack([xi_3.flatten(), yi_3.flatten()]))

        # set zi_3 to 0-1 scale
        zi_3 = (zi_3-zi_3.min())/(zi_3.max() - zi_3.min())
        zi_3 = zi_3.reshape(xi_3.shape)

        cs_3 = ax.contour(xi_3, yi_3, zi_3, levels=contour_levels_3, colors=color_3, linewidths=linewidth, origin='lower',
                          alpha=alpha_3)

        if percent_3:
            ax.clabel(cs_3, fmt='%.2f', colors=color_3, fontsize=fontsize)


        k_4 = stats.gaussian_kde(np.vstack([x_data_4, y_data_4]))
        xi_4, yi_4 = np.mgrid[x_data_4.min():x_data_4.max():x_data_4.size**0.5*1j,
                              y_data_4.min():y_data_4.max():y_data_4.size**0.5*1j]

        zi_4 = k_4(np.vstack([xi_4.flatten(), yi_4.flatten()]))

        # set zi_4 to 0-1 scale
        zi_4 = (zi_4-zi_4.min())/(zi_4.max() - zi_4.min())
        zi_4 = zi_4.reshape(xi_4.shape)

        cs_4 = ax.contour(xi_4, yi_4, zi_4, levels=contour_levels_4, colors=color_4, linewidths=linewidth, origin='lower',
                          alpha=alpha_4)

        if percent_4:
            ax.clabel(cs_4, fmt='%.2f', colors=color_4, fontsize=fontsize)