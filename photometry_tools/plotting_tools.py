"""
Plotting tools for photometry and SED fitting
"""

import numpy as np
import matplotlib.pyplot as plt


class PlotPhotometry:

    @staticmethod
    def plot_multi_zoom_panel_hst_nircam_miri(hst_band_list,
                                              nircam_band_list,
                                              miri_band_list,
                                              cutout_dict,
                                              circ_pos=False, circ_rad=0.5,
                                              fontsize=20, figsize=(18, 13),
                                              vmax_hst=None, vmax_nircam=None, vmax_miri=None,
                                              ticks_hst=None, ticks_nircam=None, ticks_miri=None,
                                              cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
                                              log_scale=False,
                                              name_ra_offset=2.4, name_dec_offset=1.5,
                                              ra_tick_num=3, dec_tick_num=3):

        if vmax_hst is None:
            vmax_hst = np.max([cutout_dict['%s_img_cutout' % band].data for band in hst_band_list])
        if vmax_nircam is None:
            vmax_nircam = np.max([cutout_dict['%s_img_cutout' % band].data for band in nircam_band_list])
        if vmax_miri is None:
            vmax_miri = np.max([cutout_dict['%s_img_cutout' % band].data for band in miri_band_list])

        # build up a figure
        figure = plt.figure(figsize=figsize)

        for hst_band in hst_band_list:


        ax_hst_f275w = figure.add_axes([0.0, 0.67, 0.3, 0.3], projection=cutout_hst_f275w.wcs)
        ax_hst_f438w = figure.add_axes([0.22, 0.67, 0.3, 0.3], projection=cutout_hst_f438w.wcs)
        ax_hst_f555w = figure.add_axes([0.44, 0.67, 0.3, 0.3], projection=cutout_hst_f555w.wcs)
        ax_hst_f814w = figure.add_axes([0.66, 0.67, 0.3, 0.3], projection=cutout_hst_f814w.wcs)
        ax_color_bar_hst = figure.add_axes([0.925, 0.68, 0.015, 0.28])

        ax_jwst_f200w = figure.add_axes([0.0, 0.365, 0.3, 0.3], projection=cutout_jwst_f200w.wcs)
        ax_jwst_f300m = figure.add_axes([0.22, 0.365, 0.3, 0.3], projection=cutout_jwst_f300m.wcs)
        ax_jwst_f335m = figure.add_axes([0.44, 0.365, 0.3, 0.3], projection=cutout_jwst_f335m.wcs)
        ax_jwst_f360m = figure.add_axes([0.66, 0.365, 0.3, 0.3], projection=cutout_jwst_f360m.wcs)
        ax_color_bar_nircam = figure.add_axes([0.925, 0.375, 0.015, 0.28])

        ax_jwst_f770w = figure.add_axes([0.0, 0.06, 0.3, 0.3], projection=cutout_jwst_f770w.wcs)
        ax_jwst_f1000w = figure.add_axes([0.22, 0.06, 0.3, 0.3], projection=cutout_jwst_f1000w.wcs)
        ax_jwst_f1130w = figure.add_axes([0.44, 0.06, 0.3, 0.3], projection=cutout_jwst_f1130w.wcs)
        ax_jwst_f2100w = figure.add_axes([0.66, 0.06, 0.3, 0.3], projection=cutout_jwst_f2100w.wcs)
        ax_color_bar_miri = figure.add_axes([0.925, 0.07, 0.015, 0.28])

        if log_scale:
            cb_hst = ax_hst_f275w.imshow(np.log10(cutout_hst_f275w.data), vmin=np.log10(vmax_hst/100),
                                         vmax=np.log10(vmax_hst), cmap=cmap_hst)
            ax_hst_f438w.imshow(np.log10(cutout_hst_f438w.data), vmin=np.log10(vmax_hst/100), vmax=np.log10(vmax_hst),
                                cmap=cmap_hst)
            ax_hst_f555w.imshow(np.log10(cutout_hst_f555w.data), vmin=np.log10(vmax_hst/100), vmax=np.log10(vmax_hst),
                                cmap=cmap_hst)
            ax_hst_f814w.imshow(np.log10(cutout_hst_f814w.data), vmin=np.log10(vmax_hst/100), vmax=np.log10(vmax_hst),
                                cmap=cmap_hst)

            cb_nircam = ax_jwst_f200w.imshow(np.log10(cutout_jwst_f200w.data), vmin=np.log10(vmax_nircam/100),
                                             vmax=np.log10(vmax_nircam), cmap=cmap_nircam)
            ax_jwst_f300m.imshow(np.log10(cutout_jwst_f300m.data), vmin=np.log10(vmax_nircam/100),
                                 vmax=np.log10(vmax_nircam), cmap=cmap_nircam)
            ax_jwst_f335m.imshow(np.log10(cutout_jwst_f335m.data), vmin=np.log10(vmax_nircam/100),
                                 vmax=np.log10(vmax_nircam), cmap=cmap_nircam)
            ax_jwst_f360m.imshow(np.log10(cutout_jwst_f360m.data), vmin=np.log10(vmax_nircam/100),
                                 vmax=np.log10(vmax_nircam), cmap=cmap_nircam)

            cb_miri = ax_jwst_f770w.imshow(np.log10(cutout_jwst_f770w.data), vmin=np.log10(vmax_miri/100),
                                           vmax=np.log10(vmax_miri), cmap=cmap_miri)
            ax_jwst_f1000w.imshow(np.log10(cutout_jwst_f1000w.data), vmin=np.log10(vmax_miri/100),
                                  vmax=np.log10(vmax_miri), cmap=cmap_miri)
            ax_jwst_f1130w.imshow(np.log10(cutout_jwst_f1130w.data), vmin=np.log10(vmax_miri/100),
                                  vmax=np.log10(vmax_miri), cmap=cmap_miri)
            ax_jwst_f2100w.imshow(np.log10(cutout_jwst_f2100w.data), vmin=np.log10(vmax_miri/100),
                                  vmax=np.log10(vmax_miri), cmap=cmap_miri)
            colorbar_label = r'log(S /[MJy / sr])'
        else:
            cb_hst = ax_hst_f275w.imshow(cutout_hst_f275w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)
            ax_hst_f438w.imshow(cutout_hst_f438w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)
            ax_hst_f555w.imshow(cutout_hst_f555w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)
            ax_hst_f814w.imshow(cutout_hst_f814w.data, vmin=-vmax_hst/10, vmax=vmax_hst, cmap=cmap_hst)

            cb_nircam = ax_jwst_f200w.imshow(cutout_jwst_f200w.data, vmin=-vmax_nircam/10, vmax=vmax_nircam,
                                             cmap=cmap_nircam)
            ax_jwst_f300m.imshow(cutout_jwst_f300m.data, vmin=-vmax_nircam/10, vmax=vmax_nircam, cmap=cmap_nircam)
            ax_jwst_f335m.imshow(cutout_jwst_f335m.data, vmin=-vmax_nircam/10, vmax=vmax_nircam, cmap=cmap_nircam)
            ax_jwst_f360m.imshow(cutout_jwst_f360m.data, vmin=-vmax_nircam/10, vmax=vmax_nircam, cmap=cmap_nircam)

            cb_miri = ax_jwst_f770w.imshow(cutout_jwst_f770w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            ax_jwst_f1000w.imshow(cutout_jwst_f1000w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            ax_jwst_f1130w.imshow(cutout_jwst_f1130w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            ax_jwst_f2100w.imshow(cutout_jwst_f2100w.data, vmin=-vmax_miri/10, vmax=vmax_miri, cmap=cmap_miri)
            colorbar_label = r'S [MJy / sr]'


        figure.colorbar(cb_hst, cax=ax_color_bar_hst, ticks=ticks_hst, orientation='vertical')
        ax_color_bar_hst.set_ylabel(colorbar_label, labelpad=2, fontsize=fontsize)
        ax_color_bar_hst.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                                     labeltop=True, labelsize=fontsize)

        figure.colorbar(cb_nircam, cax=ax_color_bar_nircam, ticks=ticks_nircam, orientation='vertical')
        ax_color_bar_nircam.set_ylabel(colorbar_label, labelpad=2, fontsize=fontsize)
        ax_color_bar_nircam.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                                        labeltop=True, labelsize=fontsize)

        figure.colorbar(cb_miri, cax=ax_color_bar_miri, ticks=ticks_miri, orientation='vertical')
        ax_color_bar_miri.set_ylabel(colorbar_label, labelpad=18, fontsize=fontsize)
        ax_color_bar_miri.tick_params(axis='both', which='both', width=2, direction='in', top=True, labelbottom=False,
                                      labeltop=True, labelsize=fontsize)

        VisualizeHelper.set_lim2cutout(ax=ax_hst_f275w, cutout=cutout_hst_f275w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_hst_f438w, cutout=cutout_hst_f438w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_hst_f555w, cutout=cutout_hst_f555w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_hst_f814w, cutout=cutout_hst_f814w, cutout_pos=cutout_pos, edge_cut_ratio=100)

        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f200w, cutout=cutout_jwst_f200w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f300m, cutout=cutout_jwst_f300m, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f335m, cutout=cutout_jwst_f335m, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f360m, cutout=cutout_jwst_f360m, cutout_pos=cutout_pos, edge_cut_ratio=100)

        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f770w, cutout=cutout_jwst_f770w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f1000w, cutout=cutout_jwst_f1000w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f1130w, cutout=cutout_jwst_f1130w, cutout_pos=cutout_pos, edge_cut_ratio=100)
        VisualizeHelper.set_lim2cutout(ax=ax_jwst_f2100w, cutout=cutout_jwst_f2100w, cutout_pos=cutout_pos, edge_cut_ratio=100)

        if circ_pos:
            VisualizeHelper.plot_coord_circle(ax_hst_f275w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_hst_f438w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_hst_f555w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_hst_f814w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f200w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f300m, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f335m, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f360m, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f770w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f1000w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f1130w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')
            VisualizeHelper.plot_coord_circle(ax_jwst_f2100w, position=circ_pos, radius=circ_rad*u.arcsec, color='k')


        text = cutout_hst_f275w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f275w.text(text[0], text[1], hst_channel_list[0], color='k', fontsize=fontsize+2)

        text = cutout_hst_f438w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f438w.text(text[0], text[1], hst_channel_list[1], color='k', fontsize=fontsize+2)

        text = cutout_hst_f555w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f555w.text(text[0], text[1], hst_channel_list[2], color='k', fontsize=fontsize+2)

        text = cutout_hst_f814w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_hst_f814w.text(text[0], text[1], hst_channel_list[3], color='k', fontsize=fontsize+2)


        text = cutout_jwst_f200w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f200w.text(text[0], text[1], nircam_channel_list[0], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f300m.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f300m.text(text[0], text[1], nircam_channel_list[1], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f335m.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f335m.text(text[0], text[1], nircam_channel_list[2], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f360m.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f360m.text(text[0], text[1], nircam_channel_list[3], color='k', fontsize=fontsize+2)


        text = cutout_jwst_f770w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f770w.text(text[0], text[1], miri_channel_list[0], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f1000w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f1000w.text(text[0], text[1], miri_channel_list[1], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f1130w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f1130w.text(text[0], text[1], miri_channel_list[2], color='k', fontsize=fontsize+2)

        text = cutout_jwst_f2100w.wcs.world_to_pixel(SkyCoord(cutout_pos.ra+name_ra_offset*u.arcsec, cutout_pos.dec+name_dec_offset*u.arcsec))
        ax_jwst_f2100w.text(text[0], text[1], miri_channel_list[3], color='k', fontsize=fontsize+2)


        ax_hst_f275w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_hst_f438w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_hst_f555w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_hst_f814w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f200w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f300m.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f335m.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f360m.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f770w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f1000w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f1130w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)
        ax_jwst_f2100w.tick_params(axis='both', which='both', width=3, length=7, direction='in', color='k', labelsize=fontsize-1)



        ax_hst_f275w.coords['dec'].set_ticklabel(rotation=90)
        ax_hst_f275w.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f200w.coords['dec'].set_ticklabel(rotation=90)
        ax_jwst_f200w.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f770w.coords['dec'].set_ticklabel(rotation=90)
        ax_jwst_f770w.coords['dec'].set_axislabel('DEC. (2000.0)', minpad=0.8, fontsize=fontsize)


        ax_jwst_f770w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f770w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f1000w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f1000w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f1130w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f1130w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)

        ax_jwst_f2100w.coords['ra'].set_ticklabel(rotation=0)
        ax_jwst_f2100w.coords['ra'].set_axislabel('R.A. (2000.0)', minpad=0.8, fontsize=fontsize)


        ax_hst_f275w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f275w.coords['ra'].set_axislabel(' ')

        ax_hst_f438w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f438w.coords['ra'].set_axislabel(' ')

        ax_hst_f555w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f555w.coords['ra'].set_axislabel(' ')

        ax_hst_f814w.coords['ra'].set_ticklabel_visible(False)
        ax_hst_f814w.coords['ra'].set_axislabel(' ')


        ax_jwst_f200w.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f200w.coords['ra'].set_axislabel(' ')

        ax_jwst_f300m.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f300m.coords['ra'].set_axislabel(' ')

        ax_jwst_f335m.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f335m.coords['ra'].set_axislabel(' ')

        ax_jwst_f360m.coords['ra'].set_ticklabel_visible(False)
        ax_jwst_f360m.coords['ra'].set_axislabel(' ')


        ax_hst_f438w.coords['dec'].set_ticklabel_visible(False)
        ax_hst_f438w.coords['dec'].set_axislabel(' ')

        ax_hst_f555w.coords['dec'].set_ticklabel_visible(False)
        ax_hst_f555w.coords['dec'].set_axislabel(' ')

        ax_hst_f814w.coords['dec'].set_ticklabel_visible(False)
        ax_hst_f814w.coords['dec'].set_axislabel(' ')

        ax_jwst_f300m.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f300m.coords['dec'].set_axislabel(' ')

        ax_jwst_f335m.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f335m.coords['dec'].set_axislabel(' ')

        ax_jwst_f360m.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f360m.coords['dec'].set_axislabel(' ')

        ax_jwst_f1000w.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f1000w.coords['dec'].set_axislabel(' ')

        ax_jwst_f1130w.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f1130w.coords['dec'].set_axislabel(' ')

        ax_jwst_f2100w.coords['dec'].set_ticklabel_visible(False)
        ax_jwst_f2100w.coords['dec'].set_axislabel(' ')



        ax_hst_f275w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_hst_f438w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_hst_f555w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_hst_f814w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f200w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f300m.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f335m.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f360m.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f770w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f1000w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f1130w.coords['ra'].set_ticks(number=ra_tick_num)
        ax_jwst_f2100w.coords['ra'].set_ticks(number=ra_tick_num)

        ax_hst_f275w.coords['ra'].display_minor_ticks(True)
        ax_hst_f438w.coords['ra'].display_minor_ticks(True)
        ax_hst_f555w.coords['ra'].display_minor_ticks(True)
        ax_hst_f814w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f200w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f300m.coords['ra'].display_minor_ticks(True)
        ax_jwst_f335m.coords['ra'].display_minor_ticks(True)
        ax_jwst_f360m.coords['ra'].display_minor_ticks(True)
        ax_jwst_f770w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f1000w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f1130w.coords['ra'].display_minor_ticks(True)
        ax_jwst_f2100w.coords['ra'].display_minor_ticks(True)


        ax_hst_f275w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_hst_f438w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_hst_f555w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_hst_f814w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f200w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f300m.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f335m.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f360m.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f770w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f1000w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f1130w.coords['dec'].set_ticks(number=dec_tick_num)
        ax_jwst_f2100w.coords['dec'].set_ticks(number=dec_tick_num)


        ax_hst_f275w.coords['dec'].display_minor_ticks(True)
        ax_hst_f438w.coords['dec'].display_minor_ticks(True)
        ax_hst_f555w.coords['dec'].display_minor_ticks(True)
        ax_hst_f814w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f200w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f300m.coords['dec'].display_minor_ticks(True)
        ax_jwst_f335m.coords['dec'].display_minor_ticks(True)
        ax_jwst_f360m.coords['dec'].display_minor_ticks(True)
        ax_jwst_f770w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f1000w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f1130w.coords['dec'].display_minor_ticks(True)
        ax_jwst_f2100w.coords['dec'].display_minor_ticks(True)

        return figure

