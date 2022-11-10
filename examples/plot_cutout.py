"""
Example to create a cutout plot
"""

from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools.plotting_tools import PlotPhotometry

# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                                                  target_name='ngc0628',
                                                                  hst_data_ver='v1',
                                                                  nircam_data_ver='v0p4p2',
                                                                  miri_data_ver='v0p5')

# load all data
phangs_photometry.load_hst_nircam_miri_bands()

# get dictionary with cutouts
ra_center = 24.173946
dec_center = 15.783662
# size of image
size_of_cutout = (10, 10)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_center, dec_cutout=dec_center,
                                                     cutout_size=size_of_cutout)


# plot the cutouts

PlotPhotometry.plot_multi_zoom_panel_hst_nircam_miri(
    hst_band_list=phangs_photometry.hst_targets[phangs_photometry.target_name]['observed_bands'],
    nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
    miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],

                                              cutout_dict=cutout_dict,
                                              circ_pos=False, circ_rad=0.5,
                                              fontsize=20, figsize=(18, 13),
                                              vmax_hst=None, vmax_nircam=None, vmax_miri=None,
                                              ticks_hst=None, ticks_nircam=None, ticks_miri=None,
                                              cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
                                              log_scale=False,
                                              name_ra_offset=2.4, name_dec_offset=1.5,
                                              ra_tick_num=3, dec_tick_num=3)



