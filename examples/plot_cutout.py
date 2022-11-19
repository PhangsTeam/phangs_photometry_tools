"""
Example to create a cutout plot
"""
import os

from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools.plotting_tools import PlotPhotometry

from astropy.coordinates import SkyCoord
import astropy.units as u

# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name='ngc0628',
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')

# load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='MJy/sr')

# get dictionary with cutouts
# # embedded source
# ra_center = 24.173946 + 11 / 3600
# dec_center = 15.783662 - 10.1 / 3600

ra_center = 24.173946 - 33.5 / 3600
dec_center = 15.783662 - 27.5 / 3600
# size of image
size_of_cutout = (4, 4)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_center, dec_cutout=dec_center,
                                                     cutout_size=size_of_cutout)


circ_pos_1 = SkyCoord(ra=ra_center - 0.11 / 3600, dec=dec_center - 0.21 / 3600, unit=(u.degree, u.degree), frame='fk5')
circ_pos_2 = SkyCoord(ra=ra_center + 0.41 / 3600, dec=dec_center + 0.03 / 3600, unit=(u.degree, u.degree), frame='fk5')
circ_pos_3 = SkyCoord(ra=ra_center + 0.2 / 3600, dec=dec_center + 0.96 / 3600, unit=(u.degree, u.degree), frame='fk5')


# sort bands
hst_bands = phangs_photometry.sort_band_list(
    band_list=(phangs_photometry.hst_targets[phangs_photometry.target_name]['acs_wfc1_observed_bands'] +
               phangs_photometry.hst_targets[phangs_photometry.target_name]['wfc3_uvis_observed_bands']))


fig = PlotPhotometry.plot_cutout_panel_hst_nircam_miri(
    hst_band_list=hst_bands,
    nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
    miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],
    cutout_dict=cutout_dict,
    circ_pos=[circ_pos_1, circ_pos_2, circ_pos_3], circ_rad=[0.2, 0.2, 0.2], circ_color=['k', 'r', 'g'],
    vmax_vmin_hst=(0.05, 1.5), vmax_vmin_nircam=(0.1, 21), vmax_vmin_miri=(0.1, 21),
    # cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
    cmap_hst='Greys', cmap_nircam='Greys', cmap_miri='Greys',
    log_scale=True, axis_length=axis_length)


if not os.path.isdir('plot_output'):
    os.makedirs('plot_output')

fig.savefig('plot_output/cutout_example_1.png')
