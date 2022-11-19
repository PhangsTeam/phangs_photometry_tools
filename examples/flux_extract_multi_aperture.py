"""
Example for a circular flux extraction in multiple bands
"""
import os.path
from astropy.coordinates import SkyCoord
import astropy.units as u
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
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='mJy')

ra_center = 24.173946 - 33.5 / 3600
dec_center = 15.783662 - 27.5 / 3600
# size of image
size_of_cutout = (5, 5)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_center, dec_cutout=dec_center,
                                                     cutout_size=size_of_cutout, include_err=True)


source_pos_1 = SkyCoord(ra=ra_center - 0.11 / 3600, dec=dec_center - 0.21 / 3600, unit=(u.degree, u.degree), frame='fk5')
source_pos_2 = SkyCoord(ra=ra_center + 0.41 / 3600, dec=dec_center + 0.03 / 3600, unit=(u.degree, u.degree), frame='fk5')
source_pos_3 = SkyCoord(ra=ra_center + 0.2 / 3600, dec=dec_center + 0.96 / 3600, unit=(u.degree, u.degree), frame='fk5')
source_list = [source_pos_1, source_pos_2, source_pos_3]

aperture_dict_list = []

for source in source_list:

    # compute flux from 50% encircled energy
    aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                          recenter=True)
    # print the measured fluxes
    for band in cutout_dict['band_list']:
        print(band, ' = %.6f ' % aperture_dict['aperture_dict_%s' % band]['flux'], '+/- ',
              '%.6f mJy' % aperture_dict['aperture_dict_%s' % band]['flux_err'])
    aperture_dict_list.append(aperture_dict)

# for plotting we want to use MJy/sr thus we convert the flux and
phangs_photometry.change_hst_nircam_miri_band_units(new_unit='MJy/sr')
# cutout_dict_new = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_center, dec_cutout=dec_center,
#                                                          cutout_size=size_of_cutout, include_err=True)

# sort bands
hst_bands = phangs_photometry.sort_band_list(
    band_list=(phangs_photometry.hst_targets[phangs_photometry.target_name]['acs_wfc1_observed_bands'] +
               phangs_photometry.hst_targets[phangs_photometry.target_name]['wfc3_uvis_observed_bands']))

for source_index in range(len(source_list)):

    # plot the results
    fig = PlotPhotometry.plot_circ_flux_extraction(
        hst_band_list=hst_bands,
        nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
        miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],
        cutout_dict=cutout_dict, aperture_dict=aperture_dict_list[source_index],
        vmax_vmin_hst=(0.05, 1.5), vmax_vmin_nircam=(0.1, 21), vmax_vmin_miri=(0.1, 21),
        # cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
        cmap_hst='Greys', cmap_nircam='Greys', cmap_miri='Greys',
        log_scale=True, axis_length=axis_length)

    if not os.path.isdir('plot_output'):
        os.makedirs('plot_output')

    fig.savefig('plot_output/circ_aperture_%i.png' % source_index)
    fig.clf()
