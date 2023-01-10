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

ra = 24.173946 - 33.5 / 3600
dec = 15.783662 - 27.6 / 3600
# size of image
size_of_cutout = (5, 5)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
                                                     cutout_size=size_of_cutout, include_err=True)
source = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='fk5')

# get a list of all the hst bands in the correct order
hst_bands = phangs_photometry.sort_band_list(
    band_list=(phangs_photometry.hst_targets[phangs_photometry.target_name]['acs_wfc1_observed_bands'] +
               phangs_photometry.hst_targets[phangs_photometry.target_name]['wfc3_uvis_observed_bands']))


# compute flux from 50% encircled energy
aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                      recenter=True, recenter_rad=0.25,
                                                                      default_ee_rad=50)

# make the plot to inspect the aperture extraction
fig = PlotPhotometry.plot_circ_flux_extraction(
        hst_band_list=hst_bands,
        nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
        miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],
        cutout_dict=cutout_dict, aperture_dict=aperture_dict,
        # vmax_vmin_hst=(0.05, 1.5), vmax_vmin_nircam=(0.1, 21), vmax_vmin_miri=(0.1, 21),
        # cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
        cmap_hst='Greys', cmap_nircam='Greys', cmap_miri='Greys',
        log_scale=True, axis_length=axis_length)

if not os.path.isdir('plot_output'):
    os.makedirs('plot_output')

fig.savefig('plot_output/circ_aperture.png')
fig.clf()


# print the measured fluxes
for band in cutout_dict['band_list']:
    print(band, ' = %.6f ' % aperture_dict['aperture_dict_%s' % band]['flux'], '+/- ',
          '%.6f mJy' % aperture_dict['aperture_dict_%s' % band]['flux_err'])

print(aperture_dict)

exit()

# for plotting we want to use MJy/sr thus we convert the flux and
phangs_photometry.change_hst_nircam_miri_band_units(new_unit='MJy/sr')

# # # in case you want different cutout sizes
# cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec,
#                                                      cutout_size=[2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
#                                                      include_err=True)

figure = PlotPhotometry.plot_cigale_sed_panel(
        hst_band_list=hst_bands,
        nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
        miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],
        cutout_dict=cutout_dict, aperture_dict=aperture_dict)

figure.savefig('plot_output/test_sed.png')
figure.clf()

