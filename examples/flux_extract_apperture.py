"""
Example for a circular flux extraction in multiple bands
"""
import os.path
from astropy.coordinates import SkyCoord
import astropy.units as u
import photometry_tools.plotting_tools
from photometry_tools.analysis_tools import AnalysisTools
# from photometry_tools.plotting_tools import PlotPhotometry

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
ra_center = 24.161729333333334
dec_center = 15.777342555555556
pos = SkyCoord(ra=ra_center*u.arcsec, dec=dec_center*u.arcsec)
# size of image
size_of_cutout = (2, 2)

# get cutout dict
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_center, dec_cutout=dec_center,
                                                     cutout_size=size_of_cutout)


phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=pos, aperture_rad=None, recenter=False, recenter_rad=0.2)



