"""
Construct a data access structure for HST and JWST imaging data
"""
import numpy as np
from scipy.constants import c as speed_of_light
from pathlib import Path
from astropy.coordinates import SkyCoord
import astropy.units as u

from photometry_tools import basic_attributes, helper_func


class DataAccess(basic_attributes.PhangsDataStructure, basic_attributes.PhysParams):
    """
    Access class to organize data structure of HST, NIRCAM and MIRI imaging data
    """
    def __init__(self, hst_data_path=None, nircam_data_path=None, miri_data_path=None, target_name=None,
                 hst_data_ver='v1', nircam_data_ver='v0p4p2', miri_data_ver='v0p5'):
        """
        Parameters
        ----------
        hst_data_path : str
            Default None. Path to HST imaging data
        nircam_data_path : str
            Default None. Path to NIRCAM imaging data
        miri_data_path : str
            Default None. Path to MIRI imaging data
        target_name : str
            Default None. Target name
        """
        super().__init__()

        self.hst_data_path = Path(hst_data_path)
        self.nircam_data_path = Path(nircam_data_path)
        self.miri_data_path = Path(miri_data_path)
        self.target_name = target_name
        if (self.target_name not in self.target_list) & (self.target_name is not None):
            raise AttributeError('The target %s is not in the PHANGS photometric sample or has not been added to '
                                 'the current package version' % self.target_name)

        self.hst_data_ver = hst_data_ver
        self.nircam_data_ver = nircam_data_ver
        self.miri_data_ver = miri_data_ver

        self.hst_bands_data = {}
        self.nircam_bands_data = {}
        self.miri_bands_data = {}

    def get_hst_img_file_name(self, band):
        """

        Parameters
        ----------
        band : str

        Returns
        -------
        data_file_path : Path
        """
        if band not in self.hst_bands:
            raise AttributeError('The band <%s> is not in the list of possible HST bands it must be one of ' % band,
                                 [band for band in self.hst_bands])

        hst_data_folder = (self.hst_data_path / self.hst_ver_folder_names[self.hst_data_ver] /
                           self.hst_targets[self.target_name]['folder_name'])
        ending_of_band_file = '%s_%s_exp-drc-sci.fits' % (band.lower(), self.hst_data_ver)

        return helper_func.identify_file_in_folder(folder_path=hst_data_folder, str_in_file_name=ending_of_band_file)

    def get_hst_err_file_name(self, band):
        """

        Parameters
        ----------
        band : str

        Returns
        -------
        data_file_path : Path
        """
        if band not in self.hst_bands:
            raise AttributeError('The band <%s> is not in the list of possible HST bands it must be one of ' % band,
                                 [band for band in self.hst_bands])

        hst_data_folder = (self.hst_data_path / self.hst_ver_folder_names[self.hst_data_ver] /
                           self.hst_targets[self.target_name]['folder_name'])
        ending_of_band_file = '%s_%s_err-drc-wht.fits' % (band.lower(), self.hst_data_ver)

        return helper_func.identify_file_in_folder(folder_path=hst_data_folder, str_in_file_name=ending_of_band_file)

    def get_nircam_img_file_name(self, band):
        """

        Parameters
        ----------
        band : str

        Returns
        -------
        data_file_path : Path
        """
        if band not in self.nircam_bands:
            raise AttributeError('The band <%s> is not in the list of possible NIRCAM bands it must be one of ' % band,
                                 [band for band in self.hst_bands])

        nircam_data_folder = (self.nircam_data_path / self.nircam_ver_folder_names[self.nircam_data_ver] /
                              self.nircam_targets[self.target_name]['folder_name'])
        ending_of_band_file = 'nircam_lv3_%s_i2d_align.fits' % band.lower()

        return helper_func.identify_file_in_folder(folder_path=nircam_data_folder, str_in_file_name=ending_of_band_file)

    def get_miri_img_file_name(self, band):
        """

        Parameters
        ----------
        band : str

        Returns
        -------
        data_file_path : Path
        """
        if band not in self.miri_bands:
            raise AttributeError('The band <%s> is not in the list of possible MIRI bands it must be one of ' % band,
                                 [band for band in self.hst_bands])

        miri_data_folder = self.miri_data_path / self.miri_ver_folder_names[self.miri_data_ver]
        ending_of_band_file = '%s_miri_%s_anchored.fits' % (self.target_name, band.lower())

        return helper_func.identify_file_in_folder(folder_path=miri_data_folder, str_in_file_name=ending_of_band_file)

    def get_miri_err_file_name(self, band):
        """

        Parameters
        ----------
        band : str

        Returns
        -------
        data_file_path : Path
        """
        if band not in self.miri_bands:
            raise AttributeError('The band <%s> is not in the list of possible MIRI bands it must be one of ' % band,
                                 [band for band in self.hst_bands])

        miri_data_folder = self.miri_data_path / self.miri_ver_folder_names[self.miri_data_ver]
        ending_of_band_file = '%s_miri_%s_noisemap.fits' % (self.target_name, band.lower())

        return helper_func.identify_file_in_folder(folder_path=miri_data_folder, str_in_file_name=ending_of_band_file)

    def load_hst_band(self, band, load_err=True, flux_unit='Jy'):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        """
        # load the band observations
        img_file_name = self.get_hst_img_file_name(band=band)
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
            f_nu = sensitivity * pivot_wavelength**2 / c
            # change to get Jy
            conversion_factor = f_nu * 1e23
        else:
            raise KeyError('there is no PHOTFNU or PHOTFLAM in the header')
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            conversion_factor = conversion_factor
        elif flux_unit == 'mJy':
            # rescale to Jy
            conversion_factor *= 1e3
        elif flux_unit == 'MJy/sr':
            # get the size of one pixel in sr with the factor 1e6 for the conversion of Jy to MJy later
            pixel_area_size = img_wcs.proj_plane_pixel_area() * self.sr_per_square_deg * 1e6
            # change to MJy/sr
            conversion_factor /= pixel_area_size.value
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand!')

        img_data *= conversion_factor
        self.hst_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                   '%s_wcs_img' % band: img_wcs})
        if load_err:
            err_file_name = self.get_hst_err_file_name(band=band)
            err_data, err_header, err_wcs = helper_func.load_img(file_name=err_file_name)
            err_data *= conversion_factor
            self.hst_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                       '%s_wcs_err' % band: err_wcs})

    def load_nircam_band(self, band, load_err=True, flux_unit='Jy'):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        """
        # load the band observations
        file_name = self.get_nircam_img_file_name(band=band)
        img_data, img_header, img_wcs = helper_func.load_img(file_name=file_name, hdu_number='SCI')
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            pixel_area_size = img_wcs.proj_plane_pixel_area() * self.sr_per_square_deg
            conversion_factor = pixel_area_size.value * 1e6

        elif flux_unit == 'mJy':
            # rescale to Jy
            pixel_area_size = img_wcs.proj_plane_pixel_area() * self.sr_per_square_deg
            conversion_factor = pixel_area_size.value * 1e9
        elif flux_unit == 'MJy/sr':
            conversion_factor = 1
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand')

        img_data *= conversion_factor
        self.nircam_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                      '%s_wcs_img' % band: img_wcs})
        if load_err:
            err_data, err_header, err_wcs = helper_func.load_img(file_name=file_name, hdu_number='ERR')
            err_data *= conversion_factor
            self.nircam_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                          '%s_wcs_err' % band: err_wcs})

    def load_miri_band(self, band, load_err=True, flux_unit='Jy'):
        """

        Parameters
        ----------
        band : str
        load_err : bool
        flux_unit : str
        """
        # load the band observations
        file_name = self.get_miri_img_file_name(band=band)
        img_data, img_header, img_wcs = helper_func.load_img(file_name=file_name)
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            pixel_area_size = img_wcs.proj_plane_pixel_area() * self.sr_per_square_deg
            conversion_factor = pixel_area_size.value * 1e6

        elif flux_unit == 'mJy':
            # rescale to Jy
            pixel_area_size = img_wcs.proj_plane_pixel_area() * self.sr_per_square_deg
            conversion_factor = pixel_area_size.value * 1e9
        elif flux_unit == 'MJy/sr':
            conversion_factor = 1
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand')

        img_data *= conversion_factor
        self.miri_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                    '%s_wcs_img' % band: img_wcs})
        if load_err:
            err_file_name = self.get_miri_err_file_name(band=band)
            err_data, err_header, err_wcs = helper_func.load_img(file_name=err_file_name)
            err_data *= conversion_factor
            self.miri_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                        '%s_wcs_err' % band: err_wcs})

    def load_hst_nircam_miri_bands(self, band_list=None, flux_unit='Jy'):
        """
        wrapper to load all available HST, NIRCAM and MIRI observations into the constructor

        Parameters
        ----------
        band_list : list
        flux_unit : str
        """
        if band_list is None:
            band_list = (self.hst_targets[self.target_name]['observed_bands'] +
                         self.nircam_targets[self.target_name]['observed_bands'] +
                         self.miri_targets[self.target_name]['observed_bands'])

        for hst_band in self.hst_bands:
            if hst_band in band_list:
                self.load_hst_band(band=hst_band, flux_unit=flux_unit)
        for nircam_band in self.nircam_bands:
            if nircam_band in band_list:
                self.load_nircam_band(band=nircam_band, flux_unit=flux_unit)
        for miri_band in self.miri_bands:
            if miri_band in band_list:
                self.load_miri_band(band=miri_band, flux_unit=flux_unit)

    def get_band_cutout_dict(self, ra_cutout, dec_cutout, cutout_size, include_err=False, band_list=None):
        """

        Parameters
        ----------
        ra_cutout : float
        dec_cutout : float
        cutout_size : float or tuple
            Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.
        include_err : bool
        band_list : list

        Returns
        -------
        cutout_dict : dict
        each element in dictionary is of type astropy.nddata.Cutout2D object
        """
        if band_list is None:
            band_list = (self.hst_targets[self.target_name]['observed_bands'] +
                         self.nircam_targets[self.target_name]['observed_bands'] +
                         self.miri_targets[self.target_name]['observed_bands'])

        cutout_pos = SkyCoord(ra=ra_cutout, dec=dec_cutout, unit=(u.degree, u.degree), frame='fk5')
        cutout_dict = {'cutout_pos': cutout_pos}
        for hst_band in self.hst_bands:
            if hst_band in band_list:
                cutout_dict.update({
                    '%s_img_cutout' % hst_band:
                        helper_func.get_img_cutout(img=self.hst_bands_data['%s_data_img' % hst_band],
                                                   wcs=self.hst_bands_data['%s_wcs_img' % hst_band],
                                                   coord=cutout_pos, cutout_size=cutout_size)})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % hst_band:
                            helper_func.get_img_cutout(img=self.hst_bands_data['%s_data_err' % hst_band],
                                                       wcs=self.hst_bands_data['%s_wcs_err' % hst_band],
                                                       coord=cutout_pos, cutout_size=cutout_size)})
        for nircam_band in self.nircam_bands:
            if nircam_band in band_list:
                cutout_dict.update({
                    '%s_img_cutout' % nircam_band:
                        helper_func.get_img_cutout(img=self.nircam_bands_data['%s_data_img' % nircam_band],
                                                   wcs=self.nircam_bands_data['%s_wcs_img' % nircam_band],
                                                   coord=cutout_pos, cutout_size=cutout_size)})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % nircam_band:
                            helper_func.get_img_cutout(img=self.nircam_bands_data['%s_data_err' % nircam_band],
                                                       wcs=self.nircam_bands_data['%s_wcs_err' % nircam_band],
                                                       coord=cutout_pos, cutout_size=cutout_size)})
        for miri_band in self.miri_bands:
            if miri_band in band_list:
                cutout_dict.update({
                    '%s_img_cutout' % miri_band:
                        helper_func.get_img_cutout(img=self.miri_bands_data['%s_data_img' % miri_band],
                                                   wcs=self.miri_bands_data['%s_wcs_img' % miri_band],
                                                   coord=cutout_pos, cutout_size=cutout_size)})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % miri_band:
                            helper_func.get_img_cutout(
                                img=self.miri_bands_data['%s_data_err' % miri_band],
                                wcs=self.miri_bands_data['%s_wcs_err' % miri_band],
                                coord=cutout_pos, cutout_size=cutout_size)})

        return cutout_dict

