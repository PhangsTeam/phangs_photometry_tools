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

        # loaded data dictionaries
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
        if (band not in self.hst_acs_wfc1_bands) & (band not in self.hst_wfc3_uvis2_bands):
            raise AttributeError('The band <%s> is not in the list of possible HST bands.' % band)

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
        if (band not in self.hst_acs_wfc1_bands) & (band not in self.hst_wfc3_uvis2_bands):
            raise AttributeError('The band <%s> is not in the list of possible HST bands.' % band)

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
            raise AttributeError('The band <%s> is not in the list of possible NIRCAM bands.' % band)

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
            raise AttributeError('The band <%s> is not in the list of possible MIRI bands.' % band)

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
            raise AttributeError('The band <%s> is not in the list of possible MIRI bands.' % band)

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

        # for key in img_header.keys():
        #     print(key, img_header[key])

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
            f_nu = sensitivity * pivot_wavelength ** 2 / c
            # change to get Jy
            conversion_factor = f_nu * 1e23
        else:
            raise KeyError('there is no PHOTFNU or PHOTFLAM in the header')

        pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * self.sr_per_square_deg
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            conversion_factor = conversion_factor
        elif flux_unit == 'mJy':
            # rescale to Jy
            conversion_factor *= 1e3
        elif flux_unit == 'MJy/sr':
            # get the size of one pixel in sr with the factor 1e6 for the conversion of Jy to MJy later

            # change to MJy/sr
            conversion_factor /= (pixel_area_size_sr * 1e6)
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand!')

        img_data *= conversion_factor
        self.hst_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                    '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                    '%s_pixel_area_size_sr_img' % band: pixel_area_size_sr})
        if load_err:
            err_file_name = self.get_hst_err_file_name(band=band)
            err_data, err_header, err_wcs = helper_func.load_img(file_name=err_file_name)
            err_data = 1 / np.sqrt(err_data)
            err_data *= conversion_factor
            self.hst_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                        '%s_wcs_err' % band: err_wcs, '%s_unit_err' % band: flux_unit,
                                        '%s_pixel_area_size_sr_err' % band: pixel_area_size_sr})

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
        pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * self.sr_per_square_deg
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            conversion_factor = pixel_area_size_sr * 1e6

        elif flux_unit == 'mJy':
            # rescale to Jy
            conversion_factor = pixel_area_size_sr * 1e9
        elif flux_unit == 'MJy/sr':
            conversion_factor = 1
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand')

        # add flux zero-point correction
        if self.nircam_data_ver == 'v0p4p2':
            conversion_factor *= self.nircam_zero_point_flux_corr[band]

        img_data *= conversion_factor
        self.nircam_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                       '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                       '%s_pixel_area_size_sr_img' % band: pixel_area_size_sr})
        if load_err:
            err_data, err_header, err_wcs = helper_func.load_img(file_name=file_name, hdu_number='ERR')
            err_data *= conversion_factor
            # use the img_wcs for the version v0p4p2 because the WCS for the error band is not scaled.
            # This might be corrected in later versions.
            if self.nircam_data_ver == 'v0p4p2':
                selected_wcs = img_wcs
            else:
                selected_wcs = err_wcs
            self.nircam_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                           '%s_wcs_err' % band: selected_wcs, '%s_unit_err' % band: flux_unit,
                                           '%s_pixel_area_size_sr_err' % band: pixel_area_size_sr})

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
        pixel_area_size_sr = img_wcs.proj_plane_pixel_area().value * self.sr_per_square_deg
        # rescale data image
        if flux_unit == 'Jy':
            # rescale to Jy
            conversion_factor = pixel_area_size_sr * 1e6

        elif flux_unit == 'mJy':
            # rescale to Jy
            conversion_factor = pixel_area_size_sr * 1e9
        elif flux_unit == 'MJy/sr':
            conversion_factor = 1
        else:
            raise KeyError('flux_unit ', flux_unit, ' not understand')

        img_data *= conversion_factor
        self.miri_bands_data.update({'%s_data_img' % band: img_data, '%s_header_img' % band: img_header,
                                     '%s_wcs_img' % band: img_wcs, '%s_unit_img' % band: flux_unit,
                                     '%s_pixel_area_size_sr_img' % band: pixel_area_size_sr})
        if load_err:
            err_file_name = self.get_miri_err_file_name(band=band)
            err_data, err_header, err_wcs = helper_func.load_img(file_name=err_file_name)
            err_data *= conversion_factor
            self.miri_bands_data.update({'%s_data_err' % band: err_data, '%s_header_err' % band: err_header,
                                         '%s_wcs_err' % band: err_wcs, '%s_unit_err' % band: flux_unit,
                                         '%s_pixel_area_size_sr_err' % band: pixel_area_size_sr})

    def load_hst_nircam_miri_bands(self, band_list=None, flux_unit='Jy'):
        """
        wrapper to load all available HST, NIRCAM and MIRI observations into the constructor

        Parameters
        ----------
        band_list : list
        flux_unit : str
        """
        # geta list with all observed bands in order of wavelength
        if band_list is None:
            band_list = []
            for band in list(set(self.hst_acs_wfc1_bands + self.hst_wfc3_uvis2_bands)):
                if band in (self.hst_targets[self.target_name]['acs_wfc1_observed_bands'] +
                            self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']):
                    band_list.append(band)
            for band in self.nircam_bands:
                if band in self.nircam_targets[self.target_name]['observed_bands']:
                    band_list.append(band)
            for band in self.miri_bands:
                if band in self.miri_targets[self.target_name]['observed_bands']:
                    band_list.append(band)

        # sort band list with increasing wavelength
        band_list = self.sort_band_list(band_list=band_list)
        for band in band_list:
            if band in list(set(self.hst_acs_wfc1_bands + self.hst_wfc3_uvis2_bands)):
                self.load_hst_band(band=band, flux_unit=flux_unit)
            elif band in self.nircam_bands:
                self.load_nircam_band(band=band, flux_unit=flux_unit)
            elif band in self.miri_bands:
                self.load_miri_band(band=band, flux_unit=flux_unit)
            else:
                raise KeyError('Band is not found in possible band lists')

        # for hst_band in self.hst_acs_wfc1_bands + self.hst_wfc3_uvis2_bands:
        #     if hst_band in band_list:
        #         self.load_hst_band(band=hst_band, flux_unit=flux_unit)
        # for nircam_band in self.nircam_bands:
        #     if nircam_band in band_list:
        #         self.load_nircam_band(band=nircam_band, flux_unit=flux_unit)
        # for miri_band in self.miri_bands:
        #     if miri_band in band_list:
        #         self.load_miri_band(band=miri_band, flux_unit=flux_unit)

    def change_hst_nircam_miri_band_units(self, band_list=None, new_unit='MJy/sr'):
        """

        Parameters
        ----------
        band_list : list
        new_unit : str
        """
        if band_list is None:
            band_list = []
            for band in list(set(self.hst_acs_wfc1_bands + self.hst_wfc3_uvis2_bands)):
                if band in (self.hst_targets[self.target_name]['acs_wfc1_observed_bands'] +
                            self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']):
                    band_list.append(band)
            for band in self.nircam_bands:
                if band in self.nircam_targets[self.target_name]['observed_bands']:
                    band_list.append(band)
            for band in self.miri_bands:
                if band in self.miri_targets[self.target_name]['observed_bands']:
                    band_list.append(band)

        for band in band_list:
            self.change_band_unit(band=band, new_unit=new_unit)

    def change_band_unit(self, band, new_unit='MJy/sr'):
        """
        will change loaded data to the needed unit
        Parameters
        ----------
        band : str
        new_unit : str
        """
        if band in list(set(self.hst_acs_wfc1_bands + self.hst_wfc3_uvis2_bands)):
            old_unit = self.hst_bands_data['%s_unit_img' % band]
            conversion_factor = 1
            # change to Jy
            if 'm' in old_unit:
                conversion_factor *= 1e-3
            if 'M' in old_unit:
                conversion_factor *= 1e6
            if '/sr' in old_unit:
                conversion_factor *= self.hst_bands_data['%s_pixel_area_size_sr_err' % band]

            # change to the new unit
            if 'm' in new_unit:
                conversion_factor *= 1e3
            if 'M' in new_unit:
                conversion_factor *= 1e-6
            if '/sr' in new_unit:
                conversion_factor /= self.hst_bands_data['%s_pixel_area_size_sr_err' % band]

            self.hst_bands_data['%s_data_img' % band] *= conversion_factor
            self.hst_bands_data['%s_unit_img' % band] = new_unit
            if '%s_data_err' % band in self.hst_bands_data.keys():
                self.hst_bands_data['%s_data_err' % band] *= conversion_factor
                self.hst_bands_data['%s_unit_err' % band] = new_unit

        if band in self.nircam_bands:
            old_unit = self.nircam_bands_data['%s_unit_img' % band]

            conversion_factor = 1
            # change to Jy
            if 'm' in old_unit:
                conversion_factor *= 1e-3
            if 'M' in old_unit:
                conversion_factor *= 1e6
            if '/sr' in old_unit:
                conversion_factor *= self.nircam_bands_data['%s_pixel_area_size_sr_err' % band]

            # change to the new unit
            if 'm' in new_unit:
                conversion_factor *= 1e3
            if 'M' in new_unit:
                conversion_factor *= 1e-6
            if '/sr' in new_unit:
                conversion_factor /= self.nircam_bands_data['%s_pixel_area_size_sr_err' % band]

            self.nircam_bands_data['%s_data_img' % band] *= conversion_factor
            self.nircam_bands_data['%s_unit_img' % band] = new_unit

            if '%s_data_err' % band in self.nircam_bands_data.keys():
                self.nircam_bands_data['%s_data_err' % band] *= conversion_factor
                self.nircam_bands_data['%s_unit_err' % band] = new_unit

        if band in self.miri_bands:
            old_unit = self.miri_bands_data['%s_unit_img' % band]

            conversion_factor = 1
            # change to Jy
            if 'm' in old_unit:
                conversion_factor *= 1e-3
            if 'M' in old_unit:
                conversion_factor *= 1e6
            if '/sr' in old_unit:
                conversion_factor *= self.miri_bands_data['%s_pixel_area_size_sr_err' % band]

            # change to the new unit
            if 'm' in new_unit:
                conversion_factor *= 1e3
            if 'M' in new_unit:
                conversion_factor *= 1e-6
            if '/sr' in new_unit:
                conversion_factor /= self.miri_bands_data['%s_pixel_area_size_sr_err' % band]

            self.miri_bands_data['%s_data_img' % band] *= conversion_factor
            self.miri_bands_data['%s_unit_img' % band] = new_unit
            if '%s_data_err' % band in self.miri_bands_data.keys():
                self.miri_bands_data['%s_data_err' % band] *= conversion_factor
                self.miri_bands_data['%s_unit_err' % band] = new_unit

    def get_band_cutout_dict(self, ra_cutout, dec_cutout, cutout_size, include_err=False, band_list=None):
        """

        Parameters
        ----------
        ra_cutout : float
        dec_cutout : float
        cutout_size : float, tuple or list
            Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.
        include_err : bool
        band_list : list

        Returns
        -------
        cutout_dict : dict
        each element in dictionary is of type astropy.nddata.Cutout2D object
        """
        # geta list with all observed bands in order of wavelength
        if band_list is None:
            band_list = []
            for band in list(set(self.hst_acs_wfc1_bands + self.hst_wfc3_uvis2_bands)):
                if band in (self.hst_targets[self.target_name]['acs_wfc1_observed_bands'] +
                            self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']):
                    band_list.append(band)
            for band in self.nircam_bands:
                if band in self.nircam_targets[self.target_name]['observed_bands']:
                    band_list.append(band)
            for band in self.miri_bands:
                if band in self.miri_targets[self.target_name]['observed_bands']:
                    band_list.append(band)
            # sort bands in increasing order
            band_list = self.sort_band_list(band_list=band_list)

        if not isinstance(cutout_size, list):
            cutout_size = [cutout_size] * len(band_list)

        cutout_pos = SkyCoord(ra=ra_cutout, dec=dec_cutout, unit=(u.degree, u.degree), frame='fk5')
        cutout_dict = {'cutout_pos': cutout_pos}
        cutout_dict.update({'band_list': band_list})

        for band, band_index in zip(band_list, range(len(band_list))):
            if band in list(set(self.hst_acs_wfc1_bands + self.hst_wfc3_uvis2_bands)):
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.hst_bands_data['%s_data_img' % band],
                                                   wcs=self.hst_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.hst_bands_data['%s_data_err' % band],
                                                       wcs=self.hst_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})

            elif band in self.nircam_bands:
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.nircam_bands_data['%s_data_img' % band],
                                                   wcs=self.nircam_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.nircam_bands_data['%s_data_err' % band],
                                                       wcs=self.nircam_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})

            elif band in self.miri_bands:
                cutout_dict.update({
                    '%s_img_cutout' % band:
                        helper_func.get_img_cutout(img=self.miri_bands_data['%s_data_img' % band],
                                                   wcs=self.miri_bands_data['%s_wcs_img' % band],
                                                   coord=cutout_pos, cutout_size=cutout_size[band_index])})
                if include_err:
                    cutout_dict.update({
                        '%s_err_cutout' % band:
                            helper_func.get_img_cutout(img=self.miri_bands_data['%s_data_err' % band],
                                                       wcs=self.miri_bands_data['%s_wcs_err' % band],
                                                       coord=cutout_pos, cutout_size=cutout_size[band_index])})

        return cutout_dict

    def get_band_wave(self, band, unit='mu'):
        """
        Returns mean wavelength of a specific band
        Parameters
        ----------
        unit : str
        band : str

        Returns
        -------
        wavelength : float
        """
        if band in self.hst_targets[self.target_name]['acs_wfc1_observed_bands']:
            wave = self.hst_acs_wfc1_bands_mean_wave[band]
        elif band in self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']:
            wave = self.hst_wfc3_uvis1_bands_mean_wave[band]
        elif band in self.nircam_targets[self.target_name]['observed_bands']:
            wave = self.nircam_bands_mean_wave[band]
        elif band in self.miri_targets[self.target_name]['observed_bands']:
            wave = self.miri_bands_mean_wave[band]
        else:
            raise KeyError(band, 'not understand')

        if unit == 'angstrom':
            return wave
        if unit == 'nano':
            return wave * 1e-1
        elif unit == 'mu':
            return wave * 1e-4
        else:
            raise KeyError('return unit not understand')

    def sort_band_list(self, band_list):
        """
        sorts a band list with increasing wavelength
        Parameters
        ----------
        band_list : list

        Returns
        -------
        sorted_band_list : list
        """
        wave_list = []
        for band in band_list:
            wave_list.append(self.get_band_wave(band=band))

        # print(len(band_list))
        # print(len(wave_list))

        # sort wavelength bands
        sort = np.argsort(wave_list)
        # print(sort)
        return list(np.array(band_list)[sort])

    def create_cigale_flux_file(self, file_path, band_list, aperture_dict_list, snr=3, name_list=None,
                                redshift_list=None, dist_list=None):

        if isinstance(aperture_dict_list, dict):
            aperture_dict_list = [aperture_dict_list]

        if name_list is None:
            name_list = np.arange(start=0,  stop=len(aperture_dict_list)+1)
        if redshift_list is None:
            redshift_list = [0.0] * len(aperture_dict_list)
        if dist_list is None:
            dist_list = [self.dist_dict[self.target_name]['dist']] * len(aperture_dict_list)

        name_list = np.array(name_list, dtype=str)
        redshift_list = np.array(redshift_list)
        dist_list = np.array(dist_list)


        # create flux file
        flux_file = open(file_path, "w")
        # add header for all variables
        band_name_list = self.compute_cigale_band_name_list(band_list=band_list)
        flux_file.writelines("# id             redshift  distance   ")
        for band_name in band_name_list:
            flux_file.writelines(band_name + "   ")
            flux_file.writelines(band_name + "_err" + "   ")
        flux_file.writelines(" \n")

        # fill flux file
        for name, redshift, dist, aperture_dict in zip(name_list, redshift_list, dist_list, aperture_dict_list):
            flux_file.writelines(" %s   %f   %f  " % (name, redshift, dist))
            flux_list, flux_err_list = self.compute_cigale_flux_list(band_list=band_list, aperture_dict=aperture_dict,
                                                                     snr=snr)
            for flux, flux_err in zip(flux_list, flux_err_list):
                flux_file.writelines("%.15f   " % flux)
                flux_file.writelines("%.15f   " % flux_err)
            flux_file.writelines(" \n")

        flux_file.close()

    def compute_cigale_flux_list(self, band_list, aperture_dict, snr=3):

        flux_list = []
        flux_err_list = []

        for band in band_list:
            flux_list.append(aperture_dict['aperture_dict_%s' % band]['flux'])
            flux_err_list.append(aperture_dict['aperture_dict_%s' % band]['flux_err'])

        flux_list = np.array(flux_list)
        flux_err_list = np.array(flux_err_list)

        # compute the upper limits
        upper_limits = (flux_list < 0) | (flux_list/flux_err_list < snr)
        flux_list[upper_limits] = flux_err_list[upper_limits]
        flux_err_list[upper_limits] *= -1

        return flux_list, flux_err_list

    def compute_cigale_band_name_list(self, band_list):

        band_name_list = []
        for band in band_list:
            if band in self.hst_targets[self.target_name]['acs_wfc1_observed_bands']:
                band_name_list.append(band + '_ACS')
            elif band in self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']:
                band_name_list.append(band + '_UVIS_CHIP2')
            elif band in self.nircam_targets[self.target_name]['observed_bands']:
                band_name_list.append(band + 'jwst.nircam.' + band)
            elif band in self.miri_targets[self.target_name]['observed_bands']:
                band_name_list.append(band + 'jwst.miri.' + band)

        return band_name_list


