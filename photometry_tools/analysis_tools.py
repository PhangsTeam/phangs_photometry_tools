"""
Gather all photometric tools for HST and JWST photometric observations
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import sep

from photometry_tools import data_access, helper_func

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.table import QTable, hstack

class AnalysisTools(data_access.DataAccess):
    """
    Access class to organize data structure of HST, NIRCAM and MIRI imaging data
    """

    def __init__(self, **kwargs):
        """

        """
        super().__init__(**kwargs)

    def circular_flux_aperture_from_cutouts(self, cutout_dict, pos, apertures=None, recenter=False, recenter_rad=0.2,
                                            default_ee_rad=50):
        """

        Parameters
        ----------
        cutout_dict : dict
        pos : ``astropy.coordinates.SkyCoord``
        apertures : float or list of float
        recenter : bool
        recenter_rad : float
        default_ee_rad : int
            is either 50 or 80. This will only be

        Returns
        -------
        aperture_dict : dict
        """
        # if aperture_dict is None we use the 50% encircled energy of a point spread function in each band
        if apertures is None:
            aperture_rad_dict = {}
            for band in cutout_dict['band_list']:
                if band in self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']:
                    aperture_rad_dict.update({
                        'aperture_%s' % band:
                            self.hst_encircle_apertures_wfc3_uvis2_arcsec[band]['ee%i' % default_ee_rad]})
                if band in self.hst_targets[self.target_name]['acs_wfc1_observed_bands']:
                    aperture_rad_dict.update({
                        'aperture_%s' % band:
                            self.hst_encircle_apertures_acs_wfc1_arcsec[band]['ee%i' % default_ee_rad]})
                if band in self.nircam_bands:
                    aperture_rad_dict.update({
                        'aperture_%s' % band: self.nircam_encircle_apertures_arcsec[band]['ee%i' % default_ee_rad]})
                if band in self.miri_bands:
                    aperture_rad_dict.update({
                        'aperture_%s' % band: self.miri_encircle_apertures_arcsec[band]['ee%i' % default_ee_rad]})

        # A fixed aperture for all bands
        elif isinstance(apertures, float):
            aperture_rad_dict = {}
            for band in cutout_dict['band_list']:
                aperture_rad_dict.update({'aperture_%s' % band: apertures})
        else:
            raise KeyError('The variable aperture must be either a float or a dictionary with all the aperture values')

        # compute the fluxes
        aperture_dict = {'aperture_rad_dict': aperture_rad_dict, 'init_pos': pos, 'recenter': recenter,
                         'recenter_rad': recenter_rad}

        for band in cutout_dict['band_list']:
            aperture_dict.update({
                'aperture_dict_%s' % band: self.flux_from_circ_aperture(
                    data=cutout_dict['%s_img_cutout' % band].data, data_err=cutout_dict['%s_err_cutout' % band].data,
                    wcs=cutout_dict['%s_img_cutout' % band].wcs, pos=pos,
                    aperture_rad=aperture_rad_dict['aperture_%s' % band], band=band, recenter=recenter,
                    recenter_rad=recenter_rad)})

        return aperture_dict

    def flux_from_circ_aperture(self, data, data_err, wcs, pos, aperture_rad, band, recenter=False, recenter_rad=0.2):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        data_err : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        pos : ``astropy.coordinates.SkyCoord``
        aperture_rad : float
        band : str
        recenter : bool
        recenter_rad : float

        Returns
        -------
        aperture_band_dict : dict
        """
        if recenter:
            # compute new pos
            new_pos, source_table = self.re_center_peak(data=data, wcs=wcs, pos=pos, recenter_rad=recenter_rad)
            aperture_band_dict = {'new_pos': new_pos}
            aperture_band_dict.update({'source_table': source_table})
            # extract fluxes
            flux, flux_err = self.extract_flux_from_circ_aperture(data=data, wcs=wcs, pos=new_pos,
                                                                  aperture_rad=aperture_rad, data_err=data_err)
            aperture_band_dict.update({'flux': flux})
            aperture_band_dict.update({'flux_err': flux_err})
            aperture_band_dict.update({'wave': self.get_band_wave(band)})

        else:
            aperture_band_dict = {'new_pos': None}
            aperture_band_dict.update({'source_table': None})
            flux, flux_err = self.extract_flux_from_circ_aperture(data=data, wcs=wcs, pos=pos,
                                                                  aperture_rad=aperture_rad, data_err=data_err)
            aperture_band_dict.update({'flux': flux})
            aperture_band_dict.update({'flux_err': flux_err})
            aperture_band_dict.update({'wave': self.get_band_wave(band)})

        return aperture_band_dict

    @staticmethod
    def sep_find_peak_in_rad(data, err, pixel_coordinates, pix_radius):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        err : ``numpy.ndarray``
        pixel_coordinates : tuple
        pix_radius : float

        Returns
        -------
        new_pixel_coordinates : tuple
        source_table : ``astropy.table.Table``
        """
        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        # to detect many features we increased the de-blend threshold and lower the de-blend contrast.
        # Furthermore, we allow features of the size of 2 pixels and accept sources with an S/N of 1.5
        source_table = sep.extract(data, 1.5, err=err, minarea=2, deblend_nthresh=100, deblend_cont=0.00005)

        if len(source_table) == 0:
            new_pixel_coordinates = pixel_coordinates
        else:
            x_cords_sources = source_table['x']
            y_cords_sources = source_table['y']
            source_table_in_search_radius = np.sqrt((x_cords_sources - pixel_coordinates[0]) ** 2 +
                                                    (y_cords_sources - pixel_coordinates[1]) ** 2) < pix_radius
            if np.sum(source_table_in_search_radius) == 0:
                # print('the object detected was not in the radius')
                new_pixel_coordinates = pixel_coordinates
            elif np.sum(source_table_in_search_radius) == 1:
                # print('only one object in radius')
                new_pixel_coordinates = (x_cords_sources[source_table_in_search_radius],
                                         y_cords_sources[source_table_in_search_radius])
            else:
                # print('get brightest object')
                peak = source_table['peak']
                max_peak_in_rad = np.max(peak[source_table_in_search_radius])
                # print('max_peak_in_rad ', peak == max_peak_in_rad)
                new_pixel_coordinates = (x_cords_sources[source_table_in_search_radius * (peak == max_peak_in_rad)],
                                         y_cords_sources[source_table_in_search_radius * (peak == max_peak_in_rad)])

        return new_pixel_coordinates, source_table

    def re_center_peak(self, data, wcs, pos, recenter_rad):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        pos : ``astropy.coordinates.SkyCoord``
        recenter_rad : float

        Returns
        -------
        new_pos : ``astropy.coordinates.SkyCoord``
        source_table : ``astropy.table.Table``
        """
        # get radius in pixel scale
        pix_radius = (wcs.world_to_pixel(pos)[0] -
                      wcs.world_to_pixel(SkyCoord(ra=pos.ra + recenter_rad * u.arcsec, dec=pos.dec))[0])
        # get the coordinates in pixel scale
        pixel_coordinates = wcs.world_to_pixel(pos)

        # estimate background
        bkg = sep.Background(np.array(data, dtype=float))
        # subtract background from image
        data = data - bkg.globalback
        data = np.array(data.byteswap().newbyteorder(), dtype=float)

        # calculate new pos
        new_pixel_coordinates, source_table = self.sep_find_peak_in_rad(data=data, err=bkg.globalrms,
                                                                        pixel_coordinates=pixel_coordinates,
                                                                        pix_radius=pix_radius)
        new_pos = wcs.pixel_to_world(new_pixel_coordinates[0], new_pixel_coordinates[1])
        return new_pos, source_table

    @staticmethod
    def extract_flux_from_circ_aperture(data, wcs, pos, aperture_rad, data_err=None):
        """

        Parameters
        ----------
        data : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS``
        pos : ``astropy.coordinates.SkyCoord``
        aperture_rad : float
        data_err : ``numpy.ndarray``

        Returns
        -------
        flux : float
        flux_err : float
        """
        # estimate background
        bkg = sep.Background(np.array(data, dtype=float))
        # get radius in pixel scale
        pix_radius = (wcs.world_to_pixel(pos)[0] -
                      wcs.world_to_pixel(SkyCoord(ra=pos.ra + aperture_rad * u.arcsec, dec=pos.dec))[0])
        # get the coordinates in pixel scale
        pixel_coords = wcs.world_to_pixel(pos)

        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        if data_err is None:
            bkg_rms = bkg.rms()
            data_err = np.array(bkg_rms.byteswap().newbyteorder(), dtype=float)
        else:
            data_err = np.array(data_err.byteswap().newbyteorder(), dtype=float)

        flux, flux_err, flag = sep.sum_circle(data=data - bkg.globalback, x=np.array([float(pixel_coords[0])]),
                                              y=np.array([float(pixel_coords[1])]), r=np.array([float(pix_radius)]),
                                              err=data_err)

        return float(flux), float(flux_err)

    @staticmethod
    def sep_source_detection(data_sub, rms, psf, wcs, snr=3.0):
        """

        Parameters
        ----------
        data_sub : ``numpy.ndarray``
            background subtracted image data
        rms : float or``numpy.ndarray``
            RMS of the data
        psf : ``numpy.ndarray``
        wcs : ``astropy.wcs.WCS´´
        snr : float

        Returns
        -------
        object_table: ``astropy.table.Table``
        """

        # To get a better detection of crowded sources and blended sources we significantly increase the contrast
        # We also use a minimum area of 2 because we want to be able to select point sources
        # We use the convolved mode and provide a psf in order to get a correct source size estimate.
        sep_table = sep.extract(data=data_sub, thresh=snr, err=rms, filter_kernel=psf, minarea=2, filter_type='conv',
                                deblend_cont=0.00001)

        wcs_position = wcs.pixel_to_world(sep_table['x'], sep_table['y'])
        a = (wcs.pixel_to_world(sep_table['x'], sep_table['y']).ra -
             wcs.pixel_to_world(sep_table['x'] + sep_table['a'], sep_table['y']).ra)
        b = (wcs.pixel_to_world(sep_table['x'], sep_table['y']).dec -
             wcs.pixel_to_world(sep_table['x'], sep_table['y'] + sep_table['b']).dec)

        source_table = QTable([wcs_position.ra, wcs_position.dec, a, b, sep_table['theta']],
                              names=('ra', 'dec', 'a', 'b', 'theta'))
        return source_table

    def fit_n_gaussian_to_img(self, band, img, img_err, source_table, wcs):
        """

        Parameters
        ----------
        band : str
        img : ``numpy.ndarray``
        img_err : ``numpy.ndarray``
        source_table : ``astropy.table.Table``, dict or array
        wcs : ``astropy.wcs.WCS``

        Returns
        -------
        fit_results : ``lmfit.model.ModelResult``
        """

        # create x and y data grid for modelling
        x_grid, y_grid = helper_func.create_2d_data_mesh(data=img)

        # make sure that the gaussian fitting function is an attribute otherwise load it into the attributes
        if not hasattr(self, 'gauss2d_rot_conv_%s' % band):
            self.add_gaussian_model_band_conv(band=band)

        # create the fitting model
        fmodel = helper_func.compose_n_func_model(func=getattr(self, 'gauss2d_rot_conv_%s' % band), n=len(source_table),
                                                  running_prefix='g_', independent_vars=('x', 'y'))
        # print("lmfit parameter names:", fmodel.param_names)

        # get data parameters to get the image parameters
        img_mean, img_std, img_max = np.mean(img), np.std(img), np.max(img)
        params = helper_func.set_2d_gauss_params(fmodel=fmodel, initial_params=source_table, wcs=wcs, img_mean=img_mean,
                                                 img_std=img_std, img_max=img_max, running_prefix='g_')
        fit_result = fmodel.fit(img, x=x_grid, y=y_grid, params=params, weights=1/img_err) #, method='least_squares')
        return fit_result

    def compute_de_convolved_gaussian_model(self, fit_result):
        # create x and y data grid for modelling
        x_grid, y_grid = helper_func.create_2d_data_mesh(data=fit_result.data)

        number_gaussians = int(len(fit_result.init_params) / 6)
        print('number_gaussians ', number_gaussians)
        if number_gaussians == 0:
            return None

        if 'null_func_amp' in fit_result.params:
            number_gaussians -= 1
        print('number_gaussians ', number_gaussians)
        model_data = self.gauss2d_rot(x=x_grid, y=y_grid,
                                      amp=fit_result.best_values['g_0_amp'],
                                      x0=fit_result.best_values['g_0_x0'],
                                      y0=fit_result.best_values['g_0_y0'],
                                      sig_x=fit_result.best_values['g_0_sig_x'],
                                      sig_y=fit_result.best_values['g_0_sig_y'],
                                      theta=fit_result.best_values['g_0_theta'])
        for index in range(1, number_gaussians):
            model_data += self.gauss2d_rot(x=x_grid, y=y_grid,
                                           amp=fit_result.best_values['g_%i_amp' % index],
                                           x0=fit_result.best_values['g_%i_x0' % index],
                                           y0=fit_result.best_values['g_%i_y0' % index],
                                           sig_x=fit_result.best_values['g_%i_sig_x' % index],
                                           sig_y=fit_result.best_values['g_%i_sig_y' % index],
                                           theta=fit_result.best_values['g_%i_theta' % index])

        return model_data

    def compute_flux_gaussian_component(self, fit_result):

        print(fit_result.best_values['g_0_amp'])
        print(fit_result.params['g_0_amp'])
        print(fit_result.params['g_0_amp'].value)
        print(fit_result.params['g_0_amp'].stderr)

        flux = (2 * np.pi * fit_result.params['g_0_amp'].value * fit_result.params['g_0_sig_x'].value *
                fit_result.params['g_0_sig_y'].value)
        print(flux)
        exit()
        flux_err = flux * np.sqrt(
            (fit_result.params['g_0_amp'].stderr / fit_result.params['g_0_amp'].value) ** 2 +
            (fit_result.params['g_0_sig_x'].stderr / fit_result.params['g_0_sig_x'].value) ** 2 +
            (fit_result.params['g_0_sig_y'].stderr / fit_result.params['g_0_sig_y'].value) ** 2
        )

    @staticmethod
    def update_source_table(object_table, object_table_residuals):

        mask_delete_in_object_table = np.zeros(len(object_table), dtype=bool)
        for index in range(len(object_table)):
            x_sources = object_table['ra'][index]
            y_sources = object_table['dec'][index]
            a_sources = object_table['a'][index]
            b_sources = object_table['b'][index]
            angle_sources = object_table['theta'][index]


            # problem here !!!!
            # redo for coordinates
            objects_in_ell = helper_func.check_point_inside_ellipse(x_ell=x_sources, y_ell=y_sources, a_ell=6*a_sources,
                                                                    b_ell=6*b_sources, theta_ell=angle_sources,
                                                                    x_p=object_table_residuals['ra'],
                                                                    y_p=object_table_residuals['dec'])
            # find elements where a
            if sum(objects_in_ell) > 1:
                mask_delete_in_object_table[index] = True
        print(len(object_table))
        object_table = object_table[~mask_delete_in_object_table]
        print(len(object_table))
        # object_table = np.insert(object_table, -1, object_table_residuals)
        object_table = hstack([object_table, object_table_residuals])
        print(len(object_table))
        return object_table

    def iterative_gaussian_de_blend(self, band, ra, dec, cutout_size, initial_table=None, n_iterations=2):

        cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size, band_list=[band],
                                                include_err=True)

        # load data, uncertainties and psf
        data = np.array(cutout_dict['%s_img_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        err = np.array(cutout_dict['%s_err_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        psf = np.array(self.psf_dict['native_psf_%s' % band].byteswap().newbyteorder(), dtype=float)
        # calculate the background
        bkg = sep.Background(np.array(data, dtype=float))
        # data subtracted from a global background estimation
        data_sub = data - bkg.globalback

        # create SEP object table
        if initial_table is None:
            object_table_n1 = self.sep_source_detection(data_sub=data_sub, rms=bkg.globalrms, psf=psf,
                                                        wcs=cutout_dict['%s_img_cutout' % band].wcs)
        else:
            object_table_n1 = initial_table

        # fit a gaussian for each object
        fit_result_n1 = self.fit_n_gaussian_to_img(band=band, img=data_sub, img_err=err, source_table=object_table_n1,
                                                   wcs=cutout_dict['%s_img_cutout' % band].wcs)
        print(fit_result_n1.fit_report())

        # get de convolved model
        model_data_n1 = self.compute_de_convolved_gaussian_model(fit_result=fit_result_n1)

        # get residuals
        residuals_n1 = np.array((data_sub - fit_result_n1.best_fit).byteswap().newbyteorder(), dtype=float)
        bkg_residuals_n1 = sep.Background(np.array(residuals_n1, dtype=float))
        residuals_sub_n1 = residuals_n1 - bkg_residuals_n1.globalback
        # get source detection from residuals
        object_table_residuals_n1 = self.sep_source_detection(data_sub=residuals_sub_n1, rms=bkg_residuals_n1.globalrms,
                                                              psf=psf, wcs=cutout_dict['%s_img_cutout' % band].wcs)

        # update source table with residual detection
        object_table_n2 = self.update_source_table(object_table_n1, object_table_residuals_n1)

        # refit data with new table
        fit_result_n2 = self.fit_n_gaussian_to_img(band=band, img=data_sub, img_err=err, source_table=object_table_n2,
                                                   wcs=cutout_dict['%s_img_cutout' % band].wcs)
        print(fit_result_n2.fit_report())

        model_data_n2 = self.compute_de_convolved_gaussian_model(fit_result=fit_result_n2)

        residuals_n2 = np.array((data_sub - fit_result_n2.best_fit).byteswap().newbyteorder(), dtype=float)
        bkg_residuals_n2 = sep.Background(np.array(residuals_n2, dtype=float))
        residuals_sub_n2 = residuals_n2 - bkg_residuals_n2.globalback

        # get source detection from residuals
        object_table_residuals_n2 = self.sep_source_detection(data_sub=residuals_sub_n2, rms=bkg_residuals_n2.globalrms,
                                                              psf=psf, wcs=cutout_dict['%s_img_cutout' % band].wcs)

        fit_result_dict = {
            'data_sub': data_sub,
            'object_table_n1': object_table_n1, 'fit_result_n1': fit_result_n1, 'model_data_n1': model_data_n1,
            'residuals_n1': residuals_n1, 'object_table_residuals_n1': object_table_residuals_n1,
            'object_table_n2': object_table_n2, 'fit_result_n2': fit_result_n2, 'model_data_n2': model_data_n2,
            'residuals_n2': residuals_n2, 'object_table_residuals_n2': object_table_residuals_n2}

        return fit_result_dict
