"""
Gather all photometric tools for HST and JWST photometric observations
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
import sep
import time

from photometry_tools import data_access, helper_func, basic_attributes

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.table import QTable, vstack

from astropy.stats import sigma_clipped_stats

import os

# try:
#     import zfit
# except (ImportError, ModuleNotFoundError) as err:
#     print('zfit was not imported. Deblending functions are therefore not available')


class AnalysisTools(data_access.PhotAccess):
    """
    Access class to organize data structure of HST, NIRCAM and MIRI imaging data
    """

    def __init__(self, **kwargs):
        """

        """
        super().__init__(**kwargs)

    def circular_flux_aperture_from_cutouts(self, cutout_dict, pos, apertures=None, recenter=False, recenter_rad=0.2,
                                            default_ee_rad=50, include_err=False):
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
            if include_err:
                data_err = cutout_dict['%s_err_cutout' % band].data
            else:
                data_err = None
            aperture_dict.update({
                'aperture_dict_%s' % band: self.flux_from_circ_aperture(
                    data=cutout_dict['%s_img_cutout' % band].data, data_err=data_err,
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

    @staticmethod
    def get_sep_init_src_guess(data_sub, rms, psf, detect_rad=None, snr=3.0, minarea=2, deblend_cont=0.00001):
        """
        To get a better detection of crowded sources and blended sources we significantly increase the contrast
        We also use a minimum area of 2 because we want to be able to select point sources
        We use the convolved mode and provide a psf in order to get a correct source size estimate.

        Parameters
        ----------
        data_sub : ``numpy.ndarray``
            background subtracted image data
        rms : float or``numpy.ndarray``
            RMS of the data
        psf : ``numpy.ndarray``
        detect_rad : float
            radius in which sources are considered. If None all sources are considered
        snr : float
        minarea : int
            minimum area of the sources
        deblend_cont : float
            contrast for deblending the source

        Returns
        -------
        dict of initial guesses in pixel scale
        position x & y minor and major axis a & b and rotation angle theta
        """

        sep_table = sep.extract(data=data_sub, thresh=snr, err=rms, filter_kernel=psf, minarea=minarea,
                                filter_type='conv', deblend_cont=deblend_cont)

        if detect_rad is None:
            select_mask = np.ones(len(sep_table['x']), dtype=bool)
        else:
            central_x_pos = data_sub.shape[0]/2
            central_y_pos = data_sub.shape[1]/2
            select_mask = np.sqrt((sep_table['x'] - central_x_pos) ** 2 + (sep_table['y'] - central_y_pos) ** 2) < detect_rad

        sep_source_dict = {
            'x': sep_table['x'][select_mask],
            'y': sep_table['y'][select_mask],
            'a': sep_table['a'][select_mask],
            'b': sep_table['b'][select_mask],
            'theta': sep_table['theta'][select_mask]}

        return sep_source_dict

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
        fit_result = fmodel.fit(img, x=x_grid, y=y_grid, params=params, weights=1 / img_err, method='least_squares')
        return fit_result

    def fit_composed_model2img(self, band, img, img_err, init_pos, param_lim, mask_gauss=None):
        """

        Parameters
        ----------
        band : str
        img : ``numpy.ndarray``
        img_err : ``numpy.ndarray``

        Returns
        -------
        fit_results : ``lmfit.model.ModelResult``
        """

        # create x and y data grid for modelling
        x_grid, y_grid = helper_func.create_2d_data_mesh(data=img)

        # make sure that the gaussian fitting function is an attribute otherwise load it into the attributes
        if not hasattr(self, 'gauss2d_rot_conv_%s' % band):
            self.add_gaussian_model_band_conv(band=band)

        # make sure point source model exists
        if not hasattr(self, 'point_source_conv_%s' % band):
            self.add_point_source_model_band_conv(band=band)

        initial_x, initial_y = init_pos

        print(len(initial_x))
        if mask_gauss is None:
            mask_func1 = np.ones(len(initial_x), dtype=bool)
        else:
            mask_func1 = ~mask_gauss
        mask_gauss = ~mask_func1

        fmodel = helper_func.compose_mixed_func_model(func1=getattr(self, 'point_source_conv_%s' % band),
                                                      func2=getattr(self, 'gauss2d_rot_conv_%s' % band),
                                                      mask_func1=mask_func1, independent_vars=('x', 'y'),
                                                      running_prefix='g_')

        print("lmfit parameter names:", fmodel.param_names)

        # get data parameters to get the image parameters
        img_mean, img_std, img_max = np.mean(img), np.std(img), np.max(img)
        params = helper_func.set_mixt_model_params(fmodel=fmodel, init_pos=init_pos, param_lim=param_lim,
                                                   img_mean=img_mean,
                                                   img_std=img_std, img_max=img_max, mask_gauss=mask_gauss,
                                                   running_prefix='g_')
        print(params)

        fit_result = fmodel.fit(img, x=x_grid, y=y_grid, params=params, weights=1 / img_err, method='least_squares')

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

    def compute_flux_gaussian_component(self, fit_result, n_gauss):

        flux_dict = {}
        for index in range(n_gauss):
            flux = (2 * np.pi * fit_result.params['g_%i_amp' % index].value *
                    fit_result.params['g_%i_sig_x' % index].value *
                    fit_result.params['g_%i_sig_y' % index].value)

            if fit_result.params['g_%i_amp' % index].stderr is None:
                flux_err = -999
            else:
                flux_err = flux * np.sqrt((fit_result.params['g_%i_amp' % index].stderr /
                                           fit_result.params['g_%i_amp' % index].value) ** 2 +
                                          (fit_result.params['g_%i_sig_x' % index].stderr /
                                           fit_result.params['g_%i_sig_x' % index].value) ** 2 +
                                          (fit_result.params['g_%i_sig_y' % index].stderr /
                                           fit_result.params['g_%i_sig_y' % index].value) ** 2)
            flux_dict.update({'flux_%i' % index: flux, 'flux_err_%i' % index: flux_err})
        return flux_dict

    @staticmethod
    def update_source_table(object_table, object_table_residuals, fit_results, rms):

        mask_delete_in_object_table = np.zeros(len(object_table), dtype=bool)
        for index in range(len(object_table)):
            x_sources = object_table['ra'][index]
            y_sources = object_table['dec'][index]
            a_sources = object_table['a'][index]
            b_sources = object_table['b'][index]
            angle_sources = object_table['theta'][index]

            objects_in_ell = helper_func.check_point_inside_ellipse(x_ell=x_sources, y_ell=y_sources,
                                                                    a_ell=6 * a_sources,
                                                                    b_ell=6 * b_sources, theta_ell=angle_sources,
                                                                    x_p=object_table_residuals['ra'],
                                                                    y_p=object_table_residuals['dec'])
            # find elements where a
            if sum(objects_in_ell) > 1:
                mask_delete_in_object_table[index] = True

            if fit_results.params['g_%i_amp' % index].value < 3 * rms:
                mask_delete_in_object_table[index] = True

        object_table = object_table[~mask_delete_in_object_table]
        # object_table = np.insert(object_table, -1, object_table_residuals)
        object_table = vstack([object_table, object_table_residuals])

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
        # get the WCS
        wcs = cutout_dict['%s_img_cutout' % band].wcs

        # create SEP object table
        if initial_table is None:
            object_table_n1 = self.sep_source_detection(data_sub=data_sub, rms=bkg.globalrms, psf=psf,
                                                        wcs=wcs)
        else:
            object_table_n1 = initial_table

        # fit a gaussian for each object
        fit_result_n1 = self.fit_n_gaussian_to_img(band=band, img=data_sub, img_err=err, source_table=object_table_n1,
                                                   wcs=wcs)
        print(fit_result_n1.fit_report())

        # get de convolved model
        model_data_n1 = self.compute_de_convolved_gaussian_model(fit_result=fit_result_n1)
        flux_dict_n1 = self.compute_flux_gaussian_component(fit_result=fit_result_n1, n_gauss=len(object_table_n1))

        # get residuals
        residuals_n1 = np.array((data_sub - fit_result_n1.best_fit).byteswap().newbyteorder(), dtype=float)
        bkg_residuals_n1 = sep.Background(np.array(residuals_n1, dtype=float))
        residuals_sub_n1 = residuals_n1 - bkg_residuals_n1.globalback
        # get source detection from residuals
        object_table_residuals_n1 = self.sep_source_detection(data_sub=residuals_sub_n1, rms=bkg_residuals_n1.globalrms,
                                                              psf=psf, wcs=wcs)

        fit_result_dict = {
            'data_sub': data_sub, 'wcs': wcs, 'flux_dict_n1': flux_dict_n1,
            'object_table_n1': object_table_n1, 'fit_result_n1': fit_result_n1, 'model_data_n1': model_data_n1,
            'residuals_n1': residuals_n1, 'object_table_residuals_n1': object_table_residuals_n1}

        # update source table with residual detection
        object_table_n2 = self.update_source_table(object_table_n1, object_table_residuals_n1,
                                                   fit_results=fit_result_n1, rms=bkg_residuals_n1.globalrms)

        # refit data with new table
        fit_result_n2 = self.fit_n_gaussian_to_img(band=band, img=data_sub, img_err=err, source_table=object_table_n2,
                                                   wcs=wcs)
        print(fit_result_n2.fit_report())

        model_data_n2 = self.compute_de_convolved_gaussian_model(fit_result=fit_result_n2)
        flux_dict_n2 = self.compute_flux_gaussian_component(fit_result=fit_result_n2, n_gauss=len(object_table_n2))

        residuals_n2 = np.array((data_sub - fit_result_n2.best_fit).byteswap().newbyteorder(), dtype=float)
        bkg_residuals_n2 = sep.Background(np.array(residuals_n2, dtype=float))
        residuals_sub_n2 = residuals_n2 - bkg_residuals_n2.globalback

        # get source detection from residuals
        object_table_residuals_n2 = self.sep_source_detection(data_sub=residuals_sub_n2, rms=bkg_residuals_n2.globalrms,
                                                              psf=psf, wcs=wcs)

        fit_result_dict.update({'flux_dict_n2': flux_dict_n2, 'object_table_n2': object_table_n2,
                                'fit_result_n2': fit_result_n2, 'model_data_n2': model_data_n2,
                                'residuals_n2': residuals_n2, 'object_table_residuals_n2': object_table_residuals_n2})

        return fit_result_dict

    def de_blend_mixed_model(self, band, ra, dec, cutout_size, ):

        # get cutout
        cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size, band_list=[band],
                                                include_err=True)
        # get WCS, data, uncertainties and psf
        data = np.array(cutout_dict['%s_img_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        err = np.array(cutout_dict['%s_err_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        psf = np.array(self.psf_dict['native_psf_%s' % band].byteswap().newbyteorder(), dtype=float)
        # calculate the background
        bkg = sep.Background(np.array(data, dtype=float))
        # data subtracted from a global background estimation
        data_sub = data - bkg.globalback
        # get the WCS
        wcs = cutout_dict['%s_img_cutout' % band].wcs

        # create SEP object table
        sep_source_dict = self.get_sep_init_src_guess(data_sub=data_sub, rms=bkg.globalrms, psf=psf)

        print(sep_source_dict['a'])
        print(sep_source_dict['b'])

        param_lim = (np.ones(len(sep_source_dict['a'])) * 3, np.ones(len(sep_source_dict['b'])) * 3)

        if not hasattr(self, 'point_source_conv_%s' % band):
            self.add_point_source_model_band_conv(band=band)

        # fit
        fit_results = self.fit_composed_model2img(band=band, img=data_sub, img_err=err,
                                                  init_pos=(sep_source_dict['x'], sep_source_dict['y']),
                                                  param_lim=param_lim, mask_gauss=None)

        print(fit_results.fit_report())

        new_bkg = sep.Background(np.array((data_sub - fit_results.best_fit), dtype=float))
        new_sep_source_dict = self.get_sep_init_src_guess(data_sub=data_sub - fit_results.best_fit,
                                                          rms=new_bkg.globalrms, psf=psf)

        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 8))

        ax[0].imshow(data_sub, origin='lower')
        ax[1].imshow(fit_results.best_fit, origin='lower')
        ax[2].imshow(data_sub - fit_results.best_fit, origin='lower')

        # print(sep_source_dict)

        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                        width=sep_source_dict['a'][i] * 3,
                        height=sep_source_dict['b'][i] * 3,
                        angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[0].add_artist(e)

        for i in range(len(new_sep_source_dict['x'])):
            e = Ellipse(xy=(new_sep_source_dict['x'][i], new_sep_source_dict['y'][i]),
                        width=new_sep_source_dict['a'][i] * 3,
                        height=new_sep_source_dict['b'][i] * 3,
                        angle=new_sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[2].add_artist(e)

        plt.show()

        exit()

        # fit a gaussian for each object
        fit_result_n1 = self.fit_n_gaussian_to_img(band=band, img=data_sub, img_err=err, source_table=object_table_n1,
                                                   wcs=wcs)
        print(fit_result_n1.fit_report())

        # get de convolved model
        model_data_n1 = self.compute_de_convolved_gaussian_model(fit_result=fit_result_n1)
        flux_dict_n1 = self.compute_flux_gaussian_component(fit_result=fit_result_n1, n_gauss=len(object_table_n1))

        # get residuals
        residuals_n1 = np.array((data_sub - fit_result_n1.best_fit).byteswap().newbyteorder(), dtype=float)
        bkg_residuals_n1 = sep.Background(np.array(residuals_n1, dtype=float))
        residuals_sub_n1 = residuals_n1 - bkg_residuals_n1.globalback
        # get source detection from residuals
        object_table_residuals_n1 = self.sep_source_detection(data_sub=residuals_sub_n1, rms=bkg_residuals_n1.globalrms,
                                                              psf=psf, wcs=wcs)

        fit_result_dict = {
            'data_sub': data_sub, 'wcs': wcs, 'flux_dict_n1': flux_dict_n1,
            'object_table_n1': object_table_n1, 'fit_result_n1': fit_result_n1, 'model_data_n1': model_data_n1,
            'residuals_n1': residuals_n1, 'object_table_residuals_n1': object_table_residuals_n1}

        # update source table with residual detection
        object_table_n2 = self.update_source_table(object_table_n1, object_table_residuals_n1,
                                                   fit_results=fit_result_n1, rms=bkg_residuals_n1.globalrms)

        # refit data with new table
        fit_result_n2 = self.fit_n_gaussian_to_img(band=band, img=data_sub, img_err=err, source_table=object_table_n2,
                                                   wcs=wcs)
        print(fit_result_n2.fit_report())

        model_data_n2 = self.compute_de_convolved_gaussian_model(fit_result=fit_result_n2)
        flux_dict_n2 = self.compute_flux_gaussian_component(fit_result=fit_result_n2, n_gauss=len(object_table_n2))

        residuals_n2 = np.array((data_sub - fit_result_n2.best_fit).byteswap().newbyteorder(), dtype=float)
        bkg_residuals_n2 = sep.Background(np.array(residuals_n2, dtype=float))
        residuals_sub_n2 = residuals_n2 - bkg_residuals_n2.globalback

        # get source detection from residuals
        object_table_residuals_n2 = self.sep_source_detection(data_sub=residuals_sub_n2, rms=bkg_residuals_n2.globalrms,
                                                              psf=psf, wcs=wcs)

        fit_result_dict.update({'flux_dict_n2': flux_dict_n2, 'object_table_n2': object_table_n2,
                                'fit_result_n2': fit_result_n2, 'model_data_n2': model_data_n2,
                                'residuals_n2': residuals_n2, 'object_table_residuals_n2': object_table_residuals_n2})

        return fit_result_dict

    def zfit_pnt_src_param_dict(self, amp, x0, y0, sig, fix,
                                lower_amp, upper_amp, lower_x0, upper_x0, lower_y0, upper_y0, lower_sig, upper_sig,
                                starting_index=0, src_name=None, src_blend=None):
        """

        Parameters
        ----------
        amp : float, list or array-like
        x0 : list or array-like
        y0 : list or array-like
        sig : list or array-like
        fix : bool, list or arry like
        lower_amp : float, list or array-like
        upper_amp : float, list or array-like
        lower_x0 : float, list or array-like
        upper_x0 : float, list or array-like
        lower_y0 : float, list or array-like
        upper_y0 : float, list or array-like
        lower_sig : float, list or array-like
        upper_sig : float, list or array-like
        starting_index : int

        Returns
        -------
        dict with all parameter restrictions
        """
        # specify number of sources
        param_restrict_dict_pnt = {'n_src_pnt': len(x0)}

        # transform parameters to a list if just a float or int is given
        if isinstance(amp, float) | isinstance(amp, int):
            amp = [amp] * len(x0)
        if isinstance(lower_amp, float) | isinstance(lower_amp, int):
            lower_amp = [lower_amp] * len(x0)
        if isinstance(upper_amp, float) | isinstance(upper_amp, int):
            upper_amp = [upper_amp] * len(x0)

        if isinstance(lower_x0, float) | isinstance(lower_x0, int):
            lower_x0 = [lower_x0] * len(x0)
        if isinstance(upper_x0, float) | isinstance(upper_x0, int):
            upper_x0 = [upper_x0] * len(x0)
        if isinstance(lower_y0, float) | isinstance(lower_y0, int):
            lower_y0 = [lower_y0] * len(x0)
        if isinstance(upper_y0, float) | isinstance(upper_y0, int):
            upper_y0 = [upper_y0] * len(x0)

        if isinstance(sig, float) | isinstance(sig, int):
            sig = [sig] * len(x0)
        if isinstance(lower_sig, float) | isinstance(lower_sig, int):
            lower_sig = [lower_sig] * len(x0)
        if isinstance(upper_sig, float) | isinstance(upper_sig, int):
            upper_sig = [upper_sig] * len(x0)

        if isinstance(fix, bool) | isinstance(fix, int):
            fix = [fix] * len(x0)

        if src_name is None:
            src_name = []
            src_blend = []
            for index in range(len(x0)):
                src_name.append(int(index + self.n_total_sources))
                src_blend.append(None)

        # create parameter restriction dict
        for index in range(len(x0)):
            param_restrict_dict_pnt.update({'pnt_%i' % (index + starting_index): {
                'fix': fix[index],
                'init_amp': amp[index], 'lower_amp': lower_amp[index], 'upper_amp': upper_amp[index],
                'init_x0': x0[index], 'lower_x0': lower_x0[index], 'upper_x0': upper_x0[index],
                'init_y0': y0[index], 'lower_y0': lower_y0[index], 'upper_y0': upper_y0[index],
                'init_sig': sig[index], 'lower_sig': lower_sig[index], 'upper_sig': upper_sig[index],
                'src_name': src_name[index], 'src_blend': src_blend[index]
            }})
        self.n_total_sources += len(x0)

        return param_restrict_dict_pnt

    def zfit_pnt_src_param_dict_from_fit(self, fit_result_dict, mask, fix_mask, starting_index=0):

        # specify number of sources
        param_restrict_dict_pnt = {
            'n_src_pnt': sum(mask)
        }
        for index in range(sum(mask)):
            param_restrict_dict_pnt.update({'pnt_%i' % (index + starting_index): {
                'fix': fix_mask[mask][index]}})
            param_restrict_dict_pnt['pnt_%i' % (index + starting_index)].update({
                'src_name': fit_result_dict['param_dict_pnt_pix_scale']['src_name_pnt'][mask][index]})
            param_restrict_dict_pnt['pnt_%i' % (index + starting_index)].update({
                'src_blend': fit_result_dict['param_dict_pnt_pix_scale']['src_blend_pnt'][mask][index]})

            for param_name in ['amp', 'x0', 'y0', 'sig']:
                param_restrict_dict_pnt['pnt_%i' % (index + starting_index)].update({
                    '%s' % param_name: fit_result_dict['param_dict_pnt_pix_scale']['%s_pnt' % param_name][mask][index],
                    '%s_err' % param_name: fit_result_dict['param_dict_pnt_pix_scale']['%s_pnt_err' % param_name][mask][index],
                    'init_%s' % param_name: fit_result_dict['param_dict_pnt_pix_scale']['init_%s_pnt' % param_name][mask][index],
                    'lower_%s' % param_name: fit_result_dict['param_dict_pnt_pix_scale']['lower_%s_pnt' % param_name][mask][index],
                    'upper_%s' % param_name: fit_result_dict['param_dict_pnt_pix_scale']['upper_%s_pnt' % param_name][mask][index]})

        return param_restrict_dict_pnt

    def zfit_gauss_src_param_dict_from_fit(self, fit_result_dict, fix_mask, starting_index=0):

        # specify number of sources
        param_restrict_dict_gauss = {
            'n_src_gauss': len(fit_result_dict['param_dict_gauss_pix_scale']['fix_gauss'])
        }
        for index in range(len(fit_result_dict['param_dict_gauss_pix_scale']['fix_gauss'])):
            param_restrict_dict_gauss.update({'gauss_%i' % (index + starting_index): {
                'fix': fix_mask[index]}})
            param_restrict_dict_gauss['gauss_%i' % (index + starting_index)].update({
                'src_name': fit_result_dict['param_dict_gauss_pix_scale']['src_name_gauss'][index]})
            param_restrict_dict_gauss['gauss_%i' % (index + starting_index)].update({
                'src_blend': fit_result_dict['param_dict_gauss_pix_scale']['src_blend_gauss'][index]})

            for param_name in ['amp', 'x0', 'y0', 'sig_x', 'sig_y', 'theta']:
                param_restrict_dict_gauss['gauss_%i' % (index + starting_index)].update({
                    '%s' % param_name: fit_result_dict['param_dict_gauss_pix_scale']['%s_gauss' % param_name][index],
                    '%s_err' % param_name: fit_result_dict['param_dict_gauss_pix_scale']['%s_gauss_err' % param_name][index],
                    'init_%s' % param_name: fit_result_dict['param_dict_gauss_pix_scale']['init_%s_gauss' % param_name][index],
                    'lower_%s' % param_name: fit_result_dict['param_dict_gauss_pix_scale']['lower_%s_gauss' % param_name][index],
                    'upper_%s' % param_name: fit_result_dict['param_dict_gauss_pix_scale']['upper_%s_gauss' % param_name][index]})

        return param_restrict_dict_gauss

    def zfit_gauss_src_param_dict(self, amp, x0, y0, sig_x, sig_y, theta, fix,
                                  lower_amp, upper_amp, lower_x0, upper_x0, lower_y0, upper_y0,
                                  lower_sig_x, upper_sig_x, lower_sig_y, upper_sig_y, lower_theta, upper_theta,
                                  starting_index=0, src_name=None, src_blend=None):

        param_restrict_dict_gauss = {'n_src_gauss': len(x0)}

        # transform parameters to a list if just a float or int is given
        if isinstance(amp, float) | isinstance(amp, int):
            amp = [amp] * len(x0)
        if isinstance(lower_amp, float) | isinstance(lower_amp, int):
            lower_amp = [lower_amp] * len(x0)
        if isinstance(upper_amp, float) | isinstance(upper_amp, int):
            upper_amp = [upper_amp] * len(x0)

        if isinstance(lower_x0, float) | isinstance(lower_x0, int):
            lower_x0 = [lower_x0] * len(x0)
        if isinstance(upper_x0, float) | isinstance(upper_x0, int):
            upper_x0 = [upper_x0] * len(x0)
        if isinstance(lower_y0, float) | isinstance(lower_y0, int):
            lower_y0 = [lower_y0] * len(x0)
        if isinstance(upper_y0, float) | isinstance(upper_y0, int):
            upper_y0 = [upper_y0] * len(x0)

        if isinstance(sig_x, float) | isinstance(sig_x, int):
            sig_x = [sig_x] * len(x0)
        if isinstance(sig_y, float) | isinstance(sig_y, int):
            sig_y = [sig_y] * len(x0)

        if isinstance(lower_sig_x, float) | isinstance(lower_sig_x, int):
            lower_sig_x = [lower_sig_x] * len(x0)
        if isinstance(upper_sig_x, float) | isinstance(upper_sig_x, int):
            upper_sig_x = [upper_sig_x] * len(x0)
        if isinstance(lower_sig_y, float) | isinstance(lower_sig_y, int):
            lower_sig_y = [lower_sig_y] * len(x0)
        if isinstance(upper_sig_y, float) | isinstance(upper_sig_y, int):
            upper_sig_y = [upper_sig_y] * len(x0)

        if isinstance(theta, float) | isinstance(theta, int):
            theta = [theta] * len(x0)
        if isinstance(lower_theta, float) | isinstance(lower_theta, int):
            lower_theta = [lower_theta] * len(x0)
        if isinstance(upper_theta, float) | isinstance(upper_theta, int):
            upper_theta = [upper_theta] * len(x0)

        if isinstance(fix, bool) | isinstance(fix, int):
            fix = [fix] * len(x0)

        if src_name is None:
            src_name = []
            src_blend = []
            for index in range(len(x0)):
                src_name.append(int(index + self.n_total_sources))
                src_blend.append(None)

        for index in range(len(x0)):
            param_restrict_dict_gauss.update({'gauss_%i' % (index + starting_index): {
                'fix': fix[index],
                'init_amp': amp[index], 'lower_amp': lower_amp[index], 'upper_amp': upper_amp[index],
                'init_x0': x0[index], 'lower_x0': lower_x0[index], 'upper_x0': upper_x0[index],
                'init_y0': y0[index], 'lower_y0': lower_y0[index], 'upper_y0': upper_y0[index],
                'init_sig_x': sig_x[index], 'lower_sig_x': lower_sig_x[index], 'upper_sig_x': upper_sig_x[index],
                'init_sig_y': sig_y[index], 'lower_sig_y': lower_sig_y[index], 'upper_sig_y': upper_sig_y[index],
                'init_theta': theta[index], 'lower_theta': lower_theta[index], 'upper_theta': upper_theta[index],
                'src_name': src_name[index], 'src_blend': src_blend[index]
            }})
        return param_restrict_dict_gauss

    def zfit_pnt_src_param_dict_from_fit_old(self, amp, x0, y0, amp_err, x0_err, y0_err, sig, sig_err, src_name, src_blend,
                                         starting_index=0):

        param_restrict_dict_pnt = {'n_src_pnt': len(x0)}

        for index in range(len(x0)):
            param_restrict_dict_pnt.update({'pnt_%i' % (index + starting_index): {
                'fix': True,
                'init_amp': amp[index],
                'amp_err': amp_err[index],
                'init_x0': x0[index],
                'x0_err': x0_err[index],
                'init_y0': y0[index],
                'y0_err': y0_err[index],
                'init_sig': sig[index],
                'sig_err': sig_err[index],
                'src_name': src_name[index],
                'src_blend': src_blend[index]
            }})

        return param_restrict_dict_pnt

    def zfit_gauss_src_param_dict_from_fit_old(self, amp, x0, y0, sig_x, sig_y, theta,
                                           amp_err, x0_err, y0_err, sig_x_err, sig_y_err, theta_err,
                                           src_name, src_blend,
                                           starting_index=0):
        param_restrict_dict_gauss = {'n_src_gauss': len(x0)}
        for index in range(len(x0)):
            param_restrict_dict_gauss.update({'gauss_%i' % (index + starting_index): {
                'fix': True,
                'init_amp': amp[index], 'amp_err': amp_err[index],
                'init_x0': x0[index], 'x0_err': x0_err[index],
                'init_y0': y0[index], 'y0_err': y0_err[index],
                'init_sig_x': sig_x[index], 'sig_x_err': sig_x_err[index],
                'init_sig_y': sig_y[index], 'sig_y_err': sig_y_err[index],
                'init_theta': theta[index], 'theta_err': theta_err[index],
                'src_name': src_name[index], 'src_blend': src_blend[index]
            }})
        return param_restrict_dict_gauss

    @staticmethod
    def get_zfit_params(minimization_result, hesse_result, zfit_param_restrict_dict_pnt,
                        zfit_param_restrict_dict_gauss):
        n_src_pnt = zfit_param_restrict_dict_pnt['n_src_pnt']
        n_src_gauss = zfit_param_restrict_dict_gauss['n_src_gauss']

        param_value_array = np.array(minimization_result.values)
        error_values = list(hesse_result.values())
        if len(error_values) == 0:
            err_estimate_flag = False
        else:
            err_estimate_flag = True
        # print('error_values ', error_values)

        amp_pnt = np.zeros(n_src_pnt)
        amp_pnt_err = np.zeros(n_src_pnt)
        init_amp_pnt = np.zeros(n_src_pnt)
        lower_amp_pnt = np.zeros(n_src_pnt)
        upper_amp_pnt = np.zeros(n_src_pnt)
        x0_pnt = np.zeros(n_src_pnt)
        x0_pnt_err = np.zeros(n_src_pnt)
        init_x0_pnt = np.zeros(n_src_pnt)
        lower_x0_pnt = np.zeros(n_src_pnt)
        upper_x0_pnt = np.zeros(n_src_pnt)
        y0_pnt = np.zeros(n_src_pnt)
        y0_pnt_err = np.zeros(n_src_pnt)
        init_y0_pnt = np.zeros(n_src_pnt)
        lower_y0_pnt = np.zeros(n_src_pnt)
        upper_y0_pnt = np.zeros(n_src_pnt)
        sig_pnt = np.zeros(n_src_pnt)
        sig_pnt_err = np.zeros(n_src_pnt)
        init_sig_pnt = np.zeros(n_src_pnt)
        lower_sig_pnt = np.zeros(n_src_pnt)
        upper_sig_pnt = np.zeros(n_src_pnt)
        src_name_pnt = np.zeros(n_src_pnt, dtype=int)
        src_blend_pnt = np.zeros(n_src_pnt, dtype=object)
        fix_pnt = np.zeros(n_src_pnt, dtype=bool)

        amp_gauss = np.zeros(n_src_gauss)
        amp_gauss_err = np.zeros(n_src_gauss)
        init_amp_gauss = np.zeros(n_src_gauss)
        lower_amp_gauss = np.zeros(n_src_gauss)
        upper_amp_gauss = np.zeros(n_src_gauss)
        x0_gauss = np.zeros(n_src_gauss)
        x0_gauss_err = np.zeros(n_src_gauss)
        init_x0_gauss = np.zeros(n_src_gauss)
        lower_x0_gauss = np.zeros(n_src_gauss)
        upper_x0_gauss = np.zeros(n_src_gauss)
        y0_gauss = np.zeros(n_src_gauss)
        y0_gauss_err = np.zeros(n_src_gauss)
        init_y0_gauss = np.zeros(n_src_gauss)
        lower_y0_gauss = np.zeros(n_src_gauss)
        upper_y0_gauss = np.zeros(n_src_gauss)
        sig_x_gauss = np.zeros(n_src_gauss)
        sig_x_gauss_err = np.zeros(n_src_gauss)
        init_sig_x_gauss = np.zeros(n_src_gauss)
        lower_sig_x_gauss = np.zeros(n_src_gauss)
        upper_sig_x_gauss = np.zeros(n_src_gauss)
        sig_y_gauss = np.zeros(n_src_gauss)
        sig_y_gauss_err = np.zeros(n_src_gauss)
        init_sig_y_gauss = np.zeros(n_src_gauss)
        lower_sig_y_gauss = np.zeros(n_src_gauss)
        upper_sig_y_gauss = np.zeros(n_src_gauss)
        theta_gauss = np.zeros(n_src_gauss)
        theta_gauss_err = np.zeros(n_src_gauss)
        init_theta_gauss = np.zeros(n_src_gauss)
        lower_theta_gauss = np.zeros(n_src_gauss)
        upper_theta_gauss = np.zeros(n_src_gauss)
        src_name_gauss = np.zeros(n_src_gauss, dtype=int)
        src_blend_gauss = np.zeros(n_src_gauss, dtype=object)
        fix_gauss = np.zeros(n_src_gauss, dtype=bool)

        n_free_param_gauss = 0
        # get gauss source parameters
        for index in range(n_src_gauss):
            if zfit_param_restrict_dict_gauss['gauss_%i' % index]['fix']:
                fix_gauss[index] = True
                amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['amp']
                amp_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['amp_err']
                init_amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_amp']
                lower_amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_amp']
                upper_amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_amp']

                x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['x0']
                x0_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['x0_err']
                init_x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_x0']
                lower_x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_x0']
                upper_x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_x0']

                y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['y0']
                y0_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['y0_err']
                init_y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_y0']
                lower_y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_y0']
                upper_y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_y0']

                sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_x']
                sig_x_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_x_err']
                init_sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_sig_x']
                lower_sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_sig_x']
                upper_sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_sig_x']

                sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_y']
                sig_y_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_y_err']
                init_sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_sig_y']
                lower_sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_sig_y']
                upper_sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_sig_y']

                theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['theta']
                theta_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['theta_err']
                init_theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_theta']
                lower_theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_theta']
                upper_theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_theta']

                src_name_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['src_name']
                src_blend_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['src_blend']
            else:
                fix_gauss[index] = False
                amp_gauss[index] = param_value_array[0 + n_free_param_gauss * 6]
                init_amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_amp']
                lower_amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_amp']
                upper_amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_amp']
                x0_gauss[index] = param_value_array[1 + n_free_param_gauss * 6]
                init_x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_x0']
                lower_x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_x0']
                upper_x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_x0']
                y0_gauss[index] = param_value_array[2 + n_free_param_gauss * 6]
                init_y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_y0']
                lower_y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_y0']
                upper_y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_y0']
                sig_x_gauss[index] = param_value_array[3 + n_free_param_gauss * 6]
                init_sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_sig_x']
                lower_sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_sig_x']
                upper_sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_sig_x']
                sig_y_gauss[index] = param_value_array[4 + n_free_param_gauss * 6]
                init_sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_sig_y']
                lower_sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_sig_y']
                upper_sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_sig_y']
                theta_gauss[index] = param_value_array[5 + n_free_param_gauss * 6]
                init_theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['init_theta']
                lower_theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['lower_theta']
                upper_theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['upper_theta']
                src_name_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['src_name']
                src_blend_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['src_blend']

                if err_estimate_flag:
                    amp_gauss_err[index] = error_values[0 + n_free_param_gauss * 6]['error']
                    x0_gauss_err[index] = error_values[1 + n_free_param_gauss * 6]['error']
                    y0_gauss_err[index] = error_values[2 + n_free_param_gauss * 6]['error']
                    sig_x_gauss_err[index] = error_values[3 + n_free_param_gauss * 6]['error']
                    sig_y_gauss_err[index] = error_values[4 + n_free_param_gauss * 6]['error']
                    theta_gauss_err[index] = error_values[5 + n_free_param_gauss * 6]['error']
                else:
                    amp_gauss_err[index] = np.nan
                    x0_gauss_err[index] = np.nan
                    y0_gauss_err[index] = np.nan
                    sig_x_gauss_err[index] = np.nan
                    sig_y_gauss_err[index] = np.nan
                    theta_gauss_err[index] = np.nan

                n_free_param_gauss += 1


        # get point source parameters
        n_free_param_pnt = 0
        for index in range(n_src_pnt):
            if zfit_param_restrict_dict_pnt['pnt_%i' % index]['fix']:
                fix_pnt[index] = True
                amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['amp']
                amp_pnt_err[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['amp_err']
                init_amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_amp']
                lower_amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_amp']
                upper_amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_amp']

                x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['x0']
                x0_pnt_err[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['x0_err']
                init_x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_x0']
                lower_x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_x0']
                upper_x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_x0']

                y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['y0']
                y0_pnt_err[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['y0_err']
                init_y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_y0']
                lower_y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_y0']
                upper_y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_y0']

                sig_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['sig']
                sig_pnt_err[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['sig_err']
                init_sig_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_sig']
                lower_sig_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_sig']
                upper_sig_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_sig']

                src_name_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['src_name']
                src_blend_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['src_blend']
            else:
                fix_pnt[index] = False
                amp_pnt[index] = param_value_array[0 + n_free_param_pnt*4 + n_free_param_gauss*6]
                init_amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_amp']
                lower_amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_amp']
                upper_amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_amp']

                x0_pnt[index] = param_value_array[1 + n_free_param_pnt*4 + n_free_param_gauss*6]
                init_x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_x0']
                lower_x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_x0']
                upper_x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_x0']

                y0_pnt[index] = param_value_array[2 + n_free_param_pnt*4 + n_free_param_gauss*6]
                init_y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_y0']
                lower_y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_y0']
                upper_y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_y0']

                sig_pnt[index] = param_value_array[3 + n_free_param_pnt*4 + n_free_param_gauss*6]
                init_sig_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['init_sig']
                lower_sig_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['lower_sig']
                upper_sig_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['upper_sig']

                src_name_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['src_name']
                src_blend_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['src_blend']

                if err_estimate_flag:
                    amp_pnt_err[index] = error_values[0 + n_free_param_pnt*4 + n_free_param_gauss*6]['error']
                    x0_pnt_err[index] = error_values[1 + n_free_param_pnt*4 + n_free_param_gauss*6]['error']
                    y0_pnt_err[index] = error_values[2 + n_free_param_pnt*4 + n_free_param_gauss*6]['error']
                    sig_pnt_err[index] = error_values[3 + n_free_param_pnt*4 + n_free_param_gauss*6]['error']
                else:
                    amp_pnt_err[index] = np.nan
                    x0_pnt_err[index] = np.nan
                    y0_pnt_err[index] = np.nan
                    sig_pnt_err[index] = np.nan

                n_free_param_pnt += 1

        # point source parameters
        param_dict_pnt = {
            'amp_pnt': amp_pnt,
            'amp_pnt_err': amp_pnt_err,
            'init_amp_pnt': init_amp_pnt,
            'lower_amp_pnt': lower_amp_pnt,
            'upper_amp_pnt': upper_amp_pnt,
            'x0_pnt': x0_pnt,
            'x0_pnt_err': x0_pnt_err,
            'init_x0_pnt': init_x0_pnt,
            'lower_x0_pnt': lower_x0_pnt,
            'upper_x0_pnt': upper_x0_pnt,
            'y0_pnt': y0_pnt,
            'y0_pnt_err': y0_pnt_err,
            'init_y0_pnt': init_y0_pnt,
            'lower_y0_pnt': lower_y0_pnt,
            'upper_y0_pnt': upper_y0_pnt,
            'sig_pnt': sig_pnt,
            'sig_pnt_err': sig_pnt_err,
            'init_sig_pnt': init_sig_pnt,
            'lower_sig_pnt': lower_sig_pnt,
            'upper_sig_pnt': upper_sig_pnt,
            'src_name_pnt': src_name_pnt,
            'src_blend_pnt': src_blend_pnt,
            'fix_pnt': fix_pnt
        }

        # gaussian parameters
        param_dict_gauss = {
            'amp_gauss': amp_gauss,
            'amp_gauss_err': amp_gauss_err,
            'init_amp_gauss': init_amp_gauss,
            'lower_amp_gauss': lower_amp_gauss,
            'upper_amp_gauss': upper_amp_gauss,
            'x0_gauss': x0_gauss,
            'x0_gauss_err': x0_gauss_err,
            'init_x0_gauss': init_x0_gauss,
            'lower_x0_gauss': lower_x0_gauss,
            'upper_x0_gauss': upper_x0_gauss,
            'y0_gauss': y0_gauss,
            'y0_gauss_err': y0_gauss_err,
            'init_y0_gauss': init_y0_gauss,
            'lower_y0_gauss': lower_y0_gauss,
            'upper_y0_gauss': upper_y0_gauss,
            'sig_x_gauss': sig_x_gauss,
            'sig_x_gauss_err': sig_x_gauss_err,
            'init_sig_x_gauss': init_sig_x_gauss,
            'lower_sig_x_gauss': lower_sig_x_gauss,
            'upper_sig_x_gauss': upper_sig_x_gauss,
            'sig_y_gauss': sig_y_gauss,
            'sig_y_gauss_err': sig_y_gauss_err,
            'init_sig_y_gauss': init_sig_y_gauss,
            'lower_sig_y_gauss': lower_sig_y_gauss,
            'upper_sig_y_gauss': upper_sig_y_gauss,
            'theta_gauss': theta_gauss,
            'theta_gauss_err': theta_gauss_err,
            'init_theta_gauss': init_theta_gauss,
            'lower_theta_gauss': lower_theta_gauss,
            'upper_theta_gauss': upper_theta_gauss,
            'src_name_gauss': src_name_gauss,
            'src_blend_gauss': src_blend_gauss,
            'fix_gauss': fix_gauss
        }

        return param_dict_pnt, param_dict_gauss

    def assemble_zfit_model(self, param_dict_pnt_pix_scale, param_dict_gauss_pix_scale, band):

        best_fit = np.zeros(self.current_img_data.shape)
        best_fit_model = np.zeros(self.current_img_data.shape)

        for index in range(0, len(param_dict_pnt_pix_scale['amp_pnt'])):
            best_fit += getattr(self, 'pnt_src_sig_conv_%s' % band)(x=self.current_img_x_grid, y=self.current_img_y_grid,
                                                                amp=param_dict_pnt_pix_scale['amp_pnt'][index],
                                                                x0=param_dict_pnt_pix_scale['x0_pnt'][index],
                                                                y0=param_dict_pnt_pix_scale['y0_pnt'][index],
                                                                sig=param_dict_pnt_pix_scale['sig_pnt'][index])
            best_fit_model += self.pnt_src(x=self.current_img_x_grid, y=self.current_img_y_grid,
                                           amp=param_dict_pnt_pix_scale['amp_pnt'][index],
                                           x0=param_dict_pnt_pix_scale['x0_pnt'][index],
                                           y0=param_dict_pnt_pix_scale['y0_pnt'][index],
                                           sig=param_dict_pnt_pix_scale['sig_pnt'][index])

        for index in range(0, len(param_dict_gauss_pix_scale['amp_gauss'])):
            best_fit += getattr(self, 'gauss2d_rot_conv_%s' % band)(x=self.current_img_x_grid,
                                                                    y=self.current_img_y_grid,
                                                                    amp=param_dict_gauss_pix_scale['amp_gauss'][index],
                                                                    x0=param_dict_gauss_pix_scale['x0_gauss'][index],
                                                                    y0=param_dict_gauss_pix_scale['y0_gauss'][index],
                                                                    sig_x=param_dict_gauss_pix_scale['sig_x_gauss'][
                                                                        index],
                                                                    sig_y=param_dict_gauss_pix_scale['sig_y_gauss'][
                                                                        index],
                                                                    theta=param_dict_gauss_pix_scale['theta_gauss'][
                                                                        index])
            best_fit_model += self.gauss2d_rot(x=self.current_img_x_grid, y=self.current_img_y_grid,
                                               amp=param_dict_gauss_pix_scale['amp_gauss'][index],
                                               x0=param_dict_gauss_pix_scale['x0_gauss'][index],
                                               y0=param_dict_gauss_pix_scale['y0_gauss'][index],
                                               sig_x=param_dict_gauss_pix_scale['sig_x_gauss'][index],
                                               sig_y=param_dict_gauss_pix_scale['sig_y_gauss'][index],
                                               theta=param_dict_gauss_pix_scale['theta_gauss'][index])

        return best_fit, best_fit_model

    def compute_zfit_pnt_src_flux(self, param_dict_pnt_pix_scale, band):

        flux = np.zeros(len(param_dict_pnt_pix_scale['amp_pnt']))
        flux_err = np.zeros(len(param_dict_pnt_pix_scale['amp_pnt']))

        for index in range(0, len(param_dict_pnt_pix_scale['amp_pnt'])):
            flux[index] = np.sum(
                getattr(self, 'pnt_src_sig_conv_%s' % band)(x=self.current_img_x_grid, y=self.current_img_y_grid,
                                                           amp=param_dict_pnt_pix_scale['amp_pnt'][index],
                                                           x0=param_dict_pnt_pix_scale['x0_pnt'][index],
                                                           y0=param_dict_pnt_pix_scale['y0_pnt'][index],
                                                           sig=param_dict_pnt_pix_scale['sig_pnt'][index]))

            flux_err[index] = flux[index] / param_dict_pnt_pix_scale['amp_pnt'][index] * \
                              param_dict_pnt_pix_scale['amp_pnt_err'][index]

        return flux, flux_err

    def compute_zfit_gauss_src_flux(self, param_dict_gauss_pix_scale, band):

        flux = np.zeros(len(param_dict_gauss_pix_scale['amp_gauss']))
        flux_err = np.zeros(len(param_dict_gauss_pix_scale['amp_gauss']))

        for index in range(0, len(param_dict_gauss_pix_scale['amp_gauss'])):
            flux[index] = np.sum(getattr(self, 'gauss2d_rot_conv_%s' % band)(x=self.current_img_x_grid,
                                                                             y=self.current_img_y_grid,
                                                                             amp=
                                                                             param_dict_gauss_pix_scale['amp_gauss'][
                                                                                 index],
                                                                             x0=param_dict_gauss_pix_scale['x0_gauss'][
                                                                                 index],
                                                                             y0=param_dict_gauss_pix_scale['y0_gauss'][
                                                                                 index],
                                                                             sig_x=
                                                                             param_dict_gauss_pix_scale['sig_x_gauss'][
                                                                                 index],
                                                                             sig_y=
                                                                             param_dict_gauss_pix_scale['sig_y_gauss'][
                                                                                 index],
                                                                             theta=
                                                                             param_dict_gauss_pix_scale['theta_gauss'][
                                                                                 index]))
            analytical_flux = (param_dict_gauss_pix_scale['amp_gauss'][index] *
                               param_dict_gauss_pix_scale['sig_x_gauss'][index] *
                               param_dict_gauss_pix_scale['sig_y_gauss'][index] *
                               2 * np.pi)
            analytical_flux_err = (analytical_flux *
                                   np.sqrt((param_dict_gauss_pix_scale['amp_gauss_err'][index] /
                                            param_dict_gauss_pix_scale['amp_gauss'][index]) ** 2 +
                                           (param_dict_gauss_pix_scale['sig_x_gauss_err'][index] /
                                            param_dict_gauss_pix_scale['sig_x_gauss'][index]) ** 2 +
                                           (param_dict_gauss_pix_scale['sig_y_gauss_err'][index] /
                                            param_dict_gauss_pix_scale['sig_y_gauss'][index]) ** 2))
            flux_err[index] = flux[index] / analytical_flux * analytical_flux_err

        return flux, flux_err

    @staticmethod
    def eval_zfit_residuals(pnt_src_amp, pnt_src_amp_err, pnt_src_x, pnt_src_y, gauss_src_x, gauss_src_y,
                            new_source_table, dist_to_ellipse, gauss_detect_rad, influenz_rad):

        # mask_delete_in_object_table = np.zeros(len(object_table), dtype=bool)
        gauss_cand = np.zeros(len(pnt_src_x), dtype=bool)
        pos_x0_of_gauss_cand = np.zeros(len(pnt_src_x), dtype=float)
        pos_y0_of_gauss_cand = np.zeros(len(pnt_src_x), dtype=float)
        dist_to_gauss_cand = np.zeros(len(pnt_src_x), dtype=float)
        new_pnt_src_dict = {'x': [], 'y': [], 'a': [], 'b': []}
        derop_pnt_src = pnt_src_amp / pnt_src_amp_err < 1.0
        pnt_src_with_neighbour = np.zeros(len(pnt_src_x), dtype=bool)
        gauss_src_with_neighbour = np.zeros(len(gauss_src_x), dtype=bool)

        for index in range(len(pnt_src_x)):
            pnt_dist_to_all_candidates = np.sqrt((pnt_src_x[index] - new_source_table['x']) ** 2 +
                                                 (pnt_src_y[index] - new_source_table['y']) ** 2)
            if sum((pnt_dist_to_all_candidates > 0) & (pnt_dist_to_all_candidates < influenz_rad)):
                pnt_src_with_neighbour[index] = True

        for index in range(len(gauss_src_x)):
            gauss_dist_to_all_candidates = np.sqrt((gauss_src_x[index] - new_source_table['x']) ** 2 +
                                                   (gauss_src_y[index] - new_source_table['y']) ** 2)
            if sum((gauss_dist_to_all_candidates > 0) & (gauss_dist_to_all_candidates < influenz_rad)):
                gauss_src_with_neighbour[index] = True

        for index in range(len(new_source_table['x'])):
            x_sources = new_source_table['x'][index]
            y_sources = new_source_table['y'][index]
            a_sources = new_source_table['a'][index]
            b_sources = new_source_table['b'][index]
            angle_sources = new_source_table['theta'][index]

            objects_in_ell = helper_func.check_point_inside_ellipse(x_ell=x_sources, y_ell=y_sources,
                                                                    a_ell=dist_to_ellipse * a_sources + gauss_detect_rad,
                                                                    b_ell=dist_to_ellipse * b_sources + gauss_detect_rad,
                                                                    theta_ell=angle_sources,
                                                                    x_p=pnt_src_x, y_p=pnt_src_y)

            # find elements where just one source matches with a close neighbour
            if sum(objects_in_ell) == 1:
                gauss_cand[objects_in_ell] = True
                # get a guess how extended the source is
                print('objects_in_ell ', objects_in_ell)
                print('pnt_src_x ', pnt_src_x)
                print('x_sources ', x_sources)
                dist_to_gauss_cand[objects_in_ell] = np.sqrt((pnt_src_x[objects_in_ell] - x_sources)**2 +
                                                             (pnt_src_y[objects_in_ell] - y_sources)**2)

                pos_x0_of_gauss_cand[objects_in_ell] = np.mean([*pnt_src_x[objects_in_ell], x_sources])
                pos_y0_of_gauss_cand[objects_in_ell] = np.mean([*pnt_src_y[objects_in_ell], y_sources])
            elif sum(objects_in_ell) > 1:
                # multiple old point sources are matching with a new source.
                # we will only take the closest!
                # index of the closest point
                dist = np.sqrt((pnt_src_x - x_sources) ** 2 + (pnt_src_y - y_sources) ** 2)
                index_closest = np.where(dist == np.min(dist))

                gauss_cand[index_closest] = True
                dist_to_gauss_cand[index_closest] = np.sqrt((pnt_src_x[index_closest] - x_sources)**2 +
                                                            (pnt_src_y[index_closest] - y_sources)**2)
                pos_x0_of_gauss_cand[index_closest] = np.mean([pnt_src_x[index_closest], x_sources])
                pos_y0_of_gauss_cand[index_closest] = np.mean([pnt_src_y[index_closest], y_sources])

            else:
                new_pnt_src_dict['x'].append(x_sources)
                new_pnt_src_dict['y'].append(y_sources)
                new_pnt_src_dict['a'].append(a_sources)
                new_pnt_src_dict['b'].append(b_sources)

        return_dict = {
            'gauss_cand': gauss_cand,
            'pos_x0_of_gauss_cand': pos_x0_of_gauss_cand,
            'pos_y0_of_gauss_cand': pos_y0_of_gauss_cand,
            'dist_to_gauss_cand': dist_to_gauss_cand,
            'new_pnt_src_dict': new_pnt_src_dict,
            'derop_pnt_src': derop_pnt_src,
            'pnt_src_with_neighbour': pnt_src_with_neighbour,
            'gauss_src_with_neighbour': gauss_src_with_neighbour
        }

        return return_dict

    def generate_new_init_zfit_model(self, data_sub, psf, fit_result_dict, band, wcs, fit_rad, sig_ref, min_sig_fact,
                                     max_sig_fact, max_amp_fact, fit_iter_index, pos_var, gauss_detect_rad, init_snr=5):
        # if additional iterations will take place:
        # find sources in residuals
        new_bkg = sep.Background(np.array((data_sub - fit_result_dict['best_fit']), dtype=float))
        fit_rad_pix = helper_func.transform_world2pix_scale(length_in_arcsec=fit_rad, wcs=wcs, dim=0)
        new_sep_src_dict = self.get_sep_init_src_guess(data_sub=data_sub - fit_result_dict['best_fit'],
                                                       rms=new_bkg.globalrms, psf=psf, detect_rad=fit_rad_pix,
                                                       snr=init_snr)

        # when no source was detected just decrease the S/N and the minarea
        loop_snr_list = np.linspace(init_snr, init_snr/10, 10)
        if len(new_sep_src_dict['x']) == 0:
            for snr in loop_snr_list:
                print('snr ', snr)
                new_sep_src_dict = self.get_sep_init_src_guess(data_sub=data_sub - fit_result_dict['best_fit'],
                                                               rms=new_bkg.globalrms, psf=psf, detect_rad=fit_rad_pix,
                                                               snr=snr)
                if len(new_sep_src_dict['x']) != 0:
                    break
        if len(new_sep_src_dict['x']) == 0:
            for snr in loop_snr_list:
                new_sep_src_dict = self.get_sep_init_src_guess(data_sub=data_sub - fit_result_dict['best_fit'],
                                                               rms=new_bkg.globalrms, psf=psf, detect_rad=fit_rad_pix,
                                                               snr=snr, deblend_cont=0.000001, minarea=1)
                if len(new_sep_src_dict['x']) != 0:
                    break

        if len(new_sep_src_dict['x']) == 0:
            new_srcs_found = False
        else:
            new_srcs_found = True

        if band in self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']:
            psf_pix_rad = self.hst_encircle_apertures_wfc3_uvis2_arcsec[band]['ee%i' % 50]
        elif band in self.hst_targets[self.target_name]['acs_wfc1_observed_bands']:
            psf_pix_rad = self.hst_encircle_apertures_acs_wfc1_arcsec[band]['ee%i' % 50]
        elif band in self.nircam_bands:
            psf_pix_rad = self.nircam_encircle_apertures_arcsec[band]['ee%i' % 50]
        elif band in self.miri_bands:
            psf_pix_rad = self.miri_encircle_apertures_arcsec[band]['ee%i' % 50]
        else:
            raise KeyError('the band is not observed ! ')
        #psf_pix_rad = helper_func.transform_world2pix_scale(self.nircam_encircle_apertures_arcsec[band]['ee80'], wcs=wcs)
        print('psf_pix_rad ', psf_pix_rad)

        # evaluate residuals
        source_eval_dict = self.eval_zfit_residuals(pnt_src_amp=fit_result_dict['param_dict_pnt_pix_scale']['amp_pnt'],
                                                    pnt_src_amp_err=fit_result_dict['param_dict_pnt_pix_scale']['amp_pnt_err'],
                                                    pnt_src_x=fit_result_dict['param_dict_pnt_pix_scale']['x0_pnt'],
                                                    pnt_src_y=fit_result_dict['param_dict_pnt_pix_scale']['y0_pnt'],
                                                    gauss_src_x=fit_result_dict['param_dict_gauss_pix_scale']['x0_gauss'],
                                                    gauss_src_y=fit_result_dict['param_dict_gauss_pix_scale']['y0_gauss'],
                                                    new_source_table=new_sep_src_dict, dist_to_ellipse=3,
                                                    gauss_detect_rad=gauss_detect_rad, influenz_rad=psf_pix_rad)

        print('gauss_cand ', source_eval_dict['gauss_cand'])
        print('new_pnt_src_dict ', source_eval_dict['new_pnt_src_dict'])
        print('derop_pnt_src ', source_eval_dict['derop_pnt_src'])
        print('pnt_src_with_neighbour ', source_eval_dict['pnt_src_with_neighbour'])
        print('gauss_src_with_neighbour ', source_eval_dict['gauss_src_with_neighbour'])

        # now add point sources
        # but only add those point sources which are no gaussian candidates
        zfit_param_restrict_dict_pnt = self.zfit_pnt_src_param_dict_from_fit(
            fit_result_dict=fit_result_dict, mask=~source_eval_dict['gauss_cand'],
            fix_mask=np.invert(source_eval_dict['pnt_src_with_neighbour']))
        zfit_param_restrict_dict_gauss = self.zfit_gauss_src_param_dict_from_fit(
            fit_result_dict=fit_result_dict, fix_mask=np.invert(source_eval_dict['gauss_src_with_neighbour']))

        print('zfit_param_restrict_dict_pnt ', zfit_param_restrict_dict_pnt)
        # add new point sources
        new_zfit_param_restrict_dict_pnt = self.zfit_pnt_src_param_dict(
            amp=np.max(data_sub) * 5, x0=source_eval_dict['new_pnt_src_dict']['x'],
            y0=source_eval_dict['new_pnt_src_dict']['y'], sig=np.array(source_eval_dict['new_pnt_src_dict']['b']), fix=False,
            lower_sig=sig_ref*min_sig_fact, upper_sig=sig_ref*max_sig_fact,
            lower_amp=np.max(data_sub)/max_amp_fact, upper_amp=np.max(data_sub)*max_amp_fact,
            lower_x0=(np.array(source_eval_dict['new_pnt_src_dict']['x']) - pos_var),# -
                      #np.array(source_eval_dict['new_pnt_src_dict']['a'])*3),
            upper_x0=(np.array(source_eval_dict['new_pnt_src_dict']['x']) + pos_var),# +
                      #np.array(source_eval_dict['new_pnt_src_dict']['a'])*3),
            lower_y0=(np.array(source_eval_dict['new_pnt_src_dict']['y']) - pos_var),# -
                      #np.array(source_eval_dict['new_pnt_src_dict']['a'])*3),
            upper_y0=(np.array(source_eval_dict['new_pnt_src_dict']['y']) + pos_var),# +
                      #np.array(source_eval_dict['new_pnt_src_dict']['a'])*3),
            starting_index=zfit_param_restrict_dict_pnt['n_src_pnt'])

        n_src_pnt = zfit_param_restrict_dict_pnt['n_src_pnt'] + new_zfit_param_restrict_dict_pnt['n_src_pnt']
        zfit_param_restrict_dict_pnt.update(new_zfit_param_restrict_dict_pnt)
        zfit_param_restrict_dict_pnt['n_src_pnt'] = n_src_pnt
        print('zfit_param_restrict_dict_pnt ', zfit_param_restrict_dict_pnt)
        print('zfit_param_restrict_dict_gauss ', zfit_param_restrict_dict_gauss)

        lower_sig = sig_ref*min_sig_fact
        upper_sig = sig_ref*max_sig_fact
        initial_sig = source_eval_dict['dist_to_gauss_cand'][source_eval_dict['gauss_cand']]/5
        initial_sig[initial_sig <= lower_sig] = sig_ref

        new_zfit_param_restrict_dict_gauss = self.zfit_gauss_src_param_dict(
            amp=fit_result_dict['param_dict_pnt_pix_scale']['amp_pnt'][source_eval_dict['gauss_cand']],
            x0=source_eval_dict['pos_x0_of_gauss_cand'][source_eval_dict['gauss_cand']],
            y0=source_eval_dict['pos_y0_of_gauss_cand'][source_eval_dict['gauss_cand']],
            sig_x=initial_sig,
            sig_y=initial_sig,
            fix=False,
            theta=np.random.uniform(-np.pi/2, np.pi/2, sum(source_eval_dict['gauss_cand'])),
            lower_amp=np.max(data_sub)/max_amp_fact, upper_amp=np.max(data_sub)*max_amp_fact,
            lower_x0=(source_eval_dict['pos_x0_of_gauss_cand'][source_eval_dict['gauss_cand']] -
                      source_eval_dict['dist_to_gauss_cand'][source_eval_dict['gauss_cand']]),
            upper_x0=(source_eval_dict['pos_x0_of_gauss_cand'][source_eval_dict['gauss_cand']] +
                      source_eval_dict['dist_to_gauss_cand'][source_eval_dict['gauss_cand']]),
            lower_y0=(source_eval_dict['pos_y0_of_gauss_cand'][source_eval_dict['gauss_cand']] -
                      source_eval_dict['dist_to_gauss_cand'][source_eval_dict['gauss_cand']]),
            upper_y0=(source_eval_dict['pos_y0_of_gauss_cand'][source_eval_dict['gauss_cand']] +
                      source_eval_dict['dist_to_gauss_cand'][source_eval_dict['gauss_cand']]),
            lower_sig_x=lower_sig, upper_sig_x=upper_sig,
            lower_sig_y=lower_sig, upper_sig_y=upper_sig,
            lower_theta=-np.pi/2, upper_theta=np.pi/2,
            src_name=fit_result_dict['param_dict_pnt_pix_scale']['src_name_pnt'][source_eval_dict['gauss_cand']],
            src_blend=fit_result_dict['param_dict_pnt_pix_scale']['src_blend_pnt'][source_eval_dict['gauss_cand']],
            starting_index=zfit_param_restrict_dict_gauss['n_src_gauss'])

        n_src_gauss = zfit_param_restrict_dict_gauss['n_src_gauss'] + new_zfit_param_restrict_dict_gauss['n_src_gauss']
        zfit_param_restrict_dict_gauss.update(new_zfit_param_restrict_dict_gauss)
        zfit_param_restrict_dict_gauss['n_src_gauss'] = n_src_gauss
        print('zfit_param_restrict_dict_gauss ', zfit_param_restrict_dict_gauss)


        fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(16, 8))

        mean, median, std = sigma_clipped_stats(data_sub, sigma=3.0)

        ax[0].imshow(data_sub, vmin=-std, vmax=20*std, origin='lower')
        ax[1].imshow(fit_result_dict['best_fit'], vmin=-std, vmax=20*std, origin='lower')
        ax[2].imshow(fit_result_dict['best_fit_model'], origin='lower')
        ax[3].imshow(data_sub - fit_result_dict['best_fit'], vmin=-3*std, vmax=3*std, origin='lower')

        ax[0].scatter(fit_result_dict['param_dict_pnt_pix_scale']['x0_pnt'],
                      fit_result_dict['param_dict_pnt_pix_scale']['y0_pnt'], color='b')

        for index_pnt in range(zfit_param_restrict_dict_pnt['n_src_pnt']):
            x = zfit_param_restrict_dict_pnt['pnt_%i' % index_pnt]['init_x0']
            y = zfit_param_restrict_dict_pnt['pnt_%i' % index_pnt]['init_y0']
            ax[0].scatter(x, y, s=8, color='r')

        for index_gauss in range(zfit_param_restrict_dict_gauss['n_src_gauss']):
            x = zfit_param_restrict_dict_gauss['gauss_%i' % index_gauss]['init_x0']
            y = zfit_param_restrict_dict_gauss['gauss_%i' % index_gauss]['init_y0']
            ax[0].scatter(x, y, s=8, color='g')

        for i in range(len(new_sep_src_dict['x'])):
            e = Ellipse(xy=(new_sep_src_dict['x'][i], new_sep_src_dict['y'][i]),
                        width=new_sep_src_dict['a'][i] * 3,
                        height=new_sep_src_dict['b'][i] * 3,
                        angle=new_sep_src_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[0].add_artist(e)
            ax[0].text(new_sep_src_dict['x'][i], new_sep_src_dict['y'][i], '%i' % i)

        # plt.show()
        # plt.clf()
        # plt.close()
        # plt.cla()
        fig.savefig('plot_output/best_fit_%s_%i.png' % (band, fit_iter_index))
        fig.clf()
        plt.close()
        plt.cla()

        return zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss, new_srcs_found

    def transform_params_pix2world(self, param_dict_pnt_pix_scale, param_dict_gauss_pix_scale, wcs):

        position_pnt = wcs.pixel_to_world(param_dict_pnt_pix_scale['x0_pnt'], param_dict_pnt_pix_scale['y0_pnt'])
        x0_pnt_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_pnt_pix_scale['x0_pnt_err'],
                                                           wcs=wcs, dim=0, return_unit='arcsec')
        y0_pnt_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_pnt_pix_scale['y0_pnt_err'],
                                                           wcs=wcs, dim=1, return_unit='arcsec')
        sig_pnt = helper_func.transform_pix2world_scale(pixel_length=param_dict_pnt_pix_scale['sig_pnt'],
                                                        wcs=wcs, dim=0, return_unit='arcsec')
        sig_pnt_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_pnt_pix_scale['sig_pnt_err'],
                                                        wcs=wcs, dim=0, return_unit='arcsec')

        position_gauss = wcs.pixel_to_world(param_dict_gauss_pix_scale['x0_gauss'],
                                            param_dict_gauss_pix_scale['y0_gauss'])
        x0_gauss_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['x0_gauss_err'],
                                                             wcs=wcs, dim=0, return_unit='arcsec')
        y0_gauss_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['y0_gauss_err'],
                                                             wcs=wcs, dim=1, return_unit='arcsec')

        sig_x_gauss = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['sig_x_gauss'],
                                                            wcs=wcs, dim=0, return_unit='arcsec')
        sig_x_gauss_err = helper_func.transform_pix2world_scale(
            pixel_length=param_dict_gauss_pix_scale['sig_x_gauss_err'], wcs=wcs, dim=0, return_unit='arcsec')
        sig_y_gauss = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['sig_y_gauss'],
                                                            wcs=wcs, dim=1, return_unit='arcsec')
        sig_y_gauss_err = helper_func.transform_pix2world_scale(
            pixel_length=param_dict_gauss_pix_scale['sig_y_gauss_err'], wcs=wcs, dim=1, return_unit='arcsec')

        param_dict_pnt_wcs_scale = {'position_pnt': position_pnt, 'x0_pnt_err': x0_pnt_err, 'y0_pnt_err': y0_pnt_err,
                                    'sig_pnt': sig_pnt, 'sig_pnt_err': sig_pnt_err}
        param_dict_gauss_wcs_scale = {
            'position_gauss': position_gauss,
            'sig_x_gauss': sig_x_gauss,
            'sig_y_gauss': sig_y_gauss,
            'x0_gauss_err': x0_gauss_err,
            'y0_gauss_err': y0_gauss_err,
            'sig_x_gauss_err': sig_x_gauss_err,
            'sig_y_gauss_err': sig_y_gauss_err,
        }

        return param_dict_pnt_wcs_scale, param_dict_gauss_wcs_scale

    def get_init_guess_from_other_band(self, init_src_dict, wcs, src_combine_accuracy_pix, data_sub,
                                       sig_ref, min_sig_fact, max_sig_fact, max_amp_fact, pos_var, init_gauss):

        self.n_total_sources = init_src_dict['n_total_sources']

        x0_pnt = np.array(wcs.world_to_pixel(init_src_dict['param_dict_pnt_wcs_scale']['position_pnt'])[0], dtype=float)
        y0_pnt = np.array(wcs.world_to_pixel(init_src_dict['param_dict_pnt_wcs_scale']['position_pnt'])[1], dtype=float)
        x0_pnt_err = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_pnt_wcs_scale']['x0_pnt_err'], wcs=wcs, dim=0), dtype=float)
        y0_pnt_err = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_pnt_wcs_scale']['y0_pnt_err'], wcs=wcs, dim=1), dtype=float)
        src_name_pnt = np.array(init_src_dict['param_dict_pnt_pix_scale']['src_name_pnt'], dtype=int)
        src_blend_pnt = np.array(init_src_dict['param_dict_pnt_pix_scale']['src_blend_pnt'], dtype=object)

        x0_gauss = np.array(wcs.world_to_pixel(init_src_dict['param_dict_gauss_wcs_scale']['position_gauss'])[0],
                            dtype=float)
        y0_gauss = np.array(wcs.world_to_pixel(init_src_dict['param_dict_gauss_wcs_scale']['position_gauss'])[1],
                            dtype=float)
        x0_gauss_err = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_gauss_wcs_scale']['x0_gauss_err'], wcs=wcs, dim=0), dtype=float)
        y0_gauss_err = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_gauss_wcs_scale']['y0_gauss_err'], wcs=wcs, dim=1), dtype=float)
        sig_x_gauss = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_gauss_wcs_scale']['sig_x_gauss'], wcs=wcs, dim=0), dtype=float)
        sig_y_gauss = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_gauss_wcs_scale']['sig_y_gauss'], wcs=wcs, dim=1), dtype=float)
        sig_x_gauss_err = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_gauss_wcs_scale']['sig_x_gauss_err'], wcs=wcs, dim=0),
            dtype=float)
        sig_y_gauss_err = np.array(helper_func.transform_world2pix_scale(
            length_in_arcsec=init_src_dict['param_dict_gauss_wcs_scale']['sig_y_gauss_err'], wcs=wcs, dim=1),
            dtype=float)
        theta_gauss = np.array(init_src_dict['param_dict_gauss_pix_scale']['theta_gauss'], dtype=float)
        src_name_gauss = np.array(init_src_dict['param_dict_gauss_pix_scale']['src_name_gauss'], dtype=int)
        src_blend_gauss = np.array(init_src_dict['param_dict_gauss_pix_scale']['src_blend_gauss'], dtype=object)

        print('x0_pnt ', x0_pnt)
        print('y0_pnt ', y0_pnt)
        print('x0_pnt_err ', x0_pnt_err)
        print('y0_pnt_err ', y0_pnt_err)
        print('src_name_pnt ', src_name_pnt)
        print('src_blend_pnt ', src_blend_pnt)

        print('x0_gauss ', x0_gauss)
        print('y0_gauss ', y0_gauss)
        print('x0_gauss_err ', x0_gauss_err)
        print('y0_gauss_err ', y0_gauss_err)
        print('sig_x_gauss ', sig_x_gauss)
        print('sig_y_gauss ', sig_y_gauss)
        print('sig_x_gauss_err ', sig_x_gauss_err)
        print('sig_y_gauss_err ', sig_y_gauss_err)
        print('src_name_gauss ', src_name_gauss)
        print('src_blend_gauss ', src_blend_gauss)

        # check if the tranformed sigma of the gaussian sources is smaller than the sigma of the PSF
        delete_gauss_src = np.zeros(len(x0_gauss), dtype=bool)
        for index_gauss in range(len(x0_gauss)):
            if ((sig_x_gauss[index_gauss] < sig_ref*2) & (sig_y_gauss[index_gauss] < sig_ref*2)) | (not init_gauss):
                delete_gauss_src[index_gauss] = True
                x0_pnt = np.append(x0_pnt, x0_gauss[index_gauss])
                y0_pnt = np.append(y0_pnt, y0_gauss[index_gauss])
                x0_pnt_err = np.append(x0_pnt_err, x0_gauss_err[index_gauss])
                y0_pnt_err = np.append(y0_pnt_err, y0_gauss_err[index_gauss])
                src_name_pnt = np.append(src_name_pnt, src_name_gauss[index_gauss])
                src_blend_pnt = np.append(src_blend_pnt, src_blend_gauss[index_gauss])

        x0_gauss = np.delete(x0_gauss, delete_gauss_src, 0)
        y0_gauss = np.delete(y0_gauss, delete_gauss_src, 0)
        x0_gauss_err = np.delete(x0_gauss_err, delete_gauss_src, 0)
        y0_gauss_err = np.delete(y0_gauss_err, delete_gauss_src, 0)
        sig_x_gauss = np.delete(sig_x_gauss, delete_gauss_src, 0)
        sig_y_gauss = np.delete(sig_y_gauss, delete_gauss_src, 0)
        sig_x_gauss_err = np.delete(sig_x_gauss_err, delete_gauss_src, 0)
        sig_y_gauss_err = np.delete(sig_y_gauss_err, delete_gauss_src, 0)
        src_name_gauss = np.delete(src_name_gauss, delete_gauss_src, 0)
        src_blend_gauss = np.delete(src_blend_gauss, delete_gauss_src, 0)

        print('x0_pnt ', x0_pnt)
        print('y0_pnt ', y0_pnt)
        print('x0_pnt_err ', x0_pnt_err)
        print('y0_pnt_err ', y0_pnt_err)
        print('src_name_pnt ', src_name_pnt)
        print('src_blend_pnt ', src_blend_pnt)

        print('x0_gauss ', x0_gauss)
        print('y0_gauss ', y0_gauss)
        print('x0_gauss_err ', x0_gauss_err)
        print('y0_gauss_err ', y0_gauss_err)
        print('sig_x_gauss ', sig_x_gauss)
        print('sig_y_gauss ', sig_y_gauss)
        print('sig_x_gauss_err ', sig_x_gauss_err)
        print('sig_y_gauss_err ', sig_y_gauss_err)
        print('src_name_gauss ', src_name_gauss)
        print('src_blend_gauss ', src_blend_gauss)

        name_closest_src_pnt = np.ones(len(x0_pnt), dtype=int) * -999
        for index in range(len(x0_pnt)):
            dist = np.sqrt((x0_pnt - x0_pnt[index]) ** 2 + (y0_pnt - y0_pnt[index]) ** 2)
            # get all objects near the
            index_close_neighbours = np.where((dist < src_combine_accuracy_pix) & (dist != 0))
            # get only the closest
            print('index ', index, 'index_close_neighbours ', index_close_neighbours)
            if len(index_close_neighbours[0]) == 0:
                continue
            smallest_dist = np.min(dist[index_close_neighbours])
            index_smallest_dist = np.where(dist == smallest_dist)
            print('smallest_dist ', smallest_dist, 'index_smallest_dist ', index_smallest_dist)
            if len(index_smallest_dist[0]) > 1:
                index_smallest_dist = index_smallest_dist[0][0]

            name_closest_src_pnt[index] = src_name_pnt[index_smallest_dist]

        print('src_name_pnt ', src_name_pnt)
        print('name_closest_src_pnt ', name_closest_src_pnt)

        new_x0_pnt = []
        new_y0_pnt = []
        new_x0_pnt_err = []
        new_y0_pnt_err = []
        new_src_name_pnt = []
        new_src_blend_pnt = []

        name_list_already_blended_src = []

        for index in range(len(x0_pnt)):

            print('index ', index, ' source_name ', src_name_pnt[index], ' name_closest_src_pnt ',
                  name_closest_src_pnt[index])
            # if no closest neighbour is found, we just take this source
            if name_closest_src_pnt[index] == -999:
                new_x0_pnt.append(float(x0_pnt[index]))
                new_y0_pnt.append(float(y0_pnt[index]))
                new_x0_pnt_err.append(float(x0_pnt_err[index]))
                new_y0_pnt_err.append(float(y0_pnt_err[index]))
                new_src_name_pnt.append(int(src_name_pnt[index]))
                new_src_blend_pnt.append(src_blend_pnt[index])
                continue

            # is the current name also the closest src of the closest src?
            opponents_closest_src_name = name_closest_src_pnt[src_name_pnt == name_closest_src_pnt[index]][0]
            index_closest_src = np.where(src_name_pnt == name_closest_src_pnt[index])

            print('index ', index, ' source_name ', src_name_pnt[index],
                  ' opponents_closest_src_name ', opponents_closest_src_name, ' index_closest_src ', index_closest_src)

            if src_name_pnt[index] in name_list_already_blended_src:
                continue

            if src_name_pnt[index] == opponents_closest_src_name:
                new_x0_pnt.append(np.mean([x0_pnt[index], x0_pnt[index_closest_src[0]][0]]))
                new_y0_pnt.append(np.mean([y0_pnt[index], y0_pnt[index_closest_src[0]][0]]))
                new_x0_pnt_err.append(np.mean([x0_pnt_err[index], x0_pnt_err[index_closest_src[0]][0]]))
                new_y0_pnt_err.append(np.mean([y0_pnt_err[index], y0_pnt_err[index_closest_src[0]][0]]))
                new_src_name_pnt.append(int(self.n_total_sources + 1))
                new_src_blend_pnt.append([src_name_pnt[index], src_name_pnt[index_closest_src[0]][0]])
                name_list_already_blended_src.append(src_name_pnt[index])
                name_list_already_blended_src.append(src_name_pnt[index_closest_src[0]][0])
                self.n_total_sources += 1
            else:
                new_x0_pnt.append(float(x0_pnt[index]))
                new_y0_pnt.append(float(y0_pnt[index]))
                new_x0_pnt_err.append(float(x0_pnt_err[index]))
                new_y0_pnt_err.append(float(y0_pnt_err[index]))
                new_src_name_pnt.append(int(src_name_pnt[index]))
                new_src_blend_pnt.append(src_blend_pnt[index])
        new_x0_pnt = np.array(new_x0_pnt, dtype=float)
        new_y0_pnt = np.array(new_y0_pnt, dtype=float)
        new_x0_pnt_err = np.array(new_x0_pnt_err, dtype=float)
        new_y0_pnt_err = np.array(new_y0_pnt_err, dtype=float)
        new_src_name_pnt = np.array(new_src_name_pnt, dtype=int)
        new_src_blend_pnt = np.array(new_src_blend_pnt, dtype=object)

        print('new_x0_pnt ', new_x0_pnt)
        print('new_y0_pnt ', new_y0_pnt)
        print('new_x0_pnt_err ', new_x0_pnt_err)
        print('new_y0_pnt_err ', new_y0_pnt_err)
        print('new_src_name_pnt ', new_src_name_pnt)
        print('new_src_blend_pnt ', new_src_blend_pnt)

        # create initial fit parameter dicts
        zfit_param_restrict_dict_pnt = self.zfit_pnt_src_param_dict(
            amp=np.max(data_sub)*5, x0=new_x0_pnt, y0=new_y0_pnt, sig=sig_ref, fix=False,
            lower_sig=sig_ref*min_sig_fact, upper_sig=sig_ref*max_sig_fact,
            lower_amp=np.max(data_sub)/max_amp_fact, upper_amp=np.max(data_sub)*max_amp_fact,
            lower_x0=new_x0_pnt - src_combine_accuracy_pix - pos_var,
            upper_x0=new_x0_pnt + src_combine_accuracy_pix + pos_var,
            lower_y0=new_y0_pnt - src_combine_accuracy_pix - pos_var,
            upper_y0=new_y0_pnt + src_combine_accuracy_pix + pos_var,
            starting_index=0)

        lower_sig = sig_ref*min_sig_fact
        upper_sig = sig_ref*max_sig_fact
        sig_x_guess = sig_x_gauss
        sig_y_guess = sig_y_gauss
        sig_x_guess[sig_x_guess < lower_sig] = sig_ref
        sig_y_guess[sig_y_guess < lower_sig] = sig_ref

        zfit_param_restrict_dict_gauss = self.zfit_gauss_src_param_dict(
            amp=np.max(data_sub)*5, x0=x0_gauss, y0=y0_gauss,
            sig_x=sig_x_guess, sig_y=sig_y_guess, theta=theta_gauss, src_name=src_name_gauss, src_blend=src_blend_gauss,
            fix=False,
            lower_amp=np.max(data_sub)/max_amp_fact, upper_amp=np.max(data_sub)*max_amp_fact,
            lower_x0=x0_gauss - src_combine_accuracy_pix - pos_var, upper_x0=x0_gauss + src_combine_accuracy_pix + pos_var,
            lower_y0=y0_gauss - src_combine_accuracy_pix - pos_var, upper_y0=y0_gauss + src_combine_accuracy_pix + pos_var,
            lower_sig_x=lower_sig, upper_sig_x=upper_sig, lower_sig_y=lower_sig, upper_sig_y=upper_sig,
            lower_theta=-np.pi/2, upper_theta=np.pi/2, starting_index=0)

        return zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss

    def zfit_model2data(self, band, x_grid, y_grid, data_sub, err, zfit_param_restrict_dict_pnt,
                        zfit_param_restrict_dict_gauss, index_iter, wcs, fit_rad):

        # init fit model
        fit_rad_pix = helper_func.transform_world2pix_scale(length_in_arcsec=fit_rad, wcs=wcs, dim=0)
        self.init_zfit_models(band=band, img_x_grid=x_grid, img_y_grid=y_grid, img_data=data_sub, img_err=err,
                              n_pnt=zfit_param_restrict_dict_pnt['n_src_pnt'],
                              n_gauss=zfit_param_restrict_dict_gauss['n_src_gauss'], fit_rad_pix=fit_rad_pix)
        # create init zfit parameters
        print('zfit_param_restrict_dict_pnt ', zfit_param_restrict_dict_pnt)

        params_pnt = self.create_zfit_pnt_param_list(zfit_param_restrict_dict_pnt=zfit_param_restrict_dict_pnt,
                                                     band=band, index_iter=index_iter)
        params_gauss = self.create_zfit_gauss_param_list(zfit_param_restrict_dict_gauss=zfit_param_restrict_dict_gauss,
                                                         band=band, index_iter=index_iter)

        print('params_pnt ', params_pnt)
        print('n_pnt ', zfit_param_restrict_dict_pnt['n_src_pnt'])
        print('params_gauss ', params_gauss)
        print('n_gauss ', zfit_param_restrict_dict_gauss['n_src_gauss'])

        # init loss model and fit
        loss = zfit.loss.SimpleLoss(self.squared_loss, params_gauss + params_pnt, errordef=1)
        minimizer = zfit.minimize.Minuit()
        # fit
        start = time.time()
        minimization_result = minimizer.minimize(loss)
        end = time.time()
        print('Time of fitting ', end - start)
        print('Is Fit valid? ', minimization_result.valid)
        start = time.time()
        hesse_result = minimization_result.hesse()
        end = time.time()
        print('Time to compute Hessian ', end - start)
        print(minimization_result)

        # get fit parameters
        param_dict_pnt_pix_scale, param_dict_gauss_pix_scale = self.get_zfit_params(
            minimization_result=minimization_result,
            hesse_result=hesse_result,
            zfit_param_restrict_dict_pnt=zfit_param_restrict_dict_pnt,
            zfit_param_restrict_dict_gauss=zfit_param_restrict_dict_gauss)
        print('param_dict_pnt_pix_scale ', param_dict_pnt_pix_scale)
        print('param_dict_gauss_pix_scale ', param_dict_gauss_pix_scale)

        # convert position parameters into WCS format
        param_dict_pnt_wcs_scale, param_dict_gauss_wcs_scale = \
            self.transform_params_pix2world(param_dict_pnt_pix_scale=param_dict_pnt_pix_scale,
                                            param_dict_gauss_pix_scale=param_dict_gauss_pix_scale, wcs=wcs)
        print('param_dict_pnt_wcs_scale ', param_dict_pnt_wcs_scale)
        print('param_dict_gauss_wcs_scale ', param_dict_gauss_wcs_scale)

        # get best fit
        best_fit, best_fit_model = self.assemble_zfit_model(param_dict_pnt_pix_scale=param_dict_pnt_pix_scale,
                                                            param_dict_gauss_pix_scale=param_dict_gauss_pix_scale,
                                                            band=band)

        flux_pnt_src, flux_pnt_src_err = self.compute_zfit_pnt_src_flux(
            param_dict_pnt_pix_scale=param_dict_pnt_pix_scale, band=band)
        flux_gauss_src, flux_gauss_src_err = self.compute_zfit_gauss_src_flux(
            param_dict_gauss_pix_scale=param_dict_gauss_pix_scale,
            band=band)

        # dict to return
        fit_result_dict = {'param_dict_pnt_pix_scale': param_dict_pnt_pix_scale,
                           'param_dict_gauss_pix_scale': param_dict_gauss_pix_scale,
                           'param_dict_pnt_wcs_scale': param_dict_pnt_wcs_scale,
                           'param_dict_gauss_wcs_scale': param_dict_gauss_wcs_scale,
                           'flux_pnt_src': flux_pnt_src, 'flux_pnt_src_err': flux_pnt_src_err,
                           'flux_gauss_src': flux_gauss_src, 'flux_gauss_src_err': flux_gauss_src_err,
                           'best_fit': best_fit, 'best_fit_model': best_fit_model,
                           'n_total_sources': self.n_total_sources}

        return fit_result_dict

    def zfit_iter_de_blend(self, band, ra, dec, cutout_size, init_src_dict=None, n_iter=2,
                           src_combine_accuracy_pix=1.5, sig_ref=0.5, min_sig_fact=0.1, max_sig_fact=20, max_amp_fact=1e5, pos_var=4,
                           gauss_detect_rad=0, init_gauss=True):

        # prepare all data
        # get cutout
        cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size, band_list=[band],
                                                include_err=True)
        # get WCS, data, uncertainties and psf
        data = np.array(cutout_dict['%s_img_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        err = np.array(cutout_dict['%s_err_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        psf = np.array(self.small_psf_dict['native_psf_%s' % band].byteswap().newbyteorder(), dtype=float)
        # calculate the background
        bkg = sep.Background(np.array(data, dtype=float))
        # data subtracted from a global background estimation
        data_sub = data - bkg.globalback
        # get the WCS
        wcs = cutout_dict['%s_img_cutout' % band].wcs
        # get data grid
        x_grid, y_grid = helper_func.create_2d_data_mesh(data=data)

        # get initial fitting parameters
        if init_src_dict is None:
            # create SEP object table using standard parameters
            sep_src_dict = self.get_sep_init_src_guess(data_sub=data_sub, rms=bkg.globalrms, psf=psf, snr=5)

            # init parameters for point sources. note that `a´ is the major axis of the elliptical sources
            zfit_param_restrict_dict_pnt = self.zfit_pnt_src_param_dict(
                amp=np.max(data_sub) * 5, x0=sep_src_dict['x'], y0=sep_src_dict['y'], sig=sig_ref, fix=False,
                lower_sig=sig_ref * min_sig_fact, upper_sig=sig_ref * max_sig_fact,
                lower_amp=np.max(data_sub)/max_amp_fact, upper_amp=np.max(data_sub) * max_amp_fact,
                lower_x0=np.array(sep_src_dict['x']) - np.array(sep_src_dict['a'])*3 - pos_var,
                upper_x0=np.array(sep_src_dict['x']) + np.array(sep_src_dict['a'])*3 + pos_var,
                lower_y0=np.array(sep_src_dict['y']) - np.array(sep_src_dict['a'])*3 - pos_var,
                upper_y0=np.array(sep_src_dict['y']) + np.array(sep_src_dict['a'])*3 + pos_var)
            # no gaussian sources for initial guess
            zfit_param_restrict_dict_gauss = {'n_src_gauss': 0}
        else:
            # if from a previous fit parameters are known parameters will be transformed
            zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss = \
                self.get_init_guess_from_other_band(init_src_dict=init_src_dict, wcs=wcs, src_combine_accuracy_pix=src_combine_accuracy_pix,
                                                    data_sub=data_sub,  sig_ref=sig_ref, min_sig_fact=min_sig_fact,
                                                    max_sig_fact=max_sig_fact, max_amp_fact=max_amp_fact,
                                                    pos_var=pos_var, init_gauss=init_gauss)

        iterative_fit_result_dict = {}

        # iterative fitting
        for fit_iter_index in range(n_iter):
            fit_result_dict = self.zfit_model2data(band=band, x_grid=x_grid, y_grid=y_grid, data_sub=data_sub, err=err,
                                                   zfit_param_restrict_dict_pnt=zfit_param_restrict_dict_pnt,
                                                   zfit_param_restrict_dict_gauss=zfit_param_restrict_dict_gauss,
                                                   index_iter=fit_iter_index, wcs=wcs)

            iterative_fit_result_dict.update({'fit_results_band_%s_iter_%i' % (band, fit_iter_index): fit_result_dict})
            iterative_fit_result_dict.update({'last_iter_index': fit_iter_index})

            zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss, new_srcs_found = \
                self.generate_new_init_zfit_model(data_sub=data_sub, psf=psf, fit_result_dict=fit_result_dict,
                                                  band=band, wcs=wcs,
                                                  sig_ref=sig_ref, min_sig_fact=min_sig_fact, max_sig_fact=max_sig_fact,
                                                  max_amp_fact=max_amp_fact, fit_iter_index=fit_iter_index, pos_var=pos_var, gauss_detect_rad=gauss_detect_rad)

            if not new_srcs_found:
                break

        return iterative_fit_result_dict

    def src_deblend(self, band, ra, dec, cutout_size, fit_rad=0.3, init_src_dict=None, n_iter=2,
                    pos_var=3, sig_ref=0.5, min_sig_fact=0.1, max_sig_fact=20, max_amp_fact=1e5,
                    src_combine_accuracy_pix=1.5,
                    gauss_detect_rad=4, init_gauss=True, init_snr=5):

        # prepare all data
        # get cutout
        cutout_dict = self.get_band_cutout_dict(ra_cutout=ra, dec_cutout=dec, cutout_size=cutout_size, band_list=[band],
                                                include_err=True)
        # get WCS, data, uncertainties and psf
        data = np.array(cutout_dict['%s_img_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        err = np.array(cutout_dict['%s_err_cutout' % band].data.byteswap().newbyteorder(), dtype=float)
        psf = np.array(self.small_psf_dict['native_psf_%s' % band].byteswap().newbyteorder(), dtype=float)
        # calculate the background
        bkg = sep.Background(np.array(data, dtype=float))
        # data subtracted from a global background estimation
        data_sub = data - bkg.globalback
        # get the WCS
        wcs = cutout_dict['%s_img_cutout' % band].wcs
        # get data grid
        x_grid, y_grid = helper_func.create_2d_data_mesh(data=data)

        # get initial fitting parameters
        if init_src_dict is None:
            # only consider sources inside the radius of interest
            fit_rad_pix = helper_func.transform_world2pix_scale(length_in_arcsec=fit_rad, wcs=wcs, dim=0)
            # create SEP object table using standard parameters
            sep_src_dict = self.get_sep_init_src_guess(data_sub=data_sub, rms=bkg.globalrms, psf=psf, detect_rad=fit_rad_pix, snr=init_snr)

            # init parameters for point sources. note that `a´ is the major axis of the elliptical sources
            zfit_param_restrict_dict_pnt = self.zfit_pnt_src_param_dict(
                amp=np.max(data_sub) * 5, x0=sep_src_dict['x'], y0=sep_src_dict['y'], sig=np.array(sep_src_dict['b']), fix=False,
                lower_sig=sig_ref * min_sig_fact, upper_sig=sig_ref * max_sig_fact,
                lower_amp=np.max(data_sub)/max_amp_fact, upper_amp=np.max(data_sub) * max_amp_fact,
                lower_x0=np.array(sep_src_dict['x']) - pos_var,# - np.array(sep_src_dict['b'])*3,
                upper_x0=np.array(sep_src_dict['x']) + pos_var,# + np.array(sep_src_dict['b'])*3,
                lower_y0=np.array(sep_src_dict['y']) - pos_var,# - np.array(sep_src_dict['b'])*3,
                upper_y0=np.array(sep_src_dict['y']) + pos_var,# + np.array(sep_src_dict['b'])*3
            )
            # no gaussian sources for initial guess
            zfit_param_restrict_dict_gauss = {'n_src_gauss': 0}
        else:
            # if from a previous fit parameters are known parameters will be transformed
            zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss = \
                self.get_init_guess_from_other_band(init_src_dict=init_src_dict, wcs=wcs, src_combine_accuracy_pix=src_combine_accuracy_pix,
                                                    data_sub=data_sub,  sig_ref=sig_ref, min_sig_fact=min_sig_fact,
                                                    max_sig_fact=max_sig_fact, max_amp_fact=max_amp_fact,
                                                    pos_var=pos_var, init_gauss=init_gauss)

        iterative_fit_result_dict = {}

        # iterative fitting
        for fit_iter_index in range(n_iter):
            fit_result_dict = self.zfit_model2data(band=band, x_grid=x_grid, y_grid=y_grid, data_sub=data_sub, err=err,
                                                   zfit_param_restrict_dict_pnt=zfit_param_restrict_dict_pnt,
                                                   zfit_param_restrict_dict_gauss=zfit_param_restrict_dict_gauss,
                                                   index_iter=fit_iter_index, wcs=wcs, fit_rad=fit_rad)

            iterative_fit_result_dict.update({'fit_results_band_%s_iter_%i' % (band, fit_iter_index): fit_result_dict})
            iterative_fit_result_dict.update({'last_iter_index': fit_iter_index})

            zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss, new_srcs_found = \
                self.generate_new_init_zfit_model(data_sub=data_sub, psf=psf, fit_result_dict=fit_result_dict,
                                                  band=band, wcs=wcs, fit_rad=fit_rad,
                                                  sig_ref=sig_ref, min_sig_fact=min_sig_fact, max_sig_fact=max_sig_fact,
                                                  max_amp_fact=max_amp_fact, fit_iter_index=fit_iter_index,
                                                  pos_var=pos_var, gauss_detect_rad=gauss_detect_rad, init_snr=init_snr)

            if not new_srcs_found:
                break

        return iterative_fit_result_dict


class CigaleModelWrapper(basic_attributes.PhangsDataStructure):
    """
    Wrapper to access cigale fits
    """
    def __init__(self):

        super().__init__()

        # initial params
        self.cigale_init_params = {
            'data_file': '',
            'parameters_file': '',
            'sed_modules': ['sfh2exp', 'bc03', 'dustext', 'redshifting'],
            'analysis_method': 'savefluxes',
            'cores': 1,
        }
        # # bands used in the fit
        # self.band_dict = {
        #     # Bands to consider. To consider uncertainties too, the name of the band
        #     # must be indicated with the _err suffix. For instance: FUV, FUV_err.
        #     'bands': '',
        # }
        # parameters for the analysis
        self.analysis_params = {
            # List of the physical properties to save. Leave empty to save all the
            # physical properties (not recommended when there are many models).
            'variables': '',
            # If True, save the generated spectrum for each model.
            'save_sed': True,
            # Number of blocks to compute the models. Having a number of blocks
            # larger than 1 can be useful when computing a very large number of
            # models or to split the result file into smaller files.
            'blocks': 1
        }

        # sed model parameters
        self.ssp_params = {
            # Index of the SSP to use.
            'index': [0]
        }
        self.sfh2exp_params = {
            # e-folding time of the main stellar population model in Myr.
            'tau_main': [0.001],
            # e-folding time of the late starburst population model in Myr.
            'tau_burst': [0.001],
            # Mass fraction of the late burst population.
            'f_burst': [0.0],
            # Age of the main stellar population in the galaxy in Myr. The precision
            # is 1 Myr.
            'age': [5],
            # Age of the late burst in Myr. The precision is 1 Myr.
            'burst_age': [1],
            # Value of SFR at t = 0 in M_sun/yr.
            'sfr_0': [1.0],
            # Normalise the SFH to produce one solar mass.
            'normalise': [True]
        }
        self.bc03_params = {
            # Initial mass function: 0 (Salpeter) or 1 (Chabrier).
            'imf': [1],
            # Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05.
            'metallicity': [0.02],
            # Age [Myr] of the separation between the young and the old star
            # populations. The default value in 10^7 years (10 Myr). Set to 0 not to
            # differentiate ages (only an old population).
            'separation_age': [10]
        }
        self.bc03_ssp_params = {
            # Initial mass function: 0 (Salpeter) or 1 (Chabrier).
            'imf': [1],
            # Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05.
            'metallicity': [0.02],
            # Age [Myr] of the separation between the young and the old star
            # populations. The default value in 10^7 years (10 Myr). Set to 0 not to
            # differentiate ages (only an old population).
            'separation_age': [10]
        }
        self.nebular_params = {
            # Ionisation parameter. Possible values are: -4.0, -3.9, -3.8, -3.7,
            # -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6,
            # -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5,
            # -1.4, -1.3, -1.2, -1.1, -1.0.
            'logU': [-2.0],
            # Gas metallicity. Possible values are: 0.000, 0.0004, 0.001, 0.002,
            # 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.011, 0.012,
            # 0.014, 0.016, 0.019, 0.020, 0.022, 0.025, 0.03, 0.033, 0.037, 0.041,
            # 0.046, 0.051.
            'zgas': [0.02],
            # Electron density. Possible values are: 10, 100, 1000.
            'ne': [100],
            # Fraction of Lyman continuum photons escaping the galaxy. Possible
            # values between 0 and 1.
            'f_esc': [0.0],
            # Fraction of Lyman continuum photons absorbed by dust. Possible values
            # between 0 and 1.
            'f_dust': [0.0],
            # Line width in km/s.
            'lines_width': [100.0],
            # Include nebular emission.
            'emission': True
        }
        self.dustext_params = {
            # E(B-V), the colour excess.
            'E_BV': [0],
            # Ratio of total to selective extinction, A_V / E(B-V). The standard
            # value is 3.1 for MW using CCM89. For SMC and LMC using Pei92 the
            # values should be 2.93 and 3.16.
            'Rv': [3.1],
            # Extinction law to apply. The values are 0 for CCM, 1 for SMC, and 2
            # for LCM.
            'law': [0],
            # Filters for which the extinction will be computed and added to the SED
            # information dictionary. You can give several filter names separated by
            # a & (don't use commas).
            'filters': ['B_B90 & V_B90 & FUV']
        }
        self.dustextPHANGS_params = {
            # Attenuation at 550 nm.
            'A550': [0.0],
            'filters': 'B_B90 & V_B90 & FUV'
        }
        self.dl2014_params = {
            # Mass fraction of PAH. Possible values are: 0.47, 1.12, 1.77, 2.50,
            # 3.19, 3.90, 4.58, 5.26, 5.95, 6.63, 7.32.
            'qpah': [2.5],
            # Minimum radiation field. Possible values are: 0.100, 0.120, 0.150,
            # 0.170, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500, 0.600, 0.700, 0.800,
            # 1.000, 1.200, 1.500, 1.700, 2.000, 2.500, 3.000, 3.500, 4.000, 5.000,
            # 6.000, 7.000, 8.000, 10.00, 12.00, 15.00, 17.00, 20.00, 25.00, 30.00,
            # 35.00, 40.00, 50.00.
            'umin': [1.0],
            # Powerlaw slope dU/dM propto U^alpha. Possible values are: 1.0, 1.1,
            # 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
            # 2.6, 2.7, 2.8, 2.9, 3.0.
            'alpha': [2.0],
            # Fraction illuminated from Umin to Umax. Possible values between 0 and
            # 1.
            'gamma': [0.1],
            # Take self-absorption into account.
            'self_abs': False,
        }
        self.redshifting_params = {
            # Redshift of the objects. Leave empty to use the redshifts from the
            # input file.
            'redshift': [0.0]
        }

        # cigale model table
        self.model_table = None
        self.model_table_dict = None

    def configure_cigale_model_params(self, data_file='', parameters_file='', sed_modules=None,
                                      analysis_method='savefluxes', ncors=4):

        if sed_modules is None:
            sed_modules = ['sfh2exp', 'bc03', 'dustext', 'redshifting']

        self.cigale_init_params['data_file'] = data_file
        self.cigale_init_params['parameters_file'] = parameters_file
        self.cigale_init_params['sed_modules'] = sed_modules
        self.cigale_init_params['analysis_method'] = analysis_method
        self.cigale_init_params['cores'] = ncors

    def configurate_hst_cc_fit(self, data_file, band_list, target_name, ncores=4, met=0.02, ebv=None):
        self.configure_cigale_model_params(data_file=data_file,
                                           sed_modules=['sfh2exp', 'bc03', 'dustext', 'redshifting'],
                                           analysis_method='pdf_analysis', ncors=ncores)

        band_name_list = self.compute_cigale_band_name_list(band_list=band_list, target_name=target_name)

        #self.band_dict = {'bands': band_name_list}

        if ebv is None:
            ebv = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                     0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33,
                     0.34, 0.35000000000000003, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41000000000000003, 0.42, 0.43, 0.44,
                     0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56,
                     0.5700000000000001, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68,
                     0.6900000000000001, 0.7000000000000001, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
                     0.81, 0.8200000000000001, 0.8300000000000001, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92,
                     0.93, 0.9400000000000001, 0.9500000000000001, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04,
                     1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001,
                     1.1500000000000001, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28,
                     1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001,
                     1.4000000000000001, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5]

        self.analysis_params = {
            'variables': ['stellar.m_star', 'attenuation.E_BV', 'sfh.age'],
            'save_best_sed': True,
            'save_chi2': 'all',
            'lim_flag': 'none',
            'mock_flag': False,
            'redshift_decimals': 2,
            'blocks': 1
        }
        self.sfh2exp_params = {
            'tau_main': [0.001],
            'tau_burst': [0.001],
            'f_burst': [0.0],
            'age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22, 24, 26, 28, 30, 32, 34, 37,
                    40, 43, 46, 49, 53, 57, 61, 66, 71, 76, 82, 88, 95, 102, 110, 118, 127, 136, 147, 158, 169, 182,
                    196, 210, 226, 243, 261, 281, 302, 324, 349, 375, 403, 433, 465, 500, 537, 577, 621, 667, 717, 770,
                    828, 890, 956, 1028, 1105, 1187, 1276, 1371, 1474, 1584, 1702, 1829, 1966, 2113, 2271, 2440, 2623,
                    2819, 3029, 3255, 3499, 3760, 4041, 4343, 4667, 5015, 5390, 5793, 6225, 6690, 7190, 7727, 8304,
                    8925, 9591, 10308, 11077, 11905, 12794, 13750],
            'burst_age': [1],
            'sfr_0': [1.0],
            'normalise': [True]
        }
        self.bc03_params = {
            'imf': [1],
            'metallicity': [met],
            'separation_age': [10]
        }
        self.dustext_params = {
            'E_BV': ebv,
            'Rv': [3.1],
            'law': [0],
            'filters': ['B_B90 & V_B90 & FUV']
        }
        self.redshifting_params = {
            'redshift ': [0.0]
        }

    def configurate_hst_nircam_cc_fit(self, data_file, band_list, target_name, ncores=4):
        self.configure_cigale_model_params(data_file=data_file,
                                           sed_modules=['sfh2exp', 'bc03', 'dustext', 'redshifting'],
                                           analysis_method='pdf_analysis', ncors=ncores)

        band_name_list = self.compute_cigale_band_name_list(band_list=band_list, target_name=target_name)

        #self.band_dict = {'bands': band_name_list}

        self.analysis_params = {
            'variables': ['stellar.m_star', 'attenuation.E_BV', 'sfh.age'],
            'save_best_sed': True,
            'save_chi2': 'all',
            'lim_flag': 'none',
            'mock_flag': False,
            'redshift_decimals': 2,
            'blocks': 1
        }
        self.sfh2exp_params = {
            'tau_main': [0.001],
            'tau_burst': [0.001],
            'f_burst': [0.0],
            'age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 21, 22, 24, 26, 28, 30, 32, 34, 37,
                    40, 43, 46, 49, 53, 57, 61, 66, 71, 76, 82, 88, 95, 102, 110, 118, 127, 136, 147, 158, 169, 182,
                    196, 210, 226, 243, 261, 281, 302, 324, 349, 375, 403, 433, 465, 500, 537, 577, 621, 667, 717, 770,
                    828, 890, 956, 1028, 1105, 1187, 1276, 1371, 1474, 1584, 1702, 1829, 1966, 2113, 2271, 2440, 2623,
                    2819, 3029, 3255, 3499, 3760, 4041, 4343, 4667, 5015, 5390, 5793, 6225, 6690, 7190, 7727, 8304,
                    8925, 9591, 10308, 11077, 11905, 12794, 13750],
            'burst_age': [1],
            'sfr_0': [1.0],
            'normalise': [True]
        }
        self.bc03_params = {
            'imf': [1],
            'metallicity': [0.02],
            'separation_age': [10]
        }
        self.dustext_params = {
            'E_BV': [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                     0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33,
                     0.34, 0.35000000000000003, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41000000000000003, 0.42, 0.43, 0.44,
                     0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56,
                     0.5700000000000001, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68,
                     0.6900000000000001, 0.7000000000000001, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8,
                     0.81, 0.8200000000000001, 0.8300000000000001, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92,
                     0.93, 0.9400000000000001, 0.9500000000000001, 0.96, 0.97, 0.98, 0.99, 1.0, 1.01, 1.02, 1.03, 1.04,
                     1.05, 1.06, 1.07, 1.08, 1.09, 1.1, 1.11, 1.12, 1.1300000000000001, 1.1400000000000001,
                     1.1500000000000001, 1.16, 1.17, 1.18, 1.19, 1.2, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26, 1.27, 1.28,
                     1.29, 1.3, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.37, 1.3800000000000001, 1.3900000000000001,
                     1.4000000000000001, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46, 1.47, 1.48, 1.49, 1.5],
            'Rv': [3.1],
            'law': [0],
            'filters': ['B_B90 & V_B90 & FUV']
        }
        self.redshifting_params = {
            'redshift ': [0.0]
        }

    def compute_cigale_band_name_list(self, band_list, target_name):

        band_name_list = []
        for band in band_list:
            if band in self.hst_targets[target_name]['acs_wfc1_observed_bands']:
                band_name_list.append(band + '_ACS')
            elif band in self.hst_targets[target_name]['wfc3_uvis_observed_bands']:
                band_name_list.append(band + '_UVIS_CHIP2')
            elif band in self.nircam_targets[target_name]['observed_bands']:
                band_name_list.append('jwst.nircam.' + band)
            elif band in self.miri_targets[target_name]['observed_bands']:
                band_name_list.append('jwst.miri.' + band)

        return band_name_list

    def create_cigale_flux_file(self, file_path, band_list, target_name, flux_dict, name_list=None,
                                redshift_list=None, dist_list=None):

        n_obj = flux_dict['n_obj']

        if name_list is None:
            name_list = np.arange(start=0,  stop=n_obj+1)
        if redshift_list is None:
            redshift_list = [0.0] * n_obj
        if dist_list is None:
            dist_list = [self.dist_dict[target_name]['dist']] * n_obj

        name_list = np.array(name_list, dtype=str)
        redshift_list = np.array(redshift_list)
        dist_list = np.array(dist_list)

        # create flux file
        flux_file = open(file_path, "w")
        # add header for all variables
        band_name_list = self.compute_cigale_band_name_list(band_list=band_list, target_name=target_name)
        flux_file.writelines("# id             redshift  distance   ")
        for band_name in band_name_list:
            flux_file.writelines(band_name + "   ")
            flux_file.writelines(band_name + "_err" + "   ")
        flux_file.writelines(" \n")

        for index in range(n_obj):
            flux_file.writelines(" %s   %f   %f  " % (name_list[index], redshift_list[index], dist_list[index]))
            for band in band_list:
                flux_file.writelines("%.15f   " % flux_dict['flux_' + band][index])
                flux_file.writelines("%.15f   " % flux_dict['flux_' + band + '_err'][index])
            flux_file.writelines(" \n")
        flux_file.close()

    @staticmethod
    def replace_params_in_file(param_dict, file_name='pcigale.ini'):
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for key in param_dict.keys():
            line_index = [i for i in range(len(lines)) if lines[i].startswith(key)]
            prefix = ''
            if not line_index:
                line_index = [i for i in range(len(lines)) if lines[i].startswith('  ' + key)]
                prefix = '  '
            if not line_index:
                line_index = [i for i in range(len(lines)) if lines[i].startswith('    ' + key)]
                prefix = '    '
            if len(line_index) > 1:
                raise KeyError('There is apparently more than one line beginning with <<', key, '>>')
            if isinstance(param_dict[key], (str, bool, int, float)):
                new_line = prefix + key + ' = ' + str(param_dict[key]) + '\n'
            elif isinstance(param_dict[key], list):
                new_line = prefix + key + ' = '
                for obj in param_dict[key]:
                    new_line += str(obj) + ', '
            else:
                raise KeyError('The given parameters mus be of type str, bool, int, float or a list of these types')
            # check if there is no , at the end
            if new_line[-2:] == ', ':
                new_line = new_line[:-2]
            lines[line_index[0]] = new_line
        with open(file_name, 'w', encoding='utf-8') as file:
            file.writelines(lines)

    def run_cigale(self, delete_old_models=True, save_pdf=False):
        # initiate pcigale
        os.system('pcigale init')
        # set initial parameters
        self.replace_params_in_file(param_dict=self.cigale_init_params, file_name='pcigale.ini')
        # configurate pcigale
        os.system('pcigale genconf')
        # set module configurations
        for modul in self.cigale_init_params['sed_modules']:
            self.replace_params_in_file(param_dict=getattr(self, modul + '_params'), file_name='pcigale.ini')
        # change analysis parameters
        self.replace_params_in_file(param_dict=self.analysis_params, file_name='pcigale.ini')
        if save_pdf:
            with open('pcigale.ini', 'a') as file:
                file.write('  save_pdf = True')
        # run pcigale
        os.system('pcigale run')
        # delete old models
        if delete_old_models:
            os.system('rm -rf *_out')







    def plot_hst_nircam_miri_filters(self, ax, fontsize=15, alpha=0.3):
        for i in range(len(self.filter_colors)):
            ax.axvspan(self.filter_lam_nm[i]-self.filter_fwhm_nm[i], self.filter_lam_nm[i]+self.filter_fwhm_nm[i],
                       alpha=alpha, color=self.filter_colors[i], zorder=0)

            if (i == 3) or (i == 6):
                ax.text(self.filter_lam_nm[i], 2e-7, self.filter_labels[i], fontsize=fontsize, rotation=90)
            else:
                ax.text(self.filter_lam_nm[i], 2e-7, self.filter_labels[i], fontsize=fontsize, rotation=90, ha='center')

        for i in range(len(self.hst_lam_nm)):
            ax.axvspan(self.hst_lam_nm[i] - self.hst_fwhm[i], self.hst_lam_nm[i] + self.hst_fwhm[i], alpha=alpha,
                       color='#636363')

    def plot_hst_filters(self, ax, fontsize=15, alpha=0.3):

        for i in range(len(self.hst_lam_nm)):
            ax.axvspan(self.hst_lam_nm[i] - self.hst_fwhm[i], self.hst_lam_nm[i] + self.hst_fwhm[i], alpha=alpha,
                       color=self.hst_filter_colors[i])
            ax.text(self.hst_lam_nm[i], 2e-7, self.hst_filter_labels[i], fontsize=fontsize, rotation=90, ha='center')

    @staticmethod
    def plot_cigale_model(ax, model_file_name, cluster_mass, distance_Mpc, label=None, linestyle='-', color='k', linewidth=2):
        # read in the invidual model/sed
        mod = fits.open(model_file_name)[1].data
        # get wavelengths in nanometer
        lam_nm = mod['wavelength'] * u.nm
        # covert to micron if necessary later
        lam_um = lam_nm.to(u.um)
        # frequencies of the wavelengths
        freq = (const.c.to(u.nm/u.s) / lam_nm).to(u.Hz)

        # get Fnu in mJy units
        mod_Fnu_mJy = mod['Fnu'] * u.mJy
        # get Luminosity in W / nm / Msun units
        L_WnmMsun = mod['L_lambda_total'] * u.W / u.nm / u.Msun

        # convert Luminosity to W/Msun
        L_WMsun = L_WnmMsun * lam_nm

        # scale the Luminosity to the chosen star cluster mass
        L_W = L_WMsun * cluster_mass

        # convert distance from Mpc to meters
        distance_m = distance_Mpc.to(u.m)

        # convert luminsoity to Flux [W/m2]
        F_Wm2 = L_W / (4*np.pi*distance_m**2)

        # Convert that Flux to Flux density in Jy
        Fnu_Jy = (F_Wm2/freq).to(u.Jy)
        # convert to mJy
        Fnu_mJy = Fnu_Jy.to(u.mJy)

        # # convert to AB mag
        # Fnu_AB = Fnu_mJy.to(u.AB)

        # plot
        ax.plot(lam_nm, Fnu_mJy, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

    def load_cigale_model_block(self, model_block_file_name='out/models-block-0.fits'):
        self.model_table = Table.read(model_block_file_name)

        # create dictionary
        model_table_dict = {}
        # id
        model_table_dict.update({'id': {'value': self.model_table['id'], 'label': r'ID'}})
        # stellar parameters
        #model_table_dict.update({'ssp.index': {'value': self.model_table['ssp.index'], 'label': r'Index$_{*}$'} })


        model_table_dict.update({'sfh.age': {'value': self.model_table['sfh.age'], 'label': r'Age$_{*}$'} })
        model_table_dict.update({'stellar.metallicity': {'value': self.model_table['stellar.metallicity'], 'label': r'Metal$_{*}$'} })


        # dust parameters
        model_table_dict.update({'attenuation.E_BV': {'value': self.model_table['attenuation.E_BV'], 'label': r'E(B-V)'} })

        # model_table_dict.update({'dl2014.alpha': {'value': self.model_table['dust.alpha'], 'label': r'$\alpha_{\rm Dust}$'} })
        # model_table_dict.update({'dl2014.gamma': {'value': self.model_table['dust.gamma'], 'label': r'$\gamma_{\rm Dust}$'} })
        # model_table_dict.update({'dl2014.qpah': {'value': self.model_table['dust.qpah'], 'label': r'Q-PAH$_{\rm Dust}$'} })
        # model_table_dict.update({'dl2014.umean': {'value': self.model_table['dust.umean'], 'label': r'U-mean$_{\rm Dust}$'} })
        # model_table_dict.update({'dl2014.umin': {'value': self.model_table['dust.umin'], 'label': r'U-min$_{\rm Dust}$'} })
        # # nebular parameters
        # model_table_dict.update({'nebular.f_dust': {'value': self.model_table['nebular.f_dust'], 'label': r'f$_{\rm Dust}$'} })
        # model_table_dict.update({'nebular.f_esc': {'value': self.model_table['nebular.f_esc'], 'label': r'f${\rm esc}$'} })
        # model_table_dict.update({'nebular.lines_width': {'value': self.model_table['nebular.lines_width'], 'label': r'FWHM$_{\rm Gas}$'} })
        # model_table_dict.update({'nebular.logU': {'value': self.model_table['nebular.logU'], 'label': r'logU$_{\rm Gas}$'} })
        # model_table_dict.update({'nebular.ne': {'value': self.model_table['nebular.ne'], 'label': r'ne$_{\rm Gas}$'} })
        # model_table_dict.update({'nebular.zgas': {'value': self.model_table['nebular.zgas'], 'label': r'Metal$_{\rm Gas}$'} })

        self.model_table_dict = model_table_dict

    def create_label_str_list(self, parameter_list, list_digits=None):
        if list_digits is None:
            list_digits = [2] * len(parameter_list)
        label_list = []
        for index in range(len(self.model_table['id'])):
            label_string = ''
            for param_index, parameter in zip(range(len(parameter_list)), parameter_list):
                label_string = (label_string + self.model_table_dict[parameter]['label'] + '=' +
                                '{:.{n}f}'.format(self.model_table_dict[parameter]['value'][index],
                                                  n=list_digits[param_index]) + ' ')
            label_list.append(label_string)
        return label_list





