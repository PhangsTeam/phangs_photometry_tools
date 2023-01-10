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
from astropy.table import QTable, vstack

import zfit
import time


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

    @staticmethod
    def get_sep_init_src_guess(data_sub, rms, psf, snr=3.0):
        """

        Parameters
        ----------
        data_sub : ``numpy.ndarray``
            background subtracted image data
        rms : float or``numpy.ndarray``
            RMS of the data
        psf : ``numpy.ndarray``
        snr : float

        Returns
        -------
        dict of initial guesses in pixel scale
        position x & y minor and major axis a & b and rotation angle theta
        """

        # To get a better detection of crowded sources and blended sources we significantly increase the contrast
        # We also use a minimum area of 2 because we want to be able to select point sources
        # We use the convolved mode and provide a psf in order to get a correct source size estimate.
        sep_table = sep.extract(data=data_sub, thresh=snr, err=rms, filter_kernel=psf, minarea=2, filter_type='conv',
                                deblend_cont=0.00001)

        sep_source_dict = {
            'x': sep_table['x'],
            'y': sep_table['y'],
            'a': sep_table['a'],
            'b': sep_table['b'],
            'theta': sep_table['theta']}

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
        fit_result = fmodel.fit(img, x=x_grid, y=y_grid, params=params, weights=1/img_err, method='least_squares')
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
        params = helper_func.set_mixt_model_params(fmodel=fmodel, init_pos=init_pos, param_lim=param_lim, img_mean=img_mean,
                                                  img_std=img_std, img_max=img_max, mask_gauss=mask_gauss, running_prefix='g_')
        print(params)

        fit_result = fmodel.fit(img, x=x_grid, y=y_grid, params=params, weights=1/img_err, method='least_squares')

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
                    fit_result.params['g_%i_sig_x' %index].value *
                    fit_result.params['g_%i_sig_y' %index].value)

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

            objects_in_ell = helper_func.check_point_inside_ellipse(x_ell=x_sources, y_ell=y_sources, a_ell=6*a_sources,
                                                                    b_ell=6*b_sources, theta_ell=angle_sources,
                                                                    x_p=object_table_residuals['ra'],
                                                                    y_p=object_table_residuals['dec'])
            # find elements where a
            if sum(objects_in_ell) > 1:
                mask_delete_in_object_table[index] = True

            if fit_results.params['g_%i_amp' % index].value < 3*rms:
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
        new_sep_source_dict = self.get_sep_init_src_guess(data_sub=data_sub - fit_results.best_fit, rms=new_bkg.globalrms, psf=psf)


        fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(8, 8))

        ax[0].imshow(data_sub, origin='lower')
        ax[1].imshow(fit_results.best_fit, origin='lower')
        ax[2].imshow(data_sub - fit_results.best_fit, origin='lower')

        # print(sep_source_dict)

        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                width=sep_source_dict['a'][i]*3,
                height=sep_source_dict['b'][i]*3,
                angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[0].add_artist(e)


        for i in range(len(new_sep_source_dict['x'])):
            e = Ellipse(xy=(new_sep_source_dict['x'][i], new_sep_source_dict['y'][i]),
                width=new_sep_source_dict['a'][i]*3,
                height=new_sep_source_dict['b'][i]*3,
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

    def zfit_pnt_src_param_dict_from_src_det(self, x0, y0, a, b, max_data_val, pos_var_fact=3, max_data_fact=1e3,
                                             starting_index=0):
        param_restrict_dict_pnt = {'n_src_pnt': len(x0)}
        for index in range(len(x0)):
            max_ext = np.max([a[index], b[index]])
            param_restrict_dict_pnt.update({'pnt_%i' % (index+starting_index): {
                'fix': False,
                'init_amp': max_data_val*5, 'lower_amp': 0, 'upper_amp': max_data_val*max_data_fact,
                'init_x0': x0[index], 'lower_x0': x0[index]-max_ext*pos_var_fact,
                'upper_x0': x0[index]+max_ext*pos_var_fact,
                'init_y0': y0[index], 'lower_y0': y0[index]-max_ext*pos_var_fact,
                'upper_y0': y0[index]+max_ext*pos_var_fact
            }})
        return param_restrict_dict_pnt

    def zfit_pnt_src_param_dict_from_fit(self, amp, x0, y0, amp_err, x0_err, y0_err, starting_index=0):

        param_restrict_dict_pnt = {'n_src_pnt': len(x0)}

        for index in range(len(x0)):
            param_restrict_dict_pnt.update({'pnt_%i' % (index+starting_index): {
                'fix': True,
                'amp': amp[index],
                'amp_err': amp_err[index],
                'x0': x0[index],
                'x0_err': x0_err[index],
                'y0': y0[index],
                'y0_err': y0_err[index]
            }})

        return param_restrict_dict_pnt

    def zfit_gauss_src_param_dict(self, amp, x0, y0, sig_x, sig_y, theta,
                                  amp_lim=(0, 1), pos_var=0.5, sig_var=(0.5, 1.5), theta_var=(-np.pi/2, np.pi/2),
                                  starting_index=0):
        param_restrict_dict_gauss = {'n_src_gauss': len(x0)}
        for index in range(len(x0)):
            param_restrict_dict_gauss.update({'gauss_%i' % (index+starting_index): {
                'fix': False,
                'init_amp': amp[index], 'lower_amp': amp_lim[0], 'upper_amp': amp_lim[1],
                'init_x0': x0[index], 'lower_x0': x0[index]-pos_var, 'upper_x0': x0[index]+pos_var,
                'init_y0': y0[index], 'lower_y0': y0[index]-pos_var, 'upper_y0': y0[index]+pos_var,
                'init_sig_x': sig_x[index], 'lower_sig_x': sig_var[0], 'upper_sig_x': sig_var[1],
                'init_sig_y': sig_y[index], 'lower_sig_y': sig_var[0], 'upper_sig_y': sig_var[1],
                'init_theta': theta[index], 'lower_theta': theta_var[0], 'upper_theta': theta_var[1]
            }})
        return param_restrict_dict_gauss

    def zfit_gauss_src_param_dict_from_fit(self, amp, x0, y0, sig_x, sig_y, theta,
                                           amp_err, x0_err, y0_err, sig_x_err, sig_y_err, theta_err,
                                           starting_index=0):
        param_restrict_dict_gauss = {'n_src_gauss': len(x0)}
        for index in range(len(x0)):
            param_restrict_dict_gauss.update({'gauss_%i' % (index+starting_index): {
                'fix': True,
                'amp': amp[index], 'amp_err': amp_err[index],
                'x0': x0[index], 'x0_err': x0_err[index],
                'y0': y0[index], 'y0_err': y0_err[index],
                'sig_x': sig_x[index], 'sig_x_err': sig_x_err[index],
                'sig_y': sig_y[index], 'sig_y_err': sig_y_err[index],
                'theta': theta[index], 'theta_err': theta_err[index]
            }})
        return param_restrict_dict_gauss

    @staticmethod
    def get_zfit_params(minimization_result, hesse_result, zfit_param_restrict_dict_pnt,
                        zfit_param_restrict_dict_gauss):
        n_src_pnt = zfit_param_restrict_dict_pnt['n_src_pnt']
        n_src_gauss = zfit_param_restrict_dict_gauss['n_src_gauss']

        param_value_array = np.array(minimization_result.values)
        error_values = list(hesse_result.values())
        # print('error_values ', error_values)

        amp_pnt = np.zeros(n_src_pnt)
        amp_pnt_err = np.zeros(n_src_pnt)
        x0_pnt = np.zeros(n_src_pnt)
        x0_pnt_err = np.zeros(n_src_pnt)
        y0_pnt = np.zeros(n_src_pnt)
        y0_pnt_err = np.zeros(n_src_pnt)

        amp_gauss = np.zeros(n_src_gauss)
        amp_gauss_err = np.zeros(n_src_gauss)
        x0_gauss = np.zeros(n_src_gauss)
        x0_gauss_err = np.zeros(n_src_gauss)
        y0_gauss = np.zeros(n_src_gauss)
        y0_gauss_err = np.zeros(n_src_gauss)
        sig_x_gauss = np.zeros(n_src_gauss)
        sig_x_gauss_err = np.zeros(n_src_gauss)
        sig_y_gauss = np.zeros(n_src_gauss)
        sig_y_gauss_err = np.zeros(n_src_gauss)
        theta_gauss = np.zeros(n_src_gauss)
        theta_gauss_err = np.zeros(n_src_gauss)

        # get point source parameters
        n_free_param_pnt = 0
        for index in range(n_src_pnt):
            if zfit_param_restrict_dict_pnt['pnt_%i' % index]['fix']:
                amp_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['amp']
                amp_pnt_err[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['amp_err']
                x0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['x0']
                x0_pnt_err[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['x0_err']
                y0_pnt[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['y0']
                y0_pnt_err[index] = zfit_param_restrict_dict_pnt['pnt_%i' % index]['y0_err']
            else:
                amp_pnt[index] = param_value_array[0 + n_free_param_pnt*3]
                amp_pnt_err[index] = error_values[0 + n_free_param_pnt*3]['error']
                x0_pnt[index] = param_value_array[1 + n_free_param_pnt*3]
                x0_pnt_err[index] = error_values[1 + n_free_param_pnt*3]['error']
                y0_pnt[index] = param_value_array[2 + n_free_param_pnt*3]
                y0_pnt_err[index] = error_values[2 + n_free_param_pnt*3]['error']
                n_free_param_pnt += 1

        n_free_param_gauss = 0
        # get gauss source parameters
        for index in range(n_src_gauss):
            if zfit_param_restrict_dict_gauss['gauss_%i' % index]['fix']:
                amp_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['amp']
                amp_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['amp_err']
                x0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['x0']
                x0_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['x0_err']
                y0_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['y0']
                y0_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['y0_err']
                sig_x_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_x']
                sig_x_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_x_err']
                sig_y_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_y']
                sig_y_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['sig_y_err']
                theta_gauss[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['theta']
                theta_gauss_err[index] = zfit_param_restrict_dict_gauss['gauss_%i' % index]['theta_err']
            else:
                amp_gauss[index] = param_value_array[0 + n_free_param_gauss*6 + 3*n_free_param_pnt]
                amp_gauss_err[index] = error_values[0 + n_free_param_gauss*6 + 3*n_free_param_pnt]['error']
                x0_gauss[index] = param_value_array[1 + n_free_param_gauss*6 + 3*n_free_param_pnt]
                x0_gauss_err[index] = error_values[1 + n_free_param_gauss*6 + 3*n_free_param_pnt]['error']
                y0_gauss[index] = param_value_array[2 + n_free_param_gauss*6 + 3*n_free_param_pnt]
                y0_gauss_err[index] = error_values[2 + n_free_param_gauss*6 + 3*n_free_param_pnt]['error']
                sig_x_gauss[index] = param_value_array[3 + n_free_param_gauss*6 + 3*n_free_param_pnt]
                sig_x_gauss_err[index] = error_values[3 + n_free_param_gauss*6 + 3*n_free_param_pnt]['error']
                sig_y_gauss[index] = param_value_array[4 + n_free_param_gauss*6 + 3*n_free_param_pnt]
                sig_y_gauss_err[index] = error_values[4 + n_free_param_gauss*6 + 3*n_free_param_pnt]['error']
                theta_gauss[index] = param_value_array[5 + n_free_param_gauss*6 + 3*n_free_param_pnt]
                theta_gauss_err[index] = error_values[5 + n_free_param_gauss*6 + 3*n_free_param_pnt]['error']
                n_free_param_gauss += 1

        # point source parameters
        param_dict_pnt = {
            'amp_pnt': amp_pnt,
            'amp_pnt_err': amp_pnt_err,
            'x0_pnt': x0_pnt,
            'x0_pnt_err': x0_pnt_err,
            'y0_pnt': y0_pnt,
            'y0_pnt_err': y0_pnt_err
        }

        # gaussian parameters
        param_dict_gauss = {
            'amp_gauss': amp_gauss,
            'amp_gauss_err': amp_gauss_err,
            'x0_gauss': x0_gauss,
            'x0_gauss_err': x0_gauss_err,
            'y0_gauss': y0_gauss,
            'y0_gauss_err': y0_gauss_err,
            'sig_x_gauss': sig_x_gauss,
            'sig_x_gauss_err': sig_x_gauss_err,
            'sig_y_gauss': sig_y_gauss,
            'sig_y_gauss_err': sig_y_gauss_err,
            'theta_gauss': theta_gauss,
            'theta_gauss_err': theta_gauss_err
        }

        return param_dict_pnt, param_dict_gauss

    def assemble_zfit_model(self, param_dict_pnt_pix_scale, param_dict_gauss_pix_scale, band):

        best_fit = np.zeros(self.current_img_data.shape)
        best_fit_model = np.zeros(self.current_img_data.shape)

        for index in range(0, len(param_dict_pnt_pix_scale['amp_pnt'])):
            best_fit += getattr(self, 'pnt_src_conv_%s' % band)(x=self.current_img_x_grid, y=self.current_img_y_grid,
                                                                amp=param_dict_pnt_pix_scale['amp_pnt'][index],
                                                                x0=param_dict_pnt_pix_scale['x0_pnt'][index],
                                                                y0=param_dict_pnt_pix_scale['y0_pnt'][index])
            best_fit_model += self.pnt_src(x=self.current_img_x_grid, y=self.current_img_y_grid,
                                           amp=param_dict_pnt_pix_scale['amp_pnt'][index], x0=param_dict_pnt_pix_scale['x0_pnt'][index],
                                           y0=param_dict_pnt_pix_scale['y0_pnt'][index])

        for index in range(0, len(param_dict_gauss_pix_scale['amp_gauss'])):
            best_fit += getattr(self, 'gauss2d_rot_conv_%s' % band)(x=self.current_img_x_grid,
                                                                    y=self.current_img_y_grid,
                                                                    amp=param_dict_gauss_pix_scale['amp_gauss'][index],
                                                                    x0=param_dict_gauss_pix_scale['x0_gauss'][index],
                                                                    y0=param_dict_gauss_pix_scale['y0_gauss'][index],
                                                                    sig_x=param_dict_gauss_pix_scale['sig_x_gauss'][index],
                                                                    sig_y=param_dict_gauss_pix_scale['sig_y_gauss'][index],
                                                                    theta=param_dict_gauss_pix_scale['theta_gauss'][index])
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
            flux[index] = np.sum(getattr(self, 'pnt_src_conv_%s' % band)(x=self.current_img_x_grid, y=self.current_img_y_grid,
                                                                amp=param_dict_pnt_pix_scale['amp_pnt'][index],
                                                                x0=param_dict_pnt_pix_scale['x0_pnt'][index],
                                                                y0=param_dict_pnt_pix_scale['y0_pnt'][index]))

            flux_err[index] = flux[index] / param_dict_pnt_pix_scale['amp_pnt'][index] * param_dict_pnt_pix_scale['amp_pnt_err'][index]

        return flux, flux_err

    def compute_zfit_gauss_src_flux(self, param_dict_gauss_pix_scale, band):

        flux = np.zeros(len(param_dict_gauss_pix_scale['amp_gauss']))
        flux_err = np.zeros(len(param_dict_gauss_pix_scale['amp_gauss']))

        for index in range(0, len(param_dict_gauss_pix_scale['amp_gauss'])):
            flux[index] = np.sum(getattr(self, 'gauss2d_rot_conv_%s' % band)(x=self.current_img_x_grid,
                                                                    y=self.current_img_y_grid,
                                                                    amp=param_dict_gauss_pix_scale['amp_gauss'][index],
                                                                    x0=param_dict_gauss_pix_scale['x0_gauss'][index],
                                                                    y0=param_dict_gauss_pix_scale['y0_gauss'][index],
                                                                    sig_x=param_dict_gauss_pix_scale['sig_x_gauss'][index],
                                                                    sig_y=param_dict_gauss_pix_scale['sig_y_gauss'][index],
                                                                    theta=param_dict_gauss_pix_scale['theta_gauss'][index]))
            analytical_flux = (param_dict_gauss_pix_scale['amp_gauss'][index] *
                               param_dict_gauss_pix_scale['sig_x_gauss'][index] *
                               param_dict_gauss_pix_scale['sig_y_gauss'][index] *
                               2 * np.pi)
            analytical_flux_err = (analytical_flux *
                                   np.sqrt((param_dict_gauss_pix_scale['amp_gauss_err'][index] /
                                            param_dict_gauss_pix_scale['amp_gauss'][index])**2 +
                                           (param_dict_gauss_pix_scale['sig_x_gauss_err'][index] /
                                            param_dict_gauss_pix_scale['sig_x_gauss'][index])**2 +
                                           (param_dict_gauss_pix_scale['sig_y_gauss_err'][index] /
                                            param_dict_gauss_pix_scale['sig_y_gauss'][index])**2))
            flux_err[index] = flux[index] / analytical_flux * analytical_flux_err

        return flux, flux_err

    @staticmethod
    def eval_zfit_residuals(pnt_src_x, pnt_src_y, new_source_table, dist_to_ellipse, psf_pix_rad):

        #mask_delete_in_object_table = np.zeros(len(object_table), dtype=bool)
        gauss_cand = np.zeros(len(pnt_src_x), dtype=bool)
        new_pnt_src_dict = {'x': [], 'y': [], 'a': [], 'b': []}

        for index in range(len(new_source_table['x'])):
            x_sources = new_source_table['x'][index]
            y_sources = new_source_table['y'][index]
            a_sources = new_source_table['a'][index]
            b_sources = new_source_table['b'][index]
            angle_sources = new_source_table['theta'][index]

            objects_in_ell = helper_func.check_point_inside_ellipse(x_ell=x_sources, y_ell=y_sources,
                                                                    a_ell=dist_to_ellipse*a_sources + psf_pix_rad,
                                                                    b_ell=dist_to_ellipse*b_sources + psf_pix_rad,
                                                                    theta_ell=angle_sources,
                                                                    x_p=pnt_src_x, y_p=pnt_src_y)
            # find elements where a
            if sum(objects_in_ell) >= 1:
                gauss_cand[objects_in_ell] = True
            else:
                new_pnt_src_dict['x'].append(x_sources)
                new_pnt_src_dict['y'].append(y_sources)
                new_pnt_src_dict['a'].append(a_sources)
                new_pnt_src_dict['b'].append(b_sources)

        return gauss_cand, new_pnt_src_dict

    def generate_new_init_zfit_model(self, data_sub, psf, fit_result_dict, band, wcs):
        # if additional iterations will take place:
        # find sources in residuals
        new_bkg = sep.Background(np.array((data_sub - fit_result_dict['best_fit']), dtype=float))
        new_sep_source_dict = self.get_sep_init_src_guess(data_sub=data_sub - fit_result_dict['best_fit'],
                                                          rms=new_bkg.globalrms, psf=psf)
        print(len(new_sep_source_dict))

        psf_pix_rad = helper_func.transform_world2pix_scale(self.nircam_encircle_apertures_arcsec[band]['ee50'],
                                                            wcs=wcs)
        # evaluate residuals
        gauss_cand, new_pnt_src_dict = self.eval_zfit_residuals(pnt_src_x=fit_result_dict['param_dict_pnt_pix_scale']['x0_pnt'],
                                                                pnt_src_y=fit_result_dict['param_dict_pnt_pix_scale']['y0_pnt'],
                                                                new_source_table=new_sep_source_dict,
                                                                dist_to_ellipse=10, psf_pix_rad=psf_pix_rad)
        print('gauss_cand ', gauss_cand)
        print('new_pnt_src_dict ', new_pnt_src_dict)

        # now add point sources
        # but only add those point sources which no gaussian candidates
        zfit_param_restrict_dict_pnt =\
            self.zfit_pnt_src_param_dict_from_fit(amp=fit_result_dict['param_dict_pnt_pix_scale']['amp_pnt'][~gauss_cand],
                                                  x0=fit_result_dict['param_dict_pnt_pix_scale']['x0_pnt'][~gauss_cand],
                                                  y0=fit_result_dict['param_dict_pnt_pix_scale']['y0_pnt'][~gauss_cand],
                                                  amp_err=fit_result_dict['param_dict_pnt_pix_scale']['amp_pnt_err'][~gauss_cand],
                                                  x0_err=fit_result_dict['param_dict_pnt_pix_scale']['x0_pnt_err'][~gauss_cand],
                                                  y0_err=fit_result_dict['param_dict_pnt_pix_scale']['y0_pnt_err'][~gauss_cand])
        zfit_param_restrict_dict_gauss =\
            self.zfit_gauss_src_param_dict_from_fit(amp=fit_result_dict['param_dict_gauss_pix_scale']['amp_gauss'],
                                                    x0=fit_result_dict['param_dict_gauss_pix_scale']['x0_gauss'],
                                                    y0=fit_result_dict['param_dict_gauss_pix_scale']['y0_gauss'],
                                                    sig_x=fit_result_dict['param_dict_gauss_pix_scale']['sig_x_gauss_err'],
                                                    sig_y=fit_result_dict['param_dict_gauss_pix_scale']['sig_y_gauss_err'],
                                                    theta=fit_result_dict['param_dict_gauss_pix_scale']['theta_gauss_err'],
                                                    amp_err=fit_result_dict['param_dict_gauss_pix_scale']['amp_gauss_err'],
                                                    x0_err=fit_result_dict['param_dict_gauss_pix_scale']['x0_gauss_err'],
                                                    y0_err=fit_result_dict['param_dict_gauss_pix_scale']['y0_gauss_err'],
                                                    sig_x_err=fit_result_dict['param_dict_gauss_pix_scale']['sig_x_gauss_err'],
                                                    sig_y_err=fit_result_dict['param_dict_gauss_pix_scale']['sig_y_gauss_err'],
                                                    theta_err=fit_result_dict['param_dict_gauss_pix_scale']['theta_gauss_err'])

        print('zfit_param_restrict_dict_pnt ', zfit_param_restrict_dict_pnt)
        # add new point sources

        new_zfit_param_restrict_dict_pnt = self.zfit_pnt_src_param_dict_from_src_det(x0=new_pnt_src_dict['x'],
                                                                            y0=new_pnt_src_dict['y'],
                                                                            a=new_pnt_src_dict['a'],
                                                                            b=new_pnt_src_dict['b'],
                                                                            max_data_val=np.max(data_sub),
                                                                            starting_index=zfit_param_restrict_dict_pnt['n_src_pnt'])

        n_src_pnt = zfit_param_restrict_dict_pnt['n_src_pnt'] + new_zfit_param_restrict_dict_pnt['n_src_pnt']
        zfit_param_restrict_dict_pnt.update(new_zfit_param_restrict_dict_pnt)
        zfit_param_restrict_dict_pnt['n_src_pnt'] = n_src_pnt
        print('zfit_param_restrict_dict_pnt ', zfit_param_restrict_dict_pnt)

        print('zfit_param_restrict_dict_gauss ', zfit_param_restrict_dict_gauss)
        new_zfit_param_restrict_dict_gauss = self.zfit_gauss_src_param_dict(amp=fit_result_dict['param_dict_pnt_pix_scale']['amp_pnt'][gauss_cand],
                                                               x0=fit_result_dict['param_dict_pnt_pix_scale']['x0_pnt'][gauss_cand],
                                                               y0=fit_result_dict['param_dict_pnt_pix_scale']['y0_pnt'][gauss_cand],
                                                               sig_x=np.ones(sum(gauss_cand)),
                                                               sig_y=np.ones(sum(gauss_cand)),
                                                               theta=np.zeros(sum(gauss_cand)),
                                                               amp_lim=(0, np.max(data_sub) * 1e3),
                                                               pos_var=10,
                                                               sig_var=(0.5, 5),
                                                               theta_var=(-np.pi/2, np.pi/2))
        n_src_gauss = zfit_param_restrict_dict_gauss['n_src_gauss'] + new_zfit_param_restrict_dict_gauss['n_src_gauss']
        zfit_param_restrict_dict_gauss.update(new_zfit_param_restrict_dict_gauss)
        zfit_param_restrict_dict_gauss['n_src_gauss'] = n_src_gauss
        print('zfit_param_restrict_dict_gauss ', zfit_param_restrict_dict_gauss)

        return zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss

    def transform_params_pix2world(self, param_dict_pnt_pix_scale, param_dict_gauss_pix_scale, wcs):

        position_pnt = wcs.pixel_to_world(param_dict_pnt_pix_scale['x0_pnt'], param_dict_pnt_pix_scale['y0_pnt'])
        x0_pnt_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_pnt_pix_scale['x0_pnt_err'],
                                                           wcs=wcs, dim=0, return_unit='arcsec')
        y0_pnt_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_pnt_pix_scale['y0_pnt_err'],
                                                           wcs=wcs, dim=1, return_unit='arcsec')

        position_gauss = wcs.pixel_to_world(param_dict_gauss_pix_scale['x0_gauss'],
                                            param_dict_gauss_pix_scale['y0_gauss'])
        x0_gauss_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['x0_gauss_err'],
                                                             wcs=wcs, dim=0, return_unit='arcsec')
        y0_gauss_err = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['y0_gauss_err'],
                                                             wcs=wcs, dim=1, return_unit='arcsec')

        sigma_x_gauss = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['sig_x_gauss'],
                                                              wcs=wcs, dim=0, return_unit='arcsec')
        sig_x_gauss_err = helper_func.transform_pix2world_scale(
            pixel_length=param_dict_gauss_pix_scale['sig_x_gauss_err'], wcs=wcs, dim=0, return_unit='arcsec')
        sigma_y_gauss = helper_func.transform_pix2world_scale(pixel_length=param_dict_gauss_pix_scale['sig_y_gauss'],
                                                              wcs=wcs, dim=1, return_unit='arcsec')
        sig_y_gauss_err = helper_func.transform_pix2world_scale(
            pixel_length=param_dict_gauss_pix_scale['sig_y_gauss_err'], wcs=wcs, dim=1, return_unit='arcsec')

        param_dict_pnt_wcs_scale = {'position_pnt': position_pnt,  'x0_pnt_err': x0_pnt_err, 'y0_pnt_err': y0_pnt_err}
        param_dict_gauss_wcs_scale = {
            'position_gauss': position_gauss,
            'sigma_x_gauss': sigma_x_gauss,
            'sigma_y_gauss': sigma_y_gauss,
            'x0_gauss_err': x0_gauss_err,
            'y0_gauss_err': y0_gauss_err,
            'sig_x_gauss_err': sig_x_gauss_err,
            'sig_y_gauss_err': sig_y_gauss_err,
        }

        print('param_dict_pnt_pix_scale[x0_pnt]', param_dict_pnt_pix_scale['x0_pnt'])
        print('param_dict_pnt_pix_scale[y0_pnt]', param_dict_pnt_pix_scale['y0_pnt'])
        print('param_dict_gauss_pix_scale[x0_gauss]', param_dict_gauss_pix_scale['x0_gauss'])
        print('param_dict_gauss_pix_scale[y0_gauss]', param_dict_gauss_pix_scale['y0_gauss'])

        print('position_pnt ', position_pnt)
        print('position_gauss ', position_gauss)
        print('sigma_x_gauss ', sigma_x_gauss)
        print('sigma_y_gauss ', sigma_y_gauss)

        return param_dict_pnt_wcs_scale, param_dict_gauss_wcs_scale

    def zfit_model2data(self, band, x_grid, y_grid, data, err, zfit_param_restrict_dict_pnt,
                        zfit_param_restrict_dict_gauss, index_iter, wcs):

        # init fit model
        self.init_zfit_models(band=band, img_x_grid=x_grid, img_y_gird=y_grid, img_data=data, img_err=err,
                              n_pnt=zfit_param_restrict_dict_pnt['n_src_pnt'],
                              n_gauss=zfit_param_restrict_dict_gauss['n_src_gauss'])
        # create init zfit parameters
        params_pnt = self.create_zfit_pnt_param_list(zfit_param_restrict_dict_pnt=zfit_param_restrict_dict_pnt,
                                                     index_iter=index_iter)
        params_gauss = self.create_zfit_gauss_param_list(zfit_param_restrict_dict_gauss=zfit_param_restrict_dict_gauss,
                                                         index_iter=index_iter)

        print('params_pnt ', params_pnt)
        print('n_pnt ', zfit_param_restrict_dict_pnt['n_src_pnt'])
        print('params_gauss ', params_gauss)
        print('n_gauss ', zfit_param_restrict_dict_gauss['n_src_gauss'])

        # init loss model and fit
        loss = zfit.loss.SimpleLoss(self.squared_loss, params_pnt + params_gauss, errordef=1)
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

        print('zfit_param_restrict_dict_pnt ', zfit_param_restrict_dict_pnt)
        print('zfit_param_restrict_dict_gauss ', zfit_param_restrict_dict_gauss)
        # get fit parameters
        param_dict_pnt_pix_scale, param_dict_gauss_pix_scale = self.get_zfit_params(minimization_result=minimization_result,
                                                                hesse_result=hesse_result,
                                                                zfit_param_restrict_dict_pnt=zfit_param_restrict_dict_pnt,
                                                                zfit_param_restrict_dict_gauss=zfit_param_restrict_dict_gauss)
        print('param_dict_pnt_pix_scale ', param_dict_pnt_pix_scale)
        print('param_dict_gauss_pix_scale ', param_dict_gauss_pix_scale)

        # convert position parameters into WCS format
        param_dict_pnt_wcs_scale, param_dict_gauss_wcs_scale = \
            self.transform_params_pix2world(param_dict_pnt_pix_scale=param_dict_pnt_pix_scale,
                                            param_dict_gauss_pix_scale=param_dict_gauss_pix_scale, wcs=wcs)

        # get best fit
        best_fit, best_fit_model = self.assemble_zfit_model(param_dict_pnt_pix_scale=param_dict_pnt_pix_scale,
                                                            param_dict_gauss_pix_scale=param_dict_gauss_pix_scale, band=band)

        flux_pnt_src, flux_pnt_src_err = self.compute_zfit_pnt_src_flux(param_dict_pnt_pix_scale=param_dict_pnt_pix_scale, band=band)
        flux_gauss_src, flux_gauss_src_err = self.compute_zfit_gauss_src_flux(param_dict_gauss_pix_scale=param_dict_gauss_pix_scale,
                                                                              band=band)

        # dict to return
        fit_result_dict = {'param_dict_pnt_pix_scale': param_dict_pnt_pix_scale,
                           'param_dict_gauss_pix_scale': param_dict_gauss_pix_scale,
                           'param_dict_pnt_wcs_scale': param_dict_pnt_wcs_scale,
                           'param_dict_gauss_wcs_scale': param_dict_gauss_wcs_scale,
                           'flux_pnt_src': flux_pnt_src, 'flux_pnt_src_err': flux_pnt_src_err,
                           'flux_gauss_src': flux_gauss_src, 'flux_gauss_src_err': flux_gauss_src_err,
                           'best_fit': best_fit, 'best_fit_model': best_fit_model}

        return fit_result_dict

    def zfit_iter_de_blend(self, band, ra, dec, cutout_size, init_src_dict=None, n_iter=2,
                           pix_var=1.5):

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
            # create SEP object table
            sep_src_dict = self.get_sep_init_src_guess(data_sub=data_sub, rms=bkg.globalrms, psf=psf)
            # init all parameters
            zfit_param_restrict_dict_pnt = self.zfit_pnt_src_param_dict_from_src_det(x0=sep_src_dict['x'],
                                                                                     y0=sep_src_dict['y'],
                                                                                     a=sep_src_dict['a'],
                                                                                     b=sep_src_dict['b'],
                                                                                     max_data_val=np.max(data),
                                                                                     pos_var_fact=3, max_data_fact=1e3)
            zfit_param_restrict_dict_gauss = {'n_src_gauss': 0}
        else:
            # code access for src parameter access
            # positions init_src_dict[]

            

            zfit_param_restrict_dict_pnt = None
            zfit_param_restrict_dict_gauss = None

        for fit_iter_index in range(n_iter):
            fit_result_dict = self.zfit_model2data(band=band, x_grid=x_grid, y_grid=y_grid, data=data, err=err,
                                                   zfit_param_restrict_dict_pnt=zfit_param_restrict_dict_pnt,
                                                   zfit_param_restrict_dict_gauss=zfit_param_restrict_dict_gauss,
                                                   index_iter=fit_iter_index, wcs=wcs)

            zfit_param_restrict_dict_pnt, zfit_param_restrict_dict_gauss = \
                self.generate_new_init_zfit_model(data_sub=data_sub, psf=psf, fit_result_dict=fit_result_dict,
                                                  band=band, wcs=wcs)

        print('fit_result_dict ', fit_result_dict)

        return fit_result_dict


        # fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(8, 8))
        # ax[0].imshow(data_sub, origin='lower')
        # ax[1].imshow(fit_result_dict['best_fit'], origin='lower')
        # ax[2].imshow(fit_result_dict['best_fit_model'], origin='lower')
        # ax[3].imshow(data_sub - fit_result_dict['best_fit'], origin='lower')
        #
        # plt.show()
        #
        # exit()




        # get sources in residuals
        new_bkg = sep.Background(np.array((data_sub - best_fit), dtype=float))
        new_sep_source_dict = self.get_sep_init_src_guess(data_sub=data_sub - best_fit, rms=new_bkg.globalrms, psf=psf)
        # update source table
        ext_candidates, new_pnt_src_dict = self.update_pnt_src_table(src_x=param_dict['x0_list'],
                                                                         src_y=param_dict['y0_list'],
                                                                         new_source_table=new_sep_source_dict,
                                                                         dist_to_ellipse=3)
        print('ext_candidates ', ext_candidates)
        print('new_pnt_src_dict ', new_pnt_src_dict)

        # init parameters of new pnt sources
        new_param_restrict_dict_pnt = self.create_zfit_pnt_src_params(x=new_pnt_src_dict['x'],
                                                                      y=new_pnt_src_dict['y'],
                                                                      a=new_pnt_src_dict['a'],
                                                                      b=new_pnt_src_dict['b'],
                                                                      max_data_val=np.max(data),
                                                                      mask_pnt_src=np.ones(len(new_pnt_src_dict['x']),
                                                                                           dtype=bool))

        # get zfit params
        new_params_pnt, new_params_gauss = self.create_zfit_param_list(n_pnt=len(new_param_restrict_dict_pnt),
                                                                       param_restrict_dict_pnt=new_param_restrict_dict_pnt,
                                                                       starting_index_pnt=len(param_restrict_dict_pnt))
        print('new_params_pnt ', new_params_pnt)
        print('len(new_params_pnt) ', len(new_params_pnt))
        print('new_params_gauss ', new_params_gauss)
        print('len(new_params_gauss) ', len(new_params_gauss))

        # reinitialize fitting
        self.init_zfit_models(band=band, img_x_grid=x_grid, img_y_gird=y_grid, img_data=data, img_err=err,
                              n_pnt=len(param_restrict_dict_pnt) + len(new_param_restrict_dict_pnt), n_gauss=0)
        start = time.time()
        loss = zfit.loss.SimpleLoss(self.squared_loss, params_pnt + params_gauss + new_params_pnt + new_params_gauss, errordef=1)
        minimizer = zfit.minimize.Minuit()
        new_result = minimizer.minimize(loss)
        end = time.time()
        print('Time of fitting ', end - start)
        print('Is Fit valid? ', new_result.valid)
        print(new_result)
        start = time.time()
        new_param_hesse = new_result.hesse()
        end = time.time()
        print('Time to compute Hessian ', end - start)
        # get fit parameters
        new_param_dict = self.get_zfit_params(result=new_result, param_hesse=new_param_hesse)
        print('new_param_dict ', new_param_dict)
        new_best_fit, new_best_fit_model = self.assemble_zfit_model(param_dict=new_param_dict, band=band)

        new_new_bkg = sep.Background(np.array((data_sub - new_best_fit), dtype=float))
        new_new_sep_source_dict = self.get_sep_init_src_guess(data_sub=data_sub - new_best_fit,
                                                                 rms=new_new_bkg.globalrms, psf=psf)


        fig, ax = plt.subplots(ncols=4, nrows=2, figsize=(8, 8))

        ax[0, 0].imshow(data_sub, origin='lower')
        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                width=sep_source_dict['a'][i]*3,
                height=sep_source_dict['b'][i]*3,
                angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[0, 0].add_artist(e)
            ax[0, 0].text(sep_source_dict['x'][i], sep_source_dict['y'][i], '%i' % i)

        ax[0, 1].imshow(best_fit, origin='lower')
        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                width=sep_source_dict['a'][i]*3,
                height=sep_source_dict['b'][i]*3,
                angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[0, 1].add_artist(e)
            ax[0, 1].text(sep_source_dict['x'][i], sep_source_dict['y'][i], '%i' % i)
            ax[0, 1].scatter(param_dict['x0_list'][i], param_dict['y0_list'][i])

        ax[0, 2].imshow(best_fit_model, origin='lower')

        ax[0, 3].imshow(data_sub - best_fit, origin='lower')

        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                width=sep_source_dict['a'][i]*3,
                height=sep_source_dict['b'][i]*3,
                angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[0, 3].add_artist(e)
            ax[0, 3].scatter(param_dict['x0_list'][i], param_dict['y0_list'][i])

        for i in range(len(new_sep_source_dict['x'])):
            e = Ellipse(xy=(new_sep_source_dict['x'][i], new_sep_source_dict['y'][i]),
                width=new_sep_source_dict['a'][i]*3,
                height=new_sep_source_dict['b'][i]*3,
                angle=new_sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            e.set_linewidth(1)
            ax[0, 3].add_artist(e)


        ax[1, 0].imshow(data_sub, origin='lower')
        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                width=sep_source_dict['a'][i]*3,
                height=sep_source_dict['b'][i]*3,
                angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[1, 0].add_artist(e)
            ax[1, 0].text(sep_source_dict['x'][i], sep_source_dict['y'][i], '%i' % i)
        for j in range(len(new_sep_source_dict['x'])):
            e = Ellipse(xy=(new_sep_source_dict['x'][j], new_sep_source_dict['y'][j]),
                width=new_sep_source_dict['a'][j]*3,
                height=new_sep_source_dict['b'][j]*3,
                angle=new_sep_source_dict['theta'][j] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[1, 0].add_artist(e)
            ax[1, 0].text(new_sep_source_dict['x'][j], new_sep_source_dict['y'][j], '%i' % (j + i))

        ax[1, 1].imshow(new_best_fit, origin='lower')
        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                width=sep_source_dict['a'][i]*3,
                height=sep_source_dict['b'][i]*3,
                angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[1, 1].add_artist(e)
            ax[1, 1].text(sep_source_dict['x'][i], sep_source_dict['y'][i], '%i' % i)
        for j in range(len(new_sep_source_dict['x'])):
            e = Ellipse(xy=(new_sep_source_dict['x'][j], new_sep_source_dict['y'][j]),
                width=new_sep_source_dict['a'][j]*3,
                height=new_sep_source_dict['b'][j]*3,
                angle=new_sep_source_dict['theta'][j] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[1, 1].add_artist(e)
            ax[1, 1].text(new_sep_source_dict['x'][j], new_sep_source_dict['y'][j], '%i' % (j + i))

        ax[1, 2].imshow(new_best_fit_model, origin='lower')

        ax[1, 3].imshow(data_sub - new_best_fit, origin='lower')

        for i in range(len(sep_source_dict['x'])):
            e = Ellipse(xy=(sep_source_dict['x'][i], sep_source_dict['y'][i]),
                width=sep_source_dict['a'][i]*3,
                height=sep_source_dict['b'][i]*3,
                angle=sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[1, 3].add_artist(e)
            ax[1, 3].scatter(param_dict['x0_list'][i], param_dict['y0_list'][i])
        for i in range(len(new_sep_source_dict['x'])):
            e = Ellipse(xy=(new_sep_source_dict['x'][i], new_sep_source_dict['y'][i]),
                width=new_sep_source_dict['a'][i]*3,
                height=new_sep_source_dict['b'][i]*3,
                angle=new_sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('red')
            e.set_linewidth(1)
            ax[1, 3].add_artist(e)

        for i in range(len(new_new_sep_source_dict['x'])):
            e = Ellipse(xy=(new_new_sep_source_dict['x'][i], new_new_sep_source_dict['y'][i]),
                width=new_new_sep_source_dict['a'][i]*3,
                height=new_new_sep_source_dict['b'][i]*3,
                angle=new_new_sep_source_dict['theta'][i] * 180 / np.pi)
            e.set_facecolor('none')
            e.set_edgecolor('blue')
            e.set_linewidth(1)
            ax[1, 3].add_artist(e)


        # plt.show()
        plt.savefig('plot_output/test_zfit.png')

        exit()







