"""
Gather all photometric tools for HST and JWST photometric observations
"""
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
import sep

from photometry_tools import data_access


class AnalysisTools(data_access.DataAccess):
    """
    Access class to organize data structure of HST, NIRCAM and MIRI imaging data
    """
    def __init__(self, **kwargs):
        """

        """
        super().__init__(**kwargs)

    def circular_flux_aperture_from_cutouts(self, cutout_dict, pos, aperture_rad=None, recenter=False, recenter_rad=0.2):

        # if aperture_rad is None we use the 50% encircled energy of a point spread function in each band
        if aperture_rad is None:
            aperture_rad = []
            for band in cutout_dict['band_list']:
                if band in self.hst_bands:
                    aperture_rad.append(self.hst_encircle_apertures_uvis2_arcsec[band]['ee50'])
                if band in self.nircam_bands:
                    aperture_rad.append(self.nircam_encircle_apertures_arcsec[band]['ee50'])
                if band in self.miri_bands:
                    aperture_rad.append(self.miri_encircle_apertures_arcsec[band]['ee50'])

        # for a fixed aperture
        if isinstance(aperture_rad, float):
            aperture_rad = [aperture_rad] * len(cutout_dict['band_list'])

        # check if aperture list is confirm
        if len(aperture_rad) != len(cutout_dict['band_list']):
            raise KeyError('aperture_rad_list must have the same length as the provided band list!')

        for band in cutout_dict['band_list']:
            print(aperture_rad)
            print(band)
        exit()
    def flux_from_circ_aperture(self, data, data_err, wcs, pos, aperture_rad, recenter=False, recenter_rad=0.2):

        if recenter:
            pos = self.re_center_peak(data=data, data_err=data_err, wcs=wcs, pos=pos, plotting=True)



    @staticmethod
    def sep_peak_detect(data, err, pixel_coodinates, pix_radius):
        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        objects = sep.extract(data, 1.0, err=err)
        #print('objects ', objects)
        if len(objects) == 0:
            #print('nooo')
            return pixel_coodinates
        else:
            x_cords_sources = objects['x']
            y_cords_sources = objects['y']
            objects_in_search_radius = np.sqrt((x_cords_sources - pixel_coodinates[0])**2 +
                                               (y_cords_sources - pixel_coodinates[1])**2) < pix_radius
            if np.sum(objects_in_search_radius) == 0:
                #print('the object detected was not in the radius')
                return pixel_coodinates
            elif np.sum(objects_in_search_radius) == 1:
                #print('only one object in radius')
                return x_cords_sources[objects_in_search_radius], y_cords_sources[objects_in_search_radius]
            else:
                #print('get brightest object')
                peak = objects['peak']
                max_peak_in_rad = np.max(peak[objects_in_search_radius])
                #print('max_peak_in_rad ', peak == max_peak_in_rad)
                return (x_cords_sources[objects_in_search_radius*(peak == max_peak_in_rad)],
                        y_cords_sources[objects_in_search_radius*(peak == max_peak_in_rad)])

    def re_center_peak(self, data, data_err, wcs, pos, cutout_size, plotting=False):

        bkg = sep.Background(np.array(data, dtype=float))

        # get radius in pixel scale
        pix_radius = (wcs.world_to_pixel(pos)[0] -
                      wcs.world_to_pixel(SkyCoord(ra=pos.ra+cutout_size*u.arcsec, dec=pos.dec))[0])
        # get the coordinates in pixel scale
        pixel_coodinates = wcs.world_to_pixel(pos)
        if plotting:
            plt.imshow(data)
            plt.scatter(pixel_coodinates[0][0], pixel_coodinates[1][0])
        # re calculate peak
        data = data-bkg
        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        data_err = np.array(data_err.byteswap().newbyteorder(), dtype=float)

        pixel_coodinates = self.sep_peak_detect(data=data, err=data_err,
                                                           pixel_coodinates=pixel_coodinates,
                                                           pix_radius=pix_radius)
        if plotting:
            plt.scatter(pixel_coodinates[0][0], pixel_coodinates[1][0])
            plt.show()
        position = wcs.pixel_to_world(pixel_coodinates[0][0], pixel_coodinates[1][0])
        return position

    @staticmethod
    def extract_flux_from_circ_aperture(data, wcs, bkg, position, aperture_rad, data_err=None):
        # get radius in pixel scale
        pix_radius = (wcs.world_to_pixel(position)[0] -
                      wcs.world_to_pixel(SkyCoord(ra=position.ra+aperture_rad*u.arcsec, dec=position.dec))[0])
        # get the coordinates in pixel scale
        pixel_coords = wcs.world_to_pixel(position)

        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        if data_err is None:
            bkg_rms = bkg.rms()
            data_err = np.array(bkg_rms.byteswap().newbyteorder(), dtype=float)
        else:
            data_err = np.array(data_err.byteswap().newbyteorder(), dtype=float)

        flux, flux_err, flag = sep.sum_circle(data=data - bkg, x=pixel_coords[0], y=pixel_coords[1],
                                              r=float(pix_radius[0]), err=data_err)

        return flux, flux_err


