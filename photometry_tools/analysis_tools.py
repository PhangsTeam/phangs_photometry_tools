"""
Gather all photometric tools for HST and JWST photometric observations
"""
import numpy as np
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

    def flux_from_circ_apert(self, cutout_dict, ):
        print(cutout_dict.keys())
        print('hello')


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
            if sum(objects_in_search_radius) == 0:
                #print('the object detected was not in the radius')
                return pixel_coodinates
            elif sum(objects_in_search_radius) == 1:
                #print('only one object in radius')
                return (x_cords_sources[objects_in_search_radius], y_cords_sources[objects_in_search_radius])
            else:
                #print('get brightest object')
                peak = objects['peak']
                max_peak_in_rad = np.max(peak[objects_in_search_radius])
                #print('max_peak_in_rad ', peak == max_peak_in_rad)
                return (x_cords_sources[objects_in_search_radius*(peak == max_peak_in_rad)],
                        y_cords_sources[objects_in_search_radius*(peak == max_peak_in_rad)])

    @staticmethod
    def re_center_peak(file_name, hdu_number, cutout_pos, cutout_size,
                       file_err_name=None, hdu_err_number=None, flux_unit='mJy', obs='hst', plotting=False):

        if obs == 'hst':
            flux_cutout, flux_err_cutout = VisualizeHelper.get_hst_cutout_from_file(file_name=file_name,
                                                                                    hdu_number=hdu_number,
                                                                                    cutout_pos=cutout_pos,
                                                                                    cutout_size=cutout_size,
                                                                                    file_err_name=file_err_name,
                                                                                    hdu_err_number=hdu_err_number,
                                                                                    rescaling=flux_unit)
        elif obs == 'nircam':
            flux_cutout, flux_err_cutout = VisualizeHelper.get_jwst_cutout_from_file(file_name=file_name,
                                                                                     hdu_number=hdu_number,
                                                                                     cutout_pos=cutout_pos,
                                                                                     cutout_size=cutout_size,
                                                                                     hdu_err_number=hdu_err_number,
                                                                                     rescaling=flux_unit)
        elif obs == 'miri':
            flux_cutout = VisualizeHelper.get_jwst_cutout_from_file(file_name=file_name, hdu_number=hdu_number,
                                                                    cutout_pos=cutout_pos, cutout_size=cutout_size,
                                                                    rescaling=flux_unit)
            flux_err_cutout = VisualizeHelper.get_jwst_cutout_from_file(file_name=file_err_name,
                                                                        hdu_number=hdu_err_number,
                                                                        cutout_pos=cutout_pos,
                                                                        cutout_size=cutout_size,
                                                                        rescaling=flux_unit)
        else:
            raise KeyError('obs must be either hst, nircam or miri')

        bkg = sep.Background(np.array(flux_cutout.data, dtype=float))

        # get radius in pixel scale
        pix_radius = (flux_cutout.wcs.world_to_pixel(cutout_pos)[0] -
                      flux_cutout.wcs.world_to_pixel(SkyCoord(ra=cutout_pos.ra+cutout_size*u.arcsec, dec=cutout_pos.dec))[0])
        # get the coordinates in pixel scale
        pixel_coodinates = flux_cutout.wcs.world_to_pixel(cutout_pos)
        if plotting:
            plt.imshow(flux_cutout.data)
            plt.scatter(pixel_coodinates[0][0], pixel_coodinates[1][0])
        # re calculate peak
        data = flux_cutout.data-bkg
        data = np.array(data.byteswap().newbyteorder(), dtype=float)
        data_err = flux_err_cutout.data
        data_err = np.array(data_err.byteswap().newbyteorder(), dtype=float)

        pixel_coodinates = VisualizeHelper.sep_peak_detect(data=data, err=data_err,
                                                           pixel_coodinates=pixel_coodinates,
                                                           pix_radius=pix_radius)
        if plotting:
            plt.scatter(pixel_coodinates[0][0], pixel_coodinates[1][0])
            plt.show()
        position = flux_cutout.wcs.pixel_to_world(pixel_coodinates[0][0], pixel_coodinates[1][0])
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


