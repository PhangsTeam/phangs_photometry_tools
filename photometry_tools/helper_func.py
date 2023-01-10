"""
Collection of various helper functions. This file has no general purpose.
The idea of each function should be precised in each doc string .
"""
import os
from pathlib import Path
import warnings

from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord

from lmfit import Model
import numpy as np


def identify_file_in_folder(folder_path, str_in_file_name):
    """
    Identify a file inside a folder that contains a specific string.

    Parameters
    ----------
    folder_path : Path or str
    str_in_file_name : str

    Returns
    -------
    file_name : Path
    """

    if isinstance(folder_path, str):
        folder_path = Path(folder_path)

    identified_files = list(filter(lambda x: str_in_file_name in x, os.listdir(folder_path)))
    if not identified_files:
        raise FileNotFoundError('The data file containing the string %s does not exist.' % str_in_file_name)
    elif len(identified_files) > 1:
        raise FileExistsError('There are more than one data files containing the string %s .' % str_in_file_name)
    else:
        return folder_path / str(identified_files[0])


def load_img(file_name, hdu_number=0):
    """function to open hdu using astropy.

    Parameters
    ----------
    file_name : str or Path
        file name to open
    hdu_number : int or str
        hdu number which should be opened. can be also a string such as 'SCI' for JWST images

    Returns
    -------
    array-like, astropy.header and astropy.wcs object
    """
    # get hdu
    hdu = fits.open(file_name)
    # get header
    header = hdu[hdu_number].header
    # get WCS
    wcs = WCS(header)
    # update the header
    header.update(wcs.to_header())
    # reload the WCS and header
    header = hdu[hdu_number].header
    wcs = WCS(header)
    # load data
    data = hdu[hdu_number].data
    return data, header, wcs


def get_img_cutout(img, wcs, coord, cutout_size):
    """function to cut out a region of a larger image with an WCS.
    Parameters
    ----------
    img : ndarray
        (Ny, Nx) image
    wcs : astropy.wcs.WCS()
        astropy world coordinate system object describing the parameter image
    coord : astropy.coordinates.SkyCoord
        astropy coordinate object to point to the selected area which to cutout
    cutout_size : float or tuple
        Units in arcsec. Cutout size of a box cutout. If float it will be used for both box length.

    Returns
    -------
    cutout : astropy.nddata.Cutout2D object
        cutout object of the initial image
    """

    if isinstance(cutout_size, tuple):
        size = cutout_size * u.arcsec
    elif isinstance(cutout_size, float) | isinstance(cutout_size, int):
        size = (cutout_size, cutout_size) * u.arcsec
    else:
        raise KeyError('cutout_size must be float or tuple')

    # check if cutout is inside the image
    pix_pos = wcs.world_to_pixel(coord)
    if (pix_pos[0] > 0) & (pix_pos[0] < img.shape[1]) & (pix_pos[1] > 0) & (pix_pos[1] < img.shape[0]):
        return Cutout2D(data=img, position=coord, size=size, wcs=wcs)
    else:
        warnings.warn("The selected cutout is outside the original dataset. The data and WCS will be None",
                      DeprecationWarning)
        cut_out = type('', (), {})()
        cut_out.data = None
        cut_out.wcs = None
        return cut_out


def download_file(file_path, url, unpack=False, reload=False):
    """

    Parameters
    ----------
    file_path : str or ``pathlib.Path``
    url : str
    unpack : bool
        In case the downloaded file is zipped, this function can unpack it and remove the downloaded file,
        leaving only the extracted file
    reload : bool
        If the file is corrupted, this removes the file and reloads it

    Returns
    -------

    """
    if reload:
        # if reload file the file will be removed to re download it
        os.remove(file_path)
    # check if file already exists
    if os.path.isfile(file_path):
        print(file_path, 'already exists')
        return True
    else:
        from urllib3 import PoolManager
        # download file
        http = PoolManager()
        r = http.request('GET', url, preload_content=False)

        if unpack:
            with open(file_path.with_suffix(".gz"), 'wb') as out:
                while True:
                    data = r.read()
                    if not data:
                        break
                    out.write(data)
            r.release_conn()
            # uncompress file
            from gzip import GzipFile
            # read compressed file
            compressed_file = GzipFile(file_path.with_suffix(".gz"), 'rb')
            s = compressed_file.read()
            compressed_file.close()
            # save compressed file
            uncompressed_file = open(file_path, 'wb')
            uncompressed_file.write(s)
            uncompressed_file.close()
            # delete compressed file
            os.remove(file_path.with_suffix(".gz"))
        else:
            with open(file_path, 'wb') as out:
                while True:
                    data = r.read()
                    if not data:
                        break
                    out.write(data)
            r.release_conn()


def null_func2d(x, y, a):
    return x*a + y*a


def compose_n_func_model(func, n, independent_vars=None, running_prefix='g_'):

    if n == 1:
        return Model(func, prefix='%s0_' % running_prefix, independent_vars=independent_vars)
    else:
        fmodel = (Model(func, prefix='%s0_' % running_prefix, independent_vars=independent_vars) +
                  Model(func, prefix='%s1_' % running_prefix, independent_vars=independent_vars))
        for double_index in range(0, n-1, 2):
            if double_index != 0:
                fmodel += (Model(func, prefix='%s%i_' % (running_prefix, double_index+0), independent_vars=independent_vars) +
                           Model(func, prefix='%s%i_' % (running_prefix, double_index+1), independent_vars=independent_vars))
        if n % 2:
            fmodel += (Model(func, prefix='%s%i_' % (running_prefix, n-1), independent_vars=independent_vars) +
                       Model(func, prefix='null_func_', independent_vars=independent_vars))

        return fmodel


def compose_mixed_func_model(func1, func2, mask_func1, independent_vars=None, running_prefix='g_'):

    if len(mask_func1) == 1:
        if mask_func1[0]:
            return Model(func1, prefix='%s0_' % running_prefix, independent_vars=independent_vars)
        else:
            return Model(func2, prefix='%s0_' % running_prefix, independent_vars=independent_vars)

    else:
        func_list = []
        for bool_index in mask_func1:
            if bool_index:
                func_list.append(func1)
            else:
                func_list.append(func2)

        fmodel = (Model(func_list[0], prefix='%s0_' % running_prefix, independent_vars=independent_vars) +
                  Model(func_list[1], prefix='%s1_' % running_prefix, independent_vars=independent_vars))
        for double_index in range(0, len(mask_func1)-1, 2):
            if double_index != 0:
                fmodel += (Model(func_list[double_index+0], prefix='%s%i_' % (running_prefix, double_index+0), independent_vars=independent_vars) +
                           Model(func_list[double_index+1], prefix='%s%i_' % (running_prefix, double_index+1), independent_vars=independent_vars))
        if len(mask_func1) % 2:
            fmodel += (Model(func_list[len(mask_func1)-1], prefix='%s%i_' % (running_prefix, len(mask_func1)-1), independent_vars=independent_vars) +
                       Model(null_func2d, prefix='null_func_', independent_vars=independent_vars))

        return fmodel


def transform_ellipse_world2pix(param_table, wcs):
    x, y = wcs.world_to_pixel(SkyCoord(ra=param_table['ra'], dec=param_table['dec']))

    points_x_a_low, points_y_a_low = wcs.world_to_pixel(SkyCoord(ra=param_table['ra'], dec=param_table['dec']))
    points_x_a_high, points_y_a_high = wcs.world_to_pixel(SkyCoord(ra=param_table['ra'] + param_table['a'],
                                                                   dec=param_table['dec']))
    a = points_x_a_low - points_x_a_high

    points_x_b_low, points_y_b_low = wcs.world_to_pixel(SkyCoord(ra=param_table['ra'], dec=param_table['dec']))
    points_x_b_high, points_y_b_high = wcs.world_to_pixel(SkyCoord(ra=param_table['ra'],
                                                                   dec=param_table['dec'] + param_table['b']))
    b = points_y_b_low - points_y_b_high

    return x, y, a, b


def transform_ellipse_pix2world(param_table, wcs):
    wcs_position = wcs.pixel_to_world(param_table['x'], param_table['y'])
    a = (wcs.pixel_to_world(param_table['x'], param_table['y']).ra -
         wcs.pixel_to_world(param_table['x'] + param_table['a'], param_table['y']).ra)
    b = (wcs.pixel_to_world(param_table['x'], param_table['y']).dec -
         wcs.pixel_to_world(param_table['x'], param_table['y'] + param_table['b']).dec)

    return wcs_position.ra, wcs_position.dec, a, b


def transform_pix2world_scale(pixel_length, wcs, dim=0, return_unit='arcsec'):

    if return_unit == 'arcsec':
        return pixel_length * (wcs.proj_plane_pixel_scales()[dim]).to(u.arcsec).value


def transform_world2pix_scale(length_in_arcsec, wcs, dim=0):

    return (length_in_arcsec*u.arcsec).to(u.deg) / wcs.proj_plane_pixel_scales()[dim]


def set_2d_gauss_params(fmodel, initial_params, wcs, img_mean, img_std, img_max, running_prefix='g_'):

    x, y, a, b = transform_ellipse_world2pix(param_table=initial_params, wcs=wcs)
    theta = initial_params['theta'] * 180. / np.pi
    theta[theta < 0] += 180
    theta_min = theta - 1
    theta_max = theta + 1
    theta_min[theta_min < 0] = 0
    theta_max[theta_max > 180] = 180

    params = fmodel.make_params()

    for index in range(len(initial_params)):
        # it is very important to let the amplitude become extremely large.
        # If fact due to the PSF convolution this will be smeared out
        params['%s%i_amp' % (running_prefix, index)].set(value=img_max/3, min=0, max=1000 * img_max + 100*img_std)
        params['%s%i_x0' % (running_prefix, index)].set(value=x[index], min=x[index] - 2*a[index],
                                                        max=x[index] + 2*a[index])
        params['%s%i_y0' % (running_prefix, index)].set(value=y[index], min=y[index] - 2*b[index],
                                                        max=y[index] + 2*b[index])
        params['%s%i_sig_x' % (running_prefix, index)].set(value=a[index], min=0.01, max=2*a[index])
        params['%s%i_sig_y' % (running_prefix, index)].set(value=b[index], min=0.01, max=2*b[index])
        # params['%s%i_theta' % (running_prefix, index)].set(value=theta[index], min=theta_min[index],
        #                                                    max=theta_max[index])
        params['%s%i_theta' % (running_prefix, index)].set(value=0, min=0, max=360)

    if 'null_func_amp' in fmodel.param_names:
        params['null_func_amp'].set(value=0, vary=False)
        params['null_func_x0'].set(value=0, vary=False)
        params['null_func_y0'].set(value=0, vary=False)
        params['null_func_sig_x'].set(value=0.001, vary=False)
        params['null_func_sig_y'].set(value=0.001, vary=False)
        params['null_func_theta'].set(value=0, vary=False)

    return params


def set_mixt_model_params(fmodel, init_pos, param_lim, img_mean, img_std, img_max, mask_gauss, running_prefix='g_'):

    init_x, init_y = init_pos

    params = fmodel.make_params()

    lim_x, lim_y = param_lim

    for index in range(len(init_x)):
        # it is very important to let the amplitude become extremely large.
        # If fact due to the PSF convolution this will be smeared out
        params['%s%i_amp' % (running_prefix, index)].set(value=img_max, min=0, max=10000 * img_max + 10000*img_std)
        params['%s%i_x0' % (running_prefix, index)].set(value=init_x[index], min=init_x[index] - lim_x[index],
                                                        max=init_x[index] + lim_x[index])
        params['%s%i_y0' % (running_prefix, index)].set(value=init_y[index], min=init_y[index] - lim_y[index],
                                                        max=init_y[index] + lim_y[index])
        if mask_gauss[index]:

            params['%s%i_sig_x' % (running_prefix, index)].set(value=lim_x[index]*0.5, min=0.01, max=2*lim_x[index])
            params['%s%i_sig_y' % (running_prefix, index)].set(value=lim_y[index]*0.5, min=0.01, max=2*lim_y[index])
            params['%s%i_theta' % (running_prefix, index)].set(value=0, min=0, max=360)

    if 'null_func_a' in fmodel.param_names:
        params['null_func_a'].set(value=0, vary=False)

    return params


def create_2d_data_mesh(data):
    # create x and y data grid
    x = np.linspace(0, data.shape[0]-1, data.shape[0])
    y = np.linspace(0, data.shape[1]-1, data.shape[1])
    x_grid, y_grid = np.meshgrid(x, y)
    return x_grid, y_grid


def check_point_inside_ellipse(x_ell, y_ell, a_ell, b_ell, theta_ell, x_p, y_p):
    """

    Parameters
    ----------
    x_ell, y_ell : float
        center of ellipse
    a_ell, b_ell : float
        the minor and major diameter of the ellipse. NOT THE RADIUS!!!!
    theta_ell : float
        rotation angle of ellipse in rad
    x_p, y_p : float or array
        position of point you want to check

    Returns
    -------
    bool or array of bool
    """
    element_1 = ((np.cos(theta_ell)*(x_p - x_ell) + np.sin(theta_ell)*(y_p - y_ell))**2) / ((a_ell/2)**2)
    element_2 = ((np.sin(theta_ell)*(x_p - x_ell) - np.cos(theta_ell)*(y_p - y_ell))**2) / ((b_ell/2)**2)
    return (element_1 + element_2) <= 1


