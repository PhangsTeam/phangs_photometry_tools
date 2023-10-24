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

# from lmfit import Model
import numpy as np


from photutils.segmentation import make_2dgaussian_kernel, detect_sources, deblend_sources
from astropy.convolution import Gaussian2DKernel, convolve
from scipy.stats import gaussian_kde

import math
from scipy.spatial import ConvexHull
from scipy import odr
import photometry_tools
import dust_tools.extinction_tools
from astropy.table import QTable

import matplotlib.pyplot as plt
from matplotlib import cm
import multicolorfits as mcf


def identify_file_in_folder(folder_path, str_in_file_name_1, str_in_file_name_2=None):
    """
    Identify a file inside a folder that contains a specific string.

    Parameters
    ----------
    folder_path : Path or str
    str_in_file_name_1 : str
    str_in_file_name_2 : str

    Returns
    -------
    file_name : Path
    """

    if str_in_file_name_2 is None:
        str_in_file_name_2 = str_in_file_name_1

    if isinstance(folder_path, str):
        folder_path = Path(folder_path)
    identified_files_1 = list(filter(lambda x: str_in_file_name_1 in x, os.listdir(folder_path)))

    identified_files_2 = list(filter(lambda x: str_in_file_name_2 in x, os.listdir(folder_path)))

    if not identified_files_1 and not identified_files_2:
        raise FileNotFoundError('The data file containing the string %s or %s does not exist.' %
                                (str_in_file_name_1, str_in_file_name_2))
    elif len(identified_files_1) > 1:
        raise FileExistsError('There are more than one data files containing the string %s .' % str_in_file_name_1)
    elif len(identified_files_2) > 1:
        raise FileExistsError('There are more than one data files containing the string %s .' % str_in_file_name_2)
    else:
        if not identified_files_2:
            return folder_path / str(identified_files_1[0])
        if not identified_files_1:
            return folder_path / str(identified_files_2[0])
        if identified_files_1 and identified_files_2:
            return folder_path / str(identified_files_1[0])

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


def calc_coord_separation(ra_ref, dec_ref, ra, dec):
    coord_ref = SkyCoord(ra=ra_ref, dec=dec_ref, unit=(u.degree, u.degree), frame='fk5')
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame='fk5')

    return coord_ref.separation(coord)


def get_pixel_surface_area_sq_kp(wcs, dist):
    pixel_scale = wcs.proj_plane_pixel_scales()[0]
    return (pixel_scale.to(u.arcsec).value * dist * 4.848 * 1e-3) ** 2


def arcsec2kpc(dist_arcsec, dist):
    return dist_arcsec * dist * 4.848 * 1e-3


def get_pixel_surface_area_sq_kp_wrong(wcs, ra, dec, dist):

    print('This is not entirly correct')
    exit()

    pixel_surface_area_sq_deg = wcs.proj_plane_pixel_area()

    central_coordinates = SkyCoord(ra=ra * u.deg,
                                   dec=dec * u.deg,
                                   distance=dist*u.Mpc)

    coords_1deg_ra_offset = SkyCoord(ra=central_coordinates.ra + 1*u.deg,
                                     dec=central_coordinates.dec,
                                     distance=dist*u.Mpc)
    coords_1deg_dec_offset = SkyCoord(ra=central_coordinates.ra,
                                      dec=central_coordinates.dec + 1*u.deg,
                                      distance=dist*u.Mpc)

    offset_dist_ra_mpc = central_coordinates.separation_3d(coords_1deg_ra_offset)
    offset_dist_dec_mpc = central_coordinates.separation_3d(coords_1deg_dec_offset)

    # print('offset_dist_ra_mpc ', offset_dist_ra_mpc)
    # print('offset_dist_dec_mpc ', offset_dist_dec_mpc)

    return (pixel_surface_area_sq_deg.value * offset_dist_ra_mpc.to(u.kpc) * offset_dist_dec_mpc.to(u.kpc)).value


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
    x = np.linspace(0, data.shape[1]-1, data.shape[1])
    y = np.linspace(0, data.shape[0]-1, data.shape[0])
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


def conv_mjy2vega(flux, ab_zp, vega_zp):

    """This function (non-sophisticated as of now)
     assumes the flux are given in units of milli-Janskies"""

    """First, convert mjy to f_nu"""
    conv_f_nu = flux*np.power(10.0, -26)
    """Convert f_nu in ergs s^-1 cm^-2 Hz^-1 to AB mag"""
    ABmag = -2.5*np.log10(conv_f_nu) - 48.60
    """Convert AB mag to Vega mag"""
    vega_mag = ABmag + (vega_zp - ab_zp)

    return vega_mag

def conv_mjy2ab_mag(flux):

    """conversion of mJy to AB mag.
    See definition on Wikipedia : https://en.wikipedia.org/wiki/AB_magnitude """

    return -2.5 * np.log10(flux * 1e3) + 8.90


def conv_mag2abs_mag(mag, dist):
    """
    conversion following https://en.wikipedia.org/wiki/Absolute_magnitude
    M = m - 5*log10(d_pc) + 5
    M = m - 5*log10(d_Mpc * 10^6) + 5
    M = m - 5*log10(d_Mpc) -5*log10(10^6) + 5
    M = m - 5*log10(d_Mpc) -25

    Parameters
    ----------
    mag : float or array
        magnitude
    dist : float or array
        distance in Mpc

    Returns
    -------
    float or array
        the absolute magnitude

     """
    return mag - 25 - 5*np.log10(dist)



def sort_counterclockwise(points, centre = None):
  if centre:
    centre_x, centre_y = centre
  else:
    centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
  angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
  counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
  counterclockwise_points = [points[i] for i in counterclockwise_indices]
  return counterclockwise_points


def gauss2d(x, y, x0, y0, sig_x, sig_y):
    expo = -(((x - x0)**2)/(2 * sig_x**2) + ((y - y0)**2)/(2 * sig_y**2))
    norm_amp = 1 / (2 * np.pi * sig_x * sig_y)
    return norm_amp * np.exp(expo)


def gauss_weight_map(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, kernal_std=None, kernel_size=9):

    # bins
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh
    x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
    gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

    for color_index in range(len(x_data)):
        gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
                        sig_x=x_data_err[color_index], sig_y=y_data_err[color_index])
        gauss_map += gauss

    if kernal_std is not None:
        kernel = make_2dgaussian_kernel(kernal_std, size=kernel_size)  # FWHM = 3.0
        gauss_map = convolve(gauss_map, kernel)

    gaus_dict = {
        'x_bins_gauss': x_bins_gauss,
        'y_bins_gauss': y_bins_gauss,
        'gauss_map': gauss_map
    }

    return gaus_dict


def calc_seg(x_data, y_data, x_data_err, y_data_err, x_lim, y_lim, n_bins, threshold_fact=2, kernal_std=4.0, contrast=0.1):

    # calculate combined errors
    data_err = np.sqrt(x_data_err**2 + y_data_err**2)
    noise_cut = np.percentile(data_err, 90)

    # bins
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh
    x_mesh, y_mesh = np.meshgrid(x_bins_gauss, y_bins_gauss)
    gauss_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))
    noise_map = np.zeros((len(x_bins_gauss), len(y_bins_gauss)))

    for color_index in range(len(x_data)):
        gauss = gauss2d(x=x_mesh, y=y_mesh, x0=x_data[color_index], y0=y_data[color_index],
                        sig_x=x_data_err[color_index], sig_y=y_data_err[color_index])
        gauss_map += gauss
        if data_err[color_index] > noise_cut:
            noise_map += gauss

    gauss_map -= np.nanmean(noise_map)

    kernel = make_2dgaussian_kernel(kernal_std, size=9)  # FWHM = 3.0

    conv_gauss_map = convolve(gauss_map, kernel)
    # threshold = len(x_data) / threshold_fact
    threshold = np.nanmax(conv_gauss_map) / threshold_fact

    seg_map = detect_sources(conv_gauss_map, threshold, npixels=30)
    seg_deb_map = deblend_sources(conv_gauss_map, seg_map, npixels=30, nlevels=100, mode='linear', contrast=contrast, progress_bar=False)
    numbers_of_seg = len(np.unique(seg_deb_map))
    return_dict = {
        'gauss_map': gauss_map, 'conv_gauss_map': conv_gauss_map, 'seg_deb_map': seg_deb_map}

    return return_dict


def get_contours(ax, x_bins, y_bins, data_map, contour_index=0, save_str=None, x_label=None, y_label=None):
    cs = ax.contour(x_bins, y_bins, data_map, colors='darkgray', linewidth=2, levels=[0.01])
    p = cs.collections[0].get_paths()[contour_index]
    v = p.vertices
    # get all points from contour
    x_cont = []
    y_cont = []
    for point in v:
        x_cont.append(point[0])
        y_cont.append(point[1])

    x_cont = np.array(x_cont)
    y_cont = np.array(y_cont)
    counterclockwise_points = sort_counterclockwise(points=np.array([x_cont, y_cont]).T)

    counterclockwise_points = np.array(counterclockwise_points)

    if save_str is not None:

        x_convex_hull = counterclockwise_points[:, 0]
        y_convex_hull = counterclockwise_points[:, 1]

        x_convex_hull = np.concatenate([x_convex_hull, np.array([x_convex_hull[0]])])
        y_convex_hull = np.concatenate([y_convex_hull, np.array([y_convex_hull[0]])])

        table = QTable([x_convex_hull, y_convex_hull],  names=(x_label, y_label))
        table.write('data_output/convex_hull_%s.fits' % save_str, overwrite=True)
        # np.save('data_output/x_convex_hull_%s.npy' % save_str, x_convex_hull)
        # np.save('data_output/y_convex_hull_%s.npy' % save_str, y_convex_hull)

        # ax.scatter(x_convex_hull, y_convex_hull)

def seg2hull(seg_map, x_lim, y_lim, n_bins, seg_index=1, contour_index=0,
             save_str=None, x_label=None, y_label=None):
    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh

    # kernel = make_2dgaussian_kernel(smooth_kernel, size=9)  # FWHM = 3.0
    # conv_gauss_map = convolve(gauss_map, kernel)
    # gauss_map_seg = conv_gauss_map.copy()
    # gauss_map_seg[seg_map._data != seg_index] = np.nan

    cs = plt.contour(x_bins_gauss, y_bins_gauss, (seg_map._data != seg_index), colors='darkgray', linewidth=2, levels=[0.01])
    p = cs.collections[0].get_paths()[contour_index]
    v = p.vertices
    # get all points from contour
    x_cont = []
    y_cont = []
    for point in v:
        x_cont.append(point[0])
        y_cont.append(point[1])

    x_cont = np.array(x_cont)
    y_cont = np.array(y_cont)
    counterclockwise_points = sort_counterclockwise(points=np.array([x_cont, y_cont]).T)

    counterclockwise_points = np.array(counterclockwise_points)

    x_convex_hull = counterclockwise_points[:, 0]
    y_convex_hull = counterclockwise_points[:, 1]
    x_convex_hull = np.concatenate([x_convex_hull, np.array([x_convex_hull[0]])])
    y_convex_hull = np.concatenate([y_convex_hull, np.array([y_convex_hull[0]])])

    if save_str is not None:
        table = QTable([x_convex_hull, y_convex_hull],  names=(x_label, y_label))
        table.write('data_output/convex_hull_%s.fits' % save_str, overwrite=True)

    return x_convex_hull, y_convex_hull



def contour2hulll(gauss_dict, levels, contour_index=0, circle_index=0):

    dummy_fig, dummy_ax = plt.subplots()
    cs = dummy_ax.contour(gauss_dict['x_bins_gauss'], gauss_dict['y_bins_gauss'],
                          gauss_dict['gauss_map'] / np.nanmax(gauss_dict['gauss_map']),
                          levels=levels)

    p = cs.collections[contour_index].get_paths()[circle_index]
    v = p.vertices

    # get rif of the dummy figure
    dummy_ax.cla()
    dummy_fig.clf()

    # get all points from contour
    x_cont = []
    y_cont = []
    for point in v:
        x_cont.append(point[0])
        y_cont.append(point[1])

    x_cont = np.array(x_cont)
    y_cont = np.array(y_cont)

    # make the contour a closed loop (add the first point to the end of the array)
    x_convex_hull = np.concatenate([x_cont, np.array([x_cont[0]])])
    y_convex_hull = np.concatenate([y_cont, np.array([y_cont[0]])])

    return x_convex_hull, y_convex_hull


def plot_reg_map(ax, gauss_map, seg_map, x_lim, y_lim, n_bins, smooth_kernel=4.0,
                 color_1='Blues', color_2='Greens', color_3='Reds', color_4='Purples',
                 plot_cont_1=False, plot_cont_2=False, plot_cont_3=False, plot_cont_4=False,
                 save_str_1=None, save_str_2=None, save_str_3=None, save_str_4=None,
                 x_label='vi', y_label='ub',
                 scale_method='individual'):

    x_bins_gauss = np.linspace(x_lim[0], x_lim[1], n_bins)
    y_bins_gauss = np.linspace(y_lim[1], y_lim[0], n_bins)
    # get a mesh

    kernel = make_2dgaussian_kernel(smooth_kernel, size=9)  # FWHM = 3.0
    conv_gauss_map = convolve(gauss_map, kernel)
    gauss_map_no_seg = conv_gauss_map.copy()
    gauss_map_seg1 = conv_gauss_map.copy()
    gauss_map_seg2 = conv_gauss_map.copy()
    gauss_map_seg3 = conv_gauss_map.copy()
    gauss_map_seg4 = conv_gauss_map.copy()
    gauss_map_no_seg[seg_map._data != 0] = np.nan
    gauss_map_seg1[seg_map._data != 1] = np.nan
    gauss_map_seg2[seg_map._data != 2] = np.nan
    gauss_map_seg3[seg_map._data != 3] = np.nan
    gauss_map_seg4[seg_map._data != 4] = np.nan

    scale = np.nanmax(conv_gauss_map)
    if np.sum(seg_map._data == 0) > 0:
        ax.imshow(gauss_map_no_seg, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),

                        cmap='Greys', vmin=0, vmax=scale, interpolation='nearest', aspect='auto')
    if np.sum(seg_map._data == 1) > 0:
        if scale_method == 'individual':
            scale = np.nanmax(gauss_map_seg1)
        ax.imshow(gauss_map_seg1, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_1, vmin=0, vmax=scale, interpolation='nearest', aspect='auto')
        if plot_cont_1:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 1),
                         contour_index=0, save_str=save_str_1, x_label=x_label, y_label=y_label)
    if np.sum(seg_map._data == 2) > 0:
        if scale_method == 'individual':
            scale = np.nanmax(gauss_map_seg2)
        ax.imshow(gauss_map_seg2, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_2, vmin=0, vmax=scale, interpolation='nearest', aspect='auto')
        if plot_cont_2:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 2),
                         contour_index=0, save_str=save_str_2, x_label=x_label, y_label=y_label)
    if np.sum(seg_map._data == 3) > 0:
        if scale_method == 'individual':
            scale = np.nanmax(gauss_map_seg3)
        ax.imshow(gauss_map_seg3, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_3, vmin=0, vmax=scale, interpolation='nearest', aspect='auto')
        if plot_cont_3:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 3),
                         contour_index=0, save_str=save_str_3, x_label=x_label, y_label=y_label)
    if np.sum(seg_map._data == 4) > 0:
        if scale_method == 'individual':
            scale = np.nanmax(gauss_map_seg4)
        ax.imshow(gauss_map_seg4, origin='lower', extent=(x_lim[0], x_lim[1], y_lim[1], y_lim[0]),
                        cmap=color_4, vmin=0, vmax=scale, interpolation='nearest', aspect='auto')
        if plot_cont_4:
            get_contours(ax=ax, x_bins=x_bins_gauss, y_bins=y_bins_gauss, data_map=(seg_map._data != 4),
                         contour_index=0, save_str=save_str_4, x_label=x_label, y_label=y_label)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

def lin_func(p, x):
    gradient, intersect = p
    return gradient*x + intersect


def points_in_hull(p, hull, tol=1e-12):
    return np.all(hull.equations[:,:-1] @ p.T + np.repeat(hull.equations[:,-1][None,:], len(p), axis=0).T <= tol, 0)


def fit_line(x_data, y_data, x_data_err, y_data_err):

    # Create a model for fitting.
    lin_model = odr.Model(lin_func)

    # Create a RealData object using our initiated data from above.
    data = odr.RealData(x_data, y_data, sx=x_data_err, sy=y_data_err)

    # Set up ODR with the model and data.
    odr_object = odr.ODR(data, lin_model, beta0=[0., 1.])

    # Run the regression.
    out = odr_object.run()

    # Use the in-built pprint method to give us results.
    # out.pprint()

    gradient, intersect = out.beta
    gradient_err, intersect_err = out.sd_beta

    return {
        'gradient': gradient,
        'intersect': intersect,
        'gradient_err': gradient_err,
        'intersect_err': intersect_err
    }


def plot_reddening_vect_ubvi(ax, vi_int, ub_int, max_av, fontsize, linewidth=2):
    catalog_access = photometry_tools.data_access.CatalogAccess()

    v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
    i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4
    u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
    b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
    max_color_ext_vi = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=v_wave, wave2=i_wave, av=max_av)
    max_color_ext_ub = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=u_wave, wave2=b_wave, av=max_av)

    slope_av_vector = ((ub_int + max_color_ext_ub) - ub_int) / ((vi_int + max_color_ext_vi) - vi_int)
    angle_av_vector = - np.arctan(slope_av_vector) * 180/np.pi

    ax.annotate('', xy=(vi_int + max_color_ext_vi, ub_int + max_color_ext_ub), xycoords='data',
                xytext=(vi_int, ub_int), fontsize=fontsize,
                textcoords='data', arrowprops=dict(arrowstyle='-|>', color='k', lw=linewidth, ls='-'))

    ax.text(vi_int + 0.05, ub_int + 0.2, r'A$_{\rm V}$ = %.1f' % max_av,
            horizontalalignment='left', verticalalignment='bottom',
            rotation=angle_av_vector, fontsize=fontsize)


def plot_reddening_vect(ax, x_color_1='v', x_color_2='i',  y_color_1='u', y_color_2='b',
                        x_color_int=0, y_color_int=0, av_val=1,
                        linewidth=2, line_color='k',
                        text=False, fontsize=20, text_color='k', x_text_offset=0.1, y_text_offset=-0.3):

    catalog_access = photometry_tools.data_access.CatalogAccess()

    nuv_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F275W']*1e-4
    u_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F336W']*1e-4
    b_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F438W']*1e-4
    v_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F555W']*1e-4
    i_wave = catalog_access.hst_wfc3_uvis1_bands_mean_wave['F814W']*1e-4

    x_wave_1 = locals()[x_color_1 + '_wave']
    x_wave_2 = locals()[x_color_2 + '_wave']
    y_wave_1 = locals()[y_color_1 + '_wave']
    y_wave_2 = locals()[y_color_2 + '_wave']

    color_ext_x = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=x_wave_1, wave2=x_wave_2, av=av_val)
    color_ext_y = dust_tools.extinction_tools.ExtinctionTools.color_ext_ccm89_av(wave1=y_wave_1, wave2=y_wave_2, av=av_val)

    slope_av_vector = ((y_color_int + color_ext_y) - y_color_int) / ((x_color_int + color_ext_x) - x_color_int)

    angle_av_vector = np.arctan(color_ext_y/color_ext_x) * 180/np.pi

    ax.annotate('', xy=(x_color_int + color_ext_x, y_color_int + color_ext_y), xycoords='data',
                xytext=(x_color_int, y_color_int), fontsize=fontsize,
                textcoords='data', arrowprops=dict(arrowstyle='-|>', color=line_color, lw=linewidth, ls='-'))

    if text:
        if isinstance(av_val, int):
            arrow_text = r'A$_{\rm V}$=%i mag' % av_val
        else:
            arrow_text = r'A$_{\rm V}$=%.1f mag' % av_val
        ax.text(x_color_int + x_text_offset, y_color_int + y_text_offset, arrow_text,
                horizontalalignment='left', verticalalignment='bottom',
                transform_rotates_text=True, rotation_mode='anchor',
                rotation=angle_av_vector, fontsize=fontsize, color=text_color)


def plot_contours(ax, x, y, levels=None, legend=False, fontsize=13):

    if levels is None:
        levels = [0.0, 0.1, 0.25, 0.5, 0.68, 0.95, 0.975]


    good_values = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    x = x[good_values]
    y = y[good_values]

    k = gaussian_kde(np.vstack([x, y]))
    xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))

    #set zi to 0-1 scale
    zi = (zi-zi.min())/(zi.max() - zi.min())
    zi = zi.reshape(xi.shape)

    origin = 'lower'
    cs = ax.contour(xi, yi, zi, levels=levels,
                    linewidths=(2,),
                    origin=origin)

    labels = []
    for level in levels[1:]:
        labels.append(str(int(level*100)) + ' %')
    h1, l1 = cs.legend_elements("Z1")

    if legend:
        ax.legend(h1, labels, frameon=False, fontsize=fontsize)

def density_with_points(ax, x, y, binx=None, biny=None, threshold=1, kernel_std=2.0, save=False, save_name='',
                        cmap='inferno', scatter_size=10, scatter_alpha=0.3):

    if binx is None:
        binx = np.linspace(-1.5, 2.5, 190)
    if biny is None:
        biny = np.linspace(-2.0, 2.0, 190)

    good = np.invert(((np.isnan(x) | np.isnan(y)) | (np.isinf(x) | np.isinf(y))))

    hist, xedges, yedges = np.histogram2d(x[good], y[good], bins=(binx, biny))

    if save:
        np.save('data_output/binx.npy', binx)
        np.save('data_output/biny.npy', biny)
        np.save('data_output/hist_%s_un_smoothed.npy' % save_name, hist)

    kernel = Gaussian2DKernel(x_stddev=kernel_std)
    hist = convolve(hist, kernel)

    if save:
        np.save('data_output/hist_%s_smoothed.npy' % save_name, hist)

    over_dense_regions = hist > threshold
    mask_high_dens = np.zeros(len(x), dtype=bool)

    for x_index in range(len(xedges)-1):
        for y_index in range(len(yedges)-1):
            if over_dense_regions[x_index, y_index]:
                mask = (x > xedges[x_index]) & (x < xedges[x_index + 1]) & (y > yedges[y_index]) & (y < yedges[y_index + 1])
                mask_high_dens += mask
    print(sum(mask_high_dens) / len(mask_high_dens))
    hist[hist <= threshold] = np.nan

    cmap = cm.get_cmap(cmap)

    scatter_color = cmap(0)

    ax.imshow(hist.T, origin='lower', extent=(binx.min(), binx.max(), biny.min(), biny.max()), cmap=cmap,
              interpolation='nearest', aspect='auto')
    ax.scatter(x[~mask_high_dens], y[~mask_high_dens], color=scatter_color, marker='.', s=scatter_size, alpha=scatter_alpha)
    ax.set_ylim(ax.get_ylim()[::-1])


nuvb_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}
ub_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.2], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.0], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}
bv_label_dict = {
    1: {'offsets': [0.1, -0.1], 'ha':'center', 'va':'bottom', 'label': r'1,2,3 Myr'},
    5: {'offsets': [0.05, 0.1], 'ha':'right', 'va':'top', 'label': r'5 Myr'},
    10: {'offsets': [0.1, -0.1], 'ha':'left', 'va':'bottom', 'label': r'10 Myr'},
    100: {'offsets': [-0.1, 0.1], 'ha':'right', 'va':'center', 'label': r'100 Myr'}
}

nuvb_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.7, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [+0.05, -0.9], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
ub_annotation_dict = {
    500: {'offset': [-0.5, +0.0], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}
bv_annotation_dict = {
    500: {'offset': [-0.5, +0.3], 'label': '500 Myr', 'ha': 'right', 'va': 'center'},
    1000: {'offset': [-0.5, +0.5], 'label': '1 Gyr', 'ha': 'right', 'va': 'center'},
    13750: {'offset': [-0.0, -0.7], 'label': '13.8 Gyr', 'ha': 'left', 'va': 'center'}
}



def display_models(ax, x_color_sol=None, y_color_sol=None, age_sol=None,
                   x_color_sol50=None, y_color_sol50=None, age_sol50=None,
                   age_cut_sol=0,
                   age_cut_sol50=5e2,
                   age_dots_sol=None,
                   age_dots_sol50=None,
                   display_age_dots=True,
                   age_labels=False,
                   age_label_color='red',
                   age_label_fontsize=30,
                   y_color='ub',
                   color_sol='tab:cyan', linewidth_sol=4, linestyle_sol='-',
                   color_sol50='m', linewidth_sol50=4, linestyle_sol50='-',
                   color_age_dots_sol='b', color_age_dots_sol50='tab:pink', size_age_dots=80,
                   label_sol=None, label_sol50=None):


    ax.plot(x_color_sol[age_sol > age_cut_sol], y_color_sol[age_sol > age_cut_sol], color=color_sol,
            linewidth=linewidth_sol, linestyle=linestyle_sol, zorder=10,
            label=label_sol)
    if x_color_sol50 is not None:
        ax.plot(x_color_sol50[age_sol50 > age_cut_sol50], y_color_sol50[age_sol50 > age_cut_sol50], color=color_sol50,
            linewidth=linewidth_sol50, linestyle=linestyle_sol50, zorder=10,
            label=label_sol50)
    if display_age_dots:
        if age_dots_sol is None:
            age_dots_sol = [1, 5, 10, 100, 500, 1000, 13750]
        for age in age_dots_sol:
            ax.scatter(x_color_sol[age_sol == age], y_color_sol[age_sol == age],
                       color=color_age_dots_sol, s=size_age_dots, zorder=20)

    if x_color_sol50 is not None:
        if age_dots_sol50 is None:
            age_dots_sol50 = [500, 1000, 13750]
        for age in age_dots_sol50:
            ax.scatter(x_color_sol50[age_sol50 == age], y_color_sol50[age_sol50 == age],
                       color=color_age_dots_sol50, s=size_age_dots, zorder=20)

    if age_labels:
        label_dict = globals()['%s_label_dict' % y_color]
        for age in label_dict.keys():
            ax.text(x_color_sol[age_sol == age]+label_dict[age]['offsets'][0],
                    y_color_sol[age_sol == age]+label_dict[age]['offsets'][1],
                    label_dict[age]['label'], horizontalalignment=label_dict[age]['ha'], verticalalignment=label_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize)


        annotation_dict = globals()['%s_annotation_dict' % y_color]
        for age in annotation_dict.keys():

            ax.annotate(' ', #annotation_dict[age]['label'],
                        xy=(x_color_sol[age_sol == age], y_color_sol[age_sol == age]),
                        xytext=(x_color_sol[age_sol == age]+annotation_dict[age]['offset'][0],
                                y_color_sol[age_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data', color=age_label_color,
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkcyan', lw=3, ls='-'))
            ax.annotate(' ',
                        xy=(x_color_sol50[age_sol50 == age], y_color_sol50[age_sol50 == age]),
                        xytext=(x_color_sol[age_sol == age]+annotation_dict[age]['offset'][0],
                                y_color_sol[age_sol == age]+annotation_dict[age]['offset'][1]),
                        fontsize=age_label_fontsize, xycoords='data', textcoords='data',
                        ha=annotation_dict[age]['ha'], va=annotation_dict[age]['va'], zorder=30,
                              arrowprops=dict(arrowstyle='-|>', color='darkviolet', lw=3, ls='-'))
            ax.text(x_color_sol[age_sol == age]+annotation_dict[age]['offset'][0],
                    y_color_sol[age_sol == age]+annotation_dict[age]['offset'][1],
                    annotation_dict[age]['label'],
                    horizontalalignment=annotation_dict[age]['ha'], verticalalignment=annotation_dict[age]['va'],
                    color=age_label_color, fontsize=age_label_fontsize, zorder=40)


def create3color_rgb_img(img_r, img_g, img_b, gamma=2.2, min_max=None, gamma_rgb=2.2,
                         red_color='#FF2400', green_color='#0FFF50', blue_color='#0096FF',
                         mask_no_coverage=True):

    if min_max is None:
        min_max = [0.3, 99.9]
    grey_r = mcf.greyRGBize_image(img_r,
                                  rescalefn='asinh', scaletype='perc', min_max=min_max,
                                  gamma=gamma, checkscale=False)
    grey_g = mcf.greyRGBize_image(img_g,
                                  rescalefn='asinh', scaletype='perc', min_max=min_max,
                                  gamma=gamma, checkscale=False)
    grey_b = mcf.greyRGBize_image(img_b,
                                  rescalefn='asinh', scaletype='perc', min_max=min_max,
                                  gamma=gamma, checkscale=False)

    r = mcf.colorize_image(grey_r, red_color, colorintype='hex', gammacorr_color=gamma)
    g = mcf.colorize_image(grey_g, green_color, colorintype='hex', gammacorr_color=gamma)
    b = mcf.colorize_image(grey_b, blue_color, colorintype='hex', gammacorr_color=gamma)
    rgb_image = mcf.combine_multicolor([r, g, b], gamma=gamma_rgb, inverse=False)

    if mask_no_coverage:
        mask_no_coverage = (rgb_image == rgb_image[-1, -1])
        rgb_image[mask_no_coverage] = 1.0
        mask_no_coverage = (rgb_image[:, :, 2] == 1)
        rgb_image[mask_no_coverage] = 1.0

    return rgb_image


def create4color_rgb_img(img_r, img_g, img_b, img_p, gamma=2.2, min_max=None, gamma_rgb=2.2,
                         red_color='#FF2400', green_color='#0FFF50', blue_color='#0096FF', pink_color='#FF00FF',
                         mask_no_coverage=True):

    if min_max is None:
        min_max = [0.3, 99.9]
    grey_r = mcf.greyRGBize_image(img_r,
                                  rescalefn='asinh', scaletype='perc', min_max=min_max,
                                  gamma=gamma, checkscale=False)
    grey_g = mcf.greyRGBize_image(img_g,
                                  rescalefn='asinh', scaletype='perc', min_max=min_max,
                                  gamma=gamma, checkscale=False)
    grey_b = mcf.greyRGBize_image(img_b,
                                  rescalefn='asinh', scaletype='perc', min_max=min_max,
                                  gamma=gamma, checkscale=False)
    grey_p = mcf.greyRGBize_image(img_p,
                                  rescalefn='asinh', scaletype='perc', min_max=min_max,
                                  gamma=gamma, checkscale=False)

    r = mcf.colorize_image(grey_r, red_color, colorintype='hex', gammacorr_color=2.2)
    g = mcf.colorize_image(grey_g, green_color, colorintype='hex', gammacorr_color=2.2)
    b = mcf.colorize_image(grey_b, blue_color, colorintype='hex', gammacorr_color=2.2)
    p = mcf.colorize_image(grey_p, pink_color, colorintype='hex', gammacorr_color=2.2)
    rgb_image = mcf.combine_multicolor([r, g, b, p], gamma=gamma_rgb, inverse=False)

    if mask_no_coverage:
        mask_no_coverage = (rgb_image == rgb_image[-1, -1])
        rgb_image[mask_no_coverage] = 1.0
        mask_no_coverage = (rgb_image[:, :, 2] == 1)
        rgb_image[mask_no_coverage] = 1.0

    return rgb_image
