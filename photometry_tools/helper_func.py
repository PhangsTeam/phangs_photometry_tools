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
