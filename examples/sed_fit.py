"""
Example for a circular flux extraction in multiple bands
"""
import os.path
from astropy.coordinates import SkyCoord
import astropy.units as u
from photometry_tools.analysis_tools import AnalysisTools
from photometry_tools.plotting_tools import PlotPhotometry
from cigale_helper import cigale_wrapper

import numpy as np
import matplotlib.pyplot as plt
# initialize photometry tools
phangs_photometry = AnalysisTools(hst_data_path='/home/benutzer/data/PHANGS-HST',
                                  nircam_data_path='/home/benutzer/data/PHANGS-JWST',
                                  miri_data_path='/home/benutzer/data/PHANGS-JWST',
                                  target_name='ngc0628',
                                  hst_data_ver='v1',
                                  nircam_data_ver='v0p4p2',
                                  miri_data_ver='v0p5')

# # load all data
phangs_photometry.load_hst_nircam_miri_bands(flux_unit='mJy')
#
#
# header = phangs_photometry.hst_bands_data['F555W_header_img']
#
# for key in header.keys():
#     print(key, ' ', header[key])
#
#
# f275w_chip1 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F275W_UVIS_CHIP1.dat')
# f275w_chip2 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F275W_UVIS_CHIP2.dat')
#
# f336w_chip1 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F336W_UVIS_CHIP1.dat')
# f336w_chip2 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F336W_UVIS_CHIP2.dat')
#
# f438w_chip1 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F438W_UVIS_CHIP1.dat')
# f438w_chip2 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F438W_UVIS_CHIP2.dat')
#
# f555w_chip1 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F555W_UVIS_CHIP1.dat')
# f555w_chip2 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F555W_UVIS_CHIP2.dat')
#
# f814w_chip1 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F814W_UVIS_CHIP1.dat')
# f814w_chip2 = np.genfromtxt('/home/benutzer/software/cigale-ssp/database_builder/filters/F814W_UVIS_CHIP2.dat')



# figure = plt.figure(figsize=(17, 9))
# fontsize = 10
#
# ax = figure.add_axes([0.09, 0.09, 0.83, 0.91])
#
# ax.plot(f275w_chip1[:, 0], f275w_chip1[:, 1], linestyle='--', linewidth=3, color='b')
# ax.plot(f275w_chip2[:, 0], f275w_chip2[:, 1], linestyle='--', linewidth=3, color='orange')
#
# ax.plot(f336w_chip1[:, 0], f336w_chip1[:, 1], linestyle='--', linewidth=3, color='b')
# ax.plot(f336w_chip2[:, 0], f336w_chip2[:, 1], linestyle='--', linewidth=3, color='orange')
#
# ax.plot(f438w_chip1[:, 0], f438w_chip1[:, 1], linestyle='--', linewidth=3, color='b')
# ax.plot(f438w_chip2[:, 0], f438w_chip2[:, 1], linestyle='--', linewidth=3, color='orange')
#
# ax.plot(f555w_chip1[:, 0], f555w_chip1[:, 1], linestyle='--', linewidth=3, color='b')
# ax.plot(f555w_chip2[:, 0], f555w_chip2[:, 1], linestyle='--', linewidth=3, color='orange')
#
# ax.plot(f814w_chip1[:, 0], f814w_chip1[:, 1], linestyle='--', linewidth=3, color='b')
# ax.plot(f814w_chip2[:, 0], f814w_chip2[:, 1], linestyle='--', linewidth=3, color='orange')
#
# plt.show()
#
# exit()



ra_center = 24.173946 - 33.5 / 3600
dec_center = 15.783662 - 27.5 / 3600
# size of image
size_of_cutout = (5, 5)
axis_length = (size_of_cutout[0] - 0.1, size_of_cutout[1] - 0.1)
cutout_dict = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_center, dec_cutout=dec_center,
                                                     cutout_size=size_of_cutout, include_err=True)

source_pos_1 = SkyCoord(ra=ra_center - 0.11 / 3600, dec=dec_center - 0.21 / 3600, unit=(u.degree, u.degree), frame='fk5')
source_pos_2 = SkyCoord(ra=ra_center + 0.41 / 3600, dec=dec_center + 0.03 / 3600, unit=(u.degree, u.degree), frame='fk5')
source_pos_3 = SkyCoord(ra=ra_center + 0.2 / 3600, dec=dec_center + 0.96 / 3600, unit=(u.degree, u.degree), frame='fk5')
source_list = [source_pos_1, source_pos_2, source_pos_3]


# aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source_pos_1,
#                                                                       recenter=True, recenter_rad=0.15)

aperture_dict_list = []
for source in source_list:
    # compute flux from 50% encircled energy
    aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                          recenter=True, recenter_rad=0.15)
    aperture_dict_list.append(aperture_dict)


if not os.path.isdir('sed_fit'):
    os.makedirs('sed_fit')

phangs_photometry.create_cigale_flux_file(file_path='sed_fit/flux_file.dat', band_list=cutout_dict['band_list'],
                                          aperture_dict_list=aperture_dict_list)

exit()




band_string = 'bands = '
for index in range(len(band_names)):
    band_string += band_names[index]
    if index < (len(band_names) -1):
        band_string += ', '
print(band_string)

# flux_file = open("flux_file.dat", "w")
#
# flux_file.writelines("# id             redshift  distance   ")
# for index in range(len(band_names)):
#     flux_file.writelines(band_names[index] + "   ")
# flux_file.writelines(" \n")


# for cluster_index in range(flux_list.shape[2]):
#     flux_file.writelines(" %i   0.0   " % cluster_index)
#     for band_index in range(flux_list.shape[0]):
#         flux_file.writelines("%.15f   " % flux_list[band_index, cluster_index])
#         flux_file.writelines("%.15f   " % err_list[band_index, cluster_index])
#     flux_file.writelines(" \n")

#
# for cluster_index in [2]:
#     flux_file.writelines(" %i   0.0   18.0  " % cluster_index)
#     for band_index in range(len(new_flux_list)):
#         if cluster_dict['upper_limit_%i' % cluster_index][band_index]:
#             flux_file.writelines("%.15f   " % (np.max([cluster_dict['flux_err_%i' % cluster_index][band_index], 0]) +
#                                                cluster_dict['flux_err_%i' % cluster_index][band_index]))
#             flux_file.writelines("%.15f   " % (-cluster_dict['flux_err_%i' % cluster_index][band_index]))
#         else:
#             flux_file.writelines("%.15f   " % (cluster_dict['flux_%i' % cluster_index][band_index]))
#             flux_file.writelines("%.15f   " % (cluster_dict['flux_err_%i' % cluster_index][band_index]))
#     flux_file.writelines(" \n")
#
# flux_file.close()











aperture_dict_list = []
for source in source_list:

    # compute flux from 50% encircled energy
    aperture_dict = phangs_photometry.circular_flux_aperture_from_cutouts(cutout_dict=cutout_dict, pos=source,
                                                                          recenter=True, recenter_rad=0.15)
    # print the measured fluxes
    for band in cutout_dict['band_list']:
        print(band, ' = %.6f ' % aperture_dict['aperture_dict_%s' % band]['flux'], '+/- ',
              '%.6f mJy' % aperture_dict['aperture_dict_%s' % band]['flux_err'])
    aperture_dict_list.append(aperture_dict)

# for plotting we want to use MJy/sr thus we convert the flux and
phangs_photometry.change_hst_nircam_miri_band_units(new_unit='MJy/sr')
# cutout_dict_new = phangs_photometry.get_band_cutout_dict(ra_cutout=ra_center, dec_cutout=dec_center,
#                                                          cutout_size=size_of_cutout, include_err=True)

# # sort bands
# hst_bands = phangs_photometry.sort_band_list(
#     band_list=(phangs_photometry.hst_targets[phangs_photometry.target_name]['acs_wfc1_observed_bands'] +
#                phangs_photometry.hst_targets[phangs_photometry.target_name]['wfc3_uvis_observed_bands']))
# for source_index in range(len(source_list)):
#
#     # plot the results
#     fig = PlotPhotometry.plot_circ_flux_extraction(
#         hst_band_list=hst_bands,
#         nircam_band_list=phangs_photometry.nircam_targets[phangs_photometry.target_name]['observed_bands'],
#         miri_band_list=phangs_photometry.miri_targets[phangs_photometry.target_name]['observed_bands'],
#         cutout_dict=cutout_dict, aperture_dict=aperture_dict_list[source_index],
#         vmax_vmin_hst=(0.05, 1.5), vmax_vmin_nircam=(0.1, 21), vmax_vmin_miri=(0.1, 21),
#         # cmap_hst='Blues', cmap_nircam='Greens', cmap_miri='Reds',
#         cmap_hst='Greys', cmap_nircam='Greys', cmap_miri='Greys',
#         log_scale=True, axis_length=axis_length)
#
#     if not os.path.isdir('plot_output'):
#         os.makedirs('plot_output')
#
#     fig.savefig('plot_output/circ_aperture_%i.png' % source_index)
#     fig.clf()


figure = PlotPhotometry.plot_rgb_with_sed(cutout_dict=cutout_dict, band_name_b='F435W', band_name_g='F200W', band_name_r='F770W',
                                 aperture_dict_list=aperture_dict_list,
                                 amp_fact_r=1, amp_fact_g=1, amp_fact_b=10,
                                 axis_length=axis_length,
                                 reproject_to='F435W')


figure.savefig('plot_output/rgb_sed.png')



