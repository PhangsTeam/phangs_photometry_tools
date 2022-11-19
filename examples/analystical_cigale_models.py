import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.io import fits

from cigale_helper import cigale_wrapper as cw

# crate wrapper class object
cigale_wrapper_obj = cw.CigaleModelWrapper()

# set parameters
# cigale_wrapper_obj.sed_modules_params['ssp']['index'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# parameter_list = ['stellar.age']

# cigale_wrapper_obj.sed_modules_params['nebular']['logU'] = [-4.0, -3.0, -2.0, -1.0]
# parameter_list = ['stellar.age', 'nebular.logU']

# cigale_wrapper_obj.sed_modules_params['nebular']['zgas'] = [0.002, 0.008, 0.020, 0.041]
# parameter_list = ['stellar.age', 'nebular.zgas']

# cigale_wrapper_obj.sed_modules_params['nebular']['ne'] = [10, 100, 1000]
# parameter_list = ['stellar.age', 'nebular.ne']

# cigale_wrapper_obj.sed_modules_params['nebular']['f_esc'] = [0.0, 0.1, 0.3, 0.9]
# parameter_list = ['stellar.age', 'nebular.f_esc']

# cigale_wrapper_obj.sed_modules_params['nebular']['f_dust'] = [0.0, 0.1, 0.3, 0.9]
# parameter_list = ['stellar.age', 'nebular.f_dust']

# cigale_wrapper_obj.sed_modules_params['dustextPHANGS']['A550'] = [0.0, 0.5, 1.0, 8.0, 15.0, 30.0]
# parameter_list = ['stellar.age', 'attenuation.A550']

# cigale_wrapper_obj.sed_modules_params['dl2014']['qpah'] = [0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58, 5.26, 5.95, 6.63, 7.32]
# parameter_list = ['stellar.age', 'dl2014.qpah']

# cigale_wrapper_obj.sed_modules_params['dl2014']['umin'] = [0.100, 0.200, 0.400,  0.800, 1.500, 3.000, 7.000, 15.00, 25.00, 50.00]
# # cigale_wrapper_obj.sed_modules_params['dl2014']['gamma'] = [1.0]
# parameter_list = ['stellar.age', 'dl2014.umin']

# cigale_wrapper_obj.sed_modules_params['dl2014']['alpha'] = [1.0,  1.2, 1.6, 1.8, 2.2, 2.6, 2.8, 3.0]
# parameter_list = ['dl2014.alpha']
#
cigale_wrapper_obj.sed_modules_params['dl2014']['gamma'] = [0.0, 0.001, 0.01, 0.1, 0.3, 0.9]
parameter_list = ['dl2014.gamma']


# run cigale
cigale_wrapper_obj.create_cigale_model()

# load model into constructor
cigale_wrapper_obj.load_cigale_model_block()

# get legend list
label_list = cigale_wrapper_obj.create_label_str_list(parameter_list=parameter_list)


# set star cluster mass to scale the model SEDs
cluster_mass = 1E4 * u.Msun
# set distance to galaxy, NGC3351 = 10 Mpc, NGC1566 = 17.7 Mpc
distance_Mpc = 10 * u.Mpc


figure = plt.figure(figsize=(20, 10))
fontsize = 26
ax_models = figure.add_axes([0.08, 0.08, 0.91, 0.91])

# plot observation filters
cigale_wrapper_obj.plot_hst_nircam_miri_filters(ax=ax_models, fontsize=fontsize-10)

for id_index, id in zip(range(len(cigale_wrapper_obj.model_table_dict['id']['value'])),
						cigale_wrapper_obj.model_table_dict['id']['value']):
	cigale_wrapper_obj.plot_cigale_model(ax=ax_models, model_file_name='out/%i_best_model.fits'%id,
										 cluster_mass=cluster_mass, distance_Mpc=distance_Mpc, label=label_list[id_index])


ax_models.set_xlim(230, 3e4)
ax_models.set_ylim(7e-8, 1.5e3)

ax_models.set_xscale('log')
ax_models.set_yscale('log')

ax_models.set_xlabel('Wavelength [nm]', labelpad=-4, fontsize=fontsize)
ax_models.set_ylabel(r'F$_{\nu}$ [mJy]', fontsize=fontsize)
ax_models.tick_params(axis='both', which='both', width=2, direction='in', labelsize=fontsize)
ax_models.legend(loc='upper left', fontsize=fontsize-6, ncol=3, columnspacing=1, handlelength=1, handletextpad=0.6)

plt.figtext(0.17, 0.035, 'HST WFC3', fontsize=fontsize-4)


plt.savefig('plot_output/filter_overview.png')
plt.savefig('plot_output/filter_overview.pdf')

