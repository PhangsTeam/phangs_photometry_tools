import os
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
import astropy.constants as const
from astropy.table import Table


class CigaleModelWrapper:
    def __init__(self):
        # initial params
        self.cigale_init_params = {
            'data_file': '',
            'parameters_file': '',
            'sed_modules': ['sfh2exp', 'bc03', 'dustext', 'redshifting'],
            'analysis_method': 'savefluxes',
            'cores': 1,
        }
        self.sed_modules_params = {
            # 'ssp':
            #     {
            #         # Index of the SSP to use.
            #         'index': [0]
            #     },
            'sfh2exp':
                {
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
                },
            'bc03':
                {
                    # Initial mass function: 0 (Salpeter) or 1 (Chabrier).
                    'imf': [1],
                    # Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05.
                    'metallicity': [0.02],
                    # Age [Myr] of the separation between the young and the old star
                    # populations. The default value in 10^7 years (10 Myr). Set to 0 not to
                    # differentiate ages (only an old population).
                    'separation_age': [10]
                },
            # 'bc03_ssp':
            #     {
            #         # Initial mass function: 0 (Salpeter) or 1 (Chabrier).
            #         'imf': [1],
            #         # Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05.
            #         'metallicity': [0.02],
            #         # Age [Myr] of the separation between the young and the old star
            #         # populations. The default value in 10^7 years (10 Myr). Set to 0 not to
            #         # differentiate ages (only an old population).
            #         'separation_age': [10]
            #     },
            # 'nebular':
            #     {
            #         # Ionisation parameter. Possible values are: -4.0, -3.9, -3.8, -3.7,
            #         # -3.6, -3.5, -3.4, -3.3, -3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6,
            #         # -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5,
            #         # -1.4, -1.3, -1.2, -1.1, -1.0.
            #         'logU': [-2.0],
            #         # Gas metallicity. Possible values are: 0.000, 0.0004, 0.001, 0.002,
            #         # 0.0025, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.011, 0.012,
            #         # 0.014, 0.016, 0.019, 0.020, 0.022, 0.025, 0.03, 0.033, 0.037, 0.041,
            #         # 0.046, 0.051.
            #         'zgas': [0.02],
            #         # Electron density. Possible values are: 10, 100, 1000.
            #         'ne': [100],
            #         # Fraction of Lyman continuum photons escaping the galaxy. Possible
            #         # values between 0 and 1.
            #         'f_esc': [0.0],
            #         # Fraction of Lyman continuum photons absorbed by dust. Possible values
            #         # between 0 and 1.
            #         'f_dust': [0.0],
            #         # Line width in km/s.
            #         'lines_width': [100.0],
            #         # Include nebular emission.
            #         'emission': True},
            'dustext':
                {
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
                },
            # 'dustextPHANGS':
            #     {
            #         # Attenuation at 550 nm.
            #         'A550': [0.3],
            #         'filters': 'B_B90 & V_B90 & FUV'
            #     },
            # 'dl2014':
            #     {
            #         # Mass fraction of PAH. Possible values are: 0.47, 1.12, 1.77, 2.50,
            #         # 3.19, 3.90, 4.58, 5.26, 5.95, 6.63, 7.32.
            #         'qpah': [2.5],
            #         # Minimum radiation field. Possible values are: 0.100, 0.120, 0.150,
            #         # 0.170, 0.200, 0.250, 0.300, 0.350, 0.400, 0.500, 0.600, 0.700, 0.800,
            #         # 1.000, 1.200, 1.500, 1.700, 2.000, 2.500, 3.000, 3.500, 4.000, 5.000,
            #         # 6.000, 7.000, 8.000, 10.00, 12.00, 15.00, 17.00, 20.00, 25.00, 30.00,
            #         # 35.00, 40.00, 50.00.
            #         'umin': [1.0],
            #         # Powerlaw slope dU/dM propto U^alpha. Possible values are: 1.0, 1.1,
            #         # 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5,
            #         # 2.6, 2.7, 2.8, 2.9, 3.0.
            #         'alpha': [2.0],
            #         # Fraction illuminated from Umin to Umax. Possible values between 0 and
            #         # 1.
            #         'gamma': [0.1],
            #         # Take self-absorption into account.
            #         'self_abs': False,
            #     },
            'redshifting':
                {
                    # Redshift of the objects. Leave empty to use the redshifts from the
                    # input file.
                    'redshift': [0.0]
                }
        }
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

        # cigale filter names
        self.cigale_filter_names_uvis = {
            'F275W': 'F275W_UVIS_CHIP2',
            'F336W': 'F336W_UVIS_CHIP2',
            'F438W': 'F438W_UVIS_CHIP2',
            'F555W': 'F555W_UVIS_CHIP2',
            'F814W': 'F814W_UVIS_CHIP2',
        }

        self.cigale_filter_names_nircam = {
            'F200W': 'jwst.nircam.F200W',
            'F200W_err': 'jwst.nircam.F200W_err',
            'F300M': 'jwst.nircam.F300M',
            'F300M_err': 'jwst.nircam.F300M_err',
            'F335M': 'jwst.nircam.F335M',
            'F335M_err': 'jwst.nircam.F335M_err',
            'F360M': 'jwst.nircam.F360M',
            'F360M_err': 'jwst.nircam.F360M_err',
            'F770W': 'jwst.miri.F770W',
            'F770W_err': 'jwst.miri.F770W_err',
            'F1000W': 'jwst.miri.F1000W',
            'F1000W_err': 'jwst.miri.F1000W_err',
            'F1130W': 'jwst.miri.F1130W',
            'F1130W_err': 'jwst.miri.F1130W_err',
            'F2100W': 'jwst.miri.F2100W',
            'F2100W_er': 'jwst.miri.F2100W_err'
        }

        # filter pivot wavelengths (in wavlength order)
        filter_lam_angstrom = np.array([19886.48, 29891.21, 33620.67, 36241.76, 76393.24, 99531.16, 113085.01,
                                        207950.06]) * u.AA
        self.filter_lam_nm = filter_lam_angstrom.to(u.nm).value

        # filter FWHM (but divide by 2, so we can plot the span centered on the pivot wavelengths)
        filter_fwhm_angstrom = np.array([4689.40, 3276.84, 3576.00, 3855.31, 21048.58, 18731.30, 7128.97, 46818.15]) / 2 * u.AA
        self.filter_fwhm_nm = filter_fwhm_angstrom.to(u.nm).value

        self.filter_colors = np.array(['#bb44bb', '#6244bb', '#335ccc', '#3ec1c1', '#84ff00', '#e8f00f', '#de9221', '#f20d0d'])
        self.filter_labels = np.array(['F200W', 'F300M', 'F335M', 'F360M', 'F770W', 'F1000W', 'F1130W', 'F2100W'])

        # hst filters
        # self.hst_lam_nm = np.array([270.720, 335.485, 432.555, 530.59, 656.661, 804.810])
        # self.hst_fwhm = np.array([39.8, 51.2, 61.8, 156.2, 125.54, 153.6]) / 2

        self.hst_lam_nm = np.array([270.720, 335.485, 432.555, 530.59, 804.810])
        self.hst_fwhm = np.array([39.8, 51.2, 61.8, 156.2, 153.6]) / 2

        self.hst_filter_colors = np.array(['royalblue', 'blueviolet', 'deeppink', 'tab:red', 'darkred'])
        self.hst_filter_labels = np.array(['F275W', 'F336W', 'F438W', 'F555W', 'F814W'])

        # cigale model table
        self.model_table = None
        self.model_table_dict = None

    @staticmethod
    def replace_params_in_file(param_dict, file_name='pcigale.ini'):
        with open(file_name, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        print(param_dict.keys())
        for key in param_dict.keys():
            print('key ', key)
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

    def create_cigale_model(self, delete_old_models=True):
        # initiate pcigale
        os.system('pcigale init')
        # set initial parameters
        self.replace_params_in_file(param_dict=self.cigale_init_params, file_name='pcigale.ini')
        # configurate pcigale
        os.system('pcigale genconf')
        # set module configurations
        for module_str in self.sed_modules_params.keys():
            print(module_str)
            self.replace_params_in_file(param_dict=self.sed_modules_params[module_str], file_name='pcigale.ini')
        # change analysis parameters
        self.replace_params_in_file(param_dict=self.analysis_params, file_name='pcigale.ini')
        # run pcigale
        os.system('pcigale run')
        # delete old models
        if delete_old_models:
            os.system('rm -rf *_out')

    def compute_cigale_band_name_list(self, band_list):

        band_name_list = []
        for band in band_list:
            if band in self.hst_targets[self.target_name]['acs_wfc1_observed_bands']:
                band_name_list.append(band + '_ACS')
            elif band in self.hst_targets[self.target_name]['wfc3_uvis_observed_bands']:
                band_name_list.append(band + '_UVIS_CHIP2')
            elif band in self.nircam_targets[self.target_name]['observed_bands']:
                band_name_list.append(band + 'jwst.nircam.' + band)
            elif band in self.miri_targets[self.target_name]['observed_bands']:
                band_name_list.append(band + 'jwst.miri.' + band)

        return band_name_list

    def create_cigale_flux_file(self, file_path, band_list, aperture_dict_list, snr=3, name_list=None,
                                redshift_list=None, dist_list=None):

        if isinstance(aperture_dict_list, dict):
            aperture_dict_list = [aperture_dict_list]

        if name_list is None:
            name_list = np.arange(start=0,  stop=len(aperture_dict_list)+1)
        if redshift_list is None:
            redshift_list = [0.0] * len(aperture_dict_list)
        if dist_list is None:
            dist_list = [self.dist_dict[self.target_name]['dist']] * len(aperture_dict_list)

        name_list = np.array(name_list, dtype=str)
        redshift_list = np.array(redshift_list)
        dist_list = np.array(dist_list)


        # create flux file
        flux_file = open(file_path, "w")
        # add header for all variables
        band_name_list = self.compute_cigale_band_name_list(band_list=band_list)
        flux_file.writelines("# id             redshift  distance   ")
        for band_name in band_name_list:
            flux_file.writelines(band_name + "   ")
            flux_file.writelines(band_name + "_err" + "   ")
        flux_file.writelines(" \n")

        # fill flux file
        for name, redshift, dist, aperture_dict in zip(name_list, redshift_list, dist_list, aperture_dict_list):
            flux_file.writelines(" %s   %f   %f  " % (name, redshift, dist))
            flux_list, flux_err_list = self.compute_cigale_flux_list(band_list=band_list, aperture_dict=aperture_dict,
                                                                     snr=snr)
            for flux, flux_err in zip(flux_list, flux_err_list):
                flux_file.writelines("%.15f   " % flux)
                flux_file.writelines("%.15f   " % flux_err)
            flux_file.writelines(" \n")

        flux_file.close()

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



