"""
Classes to gather basic attributes such as folder names, observation statuses or physical constants
"""


class PhangsDataStructure:
    """
    Class gathering all attributes to specify the data structure of PHANGS HST and JWST photometric observations
    """
    def __init__(self):
        super().__init__()

        self.target_list = ['ngc0628', 'ngc7496', 'ngc1365']

        self.target_observed_hst = ['ngc0628', 'ngc7496', 'ngc1365']
        self.target_observed_nircam = ['ngc0628', 'ngc7496', 'ngc1365']
        self.target_observed_miri = ['ngc0628', 'ngc7496', 'ngc1365']

        self.hst_ver_folder_names = {'v1': 'v1.0', 'v0.9': 'v0.9'}
        self.nircam_ver_folder_names = {'v0p4p2': 'v0p4p2'}
        self.miri_ver_folder_names = {'v0p5': 'v0p5_miri'}

        self.hst_targets = {
            'ngc0628':
                {'folder_name': 'ngc628mosaic',
                 'acs_wfc1_observed_bands': ['F435W', 'F814W'],
                 'wfc3_uvis_observed_bands': ['F275W', 'F336W', 'F555W']},
            'ngc7496':
                {'folder_name': 'ngc7496',
                 'acs_wfc1_observed_bands': [],
                 'wfc3_uvis_observed_bands': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
            'ngc1365':
                {'folder_name': 'ngc1365',
                 'acs_wfc1_observed_bands': [],
                 'wfc3_uvis_observed_bands': ['F275W', 'F336W', 'F438W', 'F555W', 'F814W']},
        }
        self.nircam_targets = {
            'ngc0628':
                {'folder_name': 'ngc0628',
                 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc7496':
                {'folder_name': 'ngc7496',
                 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
            'ngc1365':
                {'folder_name': 'ngc1365',
                 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
        }
        self.miri_targets = {
            'ngc0628':
                {'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc7496':
                {'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
            'ngc1365':
                {'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
        }


class PhysParams:
    """
    Class to gather all physical params
    """
    def __init__(self):
        super().__init__()

        """
        distances need to be done!!!!! See Lee et al 2022 Table 1
        """
        self.dist_dict = {
            'ngc0628': {'dist': 9.84, 'dist_err': 0.63, 'method': 'TRGB'},
            'ngc7496': {'dist': 18.72, 'dist_err': 2.81, 'method': 'NAM'}
        }
        self.sr_per_square_deg = 0.00030461741978671  # steradians per square degree

        # zero point NIRCAM flux corrections for data from the pipeline version v0p4p2
        self.nircam_zero_point_flux_corr = {'F200W': 0.854, 'F300M': 0.997, 'F335M': 1.000, 'F360M': 1.009}

        # filter names from http://svo2.cab.inta-csic.es
        self.hst_acs_wfc1_bands = ['FR388N', 'FR423N', 'F435W', 'FR459M', 'FR462N', 'F475W', 'F502N', 'FR505N', 'F555W',
                                   'FR551N', 'F550M', 'FR601N', 'F606W', 'F625W', 'FR647M', 'FR656N', 'F658N', 'F660N',
                                   'FR716N', 'POL_UV', 'POL_V', 'G800L', 'F775W', 'FR782N', 'F814W', 'FR853N', 'F892N',
                                   'FR914M', 'F850LP', 'FR931N', 'FR1016N']
        self.hst_wfc3_uvis2_bands = ['F218W', 'FQ232N', 'F225W', 'FQ243N', 'F275W', 'F280N', 'F300X', 'F336W', 'F343N',
                                     'F373N', 'FQ378N', 'FQ387N', 'F390M', 'F390W', 'F395N', 'F410M', 'FQ422M', 'F438W',
                                     'FQ436N', 'FQ437N', 'G280', 'F467M', 'F469N', 'F475W', 'F487N', 'FQ492N', 'F502N',
                                     'F475X', 'FQ508N', 'F555W', 'F547M', 'FQ575N', 'F606W', 'F200LP', 'FQ619N',
                                     'F621M', 'F625W', 'F631N', 'FQ634N', 'F645N', 'F350LP', 'F656N', 'F657N', 'F658N',
                                     'F665N', 'FQ672N', 'FQ674N', 'F673N', 'F680N', 'F689M', 'FQ727N', 'FQ750N',
                                     'F763M', 'F600LP', 'F775W', 'F814W', 'F845M', 'FQ889N', 'FQ906N', 'F850LP',
                                     'FQ924N', 'FQ937N', 'F953N']
        self.nircam_bands = ['F070W', 'F090W', 'F115W', 'F140M', 'F150W', 'F162M', 'F164N', 'F150W2', 'F182M', 'F187N',
                             'F200W', 'F210M', 'F212N', 'F250M', 'F277W', 'F300M', 'F323N', 'F322W2', 'F335M', 'F356W',
                             'F360M', 'F405N', 'F410M', 'F430M', 'F444W', 'F460M', 'F466N', 'F470N', 'F480M']
        self.miri_bands = ['F560W', 'F770W', 'F1000W', 'F1065C', 'F1140C', 'F1130W', 'F1280W', 'F1500W', 'F1550C',
                           'F1800W', 'F2100W', 'F2300C', 'F2550W']

        # band wavelength taken from
        # http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=HST&gname2=ACS_WFC&asttype=
        self.hst_acs_wfc1_bands_mean_wave = {
            'FR388N': 3881.71,
            'FR423N': 4230.39,
            'F435W': 4360.06,
            'FR459M': 4592.76,
            'FR462N': 4620.13,
            'F475W': 4802.31,
            'F502N': 5023.13,
            'FR505N': 5050.39,
            'F555W': 5397.60,
            'FR551N': 5510.30,
            'F550M': 5588.24,
            'FR601N': 6010.50,
            'F606W': 6035.73,
            'F625W': 6352.46,
            'FR647M': 6476.15,
            'FR656N': 6560.36,
            'F658N': 6584.10,
            'F660N': 6599.50,
            'FR716N': 7160.04,
            'POL_UV': 7294.11,
            'POL_V': 7523.49,
            'G800L': 7704.08,
            'F775W': 7730.77,
            'FR782N': 7819.44,
            'F814W': 8129.21,
            'FR853N': 8528.80,
            'F892N': 8915.37,
            'FR914M': 9079.84,
            'F850LP': 9080.26,
            'FR931N': 9306.31,
            'FR1016N': 10150.22,
        }
        self.hst_wfc3_uvis1_bands_mean_wave = {
            'F218W': 2231.14,
            'FQ232N': 2327.12,
            'F225W': 2377.24,
            'FQ243N': 2420.59,
            'F275W': 2718.36,
            'F280N': 2796.98,
            'F300X': 2867.82,
            'F336W': 3365.86,
            'F343N': 3438.50,
            'F373N': 3730.19,
            'FQ378N': 3792.78,
            'FQ387N': 3873.61,
            'F390M': 3898.62,
            'F390W': 3952.50,
            'F395N': 3955.38,
            'F410M': 4109.81,
            'FQ422M': 4219.70,
            'F438W': 4338.57,
            'FQ436N': 4367.41,
            'FQ437N': 4371.30,
            'G280': 4628.43,
            'F467M': 4683.55,
            'F469N': 4688.29,
            'F475W': 4827.71,
            'F487N': 4871.54,
            'FQ492N': 4933.83,
            'F502N': 5009.93,
            'F475X': 5076.23,
            'FQ508N': 5091.59,
            'F555W': 5388.55,
            'F547M': 5459.04,
            'FQ575N': 5756.92,
            'F606W': 5999.27,
            'F200LP': 6043.00,
            'FQ619N': 6198.49,
            'F621M': 6227.39,
            'F625W': 6291.29,
            'F631N': 6304.27,
            'FQ634N': 6349.37,
            'F645N': 6453.59,
            'F350LP': 6508.00,
            'F656N': 6561.54,
            'F657N': 6566.93,
            'F658N': 6585.64,
            'F665N': 6656.23,
            'FQ672N': 6717.13,
            'FQ674N': 6730.58,
            'F673N': 6766.27,
            'F680N': 6880.13,
            'F689M': 6885.92,
            'FQ727N': 7275.84,
            'FQ750N': 7502.54,
            'F763M': 7623.09,
            'F600LP': 7656.67,
            'F775W': 7683.41,
            'F814W': 8117.36,
            'F845M': 8449.34,
            'FQ889N': 8892.56,
            'FQ906N': 9058.19,
            'F850LP': 9207.49,
            'FQ924N': 9247.91,
            'FQ937N': 9372.90,
            'F953N': 9531.11,
        }
        self.nircam_bands_mean_wave = {
            'F070W': 7088.30,
            'F090W': 9083.40,
            'F115W': 11623.89,
            'F140M': 14074.46,
            'F150W': 15104.23,
            'F162M': 16296.59,
            'F164N': 16445.95,
            'F150W2': 17865.58,
            'F182M': 18494.30,
            'F187N': 18739.65,
            'F200W': 20028.15,
            'F210M': 20982.22,
            'F212N': 21213.97,
            'F250M': 25049.39,
            'F277W': 27844.64,
            'F300M': 29940.44,
            'F323N': 32369.29,
            'F322W2': 33334.98,
            'F335M': 33675.24,
            'F356W': 35934.49,
            'F360M': 36298.10,
            'F405N': 40517.39,
            'F410M': 40886.55,
            'F430M': 42829.39,
            'F444W': 44393.50,
            'F460M': 46315.57,
            'F466N': 46545.31,
            'F470N': 47078.82,
            'F480M': 48213.27
        }
        self.miri_bands_mean_wave = {
            'F560W': 56651.28,
            'F770W': 77111.39,
            'F1000W': 99981.09,
            'F1065C': 105681.52,
            'F1140C': 113156.52,
            'F1130W': 113159.44,
            'F1280W': 128738.34,
            'F1500W': 151469.08,
            'F1550C': 155219.65,
            'F1800W': 180508.31,
            'F2100W': 209373.20,
            'F2300C': 227630.49,
            'F2550W': 254994.19
        }

        # hst encircled energy for 50% and 80% of a point source
        # the computed values are interpolated for the aperture energyenclosure for the UVIS1 and UVIS2 table found at:
        # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy
        # the interpolation procedure can be found at ``../hst_psf_ee/compute_hst_psf_ee.py``
        self.hst_encircle_apertures_wfc3_uvis1_arcsec = {
            'F275W': {'ee50': 0.0822707423580786, 'ee80': 0.21022900763358784},
            'F300X': {'ee50': 0.07183566878980892, 'ee80': 0.18605544880592925},
            'F280N': {'ee50': 0.0742747695104532, 'ee80': 0.19406145107152087},
            'F336W': {'ee50': 0.08497576396206534, 'ee80': 0.1871036106750393},
            'F343N': {'ee50': 0.06925450666336849, 'ee80': 0.16801159420289863},
            'F373N': {'ee50': 0.06637465395262997, 'ee80': 0.1573038842345773},
            'F390M': {'ee50': 0.06754620972933206, 'ee80': 0.1618176197836168},
            'F390W': {'ee50': 0.06913956513659172, 'ee80': 0.1608947211452431},
            'F395N': {'ee50': 0.06875161033065456, 'ee80': 0.16032039433148496},
            'F410M': {'ee50': 0.09371942446043166, 'ee80': 0.18274568084711132},
            'F438W': {'ee50': 0.06903736698836921, 'ee80': 0.1557715430861724},
            'F467M': {'ee50': 0.06795220286417951, 'ee80': 0.15191359135913596},
            'F469N': {'ee50': 0.06886956521739131, 'ee80': 0.15647894645642005},
            'F475W': {'ee50': 0.069850040445523, 'ee80': 0.15439822165766914},
            'F487N': {'ee50': 0.09325647899910634, 'ee80': 0.17768742058449816},
            'F475X': {'ee50': 0.0834711893424643, 'ee80': 0.1957687914096414},
            'F200LP': {'ee50': 0.07210149198176122, 'ee80': 0.1558672656136792},
            'F502N': {'ee50': 0.06777562136104677, 'ee80': 0.1479508771929825},
            'F555W': {'ee50': 0.07046691129950652, 'ee80': 0.15283876757403533},
            'F547M': {'ee50': 0.0712762460068852, 'ee80': 0.15255306603773588},
            'F350LP': {'ee50': 0.07336316039980961, 'ee80': 0.1607216494845361},
            'F606W': {'ee50': 0.07091174788741343, 'ee80': 0.15282094594594597},
            'F621M': {'ee50': 0.07030923161609948, 'ee80': 0.1496267517691134},
            'F625W': {'ee50': 0.07346899099215864, 'ee80': 0.1552398847551046},
            'F631N': {'ee50': 0.06967976144172176, 'ee80': 0.15119572661279282},
            'F645N': {'ee50': 0.06969593034760241, 'ee80': 0.14867894100255344},
            'F656N': {'ee50': 0.07031221060986903, 'ee80': 0.15098054374436287},
            'F657N': {'ee50': 0.07014802901499984, 'ee80': 0.15021556256572033},
            'F658N': {'ee50': 0.0708986229419885, 'ee80': 0.15386164171399075},
            'F665N': {'ee50': 0.0706210006299526, 'ee80': 0.1514525139664805},
            'F673N': {'ee50': 0.09633659008890062, 'ee80': 0.18216850586792785},
            'F689M': {'ee50': 0.0968180044230519, 'ee80': 0.18145735392881132},
            'F680N': {'ee50': 0.0721983626358878, 'ee80': 0.15341682419659736},
            'F600LP': {'ee50': 0.07462507022703989, 'ee80': 0.15720930232558142},
            'F763M': {'ee50': 0.07236761426978819, 'ee80': 0.15524155844155846},
            'F775W': {'ee50': 0.0733488841694809, 'ee80': 0.15742775742775744},
            'F814W': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381},
            'F845M': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381},
            'F850LP': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381},
            'F953N': {'ee50': 0.07625649913344887, 'ee80': 0.1674208144796381}
        }
        self.hst_encircle_apertures_wfc3_uvis2_arcsec = {
            'F275W': {'ee50': 0.11002563163676309, 'ee80': 0.2126182965299685},
            'F300X': {'ee50': 0.07362485839132546, 'ee80': 0.18871158725683682},
            'F280N': {'ee50': 0.07019743109621891, 'ee80': 0.18288455772113948},
            'F336W': {'ee50': 0.06656083690660827, 'ee80': 0.15241806908768826},
            'F343N': {'ee50': 0.06917672954052154, 'ee80': 0.155386012715713},
            'F373N': {'ee50': 0.06940505900113997, 'ee80': 0.15713519952352592},
            'F390M': {'ee50': 0.06846401585532019, 'ee80': 0.15556587707075403},
            'F390W': {'ee50': 0.06709837054918527, 'ee80': 0.14826257459505543},
            'F395N': {'ee50': 0.06823408871745419, 'ee80': 0.15171940763834765},
            'F410M': {'ee50': 0.09201353485224453, 'ee80': 0.17397061426801208},
            'F438W': {'ee50': 0.06631333191837725, 'ee80': 0.14449639655475485},
            'F467M': {'ee50': 0.0663031226199543, 'ee80': 0.1464906333630687},
            'F469N': {'ee50': 0.06619528826366065, 'ee80': 0.1473578475336323},
            'F475W': {'ee50': 0.06864697401920186, 'ee80': 0.14801877934272303},
            'F487N': {'ee50': 0.06836516751083176, 'ee80': 0.15293060409385922},
            'F475X': {'ee50': 0.07797421609680502, 'ee80': 0.1923851203501094},
            'F200LP': {'ee50': 0.07087352362204724, 'ee80': 0.15511143911439118},
            'F502N': {'ee50': 0.06698717750656574, 'ee80': 0.1469007055584237},
            'F555W': {'ee50': 0.06755263238774319, 'ee80': 0.14312530552387162},
            'F547M': {'ee50': 0.0684225921892018, 'ee80': 0.14788227767114526},
            'F350LP': {'ee50': 0.07050133218999785, 'ee80': 0.15470160116448328},
            'F606W': {'ee50': 0.06889893283113621, 'ee80': 0.1469464285714286},
            'F621M': {'ee50': 0.06885909850802763, 'ee80': 0.1482506682506683},
            'F625W': {'ee50': 0.07011921613035137, 'ee80': 0.15006351446718422},
            'F631N': {'ee50': 0.07010144642974017, 'ee80': 0.15391515497786035},
            'F645N': {'ee50': 0.06977947973062194, 'ee80': 0.1532566396818634},
            'F656N': {'ee50': 0.07016378100140383, 'ee80': 0.15726596491228073},
            'F657N': {'ee50': 0.07006809917355372, 'ee80': 0.15509116409537171},
            'F658N': {'ee50': 0.07000720791560186, 'ee80': 0.15564334085778783},
            'F665N': {'ee50': 0.07103805297835149, 'ee80': 0.15450959488272922},
            'F673N': {'ee50': 0.0703399377112186, 'ee80': 0.15386321626617377},
            'F689M': {'ee50': 0.07224687239366137, 'ee80': 0.15868572861800856},
            'F680N': {'ee50': 0.07155458953352282, 'ee80': 0.15526838466373355},
            'F600LP': {'ee50': 0.07462507022703989, 'ee80': 0.15720930232558142},
            'F763M': {'ee50': 0.07236761426978819, 'ee80': 0.15524155844155846},
            'F775W': {'ee50': 0.0733488841694809, 'ee80': 0.15742775742775744},
            'F814W': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292},
            'F845M': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292},
            'F850LP': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292},
            'F953N': {'ee50': 0.09630835117773019, 'ee80': 0.18683307332293292}
        }

        self.hst_encircle_apertures_acs_wfc1_arcsec = {
            'F435W': {'ee50': 0.07552552552552552, 'ee80': 0.15851063829787237},
            'F475W': {'ee50': 0.0750733137829912, 'ee80': 0.15625000000000003},
            'F502N': {'ee50': 0.07514619883040935, 'ee80': 0.15625000000000003},
            'F555W': {'ee50': 0.07529411764705882, 'ee80': 0.1563829787234043},
            'F550M': {'ee50': 0.07544378698224852, 'ee80': 0.15652173913043482},
            'F606W': {'ee50': 0.07582582582582582, 'ee80': 0.15568181818181823},
            'F625W': {'ee50': 0.07615384615384616, 'ee80': 0.15581395348837213},
            'F658N': {'ee50': 0.07640625000000001, 'ee80': 0.15681818181818186},
            'F660N': {'ee50': 0.07648902821316614, 'ee80': 0.15681818181818186},
            'F775W': {'ee50': 0.07888513513513513, 'ee80': 0.16603773584905665},
            'F814W': {'ee50': 0.08079584775086505, 'ee80': 0.17500000000000007},
            'F892N': {'ee50': 0.0914179104477612, 'ee80': 0.22096774193548396},
            'F850LP': {'ee50': 0.09393939393939393, 'ee80': 0.23529411764705882}
        }

        # nircam encircled energy for filters
        # taken from Table 2 in:
        # https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-point-spread-functions
        # version 11/11/2022
        self.nircam_encircle_apertures_arcsec = {
            'F070W': {'ee50': 0.068, 'ee80': 0.266},
            'F090W': {'ee50': 0.060, 'ee80': 0.232},
            'F115W': {'ee50': 0.055, 'ee80': 0.213},
            'F140M': {'ee50': 0.043, 'ee80': 0.187},
            'F150W': {'ee50': 0.043, 'ee80': 0.189},
            'F162M': {'ee50': 0.044, 'ee80': 0.191},
            'F164N': {'ee50': 0.044, 'ee80': 0.192},
            'F150W2': {'ee50': 0.047, 'ee80': 0.197},
            'F182M': {'ee50': 0.046, 'ee80': 0.178},
            'F187N': {'ee50': 0.047, 'ee80': 0.175},
            'F200W': {'ee50': 0.049, 'ee80': 0.176},
            'F210M': {'ee50': 0.051, 'ee80': 0.177},
            'F212N': {'ee50': 0.051, 'ee80': 0.177},
            'F250M': {'ee50': 0.057, 'ee80': 0.181},
            'F277W': {'ee50': 0.061, 'ee80': 0.195},
            'F300M': {'ee50': 0.066, 'ee80': 0.205},
            'F322W2': {'ee50': 0.067, 'ee80': 0.218},
            'F323N': {'ee50': 0.071, 'ee80': 0.220},
            'F335M': {'ee50': 0.073, 'ee80': 0.225},
            'F356W': {'ee50': 0.076, 'ee80': 0.235},
            'F360M': {'ee50': 0.077, 'ee80': 0.238},
            'F405N': {'ee50': 0.086, 'ee80': 0.263},
            'F410M': {'ee50': 0.086, 'ee80': 0.266},
            'F430M': {'ee50': 0.090, 'ee80': 0.277},
            'F444W': {'ee50': 0.092, 'ee80': 0.283},
            'F460M': {'ee50': 0.096, 'ee80': 0.295},
            'F466N': {'ee50': 0.098, 'ee80': 0.299},
            'F470N': {'ee50': 0.099, 'ee80': 0.302},
            'F480M': {'ee50': 0.101, 'ee80': 0.308}
        }
        # nircam encircled energy for filters
        # taken from Table 2 in:
        # https://jwst-docs.stsci.edu/jwst-mid-infrared-instrument/miri-performance/miri-point-spread-functions
        # version 11/11/2022
        self.miri_encircle_apertures_arcsec = {
            'F560W': {'ee50': 0.131, 'ee80': 0.422},
            'F770W': {'ee50': 0.168, 'ee80': 0.519},
            'F1000W': {'ee50': 0.209, 'ee80': 0.636},
            'F1130W': {'ee50': 0.236, 'ee80': 0.712},
            'F1280W': {'ee50': 0.266, 'ee80': 0.801},
            'F1500W': {'ee50': 0.307, 'ee80': 0.932},
            'F1800W': {'ee50': 0.367, 'ee80': 1.110},
            'F2100W': {'ee50': 0.420, 'ee80': 1.276},
            'F2550W': {'ee50': 0.510, 'ee80': 1.545}
        }
        self.miri_encircle_apertures_pixel = {
            'F560W': {'ee50': 1.177, 'ee80': 3.805},
            'F770W': {'ee50': 1.510, 'ee80': 4.672},
            'F1000W': {'ee50': 1.882, 'ee80': 5.726},
            'F1130W': {'ee50': 2.124, 'ee80': 6.416},
            'F1280W': {'ee50': 2.397, 'ee80': 7.218},
            'F1500W': {'ee50': 2.765, 'ee80': 8.395},
            'F1800W': {'ee50': 3.311, 'ee80': 10.001},
            'F2100W': {'ee50': 3.783, 'ee80': 11.497},
            'F2550W': {'ee50': 4.591, 'ee80': 13.919},
        }


class CigaleModelWrapper:
    def __init__(self):
        # initial params
        self.cigale_init_params = {
            'data_file': '',
            'parameters_file': '',
            'sed_modules': ['ssp', 'bc03_ssp', 'nebular', 'dustextPHANGS', 'dl2014', 'redshifting'],
            'analysis_method': 'savefluxes',
            'cores': 1,
        }
        self.sed_modules_params = {
            'ssp':
                {
                    # Index of the SSP to use.
                    'index': [0]
                },
            'bc03_ssp':
                {
                    # Initial mass function: 0 (Salpeter) or 1 (Chabrier).
                    'imf': [0],
                    # Metalicity. Possible values are: 0.0001, 0.0004, 0.004, 0.008, 0.02, 0.05.
                    'metallicity': [0.02],
                    # Age [Myr] of the separation between the young and the old star
                    # populations. The default value in 10^7 years (10 Myr). Set to 0 not to
                    # differentiate ages (only an old population).
                    'separation_age': [10]
                },
            'nebular':
                {
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
                    'emission': True},
            'dustextPHANGS':
                {
                    # Attenuation at 550 nm.
                    'A550': [0.3],
                    'filters': 'B_B90 & V_B90 & FUV'
                },
            'dl2014':
                {
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
                },
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

