"""
Classes to gather basic attributes such as folder names, observation statuses or physical constants
"""


class PhangsDataStructure:
    """
    Class gathering all attributes to specify the data structure of PHANGS HST and JWST photometric observations
    """
    def __init__(self):
        super().__init__()

        self.target_list = ['ngc0628']

        self.target_observed_hst = ['ngc0628']
        self.target_observed_nircam = ['ngc0628']
        self.target_observed_miri = ['ngc0628']

        self.hst_bands = ['F275W', 'F336W', 'F435W', 'F438W', 'F555W', 'F814W']
        self.nircam_bands = ['F200W', 'F300M', 'F335M', 'F360M']
        self.miri_bands = ['F770W', 'F1000W', 'F1130W', 'F2100W']

        self.hst_ver_folder_names = {'v1': 'v1.0', 'v0.9': 'v0.9'}
        self.nircam_ver_folder_names = {'v0p4p2': 'v0p4p2'}
        self.miri_ver_folder_names = {'v0p5': 'v0p5_miri'}

        self.hst_targets = {
            'ngc0628':
                {'folder_name': 'ngc628mosaic',
                 'observed_bands': ['F275W', 'F336W', 'F435W', 'F555W', 'F814W'],
                 'acs_wfc1_observed_bands': ['F435W', 'F814W'],
                 'wfc3_uvis_observed_bands': ['F275W', 'F336W', 'F555W']},
        }
        self.nircam_targets = {
            'ngc0628':
                {'folder_name': 'ngc0628',
                 'observed_bands': ['F200W', 'F300M', 'F335M', 'F360M']},
        }
        self.miri_targets = {
            'ngc0628':
                {'observed_bands': ['F770W', 'F1000W', 'F1130W', 'F2100W']},
        }


class PhysParams:
    """
    Class to gather all physical params
    """
    def __init__(self):
        super().__init__()

        """
        distances need to be done!!!!!
        """
        self.dist_dict = {
            'ngc0628': {'dist': 11, 'dist_err': 1, 'method': 'TRGB'}
        }
        self.sr_per_square_deg = 0.00030461741978671  # steradians per square degree

        # hst encircled energy for 50% and 80% of a point source
        # the computed values are interpolated for the aperture energyenclosure for the UVIS1 and UVIS2 table found at:
        # https://www.stsci.edu/hst/instrumentation/wfc3/data-analysis/photometric-calibration/uvis-encircled-energy
        # the interpolation procedure can be found at ``../hst_psf_ee/compute_hst_psf_ee.py``
        self.hst_encircle_apertures_wfc3_uvis1_arcsec = {
            'F275W': {'ee50': 0.0822707423580786, 'ee_80': 0.21022900763358784},
            'F300X': {'ee50': 0.07183566878980892, 'ee_80': 0.18605544880592925},
            'F280N': {'ee50': 0.0742747695104532, 'ee_80': 0.19406145107152087},
            'F336W': {'ee50': 0.08497576396206534, 'ee_80': 0.1871036106750393},
            'F343N': {'ee50': 0.06925450666336849, 'ee_80': 0.16801159420289863},
            'F373N': {'ee50': 0.06637465395262997, 'ee_80': 0.1573038842345773},
            'F390M': {'ee50': 0.06754620972933206, 'ee_80': 0.1618176197836168},
            'F390W': {'ee50': 0.06913956513659172, 'ee_80': 0.1608947211452431},
            'F395N': {'ee50': 0.06875161033065456, 'ee_80': 0.16032039433148496},
            'F410M': {'ee50': 0.09371942446043166, 'ee_80': 0.18274568084711132},
            'F438W': {'ee50': 0.06903736698836921, 'ee_80': 0.1557715430861724},
            'F467M': {'ee50': 0.06795220286417951, 'ee_80': 0.15191359135913596},
            'F469N': {'ee50': 0.06886956521739131, 'ee_80': 0.15647894645642005},
            'F475W': {'ee50': 0.069850040445523, 'ee_80': 0.15439822165766914},
            'F487N': {'ee50': 0.09325647899910634, 'ee_80': 0.17768742058449816},
            'F475X': {'ee50': 0.0834711893424643, 'ee_80': 0.1957687914096414},
            'F200LP': {'ee50': 0.07210149198176122, 'ee_80': 0.1558672656136792},
            'F502N': {'ee50': 0.06777562136104677, 'ee_80': 0.1479508771929825},
            'F555W': {'ee50': 0.07046691129950652, 'ee_80': 0.15283876757403533},
            'F547M': {'ee50': 0.0712762460068852, 'ee_80': 0.15255306603773588},
            'F350LP': {'ee50': 0.07336316039980961, 'ee_80': 0.1607216494845361},
            'F606W': {'ee50': 0.07091174788741343, 'ee_80': 0.15282094594594597},
            'F621M': {'ee50': 0.07030923161609948, 'ee_80': 0.1496267517691134},
            'F625W': {'ee50': 0.07346899099215864, 'ee_80': 0.1552398847551046},
            'F631N': {'ee50': 0.06967976144172176, 'ee_80': 0.15119572661279282},
            'F645N': {'ee50': 0.06969593034760241, 'ee_80': 0.14867894100255344},
            'F656N': {'ee50': 0.07031221060986903, 'ee_80': 0.15098054374436287},
            'F657N': {'ee50': 0.07014802901499984, 'ee_80': 0.15021556256572033},
            'F658N': {'ee50': 0.0708986229419885, 'ee_80': 0.15386164171399075},
            'F665N': {'ee50': 0.0706210006299526, 'ee_80': 0.1514525139664805},
            'F673N': {'ee50': 0.09633659008890062, 'ee_80': 0.18216850586792785},
            'F689M': {'ee50': 0.0968180044230519, 'ee_80': 0.18145735392881132},
            'F680N': {'ee50': 0.0721983626358878, 'ee_80': 0.15341682419659736},
            'F600LP': {'ee50': 0.07462507022703989, 'ee_80': 0.15720930232558142},
            'F763M': {'ee50': 0.07236761426978819, 'ee_80': 0.15524155844155846},
            'F775W': {'ee50': 0.0733488841694809, 'ee_80': 0.15742775742775744},
            'F814W': {'ee50': 0.07625649913344887, 'ee_80': 0.1674208144796381},
            'F845M': {'ee50': 0.07625649913344887, 'ee_80': 0.1674208144796381},
            'F850LP': {'ee50': 0.07625649913344887, 'ee_80': 0.1674208144796381},
            'F953N': {'ee50': 0.07625649913344887, 'ee_80': 0.1674208144796381}
        }
        self.hst_encircle_apertures_wfc3_uvis2_arcsec = {
            'F275W': {'ee50': 0.11002563163676309, 'ee_80': 0.2126182965299685},
            'F300X': {'ee50': 0.07362485839132546, 'ee_80': 0.18871158725683682},
            'F280N': {'ee50': 0.07019743109621891, 'ee_80': 0.18288455772113948},
            'F336W': {'ee50': 0.06656083690660827, 'ee_80': 0.15241806908768826},
            'F343N': {'ee50': 0.06917672954052154, 'ee_80': 0.155386012715713},
            'F373N': {'ee50': 0.06940505900113997, 'ee_80': 0.15713519952352592},
            'F390M': {'ee50': 0.06846401585532019, 'ee_80': 0.15556587707075403},
            'F390W': {'ee50': 0.06709837054918527, 'ee_80': 0.14826257459505543},
            'F395N': {'ee50': 0.06823408871745419, 'ee_80': 0.15171940763834765},
            'F410M': {'ee50': 0.09201353485224453, 'ee_80': 0.17397061426801208},
            'F438W': {'ee50': 0.06631333191837725, 'ee_80': 0.14449639655475485},
            'F467M': {'ee50': 0.0663031226199543, 'ee_80': 0.1464906333630687},
            'F469N': {'ee50': 0.06619528826366065, 'ee_80': 0.1473578475336323},
            'F475W': {'ee50': 0.06864697401920186, 'ee_80': 0.14801877934272303},
            'F487N': {'ee50': 0.06836516751083176, 'ee_80': 0.15293060409385922},
            'F475X': {'ee50': 0.07797421609680502, 'ee_80': 0.1923851203501094},
            'F200LP': {'ee50': 0.07087352362204724, 'ee_80': 0.15511143911439118},
            'F502N': {'ee50': 0.06698717750656574, 'ee_80': 0.1469007055584237},
            'F555W': {'ee50': 0.06755263238774319, 'ee_80': 0.14312530552387162},
            'F547M': {'ee50': 0.0684225921892018, 'ee_80': 0.14788227767114526},
            'F350LP': {'ee50': 0.07050133218999785, 'ee_80': 0.15470160116448328},
            'F606W': {'ee50': 0.06889893283113621, 'ee_80': 0.1469464285714286},
            'F621M': {'ee50': 0.06885909850802763, 'ee_80': 0.1482506682506683},
            'F625W': {'ee50': 0.07011921613035137, 'ee_80': 0.15006351446718422},
            'F631N': {'ee50': 0.07010144642974017, 'ee_80': 0.15391515497786035},
            'F645N': {'ee50': 0.06977947973062194, 'ee_80': 0.1532566396818634},
            'F656N': {'ee50': 0.07016378100140383, 'ee_80': 0.15726596491228073},
            'F657N': {'ee50': 0.07006809917355372, 'ee_80': 0.15509116409537171},
            'F658N': {'ee50': 0.07000720791560186, 'ee_80': 0.15564334085778783},
            'F665N': {'ee50': 0.07103805297835149, 'ee_80': 0.15450959488272922},
            'F673N': {'ee50': 0.0703399377112186, 'ee_80': 0.15386321626617377},
            'F689M': {'ee50': 0.07224687239366137, 'ee_80': 0.15868572861800856},
            'F680N': {'ee50': 0.07155458953352282, 'ee_80': 0.15526838466373355},
            'F600LP': {'ee50': 0.07462507022703989, 'ee_80': 0.15720930232558142},
            'F763M': {'ee50': 0.07236761426978819, 'ee_80': 0.15524155844155846},
            'F775W': {'ee50': 0.0733488841694809, 'ee_80': 0.15742775742775744},
            'F814W': {'ee50': 0.09630835117773019, 'ee_80': 0.18683307332293292},
            'F845M': {'ee50': 0.09630835117773019, 'ee_80': 0.18683307332293292},
            'F850LP': {'ee50': 0.09630835117773019, 'ee_80': 0.18683307332293292},
            'F953N': {'ee50': 0.09630835117773019, 'ee_80': 0.18683307332293292}
        }

        self.hst_encircle_apertures_acs_wfc1_arcsec = {
            'F435W': {'ee50': 0.07552552552552552, 'ee_80': 0.15851063829787237},
            'F475W': {'ee50': 0.0750733137829912, 'ee_80': 0.15625000000000003},
            'F502N': {'ee50': 0.07514619883040935, 'ee_80': 0.15625000000000003},
            'F555W': {'ee50': 0.07529411764705882, 'ee_80': 0.1563829787234043},
            'F550M': {'ee50': 0.07544378698224852, 'ee_80': 0.15652173913043482},
            'F606W': {'ee50': 0.07582582582582582, 'ee_80': 0.15568181818181823},
            'F625W': {'ee50': 0.07615384615384616, 'ee_80': 0.15581395348837213},
            'F658N': {'ee50': 0.07640625000000001, 'ee_80': 0.15681818181818186},
            'F660N': {'ee50': 0.07648902821316614, 'ee_80': 0.15681818181818186},
            'F775W': {'ee50': 0.07888513513513513, 'ee_80': 0.16603773584905665},
            'F814W': {'ee50': 0.08079584775086505, 'ee_80': 0.17500000000000007},
            'F892N': {'ee50': 0.0914179104477612, 'ee_80': 0.22096774193548396},
            'F850LP': {'ee50': 0.09393939393939393, 'ee_80': 0.23529411764705882}
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

