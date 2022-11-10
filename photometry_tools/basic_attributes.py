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
                 'observed_bands': ['F275W', 'F336W', 'F435W', 'F555W', 'F814W']},
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

        self.dist_dict = {
            'ngc0628': {'dist': 10, 'dist_err': 1, 'method': 'TRGB'}
        }
        self.sr_per_square_deg = 0.00030461741978671  # steradians per square degree

