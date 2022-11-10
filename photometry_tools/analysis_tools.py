"""
Gather all photometric tools for HST and JWST photometric observations
"""


from photometry_tools import data_access


class AnalysisTools(data_access.DataAccess):
    """
    Access class to organize data structure of HST, NIRCAM and MIRI imaging data
    """
    def __init__(self, **kwargs):
        """

        """
        super().__init__(**kwargs)
