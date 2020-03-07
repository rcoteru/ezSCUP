"""
Exception for the ezSCUP package.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

class Error(Exception):
    """Base class for exceptions in this package."""
    pass
   
#####################################################################

class NoSCUPExecutableDetected(Error):
   """Raised when attempting to access the output folder when it doesn't exist."""
   pass

class OutputFolderDoesNotExist(Error):
   """Raised when attempting to access the output folder when it doesn't exist."""
   pass

class InvalidFDFSetting(Error):
   """Raised when an invalid FDF setting is given."""
   pass

class InvalidCell(Error):
   """Raised when an invalid Cell is given."""
   pass

class InvalidLabel(Error):
   """Raised when an invalid element label is given."""
   pass

class InvalidLabelList(Error):
   """Raised when an invalid element label is given."""
   pass

class InvalidMCConfiguration(Error):
   """Raised when an invalid MCConfiguration is given."""
   pass



#####################################################################
#####################################################################


