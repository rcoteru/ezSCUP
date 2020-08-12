"""
Exceptions for the ezSCUP package.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "v2.0"

class Error(Exception):
    """Base class for the exceptions in this package."""
    pass
   
#####################################################################

class RestartNotMatching(Error):
   """Raised when attempting to load an restart file that does not match the exisiting geometry."""
   pass

class NoSCUPExecutableDetected(Error):
   """Raised when attempting to access the output folder when it doesn't exist."""
   pass

class MissingSetup(Error):
   """Raised when a simulation run is attempted without the proper setup."""

class MissingRequiredArguments(Error):
   """Raised when some of the required arguments for a function are missing"""
   pass

class PreviouslyUsedOutputFolder(Error):
   """Raised when an output folder with the same name already exists."""
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


