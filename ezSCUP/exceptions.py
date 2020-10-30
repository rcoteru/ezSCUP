"""
Exceptions for the ezSCUP package.
"""

class Error(Exception):
    """Base class for the exceptions in this package."""
    pass
   
#####################################################################

class AtomicIndexOutOfBounds(Error):
   """Raised when an invalid atomic index is given."""
   pass

class InvalidGeometryObject(Error):
   """Raised when an invalid geometry input is given."""
   pass

class NotEnoughPartials(Error):
   """Raised when no partial files were given to calculate the equilibrium geometry."""
   pass

class PositionsNotLoaded(Error):
   """Raised when attempting to access atomic position data when none was loaded."""
   pass

class GeometryNotMatching(Error):
   """Raised when attempting to load an restart file that does not match the exisiting geometry."""
   pass

class NoSCUPExecutableDetected(Error):
   """Raised when attempting to access the output folder when it doesn't exist."""
   pass

class InvalidParameterFile(Error):
   """Raised when attempting to load a non-existing parameter file."""
   pass

class MissingSetup(Error):
   """Raised when a simulation run is attempted without the proper setup."""

class PreviouslyUsedOutputFolder(Error):
   """Raised when an output folder with the same name already exists."""
   pass

class OutputFolderDoesNotExist(Error):
   """Raised when attempting to access the output folder when it doesn't exist."""
   pass

class InvalidFDFSetting(Error):
   """Raised when an invalid FDF setting is given."""
   pass

class InvalidMCConfiguration(Error):
   """Raised when an invalid combination of parameters is given."""
   pass

class InvalidLabelList(Error):
   """Raised when an invalid label list is provided."""
   pass



#####################################################################
#####################################################################


