"""
Classes to handle writing output data files.
"""

__author__ = "Raúl Coterillo"
__email__  = "raulcote98@gmail.com"
__status__ = "Development"

# package imports
import ezSCUP.settings as cfg

#####################################################################
## MODULE STRUCTURE
#####################################################################

# func save_file()

#####################################################################
## FUNCTION DEFINITIONS
#####################################################################

def save_file(fname, headers, columns, sep=","):

    """
        
        Saves data in csv-like format.

        Parameters:
        ----------

        - fname  (string): name of the output file
        - headers (list): list of column headers
        - columns (list): list of arrays to save
        - sep (string): separator between values

    """

    if len(headers) != len(columns):
        print("WARNING: Error writing output file,")
        print("length of header/column does not match.")
        print(len(headers), "!=", len(columns))
        return 0

    with open(fname, "w") as f:
        
        for h in range(len(headers)):
            f.write(str(headers[h]))
            if h != len(headers)-1:
                    f.write(sep)
        f.write("\n")

        for row in range(len(columns[0])):    
            for col in range(len(columns)):
                f.write(str(columns[col][row]))
                if col != len(columns)-1:
                    f.write(sep)
            f.write("\n")
    
    print("File " + fname + " written successfully.")
