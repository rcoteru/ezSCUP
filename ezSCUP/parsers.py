import re

# auxiliary funcs
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text 

def generate_lines_that_match(string, fp):
    for line in fp:
        if re.search(string, line):
            yield line

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

def parse_energy(fname:str, origin:str = "SP") -> dict:

    with open(fname, "r") as f:

        line = f.readline().strip()
        while (line != "Energy decomposition:"):
            line = f.readline()
            line = line.strip()

        energy = {} # in eV
        energy["reference"] = float(f.readline().strip().split()[3])
        energy["total_delta"] = float(f.readline().strip().split()[3])

        energy["lat_total_delta"] = float(f.readline().strip().split()[3])
        energy["lat_harmonic"] = float(f.readline().strip().split()[2])
        energy["lat_anharmonic"] = float(f.readline().strip().split()[2])
        energy["lat_elastic"] = float(f.readline().strip().split()[2])
        energy["lat_electrostatic"] = float(f.readline().strip().split()[2])

        energy["elec_total_delta"] = float(f.readline().strip().split()[3])
        energy["elec_one_electron"] = float(f.readline().strip().split()[2])
        energy["elec_two_electron"] = float(f.readline().strip().split()[2])
        energy["elec_electron_lat"] = float(f.readline().strip().split()[2])
        energy["elec_electrostatic"] = float(f.readline().strip().split()[2])

        energy["total_energy"] = float(f.readline().strip().split()[3])

    return energy