import os

# Path handling shortcuts

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TESTS_PATH = os.path.join(DATA_PATH, "test_data")
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
XTB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "xtb")
XTB_BINARY = os.path.join(XTB_PATH, "xtb-6.3.1", "bin", "xtb")

# Constants

EV_TO_HARTREE = (
    1 / 27.211386245988
)  # https://physics.nist.gov/cgi-bin/cuu/Value?hrev (04.06.21)
AU_TO_DEBYE = 1 / 0.3934303  # https://en.wikipedia.org/wiki/Debye (04.06.21)
ELEM_TO_ATOMNUM = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
ATOMNUM_TO_ELEM = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
    35: "Br",
    53: "I",
}
ATOM_ENERGIES_XTB = {
    "H": -0.393482763936,
    "C": -1.795110518041,
    "O": -3.769421097051,
    "N": -2.609452454630,
    "F": -4.619339964238,
    "S": -3.148271017078,
    "P": -2.377807088084,
    "Cl": -4.482525134961,
    "Br": -4.048339371234,
    "I": -3.779630263390,
}
