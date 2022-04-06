import os
import numpy as np
from os import listdir
from os.path import isfile, join
import glob
from pathlib import Path

data_path = Path(r"C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/rawdata/")
file_list = [str(pp) for pp in data_path.glob("**/*.mat")]

print(file_list)


