from os import listdir
from os.path import isfile, join

data_path = "C:/Users/juanr/Desktop/Universidad/Master/Practicas/programa calculo t1t2/NMR-t1-t2-calculator/data"

onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

def path_file_concatenate_strings(i):
    path = "./data/"
    filename = onlyfiles[i]
    str = path + filename
    return str

path_0 = path_file_concatenate_strings(0)
print(path_0)
print(len(onlyfiles))
