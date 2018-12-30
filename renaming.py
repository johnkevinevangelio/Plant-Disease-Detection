import os
from os import listdir
from os.path import isfile, join

source = "/home/pi/Desktop/virtualenvs/PD/PlantDiseaseDetection/trainadd"


files = [f for f in listdir(source) if isfile(join(source, f))]
print(len(files))
for i, n in zip(files,range(len(files))):
    old_file = join(source, i)
    new_file = join(source,"h (100%s).jpg" %n)
    os.rename(old_file, new_file)
