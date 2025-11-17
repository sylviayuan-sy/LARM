from glob import glob
import os

def prepare_assets():
    if not os.path.exists('./haven'):
        os.system("wget --no-verbose 1pb.skis.ltd/rendering/hdri.zip/hdri.zip")
        os.system("unzip hdri.zip")
    if not os.path.exists('./metadata'):
        os.system("wget --no-verbose 1pb.skis.ltd/rendering/meta.zip/meta.zip")
        os.system("unzip meta.zip")

prepare_assets()

paths = glob("./haven/hdris/*/*.hdr")

with open("./hdri.txt", "w") as f:
    for p in paths:
        f.write(p + "\n")
        
        
        