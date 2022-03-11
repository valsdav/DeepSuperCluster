import os
import pandas as pd 
import glob
import gzip
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputdir", type=str, help="inputfolder", required=True)
parser.add_argument("-o","--outputfile", type=str, help="outputfile",required=True)
args = parser.parse_args()

                    
for ty in ["seed", "event"]:
    data = []
    for f in glob.glob(args.inputdir+"/{}_data*.tar.gz".format(ty)):
        print(f)
        with gzip.open(f, "rt") as file:
            data.append(pd.read_csv(file, sep=";"))

    A = pd.concat(data)

    store = pd.HDFStore(args.outputfile.replace("{type}", ty))
    store['df'] = A  # save it
    store.close()        
