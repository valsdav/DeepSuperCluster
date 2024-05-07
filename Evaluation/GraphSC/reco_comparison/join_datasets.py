import os
import pandas as pd 
import glob
import gzip
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputdir", type=str, help="inputfolder", required=True)
parser.add_argument("-o","--outputfile", type=str, help="outputfile",required=True)
parser.add_argument("-n","--number", type=int, help="number of files to merge", default=-1)
args = parser.parse_args()

                    
for ty in ["seed", "event","object"]:
    data = []
    ifile = 0
    for f in glob.glob(args.inputdir+"/{}_data*.tar.gz".format(ty)):
        if args.number > 0 and ifile >= args.number: break
        ifile += 1
        print(f)
        with gzip.open(f, "rt") as file:
            def convert_to_list(val):
                if val == "":
                    return []
                else:
                    return ast.literal_eval(val)

            # Load the DataFrame, applying the conversion function to the 'Col3' column
            df = pd.read_csv(file, sep=";", converters={
                'ele_clsAdded_eta': convert_to_list,
                'ele_clsAdded_phi': convert_to_list,
                'ele_clsAdded_energy': convert_to_list,
                'ele_clsRemoved_eta': convert_to_list,
                'ele_clsRemoved_phi': convert_to_list,
                'ele_clsRemoved_energy': convert_to_list
            })
            data.append(df)
    if len(data)==0: continue
    A = pd.concat(data)

    store = pd.HDFStore(args.outputfile.replace("{type}", ty))
    store['df'] = A  # save it
    store.close()        
