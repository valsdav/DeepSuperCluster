import json
import random
import os
import numpy as np
import collections
import time
import sys
import glob
import gzip
import argparse 
import awkward as ak
from glob import glob

# source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

parser = argparse.ArgumentParser()
parser.add_argument("-n","--name", type=str, help="Job name", required=True)
parser.add_argument("-i","--inputdir", type=str, help="inputdir", required=True)
parser.add_argument("-o","--outputdir", type=str, help="Outputdirectory",required=True)
parser.add_argument("-g","--groupfiles", type=int, help="N. input file for each output file",required=True)
args = parser.parse_args()


def load_iter(files):
    for filename in files:
        with gzip.open(filename, "rt") as file:
            lines = []
            for line in file.readlines():
                line = line[line.rfind('{"window_index"'):]
                if "}" in line:
                    line = line.replace("NaN", "-999")
                    lines.append(line)
            yield "[{}]".format(",".join(lines))
                                   
   

    
if __name__ == "__main__":
    
    outputdir =args.outputdir

    os.makedirs(outputdir, exist_ok=True)

    inputfiles = glob(args.inputdir + "/**.ndjson.tar.gz", recursive=True)
    ninputfiles = len(inputfiles)
        
    print("Start reading files")

        
    ig = 0
    iG = 1
    arrays = []
    for file in load_iter(inputfiles):
        print("Processing file {}, in group {}".format(ig, iG))
        arr = ak.from_json(file)
        arrays.append(arr)
        ig +=1
        if ig == args.groupfiles:
            totA = ak.concatenate(arrays)
            ak.to_parquet(totA, args.outputdir + "/"+args.name + "_{}.parquet".format(iG))
            iG+=1
            ig = 0
            arrays.clear()
        

    if len(arrays):
        totA = ak.concatenate(arrays)
        ak.to_parquet(totA, args.outputdir + "/"+args.name + "_{}.parquet".format(iG))
        arrays.clear()
        
