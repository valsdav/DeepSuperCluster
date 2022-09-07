import os
import argparse 
import awkward as ak

# source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

parser = argparse.ArgumentParser()
parser.add_argument("-i","--inputdir", type=str, help="Dataset directory",required=True)
args = parser.parse_args()

print(f"Finalizing awkward parquet dataset in folder: {args.inputdir}")

ak.to_parquet.dataset(args.inputdir)

# Getting the metadata
df = ak.from_parquet(args.inputdir, lazy=True)
with open(f"{args.inputdir}/dataset_metadata.txt","w") as o:
    o.write(str(df.type))

print("Done!")
