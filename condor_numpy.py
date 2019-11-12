import sys
import os
import argparse
import random
from math import *

with open("command.txt", "w") as of:
    of.write(" ".join(["python"]+sys.argv))

'''
This scripts runs hadd on single crystal files to 
group them in strips reading a DOF file
'''
parser = argparse.ArgumentParser()

#parser.add_argument("-f", "--files", type=str, help="input file", required=True)
parser.add_argument("-i", "--inputdir", type=str, help="Inputdir", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
parser.add_argument("-e", "--eos", type=str, default="user", help="EOS instance user/cms", required=False)
parser.add_argument("--weta", type=int,  help="Window eta width", default=10)
parser.add_argument("--wphi", type=int,  help="Window phi width", default=20)
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=15)

parser.add_argument("--redo", action="store_true", default=False, help="Redo all files")
args = parser.parse_args()


# Prepare condor jobs
condor = '''executable              = run_numpy_script.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = cluster_tonumpy_v2.py

+JobFlavour             = "{queue}"
queue arguments from arguments.txt
'''

condor = condor.replace("{queue}", args.queue)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_95apython3/x86_64-centos7-gcc7-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
WETA=$4;
WPHI=$5;
MAXNOCALO=$6;


echo -e ">>> copy";
xrdcp --nopbar -f root://eos{eosinstance}.cern.ch/${INPUTFILE} input.root;

echo -e "Running numpy dumper.."

python cluster_tonumpy_v2.py -i input.root --weta ${WETA} --wphi ${WPHI} --maxnocalow ${MAXNOCALO};

echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  data_calo.root root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/data_calo_${JOBID}.root;
xrdcp -f --nopbar  data_nocalo.root root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/data_nocalo_${JOBID}.root;

echo -e "DONE";
'''

script = script.replace("{eosinstance}", args.eos)

arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

inputfiles = [ f for f in os.listdir(args.inputdir)]

jobid = 0
for ifile in inputfiles:
    jobid +=1
    inputfile = args.inputdir + "/" + ifile

    arguments.append("{} {} {} {} {} {}".format(
            jobid,inputfile, args.outputdir, args.weta, args.wphi, args.maxnocalow))

print("Njobs: ", len(arguments))
    
with open("condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open("arguments.txt", "w") as args:
    args.write("\n".join(arguments))

with open("run_numpy_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




