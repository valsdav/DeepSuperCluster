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
parser.add_argument("--weta", type=float, nargs=2,  help="Window eta widths (barrel,endcap)", default=[0.3,0.3])
parser.add_argument("--wphi", type=float, nargs=2, help="Window phi widths (barrel, endcap)", default=[0.7,0.7])
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=15)
parser.add_argument("--redo", action="store_true", default=False, help="Redo all files")
args = parser.parse_args()


# Prepare condor jobs
condor = '''executable              = run_numpy_script.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = cluster_tonumpy_simple.py, windows_creator.py

+JobFlavour             = "{queue}"
queue arguments from arguments.txt

+AccountingGroup = "group_u_CMS.CAF.COMM"
'''

condor = condor.replace("{queue}", args.queue)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_96python3/x86_64-centos7-gcc8-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
WETA_EB=$4;
WETA_EE=$5;
WPHI_EB=$6;
WPHI_EE=$7;
MAXNOCALO=$8;

echo -e ">>> copy";
xrdcp --nopbar -f root://eos{eosinstance}.cern.ch/${INPUTFILE} input.root;

echo -e "Running numpy dumper.."

python cluster_tonumpy_simple.py -i input.root -o output.pkl --weta ${WETA_EB} ${WETA_EE}\
                     --wphi ${WPHI_EB} ${WPHI_EE} --maxnocalow ${MAXNOCALO};

echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output.pkl root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.pkl;

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

    arguments.append("{} {} {} {} {} {} {} {}".format(
            jobid,inputfile, args.outputdir, *args.weta, *args.wphi, args.maxnocalow))

print("Njobs: ", len(arguments))
    
with open("condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open("arguments.txt", "w") as args:
    args.write("\n".join(arguments))

with open("run_numpy_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




