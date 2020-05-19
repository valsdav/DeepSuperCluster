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
parser.add_argument("-nfg", "--nfile-group", type=int, help="How many files per numpy file", required=True)
parser.add_argument("-tf", "--test-fraction", type=float, help="Fraction of files for testing", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-a","--assoc-strategy", type=str, help="Association strategy", required=True)
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
parser.add_argument("-e", "--eos", type=str, default="user", help="EOS instance user/cms", required=False)
parser.add_argument("--weta", type=float, nargs=2,  help="Window eta widths (barrel,endcap)", default=[0.3,0.3])
parser.add_argument("--wphi", type=float, nargs=2, help="Window phi widths (barrel, endcap)", default=[0.7,0.7])
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=15)
parser.add_argument("--min-et-seed", type=float,  help="Min Et of the seeds", default=1)
parser.add_argument("--redo", action="store_true", default=False, help="Redo all files")
args = parser.parse_args()


# Prepare condor jobs
condor = '''executable              = run_numpy_script.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = cluster_tonumpy_dynamic.py, windows_creator_dynamic.py, calo_association.py

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
ASSOC=$4;
MAXNOCALO=$5;
ET_SEED=$6;


echo -e "Running numpy dumper.."

python cluster_tonumpy_dynamic.py -i ${INPUTFILE} -o output.pkl \
            -a ${ASSOC} --maxnocalow ${MAXNOCALO} --min-et-seed ${ET_SEED};

echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output.pkl root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.pkl;

echo -e "DONE";
'''

script = script.replace("{eosinstance}", args.eos)

arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)
    os.makedirs(args.outputdir +"/training")
    os.makedirs(args.outputdir +"/testing")

inputfiles = [ f for f in os.listdir(args.inputdir)]
ninputfiles = len(inputfiles)
template_inputfile = "cluster_job{}_step2_output.root"

print("N input files: ", ninputfiles)

nfiles_testing = int( ninputfiles * args.test_fraction)
nfiles_training = ninputfiles - nfiles_testing
print("N. training files {}, N. testing files {}".format(nfiles_training, nfiles_testing))

jobid = 0
files_groups = []
ifile_used = 0
ifile_curr = 0

while ifile_used < nfiles_training:
    while (template_inputfile.format(ifile_curr) not in inputfiles): 
        ifile_curr +=1
    
    files_groups.append(args.inputdir + "/" + template_inputfile.format(ifile_curr))
    ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/training", args.assoc_strategy,
                args.maxnocalow, args.min_et_seed))
        files_groups = []
        ifile_group = 0

print ("N files used for training: {}, Last id file used: {}".format(ifile_used+1, ifile_curr))

# Join also the last group
arguments.append("{} {} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/training", args.assoc_strategy,
                args.maxnocalow, args.min_et_seed))


######## testing
ifile_group = 0
files_groups = []
ifile_used = 0
#ifile_curr from training cycle

while ifile_used < nfiles_testing:
    while (template_inputfile.format(ifile_curr) not in inputfiles): 
        ifile_curr +=1
    
    files_groups.append(args.inputdir + "/" + template_inputfile.format(ifile_curr))
    ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/testing", args.assoc_strategy,
                args.maxnocalow, args.min_et_seed))
        files_groups = []
        ifile_group = 0

print ("N files used for training: {}, Last id file used: {}".format(ifile_used+1, ifile_curr))

#join also the last group
arguments.append("{} {} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/testing", args.assoc_strategy,
                args.maxnocalow, args.min_et_seed))

print("Njobs: ", len(arguments))
    
with open("condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open("arguments.txt", "w") as args:
    args.write("\n".join(arguments))

with open("run_numpy_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




