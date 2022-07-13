
import sys
import os
import argparse
import random
from math import *


'''
This scripts runs hadd on single crystal files to 
group them in strips reading a DOF file
'''
parser = argparse.ArgumentParser()

#parser.add_argument("-f", "--files", type=str, help="input file", required=True)
parser.add_argument("-i", "--inputdir", type=str, help="Inputdir", required=True)
parser.add_argument("-nfg", "--nfile-group", type=int, help="How many files per numpy file", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
parser.add_argument("-e", "--eos", type=str, default="user", help="EOS instance user/cms", required=False)
parser.add_argument('-c', "--compress", action="store_true",  help="Compress output")
parser.add_argument('-cf', '--condor-folder', type=str, default="condor_run")
args = parser.parse_args()


# Prepare condor jobs
condor = '''executable              = condor_script.sh
output                  = output/file.$(ClusterId).$(ProcId).out
error                   = error/file.$(ClusterId).$(ProcId).err
log                     = log/file.$(ClusterId).log
transfer_input_files    = ../calo_match_dataset.py, ../calo_association.py

+JobFlavour             = "{queue}"
queue arguments from arguments.txt

+AccountingGroup = "group_u_CMS.CAF.COMM"
'''

condor = condor.replace("{queue}", args.queue)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;

echo -e "Running reco comparison dumper.."

python calo_match_dataset.py -i ${INPUTFILE} -o output.csv;

{compress}
echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output.{finalext} root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/data_${JOBID}.{finalext};
echo -e "DONE";
'''

script = script.replace("{eosinstance}", args.eos)
if args.compress:
    script = script.replace("{compress}", 'tar -zcf output.csv.tar.gz output.csv')
    script = script.replace("{finalext}", 'csv.tar.gz')
else:
    script = script.replace("{compress}", '')
    script = script.replace("{finalext}", 'csv')
    
arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

#Setup  condor dirs
os.makedirs(f"./{args.condor_folder}/error", exist_ok=True)
os.makedirs(f"./{args.condor_folder}/log", exist_ok=True)
os.makedirs(f"./{args.condor_folder}/output", exist_ok=True)
with open(f"{args.condor_folder}/command.txt", "w") as of:
    of.write(" ".join(["python"]+sys.argv))


from glob import glob
    
inputfiles = glob(args.inputdir + "/**/**.root", recursive=True)
ninputfiles = len(inputfiles)
# template_inputfile = "cluster_job{}_step2_output.root"


print("N input files: ", ninputfiles)

jobid = 0
files_groups = []
ifile_used = 0
ifile_curr = 0


for file in inputfiles:
    files_groups.append(file)
    ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir))
        files_groups = []
        ifile_group = 0

# Join also the last group
if len(files_groups):
    arguments.append("{} {} {}".format(
                    jobid+1,"#_#".join(files_groups), args.outputdir))


print("Njobs: ", len(arguments))

with open( args.condor_folder + "/condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open( args.condor_folder + "/arguments.txt", "w") as argms:
    argms.write("\n".join(arguments))

with open( args.condor_folder + "/condor_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




