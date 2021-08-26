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
parser.add_argument("-nfg", "--nfile-group", type=int, help="How many files per tfrecord file", required=True)
parser.add_argument("-tf", "--test-fraction", type=float, help="Fraction of files for testing", required=True)
parser.add_argument("-w", "--weights", type=str, help="Weights file", required=False)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-f", "--flag", type=str, help="Flag",required=True )
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
args = parser.parse_args()


# Prepare condor jobs
condor = '''executable              = run_tfrecord_script.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = ../convert_tfrecord_dataset_allinfo_jet.py {WEIGHTS}

+JobFlavour             = "{queue}"
queue arguments from arguments.txt

+AccountingGroup = "group_u_CMS.CAF.COMM"
'''

condor = condor.replace("{queue}", args.queue)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc10-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
FLAG=$4;


echo -e "Running tfrecord dumper.."

mkdir output;
python convert_tfrecord_dataset_allinfo_jet.py -i ${INPUTFILE} -o ./output -n records_$JOBID -f $FLAG {WEIGHTS};

echo -e "Copying result to: $OUTPUTDIR";
rsync -avz output/ ${OUTPUTDIR}

echo -e "DONE";
'''


arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)
    os.makedirs(args.outputdir +"/training")
    os.makedirs(args.outputdir +"/testing")

if args.weights:
    script = script.replace("{WEIGHTS}","-w "+args.weights)
    condor = condor.replace("{WEIGHTS}",", "+args.weights)
else:
    script = script.replace("{WEIGHTS}","")
    condor = condor.replace("{WEIGHTS}","")

inputfiles = [ f for f in os.listdir(args.inputdir) if 'tar.gz' in f]
ninputfiles = len(inputfiles)
template_inputfile = "clusters_data_{}.ndjson.tar.gz"


print("N input files: ", ninputfiles)

nfiles_testing = int( ninputfiles * args.test_fraction)
nfiles_training = ninputfiles - nfiles_testing
print("N. training files {}, N. testing files {}".format(nfiles_training, nfiles_testing))

jobid = 0
files_groups = []
ifile_used = 0
ifile_curr = 0

while ifile_used < nfiles_training:
    if(template_inputfile.format(ifile_curr) in inputfiles):
        files_groups.append(args.inputdir + "/" + template_inputfile.format(ifile_curr))
        ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/training", args.flag))
        files_groups = []
        ifile_group = 0

# print(files_groups)
print ("N files used for training: {}, Last id file used: {}".format(ifile_used+1, ifile_curr))

if len(files_groups)>0:
# Join also the last group
    arguments.append("{} {} {} {}".format(
                    jobid+1,"#_#".join(files_groups), args.outputdir +"/training",args.flag))


######## testing
ifile_group = 0
files_groups = []
ifile_used = 0
#ifile_curr from training cycle

while ifile_used < nfiles_testing:
    if(template_inputfile.format(ifile_curr) in inputfiles):
        files_groups.append(args.inputdir + "/" + template_inputfile.format(ifile_curr))
        ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/testing",args.flag))
        files_groups = []
        ifile_group = 0

print ("N files used for testing: {}, Last id file used: {}".format(ifile_used+1, ifile_curr))

if len(files_groups)>0:
    #join also the last group
    arguments.append("{} {} {} {}".format(
                    jobid+1,"#_#".join(files_groups), args.outputdir +"/testing",args.flag))

print("Njobs: ", len(arguments))
    
with open("condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open("arguments.txt", "w") as args:
    args.write("\n".join(arguments))

with open("run_tfrecord_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




