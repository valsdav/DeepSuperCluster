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
parser.add_argument("--wp-file", type=str,  help="File with sim fraction thresholds")
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
parser.add_argument("-e", "--eos", type=str, default="user", help="EOS instance user/cms", required=False)
parser.add_argument("--weta", type=float, nargs=2,  help="Window eta widths (barrel,endcap)", default=[0.3,0.3])
parser.add_argument("--wphi", type=float, nargs=2, help="Window phi widths (barrel, endcap)", default=[0.7,0.7])
parser.add_argument("--maxnocalow", type=int,  help="Number of no calo window per event", default=15)
parser.add_argument("--min-et-seed", type=float,  help="Min Et of the seeds", default=1)
parser.add_argument("-ov","--overlap", action="store_true",  help="Overlapping window mode", default=False)
parser.add_argument("--pu-limit", type=float,  help="SimEnergy PU limit", default=1e6)
parser.add_argument('-c', "--compress", action="store_true",  help="Compress output")
parser.add_argument("--redo", action="store_true", default=False, help="Redo all files")
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
args = parser.parse_args()


# Prepare condor jobs
condor = '''executable              = run_ndjson_script.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = ../cluster_ndjson_dynamic_general.py, ../windows_creator_general.py, ../calo_association.py, ../simScore_WP/{wp_file}, ../Mustache.C

+JobFlavour             = "{queue}"
queue arguments from arguments.txt

+AccountingGroup = "group_u_CMS.CAF.COMM"
'''

condor = condor.replace("{queue}", args.queue)
condor = condor.replace("{wp_file}", args.wp_file)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_101/x86_64-centos7-gcc11-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
ASSOC=$4;
WPFILE=$5;
MAXNOCALO=$6;
ET_SEED=$7;
OVERLAP=$8;
PULIM=$9;


echo -e "Running ndjson dumper.."

python cluster_ndjson_dynamic_global_nooverlap.py -i ${INPUTFILE} -o output.ndjson \
            -a ${ASSOC} --wp-file ${WPFILE} --min-et-seed ${ET_SEED} --maxnocalow $MAXNOCALO \
           --overlap ${OVERLAP} --pu_limit ${PULIM} {debug};

{compress}
echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output.{output_ext} root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.{output_ext};
xrdcp -f --nopbar  output.meta.csv root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/clusters_data_${JOBID}.meta.csv;

echo -e "DONE";
'''

script = script.replace("{eosinstance}", args.eos)
if args.compress:
    script = script.replace("{compress}", 'tar -zcf output.ndjson.tar.gz output.ndjson')
    script = script.replace("{output_ext}", 'ndjson.tar.gz')
else:
    script = script.replace("{compress}", '')
    script = script.replace("{output_ext}", 'ndjson')
if args.debug:
    script = script.replace("{debug}", "--debug")
else: 
    script = script.replace("{debug}", "")

arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)
    os.makedirs(args.outputdir +"/training")
    os.makedirs(args.outputdir +"/testing")

inputfiles = [ f for f in os.listdir(args.inputdir)]
ninputfiles = len(inputfiles)
# template_inputfile = "cluster_job{}_step2_output.root"

wp_file = os.path.split(args.wp_file)[1]

print("N input files: ", ninputfiles)

nfiles_testing = int( ninputfiles * args.test_fraction)
nfiles_training = ninputfiles - nfiles_testing
print("N. training files {}, N. testing files {}".format(nfiles_training, nfiles_testing))

jobid = 0
files_groups = []
ifile_used = 0
ifile_curr = 0

files_training = inputfiles[:nfiles_training]
files_testing = inputfiles[nfiles_training:]

for file in files_training:
    files_groups.append(args.inputdir + "/" + file)
    ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {} {} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/training", args.assoc_strategy, wp_file,
                args.maxnocalow, args.min_et_seed, args.overlap, args.pu_limit))
        files_groups = []
        ifile_group = 0

print ("N files used for training: {}, Last id file used: {}".format(ifile_used+1, ifile_curr))

# Join also the last group
if len(files_groups):
    arguments.append("{} {} {} {} {} {} {} {} {}".format(
                    jobid+1,"#_#".join(files_groups), args.outputdir +"/training", args.assoc_strategy,wp_file,
                    args.maxnocalow, args.min_et_seed, args.overlap, args.pu_limit))


# ######## testing

files_groups = []
ifile_used = 0
ifile_curr = 0

for file in files_testing:
    files_groups.append(args.inputdir + "/" + file)
    ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir +"/testing", args.assoc_strategy, wp_file,
                args.maxnocalow, args.min_et_seed,args.overlap, args.pu_limit))
        files_groups = []
        ifile_group = 0

print ("N files used for testing: {}, Last id file used: {}".format(ifile_used+1, ifile_curr))

# Join also the last group
if len(files_groups):
    arguments.append("{} {} {} {} {} {} {} {}".format(
                jobid+1,"#_#".join(files_groups), args.outputdir +"/testing", args.assoc_strategy,wp_file,
                args.maxnocalow, args.min_et_seed,args.overlap, args.pu_limit))


print("Njobs: ", len(arguments))
    
with open("condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open("arguments.txt", "w") as args:
    args.write("\n".join(arguments))

with open("run_ndjson_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




