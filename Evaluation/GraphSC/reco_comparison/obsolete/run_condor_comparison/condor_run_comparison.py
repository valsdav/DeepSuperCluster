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
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("-a","--assoc-strategy", type=str, help="Association strategy", required=True)
parser.add_argument("--wp-file", type=str,  help="File with sim fraction thresholds")
parser.add_argument("-q", "--queue", type=str, help="Condor queue", default="longlunch", required=True)
parser.add_argument("-e", "--eos", type=str, default="user", help="EOS instance user/cms", required=False)
parser.add_argument('-c', "--compress", action="store_true",  help="Compress output")
parser.add_argument("--redo", action="store_true", default=False, help="Redo all files")
parser.add_argument("-d","--debug", action="store_true",  help="debug", default=False)
args = parser.parse_args()


# Prepare condor jobs
condor = '''executable              = run_reco_comparison_script.sh
output                  = output/strips.$(ClusterId).$(ProcId).out
error                   = error/strips.$(ClusterId).$(ProcId).err
log                     = log/strips.$(ClusterId).log
transfer_input_files    = ../run_reco_comparison.py, ../reco_comparison.py, ../calo_association.py, ../simScore_WP/{wp_file}, ../Mustache.C

+JobFlavour             = "{queue}"
queue arguments from arguments.txt

+AccountingGroup = "group_u_CMS.CAF.COMM"
'''

condor = condor.replace("{queue}", args.queue)
condor = condor.replace("{wp_file}", args.wp_file)

script = '''#!/bin/sh -e

source /cvmfs/sft.cern.ch/lcg/views/LCG_99/x86_64-centos7-gcc10-opt/setup.sh

JOBID=$1;  
INPUTFILE=$2;
OUTPUTDIR=$3;
ASSOC=$4;
WPFILE=$5;

echo -e "Running reco comparison dumper.."

python run_reco_comparison.py -i ${INPUTFILE} -o output_{type}.csv \
            -a ${ASSOC} --wp-file ${WPFILE} {debug};

{compress}
echo -e "Copying result to: $OUTPUTDIR";
xrdcp -f --nopbar  output_seeds.{output_ext} root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/seed_data_${JOBID}.{output_ext};
xrdcp -f --nopbar  output_event.{output_ext} root://eos{eosinstance}.cern.ch/${OUTPUTDIR}/event_data_${JOBID}.{output_ext};
echo -e "DONE";
'''

script = script.replace("{eosinstance}", args.eos)
if args.compress:
    script = script.replace("{compress}", 'tar -zcf output_seeds.csv.tar.gz output_seeds.csv && tar -zcf output_event.csv.tar.gz output_event.csv')
    script = script.replace("{output_ext}", 'csv.tar.gz')
else:
    script = script.replace("{compress}", '')
    script = script.replace("{output_ext}", 'csv')
if args.debug:
    script = script.replace("{debug}", "--debug")
else: 
    script = script.replace("{debug}", "")

arguments= []
if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)

inputfiles = [ f for f in os.listdir(args.inputdir)]
ninputfiles = len(inputfiles)
# template_inputfile = "cluster_job{}_step2_output.root"

wp_file = args.wp_file

print("N input files: ", ninputfiles)

jobid = 0
files_groups = []
ifile_used = 0
ifile_curr = 0


for file in inputfiles:
    files_groups.append(args.inputdir + "/" + file)
    ifile_used +=1 
    ifile_curr +=1

    if len(files_groups) == args.nfile_group:
        jobid +=1
        #join input files by ;
        arguments.append("{} {} {} {} {}".format(
                jobid,"#_#".join(files_groups), args.outputdir, args.assoc_strategy, wp_file))
        files_groups = []
        ifile_group = 0

# Join also the last group
if len(files_groups):
    arguments.append("{} {} {} {} {}".format(
                    jobid+1,"#_#".join(files_groups), args.outputdir, args.assoc_strategy,wp_file))


print("Njobs: ", len(arguments))
    
with open("condor_job.txt", "w") as cnd_out:
    cnd_out.write(condor)

with open("arguments.txt", "w") as args:
    args.write("\n".join(arguments))

with open("run_reco_comparison_script.sh", "w") as rs:
    rs.write(script)

#os.system("condor_submit condor_job.txt")




