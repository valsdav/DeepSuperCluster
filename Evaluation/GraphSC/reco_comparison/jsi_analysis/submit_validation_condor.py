import htcondor
import argparse
import os
import json

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)

parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=str, help="Base dir", default=os.getcwd())
parser.add_argument("--cmssw-tar", type=str, help="CMSSW tar", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
parser.add_argument("--dataset", type=str, help="Dataset", required=False)
parser.add_argument("-p", "--proxy", type=str, help="Proxy key", required=False)
parser.add_argument("--dry", help="Try run, print commands", action="store_true")
parser.add_argument("--limit", help="Limit number of files", type=int)
args = parser.parse_args()

# Checking the input files exists
os.makedirs(args.basedir, exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs", exist_ok=True)
os.makedirs(f"{args.outputdir}", exist_ok=True)

files = [ ]
command = f'dasgoclient -json -query="file dataset={args.dataset} site=T0_CH_CERN_Disk"'
print(f"Executing query: {command}")
filesjson = json.loads(os.popen(command).read())
for fj in filesjson:
    f = fj["file"][0]
    files.append(f['name'])
if args.limit:
    files = files[:args.limit]

script = """#!/bin/sh -e
export X509_USER_PROXY={PROXYKEY}
voms-proxy-info

cp -r {CMSSWDIR}.tar.gz ./DeepSC_JPsi_12_5_0.tar.gz
tar -xzvf DeepSC_JPsi_12_5_0.tar.gz
cd DeepSC_JPsi_12_5_0/src
echo -e "evaluate"
eval `scramv1 ru -sh`

JOBID=$1;  
INPUTFILE=$2;
cd RecoSimStudies/Dumpers/crab/

cmsRun MiniAOD_fromRaw_Run3_rereco_DeepSC_algoA_Data2022_cfg.py inputFile=${INPUTFILE} outputFile=../../../output.root

cd ../../..;
cd Bmmm/Analysis/test/;
python3 inspector_bmmm_analysis.py --inputFiles ../../../output.root --filename output_${JOBID}.root

xrcdp output_${JOBID}.root {OUTPUTDIR};

"""

script = script.replace("{CMSSWDIR}", args.cmssw_tar)
script = script.replace("{PROXYKEY}", args.proxy)
script = script.replace("{OUTPUTDIR}", args.outputdir)

with open("run_jpsi.sh", "w") as o:
    o.write(script)


sub = htcondor.Submit()
sub['Executable'] = "run_jpsi.sh"
sub['Error'] = args.basedir+"/condor_logs/error/deepsc-jpsi-$(ClusterId).$(ProcId).err"
sub['Output'] = args.basedir+"/condor_logs/output/deepsc-jpsi-$(ClusterId).$(ProcId).out"
sub['Log'] = args.basedir+"/condor_logs/log/deepsc-jpsi-$(ClusterId).log"
sub['MY.SendCredential'] = True
sub['+JobFlavour'] = '"tomorrow"'
sub['+AccountingGroup'] = "group_u_CMS.CAF.COMM"
sub["when_to_transfer_output"] = "ON_EXIT"
sub['request_cpus'] = '2'


log =  open(f"{args.basedir}/condor_logs/deepsc_jpsi_jobs.csv", "a")

if args.dry:
    with open(f"{args.basedir}/args.txt", "w") as a:
        for i, file in enumerate(files):
            a.write(f"{i} {file}\n")
        exit(0)
    
schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    for i, file in enumerate(files):
        sub["arguments"] = f"{i} {file}"
        cluster_id = sub.queue(txn)
        print(cluster_id)
        # Saving the log
        log.write(f"{cluster_id};{i};{file}\n")


log.close()
