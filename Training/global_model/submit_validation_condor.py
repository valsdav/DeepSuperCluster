import htcondor
import argparse
import os

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)


parser = argparse.ArgumentParser()
parser.add_argument("--basedir", type=str, help="Base dir", default=os.getcwd())
parser.add_argument("--model-config", type=str, help="Model configuration", required=True)
parser.add_argument("--model-weights", type=str, help="Model weights", required=True)
parser.add_argument("-o", "--outputdir", type=str, help="Outputdir", required=True)
args = parser.parse_args()

# Checking the input files exists
os.makedirs(args.basedir, exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs", exist_ok=True)
os.makedirs(f"{args.outputdir}", exist_ok=True)

if not os.path.exists(args.model_config):
    raise ValueError(f"Config file does not exists: {args.model_config}")

sub = htcondor.Submit()
sub['Executable'] = "run_validation_condor.sh"
sub["arguments"] = f"{args.model_config}  {args.model_weights} {args.outputdir}"
sub['Error'] = args.basedir+"/condor_logs/error/validation-$(ClusterId).$(ProcId).err"
sub['Output'] = args.basedir+"/condor_logs/output/validation-$(ClusterId).$(ProcId).out"
sub['Log'] = args.basedir+"/condor_logs/log/validation-$(ClusterId).log"
sub['MY.SendCredential'] = True
sub['+JobFlavour'] = '"microcentury"'
sub["transfer_input_files"] = "run_validation_dataset_awk.py, awk_data.py, loader_awk.py"
sub["when_to_transfer_output"] = "ON_EXIT"
sub['request_cpus'] = '3'
sub['request_gpus'] = '1'

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue(txn)
    print(cluster_id)
    # Saving the log
    with open(f"{args.basedir}/condor_logs/validation_jobs.csv", "a") as l:
        l.write(f"{cluster_id};{args.model_config};{args.model_weights};{args.basedir};{args.outputdir}\n")

    
