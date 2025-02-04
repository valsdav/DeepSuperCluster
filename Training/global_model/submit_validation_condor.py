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
parser.add_argument("--conf-overwrite", type=str, help="Validation config overwrite", required=False)
parser.add_argument("--gpu", action="store_true", help="Request GPU")
parser.add_argument("--dry", help="Try run, print commands", action="store_true")
parser.add_argument("--diff-model", action="store_true", help="Model with different outputs than baseline model")
parser.add_argument("--flavour", type=str, help='Choose dataset: ele1, gamma1, ele2, gamma2', required=False)
args = parser.parse_args()

# Checking the input files exists
os.makedirs(args.basedir, exist_ok=True)
os.makedirs(f"{args.basedir}/condor_logs", exist_ok=True)
os.makedirs(f"{args.outputdir}", exist_ok=True)

if not os.path.exists(args.model_config):
    raise ValueError(f"Config file does not exists: {args.model_config}")

sub = htcondor.Submit()
sub['Executable'] = "run_validation_condor.sh"
sub["arguments"] = f"{args.model_config}  {args.model_weights} {args.outputdir} {args.conf_overwrite} {args.flavour} {str(args.diff_model)}"
sub['Error'] = args.basedir+"/condor_logs/error/validation-$(ClusterId).$(ProcId).err"
sub['Output'] = args.basedir+"/condor_logs/output/validation-$(ClusterId).$(ProcId).out"
sub['Log'] = args.basedir+"/condor_logs/log/validation-$(ClusterId).log"
sub['MY.SendCredential'] = True
sub['+JobFlavour'] = '"longlunch"'
sub["transfer_input_files"] = "run_validation_dataset_awk.py, awk_data.py, loader_awk.py"
sub["when_to_transfer_output"] = "ON_EXIT"
sub['request_cpus'] = '3'
if args.gpu:
    sub['request_gpus'] = '1'

#if args.conf_overwrite:
    #sub["arguments"] += f' {args.basedir}/{args.conf_overwrite}'
    #sub["transfer_input_files"] += f", {args.basedir}/{args.conf_overwrite}"
#else:
    #sub["arguments"] += f' None'

if args.dry:
    print("run_validation_condor.sh "+ sub["arguments"])
    print("no job submitted")
    exit(0)

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue(txn)
    print(cluster_id)
    # Saving the log
    with open(f"{args.basedir}/condor_logs/validation_jobs.csv", "a") as l:
        l.write(f"{cluster_id};{args.model_config};{args.model_weights};{args.basedir};{args.outputdir}\n")

    
