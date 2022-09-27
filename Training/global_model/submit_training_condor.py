import htcondor
import argparse 

col = htcondor.Collector()
credd = htcondor.Credd()
credd.add_user_cred(htcondor.CredTypes.Kerberos, None)


parser = argparse.ArgumentParser()

parser.add_argument("--config", type=str, help="Config", required=True)
parser.add_argument("--model", type=str, help="Model .py", required=True)
args = parser.parse_args()

base_dir= "/afs/cern.ch/work/d/dvalsecc/private/Clustering_tools/DeepSuperCluster/Training/global_model/"

sub = htcondor.Submit()
sub['Executable'] = base_dir
sub["arguments"] = args.config +" "+args.model
sub['Error'] = base_dir+"/condor_logs/error/training-$(ClusterId).$(ProcId).err"
sub['Output'] = base_dir+"/condor_logs/output/training-$(ClusterId).$(ProcId).out"
sub['Log'] = base_dir+"/condor_logs/log/training-$(ClusterId).log"
sub['MY.SendCredential'] = True
sub['+JobFlavour'] = '"tomorrow"'
sub["transfer_input_files"] = "trainer_awk.py, awk_data.py, plot_loss.py"
sub["when_to_transfer_output"] = "ON_EXIT"
sub['request_cpus'] = '3'
sub['request_gpus'] = '1'

schedd = htcondor.Schedd()
with schedd.transaction() as txn:
    cluster_id = sub.queue(txn)
    print(cluster_id)

    
