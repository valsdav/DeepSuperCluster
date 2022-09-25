'''
This script extract the normalization factors from
the awkward datasets and save them in json format for easy use. 
'''
import argparse
import awkward as ak
from awk_data import default_features_dict

parser = argparse.ArgumentParser()
parser.add_argument("-i","--input-folders", type=str, nargs="+", help="List of folders with parquet dataset to use", required=True)
parser.add_argument("-n","--n-samples", type=int, help="Number of sample to use", default=1000000)
parser.add_argument("-o", "--output-file", type=str, required=True)
args = parser.parse_args()


df_tot = ak.concatenate([ak.from_parquet(folder, lazy=True, use_threads=True)[0:args.n_samples]
                         for folder in args.input_folders])

norm_factor = { "cluster" : { "mean" : {}, "max": {}, "min": {}, "std": {}},
               "window":  { "mean" : {}, "max": {}, "min": {}, "std": {}}}

for cl in default_features_dict["cl_features"]:
    print(cl)
    norm_factor["cluster"]["mean"][cl] = ak.mean(df_tot.cl_features[cl])
    norm_factor["cluster"]["std"][cl] = ak.std(df_tot.cl_features[cl])
    norm_factor["cluster"]["min"][cl] = ak.min(df_tot.cl_features[cl])
    norm_factor["cluster"]["max"][cl] = ak.max(df_tot.cl_features[cl])
    
for wi in default_features_dict["window_features"]:
    print(wi)
    norm_factor["window"]["mean"][wi] = ak.mean(df_tot.window_features[wi])
    norm_factor["window"]["std"][wi] = ak.std(df_tot.window_features[wi])
    norm_factor["window"]["min"][wi] = ak.min(df_tot.window_features[wi])
    norm_factor["window"]["max"][wi] = ak.max(df_tot.window_features[wi])

# Save the output in json record format
norm_fact_awk = ak.Record(norm_factor)
ak.to_json(norm_fact_awk, args.output_file, pretty=True)
