#!/usr/bin/env python
# coding: utf-8

import awkward as ak
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import numpy as np
import pandas as pd
from coffea.util import load
import hist
import os
from pprint import pprint 
hep.style.use(hep.style.ROOT)
import vector
vector.register_awkward()
import sys

import numba
from collections.abc import Iterable
from coffea import processor
from coffea.processor import accumulate
from coffea.processor.accumulator import column_accumulator
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from distributed import Client


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-o","--output-folder", type=str)
parser.add_argument("--input-deepsc", type=str)
parser.add_argument("--input-mustache", type=str)
parser.add_argument("--n-workers", type=int)
args = parser.parse_args()

output_folder = args.output_folder
os.makedirs(output_folder,exist_ok=True)
input_folder_must = args.input_mustache
input_folder_deep = args.input_deepsc


@numba.njit
def genparticle_mother_pdgId(genparticle_mother_index, genparticle_pdgid,genparticle_status, builder):
    for mother_index,pdgId, status  in zip(genparticle_mother_index, genparticle_pdgid,genparticle_status):
        builder.begin_list()
        
        for i in range(len(pdgId)):
            if status[i]!=1 or abs(pdgId[i])!=11: continue
            builder.begin_record()
            builder.field('index').append(i)
            genpart_i = i
            while True:
                if mother_index[genpart_i] == -1:
                    builder.field("pdgId").append(pdgId[genpart_i])
                    break
                else:
                    genpart_i = mother_index[genpart_i]
            builder.end_record()
        builder.end_list()
    return builder


@numba.njit
def get_matching_pairs_indices(idx_1, idx_2, builder, builder2):
    for ev_q, ev_j in zip(idx_1, idx_2):
        builder.begin_list()
        builder2.begin_list()
        q_done = []
        j_done = []
        for i, (q, j) in enumerate(zip(ev_q, ev_j)):
            if q not in q_done:
                if j not in j_done:
                    builder.append(i)
                    q_done.append(q)
                    j_done.append(j)
                else:
                    builder2.append(i)
        builder.end_list()
        builder2.end_list()
    return builder, builder2


# This function takes as arguments the indices of two collections of objects that have been
# previously matched. The idx_matched_obj2 indices are supposed to be ordered but they can have missing elements.
# The idx_matched_obj indices are matched to the obj2 ones and no order is required on them.
# The function return an array of the dimension of the maxdim_obj2 (akward dimensions) with the indices of idx_matched_obk
# matched to the elements in idx_matched_obj2. None values are included where
# no match has been found.
@numba.njit
def get_matching_objects_indices_padnone(
    idx_matched_obj, idx_matched_obj2, maxdim_obj2, deltaR, builder, builder2, builder3
):
    for ev1_match, ev2_match, nobj2, dr in zip(
        idx_matched_obj, idx_matched_obj2, maxdim_obj2, deltaR
    ):
        # print(ev1_match, ev2_match)
        builder.begin_list()
        builder2.begin_list()
        builder3.begin_list()
        row1_length = len(ev1_match)
        missed = 0
        for i in range(nobj2):
            # looping on the max dimension of collection 2 and checking if the current index i
            # is matched, e.g is part of ev2_match vector.
            if i in ev2_match:
                # if this index is matched, then take the ev1_match and deltaR results
                # print(i, row1_length)
                builder2.append(i)
                if i - missed < row1_length:
                    builder.append(ev1_match[i - missed])
                    builder3.append(dr[i - missed])
            else:
                # If it is missing a None is added and the missed  is incremented
                # so that the next matched one will get the correct element assigned.
                builder.append(None)
                builder2.append(None)
                builder3.append(None)
                missed += 1
        builder.end_list()
        builder2.end_list()
        builder3.end_list()
    return builder, builder2, builder3


def metric_pt(obj, obj2):
    return abs(obj.pt - obj2.pt)


def get_unique_match(obj1, obj2, deltaRmax=0.2):
    a, b = ak.unzip(
       ak.cartesian([obj1, obj2], axis=1, nested=True)
        )
    deltaR = ak.flatten(a.deltaR(b), axis=2)
    # Keeping only the pairs with a deltaR min
    maskDR = deltaR < deltaRmax

    # Get the indexing to sort the pairs sorted by deltaR without any cut
    idx_pairs_sorted = ak.argsort(deltaR, axis=1)
    pairs = ak.argcartesian([obj1, obj2])
    # Sort all the collection over pairs by deltaR
    pairs_sorted = pairs[idx_pairs_sorted]
    deltaR_sorted = deltaR[idx_pairs_sorted]
    maskDR_sorted = maskDR[idx_pairs_sorted]
    idx_obj, idx_obj2 = ak.unzip(pairs_sorted)

    # Now get only the matching indices by looping over the pairs in order of deltaR.
    # The result contains the list of pairs that are considered valid
    _idx_matched_pairs, _idx_missed_pairs = get_matching_pairs_indices(
        ak.without_parameters(idx_obj, behavior={}),
        ak.without_parameters(idx_obj2, behavior={}),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
    )

    idx_matched_pairs = _idx_matched_pairs.snapshot()
    # The indices related to the invalid jet matches are skipped
    # idx_missed_pairs  = _idx_missed_pairs.snapshot()
    # Now let's get get the indices of the objects corresponding to the valid pairs
    idx_matched_obj = idx_obj[idx_matched_pairs]
    idx_matched_obj2 = idx_obj2[idx_matched_pairs]
    # Getting also deltaR and maskDR of the valid pairs
    deltaR_matched = deltaR_sorted[idx_matched_pairs]
    maskDR_matched = maskDR_sorted[idx_matched_pairs]



    # We get the indices needed to reorder the second collection
    # and we use them to re-order also the other collection (same dimension of the valid pairs)
    obj2_order = ak.argsort(idx_matched_obj2)
    idx_obj_obj2sorted = idx_matched_obj[obj2_order]
    idx_obj2_obj2sorted = idx_matched_obj2[obj2_order]
    deltaR_obj2sorted = deltaR_matched[obj2_order]
    maskDR_obj2sorted = maskDR_matched[obj2_order]
    # Here we apply the deltaR + pT requirements on the objects and on deltaR
    idx_obj_masked = idx_obj_obj2sorted[maskDR_obj2sorted]
    idx_obj2_masked = idx_obj2_obj2sorted[maskDR_obj2sorted]
    # Getting also the deltaR of the masked pairs
    deltaR_masked = deltaR_obj2sorted[maskDR_obj2sorted]
    # N.B. We are still working only with indices not final objects

    # Now we have the object in the collection 1 ordered as the collection 2,
    # but we would like to have an ak.Array of the dimension of the collection 2, with "None"
    # in the places where there is not matching.
    # We need a special function for that, building the ak.Array of object from collection 1, with the dimension of collection 2, with None padding.
    (
        _idx_obj_padnone,
        _idx_obj2_padnone,
        _deltaR_padnone,
    ) = get_matching_objects_indices_padnone(
        ak.without_parameters(idx_obj_masked, behavior={}),
        ak.without_parameters(idx_obj2_masked, behavior={}),
        ak.without_parameters(ak.num(obj2), behavior={}),
        ak.without_parameters(deltaR_masked, behavior={}),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
        ak.ArrayBuilder(),
    )
    idx_obj_padnone = _idx_obj_padnone.snapshot()
    idx_obj2_padnone = _idx_obj2_padnone.snapshot()
    deltaR_padnone = _deltaR_padnone.snapshot()

    # Finally the objects are sliced through the padded indices
    # In this way, to a None entry in the indices will correspond a None entry in the object
    matched_obj = obj1[idx_obj_padnone]
    matched_obj2 = obj2[idx_obj2_padnone]

    return (
        matched_obj,
        matched_obj2,
        deltaR_padnone,
        idx_obj_padnone,
        idx_obj2_padnone,
        deltaR_masked
    )



class GenMatchingProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        dataset = events.metadata['dataset']
        file = events.metadata["filename"]
        
        g = file.split("/")
        runId = int("1" + g[-2] + "_" + g[-1].replace("output_","").replace(".root","") )
        
        vector.register_awkward()
     
        gen_mask_ele = (events.genParticle_status==1)&(abs(events.genParticle_pdgId)==11)&(events.genParticle_statusFlag&1==1)
        genParticles = ak.with_name(ak.zip({
                "pt": events.genParticle_pt[gen_mask_ele],
                "eta": events.genParticle_eta[gen_mask_ele],
                "phi": events.genParticle_phi[gen_mask_ele],
                "M": ak.zeros_like(events.genParticle_pt[gen_mask_ele])
            }), name="Momentum4D")

        sc = ak.with_name(ak.zip({
            "pt": events.superCluster_energy / np.cosh(events.superCluster_eta),
            "eta": events.pfCluster_eta[events.superCluster_seedIndex],  # seed Eta
            "phi": events.pfCluster_phi[events.superCluster_seedIndex],
            "E": events.superCluster_energy,
        }), name="Momentum4D")
        
        
        electrons = ak.with_name(ak.zip({
            "pt": events.patElectron_pt,
            "eta": events.patElectron_eta,
            "phi": events.patElectron_phi,
            "M": ak.zeros_like(events.patElectron_pt),
        }), name="Momentum4D")
        
        matched_sc, matched_genpart_sc, deltaR_sc, idx_sc, idx_genpart_sc, deltaR_masked_sc = get_unique_match(sc, genParticles, deltaRmax=0.2)
        matched_ele, matched_genpart_ele, deltaR_ele, idx_ele, idx_genpart_ele, deltaR_masked_ele = get_unique_match(electrons, genParticles, deltaRmax=0.2)
        is_genmatched_sc = ~ak.is_none(matched_genpart_sc, axis=1)
        is_genmatched_ele = ~ak.is_none(matched_genpart_ele, axis=1)
        
        
        genParticle_fields = ["genParticle_eta","genParticle_phi","genParticle_pt","genParticle_pdgId"]
        patElectron_fields = [f for f in events.fields if f.startswith("patElectron") if f not in ["patElectron_overlapPhotonIndices"]]
        SC_fields = [f for f in events.fields if f.startswith("superCluster") and f not in ["superCluster_nXtals","superCluster_psCluster_energy",
                                                                                        "superCluster_psCluster_eta","superCluster_psCluster_phi"]]
        
        output = {}

        output["N_genmatched_sc"] =  column_accumulator(ak.to_numpy(
                                 ak.sum(is_genmatched_sc, axis=1), allow_missing=False))

        output["N_genmatched_ele"] =  column_accumulator(ak.to_numpy(
                                 ak.sum(is_genmatched_ele, axis=1), allow_missing=False))
        
        output["N_nongenmatched_sc"] =  column_accumulator(ak.to_numpy(
                                 ak.sum(~is_genmatched_sc, axis=1), allow_missing=False))

        output["N_nongenmatched_ele"] =  column_accumulator(ak.to_numpy(
                                 ak.sum(~is_genmatched_ele, axis=1), allow_missing=False))

        for k in ["obsPU", "truePU", "nVtx", "eventId", ]:
            data_struct = ak.ones_like(idx_genpart_sc[is_genmatched_sc])
            output[k+"_sc"] = column_accumulator(ak.to_numpy(
                ak.flatten(events[k]*data_struct), allow_missing=False))
            output["runId_sc"] = column_accumulator(ak.to_numpy(
                ak.flatten(runId*data_struct), allow_missing=False))
            
            data_struct = ak.ones_like(idx_genpart_ele[is_genmatched_ele])
            output[k+"_ele"] = column_accumulator(ak.to_numpy(
                ak.flatten(events[k]*data_struct), allow_missing=False))
            output["runId_ele"] = column_accumulator(ak.to_numpy(
                ak.flatten(runId*data_struct), allow_missing=False))
            
            
        for k in genParticle_fields:
            output[k + "_sc"] =  column_accumulator(ak.to_numpy(
                        ak.flatten(
                            events[k][gen_mask_ele][idx_genpart_sc[is_genmatched_sc]], axis=None),
                        allow_missing=False))  
            output[k + "_ele"] =  column_accumulator(ak.to_numpy(
                        ak.flatten(
                            events[k][gen_mask_ele][idx_genpart_ele[is_genmatched_ele]], axis=None),
                        allow_missing=False))  
            
            # Saving also the non matched genParticle
            output[k +"_notmatched_sc"] = column_accumulator(ak.to_numpy(
                    ak.flatten(events[k][gen_mask_ele][~is_genmatched_sc], axis=None
                    ), allow_missing=False))
            
            output[k +"_notmatched_ele"] = column_accumulator(ak.to_numpy(
                    ak.flatten(events[k][gen_mask_ele][~is_genmatched_sc], axis=None
                    ), allow_missing=False))
        

        output["genParticle_index_sc"] = column_accumulator(ak.to_numpy(
                        ak.flatten(idx_genpart_sc[is_genmatched_sc], axis=None), allow_missing=False))
        output["genParticle_index_ele"] = column_accumulator(ak.to_numpy(
                        ak.flatten(idx_genpart_ele[is_genmatched_ele], axis=None), allow_missing=False))

        

        for k in SC_fields:
            output[k] =  column_accumulator(ak.to_numpy(
                        ak.flatten(
                            events[k][idx_sc[is_genmatched_sc]], axis=None),
                        allow_missing=False))

        for k in patElectron_fields:
            output[k] =  column_accumulator(ak.to_numpy(
                        ak.flatten(
                            events[k][idx_ele[is_genmatched_ele]], axis=None),
                        allow_missing=False))
    

        return {
            dataset: output
                
        }

    def postprocess(self, accumulator):
        return accumulator


# Dask cluster

from dask_lxplus import CernCluster
import socket
n_port = 8790

log_folder = os.getcwd()+"/condor_log"
print("Starting Dask cluster")
# # Creating a CERN Cluster, special configuration for dask-on-lxplus
# cluster = CernCluster(
#     cores=1,
#     memory="2000MB",
#     disk="5GB",
#     image_type="singularity",
#     worker_image="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/batch-team/dask-lxplus/lxdask-cc7:latest",
#     death_timeout="3600",
#     scheduler_options={"port": n_port, "host": socket.gethostname()},
#     log_directory = log_folder,
#     # shared_temp_directory="/tmp"
#     job_extra={
#         "log": f"{log_folder}/dask_job_output.log",
#         "output": f"{log_folder}/dask_job_output.out",
#         "error": f"{log_folder}/dask_job_output.err",
#         "should_transfer_files": "Yes",
#         "when_to_transfer_output": "ON_EXIT",
#         "+JobFlavour": f'"microcentury"'
#     },
#     env_extra=["source /afs/cern.ch/work/d/dvalsecc/private/Clustering_tools/DeepSuperCluster/myenv/bin/activate"],
# )
from dask_jobqueue import SLURMCluster, HTCondorCluster
cluster = SLURMCluster(
                queue="short",
                cores=1,
                processes=1,
                memory="3GB",
                walltime="01:00:00",
                env_extra=[f"source {sys.prefix}/bin/activate"],
                local_directory=f"{os.getcwd()}/logs",
            )



# cluster.scale(args.n_workers)
# client = Client(cluster)
# print("Waiting for the first job to start")
# client.wait_for_workers(1)
# print("Ready to start")

# In[9]:


from glob import glob
fileset = {
    "DeepSC": glob(input_folder_deep+"/*/*.root", recursive=True),
    "Mustache": glob(input_folder_must+"/*/*.root", recursive=True)
}

output = processor.run_uproot_job(fileset,
                                  treename="recosimdumper/caloTree",
                                  processor_instance=GenMatchingProcessor(),
                                  executor= processor.iterative_executor,
                                  executor_args={
                                      #'client': client,
                                      'schema': BaseSchema,
                                  },
                                  chunksize=200,
                                )

# iterative_run = processor.Runner(
#             executor = processor.DaskExecutor(
#                 client=client,
#             ),
#             schema=BaseSchema,
#             chunksize=200,
#         )

# out = iterative_run(
#     fileset,
#     treename=
#     processor_instance=GenMatchingProcessor(),
# )

from coffea.util import save, load
save(out, f"{output_folder}/output_genmatching.coffea")


cluster.close()




