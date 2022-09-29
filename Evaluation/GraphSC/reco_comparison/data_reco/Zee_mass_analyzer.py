import awkward as ak
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import mplhep as hep
import numpy as np
import pandas as pd
import hist
import os
from pprint import pprint 
from glob import glob
hep.style.use(hep.style.ROOT)

from coffea import processor
from coffea.nanoevents.methods import candidate
from coffea.nanoevents import NanoEventsFactory, BaseSchema
from coffea.processor import column_accumulator
from coffea.util import load, save
import vector



output_folder = "/eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/RecoPlots/DataPlots/Zee_Run2018D/"
input_folder_must = "/eos/user/r/rdfexp/ecal/cluster/raw_files/EGamma/Dumper_Mustache_bugFix/220919_211119"
input_folder_deep = "/eos/user/r/rdfexp/ecal/cluster/raw_files/EGamma/Dumper_DeepSC_algoA_bugFix/220919_211359"

fileset = {
    "DeepSC": glob(input_folder_deep+"/*/*.root", recursive=True),
    "Mustache": glob(input_folder_must+"/*/*.root", recursive=True)
}

print(f"Working on DeepSC files: {len(fileset['DeepSC'])} and Mustache files {len(fileset['Mustache'])}")


patEle_fields = list(map( lambda k: k.strip().replace("patElectron_",""),
                         open("patElectron_fields.txt").readlines()))

class ZeeProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        dataset = events.metadata['dataset']
        vector.register_awkward()
        electrons = ak.zip({k:events["patElectron_" + k] for k in patEle_fields},
                           with_name="Momentum4D")
        
        nele_init = ak.num(electrons, axis=1)
        
        id_mask = (electrons.egmCutBasedElectronIDloose==1)
        # Trying to replicate the skim selection
        #https://cmssdt.cern.ch/lxr/source/DPGAnalysis/Skims/python/ZElectronSkim_cff.py
#         mask_skim = (electrons.pt >= 10) &\
#                     ( ((abs(electrons.eta)<=1.4442) &(\
#                         (electrons.full5x5_refinedSCSigmaIEtaIEta < 0.0128)&\
#                         (electrons.deltaEtaIn < 0.00523)&\
#                         (electrons.deltaPhiIn < 0.159)&\
#                         (electrons.HoE < 0.247)&\
#                         (electrons.misHits<2)\
#                         ))|
#                         ((abs(electrons.eta)>=1.4442)&(abs(electrons.eta)<=2.5)&\
#                          (electrons.full5x5_refinedSCSigmaIEtaIEta < 0.0445)&\
#                          (electrons.deltaEtaIn < 0.00984)&\
#                          (electrons.deltaPhiIn < 0.157)&\
#                          (electrons.HoE < 0.0982)&\
#                          (electrons.misHits<3)
#                         )
#                     )
        
        # The only missing cuts
        #relCombIsolationWithEACut      = cms.vdouble(0.168  ,0.185  )  , # relCombIsolationWithEALowPtCut
        #EInverseMinusPInverseCut       = cms.vdouble(0.193  ,0.0962   )  ,                
        
        electrons = electrons[id_mask]
        nele_clean = ak.num(electrons, axis=1)
        # Ask for at least two remaining electrons
        electrons = electrons[ak.num(electrons, axis=1)>=2]
        lead_ele = electrons[:, 0]
        sublead_ele = electrons[:,1]
        Z =  lead_ele + sublead_ele
        # Cleaning the event asking for at lead Z.mass 40, leading 20 and sublead 10
        event_mask = (Z.mass>40.)&(lead_ele.pt >= 20.)&(sublead_ele.pt >= 10.)
        
        electrons = electrons[event_mask]
        Z = Z[event_mask]
        lead_ele = lead_ele[event_mask]
        sublead_ele = sublead_ele[event_mask]
        
          
        efficiency_hist = hist.new.Integer(start=0, stop=10, name="nele_initial",label="N. ele initial")\
                                  .Integer(start=0, stop=10, name="nele_clean", label="N. ele cleaned")\
                                  .Double().fill(nele_init, nele_clean)
        
        
        hist_leading_ele = hist.new.Reg(name="et", label="Leading electron $E_T$",
                                       bins=50, start=0, stop=200,)\
                            .Var([0, 0.5, 1, 1.4442, 1.566, 2., 2.5],name="eta", 
                                 label="Leading electron $\eta$")\
                            .Integer(start=1, stop=15, name="ncls", label="Number of PF Clusters")\
                            .IntCategory([0,1,2,3,4], name="class",label="electron class")\
                            .Double()\
                            .fill(lead_ele.et, 
                                  abs(lead_ele.eta),
                                  lead_ele.nPFClusters,
                                  lead_ele.classification)
        
        hist_subleading_ele = hist.new.Reg(name="et", label="Subleading electron $E_T$",
                                       bins=50, start=0, stop=200)\
                            .Var([0, 0.5, 1, 1.4442, 1.566, 2., 2.5],name="eta", 
                                 label="Subleading electron $\eta$")\
                            .Integer(start=1, stop=15, name="ncls", label="Number of PF Clusters")\
                            .IntCategory([0,1,2,3,4], name="class",label="electron class")\
                            .Double()\
                            .fill(sublead_ele.et, 
                                  abs(sublead_ele.eta),
                                  sublead_ele.nPFClusters,
                                  sublead_ele.classification)
   

        hist_Z = hist.new.Reg(name="zmass", bins=120, start=60, stop=120, label="Zmass")\
                             .Reg(name="et", label="Leading electron $E_T$",
                                       bins=30, start=0, stop=200)\
                            .Var([0, 0.5, 1, 1.4442, 1.566, 2., 2.5],name="eta", 
                                 label="Leading electron $\eta$")\
                            .Integer(start=1, stop=15, name="ncls", label="Number of PF Clusters")\
                            .IntCategory([0,1,2,3,4], name="class",label="electron class")\
                            .Double()\
                            .fill(Z.mass,
                                  lead_ele.et,
                                  abs(lead_ele.eta),
                                  lead_ele.nPFClusters,
                                  lead_ele.classification)
        
        return {
            dataset: {
                "entries": len(events),
                "histos": {
                    "Z": hist_Z,
                    "nele_eff": efficiency_hist,
                    "ele_lead": hist_leading_ele,
                    "ele_sublead": hist_subleading_ele
                },
                "Z": {
                    "mass": column_accumulator(ak.to_numpy(Z.mass)),
                    "ele_et": column_accumulator(ak.to_numpy(lead_ele.et)),
                    "ele_eta": column_accumulator(ak.to_numpy(lead_ele.eta)),
                    "ele_class": column_accumulator(ak.to_numpy(lead_ele.classification)),
                }       
            }
        }

    def postprocess(self, accumulator):
        pass



iterative_run = processor.Runner(
    executor = processor.FuturesExecutor(compression=None, workers=20),
    schema=BaseSchema    
)

out = iterative_run(
    fileset,
    treename="recosimdumper/caloTree",
    processor_instance=ZeeProcessor(),
)

save(out, "output.coffea")
    
# from distributed import Client
# from dask_lxplus import CernCluster
# import socket



# n_port = 8786
# with CernCluster(
#     cores=4,
#     memory='8000MB',
#     disk='5000MB',
#     death_timeout = '4000',
#     lcg = True,
#     nanny = False,
#     container_runtime = "none",
#     log_directory = "/eos/user/d/dvalsecc/dask_condor/log",
#     scheduler_options={
#         'port': n_port,
#         'host': socket.gethostname(),
#         },
#     job_extra={
#         '+JobFlavour': 'longlunch',
#         },
#     extra = ['--worker-port 10000:10100']
#     ) as cluster:
    
#     print(cluster.job_script())


#     with Client(cluster) as client:
#         # scaling the job
#         cluster.scale(16)        
#         iterative_run = processor.Runner(
#             executor = processor.DaskExecutor(
#                 client=client,
#             ),
#             schema=BaseSchema,
#         )
        
#         out = iterative_run(
#             fileset,
#             treename="recosimdumper/caloTree",
#             processor_instance=ZeeProcessor(),
#         )

#         save(out, "output.coffea")

