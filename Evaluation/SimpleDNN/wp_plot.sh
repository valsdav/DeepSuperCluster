#!/bin/sh

# python working_points_plots_elegammasep.py -f ele  -iv 1 \
#    -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/validation_plots/deepSC_v15_optwpelegamma_ele \
#    -m ../models/v15_optwpelegamma/model_v1_EB.hd5   \
#    -s ../models/v15_optwpelegamma/scaler_model_v1_EB.pkl \
#    -e 0-0.4 0.4-0.8 0.8-1.2 1.2-1.479

# python working_points_plots_elegammasep.py -f ele  -iv 1 \
#    -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/validation_plots/deepSC_v15_optwpelegamma_ele \
#    -m ../models/v15_optwpelegamma/model_v2_EE.hd5   \
#    -s ../models/v15_optwpelegamma/scaler_model_v2_EE.pkl \
#    -e 1.479-1.75 1.75-2 2-2.25 2.25-3

# python working_points_plots_elegammasep.py -f gamma  -iv 1 \
#    -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/validation_plots/deepSC_v15_optwpelegamma_gamma \
#    -m ../models/v15_optwpelegamma/model_v1_EB.hd5   \
#    -s ../models/v15_optwpelegamma/scaler_model_v1_EB.pkl \
#    -e 0-0.4 0.4-0.8 0.8-1.2 1.2-1.479

# python working_points_plots_elegammasep.py -f gamma  -iv 1 \
#    -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/validation_plots/deepSC_v15_optwpelegamma_gamma \
#    -m ../models/v15_optwpelegamma/model_v2_EE.hd5   \
#    -s ../models/v15_optwpelegamma/scaler_model_v2_EE.pkl \
#    -e 1.479-1.75 1.75-2 2-2.25 2.25-3


# python working_points_plots_elegammasep.py -f ele  -iv 1 \
#    -o deepSC_v16_elegammasep_ele \
#    -m ../models/v16_elegammasep/model_v1_EB_ele.hd5   \
#    -s ../models/v16_elegammasep/scaler_model_v1_EB_ele.pkl \
#    -e 0-0.4 0.4-0.8 0.8-1.2 1.2-1.479 -d EB

# python working_points_plots_elegammasep.py -f ele  -iv 1 \
#    -o deepSC_v16_elegammasep_ele \
#    -m ../models/v16_elegammasep/model_v1_EE_ele.hd5   \
#    -s ../models/v16_elegammasep/scaler_model_v1_EE_ele.pkl \
#    -e 1.479-1.75 1.75-2 2-2.25 2.25-3 -d EE

# python working_points_plots_elegammasep.py -f gamma  -iv 1 \
#    -o deepSC_v16_elegammasep_gamma \
#    -m ../models/v16_elegammasep/model_v1_EB_gamma.hd5   \
#    -s ../models/v16_elegammasep/scaler_model_v1_EB_gamma.pkl \
#    -e 0-0.4 0.4-0.8 0.8-1.2 1.2-1.479 -d EB

# python working_points_plots_elegammasep.py -f gamma  -iv 1 \
#    -o deepSC_v16_elegammasep_gamma \
#    -m ../models/v16_elegammasep/model_v1_EE_gamma.hd5   \
#    -s ../models/v16_elegammasep/scaler_model_v1_EE_gamma.pkl \
#    -e 1.479-1.75 1.75-2 2-2.25 2.25-3  -d EE


python working_points_plots_v2.py -iv 1   \
    -o /eos/user/d/dvalsecc/www/ECAL/Clustering/DeepCluster/validation_plots/deepSC_v18_bayesresult  \
    -m ../models/v18_bayesresult/model_v1.hd5 \
    -s ../models/v18_bayesresult/scaler_model_v1.pkl \
    -e  0-0.4 0.4-0.8 0.8-1.2 1.2-1.479  1.479-1.75 1.75-2 2-2.25 2.25-3 
