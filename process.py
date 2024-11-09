import pickle
import numpy as np

with open('satellite_k10.pkl', 'rb') as f:
    data = pickle.load(f)

n_data = 1549
n_queries = 1464

gd = ""
iss = ""
fe = ""
fm = ""

for k, v in data.items():
    gld = np.mean(v['global_descs'])
    ins = np.mean(v['index_search'])
    fee = np.mean(v['feature_extraction'])
    fem = np.mean(v['feature_matching'])
    gld_std = np.std(v['global_descs'])
    ins_std = np.std(v['index_search'])
    fee_std = np.std(v['feature_extraction'])
    fem_std = np.std(v['feature_matching'])
    # print(f"{k:15}\tGD {gld:.4f} ± {gld_std:.4f}")
    # print(f"{'':15}\tIS {ins:.4f} ± {ins_std:.4f}")
    # print(f"{'':15}\tFE {fee:.4f} ± {fee_std:.4f}")
    # print(f"{'':15}\tFM {fem:.4f} ± {fem_std:.4f}")

    gd += f"{gld :.4f}\n"
    iss += f"{ins :.4f}\n"
    fe += f"{fee :.4f}\n"
    fm += f"{fem :.4f}\n"
    

print(gd, iss, fe, fm, sep='\n\n')
