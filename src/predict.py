import torch
import warnings
import sys
import os.path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from calc_descriptors import calc_geo_props, calc_rdfs
import pickle

# Define globally, needed by Net3 class and main function
use_cuda = torch.cuda.is_available()
device = torch.device('cuda:0' if use_cuda else 'cpu')

# Class for 3-layer models
class Net3(nn.Module):
    
    def __init__(self):
        super(Net3, self).__init__()

    # forward function applies activation function
    # to input and sends it to output (x is input tensor)
    def forward(self, x):
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(F.relu(self.hidden1(x)))
        x = self.dropout(F.relu(self.hidden2(x)))
        x = self.dropout(F.relu(self.hidden3(x)))
        x = F.relu(self.output(x))
        return x

def main(cif, zeo_exe, discard_geo=True):
    start_time = datetime.now()
    print("\nStart: ", start_time.strftime("%c"))

    print(f"Predicting adsorption properties for: {cif}")
    print("Device for prediction: ", device)

    print("\n\tComputing geometric descriptors...")
    geo_props = calc_geo_props(cif,
                               zeo_exe=zeo_exe,
                               discard_geo=discard_geo)

    print("\n\tComputing RDFs...")
    rdfs = calc_rdfs(name=cif,
                     props=["electronegativity",
                            "vdWaalsVolume",
                            "polarizability"],
                     smooth=-10,
                     factor=0.001)

    features = rdfs + geo_props

    with open("/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    features = scaler.transform(np.array([features]))
    features = torch.FloatTensor(features).to(device)

    print("\n\tLoading in PyTorch models...")
    wc_model = torch.load("/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/wc_model.pt", map_location=device)
    wc_model.eval()
    sel_model = torch.load("/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/sel_model.pt", map_location=device)
    sel_model.eval()

    print("\n\tMaking predictions on the dataset...")
    y_predict_wc = wc_model(features)
    y_predict_sel = sel_model(features)

    # Move predictions back to CPU and detach to convert to NumPy
    y_predict_wc = y_predict_wc.cpu().detach().numpy()
    y_predict_sel = y_predict_sel.cpu().detach().numpy()

    # Print results
    print("\nPredicted working capacity (mmol/g): {}".format(
          np.round(float(y_predict_wc[0][0]), 2)))
    print("Predicted CO2/N2 selectivity: {}\n".format(
          np.round(float(y_predict_sel[0][0]), 2)))

    print("Successful termination.")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("End: ", end_time.strftime("%c"))
    print("Total time: {0:.1f} s\n".format(elapsed_time.total_seconds()))

if __name__ == "__main__":
    # ----------------------------------------------------------
    # Hard-code your paths here:
    #   1) Path to the .cif file of your MOF
    #   2) Path to the Zeo++ executable (network binary)
    # ----------------------------------------------------------
    cif_path = "/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/tests/test1.cif"
    zeo_exe_path = "/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/zeo++-0.3/network"

    # Whether to discard intermediate geometric property files
    discard_geo_props = True

    # Call the main function with hard-coded paths
    main(cif=cif_path,
         zeo_exe=zeo_exe_path,
         discard_geo=discard_geo_props)

