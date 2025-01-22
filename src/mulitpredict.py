import torch
import warnings
import sys
import os
import glob
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
    def forward(self, x):
        if device == 'cuda:0':
            x.cuda(device)
        x = self.dropout(F.relu(self.hidden1(x)))
        x = self.dropout(F.relu(self.hidden2(x)))
        x = self.dropout(F.relu(self.hidden3(x)))
        x = F.relu(self.output(x))
        return x

def main(cif, zeo_exe, discard_geo=True):
    """
    Run the adsorption model for a single CIF. Returns:
        (predicted_working_capacity, predicted_selectivity)
    """
    start_time = datetime.now()
    print("\nStart: ", start_time.strftime("%c"))
    print(f"Predicting adsorption properties for: {cif}")
    print("Device for prediction: ", device)

    print("\n\tComputing geometric descriptors...")
    geo_props = calc_geo_props(cif, zeo_exe=zeo_exe, discard_geo=discard_geo)

    print("\n\tComputing RDFs...")
    rdfs = calc_rdfs(
        name=cif,
        props=["electronegativity", "vdWaalsVolume", "polarizability"],
        smooth=-10,
        factor=0.001
    )

    features = rdfs + geo_props

    # Load scaler
    with open("/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Scale features
    features = scaler.transform(np.array([features]))
    features = torch.FloatTensor(features).to(device)

    print("\n\tLoading in PyTorch models...")
    wc_model = torch.load(
        "/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/wc_model.pt",
        map_location=device
    )
    wc_model.eval()
    sel_model = torch.load(
        "/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/sel_model.pt",
        map_location=device
    )
    sel_model.eval()

    print("\n\tMaking predictions on the dataset...")
    y_predict_wc = wc_model(features)
    y_predict_sel = sel_model(features)

    # Move predictions back to CPU to convert to NumPy
    y_predict_wc = y_predict_wc.cpu().detach().numpy()
    y_predict_sel = y_predict_sel.cpu().detach().numpy()

    predicted_wc = np.round(float(y_predict_wc[0][0]), 2)
    predicted_sel = np.round(float(y_predict_sel[0][0]), 2)

    # Print results
    print(f"\nPredicted working capacity (mmol/g): {predicted_wc}")
    print(f"Predicted CO2/N2 selectivity: {predicted_sel}\n")

    print("Successful termination.")
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("End: ", end_time.strftime("%c"))
    print(f"Total time: {elapsed_time.total_seconds():0.1f} s\n")

    # Return the predictions to be stored in CSV
    return predicted_wc, predicted_sel


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # Instead of a single CIF, point to a DIRECTORY containing multiple .cif files
    # ------------------------------------------------------------------------
    cif_dir = "/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/passing_cifs/"
    zeo_exe_path = "/Users/moinkhwaja/Documents/GitHub/PCCC-MOF-Adsorption-Model/src/zeo++-0.3/network"
    discard_geo_props = True

    # Use glob to find all CIF files in the directory
    cif_files = glob.glob(os.path.join(cif_dir, "*.cif"))
    print(f"Found {len(cif_files)} CIF files in {cif_dir}:")

    # List to hold our results
    results = []

    for cif_file in cif_files:
        print("  ", cif_file)
    print()

    # Process each CIF, capture results
    for cif_file in cif_files:
        wc, sel = main(cif=cif_file, zeo_exe=zeo_exe_path, discard_geo=discard_geo_props)
        # Append a row with: CIF filename, predicted WC, predicted selectivity
        # os.path.basename(...) to get just the file name, not the full path
        results.append([os.path.basename(cif_file), wc, sel])

    # ------------------------------------------------------------------------
    # Create a CSV with these results
    # ------------------------------------------------------------------------
    df = pd.DataFrame(results, columns=["cif_name", "predicted_working_capacity", "predicted_selectivity"])
    out_csv = "prediction_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}")

