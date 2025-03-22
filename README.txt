Use conda activate predictor for this folder's environment
Use this environment for constructing the disruption prediction model, not for MDS D3D data acquisition
    -   Use the other environment and folder conda activate mldp for data acquisition

PlasmaDataset can be used to source dataset from JTEXT data and organizes it into feature dictionaries, ready for exporting
 - Features, splits, and dataset source fractions can be assigned to specify dataset parameters

PlasmaLSTM organizes an ML-ready dataset from PlasmaDataset and creates an appropriate LSTM model to train and evaluate
 - Also exports trained LSTM model and model statistics along with dataset stats to a log (CSV/JSON)

Goal of this program is to generate and train LSTM models on JTEXT data and attempt to find a combination of features and parameters most profitable in accurate disruption prediction - eventually in transfer prediction on DIII-D

