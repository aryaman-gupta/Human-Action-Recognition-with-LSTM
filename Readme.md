This system implements Human Action Recognition from skeleton data. It uses a feature representation called Skeletal Quad that is scale and body orientation invariant. It also utilizes LSTM networks for their strength in modelling the dynamics of a temporal sequence.

WriteFile reads from the dataset and writes the features of each sequence into a text file from where it can be read by a dataloader. It utilizes the SimilarityNormTransform to perform the quad encoding. Each frame is encoded by 25 quads, each of length 6. Therefore, the total number of features representing an action sequence are 6 x 25 x no. of frames, and frame wise features can be easily extracted by picking 6 x 25 features at a time.

Main contains the dataloader and the LSTM code. An action sequence is passed to the LSTM network, and the output of the final frame is compared with the actual label while training.
