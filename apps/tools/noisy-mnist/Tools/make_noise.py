import numpy as np

FilenamePrefix_Test = "./Test/Data"
FilenamePrefix_TestNoise = "./Test_Noise/Data"
NumberOfPatterns_Test = 10000

Temp = np.load(FilenamePrefix_Test + "0.npz")
Temp = Temp['Data']

for i in np.arange(0,NumberOfPatterns_Test):
    print(i)
    Data = np.random.random(Temp.shape)
    np.savez(FilenamePrefix_TestNoise + str(i) + ".npz", Data = Data)

