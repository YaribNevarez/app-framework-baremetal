import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def OnOffFiler(Data):                                                                                                                                                        
    Data = Data.astype(np.single)                                                                                                                                            
    OutputData = np.zeros((Data.shape[0],Data.shape[1],2))                                                                                                                   
    Data = 2.0 * (Data / Data.max()) - 1.0                                                                                                                                   
    OutputData[:,:,0] = np.where(Data < 0, -Data, 0)                                                                                                                         
    OutputData[:,:,1] = np.where(Data > 0, Data, 0)                                                                                                                          
    return OutputData                                                                                                                                                        
                                                                                                                                                                             
def SliceAndDice(Data, WindowSizeX, WindowSizeY):                                                                                                                            
    OutputData = np.zeros((Data.shape[0] + 1 - WindowSizeX,                                                                                                                  
                           Data.shape[1] + 1 - WindowSizeY,                                                                                                                  
                           Data.shape[2] * WindowSizeX * WindowSizeY )).astype(np.float32)                                                                                   
                                                                                                                                                                             
    for x in range(0, OutputData.shape[0]):                                                                                                                                  
        for y in range(0, OutputData.shape[1]):                                                                                                                              
                                                                                                                                                                             
            Temp = Data[x:x+WindowSizeX,y:y+WindowSizeY,:]                                                                                                                   
                                                                                                                                                                             
            for X in range(0, Temp.shape[0]):                                                                                                                                
                for Y in range(0, Temp.shape[1]):                                                                                                                            
                                                                                                                                                                             
                    z = (X + Y * Temp.shape[0])*2                                                                                                                            
                    OutputData[x,y,z:z+2] = Temp[X,Y,:]                                                                                                                      
            OutputData[x,y,:] = OutputData[x,y,:] / OutputData[x,y,:].sum()                                                                                                  
    return OutputData   

def Loader(PatternID, Noise_Amplitude, Bit_Compression, working_directory):

    if Bit_Compression > 0:                                                                                                                                              
        Bit_Compression = np.power(2.0,Bit_Compression)-1.0                                                                                                         

    Temp = np.load(working_directory + "/Test/Data" + str(PatternID) + ".npz")
    Label = Temp['Label']
    A = Temp['Data'] / 255

    Temp = np.load(working_directory + "/Test_Noise/Data" + str(PatternID) + ".npz")
    B = Temp['Data']
                                                                                                                                                                             
    C = A * (1.0 - Noise_Amplitude) + B * Noise_Amplitude
                                                                                                                                                                             
    if Bit_Compression > 0:
        C = np.round(C.astype(np.float64) * Bit_Compression) / Bit_Compression
                                                                                                                                                                             
    C = np.where(C > 1, 1, C)                                                                                                                                        
    Data = C.astype(np.float32)   

    return Data, Label


Bit_Compression = 0 # 0 (don't compress), 1, ..., N bits 

# detect the current working directory and print it
working_directory = os.getcwd()
print ("The current working directory is %s" % working_directory)

for Noise_Amplitude in np.arange(start=0, stop=105, step=5): # Noise_Amplitude = 0.0 .... 1.0
    
    path = working_directory + "/NOISY_MNIST/N_{}".format(Noise_Amplitude)

    if os.path.exists(path) == False:
        os.makedirs(path)
    
    for PatternID in np.arange(0,100): # for test 0...9999
        Data_Raw, Label = Loader(PatternID, Noise_Amplitude/100, Bit_Compression, working_directory)
        print(Label)

        #plt.imshow(Data_Raw)
        #plt.show()

        Data_SbS = OnOffFiler(Data_Raw)
        Data_SbS = SliceAndDice(Data_SbS,5,5)

        file_name = path + "/in" + str(PatternID) + ".bin"

        with open(file_name, mode='wb') as f:
            Data_SbS.tofile(f)
            f.write(bytes([Label]))
            print(Label)

        #### do your export thing...


