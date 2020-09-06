import numpy as np
#import matplotlib.pyplot as plt

#[offset] [type]          [value]          [description]
#0000     32 bit integer  0x00000801(2049) magic number (MSB first)
#0004     32 bit integer  60000            number of items
#0008     unsigned byte   ??               label
#0009     unsigned byte   ??               label
#........
#xxxx     unsigned byte   ??               label
#The labels values are 0 to 9.

class ReadLabel:
    """Class for reading the labels from an MNIST label file"""
    
    def __init__(self, Filename):
        self.Filename = Filename
        self.Data = self.ReadFromFile(Filename)
    
    def ReadFromFile(self, Filename):
        Int32Data = np.dtype(np.uint32)
        Int32Data = Int32Data.newbyteorder('>')
        File = open(Filename,'rb')
        
        MagicFlag = np.frombuffer(File.read(4),Int32Data)[0]
        
        if MagicFlag != 2049:
            Data = np.zeros(0)
            NumberOfElements = 0
        else:    
            NumberOfElements = np.frombuffer(File.read(4),Int32Data)[0]
        
        if NumberOfElements < 1:
            Data = np.zeros(0)
        else:
            Data = np.frombuffer(File.read(NumberOfElements), dtype=np.uint8)
        
        File.close()
        
        return Data

#[offset] [type]          [value]          [description]
#0000     32 bit integer  0x00000803(2051) magic number
#0004     32 bit integer  60000            number of images
#0008     32 bit integer  28               number of rows
#0012     32 bit integer  28               number of columns
#0016     unsigned byte   ??               pixel
#0017     unsigned byte   ??               pixel
#........
#xxxx     unsigned byte   ??               pixel
# Pixels are organized row-wise. 
# Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

class ReadPicture:
    """Class for reading the images from an MNIST image file"""
    
    def __init__(self, Filename):
        self.Filename = Filename
        self.Data = self.ReadFromFile(Filename)
    
    def ReadFromFile(self, Filename):
        Int32Data = np.dtype(np.uint32)
        Int32Data = Int32Data.newbyteorder('>')
        File = open(Filename,'rb')
        
        MagicFlag = np.frombuffer(File.read(4),Int32Data)[0]
        
        if MagicFlag != 2051:
            Data = np.zeros(0)
            NumberOfElements = 0
        else:    
            NumberOfElements = np.frombuffer(File.read(4),Int32Data)[0]
        
        if NumberOfElements < 1:
            Data = np.zeros(0)
            NumberOfRows = 0
        else:
            NumberOfRows = np.frombuffer(File.read(4),Int32Data)[0]
        
        if NumberOfRows != 28:
            Data = np.zeros(0)
            NumberOfColumns = 0
        else:
            NumberOfColumns = np.frombuffer(File.read(4),Int32Data)[0]
            
        if NumberOfColumns != 28:
            Data = np.zeros(0)
        else:
            Data = np.frombuffer(File.read(NumberOfElements * NumberOfRows * NumberOfColumns ), dtype=np.uint8)
            Data = Data.reshape(NumberOfElements, NumberOfColumns, NumberOfRows)
            
        File.close()
        
        return Data

def ProprocessDataSet(FilenameIn_Image, FilenameIn_Label, FilenameOut):
    Pictures = ReadPicture(FilenameIn_Image)
    Labels = ReadLabel(FilenameIn_Label)
    
    for PatternID in range(0, len(Labels.Data)):
        print(str(PatternID) + " of " + str(len(Labels.Data)-1) )
        d = Pictures.Data[PatternID,:,:]
        d = d.astype(np.float32)
        l = Labels.Data[PatternID]
        np.savez(FilenameOut + str(PatternID) + '.npz',Label=l,Data=d)


ProprocessDataSet('t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte','Test/Data')
ProprocessDataSet('train-images-idx3-ubyte','train-labels-idx1-ubyte','Train/Data')

#plt.imshow(Filtered[:,:,1], cmap="gray")

