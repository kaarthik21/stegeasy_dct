import shutil
import cv2
import numpy as np
# numpy used less space and faster than lists
import itertools
from PIL import Image

quant = np.array([[16,11,10,16,24,40,51,61],      # QUANTIZATION TABLE
                    [12,12,14,19,26,58,60,55],    # required for DCT
                    [14,13,16,24,40,57,69,56],
                    [14,17,22,29,51,87,80,62],
                    [18,22,37,56,68,109,103,77],
                    [24,35,55,64,81,104,113,92],
                    [49,64,78,87,103,121,120,101],
                    [72,92,95,98,112,100,103,99]])


class DCT():    
    def __init__(self): # Constructor
        self.message = None
        self.allBits = None
        self.oriCol = 0
        self.oriRow = 0
        self.numBits = 0   
    #encoding part : 
    def encode_image(self,img,secret_msg):
        secret=secret_msg
        self.message = str(len(secret))+'*'+secret
        self.allBits = self.ConverttoBits()
        # ConverttoBits() returns array bit of message
        # get size of image in pixels
        row,col = img.shape[:2]
        # cv2 saves image in rows and columns and chanels, saving first 2 of 3 to row,column
        self.oriRow, self.oriCol = row, col  
        if((col/8)*(row/8)<len(secret)):
        # pixel size of 8 bit image
            print("Error: Message too large to encode in image")
            return False
        # make divisible by 8x8
        if row%8 != 0 or col%8 != 0:
            img = self.divby8(img, row, col)
            # divby8 func for rezing div by 8
        
        row,col = img.shape[:2]
        bImg,gImg,rImg = cv2.split(img)
        
        # split image into RGB channels
        # As cv2 gives bgr instead of rgb
        print("BLUE")
        print(bImg)
        print("GREEN")
        print(gImg)
        print("RED")
        print(rImg)


        bImg = np.float32(bImg)
        # message to be hid in blue channel so converted to type float32 for dct function
        imgBlocks = [np.round(bImg[i:i+8, j:j+8]) for (i,j) in itertools.product(range(0,row,8),range(0,col,8))]
        # range(start,stop,step)
        # Break into 8x8 blocks
        # itertools.product gives all possible combinations of 2 arrays in pattern, can -128 to bImg

        #Blocks are run through DCT function
        dctBlocks = [np.round(cv2.dct(img_Block)) for img_Block in imgBlocks]
        
        #blocks then run through quantization table
        quantizedDCT = [np.round(dct_Block/quant) for dct_Block in dctBlocks]
        
        #set LSB in DC value corresponding bit of message
        messIndex = 0
        letterIndex = 0
        for quantizedBlock in quantizedDCT:
            #find LSB in DC coeff and replace with message bit
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            # uint8 - unsigned 8-bit dayabyte from 0-255
            DC = np.unpackbits(DC)
            # unpack uint8 elements to binary
            DC[7] = self.allBits[messIndex][letterIndex]
            # first letter and first value of ascii
            DC = np.packbits(DC)
            # packs binary value to uint8 bits
            
            DC = np.float32(DC)
            DC= DC-255
            quantizedBlock[0][0] = DC
            letterIndex = letterIndex+1
            if letterIndex == 8:
                letterIndex = 0
                # 8 bit form
                messIndex = messIndex + 1
                if messIndex == len(self.message):
                    break
        #blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
                
        sImg=[]
        #sImg is changed bImg
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
                    # block[rowBlockNum] will be appended to sImg
        sImg = np.array(sImg).reshape(row, col)
        #converted from type float32
        sImg = np.uint8(sImg)
        #show(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        return sImg

    def decode_image(self,img):
        row,col = img.shape[:2]
        messSize = None
        messageBits = []
        buff = 0
        #split image into RGB channels
        bImg,gImg,rImg = cv2.split(img)
         #message hid in blue channel so converted to type float32 for dct function
        bImg = np.float32(bImg)
        #break into 8x8 blocks
        imgBlocks = [bImg[j:j+8, i:i+8]-128 for (j,i) in itertools.product(range(0,row,8),
                                                                       range(0,col,8))]    
        #blocks run through quantization table
        #quantizedDCT = [dct_Block/ (quant) for dct_Block in dctBlocks]
        quantizedDCT = [img_Block/quant for img_Block in imgBlocks]
        i=0
        #message extracted from LSB of DC coeff
        for quantizedBlock in quantizedDCT:
            DC = quantizedBlock[0][0]
            DC = np.uint8(DC)
            DC = np.unpackbits(DC)
            if DC[7] == 1:
                buff+=(0 & 1) << (7-i)
            elif DC[7] == 0:
                buff+=(1&1) << (7-i)
            i=1+i
            if i == 8:
                messageBits.append(chr(buff))
                buff = 0
                i =0
                if messageBits[-1] == '*' and messSize is None:
                    try:
                        messSize = int(''.join(messageBits[:-1]))
                    except:
                        pass
            if len(messageBits) - len(str(messSize)) - 1 == messSize:
                return ''.join(messageBits)[len(str(messSize))+1:]
        #blocks run inversely through quantization table
        sImgBlocks = [quantizedBlock *quant+128 for quantizedBlock in quantizedDCT]
        #blocks run through inverse DCT
        #sImgBlocks = [cv2.idct(B)+128 for B in quantizedDCT]
        #puts the new image back together
        sImg=[]
        for chunkRowBlocks in self.chunks(sImgBlocks, col/8):
            for rowBlockNum in range(8):
                for block in chunkRowBlocks:
                    sImg.extend(block[rowBlockNum])
        sImg = np.array(sImg).reshape(row, col)
        #converted from type float32
        sImg = np.uint8(sImg)
        sImg = cv2.merge((sImg,gImg,rImg))
        ##sImg.save(img)
        #dct_decoded_image_file = "dct_" + original_image_file
        #cv2.imwrite(dct_decoded_image_file,sImg)
        return ''



    
    def chunks(self, l, n):
        m = int(n)
        for i in range(0, len(l), m):
            yield l[i:i + m]
            # yield is same as return, withoyt changing local variables
            # : - list slicing
    def divby8(self,img, row, col):
        img = cv2.resize(img,(col+(8-col%8),row+(8-row%8)))    
        # param-input image,scafefactorx,setfactory
        return img
    def ConverttoBits(self):
        bits = []
        for i in self.message:
            binval = bin(ord(i))[2:].rjust(8,'0')
            # 2: for taking off '0b' as its base2
            # Convert text to ascii and then to binary
            # rjust is used to give 0 in front for total of 8 unit places
            bits.append(binval)
        self.numBits = bin(len(bits))[2:].rjust(8,'0')
        return bits





# PROGRAM STARTS HERE
original_image = ""
final_image_file = ""

while True:
    n = input("1.ENCODE\n2.DECODE\nPRESS ANY BUTTON TO EXIT")
    if n == "1":

        original_image = input("Enter the name of the file with extension : ")

        img = cv2.imread(original_image, cv2.IMREAD_UNCHANGED)
        # img = cv2.imread(path, way of img reading,greyscale,etc)
        # imread load image, cv2.IMREAD_UNCHANGED for loading same img
        text = input("Enter the message you want to hide: ")
        print("The message length is: ",len(text))
        encoded_img = DCT().encode_image(img, text)
        final_image_file = "encoded_" + original_image
        cv2.imwrite(final_image_file,encoded_img)
        # Save image,param=(filename,image)
        print("Encoded images were saved!")


    elif n == "2":

        img_to_decode = input("Enter the image to be decoded with extension:")
        img = cv2.imread(img_to_decode, cv2.IMREAD_UNCHANGED)

        decoded_text = DCT().decode_image(img) 
        #print(decoded_text)
        file = open("DECODED_"+img_to_decode+".txt","w")
        file.write(decoded_text) # saving hidden text as text file
        file.close()
        print("Hidden texts are also saved as text file!")
    
 
    else:
        print("Closed!")
        break
