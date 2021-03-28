import torch
import torch.nn as nn
import torch.nn.functional as F




def conv3x3_bn_relu(inp_activation, output_activation, BN=True, activation = True):
    "con 3x3 + Batchnormalization + relu"
    layer = [nn.Conv2d(inp_activation, output_activation, 3, padding = 1)]
    for i, j in zip([nn.BatchNorm2d(output_activation), nn.ReLU(inplace=True)],[BN, activation]):
        if j==True:
            layer.append(i)
    return nn.Sequential(*layer)
    
    
    
class conv_block(nn.Module):
    
    def __init__(self, inp_activation, list_filter, BN = False):
        super().__init__()
        self.conv3x3_1 = conv3x3_bn_relu(inp_activation, list_filter, BN=BN)
        self.conv3x3_2 = conv3x3_bn_relu(list_filter, list_filter, BN=BN)
    def forward(self , inp):
        c = self.conv3x3_1(inp)
        c = self.conv3x3_2(c)
        return c
        
class Unet(nn.Module):
    def __init__(self, n_class):
        
        super().__init__()
        self.en_block1 = conv_block(16,32)
        self.en_block2 = conv_block(32,64)
        self.en_block3 = conv_block(64,128)
        self.en_block4 = conv_block(128,256)
        self.en_block5 = conv_block(256,512)
        self.en_block6 = conv_block(512, 1024)

        
        self.transpose5 = nn.ConvTranspose2d(1024,512,2,2)
        self.transpose4 = nn.ConvTranspose2d(512,256,2,2)

        self.transpose3 = nn.ConvTranspose2d(256,128,2,2)
        self.transpose2 = nn.ConvTranspose2d(128,64,2,2)
        self.transpose1 = nn.ConvTranspose2d(64,32,2,2)
        
        self.de_block1 = conv_block(64,32)
        self.de_block2 = conv_block(128,64)
        self.de_block3 = conv_block(256,128)

        self.de_block4 = conv_block(512, 256)
        self.de_block5 = conv_block(1024, 512)
        self.out_conv = nn.Conv2d(32, n_class, 1)

        
    

    def forward(self, inp):
        el1 = self.en_block1(inp) #  (32,h,w)
        print('el1',el1.shape)
        max1 = nn.MaxPool2d(2)(el1) # (32,h//2, w//2)
        print('max1',max1.shape)

        el2 = self.en_block2(max1)    #(64, h//2, w//2)
        print('el2',el2.shape)

        max2 = nn.MaxPool2d(2)(el2)  #(64, h//4, w//4)
        print('max2',max2.shape)

        el3 = self.en_block3(max2)    #(128, h//4, w//4)
        print('el3',el3.shape)

        max3 = nn.MaxPool2d(2)(el3)  #(128, h//8, w//8)
        print('max3',max3.shape)


        el4 = self.en_block4(max3)    #(256, h//8, w//8)
        print('el4',el4.shape)

        max4 = nn.MaxPool2d(2)(el4)  #(256, h//16, w//16)
        print('max4',max4.shape)

        el5 = self.en_block5(max4)  #(512, h//16, w//16)
        print('el5',el5.shape)

        max5 = nn.MaxPool2d(2)(el5)  #(512, h//32, w//32)
        print('max5',max5.shape)

        
        el6 = self.en_block6(max5)  #(1024, h//32, w//32)
        print('el6',el6.shape)


        tl5 = self.transpose5(el6)  #(512, h//16, w//16)
        print('tl5',tl5.shape)

        cat5 = torch.cat([tl5, el5], 1) #(1024, h//16, h//16 )
        print('cat5',cat5.shape)

        d5 =  self.de_block5(cat5)      #(512, h//16, w//16
        print('d5',d5.shape)

        
        tl4 = self.transpose4(d5)       #(256, h//8, w//8)
        cat4 = torch.cat([tl4, el4], 1) #(512, h//8, w//8)
        d4 =  self.de_block4(cat4)     #(256, h//8, w//8)
        
        tl3 = self.transpose3(d4)        #(128, h//4, w//4)
        cat3 = torch.cat([tl3, el3], 1)  #(256, h//4, w//4)
        d3 =  self.de_block3(cat3)        #(128, h//4, w//4)
        
        
        tl2 = self.transpose2(d3)          #(64, h//2, w//2)
        cat2 = torch.cat([tl2, el2], 1)   #(128, h//2, w//2)
        d2 =  self.de_block2(cat2)         #(64, h//2, w//2)
        
        tl1 = self.transpose1(d2)          #(32, h, w)
        cat1 = torch.cat([tl1, el1], 1) #(64, h, w)
        d1 =  self.de_block1(cat1)        #(32, h, w)
        output = self.out_conv(d1) 

        return output


