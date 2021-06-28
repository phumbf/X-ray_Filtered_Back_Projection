'''A quick Python script exploring some basic X-ray CT algorithm concepts - specifically the filtered back projection algorithm
https://en.wikipedia.org/wiki/Tomographic_reconstruction#Back_Projection_Algorithm
'''

from PIL import Image
from scipy import ndimage
from scipy.fft import ifft, fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
import sys

#Print usage options
def Usage():
    print('''
    Filtered Back Projection -- Python implementation
    Run Options:

    "-draw" - draw option to draw after each step for debugging

    "-reco" - reco option - follow with either "interp" or "standard" 
    to control the method used for back projection

    "-pad" - add padding to the input image with the pad size specified 
    as the argument after this option

    "-img" - add path to image as second argument

    "-help" - print the run options

    Program currently includes a single filter option and has been built as a 
    pedagogical exercise.

    ''')

#Pad an image
def PadImage(image,padsize):
    return np.pad(image,(padsize,padsize),'constant',constant_values=0)

#Unpad an image
def UnPadImage(padsize,padded):

    #Remove top, bottom, left and right padding
    return padded[padsize:-1*padsize,padsize:-1*padsize]

#Import image using Pillow library
def GetImage(filename,padsize):

    image = Image.open(filename)
    #Convert the image class into a numpy array
    data = np.asarray(image)
    sha = data.shape
   
    print('Original image shape is:',data.shape)

    #Ensure images only have two dimensions
    if len(sha) > 2:
        data = data[:,:,0]
        print('Correcting image shape to have two dimensions')
        print('New image shape is:',data.shape)

    #Pad 
    if padsize != 0:
        data = PadImage(data,padsize)

    #Check if image is square
    data = Squarify(data)

    return data

#Squarify image
def Squarify(data):

    sha = data.shape
    pad = 0

    #Check to see whether the image is a square
    #If not, zero pad till is square

    if(sha[0] != sha[1]):
        print('Not square - padding to square')

        if(sha[0] > sha[1]):
            pad = ((0,0),(0,sha[0]-sha[1]))
        else:
            pad = ((0,sha[1]-sha[0]),(0,0))
        
        data = np.pad(data,pad,mode='constant',constant_values=0)
        print('New image shape is:',data.shape)

    return data

#Perform Radon transform
def RadonTrans(data,nstep,draw):

    #Radon matrix needs to be dimensions of largest dimension of image x nsteps
    sinogram = np.zeros((max(data.shape),nstep), dtype='float64')

    for s in range(nstep):
        theta = 180/nstep
        #Perform a rotation of the object
        data = ndimage.rotate(data,theta,reshape=False)

        #Sum the elements and fill the Radon transform matrix
        sinogram[:,s] = data.sum(0)
        
    if draw:
        fig, ax = plt.subplots()
        ax.matshow(sinogram,cmap='gray')
        plt.savefig('results/sinogram.png')

    return sinogram

#Perform reconstruction to retrieve original image
def Recon(sinogram,nstep,draw,recotype):

    #Prepare for fourier filtering by scaling up image via zero padding
    #Yields freq domain points which are more closely spaced
    #Minimum padding of 64
    #See Kak & Stanley Ch2 
  
   #sinosize= sinogram.shape[0]
    sinosize= max(sinogram.shape)
    padsize = int(max(64,2 ** (np.ceil(np.log2(2*sinosize)))))
    padwidth = ((0,padsize - sinosize),(0,0))
    #Perform padding
    sinopad = np.pad(sinogram,padwidth,mode='constant',constant_values=0)

    #Create ramp filter - lines taken from skimage package code
    n = np.concatenate((np.arange(1, padsize / 2 + 1, 2, dtype=int),
                        np.arange(padsize / 2 - 1, 0, -2, dtype=int)))
    f = np.zeros(padsize)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    ffilter = 2 * np.real(fft(f)) 
    #Edit ramp filter for shepp-logan filter
    omega = np.pi * fftfreq(padsize)[1:]
    ffilter[1:] *= np.sin(omega) / omega
    #Create new array dimension so that can be handled properly
    # i.e. for matrix operations
    ffilter = ffilter[:,np.newaxis]

    #FT sinogram to apply the filter
    sinoft = fft(sinopad, axis=0) * ffilter
    sinofilt = ifft(sinoft,axis=0)
    #Only take real part and remove padding
    sinofilt = np.real(sinofilt)[:sinosize,:]
    
    if draw:
        fig, ax = plt.subplots()
        ax.matshow(sinofilt,cmap='gray')
        plt.savefig('results/filtsinogram.png')

    #Reconstruct through back propagation
    reconstructed = np.zeros((sinosize,sinosize))
    
    #Interpolation method
    if recotype == 'interp':
        print("Interpolation back projection")
        radius = sinosize // 2
        xpr, ypr = np.mgrid[:sinosize, :sinosize] - radius
        x = np.arange(sinosize) - sinosize // 2
    
        for col, s in zip(sinofilt.T, range(nstep)):

            #Determine theta and convert to radians
            theta = np.deg2rad(s*180/nstep)
            #t is the new matrix to be interpolated
            t = ypr * np.cos(theta) - xpr * np.sin(theta)

            #Linear interpolation
            reconstructed += np.interp(t,xp=x,fp=col,left=0,right=0)

        #rotate back properly
        reconstructed = np.fliplr(ndimage.rotate(reconstructed,180,reshape=False))

    #Standard, no interpolation method
    elif recotype == 'standard':
        print("Standard back projection")

        for s in range(nstep):
            theta = 180/nstep
            reconstructed = ndimage.rotate(reconstructed,theta,reshape=False)
            reconstructed[:,:] += sinofilt[:,s]
    
        #rotate back
        reconstructed = ndimage.rotate(reconstructed,180,reshape=False)
    
    #If neither interpolation nor standard
    else:
        print('ERROR: Option must either be standard or interp')

    if draw:
        fig, ax = plt.subplots()
        ax.matshow(reconstructed,cmap='gray')
        plt.savefig('results/' + recotype +'_reco.png')

    return reconstructed

#FILTERED BACK PROJECTION ALGORITHM
#Decide on number of steps
filename = ""
draw = False
recotype = "interp"
padsize = 0

#Command Line argument info
for i in range(len(sys.argv)):
    if sys.argv[i] == "-draw":
        draw = True
    elif sys.argv[i] == "-reco":
        recotype = sys.argv[i+1]
    elif sys.argv[i] == '-pad':
        padsize = int(sys.argv[i+1])
    elif sys.argv[i] == '-img':
        filename = sys.argv[i+1]
    elif sys.argv[i] == '-help':
        Usage()
        sys.exit()

##Get Image 
data = GetImage(filename,padsize)
nstep = max(data.shape)
print("Got image")

##pad image
padimage = data
if padsize != 0:
    padimage = PadImage(data,padsize)

#Squarify image
padimage = Squarify(padimage)

##Radon Transform 
sinogram = RadonTrans(data,nstep,draw)
print("Radon Transformed image")

##Reconstruct Image
recons = Recon(sinogram,nstep,draw, recotype)

print("Image Reconstructed")

##If padded - undo padding for drawing
recons = UnPadImage(padsize,recons)
data = UnPadImage(padsize,data)

print("Removing any pads")

##Calculate Error Image
error = data - recons
fig, ax = plt.subplots(1, 4, constrained_layout = True)
fig.suptitle('Filtered Back Projection using ' + recotype + ' BP',y=0.78)
ax[0].matshow(data,cmap='gray')
ax[0].get_xaxis().set_visible(False)
ax[0].get_yaxis().set_visible(False)
ax[0].set_title('Original Image')
ax[1].matshow(sinogram,cmap='gray')
ax[1].get_xaxis().set_visible(False)
ax[1].get_yaxis().set_visible(False)
ax[1].set_title('(Padded) Sinogram')
ax[2].matshow(recons,cmap='gray')
ax[2].get_xaxis().set_visible(False)
ax[2].get_yaxis().set_visible(False)
ax[2].set_title('Reconstructed')
ax[3].matshow(error,cmap='gray')
ax[3].get_xaxis().set_visible(False)
ax[3].get_yaxis().set_visible(False)
ax[3].set_title('Error')
plt.savefig('results/'+ recotype +'_results.png')
