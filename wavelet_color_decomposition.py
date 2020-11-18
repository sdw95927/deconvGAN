import cv2
import numpy as np
import matplotlib.pyplot        as plt

import pywt
from scipy.stats import kurtosis
from sklearn.decomposition import FastICA

def convert_to_optical_densities(rgb, r0, g0, b0):
    OD = rgb.astype(float)
    OD[:,:,0] /= r0
    OD[:,:,1] /= g0
    OD[:,:,2] /= b0
    return -np.log(OD+0.00001)

def wavelet_decomposition(image, Io=255, Level=5, NumBands=20, wname='db8', showResult = False):
    '''
    Input: image: h*w*3 array; 
    Io: max image value; 
    Level, NumBands, wname: parameters for wavelet decomposition
    '''
    # Convert uint8 image to OD values (-log(uint8/255 + 0.0001))
    OD = convert_to_optical_densities(image,Io,Io,Io)
    
    # reshape image on row per pixel
    rOD = np.reshape(OD,(-1,3))
    
    Statcount = 0
    StatMesur = np.zeros((3,Level*4))
    
    # Create OD_M
    B = image[:, :, 2]
    G = image[:, :, 1]
    R = image[:, :, 0]
    RGB = np.array([R.reshape(-1), G.reshape(-1), B.reshape(-1)])
    OD_M = -np.log(RGB/255 + 0.00001)
    
    # Wavelet decomposition
    Bands = []
    cA0 = OD[:, :, 0]
    cA1 = OD[:, :, 1]
    cA2 = OD[:, :, 2]
    for i in range(Level):
        cA0, (cH0, cV0, cD0) = pywt.dwt2(data=cA0, wavelet='db8')
        cA1, (cH1, cV1, cD1) = pywt.dwt2(data=cA1, wavelet='db8')
        cA2, (cH2, cV2, cD2) = pywt.dwt2(data=cA2, wavelet='db8')
        LL = np.zeros((cA0.shape[0], cA0.shape[1], 3))
        LH = np.zeros((cA0.shape[0], cA0.shape[1], 3))
        HL = np.zeros((cA0.shape[0], cA0.shape[1], 3))
        HH = np.zeros((cA0.shape[0], cA0.shape[1], 3))
        LL[:, :, 0] = cA0; LL[:, :, 1] = cA1; LL[:, :, 2] = cA2
        LH[:, :, 0] = cH0; LH[:, :, 1] = cH1; LH[:, :, 2] = cH2
        HL[:, :, 0] = cV0; HL[:, :, 1] = cV1; HL[:, :, 2] = cV2
        HH[:, :, 0] = cD0; HH[:, :, 1] = cD1; HH[:, :, 2] = cD2
        Bands.append([LL, LH, HL, HH])

        # Normalized bands to have zero mean and unit variance
        LL_l = (LL-np.mean(LL))/np.std(LL)
        LH_l = (LH-np.mean(LH))/np.std(LH)
        HL_l = (HL-np.mean(HL))/np.std(HL)
        HH_l = (HH-np.mean(HH))/np.std(HH)

        # Compute Non-Gaussian Messures
        Kor = [abs(kurtosis(LL_l.flatten())-3),abs(kurtosis(LH_l.flatten())-3),
               abs(kurtosis(HL_l.flatten())-3),abs(kurtosis(HH_l.flatten())-3)]

        z = 0
        for s in range(Statcount, Statcount + 4):
            StatMesur[0, s] = Kor[z]
            StatMesur[1, s] = i #level
            StatMesur[2, s] = z #band
            z = z+1
        Statcount = Statcount + 4
    
    # Sort Kourtosis matrix
    d2 = sorted(range(len(StatMesur[0, :])), key=lambda k: StatMesur[0, k], reverse=True)
    StatMesur = StatMesur[:, d2]
    
    # Concentrate subbands
    Coff = Bands[0][0]
    B = Coff[:, :, 2]
    G = Coff[:, :, 1]
    R = Coff[:, :, 0]
    Coff = [B.flatten(),G.flatten(),R.flatten()]
    FinalSignal = Coff
    for i in range(NumBands):
        Coff = Bands[np.int(StatMesur[1, i])][np.int(StatMesur[2, i])] # Bands[Level][Band]
        B = Coff[:, :, 2]
        G = Coff[:, :, 1]
        R = Coff[:, :, 0]
        Coff = [B.flatten(),G.flatten(),R.flatten()]
        FinalSignal = np.concatenate((FinalSignal, Coff), axis = 1)
    
    # apply ICA
    ica = FastICA()
    A = ica.fit(FinalSignal.T).mixing_ # Mixing matrix, [No. of features, No. of components]
    # A = ica.fit(FinalSignal.T).transform(FinalSignal.T)

    # Compute OD and density image and stain matrix
    Ref_Vecs = abs(A)

    # Normalize stain vector
    for z in range(3):
        # Normalize vector length
        length = (Ref_Vecs[0, z]**2 + Ref_Vecs[1, z]**2 + Ref_Vecs[2, z]**2)**0.5
        if length != 0.0:
            Ref_Vecs[0, z] = Ref_Vecs[0, z]/length
            Ref_Vecs[1, z] = Ref_Vecs[1, z]/length
            Ref_Vecs[2, z] = Ref_Vecs[2, z]/length
    
    # Sort to start with H
    ''' 
    Ref_Vecs: 
        [[Hr, Er, Br]
         [Hg, Eg, Bg]
         [Hb, Eb, Bb]]
    First column: lowest blue OD (H)
    Second column: lowest red OD (E)
    '''
    Temp = Ref_Vecs.copy()
    c = np.argmin(Temp[2, :])
    Ref_Vecs[:, 0] = Temp[:, c]
    Temp = np.delete(Temp, c, axis=1)
    c = np.argmin(Temp[0, :])
    Ref_Vecs[:, 1] = Temp[:, c]
    Temp = np.delete(Temp, c, axis=1)
    Ref_Vecs[:, 2] = Temp[:, 0]
    
    # Compute desity matrix and show results
    d = np.dot(np.linalg.inv(Ref_Vecs), OD_M)
    
    if showResult:
        H = Io*np.exp(-np.dot(np.array([Ref_Vecs[:, 0]]).T, np.array([d[0, :]])))
        H = np.reshape(H.T, image.shape)
        np.clip(H, 0, 255, out=H)
        H = np.uint8(H)
        plt.imshow(H)
        plt.show()
        E = Io*np.exp(-np.dot(np.array([Ref_Vecs[:, 1]]).T, np.array([d[1, :]])))
        E = np.reshape(E.T, image.shape)
        np.clip(E, 0, 255, out=E)
        E = np.uint8(E)
        plt.imshow(E)
        plt.show()
        B = Io*np.exp(-np.dot(np.array([Ref_Vecs[:, 2]]).T, np.array([d[2, :]])))
        B = np.reshape(B.T, image.shape)
        np.clip(B, 0, 255, out=B)
        B = np.uint8(B)
        plt.imshow(B)
        plt.show()
    
    # Return H channel stain density
    return np.reshape(d[0, :], image.shape[0:2])