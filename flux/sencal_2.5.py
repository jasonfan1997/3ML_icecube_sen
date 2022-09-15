# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 13:31:53 2022

@author: fkt83
"""
import matplotlib 
font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator,UnivariateSpline,Akima1DInterpolator
import glob
declist=np.arange(-85,85,2.5)
#declist=[40.0]
index = 2.5
extension = 0
sen=[]
threesig=[]
fivsig=[]
#ext=["","_0.5","_1.0"]
ext=[""]
threesigTS=[]
for e in ext:
    sen=[]
    threesig=[]
    fivsig=[]
    
    for dec in declist:
        filelist = glob.glob("./BG/*_dec"+str(dec)+"_*npy")
        TS=[]
        for f in filelist:
            tempdata=np.load(f)
            TS.append(tempdata[:,0])
        TS=np.array(TS).flatten()
        TS=np.sort(TS)
        

        
        Median =  np.median(TS)
        threesigma = np.percentile(TS, 99.865)
        #print(dec,threesigma,np.percentile(TS, 99.865))
        #fivesigma = threesigma
        # Median=np.median(TS)
        # threesigma = np.percentile(TS, 99.865)
        fivesigma = np.percentile(TS, 100-(100-99.99994)/2)
        # print(dec,threesigma)
        threesigTS.append(threesigma)

        
        if dec >= -5 and index==3 :
            n_sigs = np.r_[25:72:5]
        elif dec == -5 and index ==3 and extension > 0:
            n_sigs = np.r_[15:62:5]
        else:
            n_sigs = np.r_[2:10:1, 10:30.1:2]
            #n_sigs = np.r_[2:10:1]
        listthreesigmasignal=[]
        listfivesigmasignal=[]
        listmedsignal=[]
        for i in n_sigs:
            signalfile = glob.glob("./combine_signal"+e+"_"+str(index)+"/*_dec"+str(dec)+"_ninj"+str(i)+"*index_"+str(index)+"*npy")
            signalTS = []
            for sf in signalfile:
                tempdata=np.load(sf)
                signalTS.append(tempdata[:,0])
            signalTS=np.array(signalTS).flatten()
            listmedsignal.append((signalTS>Median).sum()/len(signalTS))
            listthreesigmasignal.append((signalTS>threesigma).sum()/len(signalTS))
            listfivesigmasignal.append((signalTS>fivesigma).sum()/len(signalTS))
        nsiglist = np.linspace(n_sigs[0],n_sigs[-1]+20,20000)
        medsignalspline = PchipInterpolator(n_sigs,listmedsignal)
        temp=medsignalspline(nsiglist)
        sen.append(nsiglist[np.where((temp>0.9))[0][0]])
            
        threesignalspline = PchipInterpolator(n_sigs,listthreesigmasignal)
        temp=threesignalspline(nsiglist)
        try:
            threesig.append(nsiglist[np.where((temp>0.5))[0][0]])
        except:
            print(e+" "+str(dec)+" out of bound for 3 sigma")
            threesig.append(0)
        #threesig.append(nsiglist[np.where((temp>0.5))[0][0]])
        fivesignalspline = PchipInterpolator(n_sigs,listfivesigmasignal)
        temp=fivesignalspline(nsiglist)
        try:
            fivsig.append(nsiglist[np.where((temp>0.5))[0][0]])
        except:
            print(e+" "+str(dec)+" out of bound for 5 sigma")
            fivsig.append(0)
        np.save("sen_v42"+e+"_"+str(index)+".npy",np.array(sen))
        np.save("threesig_v42"+e+"_"+str(index)+".npy",np.array(threesig))
        np.save("fivsig_v42"+e+"_"+str(index)+".npy",np.array(fivsig))
        
#.save("v41_threesig.npy",np.array(threesigTS))