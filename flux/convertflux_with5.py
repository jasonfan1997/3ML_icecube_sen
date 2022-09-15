import os, sys, glob, abc
import numpy as np
from matplotlib import pyplot as plt, colors
import mla
from mla.threeml import spectral
import imp
import astromodels
import warnings
import threeML
import pickle
from functools import partial
from mla.threeml.IceCubeLike import IceCubeLike
from threeML import *
import numpy.lib.recfunctions as rf
def read(filelist):
    data = []

    for f in sorted(filelist):
        x = np.load(f)
        if len(data) == 0: data = x.copy()
        else: data = np.concatenate([data, x])

    try:
        data=rf.append_fields(data, 'sindec',
                              np.sin(data['dec']),
                              usemask=False)
    except:
        pass
    return data
import argparse

p = argparse.ArgumentParser(description="Calculates Sensitivity and Discovery",
                            formatter_class=argparse.RawTextHelpFormatter)
p.add_argument("--index", default=2.0, type=float,
               help="Spectral Index (default=2.0)")
p.add_argument("--wrkdir", default="/data/condor_builds/users/jasonfan/3ml_sensitivity/result_dir/", type=str,
               help="Output directory (default:res/data/condor_builds/users/jasonfan/3ml_sensitivity/result_dir/ult_dir)")
p.add_argument("--dec", type=float,
               help="Dec")

args = p.parse_args()               
index = args.index ###gamma
wrkdir = args.wrkdir
dec=args.dec

sen = np.load("/data/condor_builds/users/jasonfan/3ml_sensitivity/flux/sen.npy")
threesig = np.load("/data/condor_builds/users/jasonfan/3ml_sensitivity/flux/threesig.npy")
fivsig = np.load("/data/condor_builds/users/jasonfan/3ml_sensitivity/flux/fivsig.npy")
declist = np.arange(-85,85,2.5)
decindex=np.argwhere(declist==dec)[0][0]
print("Injected neutrino sen :{} , 3 sigma discovery: {}, 5 sigma discovery:　{}".format(round(sen[decindex],3),round(threesig[decindex],3),round(fivsig[decindex],3)))

injection_spectrum = spectral.PowerLaw(1e3,1,-index)
#Loading all the data , MC and good run list
DATA_PATH = "/data/i3store/users/mjlarson/ps_tracks/version-004-p00"

# Read in all of the MC files 
sim_files = DATA_PATH + "/IC86*MC*npy"
sim = np.load([i for i in glob.glob(sim_files) if "2011" not in i][0])
sim=rf.append_fields(sim, 'sindec',
                     np.sin(sim['dec']),
                     usemask=False)


# Read in all of the data files
data_files = DATA_PATH + "/IC86_*exp.npy"
listofdata = []
data = read([i for i in glob.glob(data_files)])


# Set the angular error floor to 0.2 degrees
#data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(2) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC86*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = grl['stop'][-1]-grl['start'][0]
background_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
if 'sindec' not in data.dtype.names:
    data = rf.append_fields(
        data,
        'sindec',
        np.sin(data['dec']),
        usemask=False,
    )
if 'sindec' not in sim.dtype.names:
    sim = rf.append_fields(
        sim,
        'sindec',
        np.sin(sim['dec']),
        usemask=False,
    )
config = mla.generate_default_config([
    mla.threeml.data_handlers.ThreeMLDataHandler,
    mla.PointSource,
    mla.SpatialTermFactory,
    mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory,
    mla.TimeTermFactory,
    mla.LLHTestStatisticFactory
])
config['PointSource']['name'] = 'temp'
config['PointSource']['ra'] = mla.ra_to_rad(5, 9, 25.9645434784)
config['PointSource']['dec'] = np.deg2rad(dec)
config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(3)
config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
source = mla.PointSource(config['PointSource'])

data_handler = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl),background_time_profile=background_time_profile,signal_time_profile=inject_signal_time_profile)
data_handler.injection_spectrum = injection_spectrum
spatial_term_factory = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler, source)
energy_term_factory = mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory(config['ThreeMLPSEnergyTermFactory'], data_handler,source)
time_term_factory = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory,energy_term_factory,time_term_factory])
icecube=IceCubeLike("temp",data,data_handler,llh_factory,source,verbose=False)

#Loading all the data , MC and good run list
DATA_PATH = "/data/i3store/users/mjlarson/ps_tracks/version-004-p00"

# Read in all of the MC files 
sim_files = DATA_PATH + "/IC79*MC*npy"
sim = np.load([i for i in glob.glob(sim_files)][0])
#n_keep = int(0.5*len(sim)) #server don't have that much memory for me to 
#sim = np.random.choice(sim, n_keep)
sim=rf.append_fields(sim, 'sindec',
                     np.sin(sim['dec']),
                     usemask=False)


# Read in all of the data files
data_files = DATA_PATH + "/IC79_*exp.npy"
listofdata = []
data = read([i for i in glob.glob(data_files)])


# Set the angular error floor to 0.2 degrees
#data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(2) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC79*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = grl['stop'][-1]-grl['start'][0]
background_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
if 'sindec' not in data.dtype.names:
    data = rf.append_fields(
        data,
        'sindec',
        np.sin(data['dec']),
        usemask=False,
    )
if 'sindec' not in sim.dtype.names:
    sim = rf.append_fields(
        sim,
        'sindec',
        np.sin(sim['dec']),
        usemask=False,
    )
config = mla.generate_default_config([
    mla.threeml.data_handlers.ThreeMLDataHandler,
    mla.PointSource,
    mla.SpatialTermFactory,
    mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory,
    mla.TimeTermFactory,
    mla.LLHTestStatisticFactory
])
config['PointSource']['name'] = 'temp_79'
config['PointSource']['ra'] = mla.ra_to_rad(5, 9, 25.9645434784)
config['PointSource']['dec'] = np.deg2rad(dec)
config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(3)
config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
source = mla.PointSource(config['PointSource'])

data_handler_79 = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl),background_time_profile=background_time_profile,signal_time_profile=inject_signal_time_profile)
data_handler_79.injection_spectrum = injection_spectrum
spatial_term_factory_79 = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler_79, source)
energy_term_factory_79 = mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory(config['ThreeMLPSEnergyTermFactory'], data_handler_79,source)
time_term_factory_79 = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory_79 = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory_79,energy_term_factory_79,time_term_factory_79])
icecube_79=IceCubeLike("temp_79",data,data_handler_79,llh_factory_79,source,verbose=False)
               
#Loading all the data , MC and good run list
DATA_PATH = "/data/i3store/users/mjlarson/ps_tracks/version-004-p00"

# Read in all of the MC files 
sim_files = DATA_PATH + "/IC59*MC*npy"
sim = np.load([i for i in glob.glob(sim_files)][0])
#n_keep = int(0.5*len(sim)) #server don't have that much memory for me to 
#sim = np.random.choice(sim, n_keep)
sim=rf.append_fields(sim, 'sindec',
                     np.sin(sim['dec']),
                     usemask=False)


# Read in all of the data files
data_files = DATA_PATH + "/IC59_*exp.npy"
listofdata = []
data = read([i for i in glob.glob(data_files)])


# Set the angular error floor to 0.2 degrees
#data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(3) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC59*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = grl['stop'][-1]-grl['start'][0]
background_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
if 'sindec' not in data.dtype.names:
    data = rf.append_fields(
        data,
        'sindec',
        np.sin(data['dec']),
        usemask=False,
    )
if 'sindec' not in sim.dtype.names:
    sim = rf.append_fields(
        sim,
        'sindec',
        np.sin(sim['dec']),
        usemask=False,
    )
config = mla.generate_default_config([
    mla.threeml.data_handlers.ThreeMLDataHandler,
    mla.PointSource,
    mla.SpatialTermFactory,
    mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory,
    mla.TimeTermFactory,
    mla.LLHTestStatisticFactory
])
config['PointSource']['name'] = 'temp_59'
config['PointSource']['ra'] = mla.ra_to_rad(5, 9, 25.9645434784)
config['PointSource']['dec'] = np.deg2rad(dec)
config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(3)
config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
source = mla.PointSource(config['PointSource'])


data_handler_59 = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl),background_time_profile=background_time_profile,signal_time_profile=inject_signal_time_profile)
data_handler_59.injection_spectrum = injection_spectrum
spatial_term_factory_59 = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler_59, source)
energy_term_factory_59 = mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory(config['ThreeMLPSEnergyTermFactory'], data_handler_59,source)
time_term_factory_59 = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory_59 = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory_59,energy_term_factory_59,time_term_factory_59])
icecube_59=IceCubeLike("temp_59",data,data_handler_59,llh_factory_59,source,verbose=False)               
        
#Loading all the data , MC and good run list
DATA_PATH = "/data/i3store/users/mjlarson/ps_tracks/version-004-p00"

# Read in all of the MC files 
sim_files = DATA_PATH + "/IC40*MC*npy"
sim = np.load([i for i in glob.glob(sim_files)][0])
#n_keep = int(0.5*len(sim)) #server don't have that much memory for me to 
#sim = np.random.choice(sim, n_keep)
sim=rf.append_fields(sim, 'sindec',
                     np.sin(sim['dec']),
                     usemask=False)


# Read in all of the data files
data_files = DATA_PATH + "/IC40_*exp.npy"
listofdata = []
data = read([i for i in glob.glob(data_files)])


# Set the angular error floor to 0.2 degrees
#data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(2) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC40*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = grl['stop'][-1]-grl['start'][0]
background_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':grl['start'][0], 'length':bkg_days})
if 'sindec' not in data.dtype.names:
    data = rf.append_fields(
        data,
        'sindec',
        np.sin(data['dec']),
        usemask=False,
    )
if 'sindec' not in sim.dtype.names:
    sim = rf.append_fields(
        sim,
        'sindec',
        np.sin(sim['dec']),
        usemask=False,
    )
config = mla.generate_default_config([
    mla.threeml.data_handlers.ThreeMLDataHandler,
    mla.PointSource,
    mla.SpatialTermFactory,
    mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory,
    mla.TimeTermFactory,
    mla.LLHTestStatisticFactory
])
config['PointSource']['name'] = 'temp_40'
config['PointSource']['ra'] = mla.ra_to_rad(5, 9, 25.9645434784)
config['PointSource']['dec'] = np.deg2rad(dec)
config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(3)
config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
source = mla.PointSource(config['PointSource'])

data_handler_40 = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl),background_time_profile=background_time_profile,signal_time_profile=inject_signal_time_profile)
data_handler_40.injection_spectrum = injection_spectrum
spatial_term_factory_40 = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler_40, source)
energy_term_factory_40 = mla.threeml.sob_terms.ThreeMLPSEnergyTermFactory(config['ThreeMLPSEnergyTermFactory'], data_handler_40,source)
time_term_factory_40 = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory_40 = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory_40,energy_term_factory_40,time_term_factory_40])
icecube_40=IceCubeLike("temp_40",data,data_handler_40,llh_factory_40,source,verbose=False)   

warnings.filterwarnings("ignore")
analysislist = mla.threeml.IceCubeLike.icecube_analysis([icecube,icecube_79,icecube_59,icecube_40])



senflux = analysislist.cal_injection_fluxnorm(sen[decindex])
threesigflux = analysislist.cal_injection_fluxnorm(threesig[decindex])
fivesigflux = analysislist.cal_injection_fluxnorm(fivsig[decindex])
print("Flux sen :{} , 3 sigma discovery: {}, 5 sigma discovery: {}".format(senflux,threesigflux,fivesigflux))
result=np.array([dec,sen[decindex],threesig[decindex],fivsig[decindex],senflux,threesigflux,fivesigflux])
np.save("/data/condor_builds/users/jasonfan/3ml_sensitivity/flux/result/"+str(dec)+".npy",result)
#np.save("/data/condor_builds/users/jasonfan/3ml_sensitivity/flux/"+str(dec)+".npy",result)

