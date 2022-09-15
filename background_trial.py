import os, sys, glob, abc
import numpy as np
from matplotlib import pyplot as plt, colors
import mla
from mla.threeml import spectral
from mla.threeml.IceCubeLike import IceCubeLike
import imp
import astromodels
import warnings
import threeML
import pickle
from functools import partial

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
p.add_argument("--wrkdir", default="/data/condor_builds/users/jasonfan/3ml_sensitivity_v42/result_dir/", type=str,
               help="Output directory (default:res/data/condor_builds/users/jasonfan/3ml_sensitivity_v42/result_dir/ult_dir)")
p.add_argument("--nscramble", default=1000, type=int,
               help="Number of background only scrambles used "
               "to measure TS distribution (default=1000)")
p.add_argument("--dec", type=float,
               help="Dec")
p.add_argument("--extension", default=0, type=float,
               help="extension(degree)")
p.add_argument("--surfix",default="", type=str,
               help="surfix")
p.add_argument("--energysob", default=0, type=int,
               help="energysob")
args = p.parse_args()               
index = args.index ###gamma
wrkdir = args.wrkdir
nscramble = args.nscramble
dec=args.dec
surfix=args.surfix
extension=args.extension
energysob = args.energysob
if extension == 0:
    surfixextension = ""
else:
    surfixextension = "_"+str(extension)


injection_spectrum = spectral.PowerLaw(1e3,1,-index)


DATA_PATH = "/data/i3store/users/analyses/ps_tracks/version-004-p02"

# Read in all of the MC files 
sim_files = DATA_PATH + "/IC86*MC*npy"
sim = np.load([i for i in glob.glob(sim_files) if "2011" not in i][0])
#sim = mla.trimsim(sim,0.3)
sim=rf.append_fields(sim, 'sindec',
                     np.sin(sim['dec']),
                     usemask=False)


# Read in all of the data files
data_files = DATA_PATH + "/IC86_*exp.npy"
listofdata = []
data = read([i for i in glob.glob(data_files)])


# Set the angular error floor to 0.2 degrees
data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(2) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC86*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = np.sort(grl['stop'])[-1]-np.sort(grl['start'])[0]
background_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})

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

if extension == 0:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.PointSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['PointSource']['name'] = 'temp'
    config['PointSource']['ra'] = 0
    config['PointSource']['dec'] = np.deg2rad(dec)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    
    source = mla.PointSource(config['PointSource'])

else:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.GaussianExtendedSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['GaussianExtendedSource']['name'] = 'temp'
    config['GaussianExtendedSource']['ra'] = 0
    config['GaussianExtendedSource']['dec'] = np.deg2rad(dec)
    config['GaussianExtendedSource']['sigma'] = np.deg2rad(extension)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    source = mla.GaussianExtendedSource(config['GaussianExtendedSource'])
    
config['ThreeMLPSIRFEnergyTermFactory']['backgroundSOBoption']=energysob
data_handler = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl))
data_handler.injection_spectrum = injection_spectrum
spatial_term_factory = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler, source)
energy_term_factory = mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory(config['ThreeMLPSIRFEnergyTermFactory'], data_handler,source)
time_term_factory = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory,energy_term_factory])#,time_term_factory])
icecube=IceCubeLike("temp",data,data_handler,llh_factory,source,verbose=False,livetime = livetime)

#Loading all the data , MC and good run list
DATA_PATH = "/data/i3store/users/analyses/ps_tracks/version-004-p02"

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
data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(2) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC79*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = np.sort(grl['stop'])[-1]-np.sort(grl['start'])[0]
background_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
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
if extension == 0:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.PointSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['PointSource']['name'] = 'temp'
    config['PointSource']['ra'] = 0
    config['PointSource']['dec'] = np.deg2rad(dec)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    source = mla.PointSource(config['PointSource'])

else:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.GaussianExtendedSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['GaussianExtendedSource']['name'] = 'temp'
    config['GaussianExtendedSource']['ra'] = 0
    config['GaussianExtendedSource']['dec'] = np.deg2rad(dec)
    config['GaussianExtendedSource']['sigma'] = np.deg2rad(extension)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    source = mla.GaussianExtendedSource(config['GaussianExtendedSource'])
config['ThreeMLPSIRFEnergyTermFactory']['backgroundSOBoption']=energysob

data_handler_79 = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl))
data_handler_79.injection_spectrum = injection_spectrum    
spatial_term_factory_79 = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler_79, source)
energy_term_factory_79 = mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory(config['ThreeMLPSIRFEnergyTermFactory'], data_handler_79,source)
time_term_factory_79 = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory_79 = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory_79,energy_term_factory_79])#,time_term_factory_79])
icecube_79=IceCubeLike("temp_79",data,data_handler_79,llh_factory_79,source,verbose=False,livetime = livetime)
               
#Loading all the data , MC and good run list
DATA_PATH = "/data/i3store/users/analyses/ps_tracks/version-004-p02"

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
data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(3) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC59*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = np.sort(grl['stop'])[-1]-np.sort(grl['start'])[0]
background_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
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
if extension == 0:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.PointSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['PointSource']['name'] = 'temp'
    config['PointSource']['ra'] = 0
    config['PointSource']['dec'] = np.deg2rad(dec)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    source = mla.PointSource(config['PointSource'])

else:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.GaussianExtendedSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['GaussianExtendedSource']['name'] = 'temp'
    config['GaussianExtendedSource']['ra'] = 0
    config['GaussianExtendedSource']['dec'] = np.deg2rad(dec)
    config['GaussianExtendedSource']['sigma'] = np.deg2rad(extension)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    source = mla.GaussianExtendedSource(config['GaussianExtendedSource'])
config['ThreeMLPSIRFEnergyTermFactory']['backgroundSOBoption']=energysob
data_handler_59 = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl))
data_handler_59.injection_spectrum = injection_spectrum    

spatial_term_factory_59 = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler_59, source)
energy_term_factory_59 = mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory(config['ThreeMLPSIRFEnergyTermFactory'], data_handler_59,source)
time_term_factory_59 = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory_59 = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory_59,energy_term_factory_59])#,time_term_factory_59])
icecube_59=IceCubeLike("temp_59",data,data_handler_59,llh_factory_59,source,verbose=False,livetime = livetime)               
        
#Loading all the data , MC and good run list
DATA_PATH = "/data/i3store/users/analyses/ps_tracks/version-004-p02"

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
data['angErr'][data['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
sim['angErr'][sim['angErr']<np.deg2rad(0.2)] = np.deg2rad(0.2)
np.random.seed(2) #for reproduce
data['ra'] = np.random.uniform(0, 2*np.pi, size=len(data))
grlfile = DATA_PATH + "/GRL/IC40*_exp.npy"
grl = read([i for i in glob.glob(grlfile)])
livetime = np.sum(grl['livetime'])
bkg_days = np.sort(grl['stop'])[-1]-np.sort(grl['start'])[0]
background_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
inject_signal_time_profile = mla.time_profiles.UniformProfile({'start':np.sort(grl['start'])[0], 'length':bkg_days})
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
if extension == 0:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.PointSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['PointSource']['name'] = 'temp'
    config['PointSource']['ra'] = 0
    config['PointSource']['dec'] = np.deg2rad(dec)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    source = mla.PointSource(config['PointSource'])

else:
    config = mla.generate_default_config([
        mla.threeml.data_handlers.ThreeMLDataHandler,
        mla.GaussianExtendedSource,
        mla.SpatialTermFactory,
        mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory,
        mla.TimeTermFactory,
        mla.LLHTestStatisticFactory
    ])
    config['GaussianExtendedSource']['name'] = 'temp'
    config['GaussianExtendedSource']['ra'] = 0
    config['GaussianExtendedSource']['dec'] = np.deg2rad(dec)
    config['GaussianExtendedSource']['sigma'] = np.deg2rad(extension)
    config['ThreeMLDataHandler']['dec_bandwidth (rad)'] = np.deg2rad(1)
    config['ThreeMLDataHandler']['dec_cut_location']=np.deg2rad(dec)
    source = mla.GaussianExtendedSource(config['GaussianExtendedSource'])
config['ThreeMLPSIRFEnergyTermFactory']['backgroundSOBoption']=energysob

data_handler_40 = mla.threeml.data_handlers.ThreeMLDataHandler(
    config['ThreeMLDataHandler'], sim, (data, grl))
data_handler_40.injection_spectrum = injection_spectrum 
spatial_term_factory_40 = mla.SpatialTermFactory(config['SpatialTermFactory'], data_handler_40, source)
energy_term_factory_40 = mla.threeml.sob_terms.ThreeMLPSIRFEnergyTermFactory(config['ThreeMLPSIRFEnergyTermFactory'], data_handler_40,source)
time_term_factory_40 = mla.TimeTermFactory(config['TimeTermFactory'],background_time_profile,inject_signal_time_profile)
llh_factory_40 = mla.LLHTestStatisticFactory(config['LLHTestStatisticFactory'],[spatial_term_factory_40,energy_term_factory_40])#,time_term_factory_40])
icecube_40=IceCubeLike("temp_40",data,data_handler_40,llh_factory_40,source,verbose=False,livetime = livetime) 

warnings.filterwarnings("ignore")
result=np.empty((nscramble,4))
analysislist = mla.threeml.IceCubeLike.icecube_analysis([icecube,icecube_79,icecube_59,icecube_40])
fitfailed=0
folder="/BG"+surfixextension
try:
    os.mkdir(wrkdir+folder)
except FileExistsError:
    pass
    
if os.path.exists(wrkdir+folder+"/BGTrial_dec"+str(dec)+surfixextension+"_" + surfix+".npy"):
    quit()

analysislist.newton_flux_norm = True
grid_minimizer = GlobalMinimization("grid")
if analysislist.newton_flux_norm:
    local_minimizer = LocalMinimization("scipy")
    my_grid = {'TXS.spectrum.main.Powerlaw.index': np.linspace(-1.1, -3.99, 5)}
else:
    local_minimizer = LocalMinimization("minuit")
    my_grid = {'TXS.spectrum.main.Powerlaw.index': np.linspace(-2, -3.99, 3)}

grid_minimizer.setup(
    second_minimization=local_minimizer, grid=my_grid)

for ntrial in range(nscramble): 
    analysislist.injection(poisson=True)
    TXS_sp = astromodels.Powerlaw(piv=100e3) #In GeV.Setting a pivot energy is important
    
    TXS_sp.K.fix = analysislist.newton_flux_norm
    #otherwise the minimizer will not find the current flux norm and spectral index minimum(as they are correlated)
    TXS_sp.K.bounds = (1e-50,1e-15)
    TXS_sp.index.bounds = (-4,-1)
    TXS_sp.K = 1e-23
    TXS_sp.index = -2
    if extension == 0:
        TXS =  mla.threeml.IceCubeLike.NeutrinoPointSource("TXS", ra=0, dec=dec, spectral_shape=TXS_sp)
    else:
        TXS_spatial = astromodels.Gaussian_on_sphere()

        TXS = mla.threeml.IceCubeLike.NeutrinoExtendedSource("TXS",spatial_shape = TXS_spatial, spectral_shape=TXS_sp)
        TXS.spatial_shape.lon0=0
        TXS.spatial_shape.lon0.fix=True
        TXS.spatial_shape.lat0=dec
        TXS.spatial_shape.lat0.fix=True
        TXS.spatial_shape.sigma=extension
        TXS.spatial_shape.sigma.fix=True
    model = astromodels.Model(TXS)
    
    IceCubedata = threeML.DataList(analysislist)
    #IceCubedata = threeML.DataList(icecube_59)
    jl = threeML.JointLikelihood(model, IceCubedata)
    


    jl.set_minimizer(grid_minimizer)
    #jl.set_minimizer("minuit");
    try:
        jl.fit(quiet=True,compute_covariance=False)

        ns = 0
        for objecticecube in analysislist.listoficecubelike:
            ns += objecticecube.get_current_fit_ns()
        if -jl.current_minimum < 0:
            result[ntrial,0] = 0
            result[ntrial,1] = 0
            result[ntrial,2] = 4
            result[ntrial,1] = 0
            fitfailed +=1
        result[ntrial,0] = -jl.current_minimum
        result[ntrial,1] = analysislist.get_current_fit_ns()
        result[ntrial,2] = -jl.likelihood_model.TXS.spectrum.main.Powerlaw.index.value
        result[ntrial,3] = jl.likelihood_model.TXS.spectrum.main.Powerlaw.K.value
    except:
        try:
            if -jl.current_minimum < 0:
                result[ntrial,0] = 0
                result[ntrial,1] = 0
                result[ntrial,2] = 4
                result[ntrial,1] = 0
                fitfailed +=1
            result[ntrial,0] = -jl.current_minimum
            result[ntrial,1] = analysislist.get_current_fit_ns()
            result[ntrial,2] = -jl.likelihood_model.TXS.spectrum.main.Powerlaw.index.value
            result[ntrial,3] = jl.likelihood_model.TXS.spectrum.main.Powerlaw.K.value
        except:
            result[ntrial,0] = 0
            result[ntrial,1] = 0
            result[ntrial,2] = 4
            result[ntrial,3] = 0
            fitfailed +=1
        print("failed : " + str(fitfailed) +" in " + str(ntrial+1) + "trials")

    
np.save(wrkdir+folder+"/BGTrial_dec"+str(dec)+surfixextension+"_" + surfix+".npy",result)