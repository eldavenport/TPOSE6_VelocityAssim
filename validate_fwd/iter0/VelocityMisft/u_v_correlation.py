import sys
sys.path.append('../../')
import xarray as xr
from open_tpose import tpose2012to2016
import numpy as np
import warnings
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
from matplotlib.colors import TwoSlopeNorm
import cmocean.cm as cmo
import matplotlib.colors as colors
from scipy import stats
from scipy.signal import detrend, butter, sosfiltfilt
plt.rcParams['font.size'] = 13
import numpy.ma as ma
warnings.filterwarnings("ignore")
plt.rcParams['font.size'] = 13

prefix = ['diag_state']

ds = tpose2012to2016(prefix)

N = len(ds.time)
ds['time'] = range(0,N,1)
ds['XC'] = ds.XC.astype(float)
ds['YC'] = ds.YC.astype(float)
ds['Z'] = ds.Z.astype(float)
ds['XG'] = ds.XG.astype(float)
ds['YG'] = ds.YG.astype(float)

# ---------------------------------------------------------------TAO Data ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N140W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)

dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
U_TAO = dsTAO.u_1205.transpose('time','depth','lat','lon')
U_TAO = U_TAO/100 #convert from cm/s to m/s
U_TAO.data[U_TAO.data > 50] = np.nan # change 9999s to nans
latidx = 0
lonTAO140 = 0

U_TAO_140 = U_TAO[:,:,latidx,lonTAO140]

# sample these locations from the TPOSE data
U6_140 = ds.UVEL.interp(XG=[220.0],YC=[U_TAO_140.lat],Z=U_TAO_140.depth,time=U_TAO_140.time,method='linear')

temp = U6_140.values
U6_140 = U_TAO_140.copy(deep=True)
U6_140.values = temp[:,:,0,0]
U6_140 = U6_140 + U_TAO_140 - U_TAO_140

V_TAO = dsTAO.v_1206.transpose('time','depth','lat','lon')
V_TAO = V_TAO/100 #convert from cm/s to m/s
V_TAO.data[V_TAO.data > 50] = np.nan # change 9999s to nans

V_TAO_140 = V_TAO[:,:,latidx,lonTAO140]

# sample these locations from the TPOSE data
V6_140 = ds.VVEL.interp(XC=[220.0],YG=[V_TAO_140.lat],Z=V_TAO_140.depth,time=V_TAO_140.time,method='linear')

temp = V6_140.values
V6_140 = V_TAO_140.copy(deep=True)
V6_140.values = temp[:,:,0,0]
V6_140 = V6_140 + V_TAO_140 - V_TAO_140

# ---------------------------------------------------------------TAO Data ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N140W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)
latidx = 0
lonTAO140 = 0

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
Tdepths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_140 = T_TAO[:,:,latidx,lonTAO140]

# sample these locations from the TPOSE data
T6_140 = ds.THETA.interp(XC=[220.0],YC=[T_TAO_140.lat],Z=T_TAO_140.depth,time=T_TAO_140.time,method='linear')

temp = T6_140.values
T6_140 = T_TAO_140.copy(deep=True)
T6_140.values = temp[:,:,0,0]
T6_140 = T6_140 + T_TAO_140 - T_TAO_140

zMax = -35
zMin = -250
Udepthli = np.argmin(np.abs(depths - zMax))
Udepthui = np.argmin(np.abs(depths - zMin)) + 1
Tdepthli = np.argmin(np.abs(Tdepths - zMax))
Tdepthui = np.argmin(np.abs(Tdepths - zMin)) + 1

depth_levels = np.array([-30,-70,-110,-150,-200])
depth_level_idx_140 = np.zeros_like(depth_levels)
depth_level_idxT_140 = np.zeros_like(depth_levels)
i=0
for level in depth_levels:
    depth_level_idx_140[i] = np.argmin(np.abs(depths - level))
    depth_level_idxT_140[i] = np.argmin(np.abs(Tdepths - level))
    i = i+1

# ----------------------------------------------------------- 170W ------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N170W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
U_TAO = dsTAO.u_1205.transpose('time','depth','lat','lon')
U_TAO = U_TAO/100 #convert from cm/s to m/s
U_TAO.data[U_TAO.data > 50] = np.nan # change 9999s to nans
latidx = 0
lonTAO170 = 0

U_TAO_170 = U_TAO[:,:,latidx,lonTAO170]

# sample these locations from the TPOSE data
U6_170 = ds.UVEL.interp(XG=[190.0],YC=[U_TAO_170.lat],Z=U_TAO_170.depth,time=U_TAO_170.time,method='linear')

temp = U6_170.values
U6_170 = U_TAO_170.copy(deep=True)
U6_170.values = temp[:,:,0,0]
U6_170 = U6_170 + U_TAO_170 - U_TAO_170

V_TAO = dsTAO.v_1206.transpose('time','depth','lat','lon')
V_TAO = V_TAO/100 #convert from cm/s to m/s
V_TAO.data[V_TAO.data > 50] = np.nan # change 9999s to nans

V_TAO_170 = V_TAO[:,:,latidx,lonTAO170]

# sample these locations from the TPOSE data
V6_170 = ds.VVEL.interp(XC=[220.0],YG=[V_TAO_170.lat],Z=V_TAO_170.depth,time=V_TAO_170.time,method='linear')

temp = V6_170.values
V6_170 = V_TAO_170.copy(deep=True)
V6_170.values = temp[:,:,0,0]
V6_170 = V6_170 + V_TAO_170 - V_TAO_170

print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N170W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)
latidx = 0
lonTAO170 = 0

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
Tdepths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_170 = T_TAO[:,:,latidx,lonTAO170]

# sample these locations from the TPOSE data
T6_170 = ds.THETA.interp(XC=[190.0],YC=[T_TAO_170.lat],Z=T_TAO_170.depth,time=T_TAO_170.time,method='linear')

temp = T6_170.values
T6_170 = T_TAO_170.copy(deep=True)
T6_170.values = temp[:,:,0,0]
T6_170 = T6_170 + T_TAO_170 - T_TAO_170

depth_level_idx_170 = np.zeros_like(depth_levels)
depth_level_idxT_170 = np.zeros_like(depth_levels)
i=0
for level in depth_levels:
    depth_level_idx_170[i] = np.argmin(np.abs(depths - level))
    depth_level_idxT_170[i] = np.argmin(np.abs(Tdepths - level))
    i = i+1
# ---------------------------------------------------------------110W ---------------------------------------------------------------------------------
print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/ADCP_2012to2016_0N110W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
depths = dsTAO.depth.data
U_TAO = dsTAO.u_1205.transpose('time','depth','lat','lon')
U_TAO = U_TAO/100 #convert from cm/s to m/s
U_TAO.data[U_TAO.data > 50] = np.nan # change 9999s to nans
latidx = 0
lonTAO110 = 0

U_TAO_110 = U_TAO[:,:,latidx,lonTAO110]

# sample these locations from the TPOSE data
U6_110 = ds.UVEL.interp(XG=[250.0],YC=[U_TAO_110.lat],Z=U_TAO_110.depth,time=U_TAO_110.time,method='linear')

temp = U6_110.values
U6_110 = U_TAO_110.copy(deep=True)
U6_110.values = temp[:,:,0,0]
U6_110 = U6_110 + U_TAO_110 - U_TAO_110

V_TAO = dsTAO.v_1206.transpose('time','depth','lat','lon')
V_TAO = V_TAO/100 #convert from cm/s to m/s
V_TAO.data[V_TAO.data > 50] = np.nan # change 9999s to nans

V_TAO_110 = V_TAO[:,:,latidx,lonTAO110]

# sample these locations from the TPOSE data
V6_110 = ds.VVEL.interp(XC=[220.0],YG=[V_TAO_110.lat],Z=V_TAO_110.depth,time=V_TAO_110.time,method='linear')

temp = V6_110.values
V6_110 = V_TAO_110.copy(deep=True)
V6_110.values = temp[:,:,0,0]
V6_110 = V6_110 + V_TAO_110 - V_TAO_110

print('Starting TAO')
TAO_file = '/data/SO3/edavenport/TAO_2012to2016_daily/T_2012to2016_0N110W_daily.cdf' # Get the right TAO data 
dsTAO = xr.open_dataset(TAO_file,decode_times=False)
n = len(dsTAO.time)
latidx = 0
lonTAO110 = 0

if n < N:
    diff = N - n
    dsTAO['time'] = range(diff,N)
else:
    dsTAO['time'] = range(0,n)

dsTAO['depth'] = -1*dsTAO.depth
Tdepths = dsTAO.depth.data
T_TAO = dsTAO.T_20.transpose('time','depth','lat','lon')
T_TAO.data[T_TAO.data > 50] = np.nan # change 9999s to nans

T_TAO_110 = T_TAO[:,:,latidx,lonTAO110]

# sample these locations from the TPOSE data
T6_110 = ds.THETA.interp(XC=[250.0],YC=[T_TAO_110.lat],Z=T_TAO_110.depth,time=T_TAO_110.time,method='linear')

temp = T6_110.values
T6_110 = T_TAO_110.copy(deep=True)
T6_110.values = temp[:,:,0,0]
T6_110 = T6_110 + T_TAO_110 - T_TAO_110

depth_level_idx_110 = np.zeros_like(depth_levels)
depth_level_idxT_110 = np.zeros_like(depth_levels)
i=0
for level in depth_levels:
    depth_level_idx_110[i] = np.argmin(np.abs(depths - level))
    depth_level_idxT_110[i] = np.argmin(np.abs(Tdepths - level))
    i = i+1
# ---------------------------------------------------------------plotting  ---------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(22,22),nrows=5,ncols=3)
for i in range(len(depth_levels)):
    zon_corr = (ma.corrcoef(ma.masked_invalid(U_TAO_170[:,depth_level_idx_170[i]]),ma.masked_invalid(U6_170[:,depth_level_idx_170[i]])))
    # mer_corr = (ma.corrcoef(ma.masked_invalid(V_TAO_170[:,depth_level_idx_170[i]]),ma.masked_invalid(V6_170[:,depth_level_idx_170[i]])))
    ax[i,0].plot(U_TAO_170.time,U_TAO_170[:,depth_level_idx_170[i]],label='TAO U',color='tab:blue')
    label = 'TPOSE U' + ' (' + str(round(zon_corr[0,1],2)) + ')'
    ax[i,0].plot(U6_170.time,U6_170[:,depth_level_idx_170[i]],label=label,color='tab:orange')
    # ax[i,0].plot(V_TAO_170.time,V_TAO_170[:,depth_level_idx_170[i]],label='TAO V',color='tab:blue',linestyle='--')
    # label = 'TPOSE V' + ' (' + str(round(mer_corr[0,1],2)) + ')'
    # ax[i,0].plot(V6_170.time,V6_170[:,depth_level_idx_170[i]],label=label,color='tab:orange',linestyle='--')
    ax[i,0].set_title('0N,170W ' + str(U_TAO_170.depth[depth_level_idx_170[i]].data)+ ' m')
    ax[i,0].set_xlim(0,len(U_TAO_170.time))
    ax[i,0].set_xlabel('Time (days)')
    ax[i,0].set_ylabel('m/s')
    ax[i,0].legend(loc='upper right')
    ax[i,0].axhline(0.0,color='tab:gray',linewidth=0.75)
    ax[i,0].set_ylim(-0.75,1.5)

    zon_corr = (ma.corrcoef(ma.masked_invalid(U_TAO_140[:,depth_level_idx_140[i]]),ma.masked_invalid(U6_140[:,depth_level_idx_140[i]])))
    # mer_corr = (ma.corrcoef(ma.masked_invalid(V_TAO_140[:,depth_level_idx_140[i]]),ma.masked_invalid(V6_140[:,depth_level_idx_140[i]])))
    ax[i,1].plot(U_TAO_140.time,U_TAO_140[:,depth_level_idx_140[i]],label='TAO U',color='tab:blue')
    label = 'TPOSE U' + ' (' + str(round(zon_corr[0,1],2)) + ')'
    ax[i,1].plot(U6_140.time,U6_140[:,depth_level_idx_140[i]],label=label,color='tab:orange')
    # ax[i,1].plot(V_TAO_140.time,V_TAO_140[:,depth_level_idx_140[i]],label='TAO V',color='tab:blue',linestyle='--')
    # label = 'TPOSE V' + ' (' + str(round(mer_corr[0,1],2)) + ')'
    # ax[i,1].plot(V6_140.time,V6_140[:,depth_level_idx_140[i]],label=label,color='tab:orange',linestyle='--')
    ax[i,1].set_title('0N,140W ' +str(U_TAO_140.depth[depth_level_idx_140[i]].data)+ ' m')
    ax[i,1].set_xlim(0,len(U_TAO_140.time))
    ax[i,1].set_xlabel('Time (days)')
    ax[i,1].set_ylabel('m/s')
    ax[i,1].legend(loc='upper right')
    ax[i,1].axhline(0.0,color='tab:gray',linewidth=0.75)
    ax[i,1].set_ylim(-1.0,2.0)

    zon_corr = (ma.corrcoef(ma.masked_invalid(U_TAO_110[:,depth_level_idx_110[i]]),ma.masked_invalid(U6_110[:,depth_level_idx_110[i]])))
    # mer_corr = (ma.corrcoef(ma.masked_invalid(V_TAO_110[:,depth_level_idx_110[i]]),ma.masked_invalid(V6_110[:,depth_level_idx_110[i]])))
    ax[i,2].plot(U_TAO_110.time,U_TAO_110[:,depth_level_idx_110[i]],label='TAO U',color='tab:blue')
    label = 'TPOSE U' + ' (' + str(round(zon_corr[0,1],2)) + ')'
    ax[i,2].plot(U6_110.time,U6_110[:,depth_level_idx_110[i]],label=label,color='tab:orange')
    # ax[i,2].plot(V_TAO_110.time,V_TAO_110[:,depth_level_idx_110[i]],label='TAO V',color='tab:blue',linestyle='--')
    # label = 'TPOSE V' + ' (' + str(round(mer_corr[0,1],2)) + ')'
    # ax[i,2].plot(V6_110.time,V6_110[:,depth_level_idx_110[i]],label=label,color='tab:orange',linestyle='--')
    ax[i,2].set_title('0N,110W ' +str(U_TAO_110.depth[depth_level_idx_110[i]].data)+ ' m')
    ax[i,2].set_xlim(0,len(U_TAO_110.time))
    ax[i,2].set_xlabel('Time (days)')
    ax[i,2].set_ylabel('m/s')
    ax[i,2].legend(loc='upper right')
    ax[i,2].axhline(0.0,color='tab:gray',linewidth=0.75)
    ax[i,2].set_ylim(-0.75,2.5)

plt.tight_layout()
plt.savefig('U_time_series_2012to2016.png',format='png')

fig, ax = plt.subplots(figsize=(22,22),nrows=5,ncols=3)
for i in range(len(depth_levels)):
    t_corr = ma.corrcoef(ma.masked_invalid(T_TAO_170[:,depth_level_idxT_170[i]]),ma.masked_invalid(T6_170[:,depth_level_idxT_170[i]]))
    ax[i,0].plot(T_TAO_170.time,T_TAO_170[:,depth_level_idxT_170[i]],label='TAO T',color='tab:blue')
    label = 'TPOSE T' + ' (' + str(round(t_corr[0,1],2)) + ')'
    ax[i,0].plot(T6_170.time,T6_170[:,depth_level_idxT_170[i]],label=label,color='tab:orange')
    ax[i,0].set_title('0N,170W '+str(T_TAO_170.depth[depth_level_idxT_170[i]].data)+ ' m')
    ax[i,0].set_xlabel('Time (days)')
    ax[i,0].set_ylabel('deg C')
    ax[i,0].legend(loc='upper right')
    ax[i,0].set_xlim(0,len(T_TAO_170.time))

    t_corr = ma.corrcoef(ma.masked_invalid(T_TAO_140[:,depth_level_idxT_140[i]]),ma.masked_invalid(T6_140[:,depth_level_idxT_140[i]]))
    ax[i,1].plot(T_TAO_140.time,T_TAO_140[:,depth_level_idxT_140[i]],label='TAO T',color='tab:blue')
    label = 'TPOSE T' + ' (' + str(round(t_corr[0,1],2)) + ')'
    ax[i,1].plot(T6_140.time,T6_140[:,depth_level_idxT_140[i]],label=label,color='tab:orange')
    ax[i,1].set_title('0N,140W '+str(T_TAO_140.depth[depth_level_idxT_140[i]].data)+ ' m')
    ax[i,1].set_xlabel('Time (days)')
    ax[i,1].set_ylabel('deg C')
    ax[i,1].legend(loc='upper right')
    ax[i,1].set_xlim(0,len(T_TAO_140.time))

    t_corr = ma.corrcoef(ma.masked_invalid(T_TAO_110[:,depth_level_idxT_110[i]]),ma.masked_invalid(T6_110[:,depth_level_idxT_110[i]]))
    ax[i,2].plot(T_TAO_110.time,T_TAO_110[:,depth_level_idxT_110[i]],label='TAO T',color='tab:blue')
    label = 'TPOSE T' + ' (' + str(round(t_corr[0,1],2)) + ')'
    ax[i,2].plot(T6_110.time,T6_110[:,depth_level_idxT_110[i]],label=label,color='tab:orange')
    ax[i,2].set_title('0N,110W '+str(T_TAO_110.depth[depth_level_idxT_110[i]].data)+ ' m')
    ax[i,2].set_xlabel('Time (days)')
    ax[i,2].set_ylabel('deg C')
    ax[i,2].legend(loc='upper right')
    ax[i,2].set_xlim(0,len(T_TAO_110.time))

plt.tight_layout()
plt.savefig('T_time_series_2012to2016.png',format='png')

# crop the time series to the depths we are interested in and the first two years (after that there are some large gaps in TAO data)
U6_140 = U6_140[:,Udepthli:Udepthui]
U_TAO_140 = U_TAO_140[:,Udepthli:Udepthui]

V6_140 = V6_140[:,Udepthli:Udepthui]
V_TAO_140 = V_TAO_140[:,Udepthli:Udepthui]
depths = depths[Udepthli:Udepthui]

# T6_140 = T6_140[:,Tdepthli:Tdepthui]
# T_TAO_140 = T_TAO_140[:,Tdepthli:Tdepthui]
# Tdepths = Tdepths[Tdepthli:Tdepthui]

for z in range(len(depths)):
    signal = V_TAO_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    V_TAO_140[:,z] = signal
    signal = U_TAO_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    U_TAO_140[:,z] = signal
    signal = V6_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    V6_140[:,z] = signal
    signal = U6_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    U6_140[:,z] = signal

for z in range(len(Tdepths)):
    signal = T_TAO_140[:,z]
    signal[np.logical_not(np.isnan(signal))] = detrend(signal[np.logical_not(np.isnan(signal))])
    T_TAO_140[:,z] = signal

V_anom = V_TAO_140 - np.nanmean(V_TAO_140,axis=0)
Vprime_filt = V_anom

U_anom = U_TAO_140 - np.nanmean(U_TAO_140,axis=0)
Uprime_filt = U_anom

V_TP_anom = V6_140 - np.nanmean(V6_140,axis=0)
Vprime_TP_filt = V_TP_anom

U_TP_anom = U6_140 - np.nanmean(U6_140,axis=0)
Uprime_TP_filt = U_TP_anom

T_TP_anom = T6_140 - np.nanmean(T6_140,axis=0)
Tprime_TP_filt = T_TP_anom

T_anom = T_TAO_140 - np.nanmean(T_TAO_140,axis=0)
Tprime_filt = T_anom

temp = U6_140.copy(deep=True)
temp.values = Uprime_filt
Uprime_filt = temp

temp = V6_140.copy(deep=True)
temp.values = Vprime_filt
Vprime_filt = temp

temp = U6_140.copy(deep=True)
temp.values = Uprime_TP_filt
Uprime_TP_filt = temp

temp = V6_140.copy(deep=True)
temp.values = Vprime_TP_filt
Vprime_TP_filt = temp

temp = T6_140.copy(deep=True)
temp.values = Tprime_TP_filt
Tprime_TP_filt = temp

temp = T6_140.copy(deep=True)
temp.values = Tprime_filt
Tprime_filt = temp

U_corr = xr.corr(U6_140, U_TAO_140, dim="time") # correlation in time of the zonal velocity at every depth
V_corr = xr.corr(V6_140, V_TAO_140, dim="time") # correlation in time of the meridional velocity at every depth
T_corr = xr.corr(T6_140, T_TAO_140, dim="time") # correlation in time of the temperature at every depth

U_corr_170 = xr.corr(U6_170, U_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth
V_corr_170 = xr.corr(V6_170, V_TAO_170, dim="time") # correlation in time of the meridional velocity at every depth
T_corr_170 = xr.corr(T6_170, T_TAO_170, dim="time") # correlation in time of the temperature at every depth

U_corr_110 = xr.corr(U6_110, U_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth
V_corr_110 = xr.corr(V6_110, V_TAO_110, dim="time") # correlation in time of the meridional velocity at every depth
T_corr_110 = xr.corr(T6_110, T_TAO_110, dim="time") # correlation in time of the temperature at every depth

fig, ax = plt.subplots(figsize=(15,7),ncols=3)
ax[0].plot(U_corr_170,U_TAO_170.depth,linewidth=2.0,label='u')
ax[0].plot(V_corr_170,V_TAO_170.depth,linewidth=2.0,label='v')
ax[0].plot(T_corr_170,T_TAO_170.depth,linewidth=2.0,label='T')
ax[0].set_xlabel('Correlation')
ax[0].set_ylabel('Depth (m)')
ax[0].set_title('0N,170W')
ax[0].legend()
ax[0].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[0].set_xlim(-0.5,1)
ax[0].set_ylim(-250,0)

ax[1].plot(U_corr,U_TAO_140.depth,linewidth=2.0,label='u')
ax[1].plot(V_corr,V_TAO_140.depth,linewidth=2.0,label='v')
ax[1].plot(T_corr,T_TAO_140.depth,linewidth=2.0,label='T')
ax[1].set_xlabel('Correlation')
ax[1].set_ylabel('Depth (m)')
ax[1].set_title('0N,140W')
ax[1].legend()
ax[1].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[1].set_xlim(-0.5,1)
ax[1].set_ylim(-250,0)

ax[2].plot(U_corr_110,U_TAO_110.depth,linewidth=2.0,label='u')
ax[2].plot(V_corr_110,V_TAO_110.depth,linewidth=2.0,label='v')
ax[2].plot(T_corr_110,T_TAO_110.depth,linewidth=2.0,label='T')
ax[2].set_xlabel('Correlation')
ax[2].set_ylabel('Depth (m)')
ax[2].set_title('0N,110W')
ax[2].legend()
ax[2].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[2].set_xlim(-0.5,1)
ax[2].set_ylim(-250,0)

plt.tight_layout()
plt.savefig('V_U_T_correlation_by_depth.png',format='png')

frac_error_var_U = (U6_140 - U_TAO_140).var(dim='time') / U_TAO_140.var(dim='time') # should be a function of depth
frac_error_var_V = (V6_140 - V_TAO_140).var(dim='time') / V_TAO_140.var(dim='time')
frac_error_var_T = (T6_140 - T_TAO_140).var(dim='time') / T_TAO_140.var(dim='time')

fig, ax = plt.subplots(figsize=(5,10))
ax.plot(frac_error_var_U,U_TAO_140.depth,label='U')
ax.plot(frac_error_var_V,V_TAO_140.depth,label='V')
ax.plot(frac_error_var_T,T_TAO_140.depth,label='T')
ax.set_xlabel('Fractional Error Variance')
ax.set_ylabel('Depth (m)')
ax.set_xlim(0,1.55)
ax.legend()
ax.set_ylim(-250,0)

plt.tight_layout()
plt.savefig('V_U_T_fracErrorVar.png',format='png')

# should show a combined plot with frac error var, correlation, and u'v' TPOSE with u'v' TAO
uv_flux_TP = Uprime_TP_filt*Vprime_TP_filt
uv_flux_TAO = Uprime_filt*Vprime_filt
uu_TP = Uprime_TP_filt*Uprime_TP_filt
uu_TAO = Uprime_filt*Uprime_filt
vv_TP = Vprime_TP_filt*Vprime_TP_filt
vv_TAO = Vprime_filt*Vprime_filt

fig, ax = plt.subplots(figsize=(5,10))
ax.plot(uv_flux_TP.mean(dim='time'),U6_140.depth,label='TPOSE u\'v\'',color='tab:orange')
ax.fill_betweenx(U6_140.depth,uv_flux_TP.mean(dim='time')+uv_flux_TP.std(dim='time'),uv_flux_TP.mean(dim='time')-uv_flux_TP.std(dim='time'),color='tab:orange',label='_nolegend_',alpha=0.25)
ax.plot(uv_flux_TAO.mean(dim='time'),V_TAO_140.depth,label='TAO u\'v\'',color='tab:blue')
ax.fill_betweenx(V_TAO_140.depth,uv_flux_TAO.mean(dim='time')+uv_flux_TAO.std(dim='time'),uv_flux_TAO.mean(dim='time')-uv_flux_TAO.std(dim='time'),color='tab:blue',label='_nolegend_',alpha=0.25)
ax.plot(uu_TP.mean(dim='time'),U6_140.depth,label='TPOSE $u\'^2$',color='tab:orange',linestyle='--')
ax.fill_betweenx(U6_140.depth,uu_TP.mean(dim='time')+uu_TP.std(dim='time'),uu_TP.mean(dim='time')-uu_TP.std(dim='time'),color='tab:orange',label='_nolegend_',alpha=0.25,linestyle='--')
ax.plot(uu_TAO.mean(dim='time'),V_TAO_140.depth,label='TAO $u\'^2$',color='tab:blue',linestyle='--')
ax.fill_betweenx(V_TAO_140.depth,uu_TAO.mean(dim='time')+uu_TAO.std(dim='time'),uu_TAO.mean(dim='time')-uu_TAO.std(dim='time'),color='tab:blue',label='_nolegend_',alpha=0.25,linestyle='--')
ax.plot(vv_TP.mean(dim='time'),U6_140.depth,label='TPOSE $v\'^2$',color='tab:orange',linestyle=':')
ax.fill_betweenx(U6_140.depth,vv_TP.mean(dim='time')+vv_TP.std(dim='time'),vv_TP.mean(dim='time')-vv_TP.std(dim='time'),color='tab:orange',label='_nolegend_',alpha=0.25,linestyle=':')
ax.plot(vv_TAO.mean(dim='time'),V_TAO_140.depth,label='TAO $v\'^2$',color='tab:blue',linestyle=':')
ax.fill_betweenx(V_TAO_140.depth,vv_TAO.mean(dim='time')+vv_TAO.std(dim='time'),vv_TAO.mean(dim='time')-vv_TAO.std(dim='time'),color='tab:blue',label='_nolegend_',alpha=0.25,linestyle=':')
ax.set_xlabel('$m^2/s^2$')
ax.set_ylabel('Depth (m)')
ax.legend()
ax.set_ylim(-250,0)

plt.tight_layout()
plt.savefig('V_U_flux_uu_vv_profile.png',format='png')

fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(22,6))
uv_flux_TP.T.plot(x='time',y='depth',ax=ax[0,0],cmap=cmo.balance,vmin=-0.4,vmax=0.4,cbar_kwargs={'label':'$m^2/s^2$'})
ax[0,0].set_xlabel('Time (days)')
ax[0,0].set_ylabel('Depth (m)')
ax[0,0].set_title('u\'v\' TPOSE')

uv_flux_TAO.T.plot(x='time',y='depth',ax=ax[1,0],cmap=cmo.balance,vmin=-0.4,vmax=0.4,cbar_kwargs={'label':'$m^2/s^2$'})
ax[1,0].set_xlabel('Time (days)')
ax[1,0].set_ylabel('Depth (m)')
ax[1,0].set_title('u\'v\' TAO')

uu_TP.T.plot(x='time',y='depth',ax=ax[0,1],cmap=cmo.balance,vmin=0.0,vmax=0.8,cbar_kwargs={'label':'$m^2/s^2$'})
ax[0,1].set_xlabel('Time (days)')
ax[0,1].set_ylabel('Depth (m)')
ax[0,1].set_title('${u\'}^2$ TPOSE')

uu_TAO.T.plot(x='time',y='depth',ax=ax[1,1],cmap=cmo.balance,vmin=0.0,vmax=0.8,cbar_kwargs={'label':'$m^2/s^2$'})
ax[1,1].set_xlabel('Time (days)')
ax[1,1].set_ylabel('Depth (m)')
ax[1,1].set_title('${u\'}^2$ TAO')

vv_TP.T.plot(x='time',y='depth',ax=ax[0,2],cmap=cmo.balance,vmin=0.0,vmax=0.25,cbar_kwargs={'label':'$m^2/s^2$'})
ax[0,2].set_xlabel('Time (days)')
ax[0,2].set_ylabel('Depth (m)')
ax[0,2].set_title('${v\'}^2$ TPOSE')

vv_TAO.T.plot(x='time',y='depth',ax=ax[1,2],cmap=cmo.balance,vmin=0.0,vmax=0.25,cbar_kwargs={'label':'$m^2/s^2$'})
ax[1,2].set_xlabel('Time (days)')
ax[1,2].set_ylabel('Depth (m)')
ax[1,2].set_title('${v\'}^2$ TAO')

plt.tight_layout()
plt.savefig('V_U_flux_uu_vv_2D.png',format='png')

fig, ax = plt.subplots(nrows=2,ncols=3,figsize=(22,6))
uv_flux_TP[:750].T.plot(x='time',y='depth',ax=ax[0,0],cmap=cmo.balance,vmin=-0.4,vmax=0.4,cbar_kwargs={'label':'$m^2/s^2$'})
ax[0,0].set_xlabel('Time (days)')
ax[0,0].set_ylabel('Depth (m)')
ax[0,0].set_title('u\'v\' TPOSE')

uv_flux_TAO[:750].T.plot(x='time',y='depth',ax=ax[1,0],cmap=cmo.balance,vmin=-0.4,vmax=0.4,cbar_kwargs={'label':'$m^2/s^2$'})
ax[1,0].set_xlabel('Time (days)')
ax[1,0].set_ylabel('Depth (m)')
ax[1,0].set_title('u\'v\' TAO')

uu_TP[:750].T.plot(x='time',y='depth',ax=ax[0,1],cmap=cmo.balance,vmin=0.0,vmax=0.8,cbar_kwargs={'label':'$m^2/s^2$'})
ax[0,1].set_xlabel('Time (days)')
ax[0,1].set_ylabel('Depth (m)')
ax[0,1].set_title('${u\'}^2$ TPOSE')

uu_TAO[:750].T.plot(x='time',y='depth',ax=ax[1,1],cmap=cmo.balance,vmin=0.0,vmax=0.8,cbar_kwargs={'label':'$m^2/s^2$'})
ax[1,1].set_xlabel('Time (days)')
ax[1,1].set_ylabel('Depth (m)')
ax[1,1].set_title('${u\'}^2$ TAO')

vv_TP[:750].T.plot(x='time',y='depth',ax=ax[0,2],cmap=cmo.balance,vmin=0.0,vmax=0.25,cbar_kwargs={'label':'$m^2/s^2$'})
ax[0,2].set_xlabel('Time (days)')
ax[0,2].set_ylabel('Depth (m)')
ax[0,2].set_title('${v\'}^2$ TPOSE')

vv_TAO[:750].T.plot(x='time',y='depth',ax=ax[1,2],cmap=cmo.balance,vmin=0.0,vmax=0.25,cbar_kwargs={'label':'$m^2/s^2$'})
ax[1,2].set_xlabel('Time (days)')
ax[1,2].set_ylabel('Depth (m)')
ax[1,2].set_title('${v\'}^2$ TAO')

plt.tight_layout()
plt.savefig('V_U_flux_uu_vv_2D_2012to2013.png',format='png')

Flux_corr = xr.corr(uv_flux_TP, uv_flux_TAO, dim="time") # correlation in time of the zonal velocity at every depth
uu_corr = xr.corr(uu_TP, uu_TAO, dim="time") # correlation in time of the zonal velocity at every depth
vv_corr = xr.corr(vv_TP, vv_TAO, dim="time") # correlation in time of the zonal velocity at every depth

uv_flux_TP_170 = U6_170*V6_170
uv_flux_TAO_170 = U_TAO_170*V_TAO_170
uu_TP_170 = U6_170*U6_170
uu_TAO_170 = U_TAO_170*U_TAO_170
vv_TP_170 = V6_170*V6_170
vv_TAO_170 = V_TAO_170*V_TAO_170
Flux_corr_170 = xr.corr(uv_flux_TP_170, uv_flux_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth
uu_corr_170 = xr.corr(uu_TP_170, uu_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth
vv_corr_170 = xr.corr(vv_TP_170, vv_TAO_170, dim="time") # correlation in time of the zonal velocity at every depth

uv_flux_TP_110 = U6_110*V6_110
uv_flux_TAO_110 = U_TAO_110*V_TAO_110
uu_TP_110 = U6_110*U6_110
uu_TAO_110 = U_TAO_110*U_TAO_110
vv_TP_110 = V6_110*V6_110
vv_TAO_110 = V_TAO_110*V_TAO_110
Flux_corr_110 = xr.corr(uv_flux_TP_110, uv_flux_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth
uu_corr_110 = xr.corr(uu_TP_110, uu_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth
vv_corr_110 = xr.corr(vv_TP_110, vv_TAO_110, dim="time") # correlation in time of the zonal velocity at every depth


fig, ax = plt.subplots(figsize=(15,7),ncols=3)
ax[0].plot(Flux_corr_170,U_TAO_170.depth,linewidth=2.0,label='u\'v\'',color='tab:blue')
ax[0].plot(uu_corr_170,U_TAO_170.depth,linewidth=2.0,label='${u\'}^2$',color='tab:blue',linestyle='--')
ax[0].plot(vv_corr_170,U_TAO_170.depth,linewidth=2.0,label='${v\'}^2$',color='tab:blue',linestyle=':')
ax[0].set_xlabel('Correlation')
ax[0].set_ylabel('Depth (m)')
ax[0].set_title('0N,170W')
ax[0].legend()
ax[0].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[0].set_ylim(-250,0)
ax[0].set_xlim(-0.5,1)

ax[1].plot(Flux_corr,U_TAO_140.depth,linewidth=2.0,label='u\'v\'',color='tab:blue')
ax[1].plot(uu_corr,U_TAO_140.depth,linewidth=2.0,label='${u\'}^2$',color='tab:blue',linestyle='--')
ax[1].plot(vv_corr,U_TAO_140.depth,linewidth=2.0,label='${v\'}^2$',color='tab:blue',linestyle=':')
ax[1].set_xlabel('Correlation')
ax[1].set_ylabel('Depth (m)')
ax[1].set_title('0N,140W')
ax[1].legend()
ax[1].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[1].set_ylim(-250,0)
ax[1].set_xlim(-0.5,1)

ax[2].plot(Flux_corr_110,U_TAO_110.depth,linewidth=2.0,label='u\'v\'',color='tab:blue')
ax[2].plot(uu_corr_110,U_TAO_110.depth,linewidth=2.0,label='${u\'}^2$',color='tab:blue',linestyle='--')
ax[2].plot(vv_corr_110,U_TAO_110.depth,linewidth=2.0,label='${v\'}^2$',color='tab:blue',linestyle=':')
ax[2].set_xlabel('Correlation')
ax[2].set_ylabel('Depth (m)')
ax[2].set_title('0N,110W')
ax[2].legend()
ax[2].axvline(0.0,color='tab:gray',linewidth=0.75)
ax[2].set_ylim(-250,0)
ax[2].set_xlim(-0.5,1)

plt.tight_layout()
plt.savefig('V_U_flux_uu_vv_correlation_by_depth.png',format='png')