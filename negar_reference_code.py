# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 17:21:30 2024

@author: nno3
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:34:57 2024

@author: nno3
"""
#%%

# Import libraries
import numpy as np
import u6
import time
import pandas as pd

# Make the labjack instrument and call it "d"
d1 = u6.U6()
d2 = u6.U6()

# Get a single voltage
# GAIN x1:    -1V to +1V
# GAIN x10:    -10V to +10V
voltage1 = d1.getAIN(positiveChannel = 0,
                   resolutionIndex = 9,
                   gainIndex = 0,
                   settlingFactor = 2,
                   differential = True)
print(voltage1)

# Collect 13 voltages
voltage_list1 = []
for n in range(13):
    voltage1 = d1.getAIN(positiveChannel = 0,
                   resolutionIndex = 9,
                   gainIndex = 0,
                   settlingFactor = 2,
                   differential = True)
    voltage_list1.append(voltage1)
print(voltage_list1)

# Do statistics on those voltages
voltage_mean1 = np.mean(voltage_list1)
voltage_std1 = np.std(voltage_list1)

print('Mean')
print(voltage_mean1)
print('Standard deviation:')
print(voltage_std1)


# Get a single voltage in d2


voltage2 = d2.getAIN(positiveChannel = 0,
                   resolutionIndex = 9,
                   gainIndex = 0,
                   settlingFactor = 2,
                   differential = True)
print(voltage2)

# Collect 13 voltages
voltage_list2 = []
for n in range(13):
    voltage2 = d2.getAIN(positiveChannel = 0,
                   resolutionIndex = 9,
                   gainIndex = 0,
                   settlingFactor = 2,
                   differential = True)
    voltage_list2.append(voltage2)
print(voltage_list2)

# Do statistics on those voltages
voltage_mean2 = np.mean(voltage_list2)
voltage_std2 = np.std(voltage_list2)

print('Mean')
print(voltage_mean2)
print('Standard deviation:')
print(voltage_std2)
#%% Run an experimental loop

def get_voltage1(channel, num_avg):
    voltage_list1 = []
    for n in range(num_avg):
        voltage1 = d1.getAIN(positiveChannel = channel,
                        resolutionIndex = 9,                
                        gainIndex = 0,
                        settlingFactor = 2,
                        differential = True)
        voltage_list1.append(voltage1)
    voltage_mean1= np.mean(voltage_list1)
    # voltage_std = np.std(voltage_list)
    
    return voltage_mean1

def get_voltage2(channel, num_avg):
    voltage_list2 = []
    for n in range(num_avg):
        voltage2 = d2.getAIN(positiveChannel = channel,
                        resolutionIndex = 9,                
                        gainIndex = 0,
                        settlingFactor = 2,
                        differential = True)
    voltage_list2.append(voltage2)
    voltage_mean2= np.mean(voltage_list2)
    # voltage_std = np.std(voltage_list)
    
    return voltage_mean2



# Make a while loop
pos = 0
num_avg = 10 # Number of voltages to average
filename = (time.strftime("%Y-%m-%d %H-%M-%S")+ "Dark-light-redlight-set1-w=20-d=10-onedark for all light-d=40-one-detector-labjack-gradua1um-vb=15v-hanheldlaser635nm"+".csv")
outpath = r'C:\Users\nno3\Documents\Packaging\PD measurements\four-multimeter' 

data_list = []

while True:
    
    # Get light data
    time.sleep(60)
    print(pos)
    v1 = get_voltage1(channel = 0, num_avg = num_avg)
    v2 = get_voltage1(channel = 2, num_avg = num_avg)
    v3 = get_voltage1(channel = 4, num_avg = num_avg)
    v4 = get_voltage1(channel = 6, num_avg = num_avg)
    
    v1p = get_voltage2(channel = 0, num_avg = num_avg)
    v2p = get_voltage2(channel = 2, num_avg = num_avg)
    v3p = get_voltage2(channel = 4, num_avg = num_avg)
    v4p = get_voltage2(channel = 6, num_avg = num_avg)
    
   
    # Gather data together and make into dataframe and CSV
    data = dict(
        position = pos,
        v1 = v1,
        v2 = v2,
        v3 = v3,
        v4 = v4,
        v1p = v1p,
        v2p = v2p,
        v3p = v3p,
        v4p = v4p,
        )
    data_list.append(data)
    df = pd.DataFrame(data_list)
    df.to_csv(outpath + "\\" + filename)
    
    pos = pos + 1
#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 13:31:55 2023

@author: nno3
"""

# -*- coding: utf-8 -*- 
"""
Created on Thu May  3 15:06:56 2018
Using https://github.com/ap--/python-seabreeze
DO NOT INSTALL OMNIDriver. Windows needs to use the WinUSB drivers for this

Simple way to read, display, and save spectra from 
Ocean Optics USB spectrometers

Set backend field in Spyder IDE to "automatic" in
Tools > Preferences > IPython console > Graphics (tab)
@author: Dileep V. Reddy
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np
import time
import os
import seabreeze.spectrometers as sb
   
""" This is x-range for plotting only """
xrange = [400, 1000];    
""" Integration time in microseconds """
integrationtime = 300000;
""" Plotting refresh rate inverse in seconds """
plotwait = 0.05;

spec = sb.Spectrometer.from_serial_number()
spec.integration_time_micros(integrationtime)

spec2 = sb.Spectrometer.from_serial_number()
spec2.integration_time_micros(integrationtime)

i = 0;
paused = 0;
autoscaley = 0;
prefix_csv = './log/spectrum_'

plt.ion() 

fig, ax = plt.subplots(2)
plt.subplots_adjust(bottom=0.25)

def handle_close(evt):
    global i
    i = 1

fig.canvas.mpl_connect('close_event', handle_close)
   
class Index(object):
            
    def savecsv(self, event):
        global ll, s, prefix_csv
        if not os.path.exists('log'):
            os.makedirs('log')
        curTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        fileName  = prefix_csv + curTime + "spec1.csv"
        fileName2  = prefix_csv + curTime + "spec2.csv"
        fileName3  = prefix_csv + curTime + "spec1_fft.csv"
        fileName4  = prefix_csv + curTime + "spec2_fft.csv"
        np.savetxt(fileName, np.column_stack([ll, s]), fmt='%1.3f', delimiter=',')
        np.savetxt(fileName2, np.column_stack([ll2, s2]), fmt='%1.3f', delimiter=',')
        np.savetxt(fileName3, np.sort(np.column_stack([1/freqs/1e9, np.abs(fourier)**2]),axis=0), fmt='%1.3f', delimiter=',')
        np.savetxt(fileName4, np.sort(np.column_stack([1/freqs2/1e9, np.abs(fourier2)**2]),axis=0), fmt='%1.3f', delimiter=',')

    def savepng(self, event):
        global prefix_csv
        if not os.path.exists('log'):
            os.makedirs('log')
        curTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        fileName  = prefix_csv + curTime + ".png"
        plt.savefig(fileName)
    
def autoy(label):
    if label == 'Autoscale-Y':
        global autoscaley
        if autoscaley==0:
            autoscaley = 1
        else:
            autoscaley = 0                      
    elif label == 'Pause':
        global paused
        if paused==0:
            paused = 1
        else:
            paused = 0            
        
callback = Index()
axcsv = plt.axes([0.7, 0.05, 0.1, 0.075])
bcsv = Button(axcsv,'CSV')
bcsv.on_clicked(callback.savecsv)
aximg = plt.axes([0.81, 0.05, 0.1, 0.075])
bimg = Button(aximg,'PNG')
bimg.on_clicked(callback.savepng)
axchecky = plt.axes([0.05, 0.03, 0.2, 0.15])
checky = CheckButtons(axchecky, ('Autoscale-Y','Pause'), (False,False))
checky.on_clicked(autoy)

ll = spec.wavelengths()
lowidx = next(i for i,v in enumerate(ll) if v > xrange[0])
highidx = next(i for i,v in enumerate(ll) if v > xrange[1])
s = spec.intensities()

ll2 = spec2.wavelengths()
lowidx2 = next(i for i,v in enumerate(ll2) if v > xrange[0])
highidx2 = next(i for i,v in enumerate(ll2) if v > xrange[1])
s2 = spec2.intensities() #Interference fringes 

plt.axes(ax[0])
l, = plt.plot(ll[lowidx:highidx], s[lowidx:highidx], lw=2)
lspec2, = plt.plot(ll2[lowidx2:highidx2], s2[lowidx2:highidx2], lw=2)

plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Ocean Optics Spectrum: '+spec.model+' ('+spec.serial_number+')')
plt.grid(True)
plt.xlim(xrange)

#bgo edits
Wavelength = ll[lowidx:highidx]
c= 299792458
Freq=c/(Wavelength*1e-9)
x=np.linspace(np.max(Freq),np.min(Freq),10000)
x_inv = x[::-1] #reverse frequency array
dx = x_inv[1]-x_inv[0]

def fft_with_axes(Intensities):
    """ Performs an FFT and outputs correct frequency axes for it as well """
    #Intensities =  s[lowidx:highidx]

    yval=np.interp(x[::-1], Freq[::-1], Intensities[::-1])  #interpolate np.interp(target_x, source_x, source_y)n general will take the values in source_y (here, Intensities) and interpolate them to fit target_x (here, x2), based on the positions in source_x (here, Freq2).
    y = yval

    fourier = np.fft.fft(y) #1D discrete Fourier Transform
    freqs = np.fft.fftfreq(y.size, dx) #time
    idx = np.argsort(freqs)
    return (freqs[idx], fourier[idx])

freqs, fourier = fft_with_axes(s[lowidx:highidx])

Wavelength2 = ll2[lowidx2:highidx2]
c2= 299792458
Freq2=c2/(Wavelength2*1e-9)
x2=np.linspace(np.max(Freq2),np.min(Freq2),10000)
x_inv2 = x2[::-1]  #reverse frequency array
dx2 = x_inv2[1]-x_inv2[0]

def fft_with_axes2(Intensities):
    """ Performs an FFT and outputs correct frequency axes for it as well """
    #Intensities =  s[lowidx:highidx]

    yval2=np.interp(x2[::-1], Freq2[::-1], Intensities[::-1])
    y2 = yval2

    fourier2 = np.fft.fft(y2) #1D discrete Fourier Transform
    freqs2 = np.fft.fftfreq(y2.size, dx2) #time----FFT of frequency is time 
    idx2 = np.argsort(freqs2)
    return (freqs2[idx2], fourier2[idx2])

freqs2, fourier2 = fft_with_axes2(s2[lowidx2:highidx2])

plt.axes(ax[1])
l2, = plt.loglog(1/freqs/1e9, np.abs(fourier)**2,'g')
l2spec2, = plt.loglog(1/freqs2/1e9, np.abs(fourier2)**2,'r') ### 1/freq (time)= Frequency in GHz

plt.xlabel('Frequency (GHz)')
plt.ylabel('PSD (a.u.)') #PSD=power spectral density 
plt.grid(True)

while i == 0:
    if paused == 0:
        s = spec.intensities()
        s2 = spec2.intensities()
        plt.axes(ax[0])
        l.set_ydata( s[lowidx:highidx] )
        lspec2.set_ydata( s2[lowidx2:highidx2] )
                
        freqs, fourier = fft_with_axes( s[lowidx:highidx] )
        freqs2, fourier2 = fft_with_axes2(s2[lowidx2:highidx2])

        plt.axes(ax[1])        
        l2.set_ydata(np.abs(fourier)**2)
        l2spec2.set_ydata(np.abs(fourier2)**2)
        if autoscaley == 1:
            ax[0].relim()
            ax[0].autoscale_view(True,True,True)
            ax[1].relim()
            ax[1].autoscale_view(True,True,True)
            
        plt.show()
    plt.pause(plotwait)
    
#plt.savefig("test.png")
spec.close()
spec2.close()