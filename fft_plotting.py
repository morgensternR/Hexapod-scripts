# -*- coding: utf-8 -*-
"""
Updated: May 2025
Includes: FFT peak detection, vertical lines, custom annotations on FFT plot
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np
import time
import os
import seabreeze.spectrometers as sb

xrange = [400, 1000]
integrationtime = 300000
plotwait = 0.05

spec = sb.Spectrometer.from_serial_number()
spec.integration_time_micros(integrationtime)

spec2 = sb.Spectrometer.from_serial_number()
spec2.integration_time_micros(integrationtime)

i = 0
paused = 0
autoscaley = 0
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
        fileName = prefix_csv + curTime + "spec1.csv"
        fileName2 = prefix_csv + curTime + "spec2.csv"
        fileName3 = prefix_csv + curTime + "spec1_fft.csv"
        fileName4 = prefix_csv + curTime + "spec2_fft.csv"
        np.savetxt(fileName, np.column_stack([ll, s]), fmt='%1.3f', delimiter=',')
        np.savetxt(fileName2, np.column_stack([ll2, s2]), fmt='%1.3f', delimiter=',')
        np.savetxt(fileName3, np.sort(np.column_stack([1/freqs/1e9, np.abs(fourier)**2]), axis=0), fmt='%1.3f', delimiter=',')
        np.savetxt(fileName4, np.sort(np.column_stack([1/freqs2/1e9, np.abs(fourier2)**2]), axis=0), fmt='%1.3f', delimiter=',')

    def savepng(self, event):
        global prefix_csv
        if not os.path.exists('log'):
            os.makedirs('log')
        curTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        fileName = prefix_csv + curTime + ".png"
        plt.savefig(fileName)

def autoy(label):
    global autoscaley, paused
    if label == 'Autoscale-Y':
        autoscaley = 1 - autoscaley
    elif label == 'Pause':
        paused = 1 - paused

callback = Index()
axcsv = plt.axes([0.7, 0.05, 0.1, 0.075])
bcsv = Button(axcsv, 'CSV')
bcsv.on_clicked(callback.savecsv)
aximg = plt.axes([0.81, 0.05, 0.1, 0.075])
bimg = Button(aximg, 'PNG')
bimg.on_clicked(callback.savepng)
axchecky = plt.axes([0.05, 0.03, 0.2, 0.15])
checky = CheckButtons(axchecky, ('Autoscale-Y', 'Pause'), (False, False))
checky.on_clicked(autoy)

ll = spec.wavelengths()
lowidx = next(i for i, v in enumerate(ll) if v > xrange[0])
highidx = next(i for i, v in enumerate(ll) if v > xrange[1])
s = spec.intensities()

ll2 = spec2.wavelengths()
lowidx2 = next(i for i, v in enumerate(ll2) if v > xrange[0])
highidx2 = next(i for i, v in enumerate(ll2) if v > xrange[1])
s2 = spec2.intensities()

plt.axes(ax[0])
l, = plt.plot(ll[lowidx:highidx], s[lowidx:highidx], lw=2)
lspec2, = plt.plot(ll2[lowidx2:highidx2], s2[lowidx2:highidx2], lw=2)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity (a.u.)')
plt.title('Ocean Optics Spectrum: ' + spec.model + ' (' + spec.serial_number + ')')
plt.grid(True)
plt.xlim(xrange)

Wavelength = ll[lowidx:highidx]
c = 299792458
Freq = c / (Wavelength * 1e-9)
x = np.linspace(np.max(Freq), np.min(Freq), 10000)
x_inv = x[::-1]
dx = x_inv[1] - x_inv[0]

def fft_with_axes(Intensities):
    yval = np.interp(x[::-1], Freq[::-1], Intensities[::-1])
    fourier = np.fft.fft(yval)
    freqs = np.fft.fftfreq(yval.size, dx)
    idx = np.argsort(freqs)
    return (freqs[idx], fourier[idx])

freqs, fourier = fft_with_axes(s[lowidx:highidx])

Wavelength2 = ll2[lowidx2:highidx2]
Freq2 = c / (Wavelength2 * 1e-9)
x2 = np.linspace(np.max(Freq2), np.min(Freq2), 10000)
x_inv2 = x2[::-1]
dx2 = x_inv2[1] - x_inv2[0]

def fft_with_axes2(Intensities):
    yval2 = np.interp(x2[::-1], Freq2[::-1], Intensities[::-1])
    fourier2 = np.fft.fft(yval2)
    freqs2 = np.fft.fftfreq(yval2.size, dx2)
    idx2 = np.argsort(freqs2)
    return (freqs2[idx2], fourier2[idx2])

freqs2, fourier2 = fft_with_axes2(s2[lowidx2:highidx2])

plt.axes(ax[1])
l2, = plt.loglog(1/freqs/1e9, np.abs(fourier)**2, 'g')
l2spec2, = plt.loglog(1/freqs2/1e9, np.abs(fourier2)**2, 'r')
plt.xlabel('Frequency (GHz)')
plt.ylabel('PSD (a.u.)')
plt.grid(True)

# Peak tracking variables
peak_line1 = None
peak_line2 = None
peak_label1 = None
peak_label2 = None
custom_labels = []

while i == 0:
    if paused == 0:
        s = spec.intensities()
        s2 = spec2.intensities()

        plt.axes(ax[0])
        l.set_ydata(s[lowidx:highidx])
        lspec2.set_ydata(s2[lowidx2:highidx2])

        freqs, fourier = fft_with_axes(s[lowidx:highidx])
        freqs2, fourier2 = fft_with_axes2(s2[lowidx2:highidx2])

        plt.axes(ax[1])
        l2.set_ydata(np.abs(fourier)**2)
        l2spec2.set_ydata(np.abs(fourier2)**2)

        # Remove previous peaks and text
        if peak_line1: peak_line1.remove()
        if peak_label1: peak_label1.remove()
        if peak_line2: peak_line2.remove()
        if peak_label2: peak_label2.remove()
        for label in custom_labels:
            label.remove()
        custom_labels.clear()

        # Spectrum 1 peak
        peak_idx1 = np.argmax(np.abs(fourier)**2)
        peak_freq1 = 1 / freqs[peak_idx1] / 1e9
        peak_power1 = np.abs(fourier[peak_idx1])**2
        peak_line1 = ax[1].axvline(peak_freq1, color='black', linestyle='--', zorder=5)
        peak_label1 = ax[1].text(peak_freq1, peak_power1, f'{peak_freq1:.2f} GHz',
                                 color='black', fontsize=8, ha='left', va='bottom', zorder=6)

        # Spectrum 2 peak
        peak_idx2 = np.argmax(np.abs(fourier2)**2)
        peak_freq2 = 1 / freqs2[peak_idx2] / 1e9
        peak_power2 = np.abs(fourier2[peak_idx2])**2
        peak_line2 = ax[1].axvline(peak_freq2, color='gray', linestyle='--', zorder=5)
        peak_label2 = ax[1].text(peak_freq2, peak_power2, f'{peak_freq2:.2f} GHz',
                                 color='gray', fontsize=8, ha='left', va='bottom', zorder=6)

        # Add custom static text annotations
        custom_labels.append(ax[1].text(1.0, 1e7, "Note 1", color='blue', fontsize=9))
        custom_labels.append(ax[1].text(2.0, 1e6, "Note 2", color='green', fontsize=9))
        custom_labels.append(ax[1].text(3.0, 1e5, "Note 3", color='purple', fontsize=9))
        custom_labels.append(ax[1].text(4.0, 1e4, "Note 4", color='orange', fontsize=9))

        if autoscaley == 1:
            ax[0].relim()
            ax[0].autoscale_view(True, True, True)
            ax[1].relim()
            ax[1].autoscale_view(True, True, True)

        plt.show()
    plt.pause(plotwait)

spec.close()
spec2.close()
