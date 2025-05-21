import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import numpy as np
import time
import os
import seabreeze.spectrometers as sb
from scipy.signal import find_peaks

# --- Config ---
xrange = [400, 1000]
integrationtime = 300000  # µs
plotwait = 0.05  # seconds
i = 0
paused = 0
autoscaley = 0
prefix_csv = './log/spectrum_'

# --- Initialize Spectrometers ---
spec = sb.Spectrometer.from_serial_number()
spec.integration_time_micros(integrationtime)
spec2 = sb.Spectrometer.from_serial_number()
spec2.integration_time_micros(integrationtime)

# --- Get Wavelengths and Initial Intensities ---
ll = spec.wavelengths()
lowidx = next(i for i,v in enumerate(ll) if v > xrange[0])
highidx = next(i for i,v in enumerate(ll) if v > xrange[1])
s = spec.intensities()

ll2 = spec2.wavelengths()
lowidx2 = next(i for i,v in enumerate(ll2) if v > xrange[0])
highidx2 = next(i for i,v in enumerate(ll2) if v > xrange[1])
s2 = spec2.intensities()

# --- FFT Helpers ---
c = 299792458

def fft_with_axes(Intensities, Freq, dx, xvals):
    yval = np.interp(xvals[::-1], Freq[::-1], Intensities[::-1])
    y = yval
    fourier = np.fft.fft(y)
    freqs = np.fft.fftfreq(y.size, dx)
    idx = np.argsort(freqs)
    return freqs[idx], fourier[idx]

# --- FFT Setup ---
Freq = c / (ll[lowidx:highidx] * 1e-9)
x = np.linspace(np.max(Freq), np.min(Freq), 10000)
dx = x[1] - x[0]

Freq2 = c / (ll2[lowidx2:highidx2] * 1e-9)
x2 = np.linspace(np.max(Freq2), np.min(Freq2), 10000)
dx2 = x2[1] - x2[0]

# --- Plot Setup ---
plt.ion()
fig, ax = plt.subplots(2, 1, figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

# Wavelength Plot
plt.axes(ax[0])
l, = plt.plot(ll[lowidx:highidx], s[lowidx:highidx], lw=2, label="Spectrum 1")
lspec2, = plt.plot(ll2[lowidx2:highidx2], s2[lowidx2:highidx2], lw=2, label="Spectrum 2")
ax[0].set_xlabel('Wavelength (nm)')
ax[0].set_ylabel('Intensity (a.u.)')
ax[0].legend()
ax[0].grid(True)
ax[0].set_xlim(xrange)

# FFT Plot
l2, = ax[1].plot([], [], 'g', label='FFT 1')
l2spec2, = ax[1].plot([], [], 'r', label='FFT 2')
peak1_line = ax[1].axvline(x=0, color='green', linestyle='--', label='Peak 1')
peak2_line = ax[1].axvline(x=0, color='red', linestyle='--', label='Peak 2')

peak1_text = ax[1].text(0.05, 0.95, '', transform=ax[1].transAxes,
                        fontsize=10, color='green', bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
peak2_text = ax[1].text(0.05, 0.85, '', transform=ax[1].transAxes,
                        fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')

peak1_text_extra = ax[1].text(0.05, 0.78, '', transform=ax[1].transAxes,
                              fontsize=10, color='green', bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
peak2_text_extra = ax[1].text(0.05, 0.68, '', transform=ax[1].transAxes,
                              fontsize=10, color='red', bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')

ax[1].set_xlabel("Frequency (GHz)")
ax[1].set_ylabel("PSD (a.u.)")
ax[1].legend()
ax[1].grid(True)
plt.tight_layout()

# --- GUI Buttons ---
class Index:
    def savecsv(self, event):
        if not os.path.exists('log'):
            os.makedirs('log')
        curTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        np.savetxt(f'log/spectrum_{curTime}_spec1.csv', np.column_stack([ll, s]), fmt='%1.3f', delimiter=',')
        np.savetxt(f'log/spectrum_{curTime}_spec2.csv', np.column_stack([ll2, s2]), fmt='%1.3f', delimiter=',')
        np.savetxt(f'log/spectrum_{curTime}_spec1_fft.csv', np.sort(np.column_stack([1/freqs/1e9, np.abs(fourier)**2]), axis=0), fmt='%1.3f', delimiter=',')
        np.savetxt(f'log/spectrum_{curTime}_spec2_fft.csv', np.sort(np.column_stack([1/freqs2/1e9, np.abs(fourier2)**2]), axis=0), fmt='%1.3f', delimiter=',')

    def savepng(self, event):
        if not os.path.exists('log'):
            os.makedirs('log')
        curTime = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
        plt.savefig(f'log/spectrum_{curTime}.png')

def autoy(label):
    global autoscaley, paused
    if label == 'Autoscale-Y':
        autoscaley = 1 - autoscaley
    elif label == 'Pause':
        paused = 1 - paused

callback = Index()
axcsv = plt.axes([0.7, 0.05, 0.1, 0.075])
aximg = plt.axes([0.81, 0.05, 0.1, 0.075])
axchecky = plt.axes([0.05, 0.03, 0.2, 0.15])

bcsv = Button(axcsv, 'CSV')
bcsv.on_clicked(callback.savecsv)
bimg = Button(aximg, 'PNG')
bimg.on_clicked(callback.savepng)
checky = CheckButtons(axchecky, ('Autoscale-Y', 'Pause'), (False, False))
checky.on_clicked(autoy)

# --- Window Close Handling ---
def handle_close(evt):
    global i
    i = 1

fig.canvas.mpl_connect('close_event', handle_close)

# --- Live Loop ---
while i == 0:
    if paused == 0:
        s = spec.intensities()
        s2 = spec2.intensities()
        l.set_ydata(s[lowidx:highidx])
        lspec2.set_ydata(s2[lowidx2:highidx2])

        freqs, fourier = fft_with_axes(s[lowidx:highidx], Freq, dx, x)
        freqs2, fourier2 = fft_with_axes(s2[lowidx2:highidx2], Freq2, dx2, x2)

        valid1 = freqs != 0
        valid2 = freqs2 != 0
        freq_ghz = 1 / freqs[valid1] / 1e9
        freq2_ghz = 1 / freqs2[valid2] / 1e9
        psd1 = np.abs(fourier[valid1])**2
        psd2 = np.abs(fourier2[valid2])**2
        psd1 = np.clip(psd1, 1e-20, None)
        psd2 = np.clip(psd2, 1e-20, None)

        l2.set_data(freq_ghz, psd1)
        l2spec2.set_data(freq2_ghz, psd2)
        ax[1].set_xlim(freq_ghz.min(), freq_ghz.max())

        if autoscaley:
            ax[0].relim()
            ax[0].autoscale_view()
            ax[1].relim()
            ax[1].autoscale_view()

        # Peak detection
        peaks1, _ = find_peaks(psd1, height=np.max(psd1)*0.1, distance=10)
        peaks2, _ = find_peaks(psd2, height=np.max(psd2)*0.1, distance=10)

        if len(peaks1) > 0:
            max1 = peaks1[np.argmax(psd1[peaks1])]
            peak1_line.set_xdata(freq_ghz[max1])
            peak1_text.set_text(f'Spec1 Peak:\n{freq_ghz[max1]:.2f} GHz\n{psd1[max1]:.2e}')
            peak1_text_extra.set_text(f'Note: Max peak\nIndex = {max1}')
        else:
            peak1_line.set_xdata(np.nan)
            peak1_text.set_text("Spec1 Peak:\nNone")
            peak1_text_extra.set_text("")

        if len(peaks2) > 0:
            max2 = peaks2[np.argmax(psd2[peaks2])]
            peak2_line.set_xdata(freq2_ghz[max2])
            peak2_text.set_text(f'Spec2 Peak:\n{freq2_ghz[max2]:.2f} GHz\n{psd2[max2]:.2e}')
            peak2_text_extra.set_text(f'Note: Max peak\nIndex = {max2}')
        else:
            peak2_line.set_xdata(np.nan)
            peak2_text.set_text("Spec2 Peak:\nNone")
            peak2_text_extra.set_text("")

        plt.pause(plotwait)

# --- Cleanup ---
spec.close()
spec2.close()
