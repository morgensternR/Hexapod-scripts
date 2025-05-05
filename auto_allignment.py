# Import hexapod class from hexapod.py in same directory - Must set as working directory if using spyder
from hexapod import Hexapod
# Import labjack class
import u6
# Misc Imports
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons
import os
# Import seabreeze spectrometer code
import seabreeze.spectrometers as sb

hexapod = Hexapod('COM7')  # Connect to the hexapod on COM3

# Get device information
print(hexapod.get_identification())


