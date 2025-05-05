#%%
from pipython import GCSDevice
from pipython import pitools

import time


class HEXAPOD:
    """
    A class to control the PI C-887 Hexapod motion controller.
    Implements the available commands for the C-887 controller.
    """
    
    def __init__(self, device_name='C-887'):
        """
        Initialize the hexapod controller.
        
        Parameters:
        -----------
        device_name : str
            The device name, default is 'C-887'
        """
        self.device_name = device_name
        self.pidevice = None
        self.connected = False
        #Attempt connection and initialize the device
        print('Look for a pop-up window to connect to the hexapod')
        self.connect()
        print('initialize connected stages...')
        self.initialize()
        
    def initialize(self):
        """Initialize the hexapod properly before movement."""
        print("Initializing hexapod...")
        
        # Define all axes
        axes = ['X', 'Y', 'Z', 'U', 'V', 'W']
        
        if not self._check_connection():
            print("Error: No connection to hexapod")
            return False
        
        try:
            # Use the pitools.startup function to properly initialize the device
            print("Starting up the hexapod...")
            pitools.startup(
                pidevice=self.pidevice,  # The PI device object
                stages=None,             # All axes to initialize
                refmodes='FRF',          # Use FRF (reference using reference position) for all axes
                servostates=True         # Enable servo for all axes
            )
            print("Hexapod initialization complete")
            return True
        except Exception as e:
            print(f"Error initializing hexapod: {e}")
            return False
    def connect(self, auto_setup=True):
        """
        Connect to the hexapod controller.
        
        Parameters:
        -----------
        auto_setup : bool
            If True, opens the interface setup dialog
            
        Returns:
        --------
        bool
            True if connection successful
        """
        try:
            self.pidevice = GCSDevice(self.device_name)
            if auto_setup:
                self.pidevice.InterfaceSetupDlg()
            else:
                # You can add specific connection parameters here if needed
                pass
                
            self.connected = True
            return True
        except Exception as e:
            print(f"Error connecting to hexapod: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """
        Disconnect from the hexapod controller.
        """
        if self.pidevice and self.connected:
            self.pidevice.CloseConnection()
            self.connected = False
            print("Disconnected from hexapod controller")
    
    def _check_connection(self):
        """
        Check if the device is connected.
        
        Returns:
        --------
        bool
            True if connected
        """
        if not self.connected or not self.pidevice:
            print("Not connected to hexapod. Call connect() first.")
            return False
        return True
    
    # Device Information Commands
    def qIDN(self):
        """Get device identification."""
        if not self._check_connection():
            return None
        return self.pidevice.qIDN()
    
    def qHLP(self):
        """Get help information."""
        if not self._check_connection():
            return None
        return self.pidevice.qHLP()
    
    def qVER(self):
        """Get version information."""
        if not self._check_connection():
            return None
        return self.pidevice.qVER()
    
    def qERR(self):
        """Get error information."""
        if not self._check_connection():
            return None
        return self.pidevice.qERR()
    
    def qECO(self, command):
        """Echo command."""
        if not self._check_connection():
            return None
        return self.pidevice.qECO(command)
    
    # Axis and System Configuration
    def qCST(self):
        """Get assignment of axis identifiers to axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qCST()
    
    def qSAI(self, axis=None):
        """Get list of all available axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qSAI(axis)
    
    def qSAI_ALL(self):
        """Get list of all available axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qSAI_ALL()
    
    def qSVO(self, axes=None):
        """Get servo mode for specified axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qSVO(axes)
    
    def SVO(self, axes, values):
        """Set servo mode for specified axes."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.SVO(axes, values)
            return True
        except Exception as e:
            print(f"Error setting servo mode: {e}")
            return False
    
    def qPOS(self, axes=None):
        """Get position of specified axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qPOS(axes)
    
    def MOV(self, axes, values=None):
        """Move specified axes to absolute positions."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.MOV(axes, values)
            return True
        except Exception as e:
            print(f"Error during absolute motion: {e}")
            return False
    
    def MVR(self, axes, values=None):
        """Move specified axes relative to current position."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.MVR(axes, values)
            return True
        except Exception as e:
            print(f"Error during relative motion: {e}")
            return False
    
    def STP(self):
        """Stop all axes."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.STP()
            return True
        except Exception as e:
            print(f"Error stopping motion: {e}")
            return False
    
    def INI(self, axes=None):
        """Initialize specified axes."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.INI(axes)
            return True
        except Exception as e:
            print(f"Error initializing axes: {e}")
            return False
    
    def IsMoving(self, axes=None):
        """Check if specified axes are moving."""
        if not self._check_connection():
            return False
        try:
            return self.pidevice.IsMoving(axes)
        except Exception as e:
            print(f"Error checking motion status: {e}")
            return False
    
    def HasPosChanged(self, axes=None):
        """Check if position of specified axes has changed."""
        if not self._check_connection():
            return False
        try:
            return self.pidevice.HasPosChanged(axes)
        except Exception as e:
            print(f"Error checking position change: {e}")
            return False
    
    def GetPosStatus(self, axes=None):
        """Get position status of specified axes."""
        if not self._check_connection():
            return None
        try:
            return self.pidevice.GetPosStatus(axes)
        except Exception as e:
            print(f"Error getting position status: {e}")
            return None
    
    # Velocity and Acceleration
    def qVEL(self, axes=None):
        """Get velocity of specified axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qVEL(axes)
    
    def VEL(self, axes, values=None):
        """Set velocity of specified axes."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.VEL(axes, values)
            return True
        except Exception as e:
            print(f"Error setting velocity: {e}")
            return False
    
    # Limits
    def qNLM(self, axes=None):
        """Get lower limits for specified axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qNLM(axes)
    
    def qPLM(self, axes=None):
        """Get upper limits for specified axes."""
        if not self._check_connection():
            return None
        return self.pidevice.qPLM(axes)
    
    def NLM(self, axes, values=None):
        """Set lower limits for specified axes."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.NLM(axes, values)
            return True
        except Exception as e:
            print(f"Error setting lower limits: {e}")
            return False
    
    def PLM(self, axes, values=None):
        """Set upper limits for specified axes."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.PLM(axes, values)
            return True
        except Exception as e:
            print(f"Error setting upper limits: {e}")
            return False
    
    # Macro Commands
    def MAC_BEG(self, macro):
        """Begin recording a macro."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.MAC_BEG(macro)
            return True
        except Exception as e:
            print(f"Error beginning macro recording: {e}")
            return False
    
    def MAC_END(self):
        """End recording a macro."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.MAC_END()
            return True
        except Exception as e:
            print(f"Error ending macro recording: {e}")
            return False
    
    def MAC_START(self, macro):
        """Start a macro."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.MAC_START(macro)
            return True
        except Exception as e:
            print(f"Error starting macro: {e}")
            return False
    
    def MAC_DEL(self, macro):
        """Delete a macro."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.MAC_DEL(macro)
            return True
        except Exception as e:
            print(f"Error deleting macro: {e}")
            return False
    
    def qMAC(self, macro=None):
        """Get macro content."""
        if not self._check_connection():
            return None
        return self.pidevice.qMAC(macro)
    
    def IsRecordingMacro(self):
        """Check if macro recording is active."""
        if not self._check_connection():
            return False
        try:
            return self.pidevice.IsRecordingMacro()
        except Exception as e:
            print(f"Error checking macro recording status: {e}")
            return False
    
    # Additional commands
    def DRV(self, axes, values=None):
        """Set drive mode for specified axes."""
        if not self._check_connection():
            return False
        try:
            self.pidevice.DRV(axes, values)
            return True
        except Exception as e:
            print(f"Error setting drive mode: {e}")
            return False
    
    def qDRR(self, tables=None):
        """Get data recorder configuration."""
        if not self._check_connection():
            return None
        return self.pidevice.qDRR(tables)
    
    def wait_for_motion_completion(self, timeout=60):
        """
        Wait for motion completion on all axes.
        
        Parameters:
        -----------
        timeout : float
            Maximum time to wait in seconds
            
        Returns:
        --------
        bool
            True if motion completed within timeout
        """
        if not self._check_connection():
            return False
        
        start_time = time.time()
        while self.IsMoving():
            if time.time() - start_time > timeout:
                print(f"Timeout waiting for motion completion after {timeout} seconds")
                return False
            time.sleep(0.1)
        
        return True
    
    def __enter__(self):
        """Context manager entry point."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.disconnect()

hexapod = HEXAPOD()
#%%
# Example usage:
if __name__ == "__main__":
    # Using context manager
    with HEXAPOD() as hexapod:
        print(f"Connected to: {hexapod.qIDN()}")
        print(f"Available axes: {hexapod.qSAI_ALL()}")
        print(f"Current position: {hexapod.qPOS()}")
    
        
        # Move to a position
        hexapod.MOV({'X': 0, 'Y': 0, 'Z': 5})
        
        # Get current velocity settings
        print(f"Velocity settings: {hexapod.qVEL()}")
print('end')
# %%
import math
    
def move_in_circle(radius=3, height=0, steps=36, delay=0.1):
    """
    Move the hexapod in a circle in the X-Y plane.
    
    Args:
        radius: Radius of the circle in mm
        height: Constant Z height in mm
        steps: Number of points in the circle
        delay: Time delay between movements in seconds
    """
    print("Starting circular motion with radius:", radius)
    
    try:
        for i in range(steps):
            # Calculate angle in radians
            angle = 2 * math.pi * i / steps
            
            # Calculate X and Y coordinates for this point on the circle
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            print(f"Moving to position: X={x:.2f}, Y={y:.2f}, Z={height}")
            
            # Move to the calculated position
            hexapod.MOV({'X': x, 'Y': y, 'Z': height})
            
            # Wait a bit before the next movement
            time.sleep(delay)
        
        # Return to center position
        hexapod.MOV({'X': 0, 'Y': 0, 'Z': height})
        print("Circular motion completed")
    except Exception as e:
        print(f"Error during circular motion: {e}")


# Execute the circle motion with a radius of 15mm at 5mm height
move_in_circle(radius=15, height=0, steps=550, delay=0.001)
# %%
