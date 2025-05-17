# Import packages, 
import serial
import time
import sys
# if using serial, find com port number with pyvisa
# import pyvisa as visa
# rm = visa.ResourceManager()
# rm.list_resources()
#%%
# Try to import PI Python libraries, but continue if not available
try:
    from pipython import GCSDevice, pitools
    PI_PYTHON_AVAILABLE = True
except ImportError:
    PI_PYTHON_AVAILABLE = False
    print("Warning: PIPython library not found. Only serial communication will be available.")

class HexapodController:
    """
    A unified class to control PI Hexapod positioning systems.
    Supports both direct serial communication and the PI Python library interface.
    """
    
    def __init__(self, connection_type='serial', port=None, baudrate=115200, timeout=1, device_name='C-887'):
        """
        Initialize the Hexapod controller connection.
        
        Args:
            connection_type (str): Type of connection - 'serial' or 'pipython'
            port (str): Serial port to connect to (e.g., 'COM1') - only for serial connection
            baudrate (int): Baud rate for serial communication - only for serial connection
            timeout (float): Timeout for serial communication in seconds - only for serial connection
            device_name (str): Device name for PI Python connection - only for pipython connection
        """
        self.connection_type = connection_type.lower()
        self.connected = False
        
        # Serial connection attributes
        self.serial = None
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        
        # PI Python connection attributes
        self.device_name = device_name
        self.pidevice = None
        
        # Validate connection type
        if self.connection_type not in ['serial', 'pipython']:
            raise ValueError("Connection type must be 'serial' or 'pipython'")
            
        # Check if pipython is required but not available
        if self.connection_type == 'pipython' and not PI_PYTHON_AVAILABLE:
            raise ImportError("PIPython library is required for 'pipython' connection type")
            
        # Connect to the device
        self.connect()
        
        # When first started, turn on all axes
        self.initialize()
        
    def connect(self, auto_setup=True):
        """
        Establish connection to the hexapod controller.
        
        Args:
            auto_setup (bool): For pipython connection, if True, opens the interface setup dialog
            
        Returns:
            bool: True if connection successful
        """
        if self.connection_type == 'serial':
            return self._connect_serial()
        else:  # pipython
            return self._connect_pipython(auto_setup)
            
    def _connect_serial(self):
        """Establish serial connection to the hexapod controller."""
        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            
            if not self.serial.is_open:
                self.serial.open()
            
            # Check if connection is successful by querying device ID
            response = self.send_command("*IDN?")
            if response:
                self.connected = True
                print(f"Connected to: {response}")
                return True
            else:
                print("Failed to connect to hexapod controller")
                return False
        except Exception as e:
            print(f"Error connecting to hexapod via serial: {e}")
            self.connected = False
            return False
            
    def _connect_pipython(self, auto_setup=True):
        """Establish connection using PI Python library."""
        try:
            self.pidevice = GCSDevice(self.device_name)
            if auto_setup:
                print('Look for a pop-up window to connect to the hexapod')
                self.pidevice.InterfaceSetupDlg()
            else:
                # You can add specific connection parameters here if needed
                pass
                
            self.connected = True
            print(f"Connected to: {self.pidevice.qIDN()}")
            return True
        except Exception as e:
            print(f"Error connecting to hexapod via PIPython: {e}")
            self.connected = False
            return False
            
    def disconnect(self):
        """Close the connection to the hexapod controller."""
        if self.connection_type == 'serial':
            if self.serial and self.serial.is_open:
                self.serial.close()
        else:  # pipython
            if self.pidevice and self.connected:
                self.pidevice.CloseConnection()
                
        self.connected = False
        print("Disconnected from hexapod controller")
        
    def _check_connection(self):
        """
        Check if the device is connected.
        
        Returns:
            bool: True if connected
        """
        if not self.connected:
            print("Not connected to hexapod controller")
            return False
        return True
        
    def send_command(self, command, wait_for_response=True):
        """
        Send a command to the hexapod controller via serial connection.
        
        Args:
            command (str): Command to send
            wait_for_response (bool): Whether to wait for and return a response
            
        Returns:
            str: Response from the controller if wait_for_response is True
        """
        if self.connection_type != 'serial':
            print("send_command is only available for serial connection")
            return None
            
        if not self._check_connection() and not command == "*IDN?":
            return None
            
        # Add termination character if not present
        if not command.endswith('\n'):
            command += '\n'
            
        self.serial.write(command.encode())
        
        if wait_for_response:
            response = self.serial.readline().decode().strip()
            return response
        return None
        
    # System Commands
    def get_identification(self):
        """Get device identification."""
        if self.connection_type == 'serial':
            return self.send_command("*IDN?")
        else:  # pipython
            if not self._check_connection():
                return None
            return self.pidevice.qIDN()
            
    def initialize(self):
        """Initialize the hexapod properly before movement."""
        print("Initializing hexapod...")
        
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            # Enable servo mode on all axes
            for axis in ['X', 'Y', 'Z', 'U', 'V', 'W']:
                self.set_servo_mode(axis, 1)
            return True
        else:  # pipython
            try:
                # Use the pitools.startup function to properly initialize the device
                print("Starting up the hexapod...")
                pitools.startup(
                    pidevice=self.pidevice,  # The PI device object
                    stages=None,             # All axes to initialize
                    refmodes=None,          # Use FRF (reference using reference position) for all axes
                    servostates=True         # Enable servo for all axes
                )
                print("Hexapod initialization complete")
                return True
            except Exception as e:
                print(f"Error initializing hexapod: {e}")
                return False
                
    # Motion Control Commands
    def stop_all_axes(self):
        """Stop all axes abruptly."""
        if self.connection_type == 'serial':
            return self.send_command("STP")
        else:  # pipython
            if not self._check_connection():
                return False
            try:
                self.pidevice.STP()
                return True
            except Exception as e:
                print(f"Error stopping motion: {e}")
                return False
                
    def halt_motion(self, axis=None):
        """
        Halt motion smoothly.
        
        Args:
            axis (str, optional): Axis identifier. If None, halts all axes.
        """
        if self.connection_type == 'serial':
            cmd = "HLT"
            if axis:
                cmd += f" {axis}"
            return self.send_command(cmd)
        else:  # pipython
            # PIPython doesn't have a direct equivalent to HLT
            # Using STP as a fallback
            return self.stop_all_axes()
            
    def move_absolute(self, axes, positions=None):
        """
        Move to absolute position.
        
        Args:
            axes (str or dict): Axis identifier or dict of {axis: position}
            positions (float, optional): Target position if axes is a string
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            if isinstance(axes, dict):
                for axis, pos in axes.items():
                    self.send_command(f"MOV {axis} {pos}")
                return True
            else:
                return self.send_command(f"MOV {axes} {positions}")
        else:  # pipython
            try:
                if isinstance(axes, dict):
                    self.pidevice.MOV(axes)
                else:
                    self.pidevice.MOV(axes, positions)
                return True
            except Exception as e:
                print(f"Error during absolute motion: {e}")
                return False
                
    def move_relative(self, axes, distances=None):
        """
        Move relative to current position.
        
        Args:
            axes (str or dict): Axis identifier or dict of {axis: distance}
            distances (float, optional): Relative distance to move if axes is a string
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            if isinstance(axes, dict):
                for axis, dist in axes.items():
                    self.send_command(f"MVR {axis} {dist}")
                return True
            else:
                return self.send_command(f"MVR {axes} {distances}")
        else:  # pipython
            try:
                if isinstance(axes, dict):
                    self.pidevice.MVR(axes)
                else:
                    self.pidevice.MVR(axes, distances)
                return True
            except Exception as e:
                print(f"Error during relative motion: {e}")
                return False
                
    def set_velocity(self, axes, velocities=None):
        """
        Set closed-loop velocity.
        
        Args:
            axes (str or dict): Axis identifier or dict of {axis: velocity}
            velocities (float, optional): Velocity value if axes is a string
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            if isinstance(axes, dict):
                for axis, vel in axes.items():
                    self.send_command(f"VEL {axis} {vel}")
                return True
            else:
                return self.send_command(f"VEL {axes} {velocities}")
        else:  # pipython
            try:
                if isinstance(axes, dict):
                    self.pidevice.VEL(axes)
                else:
                    self.pidevice.VEL(axes, velocities)
                return True
            except Exception as e:
                print(f"Error setting velocity: {e}")
                return False
                
    def get_velocity(self, axes=None):
        """
        Get closed-loop velocity.
        
        Args:
            axes (str, optional): Axis identifier or list of axes
        """
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            cmd = "VEL?"
            if axes:
                cmd += f" {axes}"
            return self.send_command(cmd)
        else:  # pipython
            return self.pidevice.qVEL(axes)
            
    def get_position(self, axes=None):
        """
        Get real position.
        
        Args:
            axes (str, optional): Axis identifier or list of axes
        """
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            cmd = "POS?"
            if axes:
                cmd += f" {axes}"
            return self.send_command(cmd)
        else:  # pipython
            return self.pidevice.qPOS(axes)
            
    def get_status(self):
        """Query status word."""
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            return self.send_command("STA?")
        else:  # pipython
            # No direct equivalent in PIPython, returning a combination of status info
            try:
                is_moving = self.pidevice.IsMoving()
                error = self.pidevice.qERR()
                return {"moving": is_moving, "error": error}
            except Exception as e:
                print(f"Error getting status: {e}")
                return None
                
    def is_on_target(self, axes=None):
        """
        Get on-target state.
        
        Args:
            axes (str, optional): Axis identifier or list of axes
        """
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            cmd = "ONT?"
            if axes:
                cmd += f" {axes}"
            return self.send_command(cmd)
        else:  # pipython
            # Check if the device is still moving
            try:
                return not self.pidevice.IsMoving(axes)
            except Exception as e:
                print(f"Error checking on-target state: {e}")
                return None
                
    def set_servo_mode(self, axes, states=None):
        """
        Set servo mode.
        
        Args:
            axes (str or dict): Axis identifier or dict of {axis: state}
            states (int, optional): Servo state (0 for off, 1 for on) if axes is a string
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            if isinstance(axes, dict):
                for axis, state in axes.items():
                    self.send_command(f"SVO {axis} {state}")
                return True
            else:
                return self.send_command(f"SVO {axes} {states}")
        else:  # pipython
            try:
                if isinstance(axes, dict):
                    self.pidevice.SVO(axes)
                else:
                    self.pidevice.SVO(axes, states)
                return True
            except Exception as e:
                print(f"Error setting servo mode: {e}")
                return False
                
    def get_servo_mode(self, axes=None):
        """
        Get servo mode.
        
        Args:
            axes (str, optional): Axis identifier or list of axes
        """
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            cmd = "SVO?"
            if axes:
                cmd += f" {axes}"
            return self.send_command(cmd)
        else:  # pipython
            return self.pidevice.qSVO(axes)
    def set_soft_limits(self, axis, low_limit, high_limit):
        """
        Set soft limits for an axis.
        
        Args:
            axis (str): Axis identifier
            low_limit (float): Low position soft limit
            high_limit (float): High position soft limit
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            self.send_command(f"NLM {axis} {low_limit}")
            return self.send_command(f"PLM {axis} {high_limit}")
        else:  # pipython
            try:
                self.pidevice.NLM(axis, low_limit)
                self.pidevice.PLM(axis, high_limit)
                return True
            except Exception as e:
                print(f"Error setting soft limits: {e}")
                return False
                
    def get_soft_limits(self, axis):
        """
        Get soft limits for an axis.
        
        Args:
            axis (str): Axis identifier
            
        Returns:
            tuple: (low_limit, high_limit)
        """
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            low = self.send_command(f"NLM? {axis}")
            high = self.send_command(f"PLM? {axis}")
            try:
                return (float(low), float(high))
            except (ValueError, TypeError):
                print(f"Error parsing soft limits: {low}, {high}")
                return None
        else:  # pipython
            try:
                low = self.pidevice.qNLM(axis)
                high = self.pidevice.qPLM(axis)
                if isinstance(low, dict) and isinstance(high, dict):
                    return (low[axis], high[axis])
                return (low, high)
            except Exception as e:
                print(f"Error getting soft limits: {e}")
                return None
                
    # Coordinate System Commands
    def define_work_coordinate_system(self, name, axis_positions=None):
        """
        Define a new work coordinate system.
        
        Args:
            name (str): Name of the coordinate system
            axis_positions (dict, optional): Dictionary of axis positions {axis: position}
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            cmd = f"KSW {name}"
            if axis_positions:
                for axis, pos in axis_positions.items():
                    cmd += f" {axis} {pos}"
            return self.send_command(cmd)
        else:  # pipython
            try:
                if axis_positions:
                    # Format the command as required by PIPython
                    self.pidevice.KSW(name, axis_positions)
                else:
                    self.pidevice.KSW(name)
                return True
            except Exception as e:
                print(f"Error defining work coordinate system: {e}")
                return False
                
    def define_tool_coordinate_system(self, name, axis_positions=None):
        """
        Define a new tool coordinate system.
        
        Args:
            name (str): Name of the coordinate system
            axis_positions (dict, optional): Dictionary of axis positions {axis: position}
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            cmd = f"KST {name}"
            if axis_positions:
                for axis, pos in axis_positions.items():
                    cmd += f" {axis} {pos}"
            return self.send_command(cmd)
        else:  # pipython
            try:
                if axis_positions:
                    # Format the command as required by PIPython
                    self.pidevice.KST(name, axis_positions)
                else:
                    self.pidevice.KST(name)
                return True
            except Exception as e:
                print(f"Error defining tool coordinate system: {e}")
                return False
                
    def enable_coordinate_system(self, name):
        """
        Enable a coordinate system.
        
        Args:
            name (str): Name of the coordinate system
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            return self.send_command(f"KEN {name}")
        else:  # pipython
            try:
                self.pidevice.KEN(name)
                return True
            except Exception as e:
                print(f"Error enabling coordinate system: {e}")
                return False
                
    def get_enabled_coordinate_systems(self):
        """Get enabled coordinate systems."""
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            return self.send_command("KEN?")
        else:  # pipython
            try:
                return self.pidevice.qKEN()
            except Exception as e:
                print(f"Error getting enabled coordinate systems: {e}")
                return None
                
    # Alignment and Scanning Commands
    def fast_line_scan(self, axis, distance, threshold=None, analog_input=None, scan_direction=None):
        """
        Start fast line scan to maximum.
        
        Args:
            axis (str): Axis identifier
            distance (float): Scan distance
            threshold (float, optional): Threshold value
            analog_input (str, optional): Analog input identifier
            scan_direction (int, optional): Scan direction
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            cmd = f"FLM {axis} {distance}"
            if threshold is not None:
                cmd += f" L {threshold}"
            if analog_input is not None:
                cmd += f" A {analog_input}"
            if scan_direction is not None:
                cmd += f" D {scan_direction}"
            return self.send_command(cmd)
        else:  # pipython
            try:
                # Build parameters dictionary for PIPython
                params = {}
                if threshold is not None:
                    params['L'] = threshold
                if analog_input is not None:
                    params['A'] = analog_input
                if scan_direction is not None:
                    params['D'] = scan_direction
                    
                self.pidevice.FLM(axis, distance, **params)
                return True
            except Exception as e:
                print(f"Error during fast line scan: {e}")
                return False
                
    def fast_plane_scan(self, axis1, distance1, axis2, distance2, threshold=None, 
                        scan_line_distance=None, analog_input=None):
        """
        Start fast plane scan to maximum.
        
        Args:
            axis1 (str): First axis identifier
            distance1 (float): First axis scan distance
            axis2 (str): Second axis identifier
            distance2 (float): Second axis scan distance
            threshold (float, optional): Threshold value
            scan_line_distance (float, optional): Scan line distance
            analog_input (str, optional): Analog input identifier
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            cmd = f"FSM {axis1} {distance1} {axis2} {distance2}"
            if threshold is not None:
                cmd += f" L {threshold}"
            if scan_line_distance is not None:
                cmd += f" S {scan_line_distance}"
            if analog_input is not None:
                cmd += f" A {analog_input}"
            return self.send_command(cmd)
        else:  # pipython
            try:
                # Build parameters dictionary for PIPython
                params = {}
                if threshold is not None:
                    params['L'] = threshold
                if scan_line_distance is not None:
                    params['S'] = scan_line_distance
                if analog_input is not None:
                    params['A'] = analog_input
                    
                self.pidevice.FSM(axis1, distance1, axis2, distance2, **params)
                return True
            except Exception as e:
                print(f"Error during fast plane scan: {e}")
                return False
                
    # Macro Commands (PIPython specific)
    def macro_begin(self, macro):
        """
        Begin recording a macro.
        
        Args:
            macro (str): Name of the macro
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            print("Macro recording not supported in serial mode")
            return False
        else:  # pipython
            try:
                self.pidevice.MAC_BEG(macro)
                return True
            except Exception as e:
                print(f"Error beginning macro recording: {e}")
                return False
                
    def macro_end(self):
        """End recording a macro."""
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            print("Macro recording not supported in serial mode")
            return False
        else:  # pipython
            try:
                self.pidevice.MAC_END()
                return True
            except Exception as e:
                print(f"Error ending macro recording: {e}")
                return False
                
    def macro_start(self, macro):
        """
        Start a macro.
        
        Args:
            macro (str): Name of the macro
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            print("Macro execution not supported in serial mode")
            return False
        else:  # pipython
            try:
                self.pidevice.MAC_START(macro)
                return True
            except Exception as e:
                print(f"Error starting macro: {e}")
                return False
                
    def macro_delete(self, macro):
        """
        Delete a macro.
        
        Args:
            macro (str): Name of the macro
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            print("Macro deletion not supported in serial mode")
            return False
        else:  # pipython
            try:
                self.pidevice.MAC_DEL(macro)
                return True
            except Exception as e:
                print(f"Error deleting macro: {e}")
                return False
                
    def get_macro(self, macro=None):
        """
        Get macro content.
        
        Args:
            macro (str, optional): Name of the macro
        """
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            print("Macro query not supported in serial mode")
            return None
        else:  # pipython
            try:
                return self.pidevice.qMAC(macro)
            except Exception as e:
                print(f"Error querying macro: {e}")
                return None
                
    def is_recording_macro(self):
        """Check if macro recording is active."""
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            print("Macro recording status not supported in serial mode")
            return False
        else:  # pipython
            try:
                return self.pidevice.IsRecordingMacro()
            except Exception as e:
                print(f"Error checking macro recording status: {e}")
                return False
                
    # Helper methods
    def wait_for_motion_completion(self, timeout=60):
        """
        Wait until all motion has completed.
        
        Args:
            timeout (float): Maximum time to wait in seconds
            
        Returns:
            bool: True if motion completed, False if timed out
        """
        if not self._check_connection():
            return False
            
        start_time = time.time()
        
        if self.connection_type == 'serial':
            while time.time() - start_time < timeout:
                if self.send_command("#5") == "0":  # No motion
                    return True
                time.sleep(0.1)
            return False
        else:  # pipython
            try:
                while self.pidevice.IsMoving():
                    if time.time() - start_time > timeout:
                        print(f"Timeout waiting for motion completion after {timeout} seconds")
                        return False
                    time.sleep(0.1)
                return True
            except Exception as e:
                print(f"Error waiting for motion completion: {e}")
                return False
                
    def get_available_commands(self):
        """Get list of available commands."""
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            return self.send_command("HLP?")
        else:  # pipython
            try:
                return self.pidevice.qHLP()
            except Exception as e:
                print(f"Error getting available commands: {e}")
                return None
                
    def reboot(self):
        """Reboot the controller system."""
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            return self.send_command("RBT", wait_for_response=False)
        else:  # pipython
            try:
                self.pidevice.RBT()
                return True
            except Exception as e:
                print(f"Error rebooting controller: {e}")
                return False
                
    def set_command_level(self, level, password=None):
        """
        Set command level.
        
        Args:
            level (int): Command level
            password (str, optional): Password for higher command levels
        """
        if not self._check_connection():
            return False
            
        if self.connection_type == 'serial':
            cmd = f"CCL {level}"
            if password:
                cmd += f" {password}"
            return self.send_command(cmd)
        else:  # pipython
            try:
                if password:
                    self.pidevice.CCL(level, password)
                else:
                    self.pidevice.CCL(level)
                return True
            except Exception as e:
                print(f"Error setting command level: {e}")
                return False
                
    def get_command_level(self):
        """Get current command level."""
        if not self._check_connection():
            return None
            
        if self.connection_type == 'serial':
            return self.send_command("CCL?")
        else:  # pipython
            try:
                return self.pidevice.qCCL()
            except Exception as e:
                print(f"Error getting command level: {e}")
                return None
                
    def __enter__(self):
        """Context manager entry point."""
        if not self.connected:
            self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.disconnect()
        
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.disconnect()
        except:
            pass
    def set_pos_to_zero(self):
        for axis in ['X', 'Y', 'Z', 'U', 'V', 'W']:
            self.move_absolute(axis, 0)
# Initialize hexapod object using serial (DOESNT WORK YET)
#hexa = HexapodController(connection_type='serial', port='COM7', baudrate=115200, timeout=1, device_name='C-887')
# Initialize hexapod object using PiPython window
hexa = HexapodController(connection_type='pipython', port=None, baudrate=115200, timeout=1, device_name='C-887')



#%%

    
#%%
# Scans 2 axis and goes back to inital position
# Units are in mm
import numpy as np
import time
import csv
import os
import u6

# Make the labjack instrument and call it "d"
right_labjack = u6.U6(firstFound=False, localId=None, serial=360024796)
#left_labjack = u6.U6(firstFound=False, localId=None, serial=360011806)


def get_voltage(num_avg, labjack_object):     
        voltage_list = []
        voltage_channel_list =[]
        for channel in [0,2,4,6]:
            for n in range(num_avg):
                voltage = labjack_object.getAIN(positiveChannel = channel,
                                resolutionIndex = 9,                
                                gainIndex = 0,
                                settlingFactor = 2,
                                differential = True)
                voltage_list.append(voltage)
            voltage_mean= np.mean(voltage_list)
            
            voltage_channel_list.append(voltage_mean)
        v1, v2, v3, v4 = voltage_list[0], voltage_list[1], voltage_list[2], voltage_list[3]
        return v1, v2, v3, v4
def get_both_labjack_voltages(left_labjack, right_labjack, num_avg):
        l_v1, l_v2, l_v3, l_v4 = get_voltage(num_avg=num_avg, labjack_object=left_labjack)
        
        r_v1, r_v2, r_v3, r_v4 = get_voltage(num_avg=num_avg, labjack_object=right_labjack)
        
        return l_v1, l_v2, l_v3, l_v4, r_v1, r_v2, r_v3, r_v4

#%%%
result = get_voltage(num_avg = 10, labjack_object=right_labjack)

print(result)


#%%

# scan_2d requires the use of get_both_labjack_voltages function. Pass this into measurement_function. 

def scan_2d(hexapod_object, axis_1, axis_2, step_range, step_size, measurment_function, csv_file_name='my_csv_file.csv'):
    try:
        scan_range = np.arange(start=-step_range, stop=step_range + step_size, step=step_size)
        initial_position = hexapod_object.get_position()

        # Check if CSV file exists, if not create and write header
        file_exists = os.path.isfile(csv_file_name)
        with open(csv_file_name, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['position',  'l_v1', 'l_v2', 'l_v3', 'l_v4', 'r_v1', 'r_v2', 'r_v3', 'r_v4'])  # Header

            for axis_1_step in scan_range:
                axis_1_initial_pos = initial_position[f'{axis_1}']
                step_to_move_to_axis_1 = axis_1_initial_pos + axis_1_step
                print(f'Moving {axis_1} to {step_to_move_to_axis_1}')
                hexapod_object.move_absolute(axes=f'{axis_1}', positions=step_to_move_to_axis_1)
                time.sleep(0.1)

                for axis_2_step in scan_range:
                    axis_2_initial_pos = initial_position[f'{axis_2}']
                    step_to_move_to_axis_2 = axis_2_initial_pos + axis_2_step
                    print(f'Moving {axis_2} to {step_to_move_to_axis_2}')
                    hexapod_object.move_absolute(axes=f'{axis_2}', positions=step_to_move_to_axis_2)
                    time.sleep(0.1)

                    # Get measurement
                    #l_v1, l_v2, l_v3, l_v4, r_v1, r_v2, r_v3, r_v4 = measurment_function()
                    r_v1, r_v2, r_v3, r_v4 = measurment_function()

                    # Log the step positions and measurement
                    hexapod_position = hexapod_object.get_position()
                    axis1_pos = hexapod_position[f'{axis_1}']
                    axis2_pos = hexapod_position[f'{axis_2}']
                    
                    #writer.writerow([[f'{axis1_pos}', f'{axis2_pos}'], 
                    #                 l_v1, l_v2, l_v3, l_v4, r_v1, r_v2, r_v3, r_v4])
                    writer.writerow([[f'{axis1_pos}', f'{axis2_pos}'], 
                                      r_v1, r_v2, r_v3, r_v4])

        # Return to initial position
        for axis in initial_position.keys():
            hexapod_object.move_absolute(axis, initial_position[axis])

    except Exception as e:
        print(f"Error during scan: {e}")

   #%%         
scan_2d(hexapod_object = hexa, axis_1 = 'X', axis_2 = 'Y', step_range = 0.1, step_size = 0.05, 
    measurment_function= lambda:get_voltage(num_avg=10,labjack_object=right_labjack), csv_file_name='first_scan_test_Right_labjack.csv')


  
    #%%
