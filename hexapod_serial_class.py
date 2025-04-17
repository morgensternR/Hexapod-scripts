#%%
import serial
import time

class Hexapod:
    """
    A class to control a PI Hexapod positioning system using the PI command set.
    This class encapsulates the commands available in the PI controller and provides
    a more Pythonic interface for controlling the hexapod.
    """
    
    def __init__(self, port, baudrate=115200, timeout=1):
        """
        Initialize the Hexapod controller connection.
        
        Args:
            port (str): Serial port to connect to (e.g., 'COM1')
            baudrate (int): Baud rate for serial communication
            timeout (float): Timeout for serial communication in seconds
        """
        self.serial = serial.Serial(port, baudrate, timeout=timeout)
        self.connected = False
        self.connect()
        
    def connect(self):
        """Establish connection to the hexapod controller."""
        if not self.serial.is_open:
            self.serial.open()
        
        # Check if connection is successful by querying device ID
        response = self.send_command("*IDN?")
        if response:
            self.connected = True
            print(f"Connected to: {response}")
        else:
            print("Failed to connect to hexapod controller")
            
    def disconnect(self):
        """Close the connection to the hexapod controller."""
        if self.serial.is_open:
            self.serial.close()
        self.connected = False
        print("Disconnected from hexapod controller")
    
    def send_command(self, command, wait_for_response=True):
        """
        Send a command to the hexapod controller.
        
        Args:
            command (str): Command to send
            wait_for_response (bool): Whether to wait for and return a response
            
        Returns:
            str: Response from the controller if wait_for_response is True
        """
        if not self.connected and not command == "*IDN?":
            print("Not connected to hexapod controller")
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
        return self.send_command("*IDN?")
    
    def set_command_level(self, level, password=None):
        """
        Set command level.
        
        Args:
            level (int): Command level
            password (str, optional): Password for higher command levels
        """
        cmd = f"CCL {level}"
        if password:
            cmd += f" {password}"
        return self.send_command(cmd)
    
    def get_command_level(self):
        """Get current command level."""
        return self.send_command("CCL?")
    
    def reboot(self):
        """Reboot the controller system."""
        return self.send_command("RBT", wait_for_response=False)
    
    # Motion Control Commands
    def stop_all_axes(self):
        """Stop all axes abruptly."""
        return self.send_command("STP")
    
    def halt_motion(self, axis=None):
        """
        Halt motion smoothly.
        
        Args:
            axis (str, optional): Axis identifier. If None, halts all axes.
        """
        cmd = "HLT"
        if axis:
            cmd += f" {axis}"
        return self.send_command(cmd)
    
    def move_absolute(self, axis, position):
        """
        Move to absolute position.
        
        Args:
            axis (str): Axis identifier
            position (float): Target position
        """
        return self.send_command(f"MOV {axis} {position}")
    
    def move_relative(self, axis, distance):
        """
        Move relative to current position.
        
        Args:
            axis (str): Axis identifier
            distance (float): Relative distance to move
        """
        return self.send_command(f"MVR {axis} {distance}")
    
    def set_velocity(self, axis, velocity):
        """
        Set closed-loop velocity.
        
        Args:
            axis (str): Axis identifier
            velocity (float): Velocity value
        """
        return self.send_command(f"VEL {axis} {velocity}")
    
    def get_velocity(self, axis=None):
        """
        Get closed-loop velocity.
        
        Args:
            axis (str, optional): Axis identifier
        """
        cmd = "VEL?"
        if axis:
            cmd += f" {axis}"
        return self.send_command(cmd)
    
    # Status and Information Commands
    def get_position(self, axis=None):
        """
        Get real position.
        
        Args:
            axis (str, optional): Axis identifier
        """
        cmd = "POS?"
        if axis:
            cmd += f" {axis}"
        return self.send_command(cmd)
    
    def get_status(self):
        """Query status word."""
        return self.send_command("STA?")
    
    def is_on_target(self, axis=None):
        """
        Get on-target state.
        
        Args:
            axis (str, optional): Axis identifier
        """
        cmd = "ONT?"
        if axis:
            cmd += f" {axis}"
        return self.send_command(cmd)
    
    # Configuration Commands
    def set_servo_mode(self, axis, state):
        """
        Set servo mode.
        
        Args:
            axis (str): Axis identifier
            state (int): Servo state (0 for off, 1 for on)
        """
        return self.send_command(f"SVO {axis} {state}")
    
    def get_servo_mode(self, axis=None):
        """
        Get servo mode.
        
        Args:
            axis (str, optional): Axis identifier
        """
        cmd = "SVO?"
        if axis:
            cmd += f" {axis}"
        return self.send_command(cmd)
    
    def set_soft_limits(self, axis, low_limit, high_limit):
        """
        Set soft limits for an axis.
        
        Args:
            axis (str): Axis identifier
            low_limit (float): Low position soft limit
            high_limit (float): High position soft limit
        """
        self.send_command(f"NLM {axis} {low_limit}")
        return self.send_command(f"PLM {axis} {high_limit}")
    
    def get_soft_limits(self, axis):
        """
        Get soft limits for an axis.
        
        Args:
            axis (str): Axis identifier
            
        Returns:
            tuple: (low_limit, high_limit)
        """
        low = self.send_command(f"NLM? {axis}")
        high = self.send_command(f"PLM? {axis}")
        return (float(low), float(high))
    
    # Coordinate System Commands
    def define_work_coordinate_system(self, name, axis_positions=None):
        """
        Define a new work coordinate system.
        
        Args:
            name (str): Name of the coordinate system
            axis_positions (dict, optional): Dictionary of axis positions {axis: position}
        """
        cmd = f"KSW {name}"
        if axis_positions:
            for axis, pos in axis_positions.items():
                cmd += f" {axis} {pos}"
        return self.send_command(cmd)
    
    def define_tool_coordinate_system(self, name, axis_positions=None):
        """
        Define a new tool coordinate system.
        
        Args:
            name (str): Name of the coordinate system
            axis_positions (dict, optional): Dictionary of axis positions {axis: position}
        """
        cmd = f"KST {name}"
        if axis_positions:
            for axis, pos in axis_positions.items():
                cmd += f" {axis} {pos}"
        return self.send_command(cmd)
    
    def enable_coordinate_system(self, name):
        """
        Enable a coordinate system.
        
        Args:
            name (str): Name of the coordinate system
        """
        return self.send_command(f"KEN {name}")
    
    def get_enabled_coordinate_systems(self):
        """Get enabled coordinate systems."""
        return self.send_command("KEN?")
    
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
        cmd = f"FLM {axis} {distance}"
        if threshold is not None:
            cmd += f" L {threshold}"
        if analog_input is not None:
            cmd += f" A {analog_input}"
        if scan_direction is not None:
            cmd += f" D {scan_direction}"
        return self.send_command(cmd)
    
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
        cmd = f"FSM {axis1} {distance1} {axis2} {distance2}"
        if threshold is not None:
            cmd += f" L {threshold}"
        if scan_line_distance is not None:
            cmd += f" S {scan_line_distance}"
        if analog_input is not None:
            cmd += f" A {analog_input}"
        return self.send_command(cmd)
    
    # Helper methods
    def wait_for_motion_complete(self, timeout=60):
        """
        Wait until all motion has completed.
        
        Args:
            timeout (float): Maximum time to wait in seconds
            
        Returns:
            bool: True if motion completed, False if timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.send_command("#5") == "0":  # No motion
                return True
            time.sleep(0.1)
        return False
    
    def get_available_commands(self):
        """Get list of available commands."""
        return self.send_command("HLP?")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.disconnect()
        except:
            pass
# Example usage
hexapod = Hexapod('COM7')  # Connect to the hexapod on COM3

# Get device information
print(hexapod.get_identification())
#%%
# Enable servo mode on all axes
for axis in ['X', 'Y', 'Z', 'U', 'V', 'W']:
    hexapod.set_servo_mode(axis, 1)
#%%
# Move to absolute position
hexapod.move_absolute('Z',
                       5)

#hexapod.wait_for_motion_complete()

# Move relatively
#%%
# Define and use a work coordinate system
hexapod.define_work_coordinate_system('WORK1', {'X': 0, 'Y': 0, 'Z': 0})
hexapod.enable_coordinate_system('WORK1')

# Disconnect when done
hexapod.disconnect()

# %%
