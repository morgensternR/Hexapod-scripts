# System Commands
*IDN?       Get device identification
CCL         <Level> [<PassWord>] Set command level
CCL?        Get command level
CSV?        Get current syntax version
DEL         <uint> Delay the command interpreter (in ms; use only in macro)
ECO?        <string> Echo a string
ERR?        Get error number
HLP?        Get list of available commands
RBT         Reboot system
SSN?        Get serial number
VER?        Get version

# Status and Information Commands
#3          Get real position
#4          Query status word
#5          Request motion status
#6          Query for position change
#7          Request controller ready status
#8          Query if macro is running
#9          Get Wave Generator Status
STA?        Query status word
POS?        [<AxisID>] Get real position
ONT?        [<AxisID>] Get on-target state
FRF?        [<AxisID>] Get referencing result
LIM?        [<AxisID>] Indicate limit switches
TRS?        [<AxisID>] Indicate reference point switch

# Motion Control Commands

#24         Stop all axes
#27         System abort
FRF         [<AxisID>] Fast reference move to reference switch
HLT         [<AxisID>] Halt motion smoothly
MOV         {<AxisID> <Position>} Set target position (start absolute motion)
MOV?        [<AxisID>] Get target position
MRT         {<AxisID> <Position>} Start relative motion referring to tool coordinate system
MRW         {<AxisID> <Position>} Start relative motion referring to work coordinate system
MVR         {<AxisID> <Distance>} Set target relative to current position (start relative motion)
STP         Stop all axes abruptly
VEL         {<AxisID> <Velocity>} Set closed-loop velocity
VEL?        [<AxisID>] Get closed-loop velocity
VLS         <SystemVelocity> Set system velocity
VLS?        Get system velocity
VMO?        {<AxisID> <Position>} Query if target can be reached

# Alignment and Scanning Commands
AAP         <AxisID> <Distance> <AxisID> <Distance> ["SA" <StepSize>] ["N" <NumberOfRepetitions>] ["A" <AnalogInputID>] Automated alignment part
FDG         <Routine> <Sc Axis> <St Axis> ["ML" <MLl>] ["A" <A>] ["MIA" <MIA>] ["MAA" <MAA>] ["F" <F>] ["SP" <SP>] ["V" <V>] ["MDC" <MDC>] ["SPO" <SPO>] Defines a fast alignment gradient search routine
FDR         <Routine> <Sc Axis> <Sc Axis Range> <St Axis> <St Axis Range> ["L" <L>] ["A" <A>] ["F" <F>] ["V" <V>] ["MP1" <MP1>] ["MP2" <MP2>] ["TT" <TT>] ["CM" <CM>] ["MIIL" <MILL>] ["MAIL" <MAIL>] ["SP" <SP>] Defines a fast alignment area scan
FGC         <Routine> <SC Axis Center Position> <ST Axis Center Position> Changes the center position of a gradient search routine
FGC?        [<Routine>] Gets the current center position of a gradient search routine
FIO         <AxisID> <Distance> <AxisID> <Distance> ["S" <LinearSpiralStepSize>] ["AR" <AngularScanSize>] ["L" <Threshold>] ["A" <AnalogInputID>] Start simultaneous alignment of input and output channels
FLM         <AxisID> <Distance> ["L" <Threshold>] ["A" <AnalogInputID>] ["D" <ScanDirection>] Start fast line scan to maximum
FLS         <AxisID> <Distance> ["L" <Threshold>] ["A" <AnalogInputID>] ["D" <ScanDirection>] Start fast line scan to maximum, with stop
FRC         <Routine> {<Routine coupled>} Couples fast alignment routines to each other
FRC?        [<Routine>] Gets coupled fast alignment routines
FRH?        Lists descriptions and physical units for the routine results that can be queried with the FRR? command
FRP         <Routine> <Routine Action> Stops, pauses or resumes a fast alignment routine
FRP?        {<Routine>} Gets the current state of a fast alignment routine
FRR?        [<Routine> [<ResultID>]] Gets the results of a fast alignment routine
FRS         {<Routine>} Starts a fast alignment routine
FSA         <Axis1ID> <Distance1> <Axis2ID> <Distance2> ["L" <Threshold>] ["S" <ScanLineDistance>] ["SA" <StepSize>] ["A" <AnalogInputID>] Start fast plane scan with automated alignment at the maximum
FSC         <Axis1ID> <Distance1> <Axis2ID> <Distance2> ["L" <Threshold>] ["S" <ScanLineDistance>] ["A" <AnalogInputID>] Start fast plane scan to maximum, with stop
FSM         <Axis1ID> <Distance1> <Axis2ID> <Distance2> ["L" <Threshold>] ["S" <ScanLineDistance>] ["A" <AnalogInputID>] Start fast plane scan to maximum
FSS?        Get scanning result

# Coordinate System Commands
KCP         <NameOfCoordinateSystem1><NameOfCoordinateSystem2> Copy a coordinate system
KEN         <NameOfCoordinateSystem> Enable a coordinate system
KEN?        [<NameOfCoordinateSystem>] Get enabled coordinate systems
KET?        [<TypeOfCoordinateSystem>] Get enabled coordinate systems
KLC?        [<NameOfCoordinateSystem1>[<NameOfCoordinateSystem2>[<Item1>[<Item2>]]]] Get properties of work|tool coordinate system couples
KLN         <NameOfCoordinateSystem1><NameOfCoordinateSystem2> Links coordinate systems to chains
KLN?        [<NameOfCoordinateSystem>] Get linked coordinate system chains
KLS?        [<NameOfCoordinateSystem>[<Item1>[<Item2>]]] Get coordinate system definitions
KLT?        [<StartCoordinateSystem>[<EndCoordinateSystem>]] Evaluate coordinate system chains
KRM         <NameOfCoordinateSystem> Remove an existing coordinate system
KSD         <NameOfCoordinateSystem>[{<AxisID> <Position>}] Define a new coordinate system of type KSD
KSF         <NameOfCoordinateSystem> Define a new coordinate system of type KSF
KST         <NameOfCoordinateSystem>[{<AxisID> <Position>}] Define a new coordinate system of type KST
KSW         <NameOfCoordinateSystem>[{<AxisID> <Position>}] Define a new coordinate system of type KSW
SPI         {<PPCoordinate> <Position>} Set pivot point coordinate
SPI?        [<PPCoordinate>] Get pivot point coordinate

# Configuration and Parameters

CST         {<AxisID> <StageName>} Set assignment of stages to axes
CST?        [<AxisID>] Get assignment of stages to axes
DPA         <Password> [{<AxisID> <ParameterID>}] Reset parameters or settings to default values
HPA?        Get list of available parameters
HPV?        Get parameter value description
NLM         {<AxisID> <LowLimit>} Set low position soft limit
NLM?        [<AxisID>] Get low position soft limit
PLM         {<AxisID> <HighLimit>} Set high position soft limit
PLM?        [<AxisID>] Get high position soft limit
PUN?        [<AxisID>] Get position unit
RON?        [<AxisID>] Get reference mode
SAI?        ["ALL"] Get list of current axis identifiers
SCT         "T" <Cycle Time> Set cycle time in ms
SCT?        [<T>] Get cycle time
SPA         {<ItemID> <PamID> <PamValue>} Set volatile memory parameters
SPA?        [<ItemID> <PamID>] Get volatile memory parameters
SSL         {<AxisID> <SoftLimitState>} Set soft limit state
SSL?        [<AxisID>] Get soft limit state
SST         {<AxisID> <StepSize>} Set step size
SST?        [<AxisID>] Get step size
SVO         {<AxisID> <ServoState>} Set servo mode
SVO?        [<AxisID>] Get servo mode
TMN?        [<AxisID>] Get minimum commandable position
TMX?        [<AxisID>] Get maximum commandable position
TRA?        {<AxisID> <Position>} Get travelrange in arbitrary direction
VST?        List available stages
WPA         <Password> [{<ItemID> <PamID>}] Save Parameters To Non-Volatile Memory

# I/O and Analog Commands
DIA?        Get DIagnosis Information [{<MeasureID>}]
DIO         []{<DIOID> <OutputOn>}] Set Digital Output Lines
DIO?        [<DIOID>] Get Digital Input Lines
HDI?        Show help on diagnosis information, received with DIA?
HIB?        [<HIDeviceID> <HIDeviceButton>] Get state of HID Button
IFC?        [<InterfacePam>] Get current interface parameters
IFS         <Pswd> {<InterfacePam> <PamValue>} Set interface parameters as default values
IFS?        [<InterfacePam>] Get interface parameters as default values
NAV         {<AnalogInputID> <NumberOfReadings>} Set number of readings to be averaged
NAV?        {<AnalogInputID> <NumberOfReadings>} Get number of readings to be averaged
SGA         {<AnalogInputID> <Gain>} Set gain of analog input
SGA?        [<AnalogInputID>] Get gain of analog input
SIC         <FA InputID> <calculation type> [{<calculation parameter>}] Defines calculation settings for a fast alignment input channel
SIC?        [<FA InputID>] Gets calculation settings for a fast alignment input channel
TAC?        Tell number of analog lines
TAD?        [<AnalogInputID>] Get ADC value of input signal
TAV?        [<AnalogInputID>] Get analog input voltage
TCI?        [<FA InputID>] Gets calculated value of a fast alignment input channel
TIO?        Tell number of installed digital I/O Lines

# Data Recording and Measurement
#11         Request memory space for trajectory points
DRC         {<RecTableID> <Source> <RecOptions>} Set data recorder configuration
DRC?        [<RecTableID>] Get data recorder configuration
DRL?        [<RecTableID>] Get number of recorded points
DRR?        [<StartPoint> [<NumberOfPoints> [<RecTableID>]]] Get recorded data values
DRT         {<RecTableID> <TriggerSource> <Value>} Set data recorder trigger source
DRT?        [<RecTableID>] Get data recorder trigger source
GWD?        <AxisID> <StartPoint> <NumberOfPoints>
HDR?        Get all data recorder options
IMP         <AxisID> <Amplitude> Start impulse and response measurement
RTR         <RecordTableRate> Set record table rate (in cycles)
RTR?        Get record table rate
SRG?        <AxisID> <RegisterID>
STE         <AxisID> <Amplitude> Start step and response measurement
TNR?        Tell number of data recorder tables

# Wave Generator Commands
WAV         <WaveTableId> <AppendWave> <WaveType> <WaveTypeParameters> Set Waveform Definition
WAV?        [<WaveTableId> <WaveParameterID>] Get Waveform Definition
WCL         {<WaveTableId>} Clear Wavedata of given Waveform
WGC         [<WaveGenID> <Cycles>] Set Number of Generator Cycles
WGC?        [<WaveGenID>] Get Number Of Generator Cycles
WGO         {<WaveGenID> <StartMode>} Set Wave Generator Start/Stop Mode
WGO?        [<WaveGenID>] Get Wave Generator Start/Stop Mode
WGR         Starts Recording in Sync with Wave Generator
WGS?        [<WaveGenID> [<ItemID]] Get Status Information of Wave Generator
WMS?        [<WaveTableId>] Get Maximum Number of Values for the Waveform
WSL         [<WaveGenID> <WaveTableID>] Set Connection of Wave Table to Wave Generator
WSL?        [<WaveGenID>] Get Connection of Wave Table to Wave Generator
WTR         [<WaveGenID> <WaveTableRate> <InterpolationType>] Set Wave Generator Table Rate
WTR?        [<WaveGenID>] Get Wave Generator Table Rate
TWG?        Tell number of Wave Generators

#Macro and Programming Commands
ADD         <Variable> <Float1> <Float2> Add two values and save result to variable (use only in macro)
CPY         <Variable> <CMD?> Copy a command response into a variable (use only in macro)
JRC         <Jump> <CMD?> <OP> <Value> Jump relatively depending on condition (use only in macro)
MAC         MAC("DEF?" | "FREE?" | "END") | (("BEG" | "DEF" | "DEL" | "START") <MacroName>) | ("NSTART" <MacroName> <RepeatNumber>) Call macro function
MAC?        [<MacroName>] List macros
MAN?        <command mnemonic> List on-line manual page
MEX         <CMD?> <OP> <Value> Stop macro execution due to condition (use only in macro)
RMC?        List running macros
VAR         {<Variable> <String>} Set variable value
VAR?        [<Variable>] Get variable value
WAC         <CMD?> <OP> <value> Wait for Condition (use only in macro)

