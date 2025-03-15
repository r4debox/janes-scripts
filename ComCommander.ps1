# This script is released into the public domain.
# Anyone is free to use, modify, and distribute this software.
# For more information, please refer to <https://unlicense.org/>.
# =====================================================================
# A diagnostic tool for COM ports and USB devices.
#
# Features:
# - Lists active COM ports and retrieves device info
# - Checks driver health and resets stuck COM ports
# - Monitors serial data and auto-detects baud rate
# - Lists USB hub devices and identifies connected devices
# - Exports a diagnostic report and monitors COM port changes
# - Retrieves boot messages from serial devices
# - Monitors serial errors with advanced logging
# - Performs virtual loopback self-test
# - Fuzzes the serial port to find unintended behavior (not in depth)
#
# No License
#
# Author: Janerain
#
#Notes
#You will probably need this ~ Set-ExecutionPolicy bypass
#Remember to set the policy back to default when you are done working on someone elses machine
#Options that can seriously break hardware are stubbed and inaccessible
#
# =====================================================================
# Auto-elevation: Restart as Administrator if needed
function Test-Admin {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    if ($global:DebugMode) { Write-Debug "[DEBUG] Current user: $($currentUser.Name)" }
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

if (-not (Test-Admin)) {
    Write-Host "Restarting with Administrator privileges..."
    Start-Process powershell -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`"" -Verb RunAs
    exit
}

# Global variables & logging
$global:LogPath    = Join-Path $PSScriptRoot "COMPortLog.txt"
$global:DryRun     = $false    # Simulate actions if true
$global:DebugMode  = $false    # Toggle verbose debug output
$global:TranscriptStarted = $false

$VerbosePreference = "SilentlyContinue"
$DebugPreference   = "SilentlyContinue"

function Write-Log {
    param ([Parameter(Mandatory = $true)][string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $entry = "$timestamp - $Message"
    Add-Content -Path $global:LogPath -Value $entry
}

# ---------------------------
# Basic Diagnostic Functions
# ---------------------------

function Get-ActiveCOMPorts {
    Write-Host "`nScanning for active COM ports..."
    $portNames = [System.IO.Ports.SerialPort]::GetPortNames()
    if ($global:DebugMode) {
        Write-Debug "[DEBUG] Detected ports: $($portNames -join ', ')"
        Write-Log "[DEBUG] Detected ports: $($portNames -join ', ')"
    }
    if ($portNames.Count -gt 0) {
        Write-Host "Active COM Ports:"
        $portNames | ForEach-Object { Write-Host $_ }
        Write-Log "Active COM ports: $($portNames -join ', ')"
    }
    else {
        Write-Host "No active COM ports found."
        Write-Log "No active COM ports found."
    }
}

function Get-USBDeviceInfo {
    param ([Parameter(Mandatory = $true)][string]$Port)
    Write-Debug "Retrieving USB device info for port: $Port"
    $serial = Get-WmiObject Win32_SerialPort | Where-Object { $_.DeviceID -eq $Port }
    if ($serial) {
        $pnpId = $serial.PNPDeviceID
        Write-Debug "Found serial port; PNPDeviceID: $pnpId"
        $device = Get-PnpDevice | Where-Object { $_.InstanceId -like "*$pnpId*" }
    }
    else {
        $device = Get-PnpDevice | Where-Object { $_.FriendlyName -like "*$Port*" -or $_.InstanceId -like "*$Port*" }
    }
    if (-not $device) {
        Write-Host "No device found on $Port."
        return $null
    }
    $wmiDevice = Get-WmiObject Win32_PnPEntity | Where-Object { $_.DeviceID -like "*$($device.InstanceId)*" }
    $deviceID  = $device.InstanceId
    $driver    = Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceID -like "*$deviceID*" }
    $detailedInfo = [PSCustomObject]@{
        Port           = $Port
        InstanceId     = $device.InstanceId
        FriendlyName   = $device.FriendlyName
        Class          = $device.Class
        Status         = $device.Status
        ProblemCode    = $device.ProblemCode
        Manufacturer   = if ($wmiDevice) { $wmiDevice.Manufacturer } else { "N/A" }
        Service        = if ($wmiDevice) { $wmiDevice.Service } else { "N/A" }
        Caption        = if ($wmiDevice) { $wmiDevice.Caption } else { "N/A" }
        Description    = if ($wmiDevice) { $wmiDevice.Description } else { "N/A" }
        DeviceID       = if ($wmiDevice) { $wmiDevice.DeviceID } else { "N/A" }
        DriverName     = if ($driver) { $driver.DeviceName } else { "N/A" }
        DriverVersion  = if ($driver) { $driver.DriverVersion } else { "N/A" }
        DriverProvider = if ($driver) { $driver.DriverProviderName } else { "N/A" }
        DriverDate     = if ($driver) { $driver.DriverDate } else { "N/A" }
        InfName        = if ($driver) { $driver.InfName } else { "N/A" }
        ChipType       = "Unknown"
    }
    if ($driver) {
        $provider = $driver.DriverProviderName
        Write-Debug "Driver Provider: $provider"
        switch -Regex ($provider) {
            "FTDI"          { $detailedInfo.ChipType = "FTDI (FT232, FT2232, FT4232)"; break }
            "Silicon Labs"  { $detailedInfo.ChipType = "CP210x (Silicon Labs)"; break }
            "wch\.cn"       { $detailedInfo.ChipType = "CH340/CH341 (WCH)"; break }
            "Prolific"      { $detailedInfo.ChipType = "PL2303 (Prolific)"; break }
            "Microchip"     { $detailedInfo.ChipType = "Microchip USB-to-Serial"; break }
            "SMSC"          { $detailedInfo.ChipType = "SMSC USB-to-Serial"; break }
            Default         { $detailedInfo.ChipType = "Unknown" }
        }
        Write-Debug "Detected Chip Type: $($detailedInfo.ChipType)"
    }
    Write-Debug "Device info retrieved for port $Port."
    return $detailedInfo
}

function Check-DriverHealth {
    param ([Parameter(Mandatory = $true)][string]$DeviceID)
    Write-Debug "Checking driver health for DeviceID: $DeviceID"
    Write-Log "[DEBUG] Checking driver health for DeviceID: $DeviceID"
    $driver = Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceID -eq $DeviceID }
    if ($driver) {
        if (-not $driver.DriverVersion -or -not $driver.DriverProviderName) {
            Write-Host "Driver is missing or corrupted."
            Write-Log "Driver missing/corrupted for DeviceID: $DeviceID."
        }
        else {
            Write-Host "Driver is installed and working."
            Write-Log "Driver healthy for DeviceID: $DeviceID."
        }
    }
    else {
        Write-Host "No driver found for this device."
        Write-Log "No driver for DeviceID: $DeviceID."
    }
}

function Reset-StuckCOMPort {
    param ([Parameter(Mandatory = $true)][string]$PortName)
    Write-Host "`nResetting processes using ${PortName}..."
    Write-Debug "[DEBUG] Searching processes using ${PortName}"
    Write-Log "[DEBUG] Searching processes using ${PortName}"
    $processes = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -match $PortName }
    if ($processes) {
        foreach ($proc in $processes) {
            Write-Host "Terminating $($proc.Name) (PID: $($proc.ProcessId)) on ${PortName}"
            Write-Log "Terminating $($proc.Name) (PID: $($proc.ProcessId)) on ${PortName}"
            if (-not $global:DryRun) { Stop-Process -Id $proc.ProcessId -Force }
            else {
                Write-Host "[Dry Run] Process termination simulated."
                Write-Log "[Dry Run] Would terminate $($proc.Name) (PID: $($proc.ProcessId))."
            }
        }
        Write-Host "COM port ${PortName} reset successfully."
        Write-Log "COM port ${PortName} reset."
    }
    else {
        Write-Host "No stuck processes for ${PortName}."
        Write-Log "No stuck processes for ${PortName}."
    }
}

function Monitor-SerialData {
    param ([Parameter(Mandatory = $true)][string]$PortName, [int]$BaudRate = 9600)
    try {
        Write-Debug "Opening ${PortName} at BaudRate=$BaudRate"
        Write-Log "[DEBUG] Opening ${PortName} at baud rate $BaudRate"
        $port = New-Object System.IO.Ports.SerialPort $PortName, $BaudRate, 'None', 8, 'One'
        $port.ReadTimeout = 1000
        $port.Open()
        Write-Debug "Port ${PortName} opened. ReadTimeout=$($port.ReadTimeout)"
        Write-Log "[DEBUG] Port ${PortName} opened. ReadTimeout=$($port.ReadTimeout)"
        Write-Host "`nMonitoring ${PortName} at $BaudRate baud. (Ctrl+C to stop)"
        Write-Log "Monitoring ${PortName} at $BaudRate baud."
        while ($true) {
            try {
                if ($port.BytesToRead -gt 0) {
                    $data = $port.ReadExisting()
                    Write-Host "Received: $data"
                    Write-Log "Data on ${PortName}: $data"
                    Write-Debug "[DEBUG] Bytes available: $($port.BytesToRead)"
                }
            }
            catch {
                Write-Host "Error reading from ${PortName}: $($_)"
                Write-Log "Error reading from ${PortName}: $($_)"
            }
            Start-Sleep -Milliseconds 500
        }
        $port.Close()
    }
    catch {
        Write-Host "Error opening ${PortName}: $($_)"
        Write-Log "Error opening ${PortName}: $($_)"
    }
}

function AutoDetect-BaudRate {
    param ([Parameter(Mandatory = $true)][string]$PortName)
    Write-Host "`nAuto-detecting baud rate for ${PortName}..."
    Write-Debug "[DEBUG] Starting baud rate detection for ${PortName}"
    Write-Log "[DEBUG] Starting baud rate detection for ${PortName}"
    $baudRates = @(9600, 19200, 38400, 57600, 115200)
    foreach ($baud in $baudRates) {
        try {
            Write-Debug "[DEBUG] Testing baud rate: $baud"
            Write-Log "[DEBUG] Testing baud rate: $baud on ${PortName}"
            $port = New-Object System.IO.Ports.SerialPort $PortName, $baud, 'None', 8, 'One'
            $port.Open()
            Write-Host "Working baud rate: $baud"
            Write-Log "Detected baud rate $baud on ${PortName}."
            $port.Close()
            return $baud
        }
        catch {
            Write-Host "Failed at $baud."
            Write-Log "Baud rate $baud failed on ${PortName}."
        }
    }
    Write-Host "No working baud rate found for ${PortName}."
    Write-Log "No working baud rate found for ${PortName}."
    return $null
}

function List-USBHubDevices {
    Write-Host "`nListing USB hub devices..."
    Write-Debug "[DEBUG] Retrieving USB hub devices via WMI."
    Write-Log "[DEBUG] Retrieving USB hub devices via WMI."
    $usbDevices = Get-WmiObject Win32_USBControllerDevice | ForEach-Object {
        $_.Dependent -match 'USB' | Out-Null
        $_.Dependent
    }
    if ($usbDevices) {
        $usbDevices | ForEach-Object { Write-Host $_ }
        Write-Log "USB hub devices listed."
    }
    else {
        Write-Host "No USB hub devices found."
        Write-Log "No USB hub devices found."
    }
}

function Identify-SerialDevice {
    param ([Parameter(Mandatory = $true)][string]$PortName, [int]$BaudRate = 9600)
    try {
        Write-Debug "[DEBUG] Identifying device on ${PortName} at $BaudRate baud."
        Write-Log "[DEBUG] Identifying device on ${PortName} at $BaudRate baud."
        $port = New-Object System.IO.Ports.SerialPort $PortName, $BaudRate, 'None', 8, 'One'
        $port.ReadTimeout = 1000
        $port.Open()
        Start-Sleep -Milliseconds 500
        $port.WriteLine("ATI`r")
        Write-Debug "[DEBUG] Sent ATI command to ${PortName}."
        Write-Log "[DEBUG] Sent ATI command to ${PortName}."
        Start-Sleep -Milliseconds 500
        $response = $port.ReadExisting()
        if ($response -match "OK") {
            Write-Host "Detected modem/AT device."
            Write-Log "Identified AT device on ${PortName}."
        }
        elseif ($response -match "Arduino") {
            Write-Host "Detected Arduino board."
            Write-Log "Identified Arduino on ${PortName}."
        }
        elseif ($response -match "UBLOX") {
            Write-Host "Detected GPS module."
            Write-Log "Identified GPS module on ${PortName}."
        }
        else {
            Write-Host "Device response: $response"
            Write-Log "Unrecognized response on ${PortName}: $response"
        }
        $port.Close()
    }
    catch {
        Write-Host "Error communicating with ${PortName}: $($_)"
        Write-Log "Error communicating with ${PortName}: $($_)"
    }
}

function Export-DiagnosticReport {
    param ([string]$OutputPath = (Join-Path $PSScriptRoot "DiagnosticReport.csv"))
    Write-Host "`nGenerating diagnostic report..."
    Write-Debug "[DEBUG] Starting report export."
    Write-Log "[DEBUG] Starting report export."
    $report = @()
    $ports = [System.IO.Ports.SerialPort]::GetPortNames()
    foreach ($portName in $ports) {
        $deviceInfo = Get-USBDeviceInfo -Port $portName
        $report += [PSCustomObject]@{
            Port           = $portName
            Name           = $portName
            Description    = "N/A"
            ChipType       = $deviceInfo.ChipType
            DriverName     = $deviceInfo.DriverName
            DriverVersion  = $deviceInfo.DriverVersion
            DriverProvider = $deviceInfo.DriverProvider
            DriverStatus   = $deviceInfo.Status
        }
    }
    $report | Export-Csv -Path $OutputPath -NoTypeInformation
    Write-Host "Report exported to $OutputPath"
    Write-Log "Report exported to $OutputPath."
}

function Monitor-COMPortChanges {
    Write-Host "`nMonitoring COM port changes. (Ctrl+C to exit)"
    Write-Log "Monitoring COM port changes."
    $previousPorts = @{}
    while ($true) {
        $currentPorts = @{}
        [System.IO.Ports.SerialPort]::GetPortNames() | ForEach-Object { $currentPorts[$_] = $_ }
        foreach ($port in $currentPorts.Keys) {
            if (-not $previousPorts.ContainsKey($port)) {
                Write-Host "New COM port detected: $port"
                Write-Log "New COM port detected: $port"
            }
        }
        foreach ($port in $previousPorts.Keys) {
            if (-not $currentPorts.ContainsKey($port)) {
                Write-Host "COM port removed: $port"
                Write-Log "COM port removed: $port"
            }
        }
        if ($global:DebugMode) {
            Write-Debug "[DEBUG] Previous: $($previousPorts.Keys -join ', '); Current: $($currentPorts.Keys -join ', ')"
            Write-Log "[DEBUG] Previous: $($previousPorts.Keys -join ', '); Current: $($currentPorts.Keys -join ', ')"
        }
        $previousPorts = $currentPorts.Clone()
        Start-Sleep -Seconds 2
    }
}

function Get-SerialBootMessages {
    param ([Parameter(Mandatory = $true)][string]$PortName, [int]$BaudRate = 9600, [int]$Duration = 10)
    Write-Host "`nRetrieving boot messages from ${PortName}..."
    Write-Debug "[DEBUG] Opening ${PortName} at $BaudRate for boot capture."
    Write-Log "[DEBUG] Opening ${PortName} at $BaudRate for boot capture."
    try {
        $port = New-Object System.IO.Ports.SerialPort $PortName, $BaudRate, 'None', 8, 'One'
        $port.ReadTimeout = 1000
        $port.Open()
        Start-Sleep -Milliseconds 500
        $endTime = (Get-Date).AddSeconds($Duration)
        $bootMessages = @()
        while ((Get-Date) -lt $endTime) {
            try {
                if ($port.BytesToRead -gt 0) {
                    $data = $port.ReadExisting()
                    if ($data) {
                        $bootMessages += $data
                        Write-Host "Received: $data"
                        Write-Debug "[DEBUG] Boot message: $data"
                        Write-Log "[DEBUG] Boot message: $data"
                    }
                }
            }
            catch {
                Write-Host "Error reading from ${PortName}: $($_)"
                Write-Log "Error reading from ${PortName} during boot capture: $($_)"
            }
            Start-Sleep -Milliseconds 500
        }
        $port.Close()
        Write-Host "`nBoot capture complete."
        Write-Log "Boot capture complete for ${PortName}."
        return $bootMessages -join "`n"
    }
    catch {
        Write-Host "Error opening ${PortName}: $($_)"
        Write-Log "Error opening ${PortName} for boot capture: $($_)"
    }
}

# -------------------------------------
# Advanced Options (Secondary Menu)
# -------------------------------------

# Dump Flash Memory  stub
function Dump-FlashMemory {
    param ([Parameter(Mandatory = $true)][string]$PortName)
    Write-Host "Flash dumped to $outputPath"
    Write-Log "LOL NO"
}

# Write Flash Memory  stub
function Write-FlashMemory {
    param (
        [Parameter(Mandatory = $true)][string]$PortName,
        [Parameter(Mandatory = $true)][string]$FirmwarePath
    )
    if (-not (Test-Path $FirmwarePath)) {
        Write-Host "Firmware file not found: $FirmwarePath"
        return
    }
    Write-Log "LOL NO"
}

# Bootloader Reset: Force device into bootloader mode stub
function BootloaderReset {
    param ([Parameter(Mandatory = $true)][string]$PortName)
    Write-Host "Device should now be in bootloader mode."
    Write-Log "LOL NO"
}

# Direct Register Access: Read/write low-level registers  stub
function DirectRegisterAccess {
    param (
        [Parameter(Mandatory = $true)][string]$PortName,
        [Parameter(Mandatory = $true)][string]$Command  # e.g., "READ 0x1A" or "WRITE 0x1A 0xFF"
    )
    Write-Log "LOL NO"
}

# Advanced Options Menu
function AdvancedMenu {
    while ($true) {
        Write-Host "`n--- Advanced Options Menu (DISABLED) ---"
        Write-Host "1. Dump Flash Memory"
        Write-Host "2. Write Flash Memory"
        Write-Host "3. Bootloader Reset"
        Write-Host "4. Direct Register Access"
        Write-Host "5. Return to Main Menu"
        $advChoice = Read-Host "Select an advanced option"
        switch ($advChoice) {
            "1" {
                $port = Read-Host "Enter COM port (e.g., COM3)"
                Dump-FlashMemory -PortName $port
            }
            "2" {
                $port = Read-Host "Enter COM port (e.g., COM3)"
                $firmware = Read-Host "Enter firmware file path"
                Write-FlashMemory -PortName $port -FirmwarePath $firmware
            }
            "3" {
                $port = Read-Host "Enter COM port (e.g., COM3)"
                BootloaderReset -PortName $port
            }
            "4" {
                $port = Read-Host "Enter COM port (e.g., COM3)"
                $cmd = Read-Host "Enter register command (e.g., 'READ 0x1A' or 'WRITE 0x1A 0xFF')"
                DirectRegisterAccess -PortName $port -Command $cmd
            }
            "5" { break }
            default { Write-Host "Invalid selection. Try again." }
        }
    }
}

# ---------------------------
# Main Menu
# ---------------------------
function MainMenu {
    while ($true) {
        Write-Host "`n=== COM Port Diagnostic Menu ==="
        Write-Host "1. List Active COM Ports"
        Write-Host "2. Get Device Information"
        Write-Host "3. Check Driver Health"
        Write-Host "4. Reset Stuck COM Port"
        Write-Host "5. Monitor Serial Data"
        Write-Host "6. Auto-Detect Baud Rate"
        Write-Host "7. List USB Hub Devices"
        Write-Host "8. Identify Connected Device"
        Write-Host "9. Export Diagnostic Report"
        Write-Host "10. Monitor COM Port Changes (Dynamic)"
        Write-Host "11. Toggle Dry Run Mode (Current: $global:DryRun)"
        Write-Host "12. Toggle Debug Mode (Current: $global:DebugMode)"
        Write-Host "13. Retrieve Boot Messages from Serial Device"
        Write-Host "14. Advanced Error Logging & Analysis Tool"
        Write-Host "15. Virtual Loopback & Self-Test Framework"
        Write-Host "16. Fuzz Serial Port"
        Write-Host "17. Advanced Options"
        Write-Host "18. Exit"
        $choice = Read-Host "Select an option"
        switch ($choice) {
            "1" { Get-ActiveCOMPorts }
            "2" {
                $port = Read-Host "Enter COM port (e.g., COM3)"
                $deviceInfo = Get-USBDeviceInfo -Port $port
                if ($deviceInfo) {
                    Write-Host "Device Info:"
                    Write-Host "  Driver Name: $($deviceInfo.DriverName)"
                    Write-Host "  Chip Type:   $($deviceInfo.ChipType)"
                    Write-Host "  Driver:      $($deviceInfo.DriverProvider)"
                }
            }
            "3" {
                $port = Read-Host "Enter COM port (e.g., COM3)"
                $deviceInfo = Get-USBDeviceInfo -Port $port
                if ($deviceInfo) { Check-DriverHealth -DeviceID $deviceInfo.DeviceID }
            }
            "4" {
                $port = Read-Host "Enter COM port to reset (e.g., COM3)"
                Reset-StuckCOMPort -PortName $port
            }
            "5" {
                $port = Read-Host "Enter COM port to monitor (e.g., COM3)"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                Monitor-SerialData -PortName $port -BaudRate $baud
            }
            "6" {
                $port = Read-Host "Enter COM port for baud detection (e.g., COM3)"
                AutoDetect-BaudRate -PortName $port
            }
            "7" { List-USBHubDevices }
            "8" {
                $port = Read-Host "Enter COM port to identify device (e.g., COM3)"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                Identify-SerialDevice -PortName $port -BaudRate $baud
            }
            "9" { Export-DiagnosticReport }
            "10" { Monitor-COMPortChanges }
            "11" {
                $global:DryRun = -not $global:DryRun
                Write-Host "Dry Run mode now: $global:DryRun"
                Write-Log "Toggled Dry Run to: $global:DryRun."
            }
            "12" {
                $global:DebugMode = -not $global:DebugMode
                if ($global:DebugMode) {
                    Set-PSDebug -Trace 1
                    if (-not $global:TranscriptStarted) {
                        Start-Transcript -Path (Join-Path $PSScriptRoot "DebugTranscript.txt")
                        $global:TranscriptStarted = $true
                    }
                    $VerbosePreference = "Continue"
                    $DebugPreference   = "Continue"
                }
                else {
                    Set-PSDebug -Off
                    if ($global:TranscriptStarted) {
                        Stop-Transcript
                        $global:TranscriptStarted = $false
                    }
                    $VerbosePreference = "SilentlyContinue"
                    $DebugPreference   = "SilentlyContinue"
                }
                Write-Host "Debug mode now: $global:DebugMode"
                Write-Log "Toggled Debug mode to: $global:DebugMode."
            }
            "13" {
                $port = Read-Host "Enter COM port for boot messages (e.g., COM3)"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                $duration = Read-Host "Enter duration in seconds (default 10)"
                if ([string]::IsNullOrWhiteSpace($duration)) { $duration = 10 }
                $bootOutput = Get-SerialBootMessages -PortName $port -BaudRate $baud -Duration $duration
                Write-Host "`nCollected Boot Messages:`n$bootOutput"
            }
            "14" {
                $port = Read-Host "Enter COM port for error monitoring (e.g., COM3)"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                $duration = Read-Host "Enter duration in seconds (default 30)"
                if ([string]::IsNullOrWhiteSpace($duration)) { $duration = 30 }
                Monitor-SerialErrors -PortName $port -BaudRate $baud -Duration $duration
            }
            "15" {
                $port = Read-Host "Enter COM port for loopback test (e.g., COM3)"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                VirtualLoopbackSelfTest -PortName $port -BaudRate $baud
            }
            "16" {
                $port = Read-Host "Enter COM port for fuzzing (e.g., COM3)"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                $duration = Read-Host "Enter fuzz duration in seconds (default 30)"
                if ([string]::IsNullOrWhiteSpace($duration)) { $duration = 30 }
                $msgLength = Read-Host "Enter fuzz message length (default 10)"
                if ([string]::IsNullOrWhiteSpace($msgLength)) { $msgLength = 10 }
                $delay = Read-Host "Enter delay between messages (ms, default 500)"
                if ([string]::IsNullOrWhiteSpace($delay)) { $delay = 500 }
                Fuzz-SerialPort -PortName $port -BaudRate $baud -Duration $duration -MessageLength $msgLength -Delay $delay
            }
            "17" { AdvancedMenu }
            "18" { exit }
            default { Write-Host "Invalid selection. Try again." }
        }
    }
}

# Start the Main Menu
MainMenu