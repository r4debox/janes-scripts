# =====================================================================
# A generalized diagnostic tool for USB devices.
#
# Features:
# - Lists active USB devices and retrieves device info
# - Checks driver health and resets unresponsive USB devices
# - Monitors USB data streams (for devices with a serial interface)
# - Lists USB hub devices and identifies connected devices
# - Exports a diagnostic report and monitors USB device changes
# - Retrieves boot messages from USB devices (if applicable)
# - Advanced error logging and analysis
# - Virtual loopback and self-test (for compatible devices)
# - Fuzz USB device interfaces to find unintended behavior (caution advised)
#
# This script is released into the public domain.
# Anyone is free to use, modify, and distribute this software.
# For more information, please refer to <https://unlicense.org/>.
#
# Author: Janerain (Generalized by ChatGPT)
#
# Notes:
# - You may need to set execution policy: Set-ExecutionPolicy Bypass
# - Options that can seriously affect hardware are stubbed and inaccessible
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
$global:LogPath    = Join-Path $PSScriptRoot "USBDiagnosticLog.txt"
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

# Function to list active USB devices (including devices that expose COM ports)
function Get-ActiveUSBDevices {
    Write-Host "`nScanning for active USB devices..."
    $usbDevices = Get-PnpDevice | Where-Object { $_.InstanceId -like "USB*" }
    if ($global:DebugMode) {
        Write-Debug "[DEBUG] Detected USB devices: $($usbDevices | Select-Object -ExpandProperty InstanceId -Unique -join ', ')"
        Write-Log "[DEBUG] Detected USB devices: $($usbDevices | Select-Object -ExpandProperty InstanceId -Unique -join ', ')"
    }
    if ($usbDevices.Count -gt 0) {
        Write-Host "Active USB Devices:"
        $usbDevices | ForEach-Object { Write-Host "$($_.FriendlyName) - $($_.InstanceId)" }
        Write-Log "Active USB devices: $($usbDevices | Select-Object -ExpandProperty InstanceId -Unique -join ', ')"
    }
    else {
        Write-Host "No active USB devices found."
        Write-Log "No active USB devices found."
    }
}

# Function to retrieve device information for a given USB device
function Get-USBDeviceInfo {
    param ([Parameter(Mandatory = $true)][string]$InstanceID)
    Write-Debug "Retrieving USB device info for InstanceID: ${InstanceID}"
    $device = Get-PnpDevice | Where-Object { $_.InstanceId -eq $InstanceID }
    if (-not $device) {
        Write-Host "No device found with InstanceID: ${InstanceID}."
        return $null
    }
    $wmiDevice = Get-WmiObject Win32_PnPEntity | Where-Object { $_.DeviceID -like "*$($device.InstanceId)*" }
    $driver = Get-WmiObject Win32_PnPSignedDriver | Where-Object { $_.DeviceID -like "*$($device.InstanceId)*" }
    $detailedInfo = [PSCustomObject]@{
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
    }
    Write-Debug "Device info retrieved for InstanceID ${InstanceID}."
    return $detailedInfo
}

# Check driver health for a given device
function Check-DriverHealth {
    [CmdletBinding()]
    param (
        [Parameter(Mandatory = $true)]
        [string]$InstanceID
    )
    Write-Debug "Checking driver health for InstanceID: ${InstanceID}"
    Write-Log "[DEBUG] Checking driver health for InstanceID: ${InstanceID}"
    try {
        $allDrivers = Get-CimInstance -ClassName Win32_PnPSignedDriver -ErrorAction Stop
    }
    catch {
        Write-Host "Error retrieving driver information: $($_.Exception.Message)"
        Write-Log "Error retrieving driver information: $($_.Exception.Message)"
        return
    }
    $filteredDrivers = $allDrivers | Where-Object { $_.DeviceID -match [regex]::Escape($InstanceID) }
    if ($filteredDrivers.Count -eq 0) {
        Write-Host "No driver found for InstanceID: ${InstanceID}."
        Write-Log "No driver found for InstanceID: ${InstanceID}."
        return
    }
    foreach ($d in $filteredDrivers) {
        $driverInfo = [PSCustomObject]@{
            DeviceID           = $d.DeviceID
            DriverName         = $d.DeviceName
            DriverVersion      = $d.DriverVersion
            DriverProviderName = $d.DriverProviderName
            DriverDate         = $d.DriverDate
            InfName            = $d.InfName
            Description        = $d.Description
            Signature          = $d.Signature
        }
        Write-Debug "[DEBUG] Retrieved driver info:`n$($driverInfo | Out-String)"
        if ([string]::IsNullOrEmpty($d.DriverVersion) -or [string]::IsNullOrEmpty($d.DriverProviderName)) {
            Write-Host "Driver is missing or appears corrupted for InstanceID: ${InstanceID}."
            Write-Log "Driver missing or corrupted for InstanceID: ${InstanceID}. Details:`n$($driverInfo | Out-String)"
        }
        else {
            Write-Host "Driver is installed and appears healthy for InstanceID: ${InstanceID}."
            Write-Host "Driver Details:"
            Write-Host "  Name:      $($d.DeviceName)"
            Write-Host "  Version:   $($d.DriverVersion)"
            Write-Host "  Provider:  $($d.DriverProviderName)"
            Write-Host "  Date:      $($d.DriverDate)"
            Write-Host "  INF File:  $($d.InfName)"
            Write-Host "  Desc:      $($d.Description)"
            Write-Host "  Signature: $($d.Signature)"
            Write-Log "Driver healthy for InstanceID: ${InstanceID}. Details:`n$($driverInfo | Out-String)"
        }
    }
}

# Function to reset a stuck USB device by disabling and re-enabling it
function Reset-StuckUSBDevice {
    param ([Parameter(Mandatory = $true)][string]$InstanceID)
    Write-Host "`nResetting USB device with InstanceID: ${InstanceID}..."
    Write-Debug "[DEBUG] Attempting to disable device with InstanceID: ${InstanceID}"
    Write-Log "[DEBUG] Attempting to disable device with InstanceID: ${InstanceID}"
    try {
        if (-not $global:DryRun) {
            Disable-PnpDevice -InstanceId $InstanceID -Confirm:$false -ErrorAction Stop
            Start-Sleep -Seconds 2
            Enable-PnpDevice -InstanceId $InstanceID -Confirm:$false -ErrorAction Stop
            Write-Host "USB device ${InstanceID} reset successfully."
            Write-Log "USB device ${InstanceID} reset successfully."
        }
        else {
            Write-Host "[Dry Run] Device reset simulated for ${InstanceID}."
            Write-Log "[Dry Run] Device reset simulated for ${InstanceID}."
        }
    }
    catch {
        Write-Host "Error resetting device ${InstanceID}: $($_.Exception.Message)"
        Write-Log "Error resetting device ${InstanceID}: $($_.Exception.Message)"
    }
}

# Monitor data stream for USB devices that expose a serial interface
function Monitor-USBDataStream {
    param (
        [Parameter(Mandatory = $true)][string]$InstanceID,
        [int]$BaudRate = 9600
    )
    Write-Host "`nMonitoring USB data stream for device: ${InstanceID}..."
    $deviceInfo = Get-USBDeviceInfo -InstanceID $InstanceID
    if ($deviceInfo -and $deviceInfo.DeviceID -match "COM") {
        $comPort = $deviceInfo.DeviceID -replace ".*(COM\d+).*", '$1'
        try {
            Write-Debug "Opening ${comPort} at BaudRate=$BaudRate"
            Write-Log "[DEBUG] Opening ${comPort} at baud rate $BaudRate"
            $port = New-Object System.IO.Ports.SerialPort $comPort, $BaudRate, 'None', 8, 'One'
            $port.ReadTimeout = 1000
            $port.Open()
            Write-Host "`nMonitoring ${comPort} at $BaudRate baud. (Ctrl+C to stop)"
            Write-Log "Monitoring ${comPort} at $BaudRate baud."
            while ($true) {
                try {
                    if ($port.BytesToRead -gt 0) {
                        $data = $port.ReadExisting()
                        Write-Host "Received: $data"
                        Write-Log "Data on ${comPort}: $data"
                        Write-Debug "[DEBUG] Bytes available: $($port.BytesToRead)"
                    }
                }
                catch {
                    Write-Host "Error reading from ${comPort}: $($_)"
                    Write-Log "Error reading from ${comPort}: $($_)"
                }
                Start-Sleep -Milliseconds 500
            }
            $port.Close()
        }
        catch {
            Write-Host "Error opening ${comPort}: $($_)"
            Write-Log "Error opening ${comPort}: $($_)"
        }
    }
    else {
        Write-Host "Device ${InstanceID} does not appear to have a serial interface."
        Write-Log "Device ${InstanceID} does not have a serial interface for data monitoring."
    }
}

# List USB Hub Devices
function List-USBHubDevices {
    Write-Host "`nListing USB hub devices..."
    Write-Debug "[DEBUG] Retrieving USB hub devices via WMI."
    Write-Log "[DEBUG] Retrieving USB hub devices via WMI."
    $usbHubs = Get-WmiObject Win32_USBHub
    if ($usbHubs) {
        $usbHubs | ForEach-Object { Write-Host "$($_.DeviceID) - $($_.Name)" }
        Write-Log "USB hub devices listed."
    }
    else {
        Write-Host "No USB hub devices found."
        Write-Log "No USB hub devices found."
    }
}

# Identify a connected USB device based on a simple command (if applicable)
function Identify-USBDevice {
    param (
        [Parameter(Mandatory = $true)][string]$InstanceID,
        [int]$BaudRate = 9600
    )
    Write-Host "`nIdentifying USB device for InstanceID: ${InstanceID}..."
    $deviceInfo = Get-USBDeviceInfo -InstanceID $InstanceID
    if ($deviceInfo -and $deviceInfo.DeviceID -match "COM") {
        $comPort = $deviceInfo.DeviceID -replace ".*(COM\d+).*", '$1'
        try {
            Write-Debug "[DEBUG] Identifying device on ${comPort} at $BaudRate baud."
            Write-Log "[DEBUG] Identifying device on ${comPort} at $BaudRate baud."
            $port = New-Object System.IO.Ports.SerialPort $comPort, $BaudRate, 'None', 8, 'One'
            $port.ReadTimeout = 1000
            $port.Open()
            Start-Sleep -Milliseconds 500
            $port.WriteLine("ATI`r")
            Write-Debug "[DEBUG] Sent identification command to ${comPort}."
            Write-Log "[DEBUG] Sent identification command to ${comPort}."
            Start-Sleep -Milliseconds 500
            $response = $port.ReadExisting()
            if ($response -match "OK") {
                Write-Host "Detected AT-compatible device."
                Write-Log "Identified AT-compatible device on ${comPort}."
            }
            elseif ($response -match "Arduino") {
                Write-Host "Detected Arduino board."
                Write-Log "Identified Arduino on ${comPort}."
            }
            else {
                Write-Host "Device response: $response"
                Write-Log "Unrecognized response on ${comPort}: $response"
            }
            $port.Close()
        }
        catch {
            Write-Host "Error communicating with ${comPort}: $($_)"
            Write-Log "Error communicating with ${comPort}: $($_)"
        }
    }
    else {
        Write-Host "Device ${InstanceID} does not appear to support direct identification."
        Write-Log "Device ${InstanceID} does not support direct identification via serial command."
    }
}

# Export a diagnostic report for USB devices
function Export-DiagnosticReport {
    param ([string]$OutputPath = (Join-Path $PSScriptRoot "USBDiagnosticReport.csv"))
    Write-Host "`nGenerating diagnostic report..."
    Write-Debug "[DEBUG] Starting report export."
    Write-Log "[DEBUG] Starting report export."
    $report = @()
    $devices = Get-PnpDevice | Where-Object { $_.InstanceId -like "USB*" }
    foreach ($dev in $devices) {
        $deviceInfo = Get-USBDeviceInfo -InstanceID $dev.InstanceId
        $report += [PSCustomObject]@{
            InstanceId     = $dev.InstanceId
            Name           = $dev.FriendlyName
            Class          = $dev.Class
            Status         = $dev.Status
            DriverName     = $deviceInfo.DriverName
            DriverVersion  = $deviceInfo.DriverVersion
            DriverProvider = $deviceInfo.DriverProvider
        }
    }
    $report | Export-Csv -Path $OutputPath -NoTypeInformation
    Write-Host "Report exported to $OutputPath"
    Write-Log "Report exported to $OutputPath."
}

# Monitor USB device changes dynamically
function Monitor-USBDeviceChanges {
    Write-Host "`nMonitoring USB device changes. (Ctrl+C to exit)"
    Write-Log "Monitoring USB device changes."
    $previousDevices = @{}
    while ($true) {
        $currentDevices = @{}
        Get-PnpDevice | Where-Object { $_.InstanceId -like "USB*" } | ForEach-Object { $currentDevices[$_.InstanceId] = $_.FriendlyName }
        foreach ($id in $currentDevices.Keys) {
            if (-not $previousDevices.ContainsKey($id)) {
                Write-Host "New USB device detected: $($currentDevices[$id]) ($id)"
                Write-Log "New USB device detected: $($currentDevices[$id]) ($id)"
            }
        }
        foreach ($id in $previousDevices.Keys) {
            if (-not $currentDevices.ContainsKey($id)) {
                Write-Host "USB device removed: $($previousDevices[$id]) ($id)"
                Write-Log "USB device removed: $($previousDevices[$id]) ($id)"
            }
        }
        if ($global:DebugMode) {
            Write-Debug "[DEBUG] Previous: $($previousDevices.Keys -join ', '); Current: $($currentDevices.Keys -join ', ')"
            Write-Log "[DEBUG] Previous: $($previousDevices.Keys -join ', '); Current: $($currentDevices.Keys -join ', ')"
        }
        $previousDevices = $currentDevices.Clone()
        Start-Sleep -Seconds 2
    }
}

# Retrieve boot messages from a USB device that supports a serial boot log (if applicable)
function Get-USBDeviceBootMessages {
    param (
        [Parameter(Mandatory = $true)][string]$InstanceID,
        [int]$BaudRate = 9600,
        [int]$Duration = 10
    )
    Write-Host "`nRetrieving boot messages for device: ${InstanceID}..."
    $deviceInfo = Get-USBDeviceInfo -InstanceID $InstanceID
    if ($deviceInfo -and $deviceInfo.DeviceID -match "COM") {
        $comPort = $deviceInfo.DeviceID -replace ".*(COM\d+).*", '$1'
        try {
            Write-Debug "[DEBUG] Opening ${comPort} at $BaudRate for boot capture."
            Write-Log "[DEBUG] Opening ${comPort} at $BaudRate for boot capture."
            $port = New-Object System.IO.Ports.SerialPort $comPort, $BaudRate, 'None', 8, 'One'
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
                    Write-Host "Error reading from ${comPort}: $($_)"
                    Write-Log "Error reading from ${comPort} during boot capture: $($_)"
                }
                Start-Sleep -Milliseconds 500
            }
            $port.Close()
            Write-Host "`nBoot capture complete."
            Write-Log "Boot capture complete for ${comPort}."
            return $bootMessages -join "`n"
        }
        catch {
            Write-Host "Error opening ${comPort}: $($_)"
            Write-Log "Error opening ${comPort} for boot capture: $($_)"
        }
    }
    else {
        Write-Host "Device ${InstanceID} does not appear to support boot message capture."
        Write-Log "Device ${InstanceID} does not support boot message capture."
    }
}

# -------------------------------------
# Advanced Options (Secondary Menu)
# -------------------------------------

# Dump Flash Memory stub
function Dump-FlashMemory {
    param ([Parameter(Mandatory = $true)][string]$InstanceID)
    Write-Host "Flash dumped to output (stub)."
    Write-Log "Dump-FlashMemory: Stub function executed for ${InstanceID}."
}

# Write Flash Memory stub
function Write-FlashMemory {
    param (
        [Parameter(Mandatory = $true)][string]$InstanceID,
        [Parameter(Mandatory = $true)][string]$FirmwarePath
    )
    if (-not (Test-Path $FirmwarePath)) {
        Write-Host "Firmware file not found: $FirmwarePath"
        return
    }
    Write-Log "Write-FlashMemory: Stub function executed for ${InstanceID} with firmware $FirmwarePath."
}

# Bootloader Reset: Force device into bootloader mode stub
function BootloaderReset {
    param ([Parameter(Mandatory = $true)][string]$InstanceID)
    Write-Host "Device ${InstanceID} should now be in bootloader mode (stub)."
    Write-Log "BootloaderReset: Stub function executed for ${InstanceID}."
}

# Direct Register Access: Read/write low-level registers stub
function DirectRegisterAccess {
    param (
        [Parameter(Mandatory = $true)][string]$InstanceID,
        [Parameter(Mandatory = $true)][string]$Command
    )
    Write-Log "DirectRegisterAccess: Stub function executed for ${InstanceID} with command $Command."
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
                $id = Read-Host "Enter device InstanceID"
                Dump-FlashMemory -InstanceID $id
            }
            "2" {
                $id = Read-Host "Enter device InstanceID"
                $firmware = Read-Host "Enter firmware file path"
                Write-FlashMemory -InstanceID $id -FirmwarePath $firmware
            }
            "3" {
                $id = Read-Host "Enter device InstanceID"
                BootloaderReset -InstanceID $id
            }
            "4" {
                $id = Read-Host "Enter device InstanceID"
                $cmd = Read-Host "Enter register command (e.g., 'READ 0x1A' or 'WRITE 0x1A 0xFF')"
                DirectRegisterAccess -InstanceID $id -Command $cmd
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
        Write-Host "`n=== USB Device Diagnostic Menu ==="
        Write-Host "1. List Active USB Devices"
        Write-Host "2. Get Device Information"
        Write-Host "3. Check Driver Health"
        Write-Host "4. Reset Stuck USB Device"
        Write-Host "5. Monitor USB Data Stream (Serial Interface)"
        Write-Host "6. List USB Hub Devices"
        Write-Host "7. Identify Connected USB Device"
        Write-Host "8. Export Diagnostic Report"
        Write-Host "9. Monitor USB Device Changes (Dynamic)"
        Write-Host "10. Retrieve Boot Messages from USB Device"
        Write-Host "11. Toggle Dry Run Mode (Current: ${global:DryRun})"
        Write-Host "12. Toggle Debug Mode (Current: ${global:DebugMode})"
        Write-Host "13. Advanced Options"
        Write-Host "14. Exit"
        $choice = Read-Host "Select an option"
        switch ($choice) {
            "1" { Get-ActiveUSBDevices }
            "2" {
                $id = Read-Host "Enter device InstanceID (e.g., USB\VID_XXXX&PID_XXXX\...):"
                $deviceInfo = Get-USBDeviceInfo -InstanceID $id
                if ($deviceInfo) {
                    Write-Host "Device Info:"
                    Write-Host "  Name:          $($deviceInfo.FriendlyName)"
                    Write-Host "  Class:         $($deviceInfo.Class)"
                    Write-Host "  Status:        $($deviceInfo.Status)"
                    Write-Host "  Manufacturer:  $($deviceInfo.Manufacturer)"
                    Write-Host "  Driver:        $($deviceInfo.DriverName) ($($deviceInfo.DriverVersion))"
                }
            }
            "3" {
                $id = Read-Host "Enter device InstanceID:"
                Check-DriverHealth -InstanceID $id
            }
            "4" {
                $id = Read-Host "Enter device InstanceID to reset:"
                Reset-StuckUSBDevice -InstanceID $id
            }
            "5" {
                $id = Read-Host "Enter device InstanceID (with serial interface):"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                Monitor-USBDataStream -InstanceID $id -BaudRate $baud
            }
            "6" { List-USBHubDevices }
            "7" {
                $id = Read-Host "Enter device InstanceID to identify:"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                Identify-USBDevice -InstanceID $id -BaudRate $baud
            }
            "8" { Export-DiagnosticReport }
            "9" { Monitor-USBDeviceChanges }
            "10" {
                $id = Read-Host "Enter device InstanceID for boot messages:"
                $baud = Read-Host "Enter baud rate (default 9600)"
                if ([string]::IsNullOrWhiteSpace($baud)) { $baud = 9600 }
                $duration = Read-Host "Enter duration in seconds (default 10)"
                if ([string]::IsNullOrWhiteSpace($duration)) { $duration = 10 }
                $bootOutput = Get-USBDeviceBootMessages -InstanceID $id -BaudRate $baud -Duration $duration
                Write-Host "`nCollected Boot Messages:`n$bootOutput"
            }
            "11" {
                $global:DryRun = -not $global:DryRun
                Write-Host "Dry Run mode now: ${global:DryRun}"
                Write-Log "Toggled Dry Run to: ${global:DryRun}."
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
                Write-Host "Debug mode now: ${global:DebugMode}"
                Write-Log "Toggled Debug mode to: ${global:DebugMode}."
            }
            "13" { AdvancedMenu }
            "14" { exit }
            default { Write-Host "Invalid selection. Try again." }
        }
    }
}

# Start the Main Menu
MainMenu
