param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("Query", "Install", "Start", "Disable", "Stop", "Uninstall")]
    [string]$Operation,
    [Parameter(Mandatory = $true)][string]$TaskName,
    [Parameter(Mandatory = $true)][string]$Fingerprint,
    [Parameter(Mandatory = $true)][string]$OwnerSid,
    [Parameter(Mandatory = $true)][string]$CommandPath,
    [Parameter(Mandatory = $true)][string]$Arguments,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory,
    [string]$XmlPath
)

$ErrorActionPreference = "Stop"
$description = "ai-nd-co-agent-tools-tts:$Fingerprint"
$mutexName = "Local\ai-nd-co-agent-tools-tts-startup"
$mutex = [Threading.Mutex]::new($false, $mutexName)

function Resolve-OwnerSid {
    param([string]$Identity)
    if ([string]::IsNullOrWhiteSpace($Identity)) { return $null }
    try {
        return ([System.Security.Principal.SecurityIdentifier]::new($Identity)).Value
    } catch {
        try {
            $translated = (New-Object System.Security.Principal.NTAccount($Identity)).Translate([System.Security.Principal.SecurityIdentifier])
            return $translated.Value
        } catch {
            return $null
        }
    }
}

function Get-ExactTask {
    $task = Get-ScheduledTask -TaskName $TaskName -TaskPath "\" -ErrorAction SilentlyContinue
    if ($null -eq $task) {
        return @{ present = $false; exact = $false; owned = $false; enabled = $false; running = $false; state = "Absent" }
    }
    $action = @($task.Actions)
    $trigger = @($task.Triggers)
    $exact = (
        $task.Description -eq $description -and
        $action.Count -eq 1 -and
        $action[0].Execute -eq $CommandPath -and
        $action[0].Arguments -eq $Arguments -and
        $action[0].WorkingDirectory -eq $WorkingDirectory -and
        $task.Principal.LogonType -eq "Interactive" -and
        $task.Principal.RunLevel -eq "Limited" -and
        $trigger.Count -eq 1 -and
        [string]$trigger[0].CimClass.CimClassName -eq "MSFT_TaskLogonTrigger" -and
        (Resolve-OwnerSid ([string]$task.Principal.UserId)) -eq $OwnerSid -and
        (Resolve-OwnerSid ([string]$trigger[0].UserId)) -eq $OwnerSid -and
        $trigger[0].Delay -eq "PT15S" -and
        $trigger[0].Enabled -eq $true -and
        $task.Settings.MultipleInstances -eq "IgnoreNew" -and
        $task.Settings.DisallowStartIfOnBatteries -eq $false -and
        $task.Settings.StopIfGoingOnBatteries -eq $false -and
        $task.Settings.AllowHardTerminate -eq $true -and
        $task.Settings.RestartCount -eq 3 -and
        $task.Settings.RestartInterval -eq "PT1M" -and
        $task.Settings.ExecutionTimeLimit -eq "PT0S" -and
        $task.Settings.StartWhenAvailable -eq $true -and
        $task.Settings.RunOnlyIfNetworkAvailable -eq $false -and
        $task.Settings.Hidden -eq $true
    )
    return @{
        present = $true
        exact = [bool]$exact
        owned = ([string]$task.Description).StartsWith("ai-nd-co-agent-tools-tts:")
        enabled = [bool]$task.Settings.Enabled
        running = ($task.State -eq "Running")
        state = [string]$task.State
    }
}

try {
    if (-not $mutex.WaitOne([TimeSpan]::FromSeconds(15))) {
        throw "lifecycle mutex timeout"
    }
    $snapshot = Get-ExactTask
    if ($Operation -eq "Query") {
        $snapshot | ConvertTo-Json -Compress
        exit 0
    }
    if ($Operation -eq "Install") {
        if ($snapshot.present -and -not $snapshot.exact) { throw "foreign task" }
        if ($snapshot.exact) {
            @{ changed = $false } | ConvertTo-Json -Compress
            exit 0
        }
        if ([string]::IsNullOrWhiteSpace($XmlPath)) { throw "missing XML" }
        $xmlText = [System.IO.File]::ReadAllText($XmlPath, [System.Text.UTF8Encoding]::new($false, $true)) -replace "^\s*<\?xml\s+[^>]*\?>\s*", ""
        Register-ScheduledTask -TaskName $TaskName -TaskPath "\" -Xml $xmlText | Out-Null
        $after = Get-ExactTask
        if (-not $after.exact) { throw "installed definition mismatch" }
        @{ changed = $true } | ConvertTo-Json -Compress
        exit 0
    }
    if (-not $snapshot.exact) { throw "task identity mismatch" }
    if ($Operation -eq "Start") {
        if ($snapshot.running) {
            @{ changed = $false } | ConvertTo-Json -Compress
            exit 0
        }
        Enable-ScheduledTask -TaskName $TaskName -TaskPath "\" | Out-Null
        Start-ScheduledTask -TaskName $TaskName -TaskPath "\"
        @{ changed = $true } | ConvertTo-Json -Compress
        exit 0
    }
    if ($Operation -eq "Disable") {
        if (-not $snapshot.enabled) {
            @{ changed = $false } | ConvertTo-Json -Compress
            exit 0
        }
        Disable-ScheduledTask -TaskName $TaskName -TaskPath "\" | Out-Null
        @{ changed = $true } | ConvertTo-Json -Compress
        exit 0
    }
    if ($Operation -eq "Stop") {
        if (-not $snapshot.running) {
            @{ changed = $false } | ConvertTo-Json -Compress
            exit 0
        }
        Stop-ScheduledTask -TaskName $TaskName -TaskPath "\"
        @{ changed = $true } | ConvertTo-Json -Compress
        exit 0
    }
    Unregister-ScheduledTask -TaskName $TaskName -TaskPath "\" -Confirm:$false
    @{ changed = $true } | ConvertTo-Json -Compress
}
catch {
    [Console]::Error.WriteLine("task_scheduler_failed")
    exit 1
}
finally {
    if ($null -ne $mutex) {
        try { $mutex.ReleaseMutex() } catch { }
        $mutex.Dispose()
    }
}
