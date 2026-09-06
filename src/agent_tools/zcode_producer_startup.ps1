param(
    [ValidateSet('Probe','Start')][string]$Operation = 'Probe',
    [Parameter(Mandatory=$true)][string]$Executable,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$Sha256,
    [Parameter(Mandatory=$true)][string]$OwnerSid,
    [ValidateRange(1,60)][int]$TimeoutSeconds = 60
)
$ErrorActionPreference = 'Stop'
# Do not emit process arguments: they can contain enrolled relay credentials.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Security.Principal;
public static class ZCodeStartupToken {
  [DllImport("kernel32.dll", SetLastError=true)] static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
  [DllImport("advapi32.dll", SetLastError=true)] static extern bool OpenProcessToken(IntPtr process, uint access, out IntPtr token);
  [DllImport("advapi32.dll", SetLastError=true)] static extern bool GetTokenInformation(IntPtr token, int kind, IntPtr data, int size, out int needed);
  public static string[] Read(uint pid) {
    IntPtr process = OpenProcess(0x1000, false, pid), token = IntPtr.Zero, data = IntPtr.Zero;
    if (process == IntPtr.Zero) return null;
    try {
      if (!OpenProcessToken(process, 8, out token)) return null;
      int needed;
      data = Marshal.AllocHGlobal(4);
      if (!GetTokenInformation(token, 20, data, 4, out needed)) return null;
      int elevated = Marshal.ReadInt32(data);
      Marshal.FreeHGlobal(data); data = IntPtr.Zero;
      GetTokenInformation(token, 25, IntPtr.Zero, 0, out needed);
      data = Marshal.AllocHGlobal(needed);
      if (!GetTokenInformation(token, 25, data, needed, out needed)) return null;
      string sid = new SecurityIdentifier(Marshal.ReadIntPtr(data)).Value;
      using (WindowsIdentity identity = new WindowsIdentity(token)) {
        return new string[] { identity.User.Value, elevated.ToString(), sid.Substring(sid.LastIndexOf('-') + 1) };
      }
    } finally {
      if (data != IntPtr.Zero) Marshal.FreeHGlobal(data);
      if (token != IntPtr.Zero) CloseHandle(token);
      CloseHandle(process);
    }
  }
}
'@
function Assert-Pin {
    $item=Get-Item -LiteralPath $Executable
    if($item.PSIsContainer -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -or
       $item.FullName -ne $Executable -or $item.Name -ne 'ZCode.exe' -or
       (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash.ToLowerInvariant() -ne $Sha256) {throw 'producer_identity_drift'}
}
function Test-HighOwner($Process) {
    $token=[ZCodeStartupToken]::Read([uint32]$Process.ProcessId)
    return ($null -ne $token -and $token[0] -eq $OwnerSid -and $token[1] -eq '1' -and
        [int]$token[2] -eq 12288 -and $Process.SessionId -eq $script:sessionId)
}
function Get-Proof {
    Assert-Pin
    $all=@(Get-CimInstance Win32_Process -Filter "Name='ZCode.exe' OR Name='node.exe'")
    # Electron also uses ZCode.exe for Node app-server children without --type.
    $zcodeIds=@($all|Where-Object {$_.Name -eq 'ZCode.exe'}|ForEach-Object {$_.ProcessId})
    $desktops=@($all|Where-Object {$_.Name -eq 'ZCode.exe' -and $_.CommandLine -notmatch '--type=' -and $_.ParentProcessId -notin $zcodeIds})
    if($desktops.Count -eq 0) {return @{code='producer_absent'}}
    if($desktops.Count -ne 1) {return @{code='producer_ambiguous'}}
    $desktop=$desktops[0]
    if($desktop.ExecutablePath -ne $Executable -or [string]::IsNullOrWhiteSpace($desktop.CommandLine)) {return @{code='producer_identity_unproved'}}
    if(-not (Test-HighOwner $desktop)) {return @{code='producer_not_high_owner_session'}}
    $hosts=@($all|Where-Object {$_.ParentProcessId -eq $desktop.ProcessId -and $_.ExecutablePath -eq $Executable -and $_.CommandLine -match 'node\.mojom\.NodeService'})
    if(@($hosts|Where-Object {-not (Test-HighOwner $_)}).Count -gt 0) {return @{code='producer_child_token_unproved'}}
    $children=@($all|Where-Object {$_.ParentProcessId -in @($hosts.ProcessId) -and $_.CommandLine -match 'zcode\.cjs["\s]+app-server(?:\s|$)'})
    if($children.Count -eq 0) {return @{code='producer_waiting_backend';desktopPid=[int]$desktop.ProcessId}}
    if(@($children|Where-Object {-not (Test-HighOwner $_)}).Count -gt 0) {return @{code='producer_child_token_unproved'}}
    return @{code='producer_ready';desktopPid=[int]$desktop.ProcessId;backendCount=$children.Count;sessionId=$script:sessionId;high=$true}
}
try {
    $script:sessionId=[Diagnostics.Process]::GetCurrentProcess().SessionId
    $caller=[ZCodeStartupToken]::Read([uint32]$PID)
    if($script:sessionId -eq 0 -or $null -eq $caller -or $caller[0] -ne $OwnerSid -or $caller[1] -ne '1' -or $caller[2] -ne '12288') {throw 'startup_not_high_owner_interactive'}
    $proof=Get-Proof
    if($Operation -eq 'Start') {
        if($proof.code -eq 'producer_absent') {
            # Recheck immediately before one launch; never replace a Medium single-instance owner.
            $proof=Get-Proof
            if($proof.code -eq 'producer_absent') {
                Assert-Pin
                Start-Process -FilePath $Executable -WorkingDirectory (Split-Path -Parent $Executable) -WindowStyle Hidden|Out-Null
            }
        }
        $deadline=[DateTimeOffset]::UtcNow.AddSeconds($TimeoutSeconds)
        do {
            $proof=Get-Proof
            if($proof.code -notin @('producer_absent','producer_waiting_backend')) {break}
            Start-Sleep -Milliseconds 250
        } while([DateTimeOffset]::UtcNow -lt $deadline)
        if($proof.code -in @('producer_absent','producer_waiting_backend')) {$proof=@{code='producer_readiness_timeout'}}
    }
    $proof|ConvertTo-Json -Compress
    if($proof.code -eq 'producer_ready') {exit 0}
    exit 2
} catch {
    $code=[string]$_.Exception.Message
    if($code -notin @('producer_identity_drift','startup_not_high_owner_interactive')) {$code='producer_observation_failed'}
    @{code=$code}|ConvertTo-Json -Compress
    exit 2
}
