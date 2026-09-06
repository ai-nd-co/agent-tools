param([string]$Helper,[string]$Scenario,[string]$FixtureRoot)
$ErrorActionPreference='Stop'
$global:Task354Launches=0
$global:Task354FixtureExe=Join-Path $FixtureRoot 'ZCode.exe'
[IO.File]::WriteAllText($global:Task354FixtureExe,'disposable non-executable identity fixture')
$hash=(Get-FileHash -LiteralPath $global:Task354FixtureExe -Algorithm SHA256).Hash.ToLowerInvariant()
$owner=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
function Start-Process { param($FilePath,$WorkingDirectory,$WindowStyle)
    if($FilePath -ne $global:Task354FixtureExe -or $WindowStyle -ne 'Hidden') {throw 'bad fixture launch'}
    $global:Task354Launches++
}
function Get-CimInstance { param($ClassName,$Filter)
    if($Scenario -eq 'Absent' -or ($Scenario -eq 'Launch' -and $global:Task354Launches -eq 0)) {return @()}
    $desktop=[pscustomobject]@{Name='ZCode.exe';CommandLine='ZCode.exe';ExecutablePath=$global:Task354FixtureExe;ProcessId=$PID;ParentProcessId=1;SessionId=[Diagnostics.Process]::GetCurrentProcess().SessionId}
    if($Scenario -eq 'Medium') {$desktop.ProcessId=0}
    if($Scenario -eq 'Foreign') {$desktop.ExecutablePath='C:\foreign\ZCode.exe'}
    if($Scenario -eq 'Multiple') {return @($desktop,$desktop)}
    $hostProcess=[pscustomobject]@{Name='ZCode.exe';CommandLine='--type=utility node.mojom.NodeService';ExecutablePath=$global:Task354FixtureExe;ProcessId=$PID;ParentProcessId=$PID;SessionId=$desktop.SessionId}
    $backend=[pscustomobject]@{Name='ZCode.exe';CommandLine='ZCode.exe zcode.cjs app-server';ExecutablePath=$global:Task354FixtureExe;ProcessId=$PID;ParentProcessId=$PID;SessionId=$desktop.SessionId}
    return @($desktop,$hostProcess,$backend)
}
$operation=if($Scenario -in @('Launch','Medium','Foreign','Multiple')) {'Start'}else{'Probe'}
& $Helper -Operation $operation -Executable $global:Task354FixtureExe -Sha256 $hash -OwnerSid $owner -TimeoutSeconds 1
@{launches=$global:Task354Launches}|ConvertTo-Json -Compress
exit 0
