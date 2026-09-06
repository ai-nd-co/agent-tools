param([string]$ReceiptRoot,[string]$ReceiptSha256,[string]$PackageRoot,[string]$ManifestSha256,[string]$OutputRoot)
$ErrorActionPreference='Stop'
$owner=[Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$receipt=[IO.File]::ReadAllText((Join-Path $ReceiptRoot 'receipt.json'))|ConvertFrom-Json
$taskName=$receipt.taskName
if($taskName -notmatch '^ai-nd-co ZCode Fixture [a-f0-9]{32}$') {throw 'fixture task required'}
$helper=Join-Path $PSScriptRoot '../../src/agent_tools/zcode_startup_task.ps1'
$mediumName='ai-nd-co ZCode Fixture '+[guid]::NewGuid().ToString('N')
$created=$false
$mediumCreated=$false
$ownedXml=$null
$mediumXml=$null
function Call-Helper([string]$Operation) {
    $output=& powershell.exe -NoProfile -NonInteractive -File $helper -Operation $Operation -ReceiptRoot $ReceiptRoot -ReceiptSha256 $ReceiptSha256 -OwnerSid $owner
    return @{code=$LASTEXITCODE;result=($output|ConvertFrom-Json)}
}
try {
    $result=Call-Helper 'InstallDisabled'
    if($result.code -ne 0) {throw 'install failed'}
    $created=$true
    $ownedXml=Export-ScheduledTask -TaskName $taskName -TaskPath '\'
    if(-not (Call-Helper 'Query').result.exactDisabled) {throw 'disabled exact query failed'}
    $task=Get-ScheduledTask -TaskName $taskName -TaskPath '\'
    $originalAction=$task.Actions[0]
    $foreign=New-ScheduledTaskAction -Execute $originalAction.Execute -Argument ($originalAction.Arguments+' -NoLogo') -WorkingDirectory $originalAction.WorkingDirectory
    Set-ScheduledTask -TaskName $taskName -TaskPath '\' -Action $foreign|Out-Null
    $foreignXml=Export-ScheduledTask -TaskName $taskName -TaskPath '\'
    $ownedXml=$foreignXml
    $refused=Call-Helper 'RemoveDisabled'
    if($refused.code -ne 2 -or (Export-ScheduledTask -TaskName $taskName -TaskPath '\') -cne $foreignXml) {throw 'foreign removal not refused'}
    $original=[IO.File]::ReadAllText((Join-Path $ReceiptRoot 'registered.xml'))
    Register-ScheduledTask -TaskName $taskName -TaskPath '\' -Xml $original -Force|Out-Null
    $ownedXml=Export-ScheduledTask -TaskName $taskName -TaskPath '\'
    Enable-ScheduledTask -TaskName $taskName -TaskPath '\'|Out-Null
    $ownedXml=Export-ScheduledTask -TaskName $taskName -TaskPath '\'
    Start-ScheduledTask -TaskName $taskName -TaskPath '\'
    $deadline=[DateTimeOffset]::UtcNow.AddSeconds(180)
    $highOutput=Join-Path $OutputRoot 'high-proof.json'
    while(-not (Test-Path -LiteralPath $highOutput) -and [DateTimeOffset]::UtcNow -lt $deadline) {Start-Sleep -Milliseconds 250}
    $high=[IO.File]::ReadAllText($highOutput)|ConvertFrom-Json
    if($high.code -ne 'package_ready' -or -not $high.highOwnerSession) {throw 'actual High package proof failed'}
    Disable-ScheduledTask -TaskName $taskName -TaskPath '\'|Out-Null
    $ownedXml=Export-ScheduledTask -TaskName $taskName -TaskPath '\'
    $deadline=[DateTimeOffset]::UtcNow.AddSeconds(10)
    while((Get-ScheduledTask -TaskName $taskName -TaskPath '\').State -eq 'Running' -and [DateTimeOffset]::UtcNow -lt $deadline) {Start-Sleep -Milliseconds 100}
    $removed=Call-Helper 'RemoveDisabled'
    if($removed.code -ne 0 -or -not $removed.result.removed) {throw 'exact disabled removal failed'}
    $created=$false
    [xml]$medium=$receipt.xml
    $medium.Task.Principals.Principal.RunLevel='LeastPrivilege'
    $medium.Task.Settings.Enabled='true'
    $medium.Task.RegistrationInfo.Description=$mediumName
    $mediumOutput=Join-Path $OutputRoot 'medium-proof.json'
    $probe=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot 'zcode_startup_medium.ps1')).ProviderPath
    $medium.Task.Actions.Exec.Arguments="-NoProfile -File `"$probe`" -PackageRoot `"$PackageRoot`" -ManifestSha256 $ManifestSha256 -OwnerSid $owner -Output `"$mediumOutput`""
    Register-ScheduledTask -TaskName $mediumName -TaskPath '\' -Xml $medium.OuterXml|Out-Null
    $mediumCreated=$true
    $mediumXml=Export-ScheduledTask -TaskName $mediumName -TaskPath '\'
    Start-ScheduledTask -TaskName $mediumName -TaskPath '\'
    $deadline=[DateTimeOffset]::UtcNow.AddSeconds(180)
    while(-not (Test-Path -LiteralPath $mediumOutput) -and [DateTimeOffset]::UtcNow -lt $deadline) {Start-Sleep -Milliseconds 250}
    $mediumProof=[IO.File]::ReadAllText($mediumOutput)|ConvertFrom-Json
    if($mediumProof.highOwnerSession -or -not $mediumProof.mediumWriteRefused) {throw 'actual Medium package boundary failed'}
    @{ok=$true;actualHigh=$true;actualMedium=$true;mediumWriteRefused=$true;foreignTaskPreserved=$true;disabledInstallRemove=$true;manualStart=$true}|ConvertTo-Json -Compress
} finally {
    if($created -and (Get-ScheduledTask -TaskName $taskName -TaskPath '\' -ErrorAction SilentlyContinue)) {
      if((Export-ScheduledTask -TaskName $taskName -TaskPath '\') -ceq $ownedXml) {
        Stop-ScheduledTask -TaskName $taskName -TaskPath '\' -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $taskName -TaskPath '\' -Confirm:$false
      }
    }
    if($mediumCreated -and (Get-ScheduledTask -TaskName $mediumName -TaskPath '\' -ErrorAction SilentlyContinue)) {
      if((Export-ScheduledTask -TaskName $mediumName -TaskPath '\') -ceq $mediumXml) {
        Stop-ScheduledTask -TaskName $mediumName -TaskPath '\' -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $mediumName -TaskPath '\' -Confirm:$false
      }
    }
}
