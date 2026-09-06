param(
    [Parameter(Mandatory=$true)][ValidateSet('InstallDisabled','Query','RemoveDisabled')][string]$Operation,
    [Parameter(Mandatory=$true)][string]$ReceiptRoot,
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$ReceiptSha256,
    [Parameter(Mandatory=$true)][string]$OwnerSid
)
$ErrorActionPreference='Stop'
function Sid([string]$value) {
    try {return [Security.Principal.SecurityIdentifier]::new($value).Value}
    catch {return [Security.Principal.NTAccount]::new($value).Translate([Security.Principal.SecurityIdentifier]).Value}
}
function Current {
    return Get-ScheduledTask -TaskName $script:receipt.taskName -TaskPath '\' -ErrorAction SilentlyContinue
}
function XmlNow {return [string](Export-ScheduledTask -TaskName $script:receipt.taskName -TaskPath '\')}
function ExactDisabled($task,[string]$xml) {
    return ($null -ne $task -and -not $task.Settings.Enabled -and $task.State -ne 'Running' -and
        (XmlNow) -ceq $xml)
}
$mutex=$null
$acquired=$false
try {
    & (Join-Path $PSScriptRoot 'zcode_startup_protection.ps1') -Operation Verify -Root $ReceiptRoot -OwnerSid $OwnerSid -Quiet
    $path=Join-Path $ReceiptRoot 'receipt.json'
    if((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ReceiptSha256) {throw 'receipt mismatch'}
    $script:receipt=[IO.File]::ReadAllText($path)|ConvertFrom-Json
    if($script:receipt.schema -ne 'ai-nd-co.zcode-startup-task/v1' -or $script:receipt.ownerSid -ne $OwnerSid -or
       $script:receipt.taskName -notmatch '^ai-nd-co ZCode (Owner Logon|Fixture [a-f0-9]{32})$') {throw 'receipt schema'}
    $mutex=[Threading.Mutex]::new($false,'Local\ai-nd-co-zcode-startup')
    $acquired=$mutex.WaitOne([TimeSpan]::FromSeconds(10))
    if(-not $acquired) {throw 'task operation busy'}
    $savedPath=Join-Path $ReceiptRoot 'registered.xml'
    $task=Current
    if($Operation -eq 'Query') {
        $exact=$false
        if(Test-Path -LiteralPath $savedPath) {$exact=ExactDisabled $task ([IO.File]::ReadAllText($savedPath))}
        @{present=($null -ne $task);exactDisabled=$exact}|ConvertTo-Json -Compress
        exit 0
    }
    if($Operation -eq 'InstallDisabled') {
        if($null -ne $task -or (Test-Path -LiteralPath $savedPath)) {throw 'existing task or receipt'}
        $requested=[string]$script:receipt.xml
        [xml]$parsed=$requested
        $ns=[Xml.XmlNamespaceManager]::new($parsed.NameTable)
        $ns.AddNamespace('t','http://schemas.microsoft.com/windows/2004/02/mit/task')
        $principal=$parsed.SelectSingleNode('/t:Task/t:Principals/t:Principal',$ns)
        $action=$parsed.SelectSingleNode('/t:Task/t:Actions/t:Exec',$ns)
        $settings=$parsed.SelectSingleNode('/t:Task/t:Settings',$ns)
        $trigger=$parsed.SelectSingleNode('/t:Task/t:Triggers/t:LogonTrigger',$ns)
        if($principal.UserId -ne $OwnerSid -or $principal.LogonType -ne 'InteractiveToken' -or
           $principal.RunLevel -ne 'HighestAvailable' -or $settings.Enabled -ne 'false' -or
           $settings.AllowHardTerminate -ne 'false' -or $null -ne $settings.RestartOnFailure -or
           $trigger.UserId -ne $OwnerSid -or $trigger.Enabled -ne 'true' -or
           $parsed.SelectNodes('/t:Task/t:Actions/*',$ns).Count -ne 1 -or
           $parsed.SelectNodes('/t:Task/t:Triggers/*',$ns).Count -ne 1) {throw 'requested policy'}
        Register-ScheduledTask -TaskName $script:receipt.taskName -TaskPath '\' -Xml $requested|Out-Null
        $task=Current
        $actual=@($task.Actions)
        $actualTriggers=@($task.Triggers)
        if($actual.Count -ne 1 -or $actual[0].Execute -ne $action.Command -or
           $actual[0].Arguments -ne $action.Arguments -or $actual[0].WorkingDirectory -ne $action.WorkingDirectory -or
           (Sid $task.Principal.UserId) -ne $OwnerSid -or $task.Principal.RunLevel -ne 'Highest' -or
           $task.Principal.LogonType -ne 'Interactive' -or $task.Settings.Enabled -or $task.State -eq 'Running' -or
           $task.Settings.RestartCount -ne 0 -or $task.Settings.AllowHardTerminate -or
           $actualTriggers.Count -ne 1 -or (Sid $actualTriggers[0].UserId) -ne $OwnerSid -or
           $task.Description -ne $parsed.Task.RegistrationInfo.Description) {throw 'registered policy'}
        $saved=XmlNow
        if(-not (ExactDisabled (Current) $saved)) {throw 'registration changed'}
        $stream=[IO.File]::Open($savedPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try {$bytes=[Text.UTF8Encoding]::new($false).GetBytes($saved);$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
        & (Join-Path $PSScriptRoot 'zcode_startup_protection.ps1') -Operation Protect -Root $ReceiptRoot -OwnerSid $OwnerSid -Quiet
        if(-not (ExactDisabled (Current) $saved)) {throw 'registered task changed'}
        @{changed=$true;disabled=$true}|ConvertTo-Json -Compress
        exit 0
    }
    if(-not (Test-Path -LiteralPath $savedPath)) {throw 'registration receipt absent'}
    $saved=[IO.File]::ReadAllText($savedPath)
    if(-not (ExactDisabled $task $saved)) {throw 'foreign or active task'}
    if(-not (ExactDisabled (Current) $saved)) {throw 'task changed before removal'}
    Unregister-ScheduledTask -TaskName $script:receipt.taskName -TaskPath '\' -Confirm:$false
    @{changed=$true;removed=$true}|ConvertTo-Json -Compress
} catch {
    $reason=[string]$_.Exception.Message
    if($reason -notin @('receipt mismatch','receipt schema','task operation busy','existing task or receipt',
        'requested policy','registered policy','registration changed','registered task changed',
        'registration receipt absent','foreign or active task','task changed before removal')) {$reason='operation failed'}
    @{code='startup_task_refused';reason=$reason}|ConvertTo-Json -Compress
    exit 2
} finally {
    if($acquired) {$mutex.ReleaseMutex()}
    if($null -ne $mutex) {$mutex.Dispose()}
}
