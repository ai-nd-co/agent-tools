param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[a-f0-9]{64}$')][string]$ManifestSha256,
    [Parameter(Mandatory=$true)][string]$OwnerSid,
    [ValidateSet('Run','Probe','Stop','Status')][string]$Operation='Run',
    [string]$ProbeOutput
)
$ErrorActionPreference='Stop'
$env:PSModulePath=Join-Path ([Environment]::GetFolderPath('Windows')) 'System32\WindowsPowerShell\v1.0\Modules'
$env:PYTHONPATH=$null
$env:PYTHONHOME=$null
$env:PYTHONSTARTUP=$null
try {
    & (Join-Path $PSScriptRoot 'protect.ps1') -Operation Verify -Root $PSScriptRoot -OwnerSid $OwnerSid -Quiet
    $manifestPath=Join-Path $PSScriptRoot 'manifest.json'
    if((Get-FileHash -LiteralPath $manifestPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $ManifestSha256) {throw 'manifest mismatch'}
    $manifest=[IO.File]::ReadAllText($manifestPath)|ConvertFrom-Json
    if($manifest.schema -ne 'ai-nd-co.zcode-startup-package/v1') {throw 'manifest schema'}
    $seen=@{}
    foreach($entry in $manifest.files) {
        $relative=[string]$entry.path
        if($relative -match '[:\\]' -or @($relative.Split('/')|Where-Object {$_ -in @('','.','..')}).Count -gt 0 -or $seen.ContainsKey($relative)) {throw 'manifest path'}
        $seen[$relative]=$true
        $file=Get-Item -LiteralPath (Join-Path $PSScriptRoot $relative)
        if($file.PSIsContainer -or $file.Length -ne $entry.bytes -or
           (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) {throw 'manifest file'}
    }
    foreach($file in Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -Force -File) {
        $relative=$file.FullName.Substring($PSScriptRoot.Length+1).Replace('\','/')
        if($relative -ne 'manifest.json' -and -not $seen.ContainsKey($relative)) {throw 'unlisted file'}
    }
    $settings=[IO.File]::ReadAllText((Join-Path $PSScriptRoot 'startup.json'))|ConvertFrom-Json
    if($settings.ownerSid -ne $OwnerSid) {throw 'owner mismatch'}
    if($ProbeOutput) {
        if($Operation -ne 'Probe' -or -not [IO.Path]::IsPathRooted($ProbeOutput)) {throw 'invalid probe output'}
        $result=& (Join-Path $PSScriptRoot 'runtime/python.exe') -I -B (Join-Path $PSScriptRoot 'bootstrap.py') probe
        $code=$LASTEXITCODE
        $stream=[IO.File]::Open($ProbeOutput,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::None)
        try {$bytes=[Text.UTF8Encoding]::new($false).GetBytes([string]$result);$stream.Write($bytes,0,$bytes.Length)}finally{$stream.Dispose()}
        exit $code
    }
    & (Join-Path $PSScriptRoot 'runtime/python.exe') -I -B (Join-Path $PSScriptRoot 'bootstrap.py') $Operation.ToLowerInvariant()
    exit $LASTEXITCODE
} catch {
    [Console]::Out.Write('{"code":"startup_package_invalid"}')
    exit 2
}
