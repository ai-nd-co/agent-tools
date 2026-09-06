param([string]$PackageRoot,[string]$ManifestSha256,[string]$OwnerSid,[string]$Output)
$ErrorActionPreference='Stop'
$PackageRoot=(Resolve-Path -LiteralPath $PackageRoot).ProviderPath
if($PackageRoot -notlike 'C:\ProgramData\ai-nd-co-task354-zcode-package-fixture-*') {throw 'fixture root required'}
$proof=& (Join-Path $PackageRoot 'launch.ps1') -ManifestSha256 $ManifestSha256 -OwnerSid $OwnerSid -Operation Probe
if($LASTEXITCODE -ne 0) {throw 'probe failed'}
$parsed=$proof|ConvertFrom-Json
$denied=$false
try {[IO.File]::AppendAllText((Join-Path $PackageRoot 'startup.json'),'fixture write must fail')}
catch [UnauthorizedAccessException] {$denied=$true}
@{highOwnerSession=$parsed.highOwnerSession;mediumWriteRefused=$denied}|ConvertTo-Json -Compress|Set-Content -LiteralPath $Output -Encoding ASCII
