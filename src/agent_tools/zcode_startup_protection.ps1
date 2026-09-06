# Reused from CWR Task354 protected bootstrap at 4fa735a; keep the ACL contract aligned.
[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidateSet('Protect','Verify')][string]$Operation,
    [Parameter(Mandatory=$true)][string]$Root,
    [Parameter(Mandatory=$true)][string]$OwnerSid,
    [switch]$Quiet
)
$ErrorActionPreference = 'Stop'
$admins = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-32-544')
$system = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18')
$owner = [System.Security.Principal.SecurityIdentifier]::new($OwnerSid)
if ([System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value -ne $OwnerSid) { throw 'bootstrap_owner_mismatch' }
$rootItem = Get-Item -LiteralPath $Root -Force
$items = @($rootItem)
if ($rootItem.PSIsContainer) { $items += @(Get-ChildItem -LiteralPath $Root -Recurse -Force) }
foreach ($item in $items) {
    if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'bootstrap_link_refused' }
    if ($Operation -eq 'Protect') {
        if ($item.PSIsContainer) {
            $acl = [Security.AccessControl.DirectorySecurity]::new()
            $inherit = [Security.AccessControl.InheritanceFlags]'ContainerInherit,ObjectInherit'
        } else {
            $acl = [Security.AccessControl.FileSecurity]::new()
            $inherit = [Security.AccessControl.InheritanceFlags]::None
        }
        $acl.SetOwner($admins)
        $acl.SetAccessRuleProtection($true, $false)
        foreach ($identity in @($admins,$system)) {
            $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($identity,'FullControl',$inherit,'None','Allow'))
        }
        $acl.AddAccessRule([Security.AccessControl.FileSystemAccessRule]::new($owner,'ReadAndExecute',$inherit,'None','Allow'))
        Set-Acl -LiteralPath $item.FullName -AclObject $acl
    }
    $observed = Get-Acl -LiteralPath $item.FullName
    if ($observed.GetOwner([Security.Principal.SecurityIdentifier]).Value -notin @($admins.Value,$system.Value)) { throw 'bootstrap_owner_unprotected' }
    if (-not $observed.AreAccessRulesProtected) { throw 'bootstrap_acl_inherited' }
    foreach ($rule in $observed.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])) {
        if ($rule.AccessControlType -eq 'Allow' -and $rule.IdentityReference.Value -notin @($admins.Value,$system.Value) -and
            ([int64]$rule.FileSystemRights -band 0x000D0156) -ne 0) { throw 'bootstrap_writable' }
    }
}
# A protected child can still be replaced through DELETE_CHILD on an ancestor.
$ancestor = if ($rootItem.PSIsContainer) { $rootItem.Parent } else { $rootItem.Directory }
while ($null -ne $ancestor) {
    if (($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw 'bootstrap_ancestor_link' }
    $acl = Get-Acl -LiteralPath $ancestor.FullName
    foreach ($rule in $acl.GetAccessRules($true,$true,[Security.Principal.SecurityIdentifier])) {
        if (($rule.PropagationFlags -band [Security.AccessControl.PropagationFlags]::InheritOnly) -ne 0) { continue }
        if ($rule.AccessControlType -eq 'Allow' -and $rule.IdentityReference.Value -notin @($admins.Value,$system.Value,'S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464') -and
            ([int64]$rule.FileSystemRights -band 0x000D0040) -ne 0) { throw 'bootstrap_ancestor_replaceable' }
    }
    $ancestor = $ancestor.Parent
}
if (-not $Quiet) { [Console]::Out.Write('{"ok":true}') }
