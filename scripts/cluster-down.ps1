<#
.SYNOPSIS
  Delete the local kind cluster. Add -StopMachine to also stop the Podman machine.
#>
[CmdletBinding()]
param([switch]$StopMachine)

$ErrorActionPreference = "Stop"
$env:KIND_EXPERIMENTAL_PROVIDER = "podman"

# kind and podman report progress on stderr ("enabling experimental podman provider"), which
# Windows PowerShell 5.1 turns into a terminating error under ErrorActionPreference=Stop — the
# cluster then survives a teardown that looks like it failed. Judge them by exit code instead, as
# cluster-up.ps1 does.
function Invoke-Native {
    $exe, $rest = $args
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $exe @rest } finally { $ErrorActionPreference = $saved }
    if ($LASTEXITCODE -ne 0) { throw "command failed (exit $LASTEXITCODE): $($args -join ' ')" }
}

Invoke-Native kind delete cluster --name nexusgate
if ($StopMachine) {
    Invoke-Native podman machine stop
}
