<#
.SYNOPSIS
  Delete the local kind cluster. Add -StopMachine to also stop the Podman machine.
#>
[CmdletBinding()]
param([switch]$StopMachine)

$ErrorActionPreference = "Stop"
$env:KIND_EXPERIMENTAL_PROVIDER = "podman"

kind delete cluster --name nexusgate
if ($LASTEXITCODE -ne 0) { throw "kind delete cluster failed" }
if ($StopMachine) {
    podman machine stop
    if ($LASTEXITCODE -ne 0) { throw "podman machine stop failed" }
}
