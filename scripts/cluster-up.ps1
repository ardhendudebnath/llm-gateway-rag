<#
.SYNOPSIS
  One command: build the API image with Podman, run a local kind cluster on Podman, deploy NexusGate.

.DESCRIPTION
  Safe to re-run: it reuses the Podman machine and the cluster, rebuilds and reloads the image, and
  restarts the API pods onto it. Every kubectl call is pinned to the kind-nexusgate context, so it
  never touches another cluster in your kubeconfig.

  Secrets come from .env (NEXUSGATE_ADMIN_TOKEN, NEXUSGATE_JWT_SECRET, NEXUSGATE_QDRANT_API_KEY,
  ANTHROPIC_API_KEY, OPENAI_API_KEY). Missing NEXUSGATE_* secrets are generated once and kept
  across runs.

.EXAMPLE
  ./scripts/cluster-up.ps1             # create or update
  ./scripts/cluster-up.ps1 -Recreate   # delete and recreate the cluster first
#>
[CmdletBinding()]
param([switch]$Recreate)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Cluster = "nexusgate"
$Context = "kind-$Cluster"
$Namespace = "nexusgate"
$Image = "localhost/nexusgate-api:dev"
$env:KIND_EXPERIMENTAL_PROVIDER = "podman"

# Native tools report progress on stderr, which Windows PowerShell 5.1 turns into terminating
# errors under ErrorActionPreference=Stop. Judge them by exit code instead.
function Invoke-Native {
    $exe, $rest = $args
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($MyInvocation.ExpectingInput) { $input | & $exe @rest } else { & $exe @rest }
    } finally { $ErrorActionPreference = $saved }
    if ($LASTEXITCODE -ne 0) { throw "command failed (exit $LASTEXITCODE): $($args -join ' ')" }
}
# For probes where failure is an expected answer (e.g. "does this secret exist?").
function Invoke-Probe {
    $exe, $rest = $args
    $saved = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try { & $exe @rest 2>$null } finally { $ErrorActionPreference = $saved }
}
function Step($message) { Write-Host "`n==> $message" -ForegroundColor Cyan }
function New-Secret {
    $bytes = New-Object byte[] 32
    [Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_")
}

foreach ($tool in "podman", "kind", "kubectl") {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        throw "$tool not found on PATH. Install: winget install RedHat.Podman Kubernetes.kind Kubernetes.kubectl"
    }
}

Step "Podman machine"
# kind needs a rootful machine to create its node container.
# Windows PowerShell 5.1's ConvertFrom-Json returns a JSON array as ONE object; unroll it.
$machines = @((Invoke-Native podman machine list --format json | Out-String | ConvertFrom-Json) |
    Where-Object { $_ })
if ($machines.Count -eq 0) {
    Invoke-Native podman machine init --rootful --now
} else {
    $machine = @($machines | Where-Object { $_.Default }) + $machines | Select-Object -First 1
    $name = $machine.Name
    $rootful = (Invoke-Native podman machine inspect $name --format "{{.Rootful}}") -eq "true"
    if (-not $rootful) {
        if ($machine.Running) { Invoke-Native podman machine stop $name }
        Invoke-Native podman machine set --rootful $name
        Invoke-Native podman machine start $name
    } elseif (-not $machine.Running) {
        Invoke-Native podman machine start $name
    } else {
        Write-Host "machine '$name' is running (rootful)"
    }
}

Step "kind cluster '$Cluster'"
$exists = @(Invoke-Probe kind get clusters) -contains $Cluster
if ($exists -and $Recreate) {
    Invoke-Native kind delete cluster --name $Cluster
    $exists = $false
}
if ($exists) {
    Write-Host "reusing existing cluster"
} else {
    Invoke-Native kind create cluster --config "$Root/infra/k8s/overlays/kind/cluster.yaml" --wait 180s
}

Step "Build $Image with Podman"
Invoke-Native podman build --tag $Image --file "$Root/Containerfile" $Root

Step "Load image into kind"
$archive = Join-Path ([IO.Path]::GetTempPath()) "nexusgate-api.tar"
try {
    Invoke-Native podman save --format docker-archive --output $archive $Image
    Invoke-Native kind load image-archive $archive --name $Cluster
} finally {
    Remove-Item $archive -ErrorAction SilentlyContinue
}

Step "Secret nexusgate-secrets"
Invoke-Native kubectl --context $Context apply -f "$Root/infra/k8s/base/namespace.yaml"

$dotenv = @{}
$envFile = Join-Path $Root ".env"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile) {
        if ($line -match '^\s*([A-Z0-9_]+)\s*=\s*([^#]*?)\s*(#.*)?$') {
            $dotenv[$Matches[1]] = $Matches[2].Trim('"', "'")
        }
    }
}
$existing = @{}
$current = Invoke-Probe kubectl --context $Context -n $Namespace get secret nexusgate-secrets -o json
if ($LASTEXITCODE -eq 0 -and $current) {
    ($current | Out-String | ConvertFrom-Json).data.PSObject.Properties | ForEach-Object {
        $existing[$_.Name] = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($_.Value))
    }
}
$secrets = [ordered]@{}
$keys = "NEXUSGATE_ADMIN_TOKEN", "NEXUSGATE_JWT_SECRET", "NEXUSGATE_QDRANT_API_KEY",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY"
foreach ($key in $keys) {
    $value = $dotenv[$key]
    if (-not $value -or $value -like "change-me*") { $value = $existing[$key] }
    if (-not $value -and $key -like "NEXUSGATE_*") { $value = New-Secret }  # prod refuses defaults
    if ($value) { $secrets[$key] = $value }
}
# Through a temp file rather than --from-literal, so values don't appear in the process list.
$secretFile = [IO.Path]::GetTempFileName()
try {
    $secrets.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" } |
        Set-Content -Encoding ascii $secretFile
    $manifest = Invoke-Native kubectl --context $Context -n $Namespace create secret generic `
        nexusgate-secrets --from-env-file=$secretFile --dry-run=client -o yaml
    $manifest | Invoke-Native kubectl --context $Context apply -f -
} finally {
    Remove-Item $secretFile -ErrorAction SilentlyContinue
}
$providers = @("ANTHROPIC_API_KEY", "OPENAI_API_KEY" | Where-Object { $secrets.Contains($_) })
Write-Host ("provider keys: " + $(if ($providers) { $providers -join ", " } else { "none (mock and chaos routes only)" }))

Step "Deploy (kustomize overlay infra/k8s/overlays/kind)"
Invoke-Native kubectl --context $Context apply -k "$Root/infra/k8s/overlays/kind"
# The tag is always :dev, so restart the API onto the image that was just loaded.
Invoke-Native kubectl --context $Context -n $Namespace rollout restart deployment/api deployment/worker
$workloads = "statefulset/redis", "statefulset/qdrant", "deployment/api", "deployment/worker",
    "deployment/prometheus", "deployment/alertmanager", "deployment/grafana",
    "deployment/prometheus-adapter"
foreach ($workload in $workloads) {
    # 10 minutes: on a fresh cluster these pods wait on first-time image pulls, and giving up
    # early fails a deployment that was only slow.
    Invoke-Native kubectl --context $Context -n $Namespace rollout status $workload --timeout=600s
}

Step "NexusGate is up"
Write-Host @"
  API docs    http://localhost:8000/docs
  Prometheus  http://localhost:9090
  Grafana     http://localhost:3000
  Admin token $($secrets["NEXUSGATE_ADMIN_TOKEN"])

  Smoke test  .venv\Scripts\python scripts\smoke_test.py --admin-token <token> --prometheus-url http://localhost:9090
  Tear down   ./scripts/cluster-down.ps1
"@
