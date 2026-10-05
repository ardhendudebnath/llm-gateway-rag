{{/* Chart name, overridable. */}}
{{- define "nexusgate.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Release-scoped base name every object is prefixed with. */}}
{{- define "nexusgate.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/* <fullname>-<component>, e.g. my-release-nexusgate-api. Call with (dict "root" . "component" "api"). */}}
{{- define "nexusgate.componentName" -}}
{{- printf "%s-%s" (include "nexusgate.fullname" .root) .component | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "nexusgate.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/* Labels on every object. */}}
{{- define "nexusgate.labels" -}}
helm.sh/chart: {{ include "nexusgate.chart" . }}
app.kubernetes.io/name: {{ include "nexusgate.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: nexusgate
{{- end }}

{{/* Selector labels for one component. Call with (dict "root" . "component" "api"). */}}
{{- define "nexusgate.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nexusgate.name" .root }}
app.kubernetes.io/instance: {{ .root.Release.Name }}
app.kubernetes.io/component: {{ .component }}
{{- end }}

{{- define "nexusgate.image" -}}
{{- printf "%s:%s" .Values.image.repository (default .Chart.AppVersion .Values.image.tag) }}
{{- end }}

{{- define "nexusgate.configName" -}}
{{- printf "%s-config" (include "nexusgate.fullname" .) }}
{{- end }}

{{- define "nexusgate.secretName" -}}
{{- if .Values.secrets.existingSecret }}
{{- .Values.secrets.existingSecret }}
{{- else }}
{{- include "nexusgate.fullname" . }}
{{- end }}
{{- end }}

{{/* Where Redis is: the bundled StatefulSet, or the URL given for an external one. */}}
{{- define "nexusgate.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://%s:6379/0" (include "nexusgate.componentName" (dict "root" . "component" "redis")) }}
{{- else }}
{{- required "redis.externalUrl is required when redis.enabled=false" .Values.redis.externalUrl }}
{{- end }}
{{- end }}

{{- define "nexusgate.qdrantUrl" -}}
{{- if .Values.qdrant.enabled }}
{{- printf "http://%s:6333" (include "nexusgate.componentName" (dict "root" . "component" "qdrant")) }}
{{- else }}
{{- required "qdrant.externalUrl is required when qdrant.enabled=false" .Values.qdrant.externalUrl }}
{{- end }}
{{- end }}

{{/*
Annotations that roll the pods when their configuration changes — the job kustomize's
content-hashed ConfigMap names do in the base stack.
*/}}
{{- define "nexusgate.rolloutAnnotations" -}}
checksum/config: {{ include (print .Template.BasePath "/configmap.yaml") . | sha256sum }}
{{- if not .Values.secrets.existingSecret }}
checksum/secret: {{ include (print .Template.BasePath "/secret.yaml") . | sha256sum }}
{{- end }}
{{- end }}

{{/*
Init container shared by the API and the worker. The app creates the RediSearch index and the
Qdrant collection at start-up, so it waits for both rather than crash-looping until they are up.
*/}}
{{- define "nexusgate.waitForDeps" -}}
- name: wait-for-deps
  image: {{ include "nexusgate.image" . }}
  imagePullPolicy: {{ .Values.image.pullPolicy }}
  command:
    - python
    - -c
    - |
      import os, time, urllib.request, redis
      r = redis.Redis.from_url(os.environ["NEXUSGATE_REDIS_URL"], socket_timeout=2)
      qdrant = os.environ["NEXUSGATE_QDRANT_URL"].rstrip("/") + "/readyz"
      for _ in range(90):
          try:
              r.ping()
              urllib.request.urlopen(qdrant, timeout=2)
              break
          except Exception:
              time.sleep(2)
      else:
          raise SystemExit("redis/qdrant not reachable after 180s")
  envFrom:
    - configMapRef:
        name: {{ include "nexusgate.configName" . }}
  securityContext:
    {{- toYaml .Values.securityContext | nindent 4 }}
{{- end }}
