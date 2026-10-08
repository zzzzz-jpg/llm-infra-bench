#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# lint_k8s.sh — build every kustomize target and validate it against the real
# Kubernetes + CRD schemas. Catches the mistakes that only show up at apply
# time (wrong field names in a ServiceMonitor, a patch that silently matched
# nothing, an invalid HPA metric block).
#
#   ./scripts/lint_k8s.sh          # downloads kubeconform/kustomize to .tools/
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
mkdir -p .tools

KUBECONFORM_VERSION="${KUBECONFORM_VERSION:-v0.6.7}"
KUSTOMIZE_VERSION="${KUSTOMIZE_VERSION:-v5.4.3}"
ARCH="$(uname -m)"; [[ "$ARCH" == "x86_64" ]] && ARCH=amd64
[[ "$ARCH" == "aarch64" ]] && ARCH=arm64
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"

if [[ ! -x .tools/kubeconform ]]; then
  echo "[lint] downloading kubeconform ${KUBECONFORM_VERSION}"
  curl -fsSL "https://github.com/yannh/kubeconform/releases/download/${KUBECONFORM_VERSION}/kubeconform-${OS}-${ARCH}.tar.gz" \
    | tar xz -C .tools kubeconform
  chmod +x .tools/kubeconform
fi
if [[ ! -x .tools/kustomize ]]; then
  echo "[lint] downloading kustomize ${KUSTOMIZE_VERSION}"
  curl -fsSL "https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize%2F${KUSTOMIZE_VERSION}/kustomize_${KUSTOMIZE_VERSION}_${OS}_${ARCH}.tar.gz" \
    | tar xz -C .tools kustomize
  chmod +x .tools/kustomize
fi

CRD_CATALOG='https://raw.githubusercontent.com/datreeio/CRDs-catalog/main/{{.Group}}/{{.ResourceKind}}_{{.ResourceAPIVersion}}.json'
rc=0
for target in k8s/base k8s/overlays/prod; do
  name=$(echo "$target" | tr '/' '-')
  echo "=============================================================="
  echo "[lint] kustomize build $target"
  if ! ./.tools/kustomize build "$target" >".tools/${name}.yaml"; then
    echo "[lint] BUILD FAILED: $target" >&2; rc=1; continue
  fi
  echo "[lint]   -> $(grep -c '^kind:' ".tools/${name}.yaml") resources"
  echo "[lint] kubeconform -strict $target"
  ./.tools/kubeconform -summary -strict -ignore-missing-schemas \
    -schema-location default -schema-location "$CRD_CATALOG" \
    ".tools/${name}.yaml" || rc=1
done

for f in observability/dcgm/dcgm-exporter.yaml; do
  echo "[lint] kubeconform $f"
  ./.tools/kubeconform -summary -ignore-missing-schemas "$f" || rc=1
done

# --- guard: an overlay patch that matched nothing is silent and deadly ------
echo "[lint] overlay sanity checks"
for probe in "replicas: 3" "maxReplicas: 8" "llm.prod.example.com" "DoNotSchedule"; do
  if ! grep -q "$probe" .tools/k8s-overlays-prod.yaml; then
    echo "[lint] FAIL: prod overlay did not produce '$probe' — a patch stopped matching" >&2
    rc=1
  fi
done

echo "[lint] python yaml parse of every manifest"
python3 - <<'PY' || rc=1
import glob, sys, yaml
bad = []
for f in glob.glob("k8s/**/*.yaml", recursive=True) + glob.glob("observability/**/*.yaml", recursive=True):
    try:
        list(yaml.safe_load_all(open(f)))
    except Exception as e:
        bad.append(f"{f}: {e}")
print("\n".join(bad) if bad else "  all yaml parses")
sys.exit(1 if bad else 0)
PY

[[ $rc -eq 0 ]] && echo "[lint] PASS" || echo "[lint] FAIL" >&2
exit $rc
