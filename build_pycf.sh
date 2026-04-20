#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREFIX="${PYCF_PREFIX:-/home/users/mfr24/usr/local}"
SITE_PACKAGES="$(
python - <<'PY' "$PREFIX"
import sys
import sysconfig

prefix = sys.argv[1]
print(sysconfig.get_path("purelib", vars={"base": prefix, "platbase": prefix}))
PY
)"

cd "$ROOT_DIR"

echo "Building pycf in place..."
python setup.py build_ext --inplace

echo
if [ -d "$SITE_PACKAGES" ]; then
    echo "Removing stale pycf egg-info metadata from $SITE_PACKAGES ..."
    rm -rf "$SITE_PACKAGES"/pycf-*.egg-info
    echo
fi

echo "Target site-packages: $SITE_PACKAGES"
echo
echo "Installing pycf to $PREFIX ..."
python setup.py install --prefix="$PREFIX"

echo
echo "Done."
