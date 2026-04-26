# Sharing pycf on a server

A practical shared-server setup is to keep two virtual environments in the repo:

- `pycf/env` for private editable development work
- `pycf/env_share` for a shared non-editable installation used by other users

## Recommended layout

- `env`: created with `pip install -e .`
- `env_share`: created with `pip install .`

This is preferable because the shared environment does not depend on the live source tree in the same way that an editable install does.

## Shared environment setup

From the repository root:

```bash
python3 -m venv env_share
source env_share/bin/activate
pip install .
deactivate
```

Other users can then normally use:

```bash
source /home/usercode/pycf/env_share/bin/activate
```

provided that:

1. the repo and `env_share` directories are readable and executable by them
2. the environment was built on a compatible system
3. required system libraries such as LAPACK, GSL, and NLOpt are available to them

## Updating the shared environment

A non-editable install does not automatically follow source changes. After rebuilding or changing the code, reinstall into `env_share`:

```bash
cd /home/usercode/pycf
source env_share/bin/activate
pip install .
deactivate
```

After that, new activations of `env_share` will use the updated version.

## Development environment

For private development:

```bash
python3 -m venv env
source env/bin/activate
pip install -e .
```

This editable environment is appropriate for building and testing changes, but it is less suitable for sharing because it depends directly on the working tree.
