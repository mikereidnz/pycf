On the server, for your usual **`devel` production branch** workflow:

```bash
cd ~/dev/pycf
git fetch origin
git checkout devel
git pull --ff-only origin devel
```

Then rebuild/reinstall the editable package in your active virtual environment:

```bash
source ~/dev/pycf/env/bin/activate
python -m pip install -e ".[test,examples]"
```

Because `pycf/cfl.pyx` changed, if you want to force a clean rebuild:

```bash
rm -f pycf/cfl.c pycf/cfl*.so
find build -name 'cfl*.so' -delete 2>/dev/null
python setup.py build_ext --inplace
python -m pip install -e ".[test,examples]"
```

Check it:

```bash
git status --short --branch
python -c "import pycf; print(pycf.__version__, pycf.__file__)"
python -m pytest tests/ -q
```

You should see `devel...origin/devel` with no ahead/behind marker, and a development version ending in the new commit hash, e.g. `0.2.1.dev0+cf7b5d7`.