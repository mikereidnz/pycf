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


`python setup.py build_ext --inplace` failed because your virtualenv does not have **Cython** installed. That’s expected with `".[test,examples]"`: Cython is a **build** dependency, not a test/example runtime dependency. The later `python -m pip install -e ".[test,examples]"` is already doing the right thing: it creates an isolated build environment and installs build dependencies there.

For the server, use this simpler rebuild command and **skip `setup.py` directly**:

```bash
cd ~/dev/pycf
source env/bin/activate

git fetch origin
git checkout devel
git pull --ff-only origin devel

rm -f pycf/cfl.c pycf/cfl*.so
find build -name 'cfl*.so' -delete 2>/dev/null

python -m pip install -e ".[test,examples]"
```

Then check:

```bash
python -c "import pycf; print(pycf.__version__, pycf.__file__)"
python -m pytest tests/ -q
```

Only install Cython into the virtualenv if you specifically want to run `python setup.py build_ext --inplace` manually:

```bash
python -m pip install "Cython>=3,<4"
```