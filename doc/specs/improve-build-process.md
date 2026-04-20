# Spec: Modernise the Build Process

## Goals

1. **Build is deterministic and local** — no global packages, everything
   lives in the project's `.venv`.
2. **Build is correct across branch switches** — uses content hashes (SHA256)
   instead of Make's timestamp comparison, so switching branches always
   rebuilds exactly what changed.
3. **Single entry point** — `./run <script>` ensures you're always building
   and running the latest version, with no manual steps.

The end result:

```bash
./run examples/ceylf/exdata_example.py
```

Always executes against the latest source — incrementally rebuilding only what
changed — without requiring any global package installation.

## Background

The current build (`setup.py` calling `make` in `cfl/`) uses Make's timestamp
comparison to decide what to recompile. This is fragile when switching git
branches because git updates file timestamps unpredictably. A file whose
**content** changed may keep an older timestamp than the existing `.o`, so Make
skips it and you get a stale build.

The fix is two-fold:

1. Track source file **content hashes** instead of timestamps to decide what
   needs recompiling.
2. Wrap everything in a runner script that handles the venv, build, and
   execution in one command.

## Approach — keep the change small

Rather than extracting the C build into a separate script, we replace the
Make subprocess call **inside `setup.py` itself** with hash-based logic. This
means:

- `setup.py` still does everything it did before — build C, build Cython —
  just using content hashes instead of Make timestamps
- `python setup.py build_ext --inplace` still works exactly as before
- `./run` is just a thin wrapper: venv → `setup.py build_ext --inplace` →
  run your script

This is the smallest possible change. The only file that gets modified is
`setup.py` (swapping out the Make call). One new file is added (`run`).

## Architecture

```
┌──────────────────────────────────────────────────┐
│  ./run <script>                                  │  ← user entry point
│  1. ensures .venv exists & activated             │
│  2. ensures editable install (pip install -e .)  │
│  3. python setup.py build_ext --inplace          │
│  4. runs the user's script                       │
├──────────────────────────────────────────────────┤
│  setup.py (modified)                             │
│  - replaces `make` call with hash-based build    │
│  - same behaviour: builds C lib, touches cfl.pyx │
│    if rebuilt, then Cython builds the .so         │
│  - everything else unchanged                     │
└──────────────────────────────────────────────────┘
```

## Files changed

- `setup.py` — replace the Make subprocess block (lines 46–64) with
  hash-based C compilation logic. Everything else stays identical.
- `.gitignore` — add `.build_hashes.json` and `.venv`

## New files

- `run` — thin shell wrapper script

### What changes in `setup.py` and why

The current Make block (lines 46–64) does this:

```python
if 'clean' in sys.argv:
    subprocess.call(['make', 'clean'], cwd='./cfl')
else:
    popen = subprocess.Popen(['make'], cwd='./cfl', ...)
    # ... capture output ...
    if not "make: Nothing to be done" in output:
        subprocess.call(['touch', 'pycf/cfl.pyx'])
```

This is replaced with a `build_cfl()` function that:

1. Hashes all C source and header files (SHA256 of contents)
2. Compares against `.build_hashes.json` from the last build
3. Recompiles only `.o` files whose source or headers have different hashes
4. Re-links `libcfl.a` if any `.o` was recompiled
5. Touches `pycf/cfl.pyx` if the library was rebuilt (same as before)

The rest of `setup.py` — compiler flags, link args, Intel support, version
stamping, Extension definition, `setup()` call — is completely untouched.

**Why this is safe:** `python setup.py build_ext --inplace` still works
exactly as it did before. It builds the C library and the Cython extension in
one command. The only difference is that the C build now uses content hashes
instead of Make timestamps, which is strictly more correct.

| Before                                      | After                                        |
|---------------------------------------------|----------------------------------------------|
| `python setup.py build_ext --inplace`       | `python setup.py build_ext --inplace`        |
| (still works, same command)                 | (still works, same command)                  |
| timestamp-based (fragile across branches)   | hash-based (correct across branches)         |
| `python setup.py install` (global install)  | `./run` uses editable install in .venv       |
| must remember to build before running       | `./run` does it all automatically            |

---

## Implementation Steps

Work through these in order. Each step has a **validation** check — confirm it
passes before moving to the next step.

### Step 1 — Replace the Make block in `setup.py` with hash-based build

Replace lines 46–64 in `setup.py` (the `if 'clean' in sys.argv` / `else`
block) with a `build_cfl()` function. The function should:

1. Define the dependency graph between C sources and headers, matching the
   existing Makefile rules. The object files and their dependencies are:

   | Object file      | Source                   | Header dependencies                                         |
   |------------------|--------------------------|--------------------------------------------------------------|
   | `cfl_csr.o`      | `cfl/src/cfl_csr.c`     | `cfl_csr.h`, `cfl_error.h`                                  |
   | `cfl_tensor.o`   | `cfl/src/cfl_tensor.c`  | `cfl_tensor.h`, `cfl_csr.h`, `cfl_error.h`                  |
   | `cfl_h.o`        | `cfl/src/cfl_h.c`       | `cfl_h.h`, `cfl_tensor.h`, `cfl_config.h`, `cfl_error.h`    |
   | `cfl_sh.o`       | `cfl/src/cfl_sh.c`      | `cfl_sh.h`, `cfl_tensor.h`, `cfl_error.h`                   |
   | `cfl_min.o`      | `cfl/src/cfl_min.c`     | `cfl_min.h`, `cfl_config.h`, `cfl_error.h`                  |
   | `basinhopping.o` | `cfl/src/basinhopping.c` | `basinhopping.h`, `cfl_min.h`, `cfl_config.h`, `cfl_error.h` |
   | `cfl_h_fit.o`    | `cfl/src/cfl_h_fit.c`   | `cfl_h_fit.h`, `basinhopping.h`, `cfl_min.h`, `cfl_sh.h`, `cfl_h.h`, `cfl_error.h` |
   | `cfl_zefoz.o`    | `cfl/src/cfl_zefoz.c`   | `cfl_zefoz.h`, `cfl_h.h`, `cfl_config.h`, `cfl_error.h`     |

   All headers live in `cfl/include/`.

2. SHA256-hash the contents of every source and header file.

3. Load `.build_hashes.json` (a flat dict of `filepath → hash`). If it
   doesn't exist, treat everything as changed.

4. For each object file, check whether its source **or any of its header
   dependencies** have a different hash than what's stored. If so, recompile
   that object:

   ```
   gcc -O3 -fPIC -std=c99 -ffast-math -fno-cx-limited-range -march=native \
       -I cfl/include -I /usr/include/lapacke \
       -c cfl/src/<name>.c -o cfl/<name>.o
   ```

   Print which files are being compiled (e.g.
   `Compiling cfl_csr.o (cfl_csr.h changed)`).

5. If **any** object was recompiled, re-link the static library:

   ```
   ar rcs cfl/libcfl.a cfl/cfl_csr.o cfl/cfl_tensor.o ... (all .o files)
   ```

   And touch `pycf/cfl.pyx` to trigger a Cython rebuild.

6. If **nothing** was recompiled, print `cfl: nothing to rebuild`.

7. Write the updated hashes to `.build_hashes.json`.

8. Handle `clean` in `sys.argv`: delete all `.o` files in `cfl/`,
   `cfl/libcfl.a`, and `.build_hashes.json` (same as the old
   `make clean` path).

9. Respect the existing Intel compiler support via `CFL_CC=icc` environment
   variable, and `CFL_CFLAGS` / `CFL_LDLIBS` env vars for extra flags
   (same logic as the existing Makefile).

Call `build_cfl()` in the same place where the Make block used to be, so the
flow of `setup.py` is unchanged: build C → stamp version → define Extension →
`setup()`.

**Validation:**

```bash
# Start clean
make clean -C cfl && rm -f .build_hashes.json

# First run — should compile everything
python setup.py build_ext --inplace
# Verify: all 8 .o files compiled, libcfl.a exists, .so created,
#         .build_hashes.json created
ls cfl/*.o cfl/libcfl.a pycf/cfl.cpython-*.so .build_hashes.json

# Second run — should skip C build
python setup.py build_ext --inplace
# Verify: prints "cfl: nothing to rebuild", Cython also skips

# Modify one source — should recompile only that one
echo "" >> cfl/src/cfl_csr.c
python setup.py build_ext --inplace
git checkout cfl/src/cfl_csr.c
# Verify: only cfl_csr.o recompiled, libcfl.a re-linked, .so rebuilt

# Run the test
python -m pytest tests/exdata_test.py -v
# Verify: all 3 parametrized tests pass

# Clean
python setup.py clean
# Verify: no .o, no libcfl.a, no .build_hashes.json
```

### Step 2 — Create the `run` script

Create an executable `run` script (bash, `chmod +x`) in the project root.
This is a thin wrapper — all the build logic is in `setup.py`.

It should do the following, in order, stopping on any error:

1. **Require an argument** — print usage and exit if no script path given:
   ```
   Usage: ./run <python-script> [args...]
   ```

2. **Ensure `.venv` exists** — if `.venv/bin/python` doesn't exist:
   ```bash
   python3 -m venv .venv
   ```

3. **Activate the venv** — source `.venv/bin/activate`.

4. **Ensure dependencies are installed** — use a stamp file
   (`.venv/.pycf_installed`) to avoid running pip every time:
   ```bash
   if [ ! -f .venv/.pycf_installed ]; then
       pip install -e . && touch .venv/.pycf_installed
   fi
   ```
   This handles numpy, cython, and registers pycf as an editable package.

5. **Build** — `python setup.py build_ext --inplace`. This calls the
   hash-based C build internally, then builds Cython if needed. When nothing
   has changed, it's fast (just hash comparisons + Cython's own check).

6. **Run the user's script** with any extra arguments:
   ```bash
   python "$@"
   ```

**Validation:**

```bash
# Remove the venv to test from scratch
rm -rf .venv

# Run an example — should create venv, install, build, and run
./run examples/ceylf/exdata_example.py
# Verify: output shows crystal field fitting results

# Run again — should be fast (no rebuild, no install)
time ./run examples/ceylf/exdata_example.py
# Verify: prints "cfl: nothing to rebuild", runs quickly

# Run the tests through the runner
./run -m pytest tests/exdata_test.py -v
# Verify: all 3 parametrized tests pass
```

### Step 3 — Update `.gitignore`

Add these entries to `.gitignore`:

```
.build_hashes.json
.venv
```

`.venv` may already be gitignored via a global gitignore, but it's good to be
explicit in the repo.

**Validation:**

```bash
git status
# Verify: .build_hashes.json and .venv/ do not show as untracked
# Verify: run and the modified setup.py DO show as changes
```

### Step 4 — End-to-end branch-switching test

This is the key scenario: switching branches must always produce a correct
build.

```bash
# On the current branch, run the example
./run examples/ceylf/exdata_example.py

# Switch to master
git stash  # if needed
git checkout master

# Run again — hashes differ, triggers rebuild
./run examples/ceylf/exdata_example.py

# Switch back
git checkout <your-branch>
git stash pop  # if needed

# Run again — should rebuild only what changed
./run examples/ceylf/exdata_example.py
```

Each run should produce correct output without needing `make clean` or any
manual intervention.

### Step 5 — Commit and push

```bash
git add run setup.py .gitignore
git commit -m "Hash-based C build in setup.py and ./run entry point

- setup.py: replaced Make call with content-hash build (SHA256).
  Recompiles only .o files whose source or headers actually changed.
  Correct across branch switches.
- run: thin wrapper that manages .venv and calls setup.py + runs script
- .gitignore: added .build_hashes.json and .venv"
git push origin HEAD
```

---

## Next Steps

### Replace `setup.py` with `pyproject.toml`

Orthogonal to the build correctness problem. `distutils` is removed in
Python 3.12+ and `setup.py` is deprecated. Replace with a `pyproject.toml`
using `setuptools` as the build backend.

### Replace custom hash logic with Meson or CMake

The content-hash build in this spec is a pragmatic fix, but it's a custom
solution maintaining a dependency graph by hand. The industry-standard
approach for C + Cython projects is to use a proper build system that handles
dependency tracking, incremental builds, and cross-platform compilation
natively.

**Option A: Meson + meson-python (recommended)**

Meson is a modern build system that natively supports C and Cython. It
uses content-hash-based rebuilds out of the box. `meson-python` is the
Python packaging bridge — it replaces `setup.py` entirely.

This is what **NumPy** and **SciPy** use. For pycf it would look like:

- `meson.build` in the project root — defines the Python package
- `cfl/meson.build` — defines the C static library (sources, includes,
  dependencies on GSL/LAPACK/NLopt)
- `pycf/meson.build` — defines the Cython extension, linking against
  the C library
- `pyproject.toml` — declares `meson-python` as the build backend

Meson handles the dependency graph, incremental builds, and compiler
detection automatically. No hand-maintained hash files or dependency
tables.

Install: `pip install meson-python meson ninja`

**Option B: CMake + scikit-build-core**

CMake is the other major option. `scikit-build-core` bridges CMake with
Python packaging. There's also `cython-cmake` for Cython-specific helpers.

- `CMakeLists.txt` in root — defines the C library and Cython extension
- `pyproject.toml` — declares `scikit-build-core` as the build backend

CMake is more verbose than Meson but has a larger ecosystem and more
IDE support.

**Either option replaces:** `setup.py`, the Makefile, and the custom
hash-based build logic from this spec — all in one go. The `./run` script
would simplify to just venv setup + `pip install -e .` + run, since the
build system handles everything else.
