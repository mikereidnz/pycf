# Spec: Modernise the Build Process

## Goal

Replace the timestamp-based Make build with a content-hash-based build system
and provide a single `./run` script so that:

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

## Architecture

```
┌──────────────────────────────────────────────────┐
│  ./run <script>                                  │  ← user entry point
│  1. ensures .venv exists & activated             │
│  2. ensures editable install (pip install -e .)  │
│  3. calls build_cfl.py (hash-based C build)      │
│  4. rebuilds Cython .so if needed                │
│  5. runs the user's script                       │
├──────────────────────────────────────────────────┤
│  build_cfl.py                                    │  ← replaces `make` call
│  - hashes cfl/src/*.c + cfl/include/*.h          │
│  - compares against .build_hashes.json           │
│  - recompiles only changed .o files              │
│  - re-links libcfl.a if any .o changed           │
│  - touches pycf/cfl.pyx if library rebuilt       │
├──────────────────────────────────────────────────┤
│  setup.py (modified)                             │
│  - removes the `make` subprocess call            │
│  - removes the `touch cfl.pyx` logic             │
│  - keeps Extension/Cython/link config            │
│  - build_cfl.py is called externally by ./run    │
└──────────────────────────────────────────────────┘
```

## Existing files that will be modified

- `setup.py` — remove the Make invocation and touch logic (lines 46–64 in
  the current mfr-upgrade-python version)
- `.gitignore` — add `.build_hashes.json`

### Why modifying `setup.py` is safe

The Make call inside `setup.py` currently does two things:

1. Compiles C sources into `.o` files and links `libcfl.a`
2. Touches `pycf/cfl.pyx` if anything was rebuilt (to trigger Cython)

Both of these are taken over by `build_cfl.py`, which does the same work but
uses content hashes instead of timestamps. The `./run` script calls
`build_cfl.py` **before** `setup.py`, so by the time `setup.py` runs,
`libcfl.a` already exists and `cfl.pyx` has already been touched if needed.

After this change:

| Before                                      | After                                        |
|---------------------------------------------|----------------------------------------------|
| `python setup.py build_ext --inplace`       | `./run examples/ceylf/exdata_example.py`     |
| builds C lib + Cython + runs nothing        | builds C lib + Cython + runs your script     |
| timestamp-based (fragile across branches)   | hash-based (correct across branches)         |
| `python setup.py install` (global install)  | `./run` uses editable install in .venv       |

The old `python setup.py build_ext --inplace` command still works for the
Cython step — it just no longer triggers `make` itself. You'd need to run
`python build_cfl.py` first. But the whole point is that `./run` does
everything in one command, so there's no reason to call `setup.py` directly
anymore.

## New files

- `build_cfl.py` — hash-based C build script
- `run` — shell entry-point script

---

## Implementation Steps

Work through these in order. Each step has a **validation** check — confirm it
passes before moving to the next step.

### Step 1 — Create `build_cfl.py` with hash-based compilation

Create `build_cfl.py` in the project root. It should:

1. Define the dependency graph between C sources and headers, matching the
   existing Makefile rules. The object files and their dependencies are:

   | Object file      | Source                | Header dependencies                                    |
   |------------------|-----------------------|---------------------------------------------------------|
   | `cfl_csr.o`      | `cfl/src/cfl_csr.c`  | `cfl_csr.h`, `cfl_error.h`                             |
   | `cfl_tensor.o`   | `cfl/src/cfl_tensor.c` | `cfl_tensor.h`, `cfl_csr.h`, `cfl_error.h`           |
   | `cfl_h.o`        | `cfl/src/cfl_h.c`    | `cfl_h.h`, `cfl_tensor.h`, `cfl_config.h`, `cfl_error.h` |
   | `cfl_sh.o`       | `cfl/src/cfl_sh.c`   | `cfl_sh.h`, `cfl_tensor.h`, `cfl_error.h`              |
   | `cfl_min.o`      | `cfl/src/cfl_min.c`  | `cfl_min.h`, `cfl_config.h`, `cfl_error.h`             |
   | `basinhopping.o` | `cfl/src/basinhopping.c` | `basinhopping.h`, `cfl_min.h`, `cfl_config.h`, `cfl_error.h` |
   | `cfl_h_fit.o`    | `cfl/src/cfl_h_fit.c` | `cfl_h_fit.h`, `basinhopping.h`, `cfl_min.h`, `cfl_sh.h`, `cfl_h.h`, `cfl_error.h` |
   | `cfl_zefoz.o`    | `cfl/src/cfl_zefoz.c` | `cfl_zefoz.h`, `cfl_h.h`, `cfl_config.h`, `cfl_error.h` |

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

   Print which files are being compiled (e.g. `Compiling cfl_csr.o (cfl_csr.h changed)`).

5. If **any** object was recompiled, re-link the static library:

   ```
   ar rcs cfl/libcfl.a cfl/cfl_csr.o cfl/cfl_tensor.o ... (all .o files)
   ```

   And touch `pycf/cfl.pyx` to trigger a Cython rebuild.

6. If **nothing** was recompiled, print `cfl: nothing to rebuild` and exit
   cleanly.

7. Write the updated hashes to `.build_hashes.json`.

8. Support a `--clean` flag that deletes all `.o` files, `libcfl.a`, and
   `.build_hashes.json`.

9. Support the Intel compiler via `CFL_CC=icc` environment variable (same
   logic as the existing Makefile's `ifeq` block), and `CFL_CFLAGS` /
   `CFL_LDLIBS` env vars for extra flags.

10. The script should be importable (`import build_cfl`) as well as runnable
    (`python build_cfl.py`), so that `setup.py` or the run script can call it
    programmatically.

**Validation:**

```bash
# Start clean
make clean -C cfl && rm -f .build_hashes.json

# First run — should compile everything
python build_cfl.py
# Verify: all 8 .o files printed, libcfl.a exists, .build_hashes.json created
ls cfl/*.o cfl/libcfl.a .build_hashes.json

# Second run — should do nothing
python build_cfl.py
# Verify: prints "cfl: nothing to rebuild"

# Touch a header — should recompile dependents
touch cfl/include/cfl_error.h
python build_cfl.py
# Verify: recompiles all 8 (everything depends on cfl_error.h)

# Modify one source — should recompile only that one
echo "" >> cfl/src/cfl_csr.c && python build_cfl.py
git checkout cfl/src/cfl_csr.c
# Verify: only cfl_csr.o recompiled, libcfl.a re-linked

# Clean
python build_cfl.py --clean
# Verify: no .o, no libcfl.a, no .build_hashes.json
```

### Step 2 — Modify `setup.py` to remove the Make call

Edit `setup.py`:

1. Remove lines 46–64 (the `if 'clean' in sys.argv` / `else` block that
   calls `make` and touches `cfl.pyx`).
2. The rest of `setup.py` stays as-is: the Extension definition, link args,
   version stamping, and `setup()` call are all still needed for
   `pip install -e .` and `build_ext --inplace` to work.

`setup.py` now assumes that `cfl/libcfl.a` already exists when it runs.
The `./run` script (Step 3) calls `build_cfl.py` first to ensure this.

**Validation:**

```bash
# Build the C library with the new system
python build_cfl.py

# Then build the Cython extension via setup.py
python setup.py build_ext --inplace

# Verify the .so was created
ls pycf/cfl.cpython-*.so

# Run the test
python -m pytest tests/exdata_test.py -v
```

### Step 3 — Create the `run` script

Create an executable `run` script (bash, `chmod +x`) in the project root.

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

5. **Run the hash-based C build** — `python build_cfl.py`. This is
   incremental and fast when nothing changed.

6. **Rebuild the Cython extension if needed** — only if `cfl/libcfl.a` is
   newer than the `.so`, or if `pycf/cfl.pyx` was touched (by build_cfl.py),
   or if the `.so` doesn't exist:
   ```bash
   python setup.py build_ext --inplace
   ```
   Since `build_cfl.py` touches `cfl.pyx` when it rebuilds the library,
   distutils/Cython will naturally skip compilation when nothing changed.
   So it is fine to always run this command — it will be a no-op when the
   `.so` is up to date.

7. **Run the user's script** with any extra arguments:
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
# Verify: build_cfl.py prints "cfl: nothing to rebuild", runs quickly

# Run the tests through the runner
./run -m pytest tests/exdata_test.py -v
# Verify: all 3 parametrized tests pass
```

### Step 4 — Update `.gitignore`

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
# Verify: build_cfl.py, run, and the modified setup.py DO show as changes
```

### Step 5 — End-to-end branch-switching test

This is the key scenario: switching branches must always produce a correct
build.

```bash
# On the current branch, run the example
./run examples/ceylf/exdata_example.py

# Switch to master
git stash  # if needed
git checkout master

# Run again — build_cfl.py should detect content changes and rebuild
./run examples/ceylf/exdata_example.py

# Switch back
git checkout spec/improve-build-process-doc
git stash pop  # if needed

# Run again — should rebuild only what changed
./run examples/ceylf/exdata_example.py
```

Each run should produce correct output without needing `make clean` or any
manual intervention.

### Step 6 — Commit and push

```bash
git add build_cfl.py run setup.py .gitignore doc/specs/improve-build-process.md
git commit -m "Add hash-based build system and ./run entry point

- build_cfl.py: content-hash C build replacing Make's timestamp logic
- run: single entry point that manages venv, build, and execution
- setup.py: removed Make invocation (now handled by build_cfl.py)
- .gitignore: added .build_hashes.json and .venv"
git push origin HEAD
```

---

## Next Steps

- **Replace setup.py with pyproject.toml** — a worthwhile follow-up but
  orthogonal to the build correctness problem. Do this separately.
