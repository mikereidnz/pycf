# Python upgrade side-by-side diffs

Compared against `devel`.

## `setup.py`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
from distutils.core import setup
from distutils.extension import Extension
from distutils.spawn import find_executable

import numpy.distutils.intelccompiler
``` | ```python
from setuptools import setup, Extension
from shutil import which

try:
    import numpy.distutils.intelccompiler
except ImportError:
    pass
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
if '--compiler=intel' in sys.argv:
    if find_executable('icc') == None:
        raise RuntimeError("Cannot locate the icc compiler.")
    else:
        intelpath = find_executable('icc')[:-len('/bin/icc')]
``` | ```python
if '--compiler=intel' in sys.argv:
    icc = which('icc')
    if icc == None:
        raise RuntimeError("Cannot locate the icc compiler.")
    else:
        intelpath = icc[:-len('/bin/icc')]
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
popen = subprocess.Popen(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
version = str(popen.communicate()[0])
if popen.returncode == 0:
    f = open('pycf/__version__.py', 'w')
    f.write('\n__version__ = "%s"\n\n' % version.rstrip())
    f.close()
``` | ```python
popen = subprocess.Popen(
    ['git', 'rev-parse', '--short', 'HEAD'],
    stdout=subprocess.PIPE,
    universal_newlines=True,
)
git_revision = popen.communicate()[0].strip()
if popen.returncode != 0 or not git_revision:
    git_revision = 'unknown'

with open('pycf/__version__.py', 'w') as f:
    f.write('\n__version__ = "%s"\n\n' % git_revision)

version = '0+%s' % git_revision
``` |

## `pycf/cfl.pyx`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```cython
from __future__ import division
cimport cfl, cython
``` | ```cython
from __future__ import division
from pycf cimport cfl
cimport cython
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```cython
cdef public object sl_cap
cpdef public list labels
cpdef public str label_key
``` | ```cython
cdef public object sl_cap
cdef public list labels
cdef public str label_key
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```cython
cdef public object t_cap
cpdef public str name
cpdef public str arith_name
cpdef public int n
``` | ```cython
cdef public object t_cap
cdef public str name
cdef public str arith_name
cdef public int n
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```cython
def __mul__(x, y):
    ...
    else:
        raise TypeError("Tensors can only be multiplied by scalar numbers")
``` | ```cython
def __mul__(x, y):
    ...
    else:
        raise TypeError("Tensors can only be multiplied by scalar numbers")

def __rmul__(self, other):
    if isinstance(other, Number):
        if self.name == None:
            selfname = self.arith_name
        else:
            selfname = self.name
        tmp_name = "{0:.2f}*{1}".format(other, selfname)
        d = (self, other)
        return Tensor(<char *>tmp_name, None, None, None, self.states, data_tuple=d)
    raise TypeError("Tensors can only be multiplied by scalar numbers")
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```cython
params[p] = np.complex(x[ri], x[ri+1])
``` | ```cython
params[p] = complex(x[ri], x[ri+1])
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```cython
cdef double (*obj_f_ptr)(size_t, double *, double *, void *)
cdef void (*nls_f_ptr)(double *, void *, double *)

obj_f_ptr = <double (*)(size_t, double *, double *, void *)>PyCapsule_GetPointer(
        fit_obj.obj_f_cap, "pycfl.MinObjF")
...
nls_f_ptr = <void (*)(double *, void *, double *)>PyCapsule_GetPointer(
        fit_obj.nls_f_cap, "pycfl.NlsObjF")
``` | ```cython
cdef double (*obj_f_ptr)(size_t, double *, double *, void *) noexcept
cdef void (*nls_f_ptr)(double *, void *, double *) noexcept

obj_f_ptr = <double (*)(size_t, double *, double *, void *) noexcept>PyCapsule_GetPointer(
        fit_obj.obj_f_cap, "pycfl.MinObjF")
...
nls_f_ptr = <void (*)(double *, void *, double *) noexcept>PyCapsule_GetPointer(
        fit_obj.nls_f_cap, "pycfl.NlsObjF")
``` |

## `pycf/import_sljm.py`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
tensors[t] = cfl.Tensor(t, np.ascontiguousarray(tensor_matrices[t].indptr),
        np.ascontiguousarray(tensor_matrices[t].indices), np.ascontiguousarray(tensor_matrices[t].data), sl)
``` | ```python
tensors[t] = cfl.Tensor(
        t,
        np.ascontiguousarray(tensor_matrices[t].indptr, dtype=np.intc),
        np.ascontiguousarray(tensor_matrices[t].indices, dtype=np.intc),
        np.ascontiguousarray(tensor_matrices[t].data),
        sl)
``` |

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
tensors['MAGY'] = np.complex(0, -1) * tensors['MAGX']
``` | ```python
tensors['MAGY'] = complex(0, -1) * tensors['MAGX']
``` |

## `pycf/matel.py`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
element = np.complex(-1)**(j1 - m1)*nj.wigner_3j(...)
...
f = lambda m1, m2: np.complex(0, 1)/np.sqrt(2) * (...)
...
matel = np.zeros([l, l], dtype = np.complex)
``` | ```python
element = complex(-1)**(j1 - m1)*nj.wigner_3j(...)
...
f = lambda m1, m2: complex(0, 1)/np.sqrt(2) * (...)
...
matel = np.zeros([l, l], dtype=complex)
``` |

## `pycf/pyemp.py`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
parsed_data = data[:, 0::2] + np.complex(0,1) * data[:, 1::2]
``` | ```python
parsed_data = data[:, 0::2] + complex(0, 1) * data[:, 1::2]
``` |

## `pycf/cfl_util.py`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
scov = fmt_scov[key].format(np.complex(np.sqrt(cov[ii,ii]), np.sqrt(cov[ii+1,ii+1])))
...
rp = np.zeros(len(p), dtype=np.complex)
``` | ```python
scov = fmt_scov[key].format(complex(np.sqrt(cov[ii,ii]), np.sqrt(cov[ii+1,ii+1])))
...
rp = np.zeros(len(p), dtype=complex)
``` |

## `examples/eryso/mcmc_analysis.py`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
s = np.std(x[:,param_ord.index(pi[0])]) + np.std(x[:,param_ord.index(pi[1])])*np.complex(0,1)
``` | ```python
s = np.std(x[:,param_ord.index(pi[0])]) + np.std(x[:,param_ord.index(pi[1])]) * complex(0, 1)
``` |

## `examples/eryso/mhfit_siman.py`

| Before (`devel`) | After (`mfr-upgrade-python`) |
| --- | --- |
| ```python
I = np.complex(0,1)
``` | ```python
I = complex(0, 1)
``` |

## Local non-repo file

### `/home/users/mfr24/calculations/f1/pycfinten_test/pycf/inten_mfr.py`

| Before | After |
| --- | --- |
| ```python
import numpy as np
from scipy.special import sph_harm
from pycf.njsymbols import wigner_3j
``` | ```python
import numpy as np
from pycf.njsymbols import wigner_3j
``` |
