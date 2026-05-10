# Plan: Output of fit information, parameters, and parameter sigmas across all types of fits. 

## Issue different methods with different output and options. 

There are a varity of fitting methods in pycf with slightly different syntax and format. Some of these differences are, of course, unavoidable, but there is room for some rationalization that would aid user workflow. 

### Existing methods

#### Energy level fitting. 

For energy levels, there are several pycf methods for fitting to calculated splittings (e_fit, mh_fit). 

There is also pyfit, which replicates the functionality of e_fit and mh_fit using scipy methods. However, it does not print a list of energy levels after the fit. The user has to do that. 

The spin-Hamiltonian methods are not under consideration here as the statistical issues are different. 

#### Intensity fitting

For intensities, there is fit_altp, patterned on e_fit and mh_fit. This uses scipy minimizers. 

### Statistics and sigma

#### Energy level fitting 

For energy level fits, there is a jacobian add-on that allows the estimation of parameter uncertainties (sigma). However, this is not integrated into the normal "res" output.  

Furthermore, currently only the fitted parameter values are printed. For users, it is important to know *all* of the Hamiltonian parameters. 

#### Intensity fitting

For intensity fits, sigmas are calculated, and all parameters are printed, with estimated uncertainties for the fitted parameters. 

## Proposal: 

1. The options for the energy-level and intensity fits should be as similar as possible. 

2. All fit res.summary printouts should *always* include a listing of *all* parameters, not just the fitted parameters. This would include the sigmas if calculated. 

3. Calculating parameter uncertainties should be routine, but could be suppressed if necessary with an option (calculate_sigma = True/False). 

4. There should also be an option to output the variance-covariance matrix. 

4. Sigmas, jacobian, etc.  should be stored numerically in "res" for easy processing. 

### Current workflow: 

Some of my code, so you can see how I have been doing things. 

#### mh_fit with sigmas

    print('doing crystal-field fit...') 
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=False)
    param = ["EAVG", "FTOT", "ZETA", "C20", "C40", "C44", "C60", "C64"]
    res = cfl.e_fit(param, h, exdata, cfl_min, suppress_input=True, max_levels=65)
    fitcoeff = res["coeff"]
    print(res["summary"])
    for p in fitcoeff:
        coeff[p] = fitcoeff[p]
    pycf_coeff = coeff.copy() # keep the parameters for later 
    h.set_coeff(coeff)

    # Calculate uncertainties
    # Note that we have to call efit again because the parameters in h have changed.
    # If we don't do this call, we get the jacobian for the starting parameters. 
    efit = cfl.EFit(param, h, exdata)  
    # Now call methods on the EFit object, not the result dict
    J = efit.fd_jacobian()  # Jacobian on efit
    cov, sigma, edata = efit.covariance()
    # Uncertainties are in sigma
    # print(f"Parameter uncertainties: {sigma}")
    print('Fitted parameters and uncertainties:')
    for i, (name, val, unc) in enumerate(zip(param, efit.x0, sigma)):
        print(f"{name}: {val:.6f} +- {unc:.6f}")


#### pyfit with sigmas

    print('doing least-squares fit with scipy.optimize.least_squares...') 
    from pycf.pyfit import PyFit
    # Create your fit objects as usual
    efit = cfl.EFit(param, h, exdata)  # or MHFit for multi-Hamiltonian 
    # Wrap with PyFit
    py = PyFit(efit)
    # Run Levenberg-Marquardt via scipy
    result = py.fit(method='lm', jac='pycf', xtol=1e-10, ftol=1e-10)
    # Extract results
    print(result.message)
    fitcoeff = result.x   # optimized parameters
    for i, val in enumerate(fitcoeff):
        coeff[param[i]] = val
    print('Parameters:')
    for label, val in coeff.items():
        print(f"  {label}: {val}")
    pycf_coeff = coeff.copy() # keep the parameters for later 
    h.set_coeff(coeff)
    print("chisquare:", py.chi2(result.x))  # chi-square
    #  Get uncertainties
    cov, sigma, edata = py.covariance(result.x)
    print('Fitted parameters and uncertainties:')
    for i, (name, val, unc) in enumerate(zip(param, result.x, sigma)):
        print(f"{name}: {val:.6f} +- {unc:.6f}")
    
    # Do a dry run because pyfit does not do a printout of energy levels: 
    cfl_min = cfl.CFLMin("nlopt_bobyqa", xtol=1e-6, dry_run=True) 
    param = ["EAVG", "FTOT", "ZETA", "C20", "C40", "C44", "C60", "C64"]
    res = cfl.e_fit(param, h, exdata, cfl_min, suppress_input=True, max_levels=65)
    fitcoeff = res["coeff"]
    print(res["summary"])
    
### printing all parameters 


print(f"\nAll parameters:")
for p, v in coeff.items():
    if v != 0:
        print(f"  {p:6s} = {v}")
print()


### Intensity fit

inten_set_expt_data(spec_list, int_data_fit_list)
print('Start the fit...')
int_param = ["A210", "A230", "A430","A450", "A454","A650","A654","A670","A674"] 

.....
elif optimizer == "Nelder-Mead":
    bounds = [(-50e-10, 50e-10)] * len(int_param)  # example bounds; adjust as needed
    result = fit_altp(
        int_param,
        spec_list,
        bounds=bounds,
        dry_run=False,
        method="Nelder-Mead",
        options={"maxiter": 2500, "xatol": 1e-10, "fatol": 1e-12, "disp": True},
    )

In this case I don't have to add anything. The method prints out the fit, parameters, sigmas, intensities. 

### scipy results

In cases where a scipy method is used, there should be an option for that result to go into the res.summary output. 

## Final thoughts

Most of what is required is wrappers to reduce the amount that users need to type and to reduced user confusion. 

As far as possible, code should be common. E.g. a function to print all parameters and uncertainties that could be called in several places. 

## Discussion: 

● I reviewed plan/sigma_output.md; the proposal is solid and consistent with current pain points. The most useful change
  is a shared wrapper/output contract so e_fit/mh_fit, pyfit, and fit_altp all return comparable res data and summary
  text.

  >> Yes, this is what I want. 

  I’d implement it as a small, compatible layer:

   1. Add common kwargs across fit entry points: calculate_sigma=True, include_covariance=False, include_jacobian=False,
   print_summary=True, max_levels=....

>> Yes, that sounds good. 

   2. Standardize res keys: coeff (fitted), all_coeff, sigma, covariance, jacobian, chi2, summary, optimizer_result.

>> Good, but we should be careful *not* to change any keys that are already in e_fit and mh_fit. 

   3. Extract one shared formatter for “all parameters + fitted flag + sigma” and reuse it in energy and intensity 
  summaries.

>> Yes

   4. Add a wrapper for PyFit that optionally emits the same energy-level summary block as e_fit (so no separate dry-run
   call is needed).

>> This should be the default. A user will expect pyfit to behave the same as e_fit and mh_fit.  

   5. Include a conditioning indicator (e.g., Jacobian rank/condition) when sigma is weak/unstable, to avoid misleading 
  0.0 uncertainties.

>> Yes. In fact, the jacobian method is not very reliable for complex problems, and we use siman to do more robust estimation. However, the jacobian method gives quick fedback. 

  Small note: line 19 says mf_fit_altp; in code it is fit_altp.

>> fixed. 

## Implementation

Implemented first batch of the wrapper/output unification in `pycf` and it’s ready for you to trial in `alpha_euky3f10.py`.

### What changed (behavior)

1. **`e_fit` / `mh_fit` now support:**
   - `calculate_sigma=True` (default)
   - `include_covariance=False` (default)
   - `include_jacobian=False` (default)

   And their `res` now includes (without removing old keys):
   - `res["all_coeff"]`
   - `res["sigma"]` (by-parameter dict)
   - `res["sigma_vector"]`
   - `res["covariance"]` (when enabled)
   - `res["jacobian"]` (when enabled)
   - `res["jacobian_diagnostics"]`

2. **Energy-fit summaries now include an explicit**
   **“All Hamiltonian parameters”** table (fitted + fixed, with sigma where available).

3. **`PyFit` now has `fit_res(...)`**
   that returns an `e_fit`-style result dict and summary (so you don’t need the extra dry-run summary workaround).

4. **`fit_altp` now also accepts**
   `calculate_sigma`, `include_covariance`, `include_jacobian`, and return diagnostics fields for conditioning.

---

### How to update your workflow script now

#### A) `e_fit` / `mh_fit` path (replace manual post-fit sigma block)

```python
res = cfl.e_fit(
    param, h, exdata, cfl_min,
    suppress_input=True,
    max_levels=65,
    calculate_sigma=True,
    include_covariance=False,   # True if you want matrix in res+summary
    include_jacobian=False,     # True if you want Jacobian in res
)

print(res["summary"])
fitcoeff = res["coeff"]         # fitted subset
allcoeff = res["all_coeff"]     # full coefficient dict
sigma = res["sigma"]            # dict: param -> sigma
jacdiag = res["jacobian_diagnostics"]
```

Same idea for `mh_fit(...)`.

---

#### B) `PyFit` path (new default-style wrapper)

```python
from pycf.pyfit import PyFit

efit = cfl.EFit(param, h, exdata)
py = PyFit(efit)

res = py.fit_res(
    method="lm",
    jac="pycf",
    xtol=1e-10,
    ftol=1e-10,
    max_levels=65,
    calculate_sigma=True,
    include_covariance=False,
    include_jacobian=False,
)

print(res["summary"])
fitcoeff = res["coeff"]
allcoeff = res["all_coeff"]
sigma = res["sigma"]
```

This replaces your old pattern:
- `py.fit(...)`
- manual covariance calls
- separate `e_fit(..., dry_run=True)` just to print energy summary

---

#### C) Intensity fit (`fit_altp`) with LM

```python
res = fit_altp(
    int_param,
    spec_list,
    dry_run=False,
    minimizer="least_squares",
    method="lm",
    xtol=1e-8,
    ftol=1e-8,
    calculate_sigma=True,
    include_covariance=False,
    include_jacobian=True,
)

print(res["summary"])
# extras:
# res["jacobian_diagnostics"], res["uncertainty_diagnostics"]
```

---

I have these changes locally in:
- `pycf/cfl.pyx`
- `pycf/cfl_util.py`
- `pycf/pyfit.py`
- `pycf/inten.py`
- `tests/unit/test_pyfit.py`

(Your `plan/sigma_output.md` remains uncommitted/unmodified by me.)

### Comments on new wrappers. 

#### Header for fitting process. 

The pycf routines have a nice header when they start doing a fit: 

    =============
    e_fit summary
    =============
    pycf details
    ============

    pycf revision: 0.1.0.dev0+4660ed6  built at 2026-05-10 09:36:15
    Build comment: Build via setup.py
    Calculation started at: 2026-05-10 09:55:46
    Calculation completed at: 2026-05-10 09:55:47

Please implement similar output for pyfit and fit_altp. 

#### Location of diagnostic output and parameters. 

The energy level fits put all the diagnostics *after* the list of energy levels. 
Currently, fit_altp seems to list them in both places. 
For compatibility, I suggest putting all the diagnostics *after*. 

However, I like how the intensity output *always* lists all the intensity parameters. 

Can we modify the energy level summary so that it always lists all the Hamiltonian parameters before listing the energy levels, but has the detailed diagnostics at the end.  

## Rethink of output and fit_altp

### Scope
1. fit_altp workflow and output
2. printing of intensity parameters with spectrum lists. 

### Issues: 
1. In almost all cases the user will want multiple spectrums. Having a distinction between single and multi is confusing. 
2. Printing the Altp parameters with each spectrum is too much repetition. 
3. Workflow will now be like energies. Optional dry run, then fit. Calling of individual methods is discouraged for most users.  

### Suggestion:

1. Only have one wrapper: fit_altp. If the user only enters one spectrum and one data set, turn them into lists. 
2. Modify tests and examples to reflect this. I can easily modify the files I am working on. 
3. altp_fit should print the parameters only once, then the list of spectra. 

Is this feasible? There may be other ways to think  about it. 
