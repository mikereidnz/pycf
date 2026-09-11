# I have some issues with the results from calculations: 

## First Example: 
Folder: 
/home/users/mfr24/calculations/f6/eucaf2_c3voxygen/pycf-kieran-reproduce
File: 
eucaf2_hfs_calcs.py

Crashes with message: 
Calculation completed at: 2026-09-11 12:52:43

<sys>:0: UserWarning: Normal matrix is rank-deficient (rank 9 < 14); returning Moore-Penrose pseudo-inverse covariance.
Traceback (most recent call last):
  File "/home/users/mfr24/calculations/f6/eucaf2_c3voxygen/pycf-kieran-reproduce/eucaf2_hfs_calcs.py", line 395, in <module>
    np.savetxt("fitting_energy_summary" + string_name + ".csv", res['eigenvalue']['Hamiltonian0'], delimiter = ',')
                                                                ~~~^^^^^^^^^^^^^^
KeyError: 'eigenvalue'


## Second Example 
/home/users/mfr24/calculations/f6/euky3f10/pycf
delta_euky3f10_estimate_C22_inten.py

Crashes with message: 
================================================================================
Absorption - Brief format (compact tabular):
================================================================================
Traceback (most recent call last):
  File "/home/users/mfr24/calculations/f6/euky3f10/pycf/delta_euky3f10_estimate_C22_inten.py", line 233, in <module>
    print("\n" + gen_inten_summary(spec_abs, h, format="brief"))
                 ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: gen_inten_summary() got multiple values for argument 'format'
Command exited with non-zero status 1
1.92user 0.05system 0:00.60elapsed 326%CPU (0avgtext+0avgdata 131904maxresident)k
408inputs+0outputs (3major+20665minor)pagefaults 0swaps


Is there a subtle incompatibility with the summary generation? 