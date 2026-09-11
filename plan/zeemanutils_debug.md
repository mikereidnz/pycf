# zeemanutils debugging. 
Using zeemanutils seems to leave MZ with a nonzero value when starting the fits. 

See this file: 
/home/users/mfr24/calculations/f11/ery2o3_c3i/pycf_02/log.20260902_214313.ery2o3_c3i_abinitio_04.py.out

which contains the input at the top. 

The relevant output lines are: 
    739 MX                                    0.00000      fixed                            n/a
    740 MY                                    0.00000      fixed                            n/a
    741 MZ                                    0.05000      fixed                            n/a

which should all be zero, but MZ is 0.05. 


