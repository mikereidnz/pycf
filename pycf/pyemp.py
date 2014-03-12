#!/usr/bin/env python
# Filename = pyemp.py

# pyemp is a python module to facilitate scripting of Michael F. Reid's F-shell
# empirical crystal field theory routines.

# Copyright (C) 2012-2013 Sebastian Horvath (sebastian.horvath@gmail.com)
# 
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import sys
import os
import re
from subprocess import Popen, PIPE
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import register_projection


#TODO:
#   - Add jmcalc and sljcalc GenericErun subclasses.  This should work well 
#     with the BaseEmp execution order checking.
#   - Add Ce example to documentation.  
#   - Profile SpectrumData and optimize with cython if necessary. 


class Spectrum(dict):
    r""" 
    Implements all the defining parameters of a Spectrum object.

    Parameters
    ----------
    name : string
        The name for log and output files.
    emproot : string
        The path to the linuxemp directory.
    states : string
        The .st_ input file name - can also be specified when instantiating a
        :class:`Cfit` object.
    tensors : string
        The .mi_ and .mm_ input file names - can also be specified when
        instantiating a :class:`Cfit`, :class:`Vtrans` or :class:`Inten` object.
    addtensor : string
        Newline separated ``addten`` statements following cfit input file
        syntax.
    addassign: string
        Newline separated ``addassign`` statements following cfit input file
        syntax.
    expparams : string, optional
        Specifies values for ``exptval``, ``delta`` and ``lsq`` with their usual
        cfit input file meaning.
    splitplot : dictionary, optional 
        If specified, :class:`Cfit` will generate and parse splitplot data.
        Required keys are ``energy``, ``var`` and ``range``; see the
        :func:`SpectrumAxes.splitplot` method for further details.
    spinh : dictionary, or list of dictionaries, optional
        If specified, :class:`Cfit` will generate and parse spin Hamiltonian
        data.  Required dictionary keys are

            - ``terms``, a list of strings with possible values of 'bgs', 'ias'
              and 'iqi', which enable, respectively, Zeeman interactions,
              magnetic dipole hyperfine interactions and electric quadrupole
              hyperfine interactions;
            - ``levels``, a tuple of integers specifying the lower and upper
              energy levels for which to generate the spin Hamiltonian.

        If the argument is a list of dictonaries then spin Hamiltonian data will
        be generated for each element. 

    edconstruct : string
        String using the vtrans EDCONSTRUCT input syntax to use transformed
        tensors to construct the A tensors. 
    levels : array
        Of the form::
            
            ['initstart', 'initend', 'finstart', 'finend', 'initname', 'finname']

        where 'initstart' to 'initend' spans the range of initial energy levels,
        etc.
    edipoletensor: string
        Newline separated electric dipole tensor assignments passed to the vtrans
        input file - can also be specified when instantiating a :class:`Vtrans`
        object.
    tvals : string
        The .vi_ and .vm_ input file names - can also be specified when
        instantiating a :class:`Vtrans` or :class:`Inten` object.
    matel : string
        The vtrans matrix element .mi_ and .mm_ input file names - can also be
        specified when instantiating a :class:`Vtrans` object.
    trans : string 
        The .ti_ and .tm_ input file names - can also be specified when
        instantiating an :class:`Inten` object.
    plotargs : dictionary
        Keys of ``polarization``, ``temp``, ``linewidth``, ``npoints``,
        ``xrange`` and ``linewidth``, with some arguments optional depending on
        which plotting method is used, see the :class:`SpectrumData` and
        :class:`SpectrumErun` docstrings for details.
    plt : string
        The SpectrumErun input data file, the existence of which is also checked
        by SpectrumData to ensure Inten was run successfully.

    Returns
    -------
    object : Spectrum

    Notes
    -----
    The following list details attributes of Spectrum objects that need to be
    called explicitly for certain applications.

    sh_terms : list
        Elements are dictionaries containing spin Hamiltonian data, set by
        :class:`Cfit` if the provided Spectrum object was instantiated with the
        ``spinh`` kwarg.  Keys correspond to the ``terms`` specified by
        ``spinh`` kwarg and values are the respective spin Hamiltonian terms.
        For the term ``bgs`` this is a list of three terms for a magnetic field
        along 'x', 'y' and 'z', respectively.
    sh_energies : np.ndarray
        Contains spin Hamiltonian energy level data read from expthelp files,
        intended for use with :func:`spinhamiltonian.sh_lsq_func`.
    sh_b_l : np.ndarray
        Contains spin Hamiltonian block and level numbers.
    line_energies : np.ndarray
        Set by :class:`SpectrumData` or :class:`SpectrumErun`; energy data for
        individual transitions in wavenumbers, useful for scripting plotting
        procedure.
    line_inten : np.ndarray
        Set by :class:`SpectrumData` or :class:`SpectrumErun`; intensity data
        for individual transitions, useful for scripting plotting procedure.
    curve_energies : np.ndarray
        Set by :class:`SpectrumData` or :class:`SpectrumErun`; energy data for
        lineshapes in wavenumbers, useful for scripting plotting procedure.
    curve_inten : np.ndarry
        Set by :class:`SpectrumData` or :class:`SpectrumErun`; intensity data
        for lineshapes, useful for scripting plotting procedure.

    """    
    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.emproot = kwargs['emproot']
        del kwargs['name']
        del kwargs['emproot']
        self.erun_obj = {}

        # Check provided spectrum kwargs are valid, then add keys and values to
        # self. All valid args are removed from the args list, and the remainder
        # is set to None. 
        args = ['states', 'tensors', 'matel', 'tvals', 'trans', 'plt',
                'addtensors', 'addassign', 'expparams', 'splitplot', 'levels',
                'edconstruct', 'edipoletensor', 'edipole', 'mdipole',
                'plotargs', 'spinh']
        for arg in kwargs:
            if arg not in args:
                raise ValueError("Invalid kwarg: {}".format(arg))
            else:
                self[arg] = kwargs[arg]
                args.remove(arg)

        for arg in args:
            self[arg] = None

    def print_log(self, mode='brief'):
        r"""
        Generate a log of the executed erun programs. 

        Parameters
        ----------
        mode : string
            Allowed values are 'brief' and 'full'.  'brief' will return the erun
            input parameters of all executed erun programs, whereas 'full' will
            return the log files of all executed erun programs.

        Returns
        -------
        log : string
            The log output.
        """
        log = "\nLog data for Spectrum object {}\n".format(self.name)
        log +="=============================" + "="*len(self.name) + "\n"
        log += "Generated on {}\n\n".format(datetime.now())

        if mode == 'brief':
            log += "pyemp Spectrum parameters:\n"
            log += "--------------------------\n"
            for key in self:
                if self[key] != None:
                    log +=  "{0}:\n {1}\n".format(key, self[key])
        elif mode == 'full':
            if self.erun_obj == {}:
                log += "No record of any erun processes.  Full log file "
                log += "printing is only supported for erun processes called "
                log += "via their Spectrum wrapper methods.\n"
            for key in self.erun_obj:
                if key == 'spectrum_data':
                    log += "spectrum_data input parameters:\n"
                    log += "-------------------------------\n"
                    log += str(self['plotargs'])
                else:
                    log += "Full {} log:\n".format(key)
                    log += "----------" + "-"*len(key) + "\n"
                    log += self.erun_obj[key].print_log()
                log += "\n\n\n"
        else:
            raise ValueError("Invalid mode.")

        return(log)

    def cfit(self, **kwargs):
        r"""
        Generate and return object of type :class:`Cfit`.  See the :class:`Cfit`
        documentation for further details.
        """
        self.erun_obj['cfit'] = Cfit(self, **kwargs)

        return self.erun_obj['cfit']

    def vtrans(self, **kwargs):
        r"""
        Generate and return object of type :class:`Vtrans`.  See the
        :class:`Vtrans` documentation for further details.
        """
        self.erun_obj['vtrans'] = Vtrans(self, **kwargs)

        return self.erun_obj['vtrans']

    def inten(self, **kwargs):
        r"""
        Generate and return object of type :class:`Inten`.  See the
        :class:`Inten` documentation for further details.
        """
        self.erun_obj['inten'] = Inten(self, **kwargs)

        return self.erun_obj['inten']

    def spectrum_data(self, **kwargs):
        r"""
        Generate and return object of type :class:`SpectrumData`.  See the
        :class:`SpectrumData` documentation for further details.
        """
        self.erun_obj['spectrum_data'] = SpectrumData(self, **kwargs)

        return self.erun_obj['spectrum_data']

    def spectrum_erun(self, **kwargs):
        r"""
        Generate and return object of type :class:`SpectrumErun`.  See the
        :class:`SpectrumErun` documentation for further details.
        """
        self.erun_obj['spectrum_erun'] = SpectrumErun(self, **kwargs)

        return self.erun_obj['spectrum_erun']


class BaseEmp(object):
    r""" 
    Base class for :class:`GenericErun` and :class:`SpectrumData`.  Ensures the
    first argument is an instance of :class:`Spectrum`, and whether all
    necessary input files have been provided. 
    """
    # Ensure the first argument is of type Spectrum, then determine the workdir
    # (either the 'name' attribute of the spectrum object, or the parent
    # directory of the file pointed to by the 'name' attribute).
    def __init__(self, spectrum, process=None, infiles=None, **kwargs):
        if not isinstance(spectrum, Spectrum):
            raise TypeError("Invalid argument when instantiating {} object; "
            "the first arugment must be of type Spectrum.".format(process))
        elif os.path.isdir(spectrum.name):
            self.workdir = spectrum.name
        else:
            self.name = spectrum.name
            self.workdir = os.path.dirname(os.path.abspath(spectrum.name))
        
        # Add any provided input files to the spectrum object and ensure the
        # spectrum object contains all required input files.
        for f in infiles:
            if f in kwargs:
                spectrum[f] = kwargs[f]
            elif spectrum[f] == None:
                raise ValueError("Missing input file {0} for process {1}.  "
                "This means the input file has not been specified (either when "
                "creating the spectrum object or the {1} object) nor generated "
                "by a different erun process.  Have you run all the necessary "
                "previous erun processes?".format(f, process))

class GenericErun(BaseEmp):
    r""" 
    Base class for :class:`Cfit`, :class:`Vtrans`, :class:`Inten` and
    :class:`SpectrumErun`.  Sets up input files and handles execution of erun
    programs, along with erun log files.
    """
 
    def __init__(self, spectrum, process=None, infiles=None, **kwargs):
        BaseEmp.__init__(self, spectrum, process, infiles, **kwargs)

        # Add data for use in GenericErun methods.
        self.process = process
        self.emproot = spectrum.emproot

        # Create input file 'name_process.dat' and add a header.
        self.input_file = open('{0}_{1}.dat'.format(self.name, process), 'w+')
        self.input_file.write('% Input file for {}\n'.format(process))
        self.input_file.close()

    def erun(self, spectrum = None, outfiles = None):
        r"""
        A wrapper for the erun script.  Any subclasses of GenericErun must have
        attributes ``self.name`` and ``self.process``.  Additionally, this
        method writes the logfile output to the file ``name_process_log.txt``
        and appends names of created data files to the :class:`Spectrum`
        instance.

        Parameters
        ----------
        spectrum : Spectrum
            An instance of the Spectrum class.
        outfiles : list 
            The names of files created by the specific erun program.
        """

        if os.path.isdir(self.emproot):
            proc = Popen(['$EMPSCRIPTS/gnu_erun.csh {0} {1}_{0}.dat '
                'nolog'.format(self.process, self.name)], env={'EMPSCRIPTS':
                self.emproot + '/scripts', 'EMPBIN': self.emproot + '/bin'},
                shell=True, stdout=PIPE, stderr=PIPE)

            # Decode and write stdout to file - this corresponds to Mike's log
            # files since we pass the 'nolog' argument to erun.  As Mike's
            # programs don't write to stderr, any stderr messages are thrown
            # out.
            stdout = proc.communicate()[0].decode('ascii') 
            f = open('{0}_{1}_log.txt'.format(self.name, self.process), 'w')
            f.write(stdout)
            f.close()
            warning = re.search(r'\sWARNING\s', stdout)
            if warning:
                raise EnvironmentError("Warning detected in {0} log; check "
                "{1}_{0}_log.txt for details.".format(self.process, self.name))

            # Add the key of any files created by this process to the spectrum
            # object, with value self.name.  This is used to tell other BaseEmp
            # subclasses what files are available.
            if not outfiles == None:
                for f in outfiles:
                    spectrum[f] = self.name
        else:
            raise ValueError('emproot is not a directory.')
        
        return

    def print_log(self):
        r"""
        Print the log file generated by by erun for a given GenericErun object.
        """
        f = open('{0}_{1}_log.txt'.format(self.name, self.process), 'r')
        return f.read()
        f.close()

    def add_input(self, data):
        r"""
        Append data to erun program input file 'name_process.dat'.
        """
        self.input_file = open('{0}_{1}.dat'.format(self.name, self.process),
                'a')
        self.input_file.write("\n".join([line.lstrip() for line in \
            data.split("\n")[1:-1]]))
        self.input_file.close()


class Cfit(GenericErun):
    r"""
    Generate a cfit.dat input file and execute the cfit program.

    Parameters
    ----------
    spectrum : Spectrum
        The object must have attributes ``name``, ``emproot``, and keys
        ``addtensors``, and ``addassign`` and optionally ``states`` and
        ``tensors``; see the Spectrum docstring for a more detailed description
        of these attributes.
    states : string, optional
        The .st_ input file name; must be provided if it was not specified when
        the Spectrum object was instantiated.
    tensors : string, optional
        The .mi_ and .mm_ input file name; must be provided if it was not
        specified when the Spectrum object was instantiated.

    Returns
    -------
    object : Cfit

    Notes
    -----
    Instantiating an object of this type automatically sets the ``tvals``
    keyword of the provided :class:`Spectrum` object.  Furthermore, the Cfit log
    file can be displayed using the print_log() method.
    """

    def __init__(self, spectrum, **kwargs):
        GenericErun.__init__(self, spectrum, 'icfit', ['states', 'tensors'],
                **kwargs)

        # Set any optional arguments that have not been provided to empty
        # strings.
        blank_args = ['addtensors', 'addassign', 'expparams']
        for arg in blank_args:
            if spectrum[arg] == None:
                spectrum[arg] = ''

        base_input = """
                    READSTATES {0}
                    READTENSORS {1} \n
                    *% Add tensors
                    {2} \n
                    *% Assign parameters
                    {3} \n
                    {4} \n
                    diag \n
                    savevect {5} \n
                    {5} evects \n
                    \n\n""".format(spectrum['states'], spectrum['tensors'],
                            spectrum['addtensors'], spectrum['addassign'],
                            spectrum['expparams'], spectrum.name)
        
        
        # Check whether splitplot input has been provided; if so, append
        # relevant commands to input.
        if spectrum['splitplot'] != None:
            data = spectrum['splitplot']
            self.splitplot = data
            splitplot_input = "splitplot {0}_split.dat {1} {2} {3} {4} {5} 5 "
            "lines nokey ps \n dummy splitplot title \n\n".format(spectrum.name,
            data['energy'][0], data['energy'][1], data['var'], *data['range'])
        else:
            splitplot_input = ''
        
        # If spinh input is provided, convert it, if necessary, to a list, then
        # generate input file content for each element in addition to the
        # expthelp input. 
        if spectrum['spinh'] != None:
            if not isinstance(spectrum['spinh'], list):
                spectrum['spinh'] = [spectrum['spinh']]
            spinh_input = """*% Energy levels
                          expthelp {0}_spinh.hlp
                          {0} energy levels \n
                          """.format(spectrum.name)
            n_sh = len(spectrum['spinh'])
            self.sh_terms = [None]*n_sh
            u = [None]*n_sh
            l = [None]*n_sh
            d_sh = [None]*n_sh
            for sh_i,sh_args in enumerate(spectrum['spinh']):
                # Get spin Hamiltonian parameters and generate cfit input
                # strings.  We add a sh_terms list to the cfit object, which
                # lists all enabled terms in the canonical order.
                if any(t not in ['bgs','ias','iqi'] for t in sh_args['terms']):
                    raise ValueError("Invalid entry in terms list; valid "
                            "entries are 'bgs', 'ias' and 'iqi'.")
    
                # Zeeman term.
                if 'bgs' in sh_args['terms']:
                    bgs = "magz magx magy"
                    self.sh_terms[sh_i] = ['bgsz', 'bgsx', 'bgsy']
                else:
                    bgs = "magz"
                    self.sh_terms[sh_i] = ['bgsz']
                    # Turn on small mag to determine electronic spin label. 
                    spinh_input +="""*% Small magz to order S label.
                                  {0} 
                                  {1}
                                  {2}
                                  diag""".format('magx 0','magy 0','magz 0.001')
                if 'ias' in sh_args['terms']:
                    ias = "al"
                    self.sh_terms[sh_i] += ['ias']
                else:
                    ias = ""
                if 'iqi' in sh_args['terms']:
                    iqi = "eqhyp"
                    self.sh_terms[sh_i] += ['iqi']
                else:
                    iqi = ""
                # Get lower and upper levels.
                try:
                    (l[sh_i], u[sh_i]) = sh_args['levels']
                except KeyError:
                    raise ValueError("The spinh dictionary of {} is missing the"
                            " levels tupple".format(self.name))
                # Spin Hamiltonian dimension.
                d_sh[sh_i] = u[sh_i] - l[sh_i] + 1
                                 
                spinh_input += """*% Spin Hamiltonian input
                               spinh {0}_spinh-{1}.out {2} {3} {4} {5} {6}
                               Spin Hamiltonian for {0} \n
                               """.format(spectrum.name, sh_i, l[sh_i], u[sh_i],
                                       bgs, ias, iqi)
        else:
            spinh_input = ""
        
        # Execute cfit. 
        GenericErun.add_input(self, base_input + splitplot_input + spinh_input +
                'finish \n\n')
        GenericErun.erun(self, spectrum, ['tvals'])
        
        # Load splitplot and spinhamiltonian data. 
        if spectrum['splitplot'] != None:
            spectrum.splitplotdata = np.loadtxt('{0}_split.dat'.format(
                spectrum.name), skiprows = 4)
        
        # Parse the input for each spinh list entry.
        if spectrum['spinh'] != None:
            # Match energy level data from expthelp file; regex depends on
            # whether the nuclear spin label is present.
            if 'assign al' in spectrum['addassign'] or \
                    'assign eqhyp' in spectrum['addassign']:
                r = r'\s+(\d+)\s+(\d+)\s+[\*\d\.]+\s+\d+\s+([\d\.]+)[^[]+\[[^,]+,\s+([\d-]+)'
                f = open('{0}_spinh.hlp'.format(spectrum.name), 'r')
                match = re.findall(r, f.read())
                f.close()
                energies = np.array([m[2] for m in match], dtype=float)
                b_l = np.array([m[0:2] for m in match], dtype=int)
                Iz = np.array([m[3] for m in match], dtype = int)

            else:
                r = r'\s+(\d+)\s+(\d+)\s+[\*\d\.]+\s+\d+\s+([\d\.]+)'
                f = open('{0}_spinh.hlp'.format(spectrum.name), 'r')
                match = re.findall(r, f.read())
                f.close()
                energies = np.array([m[2] for m in match], dtype=float)
                b_l = np.array([m[0:2] for m in match], dtype=int)
                Iz = 0
            
            spectrum.sh_energies = energies
            spectrum.sh_b_l = b_l
            spectrum.sh_terms = []

            for sh_i in range(n_sh):
                # Generate and match regex for the given spin Hamiltonian
                # dimension.
                r = r'\s+([\d.e+-]+)\s+([\d.e+-]+)' * d_sh[sh_i]
                f = open('{0}_spinh-{1}.out'.format(spectrum.name, sh_i), 'r')
                data = np.array(re.findall(r, f.read()), dtype = float)
                f.close()
                
                # Calculate list that orders energy levels delimited by l and u
                # from largest to smallest I_z. 
                if Iz != 0:
                    Iz_sort = np.argsort(Iz[l[sh_i]-1:u[sh_i]])[::-1]
                else:
                    # No I_z label, so create identity sorting list.
                    Iz_sort = np.arange(u[sh_i]-l[sh_i] + 1)

                # Data is split into even and odd columns (real and imag), and
                # assigned to complex valued parsed_data.  parsed_data is then
                # split into the different spin Hamiltonian terms and sorted by
                # I_z.
                sh_terms = {}
                B = {}
                parsed_data = data[:, 0::2] + np.complex(0,1) * data[:, 1::2]
                for i,t in enumerate(self.sh_terms[sh_i]):
                    t_i = i * d_sh[sh_i]
                    term_data = parsed_data[t_i:t_i + d_sh[sh_i], 0:d_sh[sh_i]]
                    if t in ['bgsz', 'bgsx', 'bgsy']:
                        B[t] = term_data[Iz_sort][:,Iz_sort]
                    else:
                        sh_terms[t] = term_data[Iz_sort][:,Iz_sort]

                # Sort energy level blocks of equivalent Iz w.r.t. Sz, from
                # largest to smallest.  We step through magz data in 2*Sz+1 by
                # 2*Sz+1 blocks and sort by diagonal values.
                d_Iz = np.max(Iz) + 1
                d_Sz = (u[sh_i]-l[sh_i]+1)/d_Iz
                Sz_sort = np.array([], dtype = int)
                for n in np.arange(0, d_Sz*d_Iz, d_Sz):
                    Sz_sort = np.append(Sz_sort,
                            np.argsort(np.diag(B['bgsz'][n:n+d_Sz,
                            n:n+d_Sz]))[::-1]+n)
                for t in self.sh_terms[sh_i]:
                    if t in ['bgsz', 'bgsx', 'bgsy']:
                        B[t] = B[t][Sz_sort][:, Sz_sort]
                    else:
                        sh_terms[t] = sh_terms[t][Sz_sort][:, Sz_sort]
                    
                if len(B) == 3:
                    # Return list in order x, y, z. 
                    sh_terms['bgs'] = [B['bgsx'], B['bgsy'], B['bgsz']]
                
                spectrum.sh_terms += [sh_terms]


class Vtrans(GenericErun):
    r"""
    Generate a vtrans.dat input file and execute the vtrans program.

    Parameters
    ----------
    spectrum : Spectrum
        The object must have attributes ``name``, ``emproot``, and keys
        ``edconstruct``, and ``levels`` and optionally for ``states``, ``tvals``
        and ``matel``; see the Spectrum docstring for a more detailed
        description of these attributes.
    states : string, optional
        The .st_ input file name; must be provided if it was not specified when
        the Spectrum object was instantiated, or for a previous :func:`cfit`
        call.
    tvals : string, optional
        The .vi_ and .vm_ input file name; must be provided if it was not
        specified when the Spectrum object was instantiated.
    matel : string, optional
        The vtrans matrix element .mi_ and .mm_ input file name; must be
        provided if it was not specified when the Spectrum object was
        instantiated.

    Returns
    -------
    object : Vtrans

    Notes 
    -----
    Instantiating an object of this type automatically sets the ``trans``
    keyword of the provided :class:`Spectrum` object.  Furthermore, the Vtrans
    log file can be displayed using the print_log() method. 
    """
    def __init__(self, spectrum, **kwargs):
        GenericErun.__init__(self, spectrum, 'vtrans', ['states', 'tvals',
            'matel'], **kwargs)

        GenericErun.add_input(self, """
                              READSTATES {0} \n
                              READVECTS {1} \n
                              OUTFILE {2} \n
                              TRANSFORM {5} {6} {7} {8} {3} \n
                              {4} \n
                              finish \n\n""".format(
                                  spectrum['states'], spectrum['tvals'],
                                  spectrum.name, spectrum['matel'],
                                  spectrum['edconstruct'],
                                  *spectrum['levels'][0:4]))

        GenericErun.erun(self, spectrum, ['trans'])


class Inten(GenericErun):
    r"""
    Generate an inten.dat input file and execute the inten program. 
    
    Parameters
    ----------
    spectrum : Spectrum
        The object must have attributes ``name``, ``emproot``, and keys
        ``edipoletensor``, ``edipole``, ``mdipole``, and ``levels`` and
        optionally for ``states``, ``tvals`` and ``trans``; see the Spectrum
        docstring for a more detailed description of these attributes.
    states : string, optional
        The .st_ input file name; must be provided if it was not specified
        when the Spectrum object was instantiated, or for a previous
        :func:`cfit` of :func:`vtrans` call. 
    tvals : string, optional
        The .vi_ and .vm_ input file name; must be provided if it was not
        specified when the Spectrum object was instantiated, or for a
        previous :func:`vtrans` call.
    trans : string, optional
        The .ti_ and .tm_ input file name; must be provided if it was not
        specified when the Spectrum object was instantiated.

    Returns
    -------
    object : Inten
    
    Notes
    -----
    Instantiating an object of this type automatically sets the ``plt`` keyword
    of the provided :class:`Spectrum` object.  Furthermore, the Inten log file
    can be displayed using the print_log() method.

    """

    def __init__(self, spectrum, **kwargs):
        GenericErun.__init__(self, spectrum, 'inten', ['states', 'tvals',
            'trans'], **kwargs)
        
        GenericErun.add_input(self, """
                              READSTATES {0}
                              READTVALS {1}
                              ninputsets 1
                              READTENSOR {2}
                              SETUPMOM \n
                              addten edipole %
                              {3}
                              assign edipole {4}
                              assign magdipole {5} \n
                              calc \n
                              plotout {6}.plt
                              Transition intensity
                              {7} {8} {11}
                              {9} {10} {12}
                              END \n
                              finish \n\n""".format(spectrum['states'],
                                  spectrum['tvals'], spectrum['trans'],
                                  spectrum['edipoletensor'],
                                  spectrum['edipole'], spectrum['mdipole'],
                                  spectrum.name, *spectrum['levels']))

        GenericErun.erun(self, spectrum, ['plt'])


class SpectrumData(BaseEmp):
    r"""
    Natively generate spectrum data.  Intensity data is mined from the inten log
    file, which provides information such as the initial and final state of a
    transition.  
    
    Parameters
    ----------
    spectrum : Spectrum
        The object must have attributes ``plotargs`` - a dictionary that must
        have values for:

            - ``polarization`` a string, with possible values of ``isotropic``,
              ``axial``, ``sigma`` or ``pi``;
            - ``temp`` an int or float specifying the temperature;
            - ``linewidth`` an int or float specifying the Lorentzian linewidth;
            - ``npoints`` an optional argument specifying the number of points
              for the spectrum curve;

          and optionally ``plt`` - the filename of the data created by inten,
          which while unused is employed as an indicator of whether inten
          executed successfully.
    plt : string
        The filename of the data created by inten, which must be specified here
        if it is not a Spectrum attribute.
    
    Returns
    -------
    object : SpectrumData

    Notes
    -----
    Instantiating an object of this type adds a ``transitions`` attribute to the
    provided :class:`Spectrum` object, which returns a list of dictionaries,
    corresponding to distinct transitions, where each transition has keys:

    initialstate : string 
       The initial state label.
    finalstate : string 
       The final state label.
    energy : string
       The transition energy.
    isotropic : string
       The isotropic dipole strength.
    axial : string 
       The axial dipole strength.
    sigma : string 
       The sigma dipole strength.
    pi : string
       The pi dipole strength.

    """

    def __init__(self, spectrum, **kwargs):
        BaseEmp.__init__(self, spectrum, 'spectrum_data', ['plt'], **kwargs)

        def __boltzmannFact(e, t):
            r"""
            The Boltzmann factor for a given energy e and temperature t in units
            of cm^-1 K^-1.
            """
            if t < 0:
                ans = 0
            elif t == 0:
                ans = 1
            else:
                ans = np.exp(-e / (t * 0.6952))
            return(ans)
        
        def __lorentzian(x, x0, fwhm):
            """
            Lorentzian function.
            """
            gamma = fwhm / 2
            return(1 / np.pi * gamma / ((x - x0)**2 + gamma**2))
        
        # Parse data from log file generated by the inten program.
        r = re.compile(r'[^[]+(?P<initialstate>\[[\w\s,-]+[)>])[^[]+(?P<finalstate>\[[\w\s,-]+[)>])\)\s:\s+(?P<energy>[\d.e+-]+)[\n\s]+Dip\sStr\s+(?P<isotropic>[\d.e-]+)[\s*]+(?P<axial>[\d.e-]+)[\s*]+(?P<sigma>[\d.e-]+)[\s*]+(?P<pi>[\d.e-]+)')
        f = open('{0}/{1}_inten_log.txt'.format(self.workdir, self.name), 'r')
        spectrum.transitions = [m.groupdict() for m in r.finditer(f.read())]
        f.close()
       
        # Check if npoints is provided, otherwise use default. 
        if any('npoints' in arg for arg in spectrum['plotargs']):
            npoints = spectrum['plotargs']['npoints']
        else:
            npoints = 10000

        # Generate the spectrum.  For each transition the individual line
        # intensity is calculated; further a Lorentzian of corresponding height
        # is cumulatively added to yield the spectrum for the plotting region.
        transitions = spectrum.transitions
        xmin = float(transitions[0]['energy'])
        xmax = float(transitions[len(transitions) - 1]['energy'])
        curve_energies = np.linspace(xmin, xmax, npoints)
        curve_inten = np.zeros(npoints)
        line_energies = np.zeros(len(transitions))
        line_inten = np.zeros(len(transitions))
        temp =  spectrum['plotargs']['temp']
        polarization = spectrum['plotargs']['polarization']
        linewidth = float(spectrum['plotargs']['linewidth'])

        for i in range(len(transitions)):
            line_energies[i] = float(transitions[i]['energy'])
           
            # Calculate the individual line intensities.
            line_inten[i] = __boltzmannFact(line_energies[i] - line_energies[0],
                    temp) * float(transitions[i][polarization])
            # Calculate the cumulative curve intensity.  
            curve_inten += line_inten[i] * __lorentzian(curve_energies,
                    line_energies[i], linewidth)
        
        # Assign calculated values to the spectrum object.
        spectrum.line_energies = line_energies
        spectrum.line_inten = line_inten
        spectrum.curve_energies = curve_energies
        spectrum.curve_inten = curve_inten

        # Set haslabels flag for spectrumplot.
        spectrum.haslabels = True 


class SpectrumErun(GenericErun):
    r"""
    Facilitates the loading of c spectrum output files, or the execution of the
    c spectrum program and the subsequent loading of output files.

    Parameters
    ----------
    spectrum : Spectrum
        The object must have attributes ``plotargs``, a dictionary with values:

            - ``polarization`` a string, with possible values of ``isotropic``,
              ``axial``, ``sigma`` or ``pi``;
            - ``temp`` an int or float specifying the temperature, required if
              ``action = exec`` (see below);
            - ``linewidth`` an int or float specifying the Lorentzian linewidth,
              required if ``action = exec`` (see below);
            - ``xrange`` a list of the form ``[min, max]`` where ``min`` and
              ``max`` specify the lower and upper bound of the spectrum curve,
              respectively, required if ``action = exec`` (see below); 

        and optionally ``plt`` - the filename of the data created by inten.
    plt : string
        The filename of the data created by inten, which must be specified here
        if it is not a Spectrum attribute.
    action : string
        Kwarg, with allowed values of ``load`` (default) and ``exec``, to change
        between loading existing ``lines.gp`` and ``curves.gp_`` files or first
        executing the c spectrum program and then loading the resulting files.

    Returns
    -------
    object : SpectrumErun

    Notes
    -----
    The SpectrumErun log file can be displayed using the print_log() method.
  
    """
    def __init__(self, spectrum, **kwargs):
        GenericErun.__init__(self, spectrum, 'spectrum', ['plt'], **kwargs)

        def __load_data(self):
            r"""
            Load energy and intensity data generated by the spectrum program
            from the lines.gp_ and curves.gp_ files.
            """

            linedata = np.loadtxt('{}/lines.gp_'.format(self.workdir),
                    skiprows=3)
            curvedata = np.loadtxt('{}/curves.gp_'.format(self.workdir),
                    skiprows=3)
            
            # Fetch line/curve energies and intensities for the selected
            # polarization. 
            spectrum.line_energies = linedata[:, 0]
            spectrum.curve_energies = curvedata[:, 0]

            headers = ['isotropic', 'axial', 'sigma', 'pi']
            try:
                i = headers.index(spectrum['plotargs']['polarization'])
                # The first two columns are energy and wavelength.
                spectrum.line_inten = linedata[:, i + 2]
                spectrum.curve_inten = curvedata[:, i + 2]
            except ValueError:
                raise ValueError("The value of polarization in plotargs must "
                "be an element of 'isotropic', 'axial', 'sigma', or 'pi'.")
        
        # Check action keyword argument, then load or run spectrum prior to
        # loading.
        if not any('action' in s for s in kwargs) or kwargs['action'] == 'load':
            __load_data(self)
        elif kwargs['action'] == 'exec':
            input_data = """
                        getdata {0} 1 
                        polon {1}
                        temp {2} 
                        {3} 
                        linewidth {4} 
                        xrange {5} {6} \n
                        %plot 
                        printer postscript 
                        hardcopy {0}_spectrum.ps \n\n""".format(spectrum['plt'],
                                spectrum['plotargs']['polarization'],
                                spectrum['plotargs']['temp'],
                                spectrum['plotargs']['lines'],
                                spectrum['plotargs']['linewidth'],
                                *(spectrum['plotargs']['xrange']))
            GenericErun.add_input(self, input_data)
            GenericErun.erun(self)
            __load_data(self)

        else:
            raise KeyError("The erun keyword argument must either be left "
                    "blank, or have a value of 'exec' or 'load'.")

        # Set haslabels flag for spectrumplot.
        spectrum.haslabels = False


class SpectrumAxes(plt.Axes):
    r"""
    The SpectrumAxes matplotlib projection; when instantiating an axis object,
    add the ``projection = 'spectrum'`` kwarg, then run the spectrumplot method
    of the returned axis instance which takes as an argument an object of type
    Spectrum.

    To create an axis object one calls the figure constructor, which in turn
    instantiates the axis object.  As a consequence one does not directly
    instantiate the axis object so that the easiest workaround for adding a
    plotting method to the axis class is by using the projection option of the
    figure class. 

    Notes
    -----
    In order for the figure class to be aware of the spectrum projection one
    must run ``register_projection(SpectrumAxes)`` which requires the
    ``matplotlib.projections.register_projection`` module.  Importing all of the
    pycf module automatically handles this.
    """

    name = 'spectrum'

    def spectrumplot(self, spectrum, *args, **kwargs):
        r"""
        Create a spectrum plot from a :class:`Spectrum` object. 
        
        Parameters
        ----------
        spectrum : Spectrum
            The object should have the following spectrum plot specific
            keywords:

                - ``transitionlabels`` enables or disables explicit labels for
                  all plotted transitions - allowed values are ``False``
                  (default) or ``True``;
                - ``labelsize`` the fontsize for transition labels;
                - ``intencutoff`` the minimum intensity for transitions to be
                  plotted, primarily useful for suppressing uninteresting
                  transition labels;
                - ``energylabels`` if transition labels are enabled, then this
                  option appends the transition energy - allowed values are
                  ``False`` (default) or ``True``.

        *args, optional
            Additional arguments are passed passed to the vlines and plot
            routines have their usual functions.
        **kwargs, optional
            Additional keyword arguments are passed passed to the vlines and
            plot routines have their usual functions.

        """

        def __formatlabel(transition, energylabels):
            r"""
            Create a string of the form "initial state -> final state", and
            optionally append the transition energy.
            """
            tlabel = transition['initialstate'] +"->"+ transition['finalstate']
            
            # Check whether we're appending the energy.
            if energylabels:
                elabel = ", Energy: " + transition['energy']
            else:
                elabel = ""

            return(tlabel + elabel + "\n")
        
        def __labels(self, spectrum, lineindex, energylabels):
            r"""
            Label all transitions enumerated by the 'line_energies' attribute of
            the Spectrum object.
            """
            
            # Extract relevant transitions from the transitions list, and
            # corresponding energies from the energies array. 
            transitions = [spectrum.transitions[i] for i in lineindex[0]]
            line_energies = spectrum.line_energies[lineindex]
            line_inten = spectrum.line_inten[lineindex]
            
            n = len(transitions) - 1 
            label = __formatlabel(transitions[0], energylabels)
    
            for i in range(n):
                if abs(line_energies[i] - line_energies[i + 1]) <= 10**(-8):
                    label += __formatlabel(transitions[i + 1], energylabels)
                    continue
                else:
                    self.text(line_energies[i], line_inten[i], label,
                            rotation=90, ha='left', va='bottom', fontsize =
                            labelsize)
                    label = __formatlabel(transitions[i + 1], energylabels)
    
            # Explicitly plot last label, since the loop does not plot the
            # boundary case.
            self.text(line_energies[n], line_inten[n], label, rotation=90,
                    ha='left', va='bottom', fontsize = labelsize)
        
        # Determine plotting index given an intensity cutoff. 
        if 'intencutoff' in kwargs:
            lineindex = np.where(spectrum.line_inten>=kwargs['intencutoff'])
            del kwargs['intencutoff']
        else:
            lineindex = np.where(spectrum.line_inten > 10**(-10))
        
        # Check for label fontsize keyword argument.
        if 'labelsize' in kwargs:
            labelsize = kwargs['labelsize']
            del kwargs['labelsize']
        else:
            labelsize = 10

        # Check whether energy labels are enabled.
        if 'energylabels' in kwargs:
            if kwargs['energylabels'] == 'True':
                energylabels = True
            else:
                energylabels = False
            del kwargs['energylabels']
        else:
            energylabels = False

        # Generate vertical bar data from zero to transition intensity.
        ymax = spectrum.line_inten[lineindex]
        ymin = np.zeros(len(ymax))

        if 'transitionlabels' in kwargs:
            # Save value of transitionlabels and remove from kwargs, as kwargs
            # will be passed on to the plot routine. 
            transitionlabels = kwargs['transitionlabels']
            del kwargs['transitionlabels']

            if transitionlabels:
                # Check whether spectrum object supports labels.
                if not spectrum.haslabels:
                    raise ValueError("The plot data for this Spectrum object "
                    "was generated using SpectrumErun.  Transition labels are "
                    "only supported for plot data generated with SpectrumData.")
                self.vlines(spectrum.line_energies[lineindex], ymin, ymax,
                        *args, **kwargs)
                self.plot(spectrum.curve_energies, spectrum.curve_inten, *args,
                        **kwargs)
                __labels(self, spectrum, lineindex, energylabels)
            elif not transitionlabels:
                self.vlines(spectrum.line_energies[lineindex], ymin, ymax,
                        *args, **kwargs)
                self.plot(spectrum.curve_energies, spectrum.curve_inten, *args,
                        **kwargs)
            else:
                raise ValueError("Invalid option for kwarg 'transitionlabels'; "
                "allowed values are either True or False.")
        else:
            self.vlines(spectrum.line_energies[lineindex], ymin, ymax, *args,
                    **kwargs)
            self.plot(spectrum.curve_energies, spectrum.curve_inten, *args,
                    **kwargs)

    def splitplot(self, spectrum, *args, **kwargs):
        r"""
        Create a splitplot from a :class:`Spectrum` object.
        
        Parameters
        ----------
        spectrum : Spectrum
            The object must have the ``splitplotdata`` attribute.  This is
            achieved by adding the ``splitplot`` keyword to the
            :class:`Spectrum` object prior to instantiating the :class:`Cfit`
            object; ``splitplot`` is a dictionary with keys:

                - ``energy`` a list of the form ``[min, max]``, where ``min``
                  and ``max`` are integer values of the energy levels to be
                  plotted;
                - ``var`` a string of the variable name to be a varied;
                - ``range`` a list of the form ``[min, max]`` where ``min`` and
                  ``max`` are floating point numbers indicating the range over
                  which ``var`` wil be varied.

        *args, optional
            Additional arguments are passed to the vlines and plot routines have
            their usual functions.
        **kwargs, optional
            Additional keyword arguments are passed to the vlines and plot
            routines have their usual functions.
        """

        if spectrum['splitplot'] == None:
            raise ValueError("The provided Spectrum instance does not have a "
            "splitplotdata attribute. The splitplotdata attribute is created "
            "by Cfit for Spectrum objects with a splitplot kwarg.")
        else:
            data = spectrum.splitplotdata
            x = np.linspace(spectrum['splitplot']['range'][0],
                    spectrum['splitplot']['range'][1], len(data)) 

            for i in range(data.shape[1]):
                self.plot(x, data[:, i], *args, **kwargs)  

register_projection(SpectrumAxes)


