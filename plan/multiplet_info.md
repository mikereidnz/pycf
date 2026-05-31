# Plan: Output of multiple-specific fitting information in gen_e_summary (cfl_util.py)

This is a user-display enhancement that: 
1. Separates energy output into user-defined multiplets. 
2. Prints statistical information regarding the fit at the end of each multiplet on a separate line.  

A multiplet is a sequence of energy levels. 
User would input a list of (1-referenced) levels for the end (highest-energy) level in each multiplet. This would be associated with the Hamiltonian, and is useful to break up the output into chunks.  First multple starts at 1 and goes to the first number in the list. 

Within each multiplet we have: 
1.  Calculated energies (c_i)
2.  Experimental energies (e_j). Usually there are less e_j than c_i. 

Definitions: 
1. barycenter = average(c)
2. barycentre_shift = average(e - c) (over levels that have experimental data)
3. standard deviations: 
    a. sigma_total = sqrt( average(e-c)^2 )
    b. sigma_crystal_field = sqrt( average(e - c - barycentre_shift)^2 )
    >> fixed above. 

>> Weighting: This is a diagnostic, not part of the fitting statistics, so there is no need to use data weighting factors. 

Output on line after endmultiplet level: 
1. barycenter
2. if there are experimental levels in the multplet: 
    a. barycenter_shift
    b. sigma_total
    c. sigma_crystal_field

Please investigate and clarify.

Note: obviously, only absolute energy data is relevant, energy splitting is not. However, the energy-level Hamiltonian usually only uses absolute energies. 

code: 
1. Helper to store the endmultiplet list in the Hamiltonian. 
2. Modification to gen_e_summary() to implement the output. 

>> Clarifications: 
>> Last boundary can be less than n_levels. User only enters up to levels of interest. 
>> use Americal spelling: barycenter. 

## Copilot review comments

1. **Scope/fit with existing code**
   - This is a good fit for `gen_e_summary()` as a display-layer enhancement.
   - Keep multiplet stats as optional output gated by a Hamiltonian multiplet-boundary setting.

2. **Boundary validation rules (recommended)**
   - Treat user input as 1-based end-level indices.
   - Validate strictly increasing, unique, and all `>= 1`.
   - Allow final boundary `< n_levels` exactly as noted; only summarize requested range.
   - If a boundary exceeds available levels, raise a clear `ValueError`.

3. **Computation conventions**
   - Use unweighted means/standard deviations (as you clarified) for these diagnostic lines.
   - Compute residuals only on levels in the multiplet that have absolute experimental assignments.
   - Print only `barycenter` when no experimental levels exist in a multiplet block.

4. **Data-mode guard**
   - Restrict this feature to absolute-energy data display paths.
   - For non-absolute/splitting-style displays, skip multiplet diagnostics to avoid misleading output.

5. **Output consistency**
   - Use `barycenter` spelling consistently in labels, docstrings, and comments.
   - Keep the added diagnostic line compact and aligned with existing summary formatting.

### Answers to queries: 

I re-checked against `gen_e_summary` behavior, and the main **vital** items still to lock down are:

1. **`e_shift` interaction (important)**  
   `gen_e_summary` currently shifts `w` before residuals are computed. Your multiplet stats must explicitly follow the same convention (shifted display energies) or you’ll get inconsistent “Difference” vs multiplet sigmas.
>> Please explain this in more detail. 

2. **`max_levels` interaction**  
   Decide whether multiplet stats are computed on:
   - all levels in the multiplet, or
   - only displayed levels (`max_levels`).  
   I strongly recommend “displayed levels only” to match what the user sees.
>> only displayed levels. 

3. **Absolute-only with mixed datasets (`A` + `D`)**  
   The summary can contain both absolute and difference data. Multiplet stats should use only absolute assignments (`n_a` part) and ignore diff rows even when both are present.
>> yes, ignore diff rows. In practice, they are unlikely to be used for the energy-level Hamiltonian. 

4. **Boundary semantics when last boundary < available levels**  
   You clarified this, but implementation must enforce:
   - only summarize up to the last boundary,
   - leave later levels printed normally (or define they are omitted from multiplet stats).
>> Yes, later levels printed normally. 

5. **Validation and failure mode**  
   End-level list should fail fast on non-integer, non-increasing, duplicate, `<1`, or out-of-range values with clear errors (not silent clipping).
>> Yes, fail on incorrect user data. 

6. **Test coverage you should require**  
   Add tests for:
   - multiplet stats with no ex data in a block,
   - mixed `A`/`D` ExData,
   - `e_shift=True`,
   - `max_levels` truncation through a multiplet boundary,
   - invalid boundary lists.
>> Yes. 

If you lock these six, the plan looks robust and unlikely to surprise users.
