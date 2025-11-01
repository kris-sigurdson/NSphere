.. _command_line_usage:

Command-Line Usage
==================

This section details the command-line options for the executable components
of the NSphere simulation and analysis suite. All executables should be run
from the main project directory.

.. contents:: Table of Contents
   :local:
   :depth: 1

N-Spherical Shell Simulation (`nsphere`)
---------------------------------------

This is the core C program that runs the N-spherical shell simulation.

.. program:: nsphere

**Usage:**

.. code-block:: bash

   ./nsphere [options]

**Options:**

Basic Control & Execution
~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --help

   Show the usage message and exit.

.. option:: --log

   Enable writing detailed logs to ``log/nsphere.log``. Useful for debugging
   and tracking simulation progress in detail.
   [Default: Off]

.. option:: --tag <string>

   Append a custom string tag to the automatically generated suffix for most
   output filenames. Useful for distinguishing different runs with the same
   core parameters.
   [Default: None]

.. option:: --save <subarg> [subarg...]

   Controls which major data products are saved. Can specify multiple sub-arguments;
   if conflicting levels are given, the one enabling the most output takes effect.
   [Default: all]

   raw-data
     : Saves only basic particle output files (`particles.dat`, `particlesfinal.dat`).
   psi-snaps
     : In addition to `raw-data`, saves potential snapshots (`Psi_methodA_t*.dat`) and enables dynamic Psi calculation.
   full-snaps
     : In addition to `psi-snaps` output, saves full snapshot data (`Rank_Mass_Rad_VRad_*.dat`) and enables dynamic rank calculation.
   debug-energy
     : Enables energy tracking mode. In addition to `full-snaps` output, saves a detailed energy diagnostic file (`debug_energy_compare.dat`) for particle tracking.
   all
     : Saves all possible outputs, equivalent to enabling `debug-energy`.

.. option:: --restart [force]

   Enable restart mode. Looks for existing output data products (primarily
   ``all_particle_data<suffix>.dat`` and snapshot files) to determine where
   to resume processing. If complete data is found for the simulation phase,
   it may be skipped entirely.

   With 'force': regenerate ALL snapshots even if they exist.

   Incompatible with ``--readinit`` and ``--writeinit``.
   [Default: Off]

.. option:: --sim-restart [check]

   Restart incomplete simulation from last checkpoint.
   Checks for incomplete ``all_particle_data`` files and continues from
   the last complete snapshot. Creates backups and truncates files to
   ensure consistency.

   With 'check': only report status without restarting.

   [Default: Off]

.. option:: --restart-file <file>

   Specify explicit file path for restart operations (debugging).
   Overrides automatic file detection.
   [Default: Off]

.. option:: --sim-extend

   Extend a completed simulation to new Ntimes/tfinal values.
   Copies source file and continues evolution. Maintains constant
   physical timestep dt and write frequency dtwrite.

   Requires ``--extend-file`` to specify source file.
   [Default: Off]

.. option:: --extend-file <file>

   Source ``all_particle_data`` file to extend (used with ``--sim-extend``).
   File will be copied to new name based on target parameters.
   Filename must follow NSphere format: ``prefix_N_Ntimes_tfinal.dat``
   [Default: Off]

Simulation Setup
~~~~~~~~~~~~~~~

.. option:: --nparticles <int>

   Number of particles to simulate (after potential tidal stripping).
   [Default: 100000]

.. option:: --ntimesteps <int>

   Requested total number of simulation timesteps. Note: This value may be
   adjusted slightly upwards by the program to ensure alignment with the
   snapshot output schedule defined by ``--nout`` and ``--dtwrite``.
   [Default: 10000]

.. option:: --nout <int>

   Number of desired output snapshot intervals (results in ``nout + 1`` actual
   snapshots, including t=0).
   [Default: 100]

.. option:: --dtwrite <int>

   Number of simulation timesteps between each major data write/snapshot interval.
   [Default: 100]

.. option:: --snapshot-buffer <int>

   Number of snapshots to buffer in memory before writing to disk.
   Controls memory usage vs I/O frequency tradeoff.
   [Default: 100]

.. option:: --tfinal <int>

   Sets the total simulation duration as a multiple of the characteristic
   dynamical time (tdyn). Duration = ``<int>`` * tdyn.
   [Default: 5]

.. option:: --lvals-target <float>

   Select particles with angular momentum values closest to the specified target,
   rather than selecting the lowest L particles (default behavior).
   Useful for studying specific orbital families.
   [Default: Off - selects lowest L particles]

.. option:: --ftidal <float>

   Specifies the fraction of the outermost particles (by initial radius)
   to remove via tidal stripping before starting the simulation. Value must
   be between 0.0 (no stripping) and 1.0. The initial number of generated
   particles is increased to ensure ``--nparticles`` remain after stripping.
   [Default: 0.0]

Initial Conditions & I/O
~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --readinit <file>

   Read initial particle conditions (positions, velocities, etc.) directly
   from the specified binary ``<file>`` located inside the ``init/`` subdirectory,
   instead of generating them. The file must have been created previously
   using ``--writeinit``. Incompatible with ``--restart`` and ``--writeinit``.
   [Default: Off]

.. option:: --writeinit <file>

   Generate initial particle conditions and save them to the specified binary
   ``<file>`` inside the ``init/`` subdirectory. The simulation then
   proceeds normally. Incompatible with ``--restart`` and ``--readinit``.
   [Default: Off]

.. option:: --master-seed <int>

   Set master seed to derive all other seeds deterministically.
   When provided: IC seed = master_seed + 1, SIDM seed = master_seed + 2.
   Takes precedence over ``--load-seeds`` but not over direct seed options.
   [Default: time-based]

.. option:: --load-seeds

   Load IC and SIDM seeds from previous run's seed files.
   Looks for ``last_initial_seed_{suffix}.dat`` and ``last_sidm_seed_{suffix}.dat``.
   Lowest priority - only used if seeds not set by other options.
   [Default: Off]

.. option:: --init-cond-seed <int>

   Set seed specifically for initial condition generation.
   Highest priority - overrides both ``--master-seed`` and ``--load-seeds``.
   [Default: derived from master seed, loaded from file, or time-based]

Numerical Methods
~~~~~~~~~~~~~~~~

.. option:: --method <int>

   Selects the integration algorithm. [Default: 1]

   1
     : Selects the adaptive Leapfrog integrator combined with adaptive Levi-Civita regularization for close encounters (Default).
   2
     : Selects the full-step adaptive Leapfrog integrator combined with Levi-Civita regularization.
   3
     : Selects the full-step adaptive Leapfrog integrator without regularization.
   4
     : Selects the 4th-order symplectic Yoshida integrator.
   5
     : Selects the 3rd-order Adams-Bashforth predictor-corrector method.
   6
     : Selects the standard Leapfrog integrator with the velocity half-step formulation (Kick-Drift-Kick).
   7
     : Selects the standard Leapfrog integrator with the position half-step formulation (Drift-Kick-Drift).
   8
     : Selects the classic 4th-order Runge-Kutta integrator.
   9
     : Selects the simple forward Euler integration method.

.. option:: --methodtag

   Include the integration method name string (e.g., "adp.leap.adp.levi")
   in the output filename suffix, in addition to the parameters.
   [Default: Off]

.. option:: --sort <int>

   Selects the particle sorting algorithm. [Default: 1]

   1
     : Parallel Quadsort (Default).
   2
     : Sequential Quadsort.
   3
     : Parallel Insertion Sort.
   4
     : Sequential Insertion Sort.
   5
     : Parallel Radix Sort - High performance for large arrays.
   6
     : Adaptive sort - Benchmarks algorithms every 1000 sorts and switches to fastest.

Halo Profile Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. option:: --halo-mass <float>

   Total halo mass in solar masses (M☉) for the selected profile. This sets
   the overall mass normalization of the initial conditions.
   [Default: 1.15e9]

.. option:: --profile <type>

   Select the halo density profile type for initial conditions.

   - ``nfw``: NFW profile with exponential cutoff
   - ``cored``: Cored Plummer-like profile
   - ``hernquist``: Hernquist profile (supports constant-β and OM anisotropy)

   [Default: nfw]

.. option:: --scale-radius <float>

   Scale radius in kpc for the selected profile. For NFW profiles this is
   the traditional scale radius rs; for cored profiles this is the core radius;
   for Hernquist profiles this is the scale radius a.
   [Default: 23]

.. option:: --cutoff-factor <float>

   Sets the outer truncation radius as a multiple of the scale radius. The
   maximum radius rmax = cutoff-factor × scale-radius.
   [Default: 85.0]

.. option:: --falloff-factor <float>

   NFW-specific concentration parameter that controls the sharpness of the
   exponential cutoff at large radii. Only used for NFW profiles.
   [Default: 19.0]

Anisotropy Models
~~~~~~~~~~~~~~~~~

.. option:: --aniso-beta <float>

   Constant anisotropy parameter β for Hernquist profile.
   Controls velocity anisotropy: β = 0 (isotropic), β > 0 (radially biased),
   β < 0 (tangentially biased).

   Valid range: -1 ≤ β ≤ 0.5

   Only compatible with ``--profile hernquist``.
   Cannot be used with ``--aniso-factor`` or ``--aniso-betascale``.
   [Default: 0.0]

.. option:: --aniso-factor <float>

   Osipkov-Merritt anisotropy radius as multiple of scale radius.
   Sets r_a = factor × r_scale.

   Enables OM model with β(r) = r²/(r² + r_a²), which transitions from
   isotropic (β=0) at r=0 to radially biased (β→1) at large radii.

   Compatible with all profiles (NFW, Cored, Hernquist).
   Cannot be used with ``--aniso-betascale``.
   [Default: Off]

.. option:: --aniso-betascale <float>

   Alternative to ``--aniso-factor``: specify β at the scale radius directly.
   Calculates r_a/r_s = √(1/β_s - 1) automatically.

   Valid range: (0, 1)

   Cannot be used with ``--aniso-factor``.
   [Default: Off]

SIDM Physics
~~~~~~~~~~~~

.. option:: --sidm

   Enable Self-Interacting Dark Matter (SIDM) scattering physics.
   Activates particle-particle scattering with cross-section controlled by ``--sidm-kappa``.
   [Default: Off]

.. option:: --sidm-seed <int>

   Set seed specifically for SIDM scattering calculations.
   Highest priority - overrides both ``--master-seed`` and ``--load-seeds``.
   [Default: derived from master seed, loaded from file, or time-based]

.. option:: --sidm-mode <serial|parallel>

   Select SIDM execution mode.
   
   - ``serial``: Single-threaded SIDM calculations
   - ``parallel``: Multi-threaded SIDM using OpenMP (requires OpenMP support)
   
   [Default: parallel]

.. option:: --sidm-kappa <float>

   SIDM opacity parameter kappa in cm²/g. Controls the self-interaction
   cross-section strength.
   [Default: 50.0]

Last Parameters File (`lastparams.dat`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Upon successful startup, ``nsphere`` records the key parameters for the current run
into a suffixed file, typically named ``data/lastparams_<suffix>.dat`` (where
``<suffix>`` depends on the ``--tag``, ``--methodtag``, and core parameters like
``npts``, ``Ntimes``, ``tfinal``).

Crucially, it also creates or updates a standard file named ``data/lastparams.dat``
which acts as a symbolic link (on Unix-like systems) or a direct copy (on Windows)
of the most recently created suffixed parameter file.

The format of this file is a single line containing:
``<npts> <Ntimes> <tfinal_factor> [file_tag]``

This allows the ``nsphere_plot`` script (and related wrappers) to easily find
and use the parameters from the very last simulation run by default when no
``--suffix`` is specified.

Last Seed Files (`last_initial_seed.dat`, `last_sidm_seed.dat`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
In addition to recording simulation parameters, ``nsphere`` also saves the specific
seeds used for its random number generators to ensure reproducibility of stochastic
processes. These are stored in separate suffixed files, typically named:

* ``data/last_initial_seed_<suffix>.dat``
* ``data/last_sidm_seed_<suffix>.dat``

Where ``<suffix>`` is the same suffix used for the ``lastparams_<suffix>.dat`` file,
determined by the run's command-line options and core parameters.

Similar to the parameters file, standard non-suffixed versions are also created
as symbolic links (on Unix-like systems) or direct copies (on Windows):

* ``data/last_initial_seed.dat``
* ``data/last_sidm_seed.dat``

Each seed file contains a single line with one unsigned long integer representing
the seed value used for that specific random number generator component.
``last_initial_seed`` stores the seed for initial condition generation, while
``last_sidm_seed`` stores the base seed for SIDM scattering calculations.
For parallel SIDM processing, per-thread generators are seeded deterministically
from this base value.

These seed files are essential for achieving bit-for-bit reproducibility of
simulations. When ``nsphere`` is run with the same command-line parameters
(ensuring the same ``<suffix>``), the ``--load-seeds`` option instructs it to
read the matching suffixed files to initialize its random number generators, enabling exact
reproduction of previous runs. Direct seed specification via ``--master-seed``,
``--init-cond-seed``, or ``--sidm-seed`` takes precedence over loaded values.
The seed files always reflect the seeds *actually used* for each run, regardless
of their source.

Plotting & Animation (`nsphere_plot`)
------------------------------------

This Python script generates various plots and animations from the data
produced by the ``nsphere`` simulation.

.. program:: nsphere_plot

**Usage:**

.. code-block:: bash

   ./nsphere_plot [options]

**Options:**

.. option:: --suffix <SUFFIX>

   Specify the data file suffix (e.g., ``_tag_40000_1001_5``) used to find
   input data files generated by ``nsphere``.
   If omitted, the script attempts to read ``data/lastparams.dat`` to
   determine the suffix automatically from the parameters of the last ``nsphere``
   run. The ``lastparams.dat`` file is expected to contain a single line
   in the format: ``<npts> <Ntimes> <tfinal_factor> [file_tag]``.

.. option:: --start <N>

   Starting snapshot number for animations or time-series plots.
   [Default: 0]

.. option:: --end <N>

   Ending snapshot number for animations or time-series plots.
   A value of 0 means use all available snapshots up to the maximum found.
   [Default: 0]

.. option:: --step <N>

   Step size between snapshots used for animations.
   [Default: 1]

.. option:: --fps <N>

   Frames per second for output GIF animations.
   [Default: 10]

.. option:: --log

   Enable detailed logging to ``log/nsphere_plot.log``. Captures INFO and DEBUG
   messages in addition to warnings and errors.
   [Default: Off]

.. option:: --paced

   Run in "paced mode", adding artificial delays between major processing
   sections (e.g., between loading data and generating plots) for visual effect
   or to observe progress more slowly.
   [Default: Off]

.. option:: --help, -h

   Show the help message and exit.

**Visualization Control Flags:**

These flags control which types of plots or animations are generated. If none
of the ``--<type>`` flags (e.g., ``--animations``) are specified, the script
attempts to generate *all* available visualizations. If one or more ``--<type>``
flags *are* specified, *only* those types are generated.

*Run Only* Flags:

.. option:: --phase-space

   Generate only the initial phase space histogram and phase space animation.

.. option:: --phase-comparison

   Generate only the side-by-side initial vs final phase space plot and difference plot.

.. option:: --profile-plots

   Generate only the 1D profile plots (Density, Mass, Potential, f(E), etc.).

.. option:: --trajectory-plots

   Generate only particle trajectory plots and related diagnostics (Energy/Angular Momentum vs time).

.. option:: --2d-histograms

   Generate only the 2D phase space histograms (from particles*.dat and 2d_hist*.dat).

.. option:: --convergence-tests

   Generate only plots comparing results from different numerical parameters (Nintegration, Nspline).

.. option:: --animations

   Generate only the output GIF animations (Phase Space, Mass, Density, Psi).

.. option:: --energy-plots

   Generate only the Energy vs Time plots (including the debug comparison if data exists).

.. option:: --distributions

   Generate only the 1D comparison histograms of variable distributions (Radius, Velocity, etc.).

*Skip* Flags (used when running in default "all" mode):

.. option:: --no-phase-space

   Skip generating the initial phase space histogram and animation.

.. option:: --no-phase-comparison

   Skip generating the phase space comparison plots.

.. option:: --no-profile-plots

   Skip generating the 1D profile plots.

.. option:: --no-trajectory-plots

   Skip generating trajectory and related diagnostic plots.

.. option:: --no-histograms

   Skip generating all 2D histogram plots.

.. option:: --no-convergence-tests

   Skip generating the convergence test plots.

.. option:: --no-animations

   Skip generating all output GIF animations.

.. option:: --no-energy-plots

   Skip generating the Energy vs Time plots.

.. option:: --no-distributions

   Skip generating the 1D variable distribution comparison histograms.

Wrapper Scripts
~~~~~~~~~~~~~~~

These scripts provide convenient shortcuts to run specific parts of the main
``nsphere_plot`` script. They accept ``--suffix`` and ``--log`` arguments,
which are passed through to ``nsphere_plot``.

.. program:: nsphere_distributions

Generates 1D variable distribution comparison histograms (Initial vs Final).
Calls ``nsphere_plot --distributions``.

**Usage:**

.. code-block:: bash

   ./nsphere_distributions [--suffix SUFFIX] [--log]

**Options:**

.. option:: --suffix <SUFFIX>

   Data file suffix. Tries to read ``lastparams.dat`` if omitted.

.. option:: --log

   Enable detailed logging in ``nsphere_plot.py``.


.. program:: nsphere_2d_histograms

Generates 2D histogram plots (from particle files and nsphere.c output).
Calls ``nsphere_plot --2d-histograms``.

**Usage:**

.. code-block:: bash

   ./nsphere_2d_histograms [--suffix SUFFIX] [--log]

**Options:**

.. option:: --suffix <SUFFIX>

   Data file suffix. Tries to read ``lastparams.dat`` if omitted.

.. option:: --log

   Enable detailed logging in ``nsphere_plot.py``.


.. program:: nsphere_animations

Generates all standard animations (Phase Space, Mass, Density, Psi).
Calls ``nsphere_plot --animations``.

**Usage:**

.. code-block:: bash

   ./nsphere_animations [--suffix SUFFIX] [--start N] [--end N] [--step N] [--fps N] [--log]

**Options:**

.. option:: --suffix <SUFFIX>

   Data file suffix. Tries to read ``lastparams.dat`` if omitted.

.. option:: --start <N>

   Starting snapshot number. [Default: 0]

.. option:: --end <N>

   Ending snapshot number (0=auto). [Default: 0]

.. option:: --step <N>

   Snapshot step size. [Default: 1]

.. option:: --fps <N>

   Output animation frames per second. [Default: 10]

.. option:: --log

   Enable detailed logging in ``nsphere_plot.py``.