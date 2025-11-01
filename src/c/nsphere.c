/*
 * Copyright 2025 Kris Sigurdson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_hyperg.h>
#include <time.h>
#include <sys/time.h>  /* For gettimeofday */
#include <unistd.h>    /* For getpid */
#ifdef _OPENMP
#include <omp.h>
#else
// OpenMP function stubs when compiled without OpenMP
// Use __attribute__((unused)) to prevent unused function warnings
static int __attribute__((unused)) omp_get_max_threads(void) { return 1; }
static int __attribute__((unused)) omp_get_num_procs(void) { return 1; }
static int __attribute__((unused)) omp_get_thread_num(void) { return 0; }
static void __attribute__((unused)) omp_set_max_active_levels(int x) { (void)x; }
static void __attribute__((unused)) omp_set_num_threads(int x) { (void)x; }
static double __attribute__((unused)) omp_get_wtime(void) {
    // Use higher precision time function for non-OpenMP builds
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>     // For open() flags and file operations
#include <unistd.h> // For usleep function
#include "nsphere_sort.h" // Clean custom wrapper around quadsort
#include <ctype.h>
#include <fftw3.h>
#include <float.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/statvfs.h>
#endif

/**
 * @brief Three-dimensional vector structure for particle physics calculations.
 * @details Represents velocity and position vectors in 3D space for self-interacting
 *          dark matter simulations. Components are stored in Cartesian coordinates.
 */
typedef struct {
  double x; ///< x-component of the vector.
  double y; ///< y-component of the vector.
  double z; ///< z-component of the vector.
} threevector;


/**
 * @brief Parameters for the Hernquist distribution function integrand fEintegrand_hernquist.
 * @details This structure passes all necessary data, including pre-computed splines
 *          for \f$r(\Psi)\f$ and \f$M(r)\f$, physical constants, and profile-specific parameters,
 *          to the GSL integration routine for calculating \f$I(E)\f$ or \f$I(Q)\f$.
 */
typedef struct {
    double E_current_shell;         ///< Energy E (isotropic) or Q (OM) of the current shell for which I is being computed.
    gsl_spline *spline_r_of_Psi;    ///< Spline for \f$r(-\Psi)\f$: radius as function of negated potential (negation ensures monotonicity for GSL).
    gsl_interp_accel *accel_r_of_Psi; ///< Accelerator for the \f$r(-\Psi)\f$ spline.
    gsl_spline *spline_M_of_r;      ///< Spline for M(r), enclosed mass as a function of radius.
    gsl_interp_accel *accel_M_of_r;   ///< Accelerator for the M(r) spline.
    double const_G_universal;       ///< Universal gravitational constant G.
    double hernquist_a_scale;       ///< Scale radius \f$a\f$ of the Hernquist profile.
    double hernquist_normalization; ///< Density normalization constant for the Hernquist profile.
    double Psimin_global;           ///< Minimum potential value for physical range validation.
    double Psimax_global;           ///< Maximum potential value for physical range validation.
    int use_om;                     ///< Flag: 1 if using OM (E_current_shell represents Q), 0 if isotropic.
} fE_integrand_params_hernquist_t;

/**
 * @brief Parameters for OM \f$\mu\f$-integral in df_fixed_radius output.
 */
typedef struct {
    double v_sq;
    double Psi_rf_codes;
    double r_over_ra_sq;
    double Psimin;
    double Psimax;
    gsl_interp *fofQ_interp;
    double *Q_array;
    double *I_array;
    gsl_interp_accel *fofQ_accel;
} om_mu_integral_params;

/**
 * @brief Parameters for the NFW distribution function integrand fEintegrand_nfw.
 * @details This structure passes all necessary data, including pre-computed splines
 *          for \f$r(\Psi)\f$ and \f$M(r)\f$, physical constants, and profile-specific parameters,
 *          to the GSL integration routine for calculating \f$I(E)\f$.
 */
typedef struct {
    double E_current_shell;         ///< Energy E (isotropic) or Q (OM) of the current shell for which I is being computed.
    gsl_spline *spline_r_of_Psi;    ///< Spline for \f$r(-\Psi)\f$: radius as function of negated potential (negation ensures monotonicity for GSL).
    gsl_interp_accel *accel_r_of_Psi; ///< Accelerator for the \f$r(-\Psi)\f$ spline.
    gsl_spline *spline_M_of_r;      ///< Spline for M(r), enclosed mass as a function of radius.
    gsl_interp_accel *accel_M_of_r;   ///< Accelerator for the M(r) spline.
    double const_G_universal;       ///< Universal gravitational constant G.
    double profile_rc_const;        ///< Scale radius \f$r_c\f$ of the NFW profile.
    double profile_nt_norm_const;   ///< Density normalization constant (nt_nfw) for the NFW profile.
    double profile_falloff_C_const; ///< Falloff concentration parameter \f$C\f$ for power-law cutoff in NFW profile.
    double Psimin_global;           ///< Minimum potential value for physical range validation.
    double Psimax_global;           ///< Maximum potential value for physical range validation.
    int use_om;                     ///< Flag: 1 if using OM (E_current_shell represents Q), 0 if isotropic.
} fE_integrand_params_NFW_t;


// =============================================================================
// Windows‑compatibility shims
// =============================================================================
// Provide POSIX‑style helpers for MinGW/Clang:
//   • mkdir(path,mode)   → _mkdir(path)
//   • drand48 / srand48  → wrappers around ANSI rand
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
  #include <direct.h>
  #include <stdlib.h>

  /** Accept 1‑ or 2‑argument forms of mkdir on Windows. */
  #define mkdir(path, ...) _mkdir(path)

  #define NSPHERE_WINDOWS_SHIMS_DONE 1
#endif
/* ========================================================================= */

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================
/**
 * @brief Function pointer types for dynamic dispatch
 * @details These types enable runtime selection of profile-specific implementations
 *          without conditional branching in performance-critical loops.
 */
/// Function pointer for density derivative: \f$d\rho/dr\f$ (Cored/Hernquist profiles).
typedef double (*drhodr_func_t)(double r);
/// Function pointer for NFW density derivative: \f$d\rho/dr\f$ with profile parameters.
typedef double (*drhodr_nfw_func_t)(double r, double rc, double nt_nfw, double falloff_C);

// SIDM vector mathematics and cross-section function declarations
threevector make_threevector(double x, double y, double z);
double dotproduct(threevector X, threevector Y);
threevector crossproduct(threevector X, threevector Y);
double sigmatotal(double vrel, int npts, double halo_mass_for_calc, double rc_for_calc);

// Serial SIDM scattering integration function declaration
void perform_sidm_scattering_serial(double **particles, int npts, double dt, double current_time, gsl_rng *rng, long long *Nscatter_total_step, double halo_mass_for_sidm, double rc_for_sidm, int *current_scatter_counts);

// Forward declaration for the parallel SIDM scattering function (graph coloring algorithm)
void perform_sidm_scattering_parallel_graphcolor(double **particles, int npts, double dt, double current_time, gsl_rng **rng_per_thread_list, int num_threads_for_rng, long long *Nscatter_total_step, double halo_mass_for_sidm, double rc_for_sidm, int *current_scatter_counts);

// Forward declarations for trajectory buffer management functions
static void allocate_trajectory_buffers(int num_traj_particles, int nlowest, int buffer_size, int npts);
static void flush_trajectory_buffers(int items_to_write, int num_traj_particles, int nlowest);
static void cleanup_trajectory_buffers(int num_traj_particles, int nlowest);

// Forward declarations for double-precision buffer functions
static void allocate_double_precision_buffers(int npts);
static void add_snapshot_to_double_buffer(double **particles, double *phi_array, int *rank_array, int npts);
static void flush_double_buffer_to_disk(const char *tag);
static int load_from_double_buffer(const char *tag, int snapshot_index, int total_snapshots, double **particles, int npts, int *particle_ids);
static void load_particles_from_restart(const char *filename, int snapshot_index, double **particles, int npts, int block_size, int *inverse_map);


static char g_file_suffix[256] = ""; ///< Global file suffix string.
static gsl_rng *g_rng = NULL; ///< GSL Random Number Generator state.
static gsl_rng **g_rng_per_thread = NULL; ///< Array of GSL RNG states, one per OpenMP thread.
static int g_max_omp_threads_for_rng = 1; ///< Number of threads for which RNGs are allocated.

// =============================================================================
// PERSISTENT SORT BUFFER CONFIGURATION
// =============================================================================
// 
// The simulation exclusively uses a persistent global buffer (`g_sort_columns_buffer`)
// for particle data transposition during sorting operations. This strategy 
// minimizes memory allocation/deallocation overhead.

/** Global persistent buffer for particle data transposition during sorting. */
static double **g_sort_columns_buffer = NULL;
/** Stores the number of particles for which the persistent buffer is allocated. The buffer is reallocated if the number of simulation particles changes. */
static int g_sort_columns_buffer_npts = 0;

// =============================================================================
// PARALLEL SORT ALGORITHM CONFIGURATION
// =============================================================================
// Constants controlling the behavior of parallel sorting algorithms.
// These parameters tune the parallel sorting operations used when 
// OpenMP is available, affecting the partitioning of data across threads
// and the overlap required for correct merging of sorted sections.

/** 
 * Default number of sections when OpenMP is unavailable or reports few threads.
 * Provides a baseline level of partitioning even in limited thread environments.
 */
static const int PARALLEL_SORT_DEFAULT_SECTIONS = 8;

/**
 * Number of sort sections per OpenMP thread for workload distribution.
 * Multiplier used to determine total section count from available threads.
 */
static const int PARALLEL_SORT_SECTIONS_PER_THREAD = 2;

/**
 * Divisor for calculating proportional overlap between sort sections.
 * Overlap is calculated as chunk_size / OVERLAP_DIVISOR to scale with data size.
 */
static const int PARALLEL_SORT_OVERLAP_DIVISOR = 8;

/**
 * @brief Minimum required overlap between adjacent sort sections (in elements).
 * @details This ensures sufficient overlap for correct merging of sorted sections,
 *          particularly for sparse data distributions or when the calculated
 *          proportional overlap is small.
 */
static const int PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP = 32;


/**
 * Minimum average chunk size threshold for parallel sort operation.
 * If the calculated size of each chunk (total elements / number of sections)
 * falls below this threshold, the algorithm reverts to serial sorting to
 * avoid overhead from managing very small parallel tasks.
 */
static const int PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD = 128;

/**
 * @brief Applies the global suffix to a filename.
 * @details For .dat files, inserts suffix before the extension.
 *          For other files, appends suffix to the end of filename.
 *
 * @param base_filename [in] Original filename.
 * @param with_suffix [in] Flag indicating whether to apply the suffix (1=yes, 0=no).
 * @param buffer [out] Output buffer for the resulting filename.
 * @param bufsize [in] Size of the output buffer.
 */
void get_suffixed_filename(const char *base_filename, int with_suffix, char *buffer, size_t bufsize)
{
    if (!with_suffix || g_file_suffix[0] == '\0')
    {
        // No suffix to apply
        strncpy(buffer, base_filename, bufsize - 1);
        buffer[bufsize - 1] = '\0';
        return;
    }

    const char *ext = strrchr(base_filename, '.');
    if (ext && strcmp(ext, ".dat") == 0)
    {
        // For .dat files: insert suffix before extension
        size_t basename_len = ext - base_filename;
        if (basename_len + strlen(g_file_suffix) + strlen(ext) + 1 > bufsize)
        {
            // Buffer too small
            strncpy(buffer, base_filename, bufsize - 1);
            buffer[bufsize - 1] = '\0';
            return;
        }

        strncpy(buffer, base_filename, basename_len);
        buffer[basename_len] = '\0';
        strcat(buffer, g_file_suffix);
        strcat(buffer, ext);
    }
    else
    {
        // For non-dat files: append suffix to filename
        if (strlen(base_filename) + strlen(g_file_suffix) + 1 > bufsize)
        {
            // Buffer too small
            strncpy(buffer, base_filename, bufsize - 1);
            buffer[bufsize - 1] = '\0';
            return;
        }

        strcpy(buffer, base_filename);
        strcat(buffer, g_file_suffix);
    }
}

#ifndef MYBINIO_H
#define MYBINIO_H
#endif

// Forward declaration of global logging flag
extern int g_enable_logging;

/**
 * @brief Writes a formatted message to the log file with timestamp and severity level.
 *
 * @param level [in] Severity level (e.g., "INFO", "WARNING", "ERROR").
 * @param format [in] Printf-style format string.
 * @param ... [in] Variable arguments for the format string.
 *
 * @note Creates the "log" directory if it does not exist.
 * @warning Prints an error to stderr if the log file cannot be opened.
 *          Logging only occurs if the global `g_enable_logging` flag is set.
 * @see g_enable_logging
 */
void log_message(const char *level, const char *format, ...)
{
    // Only write to log file if logging is enabled
    if (g_enable_logging)
    {
        // Create log directory if it does not exist
        struct stat st = {0};
        if (stat("log", &st) == -1)
        {
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
            mkdir("log"); // Windows
#else
            mkdir("log", 0755); // Unix-like systems
#endif
        }

        // Always use a single log file regardless of file suffix
        const char *log_filename = "log/nsphere.log";

        FILE *logfile = fopen(log_filename, "a");
        if (logfile)
        {
            time_t now;
            time(&now);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

            // Include file suffix in log entries if available
            if (g_file_suffix[0] != '\0')
            {
                fprintf(logfile, "[%s] [%s] [%s] ", timestamp, level, g_file_suffix);
            }
            else
            {
                fprintf(logfile, "[%s] [%s] ", timestamp, level);
            }

            va_list args;
            va_start(args, format);
            vfprintf(logfile, format, args);
            va_end(args);

            fprintf(logfile, "\n");
            fclose(logfile);
        }
        else
        {
            // Print an error message to stderr if the log file cannot be opened
            fprintf(stderr, "Warning: Failed to open log file '%s'\n", log_filename);
        }
    }
}

/**
 * @brief Formats a byte count into a human-readable string with appropriate units.
 *
 * @param size_in_bytes [in] The size in bytes to format.
 * @param buffer [out] Output buffer for the formatted string.
 * @param buffer_size [in] Size of the output buffer.
 */
void format_file_size(long size_in_bytes, char *buffer, size_t buffer_size)
{
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = (double)size_in_bytes;

    // Find appropriate unit
    while (size >= 1024.0 && unit_index < 4)
    {
        size /= 1024.0;
        unit_index++;
    }

    // Format with appropriate precision based on size
    if (unit_index == 0)
    {
        // Bytes: no decimal places needed
        snprintf(buffer, buffer_size, "%ld %s", (long)size, units[unit_index]);
    }
    else if (size >= 10)
    {
        // Larger sizes: one decimal place
        snprintf(buffer, buffer_size, "%.1f %s", size, units[unit_index]);
    }
    else
    {
        // Small sizes: two decimal places
        snprintf(buffer, buffer_size, "%.2f %s", size, units[unit_index]);
    }
}

/**
 * @brief Creates a backup of a file with a .backup extension.
 * @details Creates a byte-for-byte copy of the source file, preserving the
 *          `all_particle_data` file state before truncation during restart
 *          operations.
 *
 * @param source_file Path to the source file to back up.
 * @return Returns 0 on success, -1 on failure.
 */
static int create_backup_file(const char *source_file)
{
    char backup_filename[512];
    snprintf(backup_filename, sizeof(backup_filename), "%s.backup", source_file);
    
    FILE *src = fopen(source_file, "rb");
    if (!src) {
        fprintf(stderr, "Error: Cannot open source file %s for backup\n", source_file);
        return -1;
    }
    
    FILE *dst = fopen(backup_filename, "wb");
    if (!dst) {
        fprintf(stderr, "Error: Cannot create backup file %s\n", backup_filename);
        fclose(src);
        return -1;
    }
    
    // Get file size for progress reporting
    fseek(src, 0, SEEK_END);
    long file_size = ftell(src);
    fseek(src, 0, SEEK_SET);
    
    // Copy file in chunks
    const size_t chunk_size = 1024 * 1024; // 1MB chunks
    unsigned char *buffer = malloc(chunk_size);
    if (!buffer) {
        fprintf(stderr, "Error: Cannot allocate buffer for file copy\n");
        fclose(src);
        fclose(dst);
        return -1;
    }
    
    size_t bytes_read;
    while ((bytes_read = fread(buffer, 1, chunk_size, src)) > 0) {
        if (fwrite(buffer, 1, bytes_read, dst) != bytes_read) {
            fprintf(stderr, "Error: Failed to write to backup file\n");
            free(buffer);
            fclose(src);
            fclose(dst);
            return -1;
        }
    }
    
    free(buffer);
    fclose(src);
    fclose(dst);
    
    char human_size[32];
    format_file_size(file_size, human_size, sizeof(human_size));
    printf("Created backup: %s (%s)\n", backup_filename, human_size);
    
    return 0;
}

/**
 * @def CLEAN_EXIT(code)
 * @brief Thread-safe exit macro that properly cleans up allocated resources.
 *
 * @details This macro ensures proper resource cleanup before program termination:
 *          - Uses an OpenMP critical section to ensure only one thread performs cleanup.
 *          - Calls `cleanup_all_particle_data()` to free dynamically allocated memory.
 *          - Exits with the specified error code.
 *
 * @param code The exit code for the program.
 *
 * @par Example
 * @code
 * if (error_condition) {
 *     CLEAN_EXIT(1);
 * }
 * @endcode
 */
#define CLEAN_EXIT(code)                 \
    do                                   \
    {                                    \
        _Pragma("omp critical")          \
        {                                \
            cleanup_all_particle_data(); \
            exit(code);                  \
        }                                \
    } while (0)

// --- Global Configuration and Macros ---

#define imin(a, b) ((a) < (b) ? (a) : (b)) ///< Minimum of two integer values.

/**
 * @brief Binary file I/O function declarations.
 * @details These functions provide platform-independent binary file I/O operations
 *          similar to fprintf and fscanf, but ensure consistent binary format
 *          regardless of host architecture.
 */
int fprintf_bin(FILE *fp, const char *format, ...);
int fscanf_bin(FILE *fp, const char *format, ...);
int fprintf_bin_dbl(FILE *fp, const char *format, ...);
int fscanf_bin_dbl(FILE *fp, const char *format, ...);

/**
 * @brief Global simulation feature flags.
 * @details These control various optional behaviors and optimizations
 *          in the simulation. Each flag can be set via command line arguments.
 */
int g_doDebug = 1;           ///< Enable detailed debug output (default: on).
int g_doDynPsi = 1;          ///< Enable dynamic potential recalculation (default: on).
int g_doDynRank = 1;         ///< Enable dynamic rank calculation per step (default: on).
int g_doAllParticleData = 1; ///< Save complete particle evolution history (default: on).
int g_doRestart = 0;         ///< Enable simulation restart from checkpoint.
int g_doRestartForce = 0;    ///< Force regeneration of all snapshots when restarting.
int g_doSimRestart = 0;      ///< Enable simulation restart detection mode (`--sim-restart`).
int g_doSimRestartCheckOnly = 0; ///< Check-only mode for `--sim-restart` (do not actually restart).
int g_restart_mode_active = 0;   ///< Flag: 1 when actively restarting simulation, 0 when only checking or not restarting.
int g_restart_snapshots_is_count = 0; ///< Flag: 1 if `restart_completed_snapshots` represents a count of snapshots; 0 if it is an index.
char *g_restart_file_override = NULL; ///< Optional explicit restart file path (`--restart-file`).
int g_restart_initial_timestep = 0; ///< Actual timestep number from which restart continues (e.g., 10001).
int skip_file_writes = 0;    ///< Skip file writes during simulation restart.
int g_doSimExtend = 0;       ///< Enable simulation extension mode (`--sim-extend`).
char *g_extend_file_source = NULL; ///< Source file to extend from (`--extend-file`).
int g_enable_logging = 0;    ///< Enable logging to file (controlled by `--log` flag).
int g_enable_sidm_scattering = 0;    ///< Enable SIDM scattering physics (0=no, 1=yes). Default is OFF.
int g_sidm_execution_mode = 1;       ///< SIDM execution mode: 0 for serial, 1 for parallel (default).
int g_use_graph_coloring_sidm = 0;   ///< Use graph coloring algorithm for parallel SIDM (eliminates double-booking).
int g_sidm_max_interaction_range = 10; ///< Maximum number of neighbors to check for SIDM scattering. Default is 10.
long long g_total_sidm_scatters = 0; ///< Global counter for total SIDM scatters.
static int g_hybrid_p_cores = 0; ///< Number of P-cores on hybrid CPUs (0 if not hybrid).
static int g_default_max_threads = 0; ///< Default max threads (all cores) for non-SIDM operations.
static double g_sidm_kappa = 50.0;           ///< SIDM opacity kappa (cm\f$^2\f$/g), default 50.0.
static int    g_sidm_kappa_provided = 0;     ///< Flag: 1 if the `--sidm-kappa` option is provided.
static int *g_particle_scatter_state = NULL; ///< Tracks recent scatter history for Adams-Bashforth integrator state reset. Indexed by original particle ID. 0=normal AB, 1=scattered in the previous step (use AB1-like step), 2=one step has passed since scattering (use AB2-like step).

static int *g_current_timestep_scatter_counts = NULL; ///< Tracks scatter counts per particle for the timestep block being processed. The counts are reset after each block write.

// Seed Management Globals
static unsigned long int g_master_seed = 0;         ///< Master seed for the simulation, if provided.
static unsigned long int g_initial_cond_seed = 0;   ///< Seed used for generating initial conditions.
static unsigned long int g_sidm_seed = 0;           ///< Seed used for SIDM calculations.
static int g_master_seed_provided = 0;              ///< Flag: 1 if the `--master-seed` option is provided.
static int g_initial_cond_seed_provided = 0;        ///< Flag: 1 if the `--init-cond-seed` option is provided.
static int g_sidm_seed_provided = 0;                ///< Flag: 1 if the `--sidm-seed` option is provided.
static int g_attempt_load_seeds = 0;                ///< Flag: 1 to attempt loading seeds from files if not provided via command line.

static const char* g_initial_cond_seed_filename_base = "data/last_initial_seed"; ///< Base name for IC seed file.
static const char* g_sidm_seed_filename_base = "data/last_sidm_seed";             ///< Base name for SIDM seed file.

// Profile parameter macros used by profile variables
#define RC 23.0                   ///< Core radius in kpc.
#define RC_NFW_DEFAULT 23.0       ///< Default NFW profile scale radius (kpc).
#define HALO_MASS_NFW 1.15e9      ///< Default NFW profile halo mass in solar masses (\f$M_{\odot}\f$).
#define HALO_MASS 1.0e12          ///< Default general halo mass (used for Cored profile by default) in \f$M_{\odot}\f$.
#define CUTOFF_FACTOR_NFW_DEFAULT 85.0  ///< Default \f$r_{\text{max}}\f$ factor for NFW profile (\f$r_{\text{max}} = \text{factor} \times r_c\f$).
#define CUTOFF_FACTOR_CORED_DEFAULT 85.0 ///< Default \f$r_{\text{max}}\f$ factor for Cored profile (\f$r_{\text{max}} = \text{factor} \times r_c\f$).
#define FALLOFF_FACTOR_NFW_DEFAULT 19.0 ///< Default falloff concentration parameter \f$C\f$ for NFW profile.
#define NUM_MINI_SUBSTEPS_BOOTSTRAP 20 ///< Number of mini Euler steps per bootstrap full step.

// Profile Selection and NFW-Specific Parameters
static int g_use_nfw_profile = 0; ///< Flag to select NFW profile for ICs (1=NFW, 0=use other profile flags).
static int g_use_hernquist_aniso_profile = 0; ///< Flag to use anisotropic Hernquist profile for ICs.
static int g_use_hernquist_numerical = 0; ///< Flag to use numerical Hernquist profile (OM-compatible).
static int g_use_numerical_isotropic = 0; ///< Flag for numerical Hernquist with true f(E).

// Sort Benchmarking Variables (for option 6)
static int g_sort_call_count = 0;    ///< Total number of sort calls made
static int g_benchmark_count = 0;    ///< Number of times benchmarking was performed
static int g_quadsort_wins = 0;      ///< Number of times quadsort was faster
static int g_radix_wins = 0;         ///< Number of times radix sort was faster
static int g_insertion_wins = 0;     ///< Number of times insertion sort was faster
static double g_total_quadsort_time = 0.0; ///< Total time spent in quadsort benchmarks (ms)
static double g_total_radix_time = 0.0;    ///< Total time spent in radix benchmarks (ms)
static double g_total_insertion_time = 0.0; ///< Total time spent in insertion benchmarks (ms)
static char g_current_best_sort[32] = "insertion_parallel"; ///< Tracks the best-performing sorting algorithm for adaptive mode.

static double g_nfw_profile_rc = RC_NFW_DEFAULT;    ///< Scale radius \f$r_c\f$ for the NFW profile (kpc). Populated from `g_scale_radius_param`.
static double g_nfw_profile_halo_mass = HALO_MASS_NFW; ///< Total halo mass for the NFW profile (\f$M_{\odot}\f$). Populated from `g_halo_mass_param`.
static double g_nfw_profile_rmax_norm_factor = CUTOFF_FACTOR_NFW_DEFAULT; ///< Cutoff radius factor for the NFW profile. Populated from `g_cutoff_factor_param`.
static double g_nfw_profile_falloff_factor = FALLOFF_FACTOR_NFW_DEFAULT; ///< Falloff concentration parameter \f$C\f$ for the NFW profile. Populated from `g_falloff_factor_param`.


// Generalized Profile Parameters (set by command line flags)
static double g_scale_radius_param = RC;        ///< Generalized scale radius (kpc), defaults to RC macro.
static double g_halo_mass_param = HALO_MASS;    ///< Generalized halo mass (\f$M_{\odot}\f$), defaults to HALO_MASS macro.
static double g_cutoff_factor_param = CUTOFF_FACTOR_CORED_DEFAULT; ///< Generalized rmax factor, defaults to Cored's default.
static char   g_profile_type_str[16] = "nfw";   ///< Profile type string ("nfw", "cored", or "hernquist"), default "nfw".

static int g_scale_radius_param_provided = 0;   ///< Flag: 1 if the `--scale-radius` option is provided.
static int g_halo_mass_param_provided = 0;      ///< Flag: 1 if the `--halo-mass` option is provided.
static int g_cutoff_factor_param_provided = 0;  ///< Flag: 1 if the `--cutoff-factor` option is provided.
static double g_falloff_factor_param = FALLOFF_FACTOR_NFW_DEFAULT; ///< Generalized falloff factor, defaults to NFW's default.
static int    g_falloff_factor_param_provided = 0;                  ///< Flag: 1 if the `--falloff-factor` option is provided.
static int g_profile_type_str_provided = 0;     ///< Flag: 1 if the `--profile` option is provided.

// =============================================================================
// ANISOTROPY MODEL PARAMETERS
// =============================================================================
// Hernquist constant-beta anisotropy
static double g_anisotropy_beta = 0.0;          ///< Anisotropy parameter \f$\beta\f$ for Hernquist profile
static int g_anisotropy_beta_provided = 0;      ///< Flag: 1 if the `--aniso-beta` option is provided.

/**
 * @brief Osipkov-Merritt Anisotropy Model Implementation
 *
 * @details The Osipkov-Merritt (OM) model provides radially-varying velocity
 *          anisotropy characterized by the anisotropy parameter:
 *
 *          \f$\beta(r) = r^2/(r^2 + r_a^2)\f$
 *
 *          where \f$r_a\f$ is the anisotropy radius. The model transitions from
 *          isotropic (\f$\beta=0\f$) at \f$r=0\f$ to radially-biased (\f$\beta \rightarrow 1\f$) as \f$r \rightarrow \infty\f$.
 *
 *          Implementation uses the augmented density method:
 *          - Define augmented density: \f$\rho_Q(r) = \rho(r)(1 + r^2/r_a^2)\f$
 *          - Use \f$Q = E - L^2/(2r_a^2)\f$ as the OM invariant
 *          - Apply Eddington inversion to \f$\rho_Q\f$ to get \f$f(Q)\f$
 *          - Sample velocities in pseudo-space where Q-surfaces are spheres
 *          - Transform to physical velocities using mapping:
 *            \f$v_r = w \cos(\theta)\f$, \f$v_t = w \sin(\theta)/\sqrt{1 + r^2/r_a^2}\f$
 *
 * @note Compatible with NFW, Cored, and numerical Hernquist profiles.
 */
// Osipkov-Merritt radially-varying anisotropy
static int g_use_om_profile = 0;                ///< Flag: 1 if OM model is requested
static double g_om_ra_scale_factor = 0.0;       ///< Anisotropy radius as multiple of scale radius
static double g_om_anisotropy_radius = 0.0;     ///< Physical anisotropy radius \f$r_a\f$ (kpc)
static int g_om_aniso_factor_provided = 0;      ///< Flag: 1 if the `--aniso-factor` or `--aniso-betascale` option is provided.

// Cored Plummer-like Profile Specific Parameters (populated from generalized flags)
static double g_cored_profile_rc = RC;              ///< Scale radius \f$r_c\f$ for the Cored profile (kpc). Populated from `g_scale_radius_param`.
static double g_cored_profile_halo_mass = HALO_MASS; ///< Total halo mass for the Cored profile (\f$M_{\odot}\f$). Populated from `g_halo_mass_param`.
static double g_cored_profile_rmax_factor = CUTOFF_FACTOR_CORED_DEFAULT;   ///< Cutoff radius factor for the Cored profile. Populated from `g_cutoff_factor_param`.

/**
 * @brief Conditional compilation macros for feature flags.
 * @details These macros provide a cleaner syntax for conditional code blocks
 *          that depend on the global feature flags.
 *
 * @def IF_DEBUG
 * @brief Macro for code blocks executed only if `g_doDebug` is true.
 * @def IF_DYNPSI
 * @brief Macro for code blocks executed only if `g_doDynPsi` is true.
 * @def IF_DYNRANK
 * @brief Macro for code blocks executed only if `g_doDynRank` is true.
 * @def IF_ALL_PART
 * @brief Macro for code blocks executed only if `g_doAllParticleData` is true.
 */
#define IF_DEBUG if (g_doDebug)
#define IF_DYNPSI if (g_doDynPsi)
#define IF_DYNRANK if (g_doDynRank)
#define IF_ALL_PART if (g_doAllParticleData)

/**
 * @brief Convolution method selection for density smoothing.
 * @details 0 = FFT-based convolution (default, faster for large datasets).
 *          1 = Direct spatial convolution (more accurate but slower).
 */
int debug_direct_convolution = 0;

// =============================================================================
// PHYSICAL CONSTANTS AND ASTROPHYSICAL PARAMETERS
// =============================================================================
//
// Core constants and unit conversion factors for astrophysical calculations
#define PI 3.14159265358979323846 ///< Mathematical constant \f$\pi\f$.
#define G_CONST 4.3e-6           ///< Newton's gravitational constant in kpc (km/sec)\f$^2\f$/\f$M_{\odot}\f$.
#define kmsec_to_kpcmyr 1.02271e-3 ///< Conversion factor: km/s to kpc/Myr.
#define VEL_CONV_SQ (kmsec_to_kpcmyr * kmsec_to_kpcmyr) ///< Velocity conversion squared (kpc/Myr)\f$^2\f$ per (km/s)\f$^2\f$.

// Mathematical utility macros
#define sqr(x) ((x) * (x))       ///< Calculates the square of a value: \f$x^2\f$.
#define cube(x) ((x) * (x) * (x)) ///< Calculates the cube of a value: \f$x^3\f$.

// Analysis and visualization parameters
#define HIST_NBINS 400          ///< Number of bins for 2D phase-space histograms (400x400 resolution).

// Sorting algorithm parameters
#define RADIX_BITS 8            ///< Number of bits processed per radix sort pass (8 bits = 256 buckets).
#define RADIX_SIZE (1 << RADIX_BITS) ///< Number of buckets for radix sort (\f$2^8 = 256\f$).
#define RADIX_MASK (RADIX_SIZE - 1)  ///< Bit mask for extracting radix bucket index (0xFF).

static double g_active_halo_mass = HALO_MASS; ///< Active halo mass for N-body force calculations.

/**
 * @brief Angular momentum selection configuration for particle filtering.
 * @details Mode 0: Select particles with the 5 lowest \f$L\f$ values.
 *          Mode 1: Select particles with \f$L\f$ values closest to \f$L_{compare}\f$.
 */
static double g_l_target_value = -1.0;  ///< Target \f$L\f$ value for `--lvals-target` mode. A negative value indicates the option was not set.
static int use_closest_to_Lcompare = 0; ///< Mode selector (0=lowest \f$L\f$, 1=closest to \f$L_{compare}\f$). Default is lowest \f$L\f$.
static double Lcompare = 0.05;          ///< Reference \f$L\f$ value for closest-match mode (Mode 1).

// =============================================================================
// GLOBAL PARTICLE DATA ARRAYS
// =============================================================================

/** @brief Global arrays for particle data processing (block storage). */
static float *L_block = NULL;    ///< Angular momentum block.
static int *Rank_block = NULL;   ///< Particle rank (sorted position) block.
static float *R_block = NULL;    ///< Radial position block.
static float *Vrad_block = NULL; ///< Radial velocity block.
static float *phi_block = NULL;   ///< Phi angle block.
static int *scatter_count_block = NULL; ///< Scatter count block.
static int *ID_block = NULL;      ///< Particle ID block (original IDs).

// =============================================================================
// TRAJECTORY BUFFERING SYSTEM
// =============================================================================
// Buffers for storing trajectory data before flushing to disk.
// This system uses fixed-size buffers to ensure constant memory usage
// regardless of the simulation length.

// --- Buffer Configuration ---
/** Size of the trajectory buffers (in timesteps), synchronized with snapshot buffer size. */
static int g_trajectory_buffer_size = 0;
/** Current index for writing into the trajectory buffers. */
static int g_trajectory_buffer_index = 0;

// --- Main Trajectory Buffers ---
/** Buffer for time values [timestep_in_buffer]. */
static double *g_time_buf = NULL;
/** Buffer for radius [particle_idx][timestep_in_buffer]. */
static double **g_trajectories_buf = NULL;
/** Buffer for radial velocity [particle_idx][timestep_in_buffer]. */
static double **g_velocities_buf = NULL;
/** Buffer for radial direction cosine mu [particle_idx][timestep_in_buffer]. */
static double **g_mu_buf = NULL;
/** Buffer for total relative energy E_rel [particle_idx][timestep_in_buffer]. */
static double **g_E_buf = NULL;
/** Buffer for angular momentum [particle_idx][timestep_in_buffer]. */
static double **g_L_buf = NULL;

// --- Lowest-L Trajectory Buffers ---
/** Buffer for radius of low-L particles [particle_idx][timestep_in_buffer]. */
static double **g_lowestL_r_buf = NULL;
/** Buffer for energy of low-L particles [particle_idx][timestep_in_buffer]. */
static double **g_lowestL_E_buf = NULL;
/** Buffer for angular momentum of low-L particles [particle_idx][timestep_in_buffer]. */
static double **g_lowestL_L_buf = NULL;

// --- File Handles ---
/** File handle for trajectories.dat, kept open for performance. */
static FILE *g_traj_file = NULL;
/** File handle for energy_and_angular_momentum_vs_time.dat. */
static FILE *g_energy_file = NULL;
/** File handle for lowest_l_trajectories.dat. */
static FILE *g_lowestL_file = NULL;
/** File handle for single_trajectory.dat. */
static FILE *g_single_traj_file = NULL;

// =============================================================================
// DOUBLE PRECISION SNAPSHOT BUFFER GLOBALS
// =============================================================================
/**
 * @brief Rolling buffer system for preserving last 4 snapshots in full double precision.
 * @details Maintains a circular buffer of the most recent snapshots to avoid precision
 *          loss during restart/extend operations. The buffer stores:
 *          - Snapshots 0, then [0,1], then [0,1,2], then [0,1,2,3]
 *          - After 4 snapshots: rolling last 4 (e.g., [77,78,79,80] for 81-snapshot run)
 *
 *          Files written: double_buffer_all_particle_data_<tag>.dat (28 bytes/particle/snap)
 *                        double_buffer_all_particle_phi_<tag>.dat (8 bytes/particle/snap)
 *
 *          Buffer size: (28 + 8) * npts * 4 = 144 * npts bytes in memory
 */

/** Buffer for particle radii (last 4 snapshots, npts particles each). */
static double *g_dbl_buf_R = NULL;

/** Buffer for particle radial velocities (last 4 snapshots, npts particles each). */
static double *g_dbl_buf_Vrad = NULL;

/** Buffer for particle angular momenta (last 4 snapshots, npts particles each). */
static double *g_dbl_buf_L = NULL;

/** Buffer for particle ranks (last 4 snapshots, npts particles each). */
static int *g_dbl_buf_Rank = NULL;

/** Buffer for particle phi angles (last 4 snapshots, npts particles each). */
static double *g_dbl_buf_phi = NULL;

/** Next write slot in rolling 4-snapshot circular buffer (range: 0-3, wraps via modulo). */
static int g_dbl_buf_current_slot = 0;

/** Number of valid snapshots in buffer (range: 0-4, saturates at 4). */
static int g_dbl_buf_count = 0;

/** Number of particles, used for buffer indexing. */
static int g_dbl_buf_npts = 0;

// --- Persistent Initial Values ---
/** Stores initial energy E_rel for tracked particles [particle_idx]. */
static double *g_E_init_vals = NULL;
/** Stores initial angular momentum \f$L\f$ for tracked particles [particle_idx]. */
static double *g_L_init_vals = NULL;

/** @brief Variables for tracking low angular momentum particles. */
static int nlowest = 5;           ///< Number of lowest angular momentum particles to track.
static int num_traj_particles = 10; ///< Number of particles to track in trajectories.dat (by original ID).
static int *chosen = NULL;        ///< Array of indices (original IDs) for selected low-L particles.

/**
 * @brief Frees all global arrays used for particle data processing.
 * @details Frees L_block, Rank_block, R_block, Vrad_block, and chosen.
 */
void cleanup_all_particle_data(void)
{
    free(L_block);
    free(Rank_block);
    free(R_block);
    free(Vrad_block);
    free(phi_block);
    free(scatter_count_block);
    free(ID_block);

    // Free double precision buffer arrays
    free(g_dbl_buf_R);
    free(g_dbl_buf_Vrad);
    free(g_dbl_buf_L);
    free(g_dbl_buf_Rank);
    free(g_dbl_buf_phi);
    g_dbl_buf_R = NULL;
    g_dbl_buf_Vrad = NULL;
    g_dbl_buf_L = NULL;
    g_dbl_buf_Rank = NULL;
    g_dbl_buf_phi = NULL;

    // Free low angular momentum tracking arrays
    free(chosen);

}

/**
 * @brief Allocates memory for the double-precision snapshot buffer system.
 * @details Allocates circular buffers to store the last 4 snapshots in full double precision.
 *          This prevents precision loss during restart/extend operations by avoiding the
 *          double→float32→double conversion inherent in the regular all_particle_data files.
 *
 *          Buffer grows incrementally: stores [0], then [0,1], then [0,1,2], then [0,1,2,3],
 *          then rolls to maintain the last 4 snapshots.
 *
 * @param npts Number of particles in the simulation.
 *
 * @note Exits via CLEAN_EXIT(1) if allocation fails.
 * @see cleanup_all_particle_data()
 */
static void allocate_double_precision_buffers(int npts)
{
    const int NUM_SLOTS = 4;
    size_t buffer_size_double = NUM_SLOTS * npts * sizeof(double);
    size_t buffer_size_int = NUM_SLOTS * npts * sizeof(int);

    g_dbl_buf_R = (double *)calloc(NUM_SLOTS * npts, sizeof(double));
    g_dbl_buf_Vrad = (double *)calloc(NUM_SLOTS * npts, sizeof(double));
    g_dbl_buf_L = (double *)calloc(NUM_SLOTS * npts, sizeof(double));
    g_dbl_buf_Rank = (int *)calloc(NUM_SLOTS * npts, sizeof(int));
    g_dbl_buf_phi = (double *)calloc(NUM_SLOTS * npts, sizeof(double));

    if (!g_dbl_buf_R || !g_dbl_buf_Vrad || !g_dbl_buf_L || !g_dbl_buf_Rank || !g_dbl_buf_phi) {
        fprintf(stderr, "ERROR: Failed to allocate double-precision snapshot buffers\n");
        fprintf(stderr, "       Requested: %.2f MB per buffer × 5 buffers = %.2f MB total\n",
                buffer_size_double / 1048576.0,
                (buffer_size_double * 4 + buffer_size_int) / 1048576.0);
        CLEAN_EXIT(1);
    }

    g_dbl_buf_npts = npts;
    g_dbl_buf_current_slot = 0;
    g_dbl_buf_count = 0;
}

/**
 * @brief Adds the current snapshot to the double-precision circular buffer.
 * @details Copies particle state (R, Vrad, L, Rank, phi) from current simulation
 *          state into the circular buffer at the current slot position. Updates
 *          buffer slot position and count. Buffer grows from 0 to 4 snapshots,
 *          then maintains rolling last 4.
 *
 * @param particles Pointer to particle state arrays [R, Vrad, L, ...]
 * @param phi_array Array of particle phi angles
 * @param rank_array Array of particle ranks (sorted positions)
 * @param npts Number of particles
 */
static void add_snapshot_to_double_buffer(double **particles, double *phi_array,
                                          int *rank_array, int npts)
{
    if (g_dbl_buf_npts != npts) {
        fprintf(stderr, "ERROR: Buffer npts mismatch (%d vs %d)\n",
                g_dbl_buf_npts, npts);
        CLEAN_EXIT(1);
    }

    // Calculate offset into circular buffer for current slot
    const int offset = g_dbl_buf_current_slot * npts;

    // Copy particle data to buffer
    for (int i = 0; i < npts; i++) {
        g_dbl_buf_R[offset + i] = particles[0][i];       // Radius
        g_dbl_buf_Vrad[offset + i] = particles[1][i];    // Radial velocity
        g_dbl_buf_L[offset + i] = particles[2][i];       // Angular momentum
        g_dbl_buf_Rank[offset + i] = rank_array[i];      // Rank
        g_dbl_buf_phi[offset + i] = phi_array[i];        // Phi angle
    }

    // Update circular buffer state
    g_dbl_buf_current_slot = (g_dbl_buf_current_slot + 1) % 4;
    if (g_dbl_buf_count < 4) {
        g_dbl_buf_count++;
    }
}

/**
 * @brief Flushes double-precision buffer to disk files.
 * @details Writes the entire 4-snapshot buffer to disk, overwriting previous
 *          buffer files. Creates two binary files:
 *          - double_buffer_all_particle_data_<tag>.dat (Rank, R, Vrad, L)
 *          - double_buffer_all_particle_phi_<tag>.dat (phi)
 *
 *          File format per particle per snapshot:
 *          - data file: int32 Rank + double R + double Vrad + double L = 28 bytes
 *          - phi file: double phi = 8 bytes
 *
 * @param tag Simulation tag for filename
 */
static void flush_double_buffer_to_disk(const char *tag)
{
    char data_filename[512];
    char phi_filename[512];

    snprintf(data_filename, sizeof(data_filename),
             "data/double_buffer_all_particle_data_%s.dat", tag);
    snprintf(phi_filename, sizeof(phi_filename),
             "data/double_buffer_all_particle_phi_%s.dat", tag);

    // Write data file (Rank, R, Vrad, L) for all snapshots in buffer
    FILE *fp_data = fopen(data_filename, "wb");
    if (!fp_data) {
        fprintf(stderr, "ERROR: Cannot create double buffer data file: %s\n",
                data_filename);
        CLEAN_EXIT(1);
    }

    // Write phi file for all snapshots in buffer
    FILE *fp_phi = fopen(phi_filename, "wb");
    if (!fp_phi) {
        fprintf(stderr, "ERROR: Cannot create double buffer phi file: %s\n",
                phi_filename);
        fclose(fp_data);
        CLEAN_EXIT(1);
    }

    // Write buffer contents (all stored snapshots, up to 4)
    const int total_particles = g_dbl_buf_count * g_dbl_buf_npts;

    for (int i = 0; i < total_particles; i++) {
        // Write to data file: Rank(int32) + R(double) + Vrad(double) + L(double)
        fprintf_bin_dbl(fp_data, "%d %f %f %f",
                       g_dbl_buf_Rank[i],
                       g_dbl_buf_R[i],
                       g_dbl_buf_Vrad[i],
                       g_dbl_buf_L[i]);

        // Write to phi file: phi(double)
        fprintf_bin_dbl(fp_phi, "%f", g_dbl_buf_phi[i]);
    }

    fclose(fp_data);
    fclose(fp_phi);
}

/**
 * @brief Attempts to load a snapshot from the double-precision buffer files.
 * @details Tries to load the specified snapshot from the double buffer files.
 *          Returns 1 if successful, 0 if snapshot not in buffer or files missing.
 *          On success, loads particle state with full double precision (no conversion).
 *
 *          Buffer stores snapshots in circular order. For 81-snapshot run, buffer
 *          contains snapshots [77, 78, 79, 80]. Snapshot 79 is located
 *          at buffer position 2.
 *
 * @param tag Simulation tag for filename
 * @param snapshot_index Snapshot number to load (e.g., 79)
 * @param total_snapshots Total snapshots written so far (e.g., 81)
 * @param particles Destination particle arrays [R, Vrad, L, ID, \f$\mu\f$, cos_phi, sin_phi]
 * @param npts Number of particles
 * @param particle_ids Destination for particle IDs (unused in this function).
 * @return 1 if loaded successfully, 0 if snapshot not available in buffer
 */
static int load_from_double_buffer(const char *tag, int snapshot_index,
                                   int total_snapshots, double **particles,
                                   int npts, int *particle_ids)
{
    (void)particle_ids; // Unused parameter
    char data_filename[512];
    char phi_filename[512];

    snprintf(data_filename, sizeof(data_filename),
             "data/double_buffer_all_particle_data_%s.dat", tag);
    snprintf(phi_filename, sizeof(phi_filename),
             "data/double_buffer_all_particle_phi_%s.dat", tag);

    // Check if buffer files exist
    FILE *fp_test = fopen(data_filename, "rb");
    if (!fp_test) {
        // Files do not exist - simulation predates double buffer feature
        return 0;
    }
    fclose(fp_test);

    // Calculate which snapshots are in the buffer
    // Buffer contains last 4 snapshots (or fewer if total_snapshots < 4)
    int snapshots_in_buffer = (total_snapshots < 4) ? total_snapshots : 4;
    int first_snapshot_in_buffer = total_snapshots - snapshots_in_buffer;

    // Check if requested snapshot is in buffer
    if (snapshot_index < first_snapshot_in_buffer) {
        // Snapshot not in buffer (too old)
        return 0;
    }

    // Calculate position in buffer (0-3) - buffer is circular, use modulo
    int buffer_position = snapshot_index % 4;

    // Open buffer files
    FILE *fp_data = fopen(data_filename, "rb");
    FILE *fp_phi = fopen(phi_filename, "rb");

    if (!fp_data || !fp_phi) {
        if (fp_data) fclose(fp_data);
        if (fp_phi) fclose(fp_phi);
        return 0;
    }

    // Seek to correct position in buffer
    long data_offset = (long)buffer_position * npts * (sizeof(int) + 3 * sizeof(double));
    long phi_offset = (long)buffer_position * npts * sizeof(double);

    fseek(fp_data, data_offset, SEEK_SET);
    fseek(fp_phi, phi_offset, SEEK_SET);

    // Read particle data with full double precision
    for (int i = 0; i < npts; i++) {
        int rank_val;
        double R_val, Vrad_val, L_val, phi_val;

        // Read from data file: Rank(int32) + R(double) + Vrad(double) + L(double)
        int items_read = fscanf_bin_dbl(fp_data, "%d %lf %lf %lf",
                                        &rank_val, &R_val, &Vrad_val, &L_val);
        if (items_read != 4) {
            fprintf(stderr, "ERROR: Failed to read from double buffer data file at particle %d\n", i);
            fclose(fp_data);
            fclose(fp_phi);
            return 0;
        }

        // Read from phi file: phi(double)
        items_read = fscanf_bin_dbl(fp_phi, "%lf", &phi_val);
        if (items_read != 1) {
            fprintf(stderr, "ERROR: Failed to read from double buffer phi file at particle %d\n", i);
            fclose(fp_data);
            fclose(fp_phi);
            return 0;
        }

        // Store in particles array (same format as load_particles_from_restart)
        particles[0][i] = R_val;                    // Radius (full double precision)
        particles[1][i] = Vrad_val;                 // Radial velocity (full double precision)
        particles[2][i] = L_val;                    // Angular momentum (full double precision)
        particles[3][i] = (double)rank_val;         // Load rank; caller rebuilds inverse_map to map this to particle ID
        particles[4][i] = 0.0;                      // mu not needed after restart
        particles[5][i] = cos(phi_val);             // cos(phi) from full precision phi
        particles[6][i] = sin(phi_val);             // sin(phi) from full precision phi
    }

    fclose(fp_data);
    fclose(fp_phi);

    // Successfully loaded from double buffer
    return 1;
}

// =============================================================================
// TRAJECTORY BUFFER HELPER FUNCTIONS
// =============================================================================

/**
 * @brief Allocates memory for all trajectory buffering arrays.
 * @details This function is called once at the start of the simulation to
 *          allocate fixed-size buffers for storing trajectory data. The buffer
 *          size is determined by the snapshot writing schedule to ensure data
 *          is flushed to disk in synchronized blocks.
 *
 * @param num_traj_particles The number of low-ID particles to track.
 * @param nlowest The number of low-L particles to track.
 * @param buffer_size The total number of timesteps the buffer can hold.
 */
static void allocate_trajectory_buffers(int num_traj_particles, int nlowest, int buffer_size, int npts)
{
    log_message("INFO", "Allocating trajectory buffers for %d timesteps.", buffer_size);

    g_trajectory_buffer_size = buffer_size;

    // Allocate main trajectory buffers
    g_trajectories_buf = (double **)malloc(num_traj_particles * sizeof(double *));
    g_velocities_buf = (double **)malloc(num_traj_particles * sizeof(double *));
    g_mu_buf = (double **)malloc(num_traj_particles * sizeof(double *));
    g_E_buf = (double **)malloc(num_traj_particles * sizeof(double *));
    g_L_buf = (double **)malloc(num_traj_particles * sizeof(double *));

    if (!g_trajectories_buf || !g_velocities_buf || !g_mu_buf || !g_E_buf || !g_L_buf) {
        fprintf(stderr, "ERROR: Failed to allocate pointers for main trajectory buffers\n");
        CLEAN_EXIT(1);
    }

    for (int i = 0; i < num_traj_particles; i++) {
        g_trajectories_buf[i] = (double *)calloc(buffer_size, sizeof(double));
        g_velocities_buf[i] = (double *)calloc(buffer_size, sizeof(double));
        g_mu_buf[i] = (double *)calloc(buffer_size, sizeof(double));
        g_E_buf[i] = (double *)calloc(buffer_size, sizeof(double));
        g_L_buf[i] = (double *)calloc(buffer_size, sizeof(double));
        if (!g_trajectories_buf[i] || !g_velocities_buf[i] || !g_mu_buf[i] || !g_E_buf[i] || !g_L_buf[i]) {
            fprintf(stderr, "ERROR: Failed to allocate trajectory buffer for particle %d\n", i);
            CLEAN_EXIT(1);
        }
    }

    // Allocate time buffer
    g_time_buf = (double *)calloc(buffer_size, sizeof(double));
    if (!g_time_buf) {
        fprintf(stderr, "ERROR: Failed to allocate time buffer\n");
        CLEAN_EXIT(1);
    }

    // Allocate lowest-L buffers if needed
    if (nlowest > 0) {
        g_lowestL_r_buf = (double **)malloc(nlowest * sizeof(double *));
        g_lowestL_E_buf = (double **)malloc(nlowest * sizeof(double *));
        g_lowestL_L_buf = (double **)malloc(nlowest * sizeof(double *));
        if (!g_lowestL_r_buf || !g_lowestL_E_buf || !g_lowestL_L_buf) {
            fprintf(stderr, "ERROR: Failed to allocate pointers for lowest-L trajectory buffers\n");
            CLEAN_EXIT(1);
        }

        for (int i = 0; i < nlowest; i++) {
            g_lowestL_r_buf[i] = (double *)calloc(buffer_size, sizeof(double));
            g_lowestL_E_buf[i] = (double *)calloc(buffer_size, sizeof(double));
            g_lowestL_L_buf[i] = (double *)calloc(buffer_size, sizeof(double));
            if (!g_lowestL_r_buf[i] || !g_lowestL_E_buf[i] || !g_lowestL_L_buf[i]) {
                fprintf(stderr, "ERROR: Failed to allocate lowest-L buffer for particle %d\n", i);
                CLEAN_EXIT(1);
            }
        }
    }

    // Allocate persistent initial value arrays for ALL particles (indexed by final rank ID)
    g_E_init_vals = (double *)calloc(npts, sizeof(double));
    g_L_init_vals = (double *)calloc(npts, sizeof(double));
    if (!g_E_init_vals || !g_L_init_vals) {
        fprintf(stderr, "ERROR: Failed to allocate initial value arrays\n");
        CLEAN_EXIT(1);
    }
}

/**
 * @brief Flushes all buffered trajectory data to their respective files.
 * @details This function writes the collected trajectory data for a block of
 *          timesteps to disk. It handles opening files on the first write and
 *          appends data on subsequent calls. It is synchronized with the main
 *          snapshot buffer flush.
 *
 * @param items_to_write The number of timesteps currently in the buffer to write.
 * @param num_traj_particles The number of low-ID particles being tracked.
 * @param nlowest The number of low-L particles being tracked.
 */
static void flush_trajectory_buffers(int items_to_write, int num_traj_particles, int nlowest) {
    if (items_to_write <= 0) return;

    char filename[512];
    const char *mode = (g_restart_mode_active || g_doSimExtend) ? "ab" : "wb"; // Use append mode for restart/extend

    // --- Flush trajectories.dat and single_trajectory.dat ---
    if (g_traj_file == NULL) {
        get_suffixed_filename("data/trajectories.dat", 1, filename, sizeof(filename));
        g_traj_file = fopen(filename, mode);
        get_suffixed_filename("data/single_trajectory.dat", 1, filename, sizeof(filename));
        g_single_traj_file = fopen(filename, mode);
        if (!g_traj_file || !g_single_traj_file) {
            fprintf(stderr, "ERROR: Could not open trajectory files for writing.\n");
            return;
        }
    }

    for (int step = 0; step < items_to_write; step++) {
        fprintf_bin(g_traj_file, "%f", g_time_buf[step]);
        for (int p = 0; p < num_traj_particles; p++) {
            fprintf_bin(g_traj_file, " %f %f %f", g_trajectories_buf[p][step], g_velocities_buf[p][step], g_mu_buf[p][step]);
        }
        fprintf_bin(g_traj_file, "\n");

        if (num_traj_particles > 0) {
            fprintf_bin(g_single_traj_file, "%f %f %f %f\n", g_time_buf[step], g_trajectories_buf[0][step], g_velocities_buf[0][step], g_mu_buf[0][step]);
        }
    }

    // --- Flush energy_and_angular_momentum_vs_time.dat ---
    if (g_energy_file == NULL) {
        get_suffixed_filename("data/energy_and_angular_momentum_vs_time.dat", 1, filename, sizeof(filename));
        g_energy_file = fopen(filename, mode);
        if (!g_energy_file) {
            fprintf(stderr, "ERROR: Could not open energy/L file for writing.\n");
            return;
        }
    }

    for (int step = 0; step < items_to_write; step++) {
        fprintf_bin(g_energy_file, "%f", g_time_buf[step]);
        for (int p = 0; p < num_traj_particles; p++) {
            fprintf_bin(g_energy_file, " %f %f %f %f", g_E_buf[p][step], g_E_init_vals[p], g_L_buf[p][step], g_L_init_vals[p]);
        }
        fprintf_bin(g_energy_file, "\n");
    }

    // --- Flush lowest_l or chosen_l trajectories.dat ---
    if (nlowest > 0 && g_lowestL_file == NULL) {
        // Use chosen_l if --lvals-target was specified, otherwise lowest_l
        const char* base_filename = (g_l_target_value >= 0.0) ? "data/chosen_l_trajectories.dat" : "data/lowest_l_trajectories.dat";
        get_suffixed_filename(base_filename, 1, filename, sizeof(filename));
        g_lowestL_file = fopen(filename, mode);
        if (!g_lowestL_file) {
            fprintf(stderr, "ERROR: Could not open lowest-L trajectory file for writing.\n");
            return;
        }
    }

    if (g_lowestL_file) {
        for (int step = 0; step < items_to_write; step++) {
            fprintf_bin(g_lowestL_file, "%f", g_time_buf[step]);
            for (int p = 0; p < nlowest; p++) {
                fprintf_bin(g_lowestL_file, " %f %f %f", g_lowestL_r_buf[p][step], g_lowestL_E_buf[p][step], g_lowestL_L_buf[p][step]);
            }
            fprintf_bin(g_lowestL_file, "\n");
        }
    }

    // Reset buffer index for next block
    g_trajectory_buffer_index = 0;
}

/**
 * @brief Frees all memory associated with the trajectory buffering system.
 * @details This function is called at the end of the simulation. It ensures
 *          any remaining data in the buffers is flushed, closes all open file
 *          handles, and deallocates all buffer memory.
 *
 * @param num_traj_particles Number of low-ID particles tracked.
 * @param nlowest Number of low-L particles tracked.
 */
static void cleanup_trajectory_buffers(int num_traj_particles, int nlowest) {
    // Flush any remaining data in the buffer
    flush_trajectory_buffers(g_trajectory_buffer_index, num_traj_particles, nlowest);

    // Close all file handles
    if (g_traj_file) { fclose(g_traj_file); g_traj_file = NULL; }
    if (g_single_traj_file) { fclose(g_single_traj_file); g_single_traj_file = NULL; }
    if (g_energy_file) { fclose(g_energy_file); g_energy_file = NULL; }
    if (g_lowestL_file) { fclose(g_lowestL_file); g_lowestL_file = NULL; }

    // Free main trajectory buffers
    if (g_trajectories_buf) {
        for (int i = 0; i < num_traj_particles; i++) free(g_trajectories_buf[i]);
        free(g_trajectories_buf);
    }
    if (g_velocities_buf) {
        for (int i = 0; i < num_traj_particles; i++) free(g_velocities_buf[i]);
        free(g_velocities_buf);
    }
    if (g_mu_buf) {
        for (int i = 0; i < num_traj_particles; i++) free(g_mu_buf[i]);
        free(g_mu_buf);
    }
    if (g_E_buf) {
        for (int i = 0; i < num_traj_particles; i++) free(g_E_buf[i]);
        free(g_E_buf);
    }
    if (g_L_buf) {
        for (int i = 0; i < num_traj_particles; i++) free(g_L_buf[i]);
        free(g_L_buf);
    }
    free(g_time_buf);

    // Free lowest-L buffers
    if (g_lowestL_r_buf) {
        for (int i = 0; i < nlowest; i++) free(g_lowestL_r_buf[i]);
        free(g_lowestL_r_buf);
    }
    if (g_lowestL_E_buf) {
        for (int i = 0; i < nlowest; i++) free(g_lowestL_E_buf[i]);
        free(g_lowestL_E_buf);
    }
    if (g_lowestL_L_buf) {
        for (int i = 0; i < nlowest; i++) free(g_lowestL_L_buf[i]);
        free(g_lowestL_L_buf);
    }

    // Free persistent initial value arrays
    free(g_E_init_vals);
    free(g_L_init_vals);
}

/**
 * @brief Find the last complete snapshot that exists across ALL output files.
 * @details When a simulation is interrupted, different files may have different
 *          amounts of data due to buffering. This function finds the last snapshot
 *          that is complete in ALL files to ensure consistency on restart.
 *
 * @param npts Number of particles
 * @param dtwrite Timestep interval for writes
 * @param num_traj_particles Number of trajectory particles
 * @param nlowest Number of lowest-L particles
 * @param file_suffix The file suffix string
 * @return The last complete snapshot number common to all files, or -1 on error
 */
static int find_last_common_complete_snapshot(int npts, int dtwrite, int num_traj_particles, int nlowest, const char* file_suffix) {
    char filename[512];
    FILE *fp;
    long file_size;
    int min_complete_snapshot = INT_MAX;

    // Bytes per snapshot/timestep for each file type
    int bytes_per_snapshot_apd = npts * 16;  // int + 3 floats
    int bytes_per_snapshot_ids = npts * 4;   // 1 int
    int bytes_per_snapshot_phi = npts * 4;   // 1 float
    int bytes_per_snapshot_scatter = npts * 4; // 1 int

    // For trajectory files (per timestep, not per snapshot)
    int bytes_per_timestep_traj = 4 + num_traj_particles * 12;  // time + particles*(r,vrad,mu)
    int bytes_per_timestep_single = 16;  // time + 1 particle*(r,vrad,mu)
    int bytes_per_timestep_energy = 4 + num_traj_particles * 16;  // time + particles*(E,E_init,L,L_init)
    int bytes_per_timestep_lowestL = 4 + nlowest * 12;  // time + particles*(r,E,L)

    // Check all_particle_data.dat
    snprintf(filename, sizeof(filename), "data/all_particle_data%s.dat", file_suffix);
    fp = fopen(filename, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fclose(fp);
        int snapshots = file_size / bytes_per_snapshot_apd;
        printf("  all_particle_data: %d complete snapshots\n", snapshots);
        if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
    }

    // Check all_particle_ids.dat
    snprintf(filename, sizeof(filename), "data/all_particle_ids%s.dat", file_suffix);
    fp = fopen(filename, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fclose(fp);
        int snapshots = file_size / bytes_per_snapshot_ids;
        printf("  all_particle_ids: %d complete snapshots\n", snapshots);
        if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
    }

    // Check all_particle_phi.dat
    snprintf(filename, sizeof(filename), "data/all_particle_phi%s.dat", file_suffix);
    fp = fopen(filename, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fclose(fp);
        int snapshots = file_size / bytes_per_snapshot_phi;
        printf("  all_particle_phi: %d complete snapshots\n", snapshots);
        if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
    }

    // Check all_particle_scatter_counts.dat
    snprintf(filename, sizeof(filename), "data/all_particle_scatter_counts%s.dat", file_suffix);
    fp = fopen(filename, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fclose(fp);
        int snapshots = file_size / bytes_per_snapshot_scatter;
        printf("  all_particle_scatter_counts: %d complete snapshots\n", snapshots);
        if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
    }

    // Check trajectory files (convert timesteps to snapshots)
    snprintf(filename, sizeof(filename), "data/trajectories%s.dat", file_suffix);
    fp = fopen(filename, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fclose(fp);
        int timesteps = file_size / bytes_per_timestep_traj;
        int snapshots = timesteps / dtwrite;  // Convert timesteps to snapshot index
        printf("  trajectories: %d timesteps = %d complete snapshots\n", timesteps, snapshots);
        if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
    }

    snprintf(filename, sizeof(filename), "data/single_trajectory%s.dat", file_suffix);
    fp = fopen(filename, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fclose(fp);
        int timesteps = file_size / bytes_per_timestep_single;
        int snapshots = timesteps / dtwrite;
        printf("  single_trajectory: %d timesteps = %d complete snapshots\n", timesteps, snapshots);
        if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
    }

    snprintf(filename, sizeof(filename), "data/energy_and_angular_momentum_vs_time%s.dat", file_suffix);
    fp = fopen(filename, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_size = ftell(fp);
        fclose(fp);
        int timesteps = file_size / bytes_per_timestep_energy;
        int snapshots = timesteps / dtwrite;
        printf("  energy_and_angular_momentum: %d timesteps = %d complete snapshots\n", timesteps, snapshots);
        if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
    }

    if (nlowest > 0) {
        const char* traj_base = (g_l_target_value >= 0.0) ? "data/chosen_l_trajectories" : "data/lowest_l_trajectories";
        snprintf(filename, sizeof(filename), "%s%s.dat", traj_base, file_suffix);
        fp = fopen(filename, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            file_size = ftell(fp);
            fclose(fp);
            int timesteps = file_size / bytes_per_timestep_lowestL;
            int snapshots = timesteps / dtwrite;
            const char* traj_name = (g_l_target_value >= 0.0) ? "chosen_l_trajectories" : "lowest_l_trajectories";
            printf("  %s: %d timesteps = %d complete snapshots\n", traj_name, timesteps, snapshots);
            if (snapshots < min_complete_snapshot) min_complete_snapshot = snapshots;
        }
    }

    // Subtract 1 to get the last complete snapshot index (0-based)
    if (min_complete_snapshot == INT_MAX) {
        return -1;  // No files found
    }

    int last_complete = min_complete_snapshot - 1;
    printf("\nLast common complete snapshot: %d (timestep %d)\n", last_complete, last_complete * dtwrite);

    return last_complete;
}

/**
 * @brief Truncate all output files to the last common complete snapshot.
 * @details Ensures all files are consistent for restart by removing partial data.
 *
 * @param last_snapshot The last complete snapshot to keep (0-based)
 * @param npts Number of particles
 * @param dtwrite Timestep interval for writes
 * @param num_traj_particles Number of trajectory particles
 * @param nlowest Number of lowest-L particles
 * @param file_suffix The file suffix string
 * @return 0 on success, -1 on error
 */
static int truncate_files_to_snapshot(int last_snapshot, int npts, int dtwrite,
                                      int num_traj_particles, int nlowest,
                                      const char* file_suffix) {
    char filename[512];
    FILE *fp;
    int snapshots_to_keep = last_snapshot + 1;  // Convert 0-based to count
    int timesteps_to_keep = last_snapshot * dtwrite + 1;  // Include timestep at snapshot boundary to prevent duplicate write

    printf("Truncating files to snapshot %d (keeping %d snapshots, %d timesteps)...\n",
           last_snapshot, snapshots_to_keep, timesteps_to_keep);

    // Truncate snapshot-based files
    struct {
        const char* name;
        long target_size;
    } snapshot_files[] = {
        {"data/all_particle_data", (long)snapshots_to_keep * npts * 16},
        {"data/all_particle_ids", (long)snapshots_to_keep * npts * 4},
        {"data/all_particle_phi", (long)snapshots_to_keep * npts * 4},
        {"data/all_particle_scatter_counts", (long)snapshots_to_keep * npts * 4},
        {NULL, 0}
    };

    for (int i = 0; snapshot_files[i].name; i++) {
        snprintf(filename, sizeof(filename), "%s%s.dat", snapshot_files[i].name, file_suffix);
        fp = fopen(filename, "r+b");
        if (fp) {
            if (ftruncate(fileno(fp), snapshot_files[i].target_size) != 0) {
                fprintf(stderr, "ERROR: Failed to truncate %s\n", filename);
                fclose(fp);
                return -1;
            }
            fclose(fp);
            printf("  Truncated %s to %ld bytes\n", filename, snapshot_files[i].target_size);
        }
    }

    // Truncate trajectory files (based on timesteps)
    struct {
        const char* name;
        long bytes_per_timestep;
    } trajectory_files[] = {
        {"data/trajectories", 4 + num_traj_particles * 12},
        {"data/single_trajectory", 16},
        {"data/energy_and_angular_momentum_vs_time", 4 + num_traj_particles * 16},
        {(g_l_target_value >= 0.0) ? "data/chosen_l_trajectories" : "data/lowest_l_trajectories", (nlowest > 0) ? (4 + nlowest * 12) : 0},
        {NULL, 0}
    };

    for (int i = 0; trajectory_files[i].name; i++) {
        if (trajectory_files[i].bytes_per_timestep == 0) continue;  // Skip if nlowest == 0

        snprintf(filename, sizeof(filename), "%s%s.dat", trajectory_files[i].name, file_suffix);
        fp = fopen(filename, "r+b");
        if (fp) {
            long target_size = timesteps_to_keep * trajectory_files[i].bytes_per_timestep;
            if (ftruncate(fileno(fp), target_size) != 0) {
                fprintf(stderr, "ERROR: Failed to truncate %s\n", filename);
                fclose(fp);
                return -1;
            }
            fclose(fp);
            printf("  Truncated %s to %ld bytes (%d timesteps)\n",
                   filename, target_size, timesteps_to_keep);
        }
    }

    printf("File truncation complete.\n\n");
    return 0;
}

/**
 * @def CLEAN_LOCAL_EXIT(code)
 * @brief Non-thread-safe exit macro for cleanup and termination.
 *
 * @details Performs resource cleanup and exits in single-threaded contexts. This
 *          macro calls `cleanup_all_particle_data()` before exiting with the
 *          specified code.
 *
 * @warning This macro is not thread-safe. Use it only in single-threaded
 *          sections of the code. For thread-safe exits, use `CLEAN_EXIT`.
 *
 * @param code The exit code for the program.
 *
 * @par Example
 * @code
 * if (error_condition) {
 *     CLEAN_LOCAL_EXIT(1);
 * }
 * @endcode
 */
#define CLEAN_LOCAL_EXIT(code)       \
    do                               \
    {                                \
        cleanup_all_particle_data(); \
        exit(code);                  \
    } while (0)

/**
 * @brief Gravitational force calculation control flag.
 * @details Used for testing and debugging orbital dynamics:
 *          - 0 = Normal gravitational force calculation (default).
 *          - 1 = Zero gravity (particles move in straight lines).
 *
 * @note Setting this to 1 is useful for validating the integration scheme
 *       independent of gravitational physics.
 */
static int use_identity_gravity = 0;

/**
 * @brief Calculates gravitational acceleration at a given radius.
 *
 * @param r [in] Radius in kpc.
 * @param current_rank [in] Particle rank (0 to npts-1), used for \f$M(r)\f$ approximation.
 * @param npts [in] Total number of particles.
 * @param G_value [in] Gravitational constant value (e.g., G_CONST).
 * @param halo_mass_value [in] Total halo mass (e.g., HALO_MASS).
 * @return Gravitational acceleration (force per unit mass) in simulation units (kpc/Myr\f$^2\f$).
 *
 * @note Returns 0.0 if `use_identity_gravity` is set to 1.
 * @note \f$M(r)\f$ is approximated as `(current_rank / npts) * halo_mass_value`.
 */
static inline double gravitational_force(double r, int current_rank, int npts, double G_value, double halo_mass_value)
{
    if (use_identity_gravity)
    {
        // Testing mode: no gravitational force
        return 0.0;
    }
    else
    {
        // Calculate gravitational force: F = -G * M(r) / r²
        // where M(r) is proportional to particle rank
        return -(VEL_CONV_SQ * G_value) * ((double)current_rank / (double)npts) * halo_mass_value / (r * r);
    }
}

/**
 * @brief Computes the effective centrifugal acceleration due to angular momentum.
 *
 * @param r [in] Radius (kpc).
 * @param ell [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr).
 * @return Centrifugal acceleration: \f$L^2/r^3\f$ (kpc/Myr\f$^2\f$).
 */
static inline double effective_angular_force(double r, double ell)
{
    return (ell * ell) / (r * r * r);
}

/**
 * @brief Alternative gravitational force calculation using transformed coordinates.
 * @details Used in the Levi-Civita regularization scheme. Calculates F/m in rho coordinates.
 *
 * @param rho [in] Transformed radial coordinate (sqrt(r)). Units: sqrt(kpc).
 * @param current_rank [in] Particle rank (0 to npts-1).
 * @param npts [in] Total number of particles.
 * @param G_value [in] Gravitational constant value (e.g., G_CONST).
 * @param halo_mass_value [in] Total halo mass (e.g., HALO_MASS).
 * @return \f$dv/d\tau\f$ in Levi-Civita coordinates (kpc\f$^2\f$/Myr\f$^2\f$).
 *
 * @note Returns 0.0 if `use_identity_gravity` is set to 1.
 * @see gravitational_force
 * @see doLeviCivitaLeapfrog
 */
static inline double gravitational_force_rho_v(double rho, int current_rank, int npts, double G_value, double halo_mass_value)
{
    if (use_identity_gravity)
    {
        // Testing mode: no gravitational force
        return 0.0;
    }
    else
    {
        // Gravitational force in transformed coordinates
        return -(VEL_CONV_SQ * G_value) * ((double)current_rank / (double)npts) * halo_mass_value / (rho * rho);
    }
}

/**
 * @brief Computes effective centrifugal acceleration in transformed coordinates.
 * @details Used in the Levi-Civita regularization scheme. Calculates \f$L^2/r^3\f$ in rho coordinates.
 *
 * @param rho [in] Transformed radial coordinate (sqrt(r)). Units: sqrt(kpc).
 * @param ell [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr).
 * @return \f$dv/d\tau\f$ centrifugal term in Levi-Civita coordinates (kpc\f$^2\f$/Myr\f$^2\f$).
 *
 * @see effective_angular_force
 * @see doLeviCivitaLeapfrog
 */
static inline double effective_angular_force_rho_v(double rho, double ell)
{
    return (ell * ell) / (rho * rho * rho * rho);
}

// =============================================================================
// Energy Debugging and Validation Subsystem
// =============================================================================

#define DEBUG_PARTICLE_ID 4    ///< Particle ID (by rank) to track for energy debugging.
#define DEBUG_MAX_STEPS 100000 ///< Maximum number of debug energy snapshots.

/** @brief Arrays for tracking energy components through simulation for debugging. */
static double dbg_approxE[DEBUG_MAX_STEPS]; ///< Theoretical model energy (per unit mass).
static double dbg_dynE[DEBUG_MAX_STEPS];    ///< Actual dynamical energy (per unit mass).
static double dbg_kinE[DEBUG_MAX_STEPS];    ///< Kinetic energy component (per unit mass).
static double dbg_potE[DEBUG_MAX_STEPS];    ///< Potential energy component (per unit mass).
static double dbg_time[DEBUG_MAX_STEPS];    ///< Simulation time at each snapshot (Myr).
static double dbg_radius[DEBUG_MAX_STEPS];  ///< Particle radius at each snapshot (kpc).
static int dbg_count = 0;                   ///< Tracks the number of recorded debug snapshots.

/** @brief Arrays for tracking total system energy diagnostics over time. */
static double *g_time_snapshots = NULL; ///< Simulation time at each diagnostic snapshot (Myr).
static double *g_total_KE = NULL;       ///< Total kinetic energy of the system at each snapshot.
static double *g_total_PE = NULL;       ///< Total potential energy of the system at each snapshot.
static double *g_total_E = NULL;        ///< Total energy (KE + PE) of the system at each snapshot.
static int g_energy_snapshots_loaded = 0; ///< Number of energy snapshots loaded from existing file in restart mode.

// Forward declaration of energy calculation function
static void calculate_system_energies(double** particles, int npts, double deltaM,
                                      double* total_KE_out, double* total_PE_out);

/**
 * @brief Records the theoretical model energy for a debug snapshot.
 *
 * @param snapIndex [in] Index of the snapshot (0 to DEBUG_MAX_STEPS - 1).
 * @param E_value [in] Theoretical energy value (per unit mass).
 * @param time_val [in] Simulation time (Myr).
 *
 * @note Updates `dbg_count` if `snapIndex` extends the recorded range.
 * @see store_debug_dynE_components
 * @see finalize_debug_energy_output
 */
static void store_debug_approxE(int snapIndex, double E_value, double time_val)
{
    if (snapIndex < 0 || snapIndex >= DEBUG_MAX_STEPS)
        return;

    dbg_approxE[snapIndex] = E_value;
    dbg_time[snapIndex] = time_val;

    // Update the total count if needed
    if (snapIndex >= dbg_count)
    {
        dbg_count = snapIndex + 1;
    }
}

/**
 * @brief Records the actual dynamical energy and its components for a debug snapshot.
 *
 * @param snapIndex [in] Index of the snapshot (0 to DEBUG_MAX_STEPS - 1).
 * @param totalE [in] Total energy (KE + PE) per unit mass.
 * @param kinE [in] Kinetic energy component (per unit mass).
 * @param potE [in] Potential energy component (per unit mass).
 * @param time_val [in] Simulation time (Myr).
 * @param radius_val [in] Particle radius (kpc).
 *
 * @note Updates `dbg_count` if `snapIndex` extends the recorded range.
 * @see store_debug_approxE
 * @see finalize_debug_energy_output
 */
static void store_debug_dynE_components(int snapIndex,
                                        double totalE,
                                        double kinE,
                                        double potE,
                                        double time_val,
                                        double radius_val)
{
    if (snapIndex < 0 || snapIndex >= DEBUG_MAX_STEPS)
        return;

    dbg_dynE[snapIndex] = totalE;
    dbg_kinE[snapIndex] = kinE;
    dbg_potE[snapIndex] = potE;
    dbg_time[snapIndex] = time_val;
    dbg_radius[snapIndex] = radius_val;

    // Update the total count if needed
    if (snapIndex >= dbg_count)
    {
        dbg_count = snapIndex + 1;
    }
}

/**
 * @brief Writes collected debug energy data to a file.
 * @details Writes the time evolution of theoretical vs. dynamic energy for the
 *          tracked particle (`DEBUG_PARTICLE_ID`) to `data/debug_energy_compare.dat`.
 *          The output includes kinetic and potential energy components. This function
 *          is called at the end of the simulation if `g_doDebug` is enabled.
 *
 * @note File writing is suppressed if `skip_file_writes` is non-zero.
 *
 * @see store_debug_approxE
 * @see store_debug_dynE_components
 */

/**
 * @brief Loads existing energy diagnostic data from a file for restart/extend operations.
 * @details Reads energy data from an existing total_energy_vs_time file to populate
 *          the global energy arrays. This ensures continuity when restarting simulations.
 *
 * @param filename [in] Path to the existing energy file
 * @param max_snapshots [in] Maximum number of snapshots to load
 * @return Number of snapshots successfully loaded
 */
static int load_existing_energy_diagnostics(const char* filename, int max_snapshots)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return 0;  // File does not exist; return 0 snapshots loaded
    }
    
    // Read the binary file structure: 5 floats per row (time, KE, PE, total_E, frac_change)
    int snapshots_loaded = 0;
    
    // The file format, written by fprintf_bin, stores floating-point numbers as binary `float` types.
    while (snapshots_loaded < max_snapshots) {
        float values[5];
        size_t items_read = fread(values, sizeof(float), 5, fp);
        
        if (items_read != 5) {
            // End of file or error
            break;
        }
        
        // Store in the global arrays (converting float to double)
        g_time_snapshots[snapshots_loaded] = (double)values[0];
        g_total_KE[snapshots_loaded] = (double)values[1];
        g_total_PE[snapshots_loaded] = (double)values[2];
        g_total_E[snapshots_loaded] = (double)values[3];
        // The fractional change (index 4) is ignored; it is recalculated when writing the final file.
        
        snapshots_loaded++;
    }
    
    fclose(fp);
    log_message("INFO", "Loaded %d energy snapshots from %s", snapshots_loaded, filename);
    return snapshots_loaded;
}

/**
 * @brief Writes the collected total system energy diagnostics to a file.
 * @details This function is called at the end of the simulation. It iterates through
 *          the global snapshot arrays and writes the time evolution of the system's
 *          total kinetic, potential, and combined energy to a file named
 *          `data/total_energy_vs_time<suffix>.dat`.
 *
 * @param noutsnaps [in] The total number of snapshots recorded.
 *
 * @note The file writing operation is skipped if the simulation is in restart mode
 *       and file writes are disabled (`skip_file_writes` is true).
 */
static void finalize_energy_diagnostics(int noutsnaps)
{
    if (skip_file_writes) {
        log_message("INFO", "Skipped writing total energy diagnostic file due to restart/skip_file_writes flag.");
        return;
    }

    char filename[256];
    get_suffixed_filename("data/total_energy_vs_time.dat", 1, filename, sizeof(filename));

    // Determine whether to append or write from scratch
    const char *mode;
    int start_index;
    
    if ((g_restart_mode_active || g_doSimExtend) && g_energy_snapshots_loaded > 0) {
        // Existing data loaded; append only new snapshots
        mode = "ab";  // Append binary mode
        start_index = g_energy_snapshots_loaded;
        log_message("INFO", "Appending energy diagnostics starting from snapshot %d", start_index);
    } else {
        // Fresh run or no existing data loaded - write complete file
        mode = "wb";  // Write binary mode
        start_index = 0;
    }

    FILE *fp = fopen(filename, mode);
    if (!fp) {
        log_message("ERROR", "Cannot open %s for writing total energy data (mode: %s).", filename, mode);
        fprintf(stderr, "Error: cannot open %s\n", filename);
        return;
    }

    // Only write header for new files (not append mode)
    if (strcmp(mode, "wb") == 0) {
        fprintf_bin(fp, "# Time(Myr)   Total_KE   Total_PE   Total_Energy   (E-E_init)/|E_init|\n");
    }

    double E_init = g_total_E[0];

    // Write only the new snapshots (from start_index to noutsnaps)
    for (int i = start_index; i < noutsnaps; i++) {
        double E_current = g_total_E[i];
        double E_frac_change = (fabs(E_init) > 1e-30) ? (E_current - E_init) / fabs(E_init) : 0.0;
        
        fprintf_bin(fp, "%.6f  %.8g  %.8g  %.8g  %.8g\n",
                    g_time_snapshots[i],
                    g_total_KE[i],
                    g_total_PE[i],
                    E_current,
                    E_frac_change);
    }
    fclose(fp);
    
    int lines_written = noutsnaps - start_index;
    if (strcmp(mode, "ab") == 0) {
        log_message("INFO", "Appended %d new energy snapshots to %s (total: %d)", 
                    lines_written, filename, noutsnaps);
    } else {
        log_message("INFO", "Wrote total energy diagnostics to %s with %d lines.", filename, noutsnaps);
    }
}

/**
 * @brief Writes collected debug energy data to debug_energy_compare.dat.
 * @details Outputs time evolution of theoretical vs dynamical energy for the
 *          tracked particle (DEBUG_PARTICLE_ID), including KE and PE components.
 *          Called at simulation end if g_doDebug is enabled.
 */
static void finalize_debug_energy_output(void)
{
    // Skip file writes if in restart mode and file writes are skipped
    if (!skip_file_writes)
    {
        // Create filename with suffix
        char filename[256];
        get_suffixed_filename("data/debug_energy_compare.dat", 1, filename, sizeof(filename));

        FILE *fp = fopen(filename, "wb"); // Binary mode for fprintf_bin
        if (!fp)
        {
            // Use log_message for consistency if logging is enabled
            log_message("ERROR", "Cannot open %s for writing debug energy data.", filename);
            // Fallback to printf for error visibility if logging unavailable
            printf("Error: cannot open %s\n", filename);
            return;
        }

        /* Output 8 columns, including KE & PE. */
        fprintf_bin(fp,
                    "# snapIdx  time(Myr)  radius(kpc)   approxE   dynE   (dyn-approx)    KE       PE\n");

        for (int i = 0; i < dbg_count; i++)
        {
            double eA = dbg_approxE[i];
            double eD = dbg_dynE[i];
            double ke = dbg_kinE[i];
            double pe = dbg_potE[i];
            double tVal = dbg_time[i];
            double rVal = dbg_radius[i];

            double dff = eD - eA;
            fprintf_bin(fp, "%4d  %.6f  %.6f  %.8g  %.8g  %.8g  %.8g  %.8g\n",
                        i, tVal, rVal, eA, eD, dff, ke, pe);
        }
        fclose(fp);

        // Only log the debug message, never print to console
        log_message("DEBUG", "Wrote %s with %d lines.", filename, dbg_count);
    }
    else
    {
        log_message("INFO", "Skipped writing debug energy file due to restart/skip_file_writes flag.");
    }
}

static double normalization; ///< Mass normalization factor for energy calculations. Calculated based on the integral of the density profile.

// =============================================================================
// ENERGY CALCULATION AND INTEGRATION STRUCTURES
// =============================================================================

/**
 * @brief Parameters for energy integration calculations.
 * @details Used as `void* params` argument in GSL integration routines,
 *          specifically for distribution function calculations (`fEintegrand`).
 */
typedef struct
{
    double E;                      ///< Energy value \f$E\f$ for isotropic or \f$Q\f$ value for OM.
    gsl_spline *splinePsi;         ///< Interpolation spline for potential \f$\Psi(r)\f$.
    gsl_spline *splinemass;        ///< Interpolation spline for enclosed mass \f$M(r)\f$.
    gsl_interp_accel *rofPsiarray; ///< Accelerator for radius lookup from potential \f$r(\Psi)\f$.
    gsl_interp_accel *massarray;   ///< Accelerator for mass lookups \f$M(r)\f$.
    int use_om;                    ///< Flag: 1 if using OM (E represents Q), 0 if isotropic (E represents energy).
} fEintegrand_params;

/**
 * @brief Parameters for Psiintegrand to support profile-specific mass integrands.
 * @details Allows Psiintegrand to call the appropriate mass integrand function
 *          based on the selected density profile.
 */
typedef struct {
    double (*massintegrand_func)(double, void *); ///< Function pointer to profile-specific mass integrand
    void *params_for_massintegrand;               ///< Parameters for the mass integrand function
} Psiintegrand_params;

/**
 * @brief Structure to track angular momentum with particle index and direction.
 * @details Used in sorting and selection of particles by angular momentum,
 *          especially when finding particles closest to a reference \f$L\f$.
 */
typedef struct
{
    double L; ///< Angular momentum value (or squared difference from Lcompare).
    int idx;  ///< Original particle index (before sorting by L).
    int sign; ///< Direction indicator (+1 or -1) or sign of (L - Lcompare).
} LAndIndex;

/**
 * @brief Comparison function for sorting double values in ascending order.
 * @details This function is designed to be used with `qsort` or other
 *          standard library sorting functions that require a comparator.
 *          It takes two void pointers, casts them to `const double*`,
 *          dereferences them, and compares their values.
 *
 * @param a [in] Pointer to the first double value.
 * @param b [in] Pointer to the second double value.
 * @return int -1 if the first double is less than the second,
 *              1 if the first double is greater than the second,
 *              0 if they are equal.
 */
int double_cmp(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;

    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

// =============================================================================
// PARTICLE SORTING AND ORDERING OPERATIONS
// =============================================================================

int compare_particles(const void *a, const void *b);

void sort_particles(double **particles, int npts);

void sort_particles_with_alg(double **particles, int npts, const char *sortAlg);

#if defined(__APPLE__) && defined(GPU_SORT_AVAILABLE)
/// Sort particles by radius using GPU radix sort (Metal).
/// Casts double precision data to float32 for GPU efficiency.
/// \param columns_to_sort_on Array of column pointers to sort (r, vr, etc)
/// \param npts Number of particles to sort
void gpu_radix_sort(double **columns_to_sort_on, int npts);

/// Automatically choose between CPU and GPU sorting based on particle count.
/// Uses GPU for N > threshold (~500k), CPU otherwise.
/// \param columns_to_sort_on Array of column pointers to sort (r, vr, etc)
/// \param npts Number of particles to sort
void hybrid_auto_sort(double **columns_to_sort_on, int npts);
#endif

void sort_rr_psi_arrays(double *rrA_spline, double *psiAarr_spline, int npts);

// =============================================================================
// PHYSICS CALCULATION FUNCTIONS
// =============================================================================

// Forward declarations for profile-specific functions
double massintegrand(double r, void *params);
double massintegrand_profile_nfwcutoff(double r, void *params);
double drhodr_profile_nfwcutoff(double r, double rc, double nt_nfw, double falloff_C_param);
double fEintegrand_nfw(double t_integration_var, void *params);
double fEintegrand_hernquist(double t_integration_var, void *params);
double om_mu_integrand(double mu, void *params);
double massintegrand_hernquist(double r, void *params);
double drhodr_hernquist(double r, double a_scale, double M_total);
double drhodr_om_hernquist(double r, double a_scale, double M_total);
double drhodr(double r);
double density_derivative_om_cored(double r);
double density_derivative_om_nfw(double r, double rc_param, double nt_nfw_scaler, double falloff_C_param);

// Forward declarations
double Psiintegrand(double rp, void *params);
double evaluatespline(gsl_spline *spline, gsl_interp_accel *acc, double value);
double fEintegrand(double t, void *params);

int cmp_LAI(const void *a, const void *b);

/**
 * @brief Checks if a given string represents a valid integer.
 * @details Allows an optional leading '+' or '-' sign. Validates that all
 *          subsequent characters in the string are digits. Returns 0 (false)
 *          for empty strings, strings containing only a sign, or strings with
 *          non-digit characters after the optional sign.
 *
 * @param str [in] The null-terminated string to check.
 * @return int 1 if the string is a valid integer, 0 otherwise.
 */
static int isInteger(const char *str)
{
    if (*str == '-' || *str == '+')
    {
        str++;
    }
    if (!*str)
    {
        return 0;
    }
    while (*str)
    {
        if (!isdigit((unsigned char)*str))
        {
            return 0;
        }
        str++;
    }
    return 1;
}

/**
 * @brief Checks if a given string represents a valid floating-point number.
 * @details Validates if the input string conforms to common floating-point number
 *          formats, including an optional leading sign ('+' or '-'), digits,
 *          at most one decimal point (if not in exponent part), and an optional
 *          exponent part (e.g., "e+10", "E-5").
 *          The function requires at least one digit to be present for a number to be
 *          considered valid (e.g., "." or "+." are not valid floats).
 *
 * @param str [in] The null-terminated string to check.
 * @return int 1 if the string is a valid float, 0 otherwise.
 */
static int isFloat(const char *str)
{
    if (*str == '-' || *str == '+')
    {
        str++;
    }
    if (!*str)
    {
        return 0;
    }

    int has_digit = 0;
    int has_decimal = 0;
    int has_exponent = 0;

    while (*str)
    {
        if (isdigit((unsigned char)*str))
        {
            has_digit = 1;
        }
        else if (*str == '.' && !has_decimal && !has_exponent)
        {
            has_decimal = 1;
        }
        else if ((*str == 'e' || *str == 'E') && !has_exponent && has_digit)
        {
            has_exponent = 1;
            str++;
            // Check for optional sign after exponent
            if (*str == '-' || *str == '+')
            {
                str++;
            }
            if (!*str || !isdigit((unsigned char)*str))
            {
                return 0; // Exponent must have at least one digit
            }
            // Preserve has_digit - valid digits exist before exponent
        }
        else
        {
            return 0;
        }
        str++;
    }

    return has_digit;
}

// =============================================================================
// COMMAND LINE ARGUMENT PROCESSING
// =============================================================================

/**
 * @brief Displays detailed usage information for command-line arguments.
 * @details Prints a comprehensive help message to stderr showing all available
 *          command-line options, their default values, and brief descriptions.
 *          Includes information about integration methods, sorting algorithms,
 *          data saving modes, and basic usage examples.
 *
 * @param prog [in] The program name to display in the usage message (typically argv[0]).
 *
 * @note Called when the user specifies `--help` or when errors occur during
 *       argument parsing.
 */
static void printUsage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --help                        Show this usage message.\n"
            "  --log                         [Default Off] Enable writing detailed logs to log/nsphere.log\n"
            "  --tag <string>                [Default Off] Add a custom tag to output filenames (e.g. \"run1\")\n"
            "  --save <subargs>              [Default all] Enable various data-saving modes (may combine any):\n"
            "                                          all           => output everything\n"
            "                                          raw-data      => only raw particle data\n"
            "                                          psi-snaps     => plus Psi snapshots\n"
            "                                          full-snaps    => plus full data snapshots\n"
            "                                          debug-energy  => plus energy diagnostics\n"
            "                                          If multiple subargs are given, the highest priority\n"
            "                                          one overrides the lower: raw-data < psi-snaps <\n"
            "                                          full-snaps < all/debug-energy.\n"
            "  --restart [force]             [Default Off] Restart processing from the last written snapshot\n"
            "                                               With 'force': regenerate ALL snapshots even if they exist\n"
            "  --sim-restart [check]         [Default Off] Restart incomplete simulation from last checkpoint\n"
            "                                               With 'check': only report status without restarting\n"
            "  --restart-file <file>         [Default Off] Specify explicit file for restart (debugging)\n"
            "  --sim-extend                  [Default Off] Extend a completed simulation to new Ntimes/tfinal\n"
            "                                               Requires --extend-file to specify source file\n"
            "  --extend-file <file>          [Default Off] Source file to extend (used with --sim-extend)\n"
            "                                               File will be copied to new name based on parameters\n"
            "\n"
            "  --nparticles <int>            [Default 100000] Number of particles\n"
            "  --ntimesteps <int>            [Default 10000] Requested total timesteps\n"
            "                                     Note: Ntimes will be adjusted to the minimum value that\n"
            "                                     satisfies the constraint (Ntimes - 1) = k*(dtwrite)*(nout)\n"
            "  --dtwrite <int>               [Default 100] Low level diskwrite interval in timesteps\n"
            "  --snapshot-buffer <int>       [Default 100] Number of snapshots to buffer before writing to disk\n"
            "  --nout <int>                  [Default 100] Number of post-processing output data snapshot times\n"
            "  --tfinal <int>                [Default 5] Final simulation time in units of the dynamical time\n"
            "  --lvals-target <float>        [Default Off] Output L values with values closest to target\n"
            "\n"
            "  --readinit <file>             [Default Off] Read initial conditions from <file> (binary)\n"
            "  --writeinit <file>            [Default Off] Write initial conditions to <file> (binary)\n"
            "  --master-seed <int>           [Default Random] Set a master seed to derive other seeds.\n"
            "  --load-seeds                  [Default Off] Load seeds from previous run's output files\n"
            "  --init-cond-seed <int>        [Default Random/Master] Set seed for IC generation.\n"
            "                                     Overrides derivation from master-seed.\n"
            "\n"
            "  --method <int>                [Default 1] Integration method (1..9):\n"
            "                                          1   Adaptive Leapfrog with Adaptive Levi-Civita\n"
            "                                          2   Full-step adaptive Leapfrog + Levi-Civita\n"
            "                                          3   Full-step adaptive Leapfrog\n"
            "                                          4   Yoshida 4th-order\n"
            "                                          5   Adams-Bashforth 3\n"
            "                                          6   Leapfrog (vel half-step)\n"
            "                                          7   Leapfrog (pos half-step)\n"
            "                                          8   Classic RK4\n"
            "                                          9   Euler\n"
            "  --methodtag                   [Default Off] Include method string in output filenames\n"
            "  --sort <int>                  [Default 1] Sorting algorithm (1..6):\n"
            "                                          1   Parallel Quadsort\n"
            "                                          2   Sequential Quadsort\n"
            "                                          3   Parallel Insertion Sort\n"
            "                                          4   Sequential Insertion Sort\n"
            "                                          5   Parallel Radix Sort\n"
            "                                          6   Adaptive sort (benchmarks & switches algorithms)\n"
            "\n"
            "  --halo-mass <float>           [Default 1.15e9] Total halo mass in M☉ for the selected profile.\n"
            "  --profile <type>              [Default nfw] Profile type for ICs: 'nfw', 'cored', or 'hernquist'.\n"
            "  --aniso-beta <float>          [Default 0.0] Constant anisotropy parameter β for Hernquist profile (-1 ≤ β ≤ 0.5).\n"
            "  --aniso-factor <float>        [Default Off] Osipkov-Merritt anisotropy radius in units of scale radius.\n"
            "                                     Enables OM model with β(r) = r²/(r² + r_a²). Compatible with all profiles.\n"
            "  --aniso-betascale <float>     [Default Off] Alternative to --aniso-factor: specify β at the scale radius.\n"
            "                                     Sets r_a/r_s = √(1/β_s - 1). Range: (0, 1). Cannot use with --aniso-factor.\n"
            "  --scale-radius <float>        [Default 23] Scale radius in kpc for the selected profile.\n"
            "  --cutoff-factor <float>       [Default 85.0] Absolute r_max in units of scale radius.\n"
            "  --falloff-factor <float>      [Default 19.0] NFW concentration parameter transition factor.\n"
            "  --ftidal <float>              [Default 0.0] Set tidal fraction outer stripping value (0.0 to 1.0)\n"
            "\n"
            "  --sidm                        [Default Off] Enable self-interacting scattering physics\n"
            "  --sidm-seed <int>             [Default Random/Master] Set seed for SIDM calculations.\n"
            "                                     Overrides derivation from master-seed.\n"
            "  --sidm-mode <serial|parallel> [Default parallel] Select SIDM execution mode.\n"
            "                                     Parallel mode requires OpenMP.\n"
            "  --sidm-kappa <float>          [Default 50.0] SIDM opacity kappa in cm^2/g.\n"
            "\n"
            "Example:\n"
            "  %s --nparticles 50000 --ntimesteps 20000 --tfinal 5 \\\n"
            "     --nout 100 --dtwrite 10 --method 6 --sort 2 --readinit initial_conditions.bin\n",
            prog, prog);
}

/**
 * @brief Displays an error message, suggests `--help`, and terminates the program.
 * @details Formats and prints an error message to stderr, includes a suggestion
 *          to use the `--help` flag for usage information, performs necessary
 *          cleanup of allocated resources, and then exits with a non-zero status.
 *
 * @param msg [in] The error message to display.
 * @param arg [in] Optional argument value that caused the error (shown in quotes), or NULL to omit.
 * @param prog [in] The program name to display in the --help usage suggestion.
 *
 * @note Calls cleanup_all_particle_data() before exiting to free allocated memory.
 * @warning This function does not return; execution is terminated.
 */
static void errorAndExit(const char *msg, const char *arg, const char *prog)
{
    if (arg != NULL)
    {
        fprintf(stderr, "Error: %s '%s'\n", msg, arg);
    }
    else
    {
        fprintf(stderr, "Error: %s\n", msg);
    }
    fprintf(stderr, "Use '%s --help' for usage information.\n", prog ? prog : "./nsphere");
    cleanup_all_particle_data();
    exit(1);
}


// =============================================================================
// PARTICLE DATA STRUCTURES AND OPERATIONS
// =============================================================================

/**
 * @brief Compact data structure for particle properties used in sorting and analysis.
 * @details Contains essential physical properties (radial position, velocity, 
 *          angular momentum) and tracking metadata (rank, original index) for each
 *          particle in the simulation. Used extensively for sorting, file I/O, and
 *          data analysis operations.
 * 
 * @note Uses compact `float` types for physical quantities to reduce memory usage
 *       when processing large particle counts.
 * @note The `rank` field is assigned during radial sorting, while `original_index`
 *       preserves the initial array position for tracking particles across snapshots.
 */
struct PartData
{
    int rank;           // Particle rank (sorted position)
    float rad;          // Radial position
    float vrad;         // Radial velocity
    float angmom;       // Angular momentum
    int original_index; // Original position in array before sorting
};

/**
 * Sorts particle data in ascending order by radial position
 *
 * Performs a quick sort of the particle data structure array,
 * using the radial position as the primary sorting key.
 * Preserves the original indices to allow tracking particles across time steps.
 *
 * @param array  Array of particle data structures to be sorted
 * @param npts   Number of particles in the array
 */
void sort_by_rad(struct PartData *array, int npts);

/**
 * @brief Appends a block of full particle data to the specified output file.
 * @details This function writes a chunk of particle data, corresponding to `block_size`
 *          timesteps, to the given binary file. The data for each particle (rank, radius,
 *          radial velocity, angular momentum) is written sequentially for each timestep
 *          within the block (step-major order). This means all particle data for step `s`
 *          is contiguous, followed by all data for step `s+1`, etc., within the block.
 *          The file is opened in append binary mode ("ab").
 *          This is used for creating the `all_particle_data.dat` file which stores
 *          the complete evolution history when `g_doAllParticleData` is enabled.
 *
 * @param filename   [in] Path to the output binary file (e.g., "data/all_particle_data<suffix>.dat").
 * @param npts       [in] Number of particles.
 * @param block_size [in] Number of timesteps of data contained in the provided `_block` arrays.
 * @param L_block    [in] Pointer to the block of angular momentum data (float array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @param Rank_block [in] Pointer to the block of particle rank data (int array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @param R_block    [in] Pointer to the block of radial position data (float array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @param Vrad_block [in] Pointer to the block of radial velocity data (float array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @note Exits via `CLEAN_EXIT(1)` if the file cannot be opened for appending.
 */
static void append_all_particle_data_chunk_to_file(const char *filename, int npts, int block_size,
                                                   float *L_block, int *Rank_block, float *R_block, float *Vrad_block)
{
    FILE *f = fopen(filename, "ab");
    if (!f)
    {
        printf("Error: cannot open %s for appending all_particle_data\n", filename);
        CLEAN_EXIT(1);
    }

    // Write particle data in step-major order
    for (int step = 0; step < block_size; step++)
    {
        for (int i = 0; i < npts; i++)
        {
            int rankval = Rank_block[step * npts + i];
            float rval = R_block[step * npts + i];
            float vval = Vrad_block[step * npts + i];
            float lval = L_block[step * npts + i];

            // Write particle data in fixed order
            fwrite(&rankval, sizeof(int), 1, f);
            fwrite(&rval, sizeof(float), 1, f);
            fwrite(&vval, sizeof(float), 1, f);
            fwrite(&lval, sizeof(float), 1, f);
        }
    }

    fclose(f);
}

/**
 * @brief Appends a block of phi angle data to the specified output file.
 * @details This function writes a chunk of phi data, corresponding to `block_size`
 *          timesteps, to the given binary file. The data is written sequentially for
 *          each timestep within the block (step-major order). The file is opened in
 *          append binary mode ("ab"). This is used for creating the corresponding
 *          `all_particle_data_phi.dat` file.
 *
 * @param filename   [in] Path to the output binary phi data file.
 * @param npts       [in] Number of particles.
 * @param block_size [in] Number of timesteps of data contained in `phi_block`.
 * @param phi_block  [in] Pointer to the block of phi data (float array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 */
static void append_all_particle_phi_data_chunk_to_file(const char *filename, int npts, int block_size,
                                                       float *phi_block)
{
    FILE *f = fopen(filename, "ab");
    if (!f)
    {
        printf("Error: cannot open %s for appending all_particle_data_phi\n", filename);
        CLEAN_EXIT(1);
    }

    // Write phi data in step-major order
    for (int step = 0; step < block_size; step++)
    {
        for (int i = 0; i < npts; i++)
        {
            float phival = phi_block[step * npts + i];
            fwrite(&phival, sizeof(float), 1, f);
        }
    }

    fclose(f);
}

/**
 * @brief Appends a block of particle scatter counts to the specified output file.
 * @details This function writes a chunk of integer data, corresponding to `block_size`
 *          timesteps, to the given binary file. Each integer represents the number of
 *          times a particle scattered in a given timestep.
 *
 * @param filename   [in] Path to the output binary scatter count data file.
 * @param npts       [in] Number of particles.
 * @param block_size [in] Number of timesteps of data contained in `scat_count_block`.
 * @param scat_count_block [in] Pointer to the block of scatter count data (int array).
 */
static void append_all_particle_scatter_counts_to_file(const char *filename, int npts, int block_size,
                                                       int *scat_count_block)
{
    FILE *f = fopen(filename, "ab");
    if (!f)
    {
        printf("Error: cannot open %s for appending all_particle_data_scatcount\n", filename);
        CLEAN_EXIT(1);
    }

    // Write scatter count data in step-major order
    for (int step = 0; step < block_size; step++)
    {
        for (int i = 0; i < npts; i++)
        {
            int count = scat_count_block[step * npts + i];
            fwrite(&count, sizeof(int), 1, f);
        }
    }

    fclose(f);
}

/**
 * @brief Appends a block of particle IDs to the specified output file.
 * @details This function writes a chunk of particle ID data, corresponding to `block_size`
 *          timesteps, to the given binary file. The data is written in step-major order
 *          to match the structure of all_particle_data.dat.
 *
 * @param filename   [in] Path to the output binary ID data file.
 * @param npts       [in] Number of particles.
 * @param block_size [in] Number of timesteps of data contained in `id_block`.
 * @param id_block   [in] Pointer to the block of particle ID data (int array).
 */
static void append_all_particle_ids_to_file(const char *filename, int npts, int block_size,
                                            int *id_block)
{
    FILE *f = fopen(filename, "ab");
    if (!f)
    {
        printf("Error: cannot open %s for appending all_particle_ids\n", filename);
        CLEAN_EXIT(1);
    }

    // Write ID data in step-major order
    for (int step = 0; step < block_size; step++)
    {
        for (int i = 0; i < npts; i++)
        {
            int id_val = id_block[step * npts + i];
            fwrite(&id_val, sizeof(int), 1, f);
        }
    }

    fclose(f);
}

/**
 * @brief Retrieves particle phi angles for a specific snapshot from the `all_particle_phi.dat` binary file.
 * @details This function reads the phi angle data for all `npts` particles corresponding to
 *          a single snapshot number (`snap`) from the specified binary file. The file is expected
 *          to be in step-major order, where each record per particle consists of a single float (phi).
 *          It calculates the correct file offset to seek to the desired snapshot.
 *          To ensure thread safety when called in parallel (e.g., during post-processing
 *          of snapshots), file I/O (seeking and reading) is performed within an
 *          OpenMP critical section named `file_access_phi`. Temporary local buffers are used
 *          for reading, and data is then copied to the caller-provided output array.
 *
 * @param filename    [in] Path to the binary phi data file (e.g., "data/all_particle_phi<suffix>.dat").
 * @param snap        [in] The snapshot number (0-indexed, corresponding to write events) to retrieve.
 * @param npts        [in] Number of particles per snapshot.
 * @param block_size  [in] Number of snapshots written per block to file.
 * @param phi_out     [out] Pointer to an array (size `npts`) to store the retrieved phi angle values.
 * @note Exits via `CLEAN_EXIT(1)` on memory allocation failure, file open failure, fseek failure, or unexpected EOF.
 */
static void retrieve_all_particle_phi_snapshot(
    const char *filename,
    int snap,
    int npts,
    int block_size,
    float *phi_out)
{
    // Allocate local (thread-private) array
    float *tmpPhi = (float *)malloc(npts * sizeof(float));

    if (!tmpPhi)
    {
        fprintf(stderr, "Error: out of memory in retrieve_all_particle_phi_snapshot!\n");
        CLEAN_EXIT(1);
    }

// Read from file in a critical section
#pragma omp critical(file_access_phi)
    {
        FILE *f = fopen(filename, "rb");
        if (!f)
        {
            fprintf(stderr, "Error: cannot open %s for reading\n", filename);
            CLEAN_EXIT(1);
        }

        // Compute offset in file.
        int block_number = snap / block_size;
        int index_in_block = snap % block_size;

        long long step_data_size = (long long)npts * 4; // 4 bytes per float.
        long long block_data_size = (long long)block_size * step_data_size;
        long long offset = block_data_size * block_number + step_data_size * index_in_block;

        if (fseek(f, offset, SEEK_SET) != 0)
        {
            fprintf(stderr, "Error: fseek failed for snap=%d in phi file\n", snap);
            fclose(f);
            CLEAN_EXIT(1);
        }

        // Read npts floats into local buffer.
        for (int i = 0; i < npts; i++)
        {
            float phi_val;
            if (fread(&phi_val, sizeof(float), 1, f) != 1)
            {
                fprintf(stderr, "Error: unexpected end of file reading snap=%d, particle %d from phi file\n", snap, i);
                fclose(f);
                CLEAN_EXIT(1);
            }
            tmpPhi[i] = phi_val;
        }

        fclose(f);
    } // End critical section

    // Copy to output arrays (outside critical section)
    for (int i = 0; i < npts; i++)
    {
        phi_out[i] = tmpPhi[i];
    }

    free(tmpPhi);
}

/**
 * @brief Retrieves particle data for a specific snapshot from the `all_particle_data.dat` binary file.
 * @details This function reads the data (rank, radius, radial velocity, angular momentum)
 *          for all `npts` particles corresponding to a single snapshot number (`snap`)
 *          from the specified binary file. The file is expected to be in step-major order,
 *          where each record per particle consists of an int (rank) and three floats (R, Vrad, L).
 *          It calculates the correct file offset to seek to the desired snapshot.
 *          To ensure thread safety when called in parallel (e.g., during post-processing
 *          of snapshots), file I/O (seeking and reading) is performed within an
 *          OpenMP critical section named `file_access`. Temporary local buffers are used
 *          for reading, and data is then copied to the caller-provided output arrays.
 *
 * @param filename    [in] Path to the binary data file (e.g., "data/all_particle_data<suffix>.dat").
 * @param snap        [in] The snapshot number (0-indexed, corresponding to write events) to retrieve.
 * @param npts        [in] Number of particles per snapshot.
 * @param block_size  [in] Number of snapshots written per block by
 *                         `append_all_particle_data_chunk_to_file`. Determines seek offset calculation.
 * @param L_out       [out] Pointer to an array (size `npts`) to store the retrieved angular momentum values.
 * @param Rank_out    [out] Pointer to an array (size `npts`) to store the retrieved particle ranks.
 * @param R_out       [out] Pointer to an array (size `npts`) to store the retrieved radial positions.
 * @param Vrad_out    [out] Pointer to an array (size `npts`) to store the retrieved radial velocities.
 * @note Exits via `CLEAN_EXIT(1)` on memory allocation failure, file open failure, fseek failure, or unexpected EOF.
 */
static void retrieve_all_particle_snapshot(
    const char *filename,
    int snap,
    int npts,
    int block_size,
    float *L_out,
    int *Rank_out,
    float *R_out,
    float *Vrad_out)
{
    // Allocate local (thread-private) arrays
    float *tmpL = (float *)malloc(npts * sizeof(float));
    int *tmpRank = (int *)malloc(npts * sizeof(int));
    float *tmpR = (float *)malloc(npts * sizeof(float));
    float *tmpV = (float *)malloc(npts * sizeof(float));

    if (!tmpL || !tmpRank || !tmpR || !tmpV)
    {
        fprintf(stderr, "Error: out of memory in retrieve_all_particle_snapshot!\n");
        CLEAN_EXIT(1);
    }

    // Status messages are handled in the ordered section of the parallel loop.

// Read from file in a critical section
#pragma omp critical(file_access)
    {
        FILE *f = fopen(filename, "rb");
        if (!f)
        {
            fprintf(stderr, "Error: cannot open %s for reading\n", filename);
            CLEAN_EXIT(1);
        }

        // Compute offset in file.
        int block_number = snap / block_size;
        int index_in_block = snap % block_size;

        long long step_data_size = (long long)npts * 16; // 16 bytes per record.
        long long block_data_size = (long long)block_size * step_data_size;
        long long offset = block_data_size * block_number + step_data_size * index_in_block;

        if (fseek(f, offset, SEEK_SET) != 0)
        {
            fprintf(stderr, "Error: fseek failed for snap=%d\n", snap);
            fclose(f);
            CLEAN_EXIT(1);
        }

        // Read npts records into local buffers.
        for (int i = 0; i < npts; i++)
        {
            int rankval;
            float rval, vval, lval;

            if (fread(&rankval, sizeof(int), 1, f) != 1 ||
                fread(&rval, sizeof(float), 1, f) != 1 ||
                fread(&vval, sizeof(float), 1, f) != 1 ||
                fread(&lval, sizeof(float), 1, f) != 1)
            {
                fprintf(stderr, "Error: unexpected EOF while reading snap=%d (particle %d)\n", snap, i);
                fclose(f);
                CLEAN_EXIT(1);
            }
            tmpRank[i] = rankval;
            tmpR[i] = rval;
            tmpV[i] = vval;
            tmpL[i] = lval;
        }
        fclose(f);
    } // End critical(file_access).

    // Copy from local buffers to output arrays
    memcpy(Rank_out, tmpRank, npts * sizeof(int));
    memcpy(R_out, tmpR, npts * sizeof(float));
    memcpy(Vrad_out, tmpV, npts * sizeof(float));
    memcpy(L_out, tmpL, npts * sizeof(float));

    // Free the temporary local arrays
    free(tmpL);
    free(tmpRank);
    free(tmpR);
    free(tmpV);
}

/**
 * @brief Load particle state from all_particle_data file for restart/extend operations.
 * @details Reads a specific snapshot from an all_particle_data file and converts it
 *          to the internal particles array format used for initial conditions.
 *          This function is used by `--sim-restart` and `--sim-extend` to load the
 *          final particle state from a completed simulation run.
 * 
 * @param filename Path to the all_particle_data file
 * @param snapshot_index Which snapshot to load (0-based)
 * @param particles Output array [5][npts] in standard IC format
 * @param npts Number of particles
 * @param block_size Block size for I/O (typically 100)
 */
static void load_particles_from_restart(
    const char *filename,
    int snapshot_index,
    double **particles,
    int npts,
    int block_size,
    int *inverse_map)
{
    // Try to extract tag from filename for double buffer lookup
    // filename format: "data/all_particle_data_<tag>.dat"
    char tag[256] = "";
    const char *data_pos = strstr(filename, "all_particle_data");
    if (data_pos) {
        const char *tag_start = data_pos + strlen("all_particle_data");
        // Copy tag (everything after "all_particle_data")
        strncpy(tag, tag_start, sizeof(tag) - 1);
        tag[sizeof(tag) - 1] = '\0';
        // Remove ".dat" extension if present
        char *dot = strstr(tag, ".dat");
        if (dot) *dot = '\0';
    }

    // Attempt to load from double buffer if tag was found
    if (strlen(tag) > 0) {
        // Calculate total snapshots from file size
        FILE *fp_check = fopen(filename, "rb");
        int total_snapshots = 0;
        if (fp_check) {
            fseek(fp_check, 0, SEEK_END);
            long file_size = ftell(fp_check);
            fclose(fp_check);
            total_snapshots = file_size / (npts * 16);  // 16 bytes per particle per snapshot
        }

        int *temp_ids = (int *)malloc(npts * sizeof(int));
        int dbl_success = load_from_double_buffer(tag, snapshot_index, total_snapshots,
                                                  particles, npts, temp_ids);
        free(temp_ids);

        if (dbl_success) {
            // Need to load particle IDs to rebuild inverse_map correctly
            // Construct IDs filename from main filename
            char ids_filename[512];
            snprintf(ids_filename, sizeof(ids_filename), "%s", filename);
            char *data_pos_ids = strstr(ids_filename, "all_particle_data");
            if (data_pos_ids) {
                char temp[512];
                size_t prefix_len = data_pos_ids - ids_filename;
                strncpy(temp, ids_filename, prefix_len);
                temp[prefix_len] = '\0';
                strcat(temp, "all_particle_ids");
                strcat(temp, data_pos_ids + strlen("all_particle_data"));
                strcpy(ids_filename, temp);
            }

            // Load particle IDs
            FILE *ids_file = fopen(ids_filename, "rb");
            if (ids_file) {
                long seek_pos = (long)snapshot_index * npts * sizeof(int);
                fseek(ids_file, seek_pos, SEEK_SET);

                int *particle_ids = (int *)malloc(npts * sizeof(int));
                size_t items_read = fread(particle_ids, sizeof(int), npts, ids_file);
                fclose(ids_file);

                if (items_read == (size_t)npts) {
                    // Rebuild inverse_map using actual particle IDs
                    for (int i = 0; i < npts; i++) {
                        int pid = particle_ids[i];
                        if (pid >= 0 && pid < npts) {
                            inverse_map[pid] = i;
                        }
                        particles[3][i] = (double)pid;  // Store actual ID in particles array
                    }
                }
                free(particle_ids);
            }

            return;  // Successfully loaded from double buffer
        } else {
            // Snapshot not found in double buffer; proceed with standard file loading.
            char data_filename_dbl[512];
            snprintf(data_filename_dbl, sizeof(data_filename_dbl),
                     "data/double_buffer_all_particle_data%s.dat", tag);

            FILE *fp_dbl_check = fopen(data_filename_dbl, "rb");
            if (fp_dbl_check) {
                fclose(fp_dbl_check);
                // File exists but snapshot not in buffer
                printf("⚠ WARNING: Snapshot %d not in double buffer (loading from float32 files)\n",
                       snapshot_index);
                printf("           Precision will be degraded from double→float32→double conversion\n");
                printf("           Waiting 10 seconds to review this warning... [");
                fflush(stdout);

                // 10-second animated progress bar (40 blocks, 0.25s each)
                const char *filled = "█";
                const char *empty = "░";
                int total_blocks = 40;

                for (int block = 0; block < total_blocks; block++) {
                    for (int j = 0; j < total_blocks; j++) {
                        printf("%s", (j <= block) ? filled : empty);
                    }
                    printf("]");
                    fflush(stdout);

                    usleep(250000);  // 250ms = 0.25 seconds

                    if (block < total_blocks - 1) {
                        printf("\r");
                        printf("           Waiting 10 seconds to review this warning... [");
                    }
                }
                printf("\n");
            } else {
                // Double buffer file does not exist - simulation predates this feature
                printf("⚠ NOTE: No double-precision buffer found for this simulation\n");
                printf("        (Simulation predates double-precision buffer implementation)\n");
                printf("        Loading from float32 files with standard precision\n");
                printf("        Waiting 5 seconds... [");
                fflush(stdout);

                // 5-second animated progress bar (20 blocks, 0.25s each)
                const char *filled = "█";
                const char *empty = "░";
                int total_blocks = 20;

                for (int block = 0; block < total_blocks; block++) {
                    for (int j = 0; j < total_blocks; j++) {
                        printf("%s", (j <= block) ? filled : empty);
                    }
                    printf("]");
                    fflush(stdout);

                    usleep(250000);  // 250ms = 0.25 seconds

                    if (block < total_blocks - 1) {
                        printf("\r");
                        printf("        Waiting 5 seconds... [");
                    }
                }
                printf("\n");
            }
        }
    }

    // Fallback: Load from single-precision float files.
    printf("Loading from float32 all_particle_data files...\n");

    // Allocate temporary arrays for reading
    float *L = (float *)malloc(npts * sizeof(float));
    int *Rank = (int *)malloc(npts * sizeof(int));
    float *R = (float *)malloc(npts * sizeof(float));
    float *Vrad = (float *)malloc(npts * sizeof(float));

    if (!L || !Rank || !R || !Vrad) {
        fprintf(stderr, "ERROR: Failed to allocate memory in load_particles_from_restart\n");
        CLEAN_EXIT(1);
    }

    // Read the snapshot from file
    retrieve_all_particle_snapshot(
        filename, snapshot_index, npts, block_size,
        L, Rank, R, Vrad
    );

    // Also retrieve the corresponding phi snapshot
    float *phi = (float *)malloc(npts * sizeof(float));
    if (!phi) {
        fprintf(stderr, "ERROR: Failed to allocate memory for phi in load_particles_from_restart\n");
        CLEAN_EXIT(1);
    }

    // Construct phi filename from main filename
    // filename is like "data/all_particle_data<suffix>.dat"
    // Construct "data/all_particle_phi<suffix>.dat"
    char phi_filename[512];
    snprintf(phi_filename, sizeof(phi_filename), "%s", filename);
    char *data_pos_phi = strstr(phi_filename, "all_particle_data");
    if (data_pos_phi) {
        // Replace "all_particle_data" with "all_particle_phi"
        char temp[512];
        // Copy prefix before "all_particle_data"
        size_t prefix_len = data_pos_phi - phi_filename;
        strncpy(temp, phi_filename, prefix_len);
        temp[prefix_len] = '\0';
        // Append "all_particle_phi" and the rest
        strcat(temp, "all_particle_phi");
        strcat(temp, data_pos_phi + strlen("all_particle_data"));
        strcpy(phi_filename, temp);
    } else {
        fprintf(stderr, "WARNING: Could not parse filename for phi data: %s\n", filename);
    }

    // Try to read phi data; if file does not exist, initialize with random values
    FILE *phi_test = fopen(phi_filename, "rb");
    if (phi_test) {
        fclose(phi_test);
        // File exists, read the phi data
        retrieve_all_particle_phi_snapshot(phi_filename, snapshot_index, npts, block_size, phi);
        printf("  - Loaded phi angles from: %s\n", phi_filename);
    } else {
        // File does not exist (old simulation); initialize phi randomly
        printf("  - No phi file found, initializing random phi angles\n");
        for (int i = 0; i < npts; i++) {
            // Initialize phi in range [-π, π] for consistency with atan2 output
            phi[i] = M_PI * (2.0 * gsl_rng_uniform(g_rng) - 1.0);
        }
    }

    // Also retrieve particle IDs
    int *particle_ids = (int *)malloc(npts * sizeof(int));
    if (!particle_ids) {
        fprintf(stderr, "ERROR: Failed to allocate memory for IDs in load_particles_from_restart\n");
        CLEAN_EXIT(1);
    }

    // Construct IDs filename from main filename
    // filename is like "data/all_particle_data<suffix>.dat"
    // Construct "data/all_particle_ids<suffix>.dat"
    char ids_filename[512];
    snprintf(ids_filename, sizeof(ids_filename), "%s", filename);
    char *data_pos_ids = strstr(ids_filename, "all_particle_data");
    if (data_pos_ids) {
        // Replace "all_particle_data" with "all_particle_ids"
        char temp[512];
        // Copy prefix before "all_particle_data"
        size_t prefix_len = data_pos_ids - ids_filename;
        strncpy(temp, ids_filename, prefix_len);
        temp[prefix_len] = '\0';
        // Append "all_particle_ids" and the rest
        strcat(temp, "all_particle_ids");
        strcat(temp, data_pos_ids + strlen("all_particle_data"));
        strcpy(ids_filename, temp);
    } else {
        fprintf(stderr, "WARNING: Could not parse filename for IDs data: %s\n", filename);
    }

    // Try to read IDs data
    FILE *ids_test = fopen(ids_filename, "rb");
    if (ids_test) {
        fclose(ids_test);
        // File exists, read the particle IDs
        // Read IDs directly - they are stored in same order as other data
        FILE *f = fopen(ids_filename, "rb");
        if (f) {
            // Seek to correct snapshot
            long seek_pos = (long)snapshot_index * npts * sizeof(int);
            fseek(f, seek_pos, SEEK_SET);

            // Read all IDs for this snapshot
            size_t items_read = fread(particle_ids, sizeof(int), npts, f);
            if (items_read != (size_t)npts) {
                fprintf(stderr, "ERROR: Failed to read IDs from %s (got %zu, expected %d)\n",
                        ids_filename, items_read, npts);
                fclose(f);
                CLEAN_EXIT(1);
            }
            fclose(f);
            printf("  - Loaded particle IDs from: %s\n", ids_filename);
        }
    } else {
        // File does not exist - simulation predates particle ID tracking; using index as fallback ID
        printf("  - No IDs file found, using index as particle ID (WARNING: may cause tracking issues)\n");
        for (int i = 0; i < npts; i++) {
            particle_ids[i] = i;
        }
    }

    // Convert to particles array format
    // particles[0][i] = radius
    // particles[1][i] = |v| (velocity magnitude)
    // particles[2][i] = L (angular momentum)
    // particles[3][i] = particle ID
    // particles[4][i] = mu (v_rad / |v|)
    // particles[5][i] = phi angle
    for (int i = 0; i < npts; i++) {
        particles[0][i] = (double)R[i];                    // Radius
        particles[2][i] = (double)L[i];                    // Angular momentum
        particles[3][i] = (double)particle_ids[i];        // Particle ID from file

        // For restart, particles[1] should contain radial velocity (for simulation)
        // NOT velocity magnitude (which is for IC generation)
        particles[1][i] = (double)Vrad[i];                 // Radial velocity for simulation
        particles[4][i] = 0.0;                             // mu not needed after restart

        // Convert phi angle to cos(phi) and sin(phi) for SIDM calculations
        particles[5][i] = cos((double)phi[i]);             // cos(phi)
        particles[6][i] = sin((double)phi[i]);             // sin(phi)
    }

    // Cleanup temporary arrays
    free(L);
    free(Rank);
    free(R);
    free(Vrad);
    free(phi);
    free(particle_ids);

    printf("Loaded snapshot %d from restart file: %s\n", snapshot_index, filename);
    printf("  - Loaded %d particles\n", npts);
    printf("  - Particle state restored from saved checkpoint\n");

    // Rebuild inverse_map after loading particles with their original IDs
    if (inverse_map != NULL) {
        for (int idx = 0; idx < npts; idx++) {
            int orig_id = (int)particles[3][idx];  // Get particle's original ID
            inverse_map[orig_id] = idx;            // Map ID to current position
        }
        printf("  - Rebuilt inverse_map for trajectory tracking\n");
    }

}

/**
 * @brief Save chosen particle IDs to a binary file.
 * @details This function saves particle IDs for multiple trajectory tracking systems to ensure
 *          continuity across restart and extend operations. It writes two sets of particle IDs:
 *          one for particles selected based on lowest angular momentum, and another for particles
 *          tracked by their original ID for general trajectory analysis.
 *
 *          The binary file format consists of:
 *          - Header (8 bytes):
 *            - int32: num_lowestl - Number of particles with lowest angular momentum
 *            - int32: num_trajectories - Number of particles for general trajectory tracking
 *          - Data section:
 *            - int32[num_lowestl]: Particle IDs selected by lowest angular momentum
 *            - int32[num_trajectories]: Particle IDs for general trajectory tracking
 *
 *          The lowest_l particles are used by lowest_l_trajectories.dat which tracks
 *          radius, energy, and angular momentum for these specific particles over time.
 *          The trajectory particles are used by trajectories.dat which tracks radius,
 *          velocity, and mu for the specified particles by their original ID.
 *          The file single_trajectory.dat uses only the first particle ID from the trajectory list.
 *
 *          This file is essential for maintaining consistent particle tracking when simulations
 *          are restarted or extended, ensuring that the same particles continue to be monitored.
 *
 * @param filename    [in] Path to the output binary file (e.g., "data/chosen_particles_<suffix>.dat")
 * @param chosen      [in] Array of particle IDs that have the lowest angular momentum values
 * @param nlowest     [in] Number of lowest angular momentum particles to save
 * @param traj_ids    [in] Array of particle IDs to track in trajectories.dat
 * @param n_traj      [in] Number of trajectory particles to save
 * @note Exits via CLEAN_EXIT(1) if file creation fails.
 */
static void save_chosen_particles(const char *filename, int *chosen, int nlowest,
                                  int *traj_ids, int n_traj) {
    FILE *fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot create chosen particles file: %s\n", filename);
        CLEAN_EXIT(1);
    }

    // Write header: num_lowestl, num_trajectories
    fwrite(&nlowest, sizeof(int), 1, fp);
    fwrite(&n_traj, sizeof(int), 1, fp);

    // Write lowest_l particle IDs
    fwrite(chosen, sizeof(int), nlowest, fp);

    // Write trajectory particle IDs
    fwrite(traj_ids, sizeof(int), n_traj, fp);

    fclose(fp);
    printf("Saved chosen particles to %s\n", filename);
    printf("  - %d lowest_l particles (IDs: ", nlowest);
    for (int i = 0; i < nlowest && i < 5; i++) {
        printf("%d%s", chosen[i], (i < nlowest-1 && i < 4) ? ", " : "");
    }
    if (nlowest > 5) printf(", ...");
    printf(")\n");
    printf("  - %d trajectory particles (IDs: ", n_traj);
    for (int i = 0; i < n_traj && i < 5; i++) {
        printf("%d%s", traj_ids[i], (i < n_traj-1 && i < 4) ? ", " : "");
    }
    if (n_traj > 5) printf(", ...");
    printf(")\n");
}

/**
 * @brief Load chosen particle IDs from a binary file.
 * @details This function loads particle IDs that were previously saved for trajectory tracking,
 *          restoring the particle selection state from a prior simulation run. It reads two
 *          sets of particle IDs: one for particles with the lowest angular momentum, and another
 *          for general trajectory tracking by original particle ID.
 *
 *          The expected binary file format is:
 *          - Header (8 bytes):
 *            - int32: num_lowestl - Number of particles with lowest angular momentum
 *            - int32: num_trajectories - Number of particles for general trajectory tracking
 *          - Data section:
 *            - int32[num_lowestl]: Particle IDs selected by lowest angular momentum
 *            - int32[num_trajectories]: Particle IDs for general trajectory tracking
 *
 *          For backward compatibility, the function also supports files with only a single
 *          integer header followed by lowest_l particle IDs. This allows older simulation
 *          outputs to be read correctly.
 *
 *          The loaded particle IDs are used to maintain continuity in trajectory tracking
 *          across simulation restarts and extensions. The lowest_l particles continue to be
 *          tracked in lowest_l_trajectories.dat, while the trajectory particles continue
 *          in trajectories.dat and single_trajectory.dat.
 *
 * @param filename      [in]  Path to the input binary file containing saved particle IDs
 * @param chosen_out    [out] Pointer to store allocated array of lowest_l particle IDs
 * @param nlowest_out   [out] Number of lowest_l particles loaded
 * @param traj_ids_out  [out] Pointer to store allocated array of trajectory particle IDs (can be NULL)
 * @param n_traj_out    [out] Number of trajectory particles loaded (can be NULL)
 * @return 1 on success, 0 on failure (file not found or read error)
 * @note If traj_ids_out is NULL, trajectory IDs are read but not returned to the caller.
 *       Memory is allocated for the output arrays; caller is responsible for freeing them.
 */
static int load_chosen_particles(const char *filename, int **chosen_out, int *nlowest_out,
                                 int **traj_ids_out, int *n_traj_out) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        return 0; // File does not exist
    }

    // Read header
    int nlowest, n_traj;
    if (fread(&nlowest, sizeof(int), 1, fp) != 1) {
        fclose(fp);
        return 0;
    }

    // Check for second header value
    long pos = ftell(fp);
    if (fread(&n_traj, sizeof(int), 1, fp) != 1) {
        // Single-header format (backward compatibility)
        fseek(fp, pos, SEEK_SET);
        n_traj = 0;
    }

    // Allocate and read lowest_l particle IDs
    int *chosen = (int *)malloc(nlowest * sizeof(int));
    if (!chosen) {
        fprintf(stderr, "Error: Failed to allocate chosen array\n");
        fclose(fp);
        CLEAN_EXIT(1);
    }

    if (fread(chosen, sizeof(int), nlowest, fp) != (size_t)nlowest) {
        free(chosen);
        fclose(fp);
        return 0;
    }

    // Read trajectory particle IDs if present
    int *traj_ids = NULL;
    if (n_traj > 0) {
        traj_ids = (int *)malloc(n_traj * sizeof(int));
        if (!traj_ids) {
            fprintf(stderr, "Error: Failed to allocate trajectory IDs array\n");
            free(chosen);
            fclose(fp);
            CLEAN_EXIT(1);
        }

        if (fread(traj_ids, sizeof(int), n_traj, fp) != (size_t)n_traj) {
            free(traj_ids);
            free(chosen);
            fclose(fp);
            return 0;
        }
    }

    fclose(fp);

    // Set outputs
    *chosen_out = chosen;
    *nlowest_out = nlowest;

    if (traj_ids_out && n_traj_out) {
        *traj_ids_out = traj_ids;
        *n_traj_out = n_traj;
    } else if (traj_ids) {
        free(traj_ids);  // Free if not returning
    }


    return 1; // Success
}

/**
 * @brief Adjusts the total number of timesteps to align with desired output snapshot intervals.
 * @details This function calculates an adjusted number of total simulation timesteps, \f$N'_{times}\f$,
 *          such that it is greater than or equal to the initially requested `Ntimes_initial` (\f$N\f$)
 *          and satisfies the constraint: \f$(N'_{times} - 1)\f$ must be an integer multiple of
 *          \f$(M - 1) \times p\f$. Here, \f$M\f$ is `nout` (number of desired output snapshot points,
 *          which means \f$M-1\f$ intervals) and \f$p\f$ is `dtwrite` (the low-level write interval
 *          in terms of simulation timesteps).
 *          This alignment ensures that exactly `nout` snapshots can be produced at intervals
 *          that are multiples of `dtwrite` and that also evenly span the total adjusted simulation duration.
 *
 * @param Ntimes_initial [in] Initially requested total number of simulation timesteps (\f$N\f$).
 * @param nout           [in] Number of desired output snapshot points (\f$M\f$). Must be >= 2 for adjustment to apply.
 * @param dtwrite        [in] The interval (in timesteps) at which low-level data is potentially written (\f$p\f$). Must be >= 1.
 * @return int The adjusted total number of timesteps (\f$N'_{times}\f$). Returns `Ntimes_initial`
 *             if `nout < 2` or `dtwrite < 1` or other edge cases where the constraint cannot be met.
 */
static int adjust_ntimesteps(int Ntimes_initial, int nout, int dtwrite)
{
    // Notation: M = nout, p = dtwrite, N = Ntimes_initial.
    // Find the smallest N' >= N such that (N' - 1) is a multiple of (M - 1) * p.

    int M = nout;
    int p = dtwrite;
    int N = Ntimes_initial;

    // Edge cases:
    if (M < 2)
    {
        // If only 0 or 1 snapshot requested, no interval constraint applies.
        return N;
    }
    if (p < 1)
    {
        // Invalid write interval.
        return N;
    }

    // The total number of intervals between M snapshots is (M - 1).
    // The total number of steps spanning these intervals must be a multiple of p.
    // Therefore, the total number of steps (N' - 1) must be a multiple of (M - 1) * p.
    // Find the smallest integer k >= 1 such that (M - 1) * k * p >= (N - 1).
    double required_steps = (double)(N - 1);
    double steps_per_output_cycle = (M - 1) * (double)p;

    // Handle case where denominator is zero (e.g., M=1 or p=0, caught above but added safety)
    if (steps_per_output_cycle <= 0) {
        return N; // Cannot satisfy constraint
    }

    double ratio = required_steps / steps_per_output_cycle;
    int k = (int)ceil(ratio);
    if (k < 1)
    {
        k = 1; // Ensure at least one full output cycle.
    }

    int Nprime_minus_1 = (M - 1) * k * p;
    int Nprime = Nprime_minus_1 + 1;

    return Nprime;
}

/**
 * @brief Comparison function for sorting LAndIndex structures by the \f$L\f$ member.
 * @details Sorts LAndIndex structures in ascending order based on their 'L'
 *          (angular momentum or squared difference from a reference \f$L\f$) value.
 *          Used with qsort for ordering particles by their \f$L\f$ values, typically
 *          for selecting particles with lowest \f$L\f$ or \f$L\f$ closest to a target.
 *
 * @param a [in] Pointer to the first LAndIndex structure.
 * @param b [in] Pointer to the second LAndIndex structure.
 * @return int -1 if a->L < b->L, 1 if a->L > b->L, 0 if equal.
 */
int cmp_LAI(const void *a, const void *b)
{
    double La = ((const LAndIndex *)a)->L;
    double Lb = ((const LAndIndex *)b)->L;

    if (La < Lb)
        return -1;
    if (La > Lb)
        return 1;
    return 0;
}

// =============================================================================
// ADAPTIVE FULL LEAPFROG STEP: r(n), v(n) --> r(n+1), v(n+1)
// =============================================================================
//
// Physics-based time integration method with adaptive step refinement:
// - Subdivide the time step h = ΔT in powers-of-2 "micro-steps"
// - Compare a (2N+1)-step "coarse" vs. a (4N+1)-step "fine" integration
// - If within tolerance, return one of {coarse, fine, rich (Richardson extrapolation)}
// - Otherwise, double N and repeat until convergence or max subdivision reached
/**
 * @brief Performs a leapfrog integration step using a fixed number of micro-steps.
 * @details This helper function implements the leapfrog (Kick-Drift-Kick) integration
 *          over a total time interval `h` by dividing it into a sequence of micro-steps.
 *          The number of micro-steps is `subSteps` (e.g., \f$2N+1\f$ for a "coarse" pass or
 *          \f$4N+1\f$ for a "fine" pass in an adaptive scheme, where \f$N\f$ is `N_subdivision_factor`).
 *          The micro-timestep sizes for kicks and drifts are adjusted based on `N_subdivision_factor`
 *          and whether it is a coarse or fine integration sequence.
 *          Sequence: Initial half-kick, \f$((\text{subSteps}-1)/2 - 1)\f$ full Drift-Kick pairs,
 *          a final full Drift, and a final half-Kick.
 *
 * @param i                   [in] Particle index (0 to npts-1), for rank in gravitational force.
 * @param npts                [in] Total number of particles.
 * @param r_in                [in] Input radial position (kpc) at the start of the total interval `h`.
 * @param v_in                [in] Input radial velocity (kpc/Myr) at the start of `h`.
 * @param ell                 [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr).
 * @param h                   [in] Total physical time interval for this leapfrog sequence (Myr).
 * @param N_subdivision_factor [in] Base subdivision factor \f$N\f$ used to determine micro-timestep sizes.
 * @param subSteps            [in] Total number of Kicks/Drifts (e.g., \f$2N+1\f$ or \f$4N+1\f$).
 * @param grav                [in] Gravitational constant G (simulation units).
 * @param r_out               [out] Pointer to store the output radial position (kpc) after time `h`.
 * @param v_out               [out] Pointer to store the output radial velocity (kpc/Myr) after time `h`.
 */
static void doMicroLeapfrog(
    int i, int npts,
    double r_in, double v_in, double ell,
    double h,     // Total physical time interval for this leapfrog sequence.
    int N,        // Base subdivision factor.
    int subSteps, // Total number of kicks/drifts.
    double grav,
    double *r_out, double *v_out)
{
    // Initialize current state with input values
    double r_curr = r_in;
    double v_curr = v_in;

    // Set up timestep sizes based on subdivision level
    double halfKick, midStep;
    if (subSteps == (2 * N + 1))
    {
        // Coarse integration (2N+1 substeps)
        halfKick = h / (2.0 * N);
        midStep = h / (1.0 * N);
    }
    else
    {
        // Fine integration (4N+1 substeps)
        halfKick = h / (4.0 * N);
        midStep = h / (2.0 * N);
    }

    // Initial half-kick (velocity update)
    double force = gravitational_force(r_curr, i, npts, grav, g_active_halo_mass);
    double dvdt = force + effective_angular_force(r_curr, ell);
    v_curr += halfKick * dvdt;

    // Middle pattern of drift-kick pairs
    int pairs = (subSteps - 1) / 2; // Total number of drift-kick pairs
    // Execute all but the last pair (last kick handled separately)
    for (int pp = 1; pp <= (pairs - 1); pp++)
    {
        // Drift: update position using current velocity
        r_curr += midStep * v_curr;

        // Kick: update velocity using forces at new position
        force = gravitational_force(r_curr, i, npts, grav, g_active_halo_mass);
        dvdt = force + effective_angular_force(r_curr, ell);
        v_curr += midStep * dvdt;
    }

    // Final full drift
    r_curr += midStep * v_curr;

    // Final half-kick
    force = gravitational_force(r_curr, i, npts, grav, g_active_halo_mass);
    dvdt = force + effective_angular_force(r_curr, ell);
    v_curr += halfKick * dvdt;

    *r_out = r_curr;
    *v_out = v_curr;
}

/**
 * @brief Performs an adaptive full leapfrog step with error control over a physical timestep `h`.
 * @details This function implements an adaptive leapfrog algorithm to advance a particle's
 *          state \f$(r, v_{rad})\f$ over a full physical timestep `h` (\f$\Delta T_{phys}\f$). It iteratively
 *          refines the integration by comparing a "coarse" integration (using \f$2N+1\f$
 *          micro-steps via `doMicroLeapfrog`) with a "fine" integration (using \f$4N+1\f$
 *          micro-steps). The subdivision factor \f$N\f$ starts at 1 and is doubled if the
 *          relative differences in final radius and velocity between coarse and fine passes
 *          exceed `radius_tol` and `velocity_tol`, respectively. This process repeats up
 *          to a maximum subdivision factor `max_subdiv`.
 *          The final state \f$(r_{out}, v_{out})\f$ for the step `h` is chosen based on `out_type`
 *          (coarse, fine, or Richardson extrapolation) once convergence is met or
 *          `max_subdiv` is reached.
 *
 * @param i             [in] Particle index (0 to npts-1), for rank in gravitational force calculation.
 * @param npts          [in] Total number of particles in the simulation.
 * @param r_in          [in] Initial radial position (kpc) at the start of the step `h`.
 * @param v_in          [in] Initial radial velocity (kpc/Myr) at the start of `h`.
 * @param ell           [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr), conserved.
 * @param h             [in] Full physical timestep size \f$\Delta T_{phys}\f$ (Myr).
 * @param radius_tol    [in] Relative convergence tolerance for radius comparison.
 * @param velocity_tol  [in] Relative convergence tolerance for velocity comparison.
 * @param max_subdiv    [in] Maximum allowed subdivision factor \f$N\f$ for micro-steps.
 * @param grav          [in] Gravitational constant G (simulation units, e.g., G_CONST).
 * @param out_type      [in] Result selection mode for converged integration:
 *                         0 for coarse result, 1 for fine result, 2 for Richardson extrapolation.
 * @param r_out         [out] Pointer to store the final radial position (kpc) after time `h`.
 * @param v_out         [out] Pointer to store the final radial velocity (kpc/Myr) after time `h`.
 */
static void doAdaptiveFullLeap(
    int i,               // Particle index for force computation
    int npts,            // Total number of particles in simulation
    double r_in,         // Initial radius at start of step
    double v_in,         // Initial velocity at start of step
    double ell,          // Angular momentum (conserved during integration)
    double h,            // Full physical timestep size ΔT
    double radius_tol,   // Convergence tolerance for radius
    double velocity_tol, // Convergence tolerance for velocity
    int max_subdiv,      // Maximum subdivision factor allowed
    double grav,         // Gravitational constant (renamed from G to avoid macro collision)
    int out_type,        // Result selection: 0=coarse, 1=fine, 2=Richardson extrapolation
    double *r_out,       // Output parameter for final radius
    double *v_out        // Output parameter for final velocity
)
{
    int N = 1; // Start with N=1 micro-steps.

    // Initialize coarse/fine results (first coarse pass).
    double r_coarse = r_in, v_coarse = v_in;
    double r_fine = r_in, v_fine = v_in;

    while (N <= max_subdiv)
    {
        // Coarse pass (2N+1 steps)
        doMicroLeapfrog(
            i, npts, r_in, v_in, ell,
            h, N, (2 * N + 1),
            grav,
            &r_coarse, &v_coarse);

        // Fine pass (4N+1 steps)
        doMicroLeapfrog(
            i, npts, r_in, v_in, ell,
            h, N, (4 * N + 1),
            grav,
            &r_fine, &v_fine);

        // Compare radius, velocity.
        double radius_diff = fabs(r_fine - r_coarse) / (fabs(r_fine) + 1.0e-30);
        double velocity_diff = fabs(v_fine - v_coarse) / (fabs(v_fine) + 1.0e-30);

        if ((radius_diff < radius_tol) && (velocity_diff < velocity_tol))
        {
            // Tolerances met => output according to out_type.
            if (out_type == 0)
            {
                *r_out = r_coarse;
                *v_out = v_coarse;
            }
            else if (out_type == 1)
            {
                *r_out = r_fine;
                *v_out = v_fine;
            }
            else
            {
                // Richardson extrapolation for higher-order result
                *r_out = 4.0 * r_fine - 3.0 * r_coarse;
                *v_out = 4.0 * v_fine - 3.0 * v_coarse;
            }
            return; // Done.
        }
        else
        {
            // Not converged, increase refinement
            N *= 2;
        }
    }

    // Exiting while(N <= max_subdiv) loop indicates convergence failure
    // Return the last available result based on out_type.
    if (out_type == 0)
    {
        *r_out = r_coarse;
        *v_out = v_coarse;
    }
    else if (out_type == 1)
    {
        *r_out = r_fine;
        *v_out = v_fine;
    }
    else
    {
        *r_out = 4.0 * r_fine - 3.0 * r_coarse;
        *v_out = 4.0 * v_fine - 3.0 * v_coarse;
    }
}

// =============================================================================
// LEVI-CIVITA REGULARIZATION
// =============================================================================
//
// Physics-based regularized time integration method:
// - Transforms coordinates (r -> ρ = √r) to handle close encounters
// - Uses fictitious time τ to integrate equations of motion
// - Maps back to physical coordinates and time after integration
// - Provides enhanced stability for high-eccentricity orbits

/**
 * @brief Calculates \f$d\rho/d\tau\f$, the derivative of the regularized coordinate \f$\rho\f$ with respect to fictitious time \f$\tau\f$.
 * @details In Levi-Civita regularization, \f$d\rho/d\tau = (1/2) \rho v_{rad}\f$, where \f$\rho = \sqrt{r}\f$
 *          and \f$v_{rad}\f$ is the radial velocity in physical units (though often represented as \f$v\f$ or \f$v_{\rho}\f$
 *          in transformed equations of motion depending on the specific formulation).
 *          This function implements this relationship.
 *
 * @param rhoVal [in] The current value of the regularized radial coordinate \f$\rho = \sqrt{r}\f$.
 * @param vVal   [in] The current radial velocity \f$v_{rad}\f$ (kpc/Myr).
 * @return double The value of \f$d\rho/d\tau\f$ (kpc\f$^{3/2}\f$/Myr).
 */
static inline double dRhoDtaufun(double rhoVal, double vVal)
{
    // dρ/dτ = 0.5 * ρ * v
    return 0.5 * rhoVal * vVal;
}

/**
 * @brief Calculates the total effective force per unit mass in Levi-Civita transformed coordinates.
 * @details This function computes \f$F_{\rho}/m = (F_{grav,\rho} + F_{centrifugal,\rho})/m\f$,
 *          where \f$F_{grav,\rho}\f$ is the gravitational force and \f$F_{centrifugal,\rho}\f$ is the
 *          effective centrifugal force, both expressed in the regularized radial coordinate \f$\rho = \sqrt{r}\f$.
 *          It calls `gravitational_force_rho_v` and `effective_angular_force_rho_v`.
 *          This combined force is used in the equations of motion for Levi-Civita regularization.
 *
 * @param i          [in] Particle index (0 to npts-1), for rank in gravitational force calculation.
 * @param npts       [in] Total number of particles.
 * @param totalmass  [in] Total halo mass of the system (\f$M_{\odot}\f$) used for gravitational force.
 * @param grav       [in] Gravitational constant G (simulation units).
 * @param ell        [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr).
 * @param rhoVal     [in] Current value of the regularized radial coordinate \f$\rho = \sqrt{r}\f$.
 * @return double    \f$dv/d\tau\f$ in Levi-Civita coordinates (kpc\f$^2\f$/Myr\f$^2\f$).
 */
static inline double forceLCfun(
    int i, int npts,
    double totalmass,
    double grav,
    double ell,
    double rhoVal)
{
    double gravPart = gravitational_force_rho_v(rhoVal, i, npts, grav, totalmass);
    double angPart = effective_angular_force_rho_v(rhoVal, ell);
    return gravPart + angPart;
}

/**
 * @brief Performs integration of particle motion over a physical time `dt` using Levi-Civita regularization.
 * @details This function implements a leapfrog-like integration scheme in Levi-Civita
 *          regularized coordinates \f$(ρ, v_{rad})\f$ and fictitious time \f$τ\f$. The physical
 *          coordinates are \f$r = ρ^2\f$, and physical time \f$t_{phys}\f$ is related to \f$τ\f$ by \f$dt_{phys} = ρ^2 dτ\f$.
 *          The integration proceeds by taking variable \f$Δτ\f$ steps (estimated based on `N_taumin`)
 *          until the accumulated physical time `t_phys` reaches or exceeds the target physical
 *          timestep `dt`. If a step overshoots `dt`, linear interpolation is used to obtain
 *          the state precisely at the target physical time.
 *          This method is particularly effective for handling close encounters where \f$r → 0\f$.
 *
 * @param i         [in] Particle index (0 to npts-1), for rank in force calculation.
 * @param npts      [in] Total number of particles.
 * @param r_in      [in] Initial physical radial position (kpc) at the start of the physical step `dt`.
 * @param v_in      [in] Initial physical radial velocity (kpc/Myr) at the start of `dt`.
 * @param ell       [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr) for the force calculation.
 * @param dt        [in] The full physical timestep \f$\Delta T_{phys}\f$ (Myr) to advance the particle.
 * @param N_taumin  [in] Target number of fictitious \f$τ\f$-steps within the `dt` interval; influences
 *                     the initial guess for \f$Δτ\f$.
 * @param grav      [in] Gravitational constant G (simulation units).
 * @param r_out     [out] Pointer to store the final physical radial position (kpc) after time `dt`.
 * @param v_out     [out] Pointer to store the final physical radial velocity (kpc/Myr) after time `dt`.
 */
static void doLeviCivitaLeapfrog(
    int i,
    int npts,
    double r_in,
    double v_in,
    double ell,
    double dt,
    int N_taumin,
    double grav,
    double *r_out,
    double *v_out)
{
    // Transform to Levi-Civita coordinates: rho = sqrt(r)
    double rho = sqrt(r_in);
    double v_rad = v_in;
    double tau = 0.0;
    double t_cur = 0.0;

    // Estimate initial tau step size based on N_taumin
    double deltaTau = 0.0;
    if (r_in > 1.0e-30 && N_taumin > 0)
    {
        deltaTau = dt / (2.0 * r_in * N_taumin);
    }
    else
    {
        deltaTau = dt / 100.0;
    }

    // Initialize integration state variables
    double rho_cur = rho;
    double v_cur = v_rad;
    double tau_cur = tau;
    double t_phys = t_cur;

    int stepCount = 0;
    int stepMax = 200000000; // Maximum step count to prevent infinite loops.

    while (1)
    {
        if (t_phys >= dt)
        {
            // Exit loop when physical time reaches target
            break;
        }

        // Leapfrog step 1: Evaluate force at current position
        double fval = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_cur);

        // Leapfrog step 2: First half-kick for velocity
        double v_half = v_cur + 0.5 * deltaTau * fval;

        // Leapfrog step 3: Full drift for position
        double rho_next = rho_cur + deltaTau * dRhoDtaufun(rho_cur, v_half);

        // Leapfrog step 4: Second half-kick with force at new position
        double fval2 = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_next);
        double v_next = v_half + 0.5 * deltaTau * fval2;

        // Update physical time using midpoint rho value
        double rho_mid = 0.5 * (rho_cur + rho_next);
        double t_next = t_phys + deltaTau * (rho_mid * rho_mid);

        double tau_next = tau_cur + deltaTau;

        // Handle case where step overshoots target time
        if (t_next >= dt)
        {
            double alpha = 0.0;
            if (fabs(t_next - t_phys) > 1.0e-30)
            {
                alpha = (dt - t_phys) / (t_next - t_phys);
            }
            else
            {
                alpha = 1.0;
            }

            // Interpolate state to exact target time
            double rho_f = rho_cur + alpha * (rho_next - rho_cur);
            double v_f = v_cur + alpha * (v_next - v_cur);

            // Transform back to physical coordinates
            double r_f = rho_f * rho_f;
            *r_out = r_f;
            *v_out = v_f;
            return;
        }

        // Update state for next iteration
        rho_cur = rho_next;
        v_cur = v_next;
        tau_cur = tau_next;
        t_phys = t_next;

        stepCount++;
        if (stepCount > stepMax)
        {
            // Safety exit if maximum iteration count exceeded
            *r_out = rho_cur * rho_cur;
            *v_out = v_cur;
            return;
        }
    }

    // Transform final state back to physical coordinates
    *r_out = rho_cur * rho_cur;
    *v_out = v_cur;

    return;
}

// =============================================================================
// ADAPTIVE FULL LEVI-CIVITA REGULARIZATION
// =============================================================================
//
// Enhanced regularization scheme that combines adaptive step sizing with
// Levi-Civita coordinate transformation for optimal performance near
// the coordinate origin.

/**
 * @brief Performs Levi-Civita regularized integration using a fixed number of micro-steps.
 * @details This helper function advances the particle state in regularized coordinates
 *          \f$(ρ, v_{rad}, t_{phys})\f$ over a total fictitious time interval `h_tau` (\f$Δτ_{total}\f$)
 *          by taking a specified number of `subSteps` fixed-size micro-steps (\f$δτ = Δτ_{total} / \text{subSteps}\f$).
 *          Each micro-step uses a leapfrog-like scheme (Kick-Drift-Kick for \f$ρ, v_{rad}\f$)
 *          and updates the accumulated physical time \f$t_{phys}\f$ using \f$dt_{phys} = δτ \cdot ρ_{mid}^2\f$.
 *          This function is called by `doSingleTauStepAdaptiveLeviCivita` for its coarse and fine passes.
 *
 * @param i         [in] Particle index (0 to npts-1), for rank in force calculation.
 * @param npts      [in] Total number of particles.
 * @param rho_in    [in] Initial \f$ρ = \sqrt{r}\f$ at the start of the `h_tau` interval.
 * @param v_in      [in] Initial radial velocity \f$v_{rad}\f$ at the start of `h_tau`.
 * @param t_in      [in] Initial accumulated physical time \f$t_{phys}\f$ at the start of `h_tau`.
 * @param subSteps  [in] Number of fixed micro-steps to perform over `h_tau`.
 * @param h_tau     [in] Total fictitious time interval \f$Δτ_{total}\f$ for this integration sequence.
 * @param grav      [in] Gravitational constant G (simulation units).
 * @param ell       [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr).
 * @param rho_out   [out] Pointer to store the final \f$ρ\f$ after `h_tau`.
 * @param v_out     [out] Pointer to store the final \f$v_{rad}\f$ after `h_tau`.
 * @param t_out     [out] Pointer to store the final accumulated \f$t_{phys}\f$ after `h_tau`.
 */
static void doMicroLeviCivita(
    int i,
    int npts,
    double rho_in,
    double v_in,
    double t_in,
    int subSteps,
    double h_tau,
    double grav,
    double ell,
    double *rho_out,
    double *v_out,
    double *t_out)
{

    double dtau = h_tau / (double)subSteps;
    double rho_curr = rho_in;
    double v_curr = v_in;
    double t_curr = t_in;

    for (int ss = 0; ss < subSteps; ss++)
    {
        // Half-kick.
        double fLC = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_curr);
        double v_half = v_curr + 0.5 * dtau * fLC;

        // Drift for rho.
        double rho_next = rho_curr + dtau * (0.5 * rho_curr * v_half);

        // Second half-kick.
        double fLC2 = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_next);
        double v_next = v_half + 0.5 * dtau * fLC2;

        // Accumulate physical time t(τ).
        // Simplest approach: use midpoint for rho =>  ρ_mid^2.
        double rho_mid = 0.5 * (rho_curr + rho_next);
        double dt_phys = dtau * (rho_mid * rho_mid);

        t_curr += dt_phys;
        rho_curr = rho_next;
        v_curr = v_next;
    }

    *rho_out = rho_curr;
    *v_out = v_curr;
    *t_out = t_curr;
}

/**
 * @brief Performs a single adaptive step in Levi-Civita coordinates over a proposed fictitious time `h_guess`.
 * @details This function integrates the equations of motion in regularized \f$(ρ, v_{rad}, t_{phys})\f$
 *          coordinates over a proposed fictitious time interval `h_guess` (\f$Δτ_{guess}\f$).
 *          It employs an adaptive refinement strategy by comparing a "coarse" integration
 *          (using \f$2N+1\f$ micro-steps via `doMicroLeviCivita`) with a "fine" integration
 *          (using \f$4N+1\f$ micro-steps). The subdivision factor \f$N\f$ starts at 1 and is
 *          doubled if the relative differences in \f$ρ\f$ and \f$v_{rad}\f$ between coarse and fine
 *          results exceed `radius_tol` and `velocity_tol`, respectively. This continues
 *          up to `max_subdiv`. The final state for the \f$Δτ_{guess}\f$ step is chosen based on
 *          `out_type` (coarse, fine, or Richardson extrapolation).
 *          The function outputs the final \f$ρ_{out}\f$, \f$v_{out}\f$ (radial velocity), and
 *          accumulated physical time \f$t_{out}\f$ corresponding to this adaptive \f$Δτ\f$ step.
 *
 * @param i             [in] Particle index (0 to npts-1), for rank in force calculation.
 * @param npts          [in] Total number of particles.
 * @param rho_in        [in] Initial regularized radial coordinate \f$ρ = \sqrt{r}\f$ at the start of \f$Δτ_{guess}\f$.
 * @param v_in          [in] Initial radial velocity \f$v_{rad}\f$ at the start of \f$Δτ_{guess}\f$.
 * @param t_in          [in] Initial accumulated physical time \f$t_{phys}\f$ at the start of \f$Δτ_{guess}\f$.
 * @param h_guess       [in] The proposed total fictitious time step \f$Δτ_{guess}\f$ for this adaptive step.
 * @param radius_tol    [in] Relative convergence tolerance for \f$ρ\f$ comparison.
 * @param velocity_tol  [in] Relative convergence tolerance for \f$v_{rad}\f$ comparison.
 * @param max_subdiv    [in] Maximum subdivision factor \f$N\f$ for micro-steps within `doMicroLeviCivita`.
 * @param grav          [in] Gravitational constant G (simulation units).
 * @param ell           [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr).
 * @param out_type      [in] Result selection for converged micro-integration: 0=coarse, 1=fine, 2=Richardson.
 * @param rho_out       [out] Pointer to store the final \f$ρ\f$ after the adaptive \f$Δτ_{guess}\f$ step.
 * @param v_out         [out] Pointer to store the final \f$v_{rad}\f$ after the adaptive \f$Δτ_{guess}\f$ step.
 * @param t_out         [out] Pointer to store the final accumulated physical time \f$t_{phys}\f$ after this step.
 */
static void doSingleTauStepAdaptiveLeviCivita(
    int i, int npts,
    double rho_in,
    double v_in,
    double t_in,
    double h_guess,
    double radius_tol,
    double velocity_tol,
    int max_subdiv,
    double grav,
    double ell,
    int out_type,
    double *rho_out,
    double *v_out,
    double *t_out)
{
    // Initialize with the smallest subdivision factor N=1
    int N = 1;

    while (N <= max_subdiv)
    {
        // COARSE integration: use subSteps = 2N+1
        double rhoC, vC, tC;
        doMicroLeviCivita(
            i, npts,
            rho_in, v_in,
            t_in,
            (2 * N + 1),
            h_guess,
            grav,
            ell,
            &rhoC, &vC, &tC);

        // FINE integration: use subSteps = 4N+1
        double rhoF, vF, tF;
        doMicroLeviCivita(
            i, npts,
            rho_in, v_in,
            t_in,
            (4 * N + 1),
            h_guess,
            grav,
            ell,
            &rhoF, &vF, &tF);

        // Compare final radius and velocity for convergence
        // Calculate relative differences between coarse and fine solutions

        double rF = rhoF * rhoF;

        double rhodif = fabs(rhoF - rhoC) / (fabs(rF) + 1.0e-30); // Relative radial difference
        double vdif = fabs(vF - vC) / (fabs(vF) + 1.0e-30);       // Relative velocity difference

        if ((rhodif < radius_tol) && (vdif < velocity_tol))
        {
            if (out_type == 0)
            {
                *rho_out = rhoC;
                *v_out = vC;
                *t_out = tC;
            }
            else if (out_type == 1)
            {
                *rho_out = rhoF;
                *v_out = vF;
                *t_out = tF;
            }
            else
            {
                // Apply Richardson extrapolation formula: result = 4*fine - 3*coarse
                // This provides a higher-order approximation by eliminating leading error terms
                double rho_rich = 4.0 * rhoF - 3.0 * rhoC; // Extrapolated ρ value
                double v_rich = 4.0 * vF - 3.0 * vC;       // Extrapolated velocity
                double t_rich = 4.0 * tF - 3.0 * tC;       // Extrapolated time value

                // Ensure non-negative radius
                if (rho_rich < 0.0)
                    rho_rich = 0.0;
                *rho_out = rho_rich;
                *v_out = v_rich;
                *t_out = t_rich;
            }
            return;
        }
        else
        {
            // Not converged, double subdivision factor and try again
            N *= 2;
        }
    }

    // Convergence not achieved within max_subdiv iterations
    // Use highest-resolution fine integration as fallback result
    {
        double rhoF, vF, tF;
        doMicroLeviCivita(
            i, npts,
            rho_in, v_in,
            t_in,
            (4 * N + 1),
            h_guess,
            grav,
            ell,
            &rhoF, &vF, &tF);

        *rho_out = rhoF;
        *v_out = vF;
        *t_out = tF;
    }
}

/**
 * @brief Performs a full adaptive leapfrog step using Levi-Civita regularization over a physical time interval `dt`.
 * @details This function integrates a particle's motion over a physical timestep `dt` (\f$\Delta T_{phys}\f$)
 *          by taking multiple adaptive steps in fictitious Levi-Civita time \f$τ\f$.
 *          It repeatedly calls `doSingleTauStepAdaptiveLeviCivita` to advance the state in
 *          \f$(ρ, v_{rad}, t_{phys})\f$ coordinates, where \f$ρ = \sqrt{r}\f$. Each call to
 *          `doSingleTauStepAdaptiveLeviCivita` takes an adaptive \f$Δτ\f$ step.
 *          The loop continues until the accumulated physical time `t_cur` (from summing \f$Δt_{phys}\f$
 *          corresponding to each \f$Δτ\f$) reaches or exceeds the target `dt`.
 *          If a \f$Δτ\f$ step overshoots `dt`, linear interpolation is used to find the
 *          state precisely at `t_cur = dt`. The final regularized state \f$(ρ_f, v_f)\f$ is then
 *          transformed back to physical coordinates \f$(r_{out}, v_{out})\f$.
 *          An initial guess for \f$Δτ\f$ is made based on `N_taumin` and the initial radius.
 *
 * @param i             [in] Particle index (0 to npts-1), used for rank in gravitational force calculation.
 * @param npts          [in] Total number of particles in the simulation.
 * @param r_in          [in] Initial physical radial position (kpc) at the start of the physical step `dt`.
 * @param v_in          [in] Initial physical radial velocity (kpc/Myr) at the start of `dt`.
 * @param ell           [in] Angular momentum per unit mass (kpc\f$^2\f$/Myr), conserved during integration.
 * @param dt            [in] The full physical timestep \f$\Delta T_{phys}\f$ (Myr) to advance the particle.
 * @param N_taumin      [in] Target number of fictitious \f$τ\f$-steps within `dt`; influences the initial \f$Δτ\f$ guess.
 * @param radius_tol    [in] Relative convergence tolerance for \f$ρ\f$ comparison within each adaptive \f$τ\f$-step.
 * @param velocity_tol  [in] Relative convergence tolerance for velocity comparison within each adaptive \f$τ\f$-step.
 * @param max_subdiv    [in] Maximum allowed subdivision factor N for micro-steps within each adaptive \f$τ\f$-step.
 * @param grav          [in] Gravitational constant G (simulation units, e.g., G_CONST).
 * @param out_type      [in] Result selection mode for micro-steps within `doSingleTauStepAdaptiveLeviCivita`:
 *                         0 for coarse, 1 for fine, 2 for Richardson extrapolation.
 * @param r_out         [out] Pointer to store the final physical radial position (kpc) after time `dt`.
 * @param v_out         [out] Pointer to store the final physical radial velocity (kpc/Myr) after time `dt`.
 */
static void doAdaptiveFullLeviCivita(
    int i,
    int npts,
    double r_in,
    double v_in,
    double ell,
    double dt, // big step in physical time
    int N_taumin,
    double radius_tol,
    double velocity_tol,
    int max_subdiv,
    double grav,
    int out_type, // 0=coarse,1=fine,2=Richardson
    double *r_out,
    double *v_out)
{
    // Handle near-zero radius edge case
    if (r_in < 1.0e-30)
    {
        *r_out = r_in;
        *v_out = v_in;
        return;
    }

    // Transform from physical to regularized coordinates
    double rho_current = sqrt(r_in);
    double v_current = v_in;

    double t_cur = 0.0; // Current physical time.

    // Estimate initial fictitious time step deltaTau
    double deltaTau = 0.0;
    if (r_in > 1.0e-30 && N_taumin > 0)
    {
        deltaTau = dt / (2.0 * r_in * N_taumin);
    }
    else
    {
        deltaTau = dt / 100.0;
    }

    int stepCount = 0;
    int stepMax = 100000000; // Safety limit on iteration count

    while (1)
    {
        if (t_cur >= dt)
            break; // Exit when physical time target is reached

        double rho_next, v_next, t_next;
        doSingleTauStepAdaptiveLeviCivita(
            i, npts,
            rho_current, v_current,
            t_cur,    // physical time in
            deltaTau, // guess for τ
            radius_tol,
            velocity_tol,
            max_subdiv,
            grav,
            ell,
            out_type,
            &rho_next, &v_next, &t_next);

        if (t_next > dt)
        {
            // Handle overshoot case with linear interpolation
            double alpha = 0.0;
            if (fabs(t_next - t_cur) > 1.0e-30)
                alpha = (dt - t_cur) / (t_next - t_cur);

            // Interpolate to exact target time dt
            double rho_final = rho_current + alpha * (rho_next - rho_current);
            double v_final = v_current + alpha * (v_next - v_current);

            double r_fin = rho_final * rho_final;
            *r_out = r_fin;
            *v_out = v_final;
            return;
        }
        else
        {
            // Update state variables for next iteration
            rho_current = rho_next;
            v_current = v_next;
            t_cur = t_next;
        }

        // Check for iteration limit
        stepCount++;
        if (stepCount > stepMax)
        {
            // Return best available result if maximum iterations reached
            double r_fin = rho_current * rho_current;
            *r_out = r_fin;
            *v_out = v_current;
            return;
        }
    }

    // Transform final regularized state back to physical coordinates
    double r_fin = rho_current * rho_current;
    *r_out = r_fin;
    *v_out = v_current;
}

// =============================================================================
// FILE I/O AND DATA MANAGEMENT SUBSYSTEM
// =============================================================================
//
// Functions for saving, loading, and managing simulation data including:
// - Initial condition generation and I/O
// - Snapshot file management
// - Binary file format utilities

static int doReadInit = 0;            ///< Flag indicating whether to read initial conditions from file (1=yes, 0=no).
static int doWriteInit = 0;           ///< Flag indicating whether to write initial conditions to file (1=yes, 0=no).
static const char *readInitFilename = NULL; ///< Filename to read initial conditions from (if doReadInit=1).
static const char *writeInitFilename = NULL; ///< Filename to write initial conditions to (if doWriteInit=1).

// =============================================================================
// INITIAL CONDITION FILE I/O FUNCTIONS
// =============================================================================

/**
 * @brief Writes particle initial conditions to a binary file.
 * @details Stores the complete initial particle state (radius, velocity, angular momentum,
 *          original index, orientation parameter \f$\mu\f$) and the particle count (`npts`)
 *          to a binary file for later retrieval via `read_initial_conditions`.
 *          Opens the file in write binary mode ("wb"). First writes the integer `npts`,
 *          then writes the 5 double-precision values for each particle sequentially.
 *
 * @param particles [in] 2D array containing particle properties [component][particle_index]. Expected components: 0=rad, 1=vel, 2=angmom, 3=orig_idx, 4=mu.
 * @param npts [in] Number of particles to write.
 * @param filename [in] Path to the output binary file.
 *
 * @note The binary file format is: `npts` (int32), followed by `npts` records,
 *       each consisting of 7 `double` values:
 *       (radius, velocity, ang. mom., orig. index, mu, cos_phi, sin_phi).
 *       cos_phi and sin_phi are the cosine and sine of the azimuthal angle used for SIDM scattering.
 * @warning Prints an error to stderr if the file cannot be opened.
 *
 * @see read_initial_conditions
 */
static void write_initial_conditions(double **particles, int npts, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open '%s' for writing initial conditions.\n", filename);
        return;
    }

    // Write the number of particles first
    fwrite(&npts, sizeof(int), 1, fp);

    for (int i = 0; i < npts; i++)
    {
        double r_val = particles[0][i];
        double v_val = particles[1][i];
        double ell_val = particles[2][i];
        double idx_val = particles[3][i];
        double mu_val = particles[4][i];
        double cos_phi_val = particles[5][i];  // cos(phi)
        double sin_phi_val = particles[6][i];  // sin(phi)

        fwrite(&r_val, sizeof(double), 1, fp);
        fwrite(&v_val, sizeof(double), 1, fp);
        fwrite(&ell_val, sizeof(double), 1, fp);
        fwrite(&idx_val, sizeof(double), 1, fp);
        fwrite(&mu_val, sizeof(double), 1, fp);
        fwrite(&cos_phi_val, sizeof(double), 1, fp);
        fwrite(&sin_phi_val, sizeof(double), 1, fp);
    }

    fclose(fp);
    log_message("INFO", "Wrote initial conditions (%d particles) to '%s'", npts, filename);
}

/**
 * @brief Reads particle initial conditions from a binary file.
 * @details Loads the complete particle state (radius, velocity, angular momentum,
 *          original index, orientation parameter \f$\mu\f$) from a binary file previously
 *          created by `write_initial_conditions`. Verifies that the number of
 *          particles read from the file matches the expected count `npts`.
 *          Opens the file in read binary mode ("rb").
 *
 * @param particles [out] 2D array to store loaded particle properties [component][particle_index]. Must be pre-allocated with dimensions [7][npts].
 * @param npts [in] Expected number of particles to read.
 * @param filename [in] Path to the input binary file.
 *
 * @note Expects 7 variables per particle. See `write_initial_conditions` for file format details.
 * @warning Prints an error to stderr if the file cannot be opened, if the particle
 *          count does not match `npts` (returns early), or if a read error occurs.
 *
 * @see write_initial_conditions
 */
static void read_initial_conditions(double **particles, int npts, const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open '%s' for reading initial conditions.\n", filename);
        CLEAN_EXIT(1);
    }

    int file_npts;
    if (fread(&file_npts, sizeof(int), 1, fp) != 1)
    {
        fprintf(stderr, "Error: failed to read npts from '%s'.\n", filename);
        fclose(fp);
        CLEAN_EXIT(1);
    }
    if (file_npts != npts)
    {
        fprintf(stderr, "Error: file npts=%d doesn't match current npts=%d.\n", file_npts, npts);
        fclose(fp);
        CLEAN_EXIT(1);
    }

    for (int i = 0; i < npts; i++)
    {
        double r_val, v_val, ell_val, idx_val, mu_val, cos_phi_val, sin_phi_val;
        if (fread(&r_val, sizeof(double), 1, fp) != 1 || fread(&v_val, sizeof(double), 1, fp) != 1 || fread(&ell_val, sizeof(double), 1, fp) != 1 || fread(&idx_val, sizeof(double), 1, fp) != 1 || fread(&mu_val, sizeof(double), 1, fp) != 1 || fread(&cos_phi_val, sizeof(double), 1, fp) != 1 || fread(&sin_phi_val, sizeof(double), 1, fp) != 1)
        {
            fprintf(stderr, "Error: partial read at i=%d in '%s'. File must use 7-variable format (r, v, L, ID, mu, cos_phi, sin_phi).\n", i, filename);
            fclose(fp);
            CLEAN_EXIT(1);
        }

        particles[0][i] = r_val;
        particles[1][i] = v_val;
        particles[2][i] = ell_val;
        particles[3][i] = idx_val;
        particles[4][i] = mu_val;
        particles[5][i] = cos_phi_val;  // cos(phi)
        particles[6][i] = sin_phi_val;   // sin(phi)
    }

    fclose(fp);
    log_message("INFO", "Read initial conditions (%d particles) from '%s'", npts, filename);
}

// =============================================================================
// SORTING ALGORITHM CONFIGURATION
// =============================================================================
///< Default sorting algorithm identifier string. Set based on command-line options.
static const char *g_defaultSortAlg = "quadsort_parallel";

/**
 * @brief Returns a human-readable description for a sort algorithm identifier string.
 * @details Maps internal sort algorithm identifiers (e.g., "quadsort_parallel")
 *          to user-friendly descriptive names (e.g., "Parallel Quadsort").
 *          This is primarily used for display purposes in command-line output
 *          or logs, providing more context than the internal short string identifiers.
 *
 * @param sort_alg [in] The internal algorithm identifier string.
 * @return const char* A descriptive name for the algorithm. If no match is found,
 *                     the input `sort_alg` string itself is returned as a fallback.
 */
const char *get_sort_description(const char *sort_alg)
{
    if (strcmp(sort_alg, "quadsort_parallel") == 0)
        return "Parallel Quadsort";
    if (strcmp(sort_alg, "quadsort") == 0)
        return "Sequential Quadsort";
    if (strcmp(sort_alg, "insertion_parallel") == 0)
        return "Parallel Insertion Sort";
    if (strcmp(sort_alg, "insertion") == 0)
        return "Sequential Insertion Sort";
    if (strcmp(sort_alg, "parallel_radix") == 0)
        return "CPU Parallel Radix Sort";
    return sort_alg; // Default fallback.
}

// =============================================================================
// BINARY FILE I/O UTILITIES
// =============================================================================

/**
 * @brief Writes binary data to a file using a printf-like format string.
 * @details Parses a format string containing simplified specifiers (`%d`, `%f`, `%g`, `%e`).
 *          Writes corresponding arguments from the variadic list (`...`) as binary data.
 *          Integer types (`%d`) are written as `int`.
 *          Floating-point types (`%f`, `%g`, `%e`) are read as `double` from args but
 *          written as `float` to the file for storage efficiency.
 *
 * @param fp [in] File pointer to write to (must be opened in binary mode).
 * @param format [in] Format string with specifiers (%d, %f, %g, %e). Other characters are ignored.
 * @param ... [in] Variable arguments matching the format specifiers.
 * @return The number of items successfully written according to the format string.
 *
 * @note Skips optional width/precision specifiers in the format string.
 * @see fscanf_bin
 */
int fprintf_bin(FILE *fp, const char *format, ...)
{
    va_list args;
    va_start(args, format);

    int count_items = 0;
    const char *p = format;

    while (*p != '\0')
    {
        if (*p == '%')
        {
            p++; // Move past '%'
            
            // Skip format modifiers until finding type specifier
            while (*p && !strchr("dfge", *p) && !(*p == 'l'))
            {
                p++;
            }

            if (*p == 'd')
            {
                // Process integer format
                int val = va_arg(args, int);
                fwrite(&val, sizeof(val), 1, fp);
                count_items++;
            }
            else if (*p == 'f' || *p == 'g' || *p == 'e')
            {
                // Process float format (stored as 4-byte float)
                double tmp = va_arg(args, double);
                float val = (float)tmp;
                fwrite(&val, sizeof(float), 1, fp);
                count_items++;
            }
        }
        if (*p)
            p++;
    }

    va_end(args);
    return count_items;
}

/**
 * @brief Reads binary data from a file using a scanf-like format string.
 * @details Parses a format string containing simplified specifiers (`%d`, `%f`, `%g`, `%e`).
 *          Reads corresponding binary data from the file and stores it in the
 *          pointer arguments provided in the variadic list (`...`).
 *          Integer types (`%d`) are read as `int`.
 *          Floating-point types (`%f`, `%g`, `%e`) are read as `float` from the file
 *          but stored into `double*` arguments provided by the caller.
 *
 * @param fp [in] File pointer to read from (must be opened in binary mode).
 * @param format [in] Format string with specifiers (%d, %f, %g, %e). Other characters are ignored.
 * @param ... [out] Variable pointer arguments matching the format specifiers (e.g., int*, double*).
 * @return The number of items successfully read and assigned. Stops reading on first failure or EOF.
 *
 * @note Skips optional width/precision specifiers in the format string.
 * @see fprintf_bin
 */
int fscanf_bin(FILE *fp, const char *format, ...)
{
    va_list args;
    va_start(args, format);

    int count_items = 0;
    const char *p = format;

    while (*p != '\0')
    {
        if (*p == '%')
        {
            p++; // Move past '%'
            
            // Skip format modifiers (including 'l' for long) until reaching type specifier
            while (*p && !strchr("dfge", *p))
            {
                p++;
            }

            if (*p == 'd')
            {
                // Process integer format
                int *iptr = va_arg(args, int *);
                size_t nread = fread(iptr, sizeof(int), 1, fp);
                if (nread == 1)
                    count_items++;
                else
                    return count_items;
            }
            else if (*p == 'f' || *p == 'g' || *p == 'e')
            {
                // Read 4-byte float from file
                float fval;
                size_t nread = fread(&fval, sizeof(float), 1, fp);
                if (nread == 1)
                {
                    // Store in caller's double pointer with type conversion
                    double *dptr = va_arg(args, double *);
                    *dptr = (double)fval;
                    count_items++;
                }
                else
                {
                    return count_items; // Stop on read failure
                }
            }
        }
        if (*p)
            p++;
    }

    va_end(args);
    return count_items;
}

/**
 * @brief Writes binary data to a file using a printf-like format string (DOUBLE PRECISION).
 * @details Writes floating-point values as 8-byte doubles, preserving full precision.
 *          Avoids the precision loss from double→float32→double conversion.
 *
 *          Used exclusively for the double_buffer_all_particle_* files which maintain
 *          the last 4 snapshots in full precision for accurate restart/extend operations.
 *
 * @param fp [in] File pointer to write to (must be opened in binary mode).
 * @param format [in] Format string with specifiers (%d, %f, %g, %e). Other characters are ignored.
 * @param ... [in] Variable arguments matching the format specifiers.
 * @return The number of items written according to the format string.
 *
 * @note Integer types (%d) written as int (4 bytes), floats (%f/%g/%e) written as double (8 bytes).
 * @see fprintf_bin (float32 version)
 * @see fscanf_bin_dbl
 */
int fprintf_bin_dbl(FILE *fp, const char *format, ...)
{
    va_list args;
    va_start(args, format);

    int count_items = 0;
    const char *p = format;

    while (*p != '\0')
    {
        if (*p == '%')
        {
            p++; // Move past '%'

            // Skip format modifiers until finding type specifier
            while (*p && !strchr("dfge", *p) && !(*p == 'l'))
            {
                p++;
            }

            if (*p == 'd')
            {
                // Process integer format: write int (4 bytes)
                int val = va_arg(args, int);
                fwrite(&val, sizeof(val), 1, fp);
                count_items++;
            }
            else if (*p == 'f' || *p == 'g' || *p == 'e')
            {
                // Process float/double format - ALWAYS store as 8-byte double (FULL PRECISION)
                // Note: va_arg promotes float to double (always receives double type)
                double val = va_arg(args, double);
                fwrite(&val, sizeof(double), 1, fp);  // Always write 8 bytes
                count_items++;
            }
        }
        if (*p)
            p++;
    }

    va_end(args);
    return count_items;
}

/**
 * @brief Reads binary data from a file using a scanf-like format string (DOUBLE PRECISION).
 * @details Reads floating-point values as 8-byte doubles, preserving full precision.
 *          Avoids the precision loss from float32→double conversion when loading snapshot data.
 *
 *          Used to restore full-precision snapshot data during restart/extend operations,
 *          avoiding the precision loss inherent in float32 storage.
 *
 * @param fp [in] File pointer to read from (must be opened in binary mode).
 * @param format [in] Format string with specifiers (%d, %f, %g, %e). Other characters are ignored.
 * @param ... [out] Variable pointer arguments matching the format specifiers (int*, double*).
 * @return The number of items successfully read and assigned. Stops reading on first failure or EOF.
 *
 * @note Integer types (%d) read as int (4 bytes), floats (%f/%g/%e) read as double (8 bytes).
 * @see fscanf_bin (float32 version)
 * @see fprintf_bin_dbl
 */
int fscanf_bin_dbl(FILE *fp, const char *format, ...)
{
    va_list args;
    va_start(args, format);

    int count_items = 0;
    const char *p = format;

    while (*p != '\0')
    {
        if (*p == '%')
        {
            p++; // Move past '%'

            // Skip format modifiers (including 'l' for long) until reaching type specifier
            while (*p && !strchr("dfge", *p))
            {
                p++;
            }

            if (*p == 'd')
            {
                // Process integer format: read int (4 bytes)
                int *iptr = va_arg(args, int *);
                size_t nread = fread(iptr, sizeof(int), 1, fp);
                if (nread == 1)
                    count_items++;
                else
                    return count_items;
            }
            else if (*p == 'f' || *p == 'g' || *p == 'e')
            {
                // Read 8-byte double from file (FULL PRECISION - no conversion)
                double *dptr = va_arg(args, double *);
                size_t nread = fread(dptr, sizeof(double), 1, fp);
                if (nread == 1)
                {
                    count_items++;
                }
                else
                {
                    return count_items; // Stop on read failure
                }
            }
        }
        if (*p)
            p++;
    }

    va_end(args);
    return count_items;
}

/**
 * @brief Parses sub-arguments for the `--save` command-line option.
 * @details This function is called when the `--save` option is encountered during
 *          command-line argument parsing. It reads subsequent arguments (until another
 *          option starting with '-' is found, or arguments end) which specify the
 *          level or type of data to save. It then sets the corresponding global data
 *          output flags (`g_doDebug`, `g_doDynPsi`, `g_doDynRank`, `g_doAllParticleData`)
 *          based on the highest priority valid sub-argument encountered.
 *          Valid sub-arguments and their priority (lowest to highest):
 *          - "raw-data": Enables `g_doAllParticleData`.
 *          - "psi-snaps": Enables `g_doAllParticleData`, `g_doDynPsi`.
 *          - "full-snaps": Enables `g_doAllParticleData`, `g_doDynPsi`, `g_doDynRank`.
 *          - "all" or "debug-energy": Enables all flags (`g_doDebug`, `g_doDynPsi`, `g_doDynRank`, `g_doAllParticleData`).
 *
 * @param argc   [in] The total argument count from `main()`.
 * @param argv   [in] The argument array from `main()`.
 * @param pIndex [in,out] Pointer to the current index in `argv`. On input, it points to the
 *                       `--save` option. On output, it is updated to point to the last
 *                       sub-argument consumed by this function.
 * @note Exits the program with an error message if an unknown sub-argument to `--save` is found.
 */
static void parseSaveArgs(int argc, char *argv[], int *pIndex)
{
    // Use an integer priority to track the highest level of saving requested
    static int savePriority = 0; // 0=none, 1=raw, 2=psi, 3=full, 4=all/debug

    // Start checking arguments after "--save"
    int i = *pIndex + 1;

    // Process arguments until the end or another option (starting with '-') is found
    while (i < argc && argv[i][0] != '-')
    {
        const char *subarg = argv[i];

        if (strcmp(subarg, "all") == 0 || strcmp(subarg, "debug-energy") == 0)
        {
            savePriority = 4; // Highest priority
        }
        else if (strcmp(subarg, "full-snaps") == 0)
        {
            if (savePriority < 3)
                savePriority = 3;
        }
        else if (strcmp(subarg, "psi-snaps") == 0)
        {
            if (savePriority < 2)
                savePriority = 2;
        }
        else if (strcmp(subarg, "raw-data") == 0)
        {
            if (savePriority < 1)
                savePriority = 1;
        }
        else
        {
            fprintf(stderr, "Error: unknown argument to --save '%s'\n", subarg);
            exit(1);
        }

        i++;
    }

    // Set global flags based on the highest priority encountered
    switch (savePriority)
    {
    case 4: // All or debug-energy
        g_doDebug = 1;
        g_doDynPsi = 1;
        g_doDynRank = 1;
        g_doAllParticleData = 1;
        break;
    case 3: // Full-snaps.
        g_doDebug = 0;
        g_doDynPsi = 1;
        g_doDynRank = 1;
        g_doAllParticleData = 1;
        break;
    case 2: // Psi-snaps.
        g_doDebug = 0;
        g_doDynPsi = 1;
        g_doDynRank = 0;
        g_doAllParticleData = 1;
        break;
    case 1: // Raw-data.
        g_doDebug = 0;
        g_doDynPsi = 0;
        g_doDynRank = 0;
        g_doAllParticleData = 1;
        break;
    default: // No saving option specified, all flags remain 0
        break;
    }

    // Update index to point to the last processed argument
    *pIndex = i - 1;
}

/**
 * @brief Remaps particle IDs to zero-based rank order within the current particle set.
 * @details Typically used after tidal stripping where a subset of particles remains.
 *          The `orig_ids` array contains potentially non-contiguous IDs of the `n` remaining particles.
 *          Sorts these IDs and replaces each with its rank (0 to n-1) within the sorted sequence.
 *          Transforms arbitrary ID values into a compact, contiguous sequence of rank IDs.
 *          The input `orig_ids` array (which is `particles[3]` in `main`) is modified in-place.
 *
 * @param orig_ids [in,out] Pointer to an array of particle IDs (stored as doubles).
 *                        These are modified in-place to become rank IDs.
 * @param n        [in] The number of elements in the `orig_ids` array (i.e., `npts` after stripping).
 * @note Uses `qsort` and a temporary array for sorting. Exits via `exit(1)` if memory
 *       allocation for the temporary array fails.
 */
static void reassign_orig_ids_with_rank(double *orig_ids, int n)
{
    if (n <= 0) return; // Handle empty or invalid input

    double *temp = (double *)malloc(n * sizeof(double));
    if (temp == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed in reassign_orig_ids_with_rank\n");
        exit(1);
    }
    
    // Step 1: Copy original IDs to temporary array
    for (int i = 0; i < n; i++)
    {
        temp[i] = orig_ids[i];
    }
    
    // Step 2: Sort temporary array to establish rank order
    qsort(temp, n, sizeof(double), double_cmp);
    
    // Step 3: Find rank of each original ID using binary search
    for (int i = 0; i < n; i++)
    {
        int low = 0, high = n - 1, rank = -1;
        while (low <= high)
        {
            int mid = (low + high) / 2;
            if (temp[mid] == orig_ids[i])
            {
                rank = mid;
                break;
            }
            else if (temp[mid] < orig_ids[i])
            {
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }
        if (rank == -1)
        {
            // Use insertion point as fallback if exact match not found
            rank = low;
        }
        
        // Step 4: Replace original ID with its rank
        orig_ids[i] = (double)rank;
    }
    free(temp);
}

// =============================================================================
// SIGNAL PROCESSING AND FILTERING UTILITIES
// =============================================================================
//
// Advanced numerical processing utilities for density field handling including:
// - FFT-based convolution for density smoothing
// - Direct Gaussian convolution for smaller datasets
// - Signal filtering and processing functions
// =============================================================================
// FFT METHODS AND CONVOLUTION IMPLEMENTATIONS
// =============================================================================
/**
 * @brief Applies Gaussian smoothing using FFT-based convolution (thread-safe via critical section).
 * @details Smooths a density field defined on a potentially non-uniform grid
 *          (`log_r_grid`) using FFT convolution with a Gaussian kernel of width
 *          `sigma_log` (defined in log-space). Uses zero-padding to avoid
 *          wrap-around artifacts. This is generally faster than direct convolution
 *          for large `grid_size`. Assumes log_r_grid is uniformly spaced.
 *
 * @param density_grid [in] Input density grid array (values corresponding to log_r_grid).
 * @param grid_size [in] Number of points in the input grid and density arrays.
 * @param log_r_grid [in] Array of logarithmic radial grid coordinates (log10(r)). Must be uniformly spaced.
 * @param sigma_log [in] Width (standard deviation) of the Gaussian kernel in log10-space.
 * @param result [out] Output array (pre-allocated, size grid_size) for the smoothed density field.
 *
 * @note Uses FFTW library for Fast Fourier Transforms (`fftw_malloc`, `fftw_plan_dft_r2c_1d`, etc.).
 *       Requires FFTW to be installed. FFTW operations are protected by `omp critical(fftw)`.
 * @note Resulting smoothed density is clamped to a minimum value of 1e-10.
 * @warning Prints errors to stderr and returns early on memory allocation failures or
 *          FFTW plan creation failures.
 * @see direct_gaussian_convolution
 * @see gaussian_convolution
 */
void fft_gaussian_convolution(
    const double *density_grid,
    int grid_size,
    const double *log_r_grid,
    double sigma_log,
    double *result)
{
#pragma omp critical(fftw)
    {
        // Create zero-padded arrays (step 1)
        int padded_size = 2 * grid_size;

        // Allocate memory for padded input array
        double *padded_input = (double *)fftw_malloc(sizeof(double) * padded_size);
        if (!padded_input)
        {
            fprintf(stderr, "Error: Failed to allocate memory for padded_input\n");
            return;
        }

        double *padded_kernel = (double *)fftw_malloc(sizeof(double) * padded_size);
        if (!padded_kernel)
        {
            fprintf(stderr, "Error: Failed to allocate memory for padded_kernel\n");
            fftw_free(padded_input);
            return;
        }

        double *padded_output = (double *)fftw_malloc(sizeof(double) * padded_size);
        if (!padded_output)
        {
            fprintf(stderr, "Error: Failed to allocate memory for padded_output\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            return;
        }

        // Initialize padded arrays with zeros
        for (int i = 0; i < padded_size; i++)
        {
            padded_input[i] = 0.0;
            padded_kernel[i] = 0.0;
        }

        // Copy input data to first half of padded array
        for (int i = 0; i < grid_size; i++)
        {
            padded_input[i] = density_grid[i];
        }

        // Create Gaussian kernel in spatial domain
        double dlog = log_r_grid[1] - log_r_grid[0]; // Grid spacing in log space.
        double norm = 0.0;

        for (int i = 0; i < grid_size; i++)
        {
            // Distance in log space.
            double x = i * dlog;

            // Gaussian kernel centered at 0.
            double kernel_val = (1.0 / (sigma_log * sqrt(2.0 * M_PI))) *
                                exp(-0.5 * (x / sigma_log) * (x / sigma_log));

            padded_kernel[i] = kernel_val;
            norm += kernel_val;
        }

        // Normalize the kernel for unit sum
        for (int i = 0; i < grid_size; i++)
        {
            padded_kernel[i] /= norm;
        }

        // Prepare for FFT computation (step 2)
        fftw_complex *fft_input = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (padded_size / 2 + 1));
        if (!fft_input)
        {
            fprintf(stderr, "Error: Failed to allocate memory for fft_input\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            return;
        }

        fftw_complex *fft_kernel = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (padded_size / 2 + 1));
        if (!fft_kernel)
        {
            fprintf(stderr, "Error: Failed to allocate memory for fft_kernel\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            return;
        }

        fftw_complex *fft_output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (padded_size / 2 + 1));
        if (!fft_output)
        {
            fprintf(stderr, "Error: Failed to allocate memory for fft_output\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            return;
        }

        fftw_plan plan_forward_input = fftw_plan_dft_r2c_1d(padded_size, padded_input, fft_input, FFTW_ESTIMATE);
        if (!plan_forward_input)
        {
            fprintf(stderr, "Error: Failed to create forward FFTW plan for input\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            fftw_free(fft_output);
            return;
        }

        fftw_plan plan_forward_kernel = fftw_plan_dft_r2c_1d(padded_size, padded_kernel, fft_kernel, FFTW_ESTIMATE);
        if (!plan_forward_kernel)
        {
            fprintf(stderr, "Error: Failed to create forward FFTW plan for kernel\n");
            fftw_destroy_plan(plan_forward_input);
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            fftw_free(fft_output);
            return;
        }

        fftw_plan plan_backward = fftw_plan_dft_c2r_1d(padded_size, fft_output, padded_output, FFTW_ESTIMATE);
        if (!plan_backward)
        {
            fprintf(stderr, "Error: Failed to create backward FFTW plan\n");
            fftw_destroy_plan(plan_forward_input);
            fftw_destroy_plan(plan_forward_kernel);
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            fftw_free(fft_output);
            return;
        }

        // Execute forward FFTs.
        fftw_execute(plan_forward_input);
        fftw_execute(plan_forward_kernel);

        // Perform complex multiplication in frequency domain (convolution in spatial domain)
        // For each frequency component, multiply signal and kernel transforms
        for (int i = 0; i < padded_size / 2 + 1; i++)
        {
            double re_in = fft_input[i][0];   // Real part of input transform
            double im_in = fft_input[i][1];   // Imaginary part of input transform
            double re_ker = fft_kernel[i][0]; // Real part of kernel transform
            double im_ker = fft_kernel[i][1]; // Imaginary part of kernel transform

            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            fft_output[i][0] = re_in * re_ker - im_in * im_ker; // Real component
            fft_output[i][1] = re_in * im_ker + im_in * re_ker; // Imaginary component
        }

        // Execute inverse FFT.
        fftw_execute(plan_backward);

        // Apply normalization to compensate for FFTW's unnormalized inverse transform
        // FFTW's implementation requires division by array length to get properly scaled result
        for (int i = 0; i < padded_size; i++)
        {
            padded_output[i] /= padded_size; // Scale by 1/N to get normalized values
        }

        // Extract the valid portion of the convolution result
        // Only the first grid_size elements contain the actual result (rest is padding)
        for (int i = 0; i < grid_size; i++)
        {
            result[i] = padded_output[i];

            // Enforce minimum density threshold to avoid numerical instability
            // Consistent with minimum threshold in direct convolution method
            if (result[i] < 1e-10)
            {
                result[i] = 1e-10;
            }
        }

        // Release FFTW resources
        fftw_destroy_plan(plan_forward_input);
        fftw_destroy_plan(plan_forward_kernel);
        fftw_destroy_plan(plan_backward);

        fftw_free(padded_input);
        fftw_free(padded_kernel);
        fftw_free(padded_output);
        fftw_free(fft_input);
        fftw_free(fft_kernel);
        fftw_free(fft_output);
    } // End of critical section.
}

/**
 * @brief Applies Gaussian smoothing using direct convolution.
 * @details Smooths a density field defined on a potentially non-uniform grid
 *          (`log_r_grid`) using direct spatial convolution with a Gaussian kernel
 *          of width `sigma_log` (defined in log-space).
 *          For each point `i` in the output `result` array, it computes a weighted
 *          sum of the input `density_grid` values:
 *          `result[i] = sum(density_grid[j] * kernel(log_r_grid[i] - log_r_grid[j])) / sum(kernel(...))`
 *          where the `kernel` is a Gaussian function \f$G(x) = (1/(\sigma\sqrt{2\pi})) \exp(-0.5(x/\sigma)^2)\f$,
 *          with `x` being the distance `log_r_grid[i] - log_r_grid[j]` and \f$\sigma\f$ = `sigma_log`.
 *          This method is generally more accurate than FFT-based convolution, especially
 *          for non-uniform grids or near boundaries, but has a higher computational
 *          cost (\f$O(N^2)\f$) which makes it slower for large `grid_size`.
 *
 * @param density_grid [in] Input density grid array (values corresponding to log_r_grid).
 * @param grid_size [in] Number of points in the input grid and density arrays.
 * @param log_r_grid [in] Array of logarithmic radial grid coordinates (log10(r)). Can be non-uniformly spaced.
 * @param sigma_log [in] Width (standard deviation) of the Gaussian kernel in log10-space.
 * @param result [out] Output array (pre-allocated, size grid_size) for the smoothed density field.
 *
 * @note Resulting smoothed density is clamped to a minimum value of 1e-10.
 *       This function is inherently thread-safe as it only reads inputs and writes
 *       to distinct elements of the output array without shared intermediate state.
 * @see fft_gaussian_convolution
 * @see gaussian_convolution
 */
void direct_gaussian_convolution(
    const double *density_grid,
    int grid_size,
    const double *log_r_grid,
    double sigma_log,
    double *result)
{
    // Precompute normalization factor for Gaussian kernel if sigma is valid
    double kernel_norm_factor = (sigma_log > 1e-15) ? (1.0 / (sigma_log * sqrt(2.0 * M_PI))) : 1.0;
    double sig_sq_inv = (sigma_log > 1e-15) ? (1.0 / (sigma_log * sigma_log)) : 0.0; // Avoid division by zero

    // Perform direct convolution
    for (int i = 0; i < grid_size; i++)
    {
        double sum = 0.0;
        double norm = 0.0;

        for (int j = 0; j < grid_size; j++)
        {
            double dlog_r = log_r_grid[i] - log_r_grid[j];
            double kernel = (sigma_log > 1e-15) ? 
                (kernel_norm_factor * exp(-0.5 * dlog_r * dlog_r * sig_sq_inv)) : 
                ((i == j) ? 1.0 : 0.0); // Delta function if sigma=0

            sum += density_grid[j] * kernel;
            norm += kernel;
        }

        // Normalize the result, handle potential division by zero if norm is too small
        result[i] = (norm > 1e-15) ? (sum / norm) : density_grid[i]; // Fallback to original value if norm is zero

        // Clamp to minimum density
        if (result[i] < 1e-10)
        {
            result[i] = 1e-10;
        }
    }
}

/**
 * @brief Controls the convolution method selection for density smoothing.
 * @details Global flag that determines whether to use direct convolution (1)
 *          or FFT-based convolution (0). Direct convolution is more accurate
 *          but slower for large grids, while FFT-based convolution is faster
 *          but may introduce artifacts.
 *
 * @note External declaration, defined elsewhere.
 * @see gaussian_convolution
 */
extern int debug_direct_convolution;

/**
 * @brief Performs Gaussian smoothing by selecting the appropriate convolution method.
 * @details Acts as a routing function that delegates density smoothing to either
 *          `direct_gaussian_convolution` or `fft_gaussian_convolution` based on the
 *          value of the global `debug_direct_convolution` flag.
 *          - If `debug_direct_convolution` is non-zero, `direct_gaussian_convolution`
 *            is called (more accurate, \f$O(N^2)\f$ complexity, suitable for smaller or
 *            non-uniform grids).
 *          - If `debug_direct_convolution` is zero (default), `fft_gaussian_convolution`
 *            is called (faster for large grids, O(N log N) complexity, requires
 *            uniformly spaced logarithmic grid).
 *
 * @param density_grid [in] Input density grid array (values corresponding to log_r_grid).
 * @param grid_size [in] Number of points in the input grid and density arrays.
 * @param log_r_grid [in] Array of logarithmic radial grid coordinates (log10(r)). Must be uniform if FFT is used.
 * @param sigma_log [in] Width (standard deviation) of the Gaussian kernel in log10-space.
 * @param result [out] Output array (pre-allocated, size grid_size) for the smoothed density field.
 *
 * @see direct_gaussian_convolution
 * @see fft_gaussian_convolution
 * @see debug_direct_convolution
 */
void gaussian_convolution(
    const double *density_grid,
    int grid_size,
    const double *log_r_grid,
    double sigma_log,
    double *result)
{
    // Select convolution method based on global configuration flag.
    if (debug_direct_convolution)
    {
        // Use direct spatial-domain convolution.
        direct_gaussian_convolution(density_grid, grid_size, log_r_grid, sigma_log, result);
    }
    else
    {
        // Use FFT-based frequency-domain convolution.
        fft_gaussian_convolution(density_grid, grid_size, log_r_grid, sigma_log, result);
    }
}

// =============================================================================
// RESTART AND RECOVERY MANAGEMENT
// =============================================================================

/**
 * @brief Finds the index of the last successfully processed and written snapshot in restart mode.
 * @details This function is called when `--restart` is active. It checks for the existence
 *          and basic integrity of snapshot data files (e.g., `Rank_Mass_Rad_VRad_unsorted_t%05d.dat`
 *          and `Rank_Mass_Rad_VRad_sorted_t%05d.dat`) corresponding to the timesteps listed
 *          in `snapshot_steps`.
 *          Integrity is checked by comparing file sizes against those of the first snapshot's
 *          files (snapshot_steps[0]), allowing for a small percentage tolerance (typically +/- 5%).
 *          The function aims to determine from which snapshot index `s` (in `snapshot_steps`)
 *          the post-processing (e.g., Rank file generation) should resume.
 *          It assumes `g_file_suffix` is correctly set to identify the relevant run's files.
 *
 * @param snapshot_steps [in] Array of integer timestep numbers for which snapshot data
 *                           was expected to be written (often these are the `dtwrite` intervals).
 * @param noutsnaps      [in] The total number of snapshot indices in the `snapshot_steps` array.
 *
 * @return int The index `s` into `snapshot_steps` corresponding to the *first snapshot that
 *             needs to be processed* (i.e., last valid snapshot index + 1).
 *             - Returns -1 if no valid snapshot files are found (implying processing should start from index 0).
 *             - Returns -2 if all expected snapshot files for all unique snapshot numbers exist and
 *               appear valid (implying no further snapshot processing is needed for this phase).
 *             Prints status messages to stdout and logs warnings/errors.
 */
static int find_last_processed_snapshot(int *snapshot_steps, int noutsnaps)
{
    // Validate input parameters before proceeding
    if (snapshot_steps == NULL || noutsnaps <= 0)
    {
        printf("ERROR: Invalid snapshot_steps or noutsnaps in find_last_processed_snapshot\n");
        return -1; // Start from beginning.
    }

    int last_valid_snap = -1;
    int last_valid_index = -1;
    long reference_unsorted_size = 0;
    long reference_sorted_size = 0;

    printf("Restart mode: Checking for existing data products with suffix '%s'...\n\n", g_file_suffix);

    // Initialize snapshot tracking variables
    int all_snapshots_checked = 1; // Assume all checked until finding problem
    int checked_count = 0;         // Count of files checked
    int unique_snapshots = 0;      // Count of unique snapshot numbers

    // Calculate number of unique snapshot numbers (may differ from noutsnaps)
    int total_writes = 0; // Determine max snapshot number.
    for (int i = 0; i < noutsnaps; i++)
    {
        if (snapshot_steps[i] > total_writes)
        {
            total_writes = snapshot_steps[i];
        }
    }
    total_writes++; // Convert from 0-based index to count.

    {
        int *seen = (int *)calloc(total_writes, sizeof(int)); // Use calloc to initialize to 0.
        if (seen)
        {
            for (int i = 0; i < noutsnaps; i++)
            {
                int snap = snapshot_steps[i];
                if (snap >= 0 && snap < total_writes && !seen[snap])
                {
                    seen[snap] = 1;
                    unique_snapshots++;
                }
            }
            free(seen);
        }
        else
        {
            unique_snapshots = noutsnaps; // Fallback if memory allocation fails.
        }
    }

    int total_expected = unique_snapshots * 2; // Total files expected (unsorted + sorted for each unique snapshot).
    printf("Detected %d unique snapshot numbers out of %d indices. Expecting %d files total.\n\n",
           unique_snapshots, noutsnaps, total_expected);

    // Check for reference snapshot file
    if (noutsnaps > 0)
    {
        int snap = snapshot_steps[0];
        char fname_unsorted[256];
        char fname_sorted[256];

        char base_filename_1[256];
        snprintf(base_filename_1, sizeof(base_filename_1), "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_1, 1, fname_unsorted, sizeof(fname_unsorted));
        char base_filename_2[256];
        snprintf(base_filename_2, sizeof(base_filename_2), "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_2, 1, fname_sorted, sizeof(fname_sorted));

        FILE *fun = fopen(fname_unsorted, "rb");
        FILE *fsort = fopen(fname_sorted, "rb");

        if (fun && fsort)
        {
            // Get file sizes.
            fseek(fun, 0, SEEK_END);
            reference_unsorted_size = ftell(fun);
            fseek(fsort, 0, SEEK_END);
            reference_sorted_size = ftell(fsort);

            // If both files have non-zero size, use as reference.
            if (reference_unsorted_size > 0 && reference_sorted_size > 0)
            {
                printf("Found reference file sizes from snapshot %d (unsorted: %ld bytes, sorted: %ld bytes)\n\n",
                       snap, reference_unsorted_size, reference_sorted_size);
                last_valid_snap = snap;
                last_valid_index = 0;
                checked_count += 2; // Count these two files.
            }

            fclose(fun);
            fclose(fsort);
        }
        else
        {
            // Only close if non-NULL.
            if (fun)
                fclose(fun);
            if (fsort)
                fclose(fsort);

            // If the very first files cannot be opened, processing should not continue.
            printf("Could not find initial snapshot files for index 0. Will start from the beginning.\n");
            return -1;
        }
    }
    else
    {
        // No snapshots to check.
        return -1;
    }

    // If reference sizes cannot be found, proper validation is not possible.
    if (reference_unsorted_size == 0 || reference_sorted_size == 0)
    {
        printf("Could not find valid reference file sizes. Will start from the beginning.\n\n");
        return -1;
    }

    printf("Checking all %d snapshots for completeness...\n\n", noutsnaps);

    // Check all snapshots except index 0 (already validated)
    for (int i = 1; i < noutsnaps; i++)
    {
        int snap = snapshot_steps[i];

        // If this snapshot index maps to the same snapshot number as a previous index,.
        // Possible duplicate snapshot numbers in calculation
        if (snap == snapshot_steps[0])
        {
            printf("Warning: Duplicate snapshot number %d (index 0 and %d)\n", snap, i);
        }

        char fname_unsorted[256];
        char fname_sorted[256];

        char base_filename_3[256];
        snprintf(base_filename_3, sizeof(base_filename_3), "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_3, 1, fname_unsorted, sizeof(fname_unsorted));

        char base_filename_4[256];
        snprintf(base_filename_4, sizeof(base_filename_4), "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_4, 1, fname_sorted, sizeof(fname_sorted));

        FILE *fun = fopen(fname_unsorted, "rb");
        FILE *fsort = fopen(fname_sorted, "rb");

        if (fun && fsort)
        {
            // Get file sizes.
            fseek(fun, 0, SEEK_END);
            long unsorted_size = ftell(fun);
            fseek(fsort, 0, SEEK_END);
            long sorted_size = ftell(fsort);

            // Compare to reference sizes.
            // Allow for some small variation (±5%).
            double unsorted_ratio = (double)unsorted_size / reference_unsorted_size;
            double sorted_ratio = (double)sorted_size / reference_sorted_size;

            if (unsorted_size > 0 && sorted_size > 0 &&
                unsorted_ratio >= 0.95 && unsorted_ratio <= 1.05 &&
                sorted_ratio >= 0.95 && sorted_ratio <= 1.05)
            {
                last_valid_snap = snap;
                last_valid_index = i;
                log_message("INFO", "Verified valid files for snapshot %d (unsorted: %ld bytes, sorted: %ld bytes)",
                            snap, unsorted_size, sorted_size);
                checked_count += 2; // Count both files as checked.
            }
            else
            {
                log_message("WARNING", "Found invalid files for snapshot %d (unsorted: %ld bytes, sorted: %ld bytes) - expected ~%ld and ~%ld bytes",
                            snap, unsorted_size, sorted_size, reference_unsorted_size, reference_sorted_size);
                fclose(fun);
                fclose(fsort);
                all_snapshots_checked = 0; // Not all snapshots are valid.
                printf("Invalid snapshot found. Will restart from this point.\n\n");
                break; // Stop at first invalid snapshot.
            }

            fclose(fun);
            fclose(fsort);
        }
        else
        {
            // File does not exist for this snapshot
            log_message("WARNING", "Missing files for snapshot %d", snap);
            // Only close if non-NULL.
            if (fun)
                fclose(fun);
            if (fsort)
                fclose(fsort);
            all_snapshots_checked = 0; // Not all snapshots are valid.
            printf("Missing snapshot found. Will restart from this point.\n");
            break; // Stop at first missing snapshot.
        }
    }

    printf("End of file check: last_valid_snap=%d, last_valid_index=%d, noutsnaps=%d, all_snapshots_checked=%d\n\n",
           last_valid_snap, last_valid_index, noutsnaps, all_snapshots_checked);
    printf("Files checked: %d out of %d expected (unique snapshots: %d)\n\n",
           checked_count, total_expected, unique_snapshots);

    if (last_valid_snap == -1)
    {
        printf("No valid data products found. Will start from the beginning.\n\n");
        return -1;
    }

    // Check if more files verified than expected - indicates duplicate snapshot numbers
    if (checked_count > total_expected)
    {
        printf("WARNING: Checked more files (%d) than expected (%d) - likely due to duplicate snapshot numbers.\n",
               checked_count, total_expected);
    }

    // Determine whether to proceed with Rank file generation
    if (unique_snapshots == 1 && all_snapshots_checked)
    {
        // Special case: Only one unique snapshot number (usually 0) and valid
        printf("WARNING: Only one unique snapshot number found (%d). Check snapshot calculation logic.\n",
               snapshot_steps[0]);
        printf("Only 1 Rank file (snapshot %d) exists. Starting from the beginning to create all files.\n",
               snapshot_steps[0]);
        return -1; // Start from beginning.
    }
    // Only say "all files exist" if:
    // 1. More than one unique snapshot exists, and
    // 2. All expected files checked and validated
    else if (unique_snapshots > 1 && checked_count >= total_expected && all_snapshots_checked)
    {
        printf("All %d data product files (for %d unique snapshots) already exist and are valid. Nothing to do.\n\n",
               checked_count, unique_snapshots);
        return -2; // Special code for "all done".
    }
    else
    {
        printf("Will restart processing from snapshot index %d (after snapshot %d)\n\n",
               last_valid_index + 1, last_valid_snap);
        return last_valid_index + 1; // Return the index of the NEXT snapshot to process.
    }
}

/**
 * @brief Main entry point for the n-sphere dark matter simulation program.
 * @details Orchestrates the overall simulation workflow:
 *          1. Parses command-line arguments using a flexible option system
 *          2. Sets up initial conditions (particle distribution, system parameters)
 *          3. Executes the selected integration method for trajectory evolution
 *          4. Outputs data products (particle states, phase diagrams, energy tracking)
 *          5. Handles restart/resume functionality when requested
 *
 * @param argc [in] Standard argument count from the command line.
 * @param argv [in] Standard array of argument strings from the command line.
 * @return Exit code: 0 for successful execution, non-zero for errors.
 *
 * @note This is a highly parallelized application that takes advantage of OpenMP
 *       when available. Performance scales with the number of available cores.
 * @warning Some simulation configurations can be very memory-intensive. For large
 *          particle counts (\f$>10^6\f$), ensure sufficient RAM is available.
 */

// Forward declarations for structures and functions used in diagnostic loop
struct RrPsiPair
{
    double rr;  ///< Radius value or x-axis value for sorting
    double psi; ///< Corresponding potential value or y-axis value
};

int compare_pair_by_first_element(const void *a, const void *b);

/**
 * @brief Check if an array is strictly monotonically increasing.
 * 
 * @param arr Array to check
 * @param n Number of elements
 * @param name Name of the array for debug messages
 * @return 1 if strictly monotonic, 0 otherwise
 */
static int check_strict_monotonicity(const double *arr, int n, const char *name) {
    int i;
    for (i = 1; i < n; i++) {
        if (arr[i] <= arr[i-1]) {
            fprintf(stderr, "MONOTONICITY_CHECK FAILED for '%s': arr[%d]=%.17e <= arr[%d]=%.17e\n", 
                       name, i, arr[i], i-1, arr[i-1]);
            fflush(stderr);
            // Print a few surrounding values for context
            for (int k = (i > 2 ? i - 2 : 0); k < (i + 3 < n ? i + 3 : n); k++) {
                fprintf(stderr, "  Context: %s[%d] = %.17e\n", name, k, arr[k]);
                fflush(stderr);
            }
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Handles the SIDM scattering phase for a single timestep.
 * @details Checks if SIDM is enabled and if not in a bootstrap phase that should skip SIDM. 
 *          If proceeding, it resets particle scatter flags, selects serial or parallel execution 
 *          based on `g_sidm_execution_mode`, calls the appropriate core scattering function 
 *          (`perform_sidm_scattering_serial` or `perform_sidm_scattering_parallel`), 
 *          updates the global total scatter count, and logs debug information if scatters occurred
 *          and debugging is enabled. The core scattering functions are responsible for updating
 *          the `g_particle_scatter_state` flags for particles that underwent scattering.
 * 
 * @param particles         [in,out] The main particle data array: `particles[component][current_sorted_index]`.
 *                              Modified in-place with post-scattering velocities/angular momenta.
 * @param npts              [in] Total number of particles.
 * @param dt                [in] The simulation timestep (Myr).
 * @param current_sim_time  [in] The current simulation time at the beginning of this step (Myr).
 * @param active_profile_rc [in] The scale radius (kpc) of the currently active profile (NFW, Cored, or Hernquist),
 *                              passed to `sigmatotal`.
 * @param current_method_display_num [in] The user-facing display number of the current integration method (for logging).
 * @param bootstrap_phase_active [in] Flag (0 or 1) indicating if a bootstrap phase (e.g., for Adams-Bashforth)
 *                               is active. If 1, SIDM scattering is skipped for this step.
 * @note This function modifies the `particles` array in-place.
 * @note It uses global variables: `g_enable_sidm_scattering`, `g_sidm_execution_mode`, 
 *       `g_rng_per_thread`, `g_max_omp_threads_for_rng`, `g_rng`, `g_total_sidm_scatters`,
 *       `g_active_halo_mass`, `g_doDebug`, and `g_particle_scatter_state`.
 */
static void handle_sidm_step(double **particles, int npts, double dt, double current_sim_time,
                             double active_profile_rc, int current_method_display_num,
                             int bootstrap_phase_active, int *current_scatter_counts) 
{
    if (!g_enable_sidm_scattering || bootstrap_phase_active) {
        return; // Skip SIDM if disabled or in a bootstrap phase that should skip SIDM
    }

    long long Nscatters_in_this_step = 0;

    if (g_sidm_execution_mode == 1) { // Parallel
        #ifdef _OPENMP
            if (g_rng_per_thread != NULL && g_max_omp_threads_for_rng > 0) {
                // Graph coloring algorithm: processes particles in 11 color groups sequentially
                perform_sidm_scattering_parallel_graphcolor(particles, npts, dt, current_sim_time,
                                               g_rng_per_thread, g_max_omp_threads_for_rng,
                                               &Nscatters_in_this_step, g_active_halo_mass, active_profile_rc, current_scatter_counts);
            } else {
                log_message("ERROR", "SIDM Parallel mode selected but per-thread RNGs not available. Skipping SIDM for step.");
                Nscatters_in_this_step = 0;
            }
        #else
            // Serial fallback if OpenMP not compiled but parallel mode selected
            log_message("WARNING", "SIDM Parallel mode selected but OpenMP not enabled. Running SIDM serially.");
            gsl_rng *rng_for_serial_fallback = (g_rng_per_thread != NULL && g_rng_per_thread[0] != NULL) ? g_rng_per_thread[0] : g_rng;
            if (rng_for_serial_fallback != NULL) {
                perform_sidm_scattering_serial(particles, npts, dt, current_sim_time, rng_for_serial_fallback,
                                             &Nscatters_in_this_step, g_active_halo_mass, active_profile_rc, current_scatter_counts);
            } else {
                log_message("ERROR", "SIDM Serial fallback: No suitable RNG available. Skipping SIDM for step.");
                Nscatters_in_this_step = 0;
            }
        #endif
    } else { // Serial SIDM execution
        gsl_rng *rng_for_serial = (g_rng_per_thread != NULL && g_rng_per_thread[0] != NULL) ? g_rng_per_thread[0] : g_rng;
        if (rng_for_serial != NULL) {
            perform_sidm_scattering_serial(particles, npts, dt, current_sim_time, rng_for_serial,
                                         &Nscatters_in_this_step, g_active_halo_mass, active_profile_rc, current_scatter_counts);
        } else {
            log_message("ERROR", "SIDM Serial mode: No suitable RNG available. Skipping SIDM for step.");
            Nscatters_in_this_step = 0;
        }
    }
    
    g_total_sidm_scatters += Nscatters_in_this_step;
    
    if (Nscatters_in_this_step > 0 && g_doDebug) {
        log_message("DEBUG", "Method %d Step: %lld SIDM scatters this step, %lld total", 
                    current_method_display_num, Nscatters_in_this_step, g_total_sidm_scatters);
    }
}

/**
 * @brief Gets the available disk space for the filesystem containing the given path.
 * @details This function uses platform-specific APIs to determine the free space
 *          available to the current user on the filesystem where `path` resides.
 *          On Windows, it uses `GetDiskFreeSpaceEx`. On POSIX-compliant systems
 *          (Linux, macOS), it uses `statvfs`.
 *
 * @param path [in] A path to a file or directory on the filesystem to check. For Windows, this can be a root directory like "C:\\". For POSIX, any path within the target filesystem, e.g., "data/".
 * @return long long Available disk space in bytes. Returns -1 on error or if the
 *                   functionality is not implemented for the current platform.
 */
long long get_available_disk_space(const char *path) {
    #ifdef _WIN32
        ULARGE_INTEGER freeBytesAvailable;
        if (GetDiskFreeSpaceEx(path, &freeBytesAvailable, NULL, NULL)) {
            return (long long)freeBytesAvailable.QuadPart;
        }
    #else
        struct statvfs stat;
        if (statvfs(path, &stat) == 0) {
            return (long long)stat.f_bavail * (long long)stat.f_frsize;
        }
    #endif
    return -1;
}

/**
 * @brief Prompts the user with a yes/no question and reads their response from stdin.
 * @details Displays the given `prompt` string followed by "[y/N]: ".
 *          Reads a line of input from the user.
 *          - Returns 1 (yes) if the first character of the input is 'y' or 'Y'.
 *          - Returns 0 (no) if the first character is 'n', 'N', or if the input is
 *            an empty line (user just pressed Enter, defaulting to No).
 *          - If any other input is received, the prompt is repeated.
 *          Handles potential EOF or read errors by defaulting to No.
 *
 * @param prompt [in] The question/prompt message to display to the user.
 * @return int 1 if the user confirms (yes), 0 otherwise (no/default).
 */
int prompt_yes_no(const char *prompt) {
    int response;
    
    while (1) {
        printf("%s [y/N]: ", prompt);
        fflush(stdout);
        
        // Read entire line
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
            // EOF or error - treat as 'N'
            return 0;
        }
        
        // Check first character
        response = buffer[0];
        
        if (response == 'y' || response == 'Y') {
            return 1;
        } else if (response == 'n' || response == 'N' || response == '\n') {
            // Empty line (Enter key) or explicit 'n'/'N'
            return 0;
        }
        // Any other input repeats the prompt
    }
}

/**
 * @brief Safe wrapper for hypergeometric function with analytical special cases.
 * @details Handles boundary cases where GSL has singularities:
 *          - \f$\beta=0.5\f$: \f$a=0\f$, so \f${}_2F_1(0,b,c,x) = 1\f$ (exact)
 *          - \f$\beta=-0.5\f$: \f${}_2F_1(2,6,4,x) = (3x^2-10x+10)/(5(x-1)^2)\f$ (analytical)
 *          - Otherwise: Uses GSL implementation
 *
 * @param beta Anisotropy parameter
 * @param E_tilde Dimensionless energy (must be < 1 for convergence)
 * @return Hypergeometric function value
 */
static inline double hyperg_2F1_safe(double beta, double E_tilde) {
    double hyperg_a = 1.0 - 2.0*beta;
    double hyperg_b = 5.0 - 2.0*beta;
    double hyperg_c = 3.5 - beta;

    if (beta == 0.5) {
        // Special case: β=0.5 → a=0 → ₂F₁(0,b,c,x) = 1
        return 1.0;
    } else if (beta == -0.5) {
        // Special case: β=-0.5 → ₂F₁(2,6,4,x) = (3x²/10 - x + 1) / (x-1)⁴
        double x = E_tilde;
        double numerator = 3.0*x*x/10.0 - x + 1.0;
        double x_minus_1 = x - 1.0;
        double denominator = x_minus_1 * x_minus_1 * x_minus_1 * x_minus_1;  // (x-1)⁴
        return numerator / denominator;
    } else if (hyperg_c > 0.0) {
        return gsl_sf_hyperg_2F1(hyperg_a, hyperg_b, hyperg_c, E_tilde);
    } else {
        return 0.0;
    }
}

/**
 * @brief Corrects negative \f$f(E)\f$ regions by linear interpolation and reconstructs \f$I(E)\f$.
 * @details Given \f$f(E)\f$ values with some negative regions, linearly interpolates \f$f(E)\f$
 *          over consecutive negative ranges, then integrates to reconstruct \f$I(E)\f$.
 *
 * @param fE_values Array of \f$f(E)\f$ values (modified in-place)
 * @param is_negative Array marking negative \f$f(E)\f$ points (1 = negative, 0 = positive)
 * @param E_array Array of energy values
 * @param I_array Array to store reconstructed \f$I(E)\f$ values (output)
 * @param n_points Number of points
 * @param label Label for logging ("f(E)" or "f(Q)")
 * @return Number of ranges corrected
 */
static int correct_negative_fE_and_rebuild_IE(
    double *fE_values,
    int *is_negative,
    double *E_array,
    double *I_array,
    int n_points,
    const char *label)
{
    // Linearly interpolate f(E) over consecutive negative ranges
    int i = 0;
    int num_ranges = 0;

    while (i < n_points) {
        if (is_negative[i]) {
            int range_start = i;

            // Find end of consecutive negative range
            while (i < n_points && is_negative[i]) {
                i++;
            }
            int range_end = i - 1;

            // Interpolate between valid neighbors
            int i_before = range_start - 1;
            int i_after = range_end + 1;

            if (i_before >= 0 && i_after < n_points &&
                fE_values[i_before] >= 0 && fE_values[i_after] >= 0) {

                double E_before = E_array[i_before];
                double E_after = E_array[i_after];
                double fE_before = fE_values[i_before];
                double fE_after = fE_values[i_after];

                // Linear interpolation
                for (int j = range_start; j <= range_end; j++) {
                    double alpha = (E_array[j] - E_before) / (E_after - E_before);
                    fE_values[j] = fE_before + alpha * (fE_after - fE_before);
                }

                num_ranges++;
                log_message("DEBUG", "Corrected f(%s) range [%d-%d] (%d points) interpolating between %d and %d",
                            label, range_start, range_end, range_end - range_start + 1, i_before, i_after);
            }
        } else {
            i++;
        }
    }

    // Reconstruct I(E) from corrected f(E) by trapezoidal integration
    I_array[0] = 0.0;
    for (int i = 1; i < n_points; i++) {
        double dE = E_array[i] - E_array[i-1];
        double fE_avg = 0.5 * (fE_values[i-1] + fE_values[i]);
        I_array[i] = I_array[i-1] + fE_avg * (sqrt(8.0) * PI * PI) * dE;
    }

    return num_ranges;
}

/**
 * @brief Checks for negative distribution function values and corrects minor artifacts.
 * @details Evaluates \f$f(E) = dI/dE\f$ at each point in the original arrays.
 *          If negative \f$f(E)\f$ values are found (where \f$I(E)\f$ decreases):
 *          - For < 20 negative points: Removes them, rebuilds spline, warns (verbose mode only)
 *          - For \f$\geq 20\f$ negative points: Prompts user to abort or continue (verbose mode only)
 *
 * @param fofE_interp_ptr Pointer to GSL interpolator (rebuilt if correction applied)
 * @param fofE_acc_ptr Pointer to GSL accelerator (rebuilt if correction applied)
 * @param E_array Array of energy values (modified in-place during correction)
 * @param I_array Array of \f$I(E)\f$ values (modified in-place during correction)
 * @param n_points_ptr Pointer to number of points (updated during correction)
 * @param label String label "f(E)" or "f(Q)" for display
 * @param profile_name String name of profile (e.g., "NFW", "Cored")
 * @param verbose If 1: show warnings/prompts. If 0: silent correction (diagnostic mode)
 * @return int 1 to abort, 0 to continue
 */
static int check_and_warn_negative_fQ(
    gsl_interp **fofE_interp_ptr,
    gsl_interp_accel **fofE_acc_ptr,
    double *E_array,
    double *I_array,
    int *n_points_ptr,
    const char *label,
    const char *profile_name,
    int verbose)
{
    gsl_interp *fofE_interp = *fofE_interp_ptr;
    gsl_interp_accel *fofE_acc = *fofE_acc_ptr;
    int n_points = *n_points_ptr;

    if (!fofE_interp || !fofE_acc || !E_array || !I_array || n_points < 2) {
        log_message("WARNING", "Skipping negative %s check - interpolator not initialized", label);
        return 0; // Continue without checking
    }

    // Calculate f(E) = dI/dE at ALL original E_array points using spline derivative
    double *fE_values = (double *)malloc(n_points * sizeof(double));
    int *is_negative_fE = (int *)calloc(n_points, sizeof(int));
    if (!fE_values || !is_negative_fE) {
        log_message("ERROR", "Failed to allocate fE arrays");
        if (fE_values) free(fE_values);
        if (is_negative_fE) free(is_negative_fE);
        return 0;
    }

    // Evaluate f(E) = dI/dE / (√8 π²) at each original point
    for (int i = 0; i < n_points; i++) {
        double deriv = gsl_interp_eval_deriv(fofE_interp, E_array, I_array, E_array[i], fofE_acc);
        fE_values[i] = deriv / (sqrt(8.0) * PI * PI);
    }

    // Exclude first and last 1% when checking for negative f(E) (avoid edge artifacts)
    int start_check = (int)(0.01 * n_points);
    int end_check = n_points - (int)(0.01 * n_points);
    if (start_check < 1) start_check = 1;
    if (end_check > n_points - 1) end_check = n_points - 1;

    // Find negative f(E) points in interior region
    int negative_count = 0;
    int first_negative_idx = -1;

    for (int i = start_check; i < end_check; i++) {
        if (fE_values[i] < -1e-15) {  // Negative beyond numerical noise
            is_negative_fE[i] = 1;
            if (negative_count == 0) {
                first_negative_idx = i;
            }
            negative_count++;
        }
    }

    log_message("INFO", "%s: Evaluated f(%s) at %d points - found %d negative in interior (indices %d to %d)",
                profile_name, label, n_points, negative_count, start_check, end_check);

    double first_negative_E = (first_negative_idx >= 0) ? E_array[first_negative_idx] : 0.0;
    double first_negative_df = (first_negative_idx >= 0) ? fE_values[first_negative_idx] : 0.0;

    if (negative_count > 0) {
        int total_checked = end_check - start_check;
        double negative_fraction = (double)negative_count / (double)total_checked;

        // Threshold for distinguishing numerical artifacts from real physics problems
        const int MINOR_ARTIFACT_THRESHOLD = 20;  // Values below threshold auto-correct silently

        if (negative_count < MINOR_ARTIFACT_THRESHOLD) {
            // Minor numerical artifacts - warn but auto-continue after delay
            if (verbose) {
                printf("\n");
                printf("===================================================================================================\n");
                printf("NOTICE: Minor Negative %s Values Detected (Likely Numerical Artifact)\n", label);
                printf("===================================================================================================\n");
                printf("\n");
                printf("Profile: %s\n", profile_name);
                printf("Distribution: %s\n", label);
                printf("\n");
                printf("Found %d negative %s values out of %d points (%.4f%%).\n",
                       negative_count, label, total_checked, negative_fraction * 100.0);
                printf("First negative at: E = %.6e, %s = %.6e\n",
                       first_negative_E, label, first_negative_df);
                printf("\n");
                printf("Assessment: This small number of negative values is likely due to numerical\n");
                printf("            artifacts at integration boundaries rather than a fundamental\n");
                printf("            physics incompatibility.\n");
                printf("\n");
                printf("Action:     Proceeding automatically with IC generation.\n");
                printf("            Negative regions will be corrected via linear interpolation.\n");
                printf("\n");
                printf("Waiting 5 seconds... [");
                fflush(stdout);

                // Progress bar: 20 blocks, update every 0.25s (20 updates = 5 seconds)
                const char *filled = "█";
                const char *empty = "░";
                int total_blocks = 20;

                for (int block = 0; block < total_blocks; block++) {
                    // Print filled blocks
                    for (int j = 0; j < total_blocks; j++) {
                        printf("%s", (j <= block) ? filled : empty);
                    }
                    printf("]");
                    fflush(stdout);

                    usleep(250000);  // 250ms = 0.25 seconds

                    // Return to start of line and reprint
                    if (block < total_blocks - 1) {
                        printf("\r");
                        printf("Waiting 5 seconds... [");
                    }
                }
                printf("\n");
                printf("================================================================================\n");
                printf("\n");
            }

            log_message("INFO", "%s: Detected %d minor negative %s values (%.4f%%), correcting with linear interpolation",
                        profile_name, negative_count, label, negative_fraction * 100.0);

            // Apply correction using helper function
            int num_ranges_corrected = correct_negative_fE_and_rebuild_IE(
                fE_values, is_negative_fE, E_array, I_array, n_points, label
            );

            // Rebuild the spline with corrected I(E) values using LINEAR interpolation
            // Linear interpolation prevents cubic overshoot that can reintroduce negativity
            gsl_interp_free(fofE_interp);
            gsl_interp_accel_free(fofE_acc);

            fofE_interp = gsl_interp_alloc(gsl_interp_linear, n_points);
            fofE_acc = gsl_interp_accel_alloc();
            gsl_interp_init(fofE_interp, E_array, I_array, n_points);

            *fofE_interp_ptr = fofE_interp;
            *fofE_acc_ptr = fofE_acc;

            if (verbose) {
                printf("Corrected %d points. Continuing...\n\n", negative_count);
            }
            log_message("INFO", "%s: Corrected %d negative %s points in %d ranges, rebuilt spline",
                        profile_name, negative_count, label, num_ranges_corrected);

            free(fE_values);
            free(is_negative_fE);
            return 0; // Continue with corrected spline

        } else {
            // Significant negative values - this is a real physics problem
            if (verbose) {
                printf("\n");
                printf("================================================================================\n");
                printf("WARNING: NEGATIVE DISTRIBUTION FUNCTION DETECTED\n");
                printf("================================================================================\n");
                printf("\n");
                printf("Profile: %s\n", profile_name);
                printf("Distribution: %s\n", label);
                printf("\n");
                printf("The Eddington inversion has produced NEGATIVE %s values in some regions.\n", label);
                printf("This indicates the model does NOT represent physically valid, stationary,\n");
                printf("self-gravitating initial conditions.\n");
                printf("\n");
                printf("Negative values found: %d out of %d points (%.2f%%)\n",
                       negative_count, total_checked, negative_fraction * 100.0);
                printf("First negative at:     E = %.6e, %s = %.6e\n",
                       first_negative_E, label, first_negative_df);
                printf("\n");
                printf("Implications:\n");
                printf("  - Particle sampling will treat negative %s regions as zero probability\n", label);
                printf("  - Some energy ranges will have NO particles assigned\n");
                printf("  - The initial conditions will NOT be in true equilibrium\n");
                printf("  - Evolution may show non-physical relaxation behavior\n");
                printf("\n");
                printf("Common causes:\n");
                printf("  - Osipkov-Merritt anisotropy incompatible with this profile at this beta\n");
                printf("  - Augmented density rho_Q violates monotonicity requirements\n");
                printf("  - Try reducing anisotropy (smaller beta or larger --aniso-factor)\n");
                printf("\n");
                printf("================================================================================\n");
                printf("\n");

                // Prompt user
                int response = prompt_yes_no("Continue with initial condition generation despite negative values?");

                if (!response) {
                    printf("\nAborting initial condition generation.\n");
                    printf("Suggestion: Try reducing anisotropy or using a different profile.\n\n");
                    free(fE_values);
                    free(is_negative_fE);
                    return 1; // Signal abort
                } else {
                    printf("\nProceeding with IC generation. Negative %s regions will be treated as zero.\n\n", label);
                    log_message("WARNING", "User chose to proceed despite negative %s values", label);
                }
            }

            log_message("WARNING", "%s: Detected %d negative %s values (%.2f%%) - %s",
                        profile_name, negative_count, label, negative_fraction * 100.0,
                        verbose ? "user chose to continue" : "silently continuing in diagnostic mode");

            free(fE_values);
            free(is_negative_fE);
            return 0; // Continue (user said yes, or silent diagnostic mode)
        }
    } else {
        int total_checked = end_check - start_check;
        log_message("INFO", "%s: All %d f(%s) values are positive (valid distribution)",
                    profile_name, total_checked, label);
        free(fE_values);
        free(is_negative_fE);
        return 0; // Continue normally
    }
}

/**
 * @brief Parse NSphere filename to extract simulation parameters
 * @details Attempts to extract N, Ntimes, and tfinal from standard NSphere filename format:
 *          prefix_tag_N_Ntimes_tfinal.dat (e.g., beta025_1000_1250000_100.dat)
 * @param filename The filename to parse
 * @param N Pointer to store extracted N value
 * @param Ntimes Pointer to store extracted Ntimes value  
 * @param tfinal Pointer to store extracted tfinal value
 * @return 1 if successful, 0 if unable to parse
 */
int parse_nsphere_filename(const char* filename, int* N, int* Ntimes, double* tfinal) {
    if (!filename || !N || !Ntimes || !tfinal) return 0;
    
    // Make a copy to work with
    char* work_str = strdup(filename);
    if (!work_str) return 0;
    
    // Remove directory path if present
    char* base = strrchr(work_str, '/');
    if (base) {
        base++;
    } else {
        base = work_str;
    }
    
    // Remove .dat extension if present
    char* ext = strstr(base, ".dat");
    if (ext) {
        *ext = '\0';
    }
    
    // Find the last three underscores (working backwards)
    // Format: prefix_tag_N_Ntimes_tfinal
    char* underscore_positions[3] = {NULL, NULL, NULL};
    int underscore_count = 0;
    
    // Scan backwards to find underscores
    for (int i = strlen(base) - 1; i >= 0 && underscore_count < 3; i--) {
        if (base[i] == '_') {
            underscore_positions[2 - underscore_count] = &base[i];
            underscore_count++;
        }
    }
    
    // Need at least 3 underscores for standard format
    if (underscore_count < 3) {
        free(work_str);
        return 0;
    }
    
    // Try to parse the three numeric values
    char* endptr;
    
    // Parse N
    long n_val = strtol(underscore_positions[0] + 1, &endptr, 10);
    if (endptr == underscore_positions[0] + 1 || *endptr != '_') {
        free(work_str);
        return 0;
    }
    
    // Parse Ntimes  
    long ntimes_val = strtol(underscore_positions[1] + 1, &endptr, 10);
    if (endptr == underscore_positions[1] + 1 || *endptr != '_') {
        free(work_str);
        return 0;
    }
    
    // Parse tfinal
    double tfinal_val = strtod(underscore_positions[2] + 1, &endptr);
    if (endptr == underscore_positions[2] + 1) {
        free(work_str);
        return 0;
    }
    
    // Store results
    *N = (int)n_val;
    *Ntimes = (int)ntimes_val;
    *tfinal = tfinal_val;
    
    free(work_str);
    return 1;
}

/**
 * @brief Main entry point for the n-sphere dark matter simulation program.
 * @details Orchestrates the overall simulation workflow:
 *          1. Parses command-line arguments.
 *          2. Sets up global parameters and logging.
 *          3. Initializes random number generators.
 *          4. Generates or loads initial conditions (ICs) for NFW, Cored Plummer-like, or Hernquist profiles.
 *             - Includes theoretical calculations for density, mass, potential, and \f$f(E)\f$ splines.
 *             - Includes a diagnostic loop to test IC generation with varied numerical parameters.
 *             - Performs particle sampling based on the derived distribution function.
 *          5. Optionally performs tidal stripping and re-assigns particle IDs.
 *          6. Converts particle velocities to physical simulation units.
 *          7. Executes the main N-body timestepping loop using a selected integration method.
 *             - Performs gravitational updates.
 *             - Optionally performs SIDM scattering via `handle_sidm_step`.
 *             - Tracks particle trajectories and energies.
 *             - Periodically writes simulation data and progress.
 *          8. If `g_doAllParticleData` is enabled, processes all particle data to generate
 *             snapshot files for Rank/Mass/Radius/Velocity/Potential/Energy/Density.
 *          9. Writes final summary plots and theoretical profiles.
 *          10. Cleans up allocated resources.
 *          Handles restart/resume functionality by checking for existing data products.
 * 
 * @param argc [in] Standard argument count from the command line.
 * @param argv [in] Standard array of argument strings from the command line.
 * @return int Exit code: 0 for successful execution, non-zero for errors.
 *
 * @note This application supports OpenMP for parallelization in various sections.
 * @warning Large particle counts or long simulations can be memory and CPU intensive.
 *          Disk space requirements for full data output can also be significant.
 */
// Forward declarations for Hernquist profile functions
static inline double density_hernquist(double r, double M, double a);
static inline double potential_hernquist(double r, double M, double a);
static inline double df_hernquist_aniso(double E_bind, double L, double M, double a);

// Forward declarations for adaptive spline utilities
double find_minimum_useful_radius(double (*potential_func)(double, void*), void* params,
                                  double scale_radius, double tolerance);
/// Function pointer for potential evaluation: \f$\Psi(r)\f$.
typedef double (*potential_function_t)(double r, void* params);

/**
 * @brief Function pointer for dynamically selected density derivative (Cored/Hernquist profiles).
 * @details Points to the appropriate \f$d\rho/dr\f$ implementation selected at runtime during
 *          IC generation. Enables profile-specific calculations without conditional branching
 *          in performance-critical loops.
 *
 *          Selection logic:
 *          - If OM model active: Points to augmented density derivative (density_derivative_om_cored)
 *          - Otherwise: Points to standard isotropic derivative (drhodr)
 *
 * @note Must be set before any integration begins.
 */
static drhodr_func_t g_density_derivative_func = NULL;

/**
 * @brief Function pointer for dynamically selected NFW density derivative.
 * @details Points to the appropriate NFW-specific \f$d\rho/dr\f$ implementation selected at
 *          runtime during IC generation. NFW derivatives require additional parameters
 *          (rc, nt_nfw, falloff_C) passed through the function signature.
 *
 *          Selection logic:
 *          - If OM model active: Points to augmented NFW derivative (density_derivative_om_nfw)
 *          - Otherwise: Points to standard NFW derivative (drhodr_profile_nfwcutoff)
 *
 * @note Must be set before any integration begins.
 */
static drhodr_nfw_func_t g_density_derivative_nfw_func = NULL;

// Structs for potential parameters
typedef struct {
    double M;    // Total mass
    double a;    // Scale radius
} hernquist_potential_params;

typedef struct {
    double M;           // Total mass  
    double rs;          // Scale radius
    double falloff;     // Falloff factor
    double Phi0;        // Central potential
    double normalization; // Mass normalization
} nfw_potential_params;

typedef struct {
    double M;    // Total mass
    double rc;   // Core radius
} cored_potential_params;

// Forward declarations for potential wrapper functions
double hernquist_potential_wrapper(double r, void* params);
double nfw_potential_wrapper(double r, void* params);
double cored_potential_wrapper(double r, void* params);

// Forward declaration for the anisotropic Hernquist IC generator
void generate_ics_hernquist_anisotropic(double **particles, int npts_initial, gsl_rng *rng, double halo_mass, double scale_radius,
                                        gsl_spline **splinemass_out, gsl_interp_accel **enclosedmass_out,
                                        gsl_spline **splinePsi_out, gsl_interp_accel **Psiinterp_out,
                                        gsl_spline **splinerofPsi_out, gsl_interp_accel **rofPsiinterp_out,
                                        gsl_interp **fofEinterp_out, gsl_interp_accel **fofEacc_out,
                                        double **radius_out, double **radius_unsorted_out, double **mass_out, double **Psivalues_out, int *num_points_out,
                                        int splines_only);

/// Struct to pass parameters to the GSL Hernquist density integrand
typedef struct {
    double M; ///< Total Mass
    double a; ///< Scale Radius
} hernquist_params;

/**
 * @brief GSL-compatible wrapper for the Hernquist density integrand: \f$4\pi r^2 \rho(r)\f$.
 * @param r Radius (kpc).
 * @param p Void pointer to hernquist_params struct.
 * @return Integrand value for mass calculation.
 */
double mass_integrand_hernquist(double r, void *p) {
    hernquist_params *params = (hernquist_params *)p;
    return 4.0 * PI * r * r * density_hernquist(r, params->M, params->a);
}


/**
 * @brief Evaluates a spline, returning boundary values when outside domain.
 * @param spline GSL spline object
 * @param acc GSL interpolation accelerator
 * @param x Point to evaluate
 * @param x_min Minimum x value in spline
 * @param x_max Maximum x value in spline  
 * @param y_min Value to return when x < x_min
 * @param y_max Value to return when x > x_max
 * @return Interpolated value or boundary value
 */
static inline double evaluatespline_with_boundary(gsl_spline *spline, gsl_interp_accel *acc,
                                                  double x, double x_min, double x_max,
                                                  double y_min, double y_max) {
    if (x < x_min) {
        return y_min;
    } else if (x > x_max) {
        return y_max;
    } else {
        return gsl_spline_eval(spline, x, acc);
    }
}

/**
 * @brief Generates initial conditions for an anisotropic Hernquist profile.
 * @details Implements a sampling algorithm for the anisotropic Hernquist distribution
 *          function \f$f(E, L)\f$. It uses a combination of inverse transform sampling
 *          for radius and direction, and rejection sampling for velocity magnitude.
 */
void generate_ics_hernquist_anisotropic(double **particles, int npts_initial, gsl_rng *rng, double halo_mass, double scale_radius,
                                        gsl_spline **splinemass_out, gsl_interp_accel **enclosedmass_out,
                                        gsl_spline **splinePsi_out, gsl_interp_accel **Psiinterp_out,
                                        gsl_spline **splinerofPsi_out, gsl_interp_accel **rofPsiinterp_out,
                                        gsl_interp **fofEinterp_out, gsl_interp_accel **fofEacc_out,
                                        double **radius_out, double **radius_unsorted_out, double **mass_out, double **Psivalues_out, int *num_points_out,
                                        int splines_only)
{
    // === Phase 1: Setup and Spline Generation ===
    log_message("INFO", "Hernquist IC Gen: Calculating theoretical profiles...");

    int num_points = 100000;
    double rmax = g_cutoff_factor_param * scale_radius;

    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);

    double *mass = (double *)malloc(num_points * sizeof(double));
    double *radius = (double *)malloc(num_points * sizeof(double));
    if (!mass || !radius) {
        fprintf(stderr, "HERNQUIST_PATH: Failed to allocate mass/radius arrays\n");
        CLEAN_EXIT(1);
    }

    hernquist_params params = { .M = halo_mass, .a = scale_radius };
    gsl_function F_mass_integrand;
    F_mass_integrand.function = &mass_integrand_hernquist;
    F_mass_integrand.params = &params;

    // Find minimum useful radius adaptively
    hernquist_potential_params hern_params = {halo_mass, scale_radius};
    double rmin = find_minimum_useful_radius(hernquist_potential_wrapper, &hern_params, 
                                           scale_radius, 1e-12);
    
    log_message("INFO", "Hernquist: Using adaptive rmin = %.3e kpc (%.3e × scale radius)", 
                rmin, rmin/scale_radius);
    
    // Use logarithmic grid starting from adaptive minimum radius
    double log_rmin = log10(rmin);
    double log_rmax = log10(rmax);
    
    for (int i = 0; i < num_points; i++) {
        double r;
        if (i == num_points - 1) {
            r = rmax;  // Ensure exact endpoint
        } else {
            // Logarithmic spacing from rmin to rmax
            double log_r = log_rmin + (log_rmax - log_rmin) * (double)i / (double)(num_points - 1);
            r = pow(10.0, log_r);
        }
        radius[i] = r;
        double integral_result, error;
        gsl_integration_qag(&F_mass_integrand, 0.0, r, 1e-9, 1e-9, 1000, GSL_INTEG_GAUSS51, w, &integral_result, &error);
        mass[i] = integral_result;
    }

    // Allocate and initialize master splines for main()
    *enclosedmass_out = gsl_interp_accel_alloc();
    *splinemass_out = gsl_spline_alloc(gsl_interp_cspline, num_points);
    gsl_spline_init(*splinemass_out, radius, mass, num_points);

    gsl_interp_accel *rofMaccel = gsl_interp_accel_alloc();
    gsl_spline *splinerofM = gsl_spline_alloc(gsl_interp_cspline, num_points);
    for (int i = 1; i < num_points; i++) {
        if (mass[i] <= mass[i-1]) mass[i] = mass[i-1] * (1.0 + 1e-12);
    }
    gsl_spline_init(splinerofM, mass, radius, num_points);
    

    double *Psivalues = (double *)malloc(num_points * sizeof(double));
    for (int i = 0; i < num_points; i++) {
        Psivalues[i] = potential_hernquist(radius[i], halo_mass, scale_radius);
    }
    *Psiinterp_out = gsl_interp_accel_alloc();
    *splinePsi_out = gsl_spline_alloc(gsl_interp_cspline, num_points);
    gsl_spline_init(*splinePsi_out, radius, Psivalues, num_points);
    
    // Create r(Psi) and a dummy f(E) spline to prevent crashes in downstream code
    double *nPsivalues = (double *)malloc(num_points * sizeof(double));
    for(int i = 0; i < num_points; i++) nPsivalues[i] = -Psivalues[i];

    // Save original unsorted radius array BEFORE sorting (for file output)
    double *radius_unsorted = (double *)malloc(num_points * sizeof(double));
    for(int i = 0; i < num_points; i++) radius_unsorted[i] = radius[i];

    // Sort first, then check for duplicates
    sort_rr_psi_arrays(nPsivalues, radius, num_points); // Sorts the pair for r(Psi)
    
    // Now check for duplicate values after sorting and remove them
    int unique_count = 1;  // Start with first element
    for (int i = 1; i < num_points; i++) {
        // Keep only values that are strictly greater than the previous
        if (nPsivalues[i] > nPsivalues[unique_count-1] + 1e-12 * fabs(nPsivalues[unique_count-1])) {
            if (i != unique_count) {
                nPsivalues[unique_count] = nPsivalues[i];
                radius[unique_count] = radius[i];
            }
            unique_count++;
        }
    }
    
    if (unique_count < num_points) {
        log_message("WARNING", "Hernquist: Removed %d duplicate nPsi values from r(Psi) spline (keeping %d unique values)", 
                    num_points - unique_count, unique_count);
    }
    
    if (unique_count < 3) {
        fprintf(stderr, "HERNQUIST_PATH: Too few unique Psi values (%d) for spline creation\n", unique_count);
        CLEAN_EXIT(1);
    }
    
    *rofPsiinterp_out = gsl_interp_accel_alloc();
    *splinerofPsi_out = gsl_spline_alloc(gsl_interp_cspline, unique_count);
    gsl_spline_init(*splinerofPsi_out, nPsivalues, radius, unique_count);
    
    // Create an approximate f(E) interpolator for diagnostics.
    // For general beta, f(E,L) is proportional to E^(5/2-beta) / L^(2*beta)
    // The marginalized f(E) requires integrating over L at fixed E
    int n_E_points = 1000; // Use a reasonable number of points for the spline
    double *E_array = (double *)malloc(n_E_points * sizeof(double));
    double *f_array = (double *)malloc(n_E_points * sizeof(double));
    if (!E_array || !f_array) {
        fprintf(stderr, "HERNQUIST_PATH: Failed to allocate arrays for f(E) spline\n");
        CLEAN_EXIT(1);
    }
    
    // Psivalues are DESCENDING; ASCENDING energy array required for GSL
    double E_min = Psivalues[num_points-1]; // Least bound (at r_max)
    double E_max = Psivalues[0];            // Most bound (at r=0)
    double GM_over_a = G_CONST * halo_mass / scale_radius;

    struct RrPsiPair *ef_pairs = (struct RrPsiPair *)malloc(n_E_points * sizeof(struct RrPsiPair));
    if (!ef_pairs) {
        fprintf(stderr, "HERNQUIST_PATH: Failed to allocate ef_pairs for sorting\n");
        CLEAN_EXIT(1);
    }

    for (int i = 0; i < n_E_points; i++) {
        // Build the energy array in DESCENDING order first, matching Psivalues
        double E_val = E_max - (E_max - E_min) * (double)i / (n_E_points - 1.0);
        double E_bind = E_val; // Using positive Psi convention: E_bind = Psi
        
        ef_pairs[i].rr = E_val; // Use the 'rr' field for Energy
        
        // Calculate marginalized f(E) using proper hypergeometric function
        double beta = g_anisotropy_beta;
        if (E_bind > 0) {
            // Convert to dimensionless energy for hypergeometric function
            double E_tilde = E_bind / GM_over_a;
            
            if (fabs(E_tilde) < 1.0) {  // Hypergeometric convergence requirement
                double power_term = pow(E_bind, 2.5 - beta);
                
                // Calculate hypergeometric function using safe wrapper
                double hyperg = hyperg_2F1_safe(beta, E_tilde);

                if (hyperg != 0.0) {
                    // f(E) proportional to E^(5/2-beta) * 2F1(...)
                    ef_pairs[i].psi = power_term * hyperg;
                } else {
                    ef_pairs[i].psi = 0.0;
                }
            } else {
                ef_pairs[i].psi = 0.0;  // Outside convergence region
            }
        } else {
            ef_pairs[i].psi = 0.0;
        }
    }

    // Sort the pairs by Energy (ef_pairs.rr) in ASCENDING order
    qsort(ef_pairs, n_E_points, sizeof(struct RrPsiPair), compare_pair_by_first_element);

    // Populate the final arrays from the sorted pairs
    for (int i = 0; i < n_E_points; i++) {
        E_array[i] = ef_pairs[i].rr;
        f_array[i] = ef_pairs[i].psi;
    }

    // Now initialize the GSL spline with the correctly sorted arrays
    *fofEacc_out = gsl_interp_accel_alloc();
    *fofEinterp_out = gsl_interp_alloc(gsl_interp_cspline, n_E_points);
    gsl_interp_init(*fofEinterp_out, E_array, f_array, n_E_points);
    
    // Cleanup
    free(E_array);
    free(f_array);
    free(ef_pairs);

    // Return arrays to main for theoretical profile output
    *radius_out = radius;
    *radius_unsorted_out = radius_unsorted;
    *mass_out = mass;
    *Psivalues_out = Psivalues;
    *num_points_out = num_points;

    if (splines_only) {
        // Return after building splines without generating particles
        return;
    }
    
    // === Phase 2: Particle Sampling ===
    log_message("INFO", "Hernquist IC Gen: Generating %d initial particle positions and velocities...", npts_initial);
    int num_envelope_points = 2000;  // Even more points for envelope
    double *r_envelope = (double *)malloc(num_envelope_points * sizeof(double));
    double *Pmax_envelope = (double *)malloc(num_envelope_points * sizeof(double));
    
    // Use logarithmic spacing for better resolution at small radii
    double log_rmin_env = log10(1e-4 * scale_radius);
    double log_rmax_env = log10(rmax);
    
    for (int i = 0; i < num_envelope_points; i++) {
        double log_r = log_rmin_env + (log_rmax_env - log_rmin_env) * (double)i / (num_envelope_points - 1.0);
        double r = pow(10.0, log_r);
        r_envelope[i] = r;
        double psi_val = potential_hernquist(r, halo_mass, scale_radius);
        // Escape velocity for positive potential Psi: vesc = sqrt(2*Psi)
        double vesc = sqrt(2.0 * psi_val);
        double p_max = 0.0;
        int v_steps = 1000;  // Fine velocity grid
        double GM_over_a = G_CONST * halo_mass / scale_radius;
        for (int j = 0; j <= v_steps; j++) {
            double v = (double)j * vesc / v_steps;
            double E_bind = psi_val - 0.5 * v * v;
            if (E_bind <= 0) continue;
            
            // Convert to dimensionless energy for hypergeometric function
            double E_tilde = E_bind / GM_over_a;
            
            // For isotropic (beta=0): f(E) ∝ E^(5/2) * 2F1(...)
            // The velocity distribution P(v) ∝ v² * f(E) for isotropic
            // Anisotropic case uses approximation for velocity distribution
            double p_current = 0.0;
            
            if (fabs(E_tilde) < 1.0) {  // Hypergeometric convergence requirement
                double beta = g_anisotropy_beta;
                double power_term = pow(E_bind, 2.5 - beta);
                
                // Calculate hypergeometric function using safe wrapper
                double hyperg = hyperg_2F1_safe(beta, E_tilde);

                if (hyperg != 0.0) {
                    // P(v) ∝ v^(2-2β) × E^(5/2-β) × hypergeometric
                    // This comes from integrating f(E,L) ∝ E^(5/2-β)/L^(2β) over angles
                    // Since L = r·v·√(1-μ²), the L^(-2β) gives v^(-2β) after integration
                    p_current = pow(v, 2.0 - 2.0*beta) * power_term * hyperg;
                }
            }
            
            if (p_current > p_max) p_max = p_current;
        }
        Pmax_envelope[i] = p_max * 1.05;  // 5% safety factor
        if (p_max <= 0.0) {
            fprintf(stderr, "ERROR: Envelope p_max=%.3e at r=%.3e is non-positive!\n", p_max, r);
            Pmax_envelope[i] = 1e-30;  // Set a minimum positive value
        }
    }
    gsl_interp_accel *Pmax_accel = gsl_interp_accel_alloc();
    // Use linear interpolation to avoid negative oscillations
    gsl_spline *spline_Pmax = gsl_spline_alloc(gsl_interp_linear, num_envelope_points);
    gsl_spline_init(spline_Pmax, r_envelope, Pmax_envelope, num_envelope_points);
    
    // Debug: Check particle radius distribution
    double r_min_particle = 1e100;
    double r_max_particle = 0.0;
    
    for (int i = 0; i < npts_initial; i++) {
        double mass_frac_sample = gsl_rng_uniform(rng);
        double mass_sample = mass_frac_sample * mass[num_points-1];
        double r = evaluatespline(splinerofM, rofMaccel, mass_sample);
        particles[0][i] = r;
        
        // Track min/max radii
        if (r < r_min_particle) r_min_particle = r;
        if (r > r_max_particle) r_max_particle = r;
        double psi_at_r = evaluatespline(*splinePsi_out, *Psiinterp_out, r);
        double vesc_at_r = sqrt(fmax(0.0, 2.0 * psi_at_r));
        double Pmax_at_r = evaluatespline(spline_Pmax, Pmax_accel, r);
        double v_accepted = -1.0;
        double GM_over_a = G_CONST * halo_mass / scale_radius;
        double max_P_seen = 0.0;  // Move this outside the if block
        
        
        if (Pmax_at_r > 1e-30) {
            int trials = 0;
            while (v_accepted < 0.0 && trials < 10000) {
                double v_cand = gsl_rng_uniform(rng) * vesc_at_r;
                // Binding energy formula: E_bind = Psi - KE
                double E_bind_cand = psi_at_r - 0.5 * v_cand * v_cand;
                if (E_bind_cand <= 0) continue;
                
                // Convert to dimensionless energy
                double E_tilde = E_bind_cand / GM_over_a;
                
                double P_target = 0.0;
                if (fabs(E_tilde) < 1.0) {  // Hypergeometric convergence
                    double beta = g_anisotropy_beta;
                    double power_term = pow(E_bind_cand, 2.5 - beta);
                    
                        // Calculate hypergeometric function using safe wrapper
                    double hyperg = hyperg_2F1_safe(beta, E_tilde);

                    if (hyperg != 0.0) {
                        // P(v) ∝ v^(2-2β) × E^(5/2-β) × hypergeometric
                        // This comes from integrating f(E,L) ∝ E^(5/2-β)/L^(2β) over angles
                        // The v^(2-2β) factor arises from L = r·v·√(1-μ²) giving L^(-2β) ∝ v^(-2β)
                        P_target = pow(v_cand, 2.0 - 2.0*beta) * power_term * hyperg;
                    }
                }
                
                if (P_target > max_P_seen) max_P_seen = P_target;
                
                double u_rej = gsl_rng_uniform(rng) * Pmax_at_r;
                if (u_rej < P_target) v_accepted = v_cand;
                trials++;
            }
        }
        if (v_accepted < 0.0) {
            log_message("WARNING", "Hernquist IC: Failed to sample velocity at r=%.3e, psi=%.3e, vesc=%.3e, Pmax=%.3e", 
                        r, psi_at_r, vesc_at_r, Pmax_at_r);
            v_accepted = 0.0;
        }
        // Sample mu = v_r/v from the standard constant-beta distribution:
        // p(mu) propto (1 - mu^2)^(-beta).
        // This is achieved by sampling mu^2 from a Beta distribution.
        double mu;
        if (fabs(g_anisotropy_beta) < 1e-10) {
            // Isotropic case (beta=0): mu is uniform in [-1, 1].
            mu = 2.0 * gsl_rng_uniform(rng) - 1.0;
        } else {
            // Anisotropic case (beta != 0):
            // 1. Sample z = mu^2 from Beta(a, b) distribution.
            //    For p(mu) propto (1-mu^2)^(-beta), the parameters are a=1/2, b=1-beta.
            //    Constraint: beta < 1. The command-line parser already constrains beta <= 0.5,
            //    so b = 1-beta is always >= 0.5, which is valid for GSL.
            double z = gsl_ran_beta(rng, 0.5, 1.0 - g_anisotropy_beta);
            mu = sqrt(z);

            // 2. Randomly assign sign.
            if (gsl_rng_uniform(rng) < 0.5) {
                mu = -mu;
            }
        }
        // Store velocity MAGNITUDE and MU separately to maintain a consistent
        // data format with the isotropic IC generators. The downstream processing
        // block will correctly calculate v_r = v_mag * mu.
        particles[1][i] = v_accepted; // Store velocity MAGNITUDE v
        particles[2][i] = r * v_accepted * sqrt(fmax(0.0, 1.0 - mu * mu)); // Angular momentum L
        particles[3][i] = (double)i; // Original ID
        particles[4][i] = mu; // Store mu
    }
    
    // Debug: Check for particles with extreme velocities or positions
    double v_max = 0.0;
    double r_min_check = 1e100;
    double r_max_check = 0.0;
    int zero_v_count = 0;
    for (int i = 0; i < npts_initial; i++) {
        double r = particles[0][i];
        double v = particles[1][i];
        if (r < r_min_check) r_min_check = r;
        if (r > r_max_check) r_max_check = r;
        if (v > v_max) v_max = v;
        if (v < 1e-10) zero_v_count++;
    }
    
    log_message("INFO", "Hernquist IC Gen: Successfully generated %d particles.", npts_initial);
    log_message("INFO", "Hernquist IC Gen: r range [%.3e, %.3e] kpc, v_max = %.3f kpc/Myr, zero_v = %d",
                r_min_check, r_max_check, v_max, zero_v_count);
    free(r_envelope);
    free(Pmax_envelope);
    gsl_spline_free(spline_Pmax);
    gsl_interp_accel_free(Pmax_accel);

    // === Phase 3: Cleanup ===
    gsl_integration_workspace_free(w);
    gsl_spline_free(splinerofM);
    gsl_interp_accel_free(rofMaccel);
    free(nPsivalues);
}

/**
 * @brief Finds the minimum useful radius where potential values are numerically distinguishable.
 * @details Starting from a safe radius, steps down logarithmically towards \f$r=0\f$ until
 *          consecutive potential values differ by less than the tolerance.
 * @param potential_func Function pointer to evaluate potential at radius \f$r\f$
 * @param params Parameters to pass to potential_func
 * @param scale_radius Characteristic scale of the profile
 * @param tolerance Relative tolerance for considering potentials equal
 * @return Minimum radius where potential is numerically distinguishable from smaller radii
 */
double find_minimum_useful_radius(double (*potential_func)(double, void*), void* params, 
                                  double scale_radius, double tolerance) {
    // Start from a safe radius
    double r_start = 0.1 * scale_radius;
    double r_min_limit = 1e-12 * scale_radius;  // Don't go smaller than this
    int n_steps = 1000;  // Number of logarithmic steps
    
    // Generate logarithmic grid from r_start down to r_min_limit
    double log_r_start = log10(r_start);
    double log_r_min = log10(r_min_limit);
    
    double r_prev = r_start;
    double psi_prev = potential_func(r_prev, params);
    
    for (int i = 1; i <= n_steps; i++) {
        double log_r = log_r_start + (log_r_min - log_r_start) * (double)i / (double)n_steps;
        double r_curr = pow(10.0, log_r);
        double psi_curr = potential_func(r_curr, params);
        
        // Check if potentials are effectively identical
        double diff = fabs(psi_curr - psi_prev);
        double scale = fmax(fabs(psi_curr), fabs(psi_prev));
        
        if (scale > 0 && diff / scale < tolerance) {
            // Found the radius where potential stops changing significantly
            return r_prev;  // Return the last distinguishable radius
        }
        
        r_prev = r_curr;
        psi_prev = psi_curr;
    }
    
    // All values distinguishable (minimum radius reached)
    return r_min_limit;
}

/**
 * @brief Wrapper function for NFW potential calculation.
 * @param r Radius (kpc).
 * @param params Void pointer to nfw_potential_params struct.
 * @return Approximate potential value at radius \a r.
 */
double nfw_potential_wrapper(double r, void* params) {
    nfw_potential_params* p = (nfw_potential_params*)params;

    // Approximate potential without integral for rmin determination
    double rho0 = p->M / (4.0 * PI * p->rs * p->rs * p->rs * p->falloff * p->falloff);

    // If r is 0 or very small, return the constant potential at center
    if (r < 1e-12 * p->rs) {
        // At r=0, NFW potential is -4πGρ₀rs²
        return -4.0 * PI * G_CONST * rho0 * p->rs * p->rs;
    }

    double x = r / p->rs;
    return -4.0 * PI * G_CONST * rho0 * p->rs * p->rs * log(1.0 + x) / x;
}

/**
 * @brief Wrapper function for Cored Plummer-like potential calculation.
 * @param r Radius (kpc).
 * @param params Void pointer to cored_potential_params struct.
 * @return Potential value at radius \a r.
 */
double cored_potential_wrapper(double r, void* params) {
    cored_potential_params* p = (cored_potential_params*)params;

    // At r=0, the cored potential is -GM/rc
    if (r < 1e-12 * p->rc) {
        return -G_CONST * p->M / p->rc;
    }

    // Simplified Plummer-like potential
    double r_eff = sqrt(r * r + p->rc * p->rc);
    return -G_CONST * p->M / r_eff;
}

/**
 * @brief Wrapper function for Hernquist potential calculation.
 * @param r Radius (kpc).
 * @param params Void pointer to hernquist_potential_params struct.
 * @return Potential value at radius \a r.
 */
double hernquist_potential_wrapper(double r, void* params) {
    hernquist_potential_params* p = (hernquist_potential_params*)params;

    // At r=0, Hernquist potential is GM/a (positive convention)
    if (r < 1e-12 * p->a) {
        return G_CONST * p->M / p->a;
    }

    return potential_hernquist(r, p->M, p->a);
}

// Forward declaration for sort test
static void parallel_radix_sort(double **columns, int n);

int main(int argc, char *argv[])
{
// Print the tool header first, regardless of OpenMP status
printf("\n===================================================================================================\n");
printf("NSphere Simulation Tool\n");
printf("===================================================================================================\n");
printf("  \n");

#ifdef _OPENMP
    /** @note OpenMP section: Configures parallel execution environment when compiled with OpenMP. */
    // OpenMP is available - configure parallel execution environment
    int max_threads = omp_get_max_threads();
    int num_processors = omp_get_num_procs();

    // Enable nested parallelism with max_active_levels
    omp_set_max_active_levels(10); // Allow up to 10 levels of nested parallelism

    // Set OpenMP to use maximum available thread parallelism
    // Store the default max_threads for non-SIDM operations
    g_default_max_threads = max_threads;

    // Detect hybrid CPU architectures (P-cores + E-cores) and optimize for SIDM
    int p_cores = 0;
    int detected_hybrid = 0;

    #ifdef __APPLE__
        // Check if this is Apple Silicon (ARM64) or Intel Mac
        FILE *fp_apple = popen("sysctl -n hw.optional.arm64 2>/dev/null", "r");
        if (fp_apple != NULL) {
            char result[10];
            if (fgets(result, sizeof(result), fp_apple) != NULL && atoi(result) == 1) {
                // Apple Silicon detected - has hybrid P+E cores
                FILE *fp_pcore = popen("sysctl -n hw.perflevel0.physicalcpu 2>/dev/null", "r");
                if (fp_pcore != NULL) {
                    char pcore_str[10];
                    if (fgets(pcore_str, sizeof(pcore_str), fp_pcore) != NULL) {
                        int detected_p_cores = atoi(pcore_str);
                        if (detected_p_cores > 0) {
                            p_cores = detected_p_cores;
                            detected_hybrid = 1;
                        }
                    }
                    pclose(fp_pcore);
                }
            }
            // Intel Macs: hw.optional.arm64 returns 0 or does not exist
            // These use all cores (no hybrid architecture)
            pclose(fp_apple);
        }
    #elif defined(__linux__)
        // Intel/AMD hybrid detection on Linux using lscpu
        FILE *fp_cpu = popen("lscpu | grep 'Core(s) per cluster' 2>/dev/null", "r");
        if (fp_cpu != NULL) {
            char line[256];
            if (fgets(line, sizeof(line), fp_cpu) != NULL) {
                // Intel hybrid detected - try to get P-core count
                FILE *fp_pcore = popen("lscpu --all --extended | grep 'Core' | awk '{print $4}' | grep -c '^P' 2>/dev/null", "r");
                if (fp_pcore != NULL) {
                    char pcore_str[10];
                    if (fgets(pcore_str, sizeof(pcore_str), fp_pcore) != NULL) {
                        int detected_p_cores = atoi(pcore_str);
                        if (detected_p_cores > 0) {
                            p_cores = detected_p_cores;
                            detected_hybrid = 1;
                        }
                    }
                    pclose(fp_pcore);
                }
            }
            pclose(fp_cpu);
        }
    #elif defined(_WIN32) || defined(_WIN64)
        // Windows hybrid detection using wmic or PowerShell
        FILE *fp_cpu = popen("wmic cpu get NumberOfPerformanceCores /value 2>NUL", "r");
        if (fp_cpu != NULL) {
            char line[256];
            while (fgets(line, sizeof(line), fp_cpu) != NULL) {
                if (strstr(line, "NumberOfPerformanceCores=") != NULL) {
                    char *eq_pos = strchr(line, '=');
                    if (eq_pos != NULL) {
                        int detected_p_cores = atoi(eq_pos + 1);
                        if (detected_p_cores > 0) {
                            p_cores = detected_p_cores;
                            detected_hybrid = 1;
                        }
                    }
                }
            }
            pclose(fp_cpu);
        }
    #endif

    // Apply P-core optimization if hybrid architecture detected
    if (detected_hybrid && p_cores > 0) {
        g_hybrid_p_cores = p_cores;

        if (g_sidm_execution_mode == 1) {
            max_threads = p_cores;
            printf("Hybrid CPU detected: Using %d P-cores for SIDM, %d total cores available\n",
                   p_cores, g_default_max_threads);
        }
    }

    omp_set_num_threads(max_threads);

    printf("OpenMP Status: ENABLED (%d logical processors, using %d threads)\n", 
           num_processors, max_threads);
    printf("\n\n");

    log_message("INFO", "Using maximum thread parallelism: %d threads", max_threads);
#else
    /** @warning OpenMP section: Warns user when compiled without OpenMP support. */
    // OpenMP is not available - warn user about performance implications
    printf("WARNING: OpenMP is NOT ENABLED in this build!\n");
    printf("This will result in significantly reduced performance.\n");
    printf("For better performance, please install OpenMP and recompile with -fopenmp flag.\n\n\n");

    log_message("WARNING", "OpenMP not available - running in single-threaded mode");
    
    // Check if --help flag is used (skip delay for help display)
    int help_requested = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help_requested = 1;
            break;
        }
    }
    
    // Add a delay to ensure the warning is noticed, but only when not showing help
    if (!help_requested) {
        printf("Continuing in single-threaded mode");
        fflush(stdout);
        for (int i = 0; i < 5; i++) {
            usleep(500000); // 500ms * 5 = 2.5 seconds
            printf(".");
            fflush(stdout);
        }
        printf("\n\n");
    }

    // Define single-thread variable for non-OpenMP compilation mode
    int max_threads __attribute__((unused)) = 1;
#endif

#ifdef _OPENMP
    /** @note Initializes FFTW thread support if compiled with OpenMP. */
    // Initialize FFTW thread support (only effective if FFTW was compiled with threading)
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);
#endif

    printf("\n");

    int npts = 100000;
    int Ntimes = 10000;
    int tfinal_factor = 5;
    int nout = 100;
    int dtwrite = 100;
    int snapshot_block_size = 100;  // Default value, can be overridden with --snapshot-buffer
    double tidal_fraction = 0.0;
    int noutsnaps;
    g_total_sidm_scatters = 0; // Initialize global SIDM scatter counter
    int *inverse_map = NULL;  // Declare inverse_map early for use in profile blocks
    char suffixed_filename[256]; // Buffer for suffixed filenames

    int method_select = 1;            // Default: option 1 (Adaptive Leapfrog with Adaptive Levi-Civita)
    int display_sort = 1;             // Default: option 1 (Parallel Quadsort)
    int include_method_in_suffix = 0; // Default: exclude method from filenames.
    char custom_tag[256] = {0};       // Default: no custom tag.

    /** @note Check for the `--help` argument first before parsing other options. */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    /** @note Handle the case where no command-line arguments are provided. */
    if (argc == 1)
    {
        printf("No command-line arguments given. Using default parameters.\n");
        printf("Run `%s --help` to learn how to adjust parameters.\n\n", argv[0]);
    }

    /** @note Main command-line argument parsing loop (strict parsing). */
    for (int i = 1; i < argc; i++)
    {
        /** @note Forbid using '=' within options; require space separation. */
        // Check for "--option=value" format, which is disallowed.
        if (strncmp(argv[i], "--", 2) == 0 && strstr(argv[i], "=") != NULL)
        {
            errorAndExit("use space, not '=' after option", argv[i], argv[0]);
        }

        if (strcmp(argv[i], "--nparticles") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--nparticles requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --nparticles", argv[i + 1], argv[0]);
            }
            npts = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--ntimesteps") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--ntimesteps requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --ntimesteps", argv[i + 1], argv[0]);
            }
            Ntimes = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tfinal") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--tfinal requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --tfinal", argv[i + 1], argv[0]);
            }
            tfinal_factor = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--nout") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--nout requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --nout", argv[i + 1], argv[0]);
            }
            nout = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--dtwrite") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--dtwrite requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --dtwrite", argv[i + 1], argv[0]);
            }
            dtwrite = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--snapshot-buffer") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--snapshot-buffer requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --snapshot-buffer", argv[i + 1], argv[0]);
            }
            snapshot_block_size = atoi(argv[++i]);
            if (snapshot_block_size < 1)
            {
                errorAndExit("--snapshot-buffer must be at least 1", argv[i], argv[0]);
            }
        }
        else if (strcmp(argv[i], "--tag") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--tag requires a string argument", NULL, argv[0]);
            }

            strncpy(custom_tag, argv[++i], 255);
            custom_tag[255] = '\0'; // Ensure null-termination.
        }
        else if (strcmp(argv[i], "--method") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--method requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --method", argv[i + 1], argv[0]);
            }
            method_select = atoi(argv[++i]);

            if (method_select < 1 || method_select > 9)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "method must be in [1..9]");
                errorAndExit(buf, NULL, argv[0]);
            }
        }
        else if (strcmp(argv[i], "--sort") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--sort requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --sort", argv[i + 1], argv[0]);
            }
            int sort_val = atoi(argv[++i]);

            if (sort_val < 1 || sort_val > 6)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "sort must be in [1..6]");
                errorAndExit(buf, NULL, argv[0]);
            }

            display_sort = sort_val;

            switch (sort_val)
            {
            case 1:
                // Parallel Quadsort
                g_defaultSortAlg = "quadsort_parallel";
                break;
            case 2:
                // Sequential Quadsort
                g_defaultSortAlg = "quadsort";
                break;
            case 3:
                // Parallel Insertion Sort
                g_defaultSortAlg = "insertion_parallel";
                break;
            case 4:
                // Sequential Insertion Sort
                g_defaultSortAlg = "insertion";
                break;
            case 5:
                // Parallel radix sort - high performance for large arrays
                g_defaultSortAlg = "parallel_radix";
                printf("INFO: Parallel radix sort selected\n");
                break;
            case 6:
                // Adaptive sorting with periodic benchmarking
                g_defaultSortAlg = "benchmark_mode";
                printf("INFO: Adaptive sort mode enabled (starts with insertion, benchmarks every 1000 sorts)\n");
                printf("      Will compare insertion, quadsort, and radix sorts\n");
                printf("      Algorithm will switch to the fastest after each benchmark\n");
                break;
            }
        }
        else if (strcmp(argv[i], "--readinit") == 0)
        {
            if (i + 1 >= argc) // Check if filename argument exists
            {
                errorAndExit("--readinit requires a file argument", NULL, argv[0]);
            }

            /** @warning Check for incompatibility with `--restart` and `--writeinit`. */
            if (g_doRestart)
            {
                errorAndExit("--readinit is incompatible with --restart. These options cannot be used together. Use either --restart OR --readinit, not both.", NULL, argv[0]);
            }
            if (doWriteInit)
            {
                errorAndExit("--readinit is incompatible with --writeinit. These options cannot be used together. Use either --readinit OR --writeinit, not both.", NULL, argv[0]);
            }

            const char* user_filename = argv[++i]; // consume next arg
            // Prefix the path with "init/" directory
            static char prefixed_read_path[512]; // Static buffer for the path
            snprintf(prefixed_read_path, sizeof(prefixed_read_path), "init/%s", user_filename);
            readInitFilename = prefixed_read_path; // Assign the prefixed path
            doReadInit = 1;
        }
        else if (strcmp(argv[i], "--writeinit") == 0)
        {
            if (i + 1 >= argc) // Check if filename argument exists
            {
                errorAndExit("--writeinit requires a file argument", NULL, argv[0]);
            }

            /** @warning Check for incompatibility with `--restart` and `--readinit`. */
            if (g_doRestart)
            {
                errorAndExit("--writeinit is incompatible with --restart. These options cannot be used together. Use either --restart OR --writeinit, not both.", NULL, argv[0]);
            }
            if (doReadInit)
            {
                errorAndExit("--writeinit is incompatible with --readinit. These options cannot be used together. Use either --writeinit OR --readinit, not both.", NULL, argv[0]);
            }

            const char* user_filename = argv[++i]; // consume next arg
            // Prefix the path with "init/" directory
            static char prefixed_write_path[512]; // Static buffer for the path
            snprintf(prefixed_write_path, sizeof(prefixed_write_path), "init/%s", user_filename);
            writeInitFilename = prefixed_write_path; // Assign the prefixed path
            doWriteInit = 1;
        }
        else if (strcmp(argv[i], "--restart") == 0)
        {
            /** @warning Check for incompatibility with `--readinit` and `--writeinit`. */
            if (doReadInit)
            {
                errorAndExit("--restart is incompatible with --readinit. These options cannot be used together. Use either --restart OR --readinit, not both.", NULL, argv[0]);
            }
            if (doWriteInit)
            {
                errorAndExit("--restart is incompatible with --writeinit. These options cannot be used together. Use either --restart OR --writeinit, not both.", NULL, argv[0]);
            }

            g_doRestart = 1;
            
            // Check for optional "force" argument
            if (i + 1 < argc && strcmp(argv[i + 1], "force") == 0) {
                g_doRestartForce = 1;
                i++; // Consume the "force" argument
                printf("Restart mode enabled with FORCE option. Will regenerate ALL snapshots from existing data.\n\n");
            } else {
                printf("Restart mode enabled. Will look for existing data products to resume processing.\n\n");
            }
        }
        else if (strcmp(argv[i], "--sim-restart") == 0)
        {
            /** @brief Enable simulation restart detection/restart mode.
             *  @details This flag has two modes:
             *           - Default: Actually restart the simulation from last complete checkpoint
             *           - With "check": Only check completion status without restarting
             */
            g_doSimRestart = 1;
            
            // Check for optional "check" argument
            if (i + 1 < argc && strcmp(argv[i + 1], "check") == 0) {
                g_doSimRestartCheckOnly = 1;
                i++; // Consume the "check" argument
                printf("Simulation restart CHECK mode enabled (status check only).\n");
            } else {
                printf("Simulation restart mode enabled (will restart if incomplete).\n");
            }
        }
        else if (strcmp(argv[i], "--restart-file") == 0)
        {
            /** @brief Specify explicit restart file path.
             *  @details Allows overriding the automatically generated filename
             *           for restart operations. Useful for debugging and recovery.
             */
            if (i + 1 >= argc) {
                errorAndExit("--restart-file requires a filename argument", NULL, argv[0]);
            }
            g_restart_file_override = argv[++i];
            printf("Using explicit restart file: %s\n", g_restart_file_override);
        }
        else if (strcmp(argv[i], "--sim-extend") == 0)
        {
            /** @brief Enable simulation extension mode.
             *  @details Extends a completed simulation to new Ntimes/tfinal values.
             *           Requires `--extend-file` to specify the source file.
             *           Copies source file to filename matching target parameters.
             */
            g_doSimExtend = 1;
            printf("Simulation extension mode enabled.\n");
        }
        else if (strcmp(argv[i], "--extend-file") == 0)
        {
            /** @brief Specify source file for simulation extension.
             *  @details Required when using `--sim-extend`.
             *           File copied to filename matching target parameters.
             */
            if (i + 1 >= argc) {
                errorAndExit("--extend-file requires a filename argument", NULL, argv[0]);
            }
            g_extend_file_source = argv[++i];
            printf("Extension source file: %s\n", g_extend_file_source);
        }
        else if (strcmp(argv[i], "--save") == 0)
        {
            /** @note Delegate parsing of subsequent arguments to parseSaveArgs. */
            parseSaveArgs(argc, argv, &i);
            // The main loop continues from the updated index 'i'.
        }
        else if (strcmp(argv[i], "--ftidal") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--ftidal requires a float argument", NULL, argv[0]);
            }
            if (!isFloat(argv[i + 1]))
            {
                errorAndExit("invalid float for --ftidal", argv[i + 1], argv[0]);
            }
            tidal_fraction = atof(argv[++i]);

            /** @warning Check range [0.0, 1.0]. */
            if (tidal_fraction < 0.0 || tidal_fraction > 1.0)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "tidal_fraction must be in [0.0..1.0]");
                errorAndExit(buf, NULL, argv[0]);
            }
        }
        else if (strcmp(argv[i], "--methodtag") == 0)
        {
            /** @note Flag to include integration method string in output filename suffix. */
            include_method_in_suffix = 1;
        }
        else if (strcmp(argv[i], "--log") == 0)
        {
            /** @note Flag to enable logging to log/nsphere.log. */
            g_enable_logging = 1;
        }
        else if (strcmp(argv[i], "--sidm") == 0)
        {
            /** @note Flag to enable Self-Interacting Dark Matter physics. */
            g_enable_sidm_scattering = 1;
            // This flag does not take a value, so 'i' is not incremented further.
        }
        else if (strcmp(argv[i], "--sidm-mode") == 0)
        {
            if (i + 1 >= argc) {
                errorAndExit("--sidm-mode requires an argument (serial or parallel)", NULL, argv[0]);
            }
            char* mode_arg = argv[++i];
            if (strcmp(mode_arg, "serial") == 0) {
                g_sidm_execution_mode = 0;
            } else if (strcmp(mode_arg, "parallel") == 0) {
                #ifndef _OPENMP
                    printf("Warning: OpenMP is not enabled in this build. SIDM will run serially despite '--sidm-mode parallel'.\n");
                    log_message("WARNING", "OpenMP not enabled, SIDM forced to serial despite --sidm-mode parallel request.");
                    g_sidm_execution_mode = 0; // Force serial if no OpenMP
                #else
                    g_sidm_execution_mode = 1;
                #endif
            } else {
                errorAndExit("Invalid argument for --sidm-mode. Use 'serial' or 'parallel'.", mode_arg, argv[0]);
            }
        }
        else if (strcmp(argv[i], "--sidm-kappa") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--sidm-kappa requires a float argument", argv[i + 1], argv[0]);
            }
            g_sidm_kappa = atof(argv[++i]);
            if (g_sidm_kappa < 0) { // Kappa can be 0 (no interaction) but not negative
                errorAndExit("--sidm-kappa must be non-negative", NULL, argv[0]);
            }
            g_sidm_kappa_provided = 1;
        }
        else if (strcmp(argv[i], "--master-seed") == 0) {
            if (i + 1 >= argc || !isInteger(argv[i + 1])) {
                errorAndExit("--master-seed requires an integer argument", argv[i + 1], argv[0]);
            }
            g_master_seed = strtoul(argv[++i], NULL, 10);
            g_master_seed_provided = 1;
        } else if (strcmp(argv[i], "--init-cond-seed") == 0) {
            if (i + 1 >= argc || !isInteger(argv[i + 1])) {
                errorAndExit("--init-cond-seed requires an integer argument", argv[i + 1], argv[0]);
            }
            g_initial_cond_seed = strtoul(argv[++i], NULL, 10);
            g_initial_cond_seed_provided = 1;
        } else if (strcmp(argv[i], "--sidm-seed") == 0) {
            if (i + 1 >= argc || !isInteger(argv[i + 1])) {
                errorAndExit("--sidm-seed requires an integer argument", argv[i + 1], argv[0]);
            }
            g_sidm_seed = strtoul(argv[++i], NULL, 10);
            g_sidm_seed_provided = 1;
        } else if (strcmp(argv[i], "--load-seeds") == 0) {
            g_attempt_load_seeds = 1;
        } else if (strcmp(argv[i], "--profile") == 0) {
            if (i + 1 >= argc) {
                errorAndExit("--profile requires a type argument ('nfw', 'cored', or 'hernquist')", NULL, argv[0]);
            }
            strncpy(g_profile_type_str, argv[++i], sizeof(g_profile_type_str) - 1);
            g_profile_type_str[sizeof(g_profile_type_str) - 1] = '\0'; // Ensure null termination
            if (strcmp(g_profile_type_str, "nfw") != 0 && strcmp(g_profile_type_str, "cored") != 0 && strcmp(g_profile_type_str, "hernquist") != 0) {
                errorAndExit("Invalid argument for --profile. Use 'nfw', 'cored', or 'hernquist'.", g_profile_type_str, argv[0]);
            }
            g_profile_type_str_provided = 1;
        } else if (strcmp(argv[i], "--scale-radius") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--scale-radius requires a float argument", argv[i + 1], argv[0]);
            }
            g_scale_radius_param = atof(argv[++i]);
            if (g_scale_radius_param <= 0) errorAndExit("--scale-radius must be positive", NULL, argv[0]);
            g_scale_radius_param_provided = 1;
        } else if (strcmp(argv[i], "--halo-mass") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) { // Ensure isFloat is robust for scientific notation
                errorAndExit("--halo-mass requires a float argument", argv[i + 1], argv[0]);
            }
            g_halo_mass_param = atof(argv[++i]);
            if (g_halo_mass_param <= 0) errorAndExit("--halo-mass must be positive", NULL, argv[0]);
            g_halo_mass_param_provided = 1;
        } else if (strcmp(argv[i], "--cutoff-factor") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--cutoff-factor requires a float argument", argv[i + 1], argv[0]);
            }
            g_cutoff_factor_param = atof(argv[++i]);
            if (g_cutoff_factor_param <= 0) errorAndExit("--cutoff-factor must be positive", NULL, argv[0]);
            g_cutoff_factor_param_provided = 1;
        } else if (strcmp(argv[i], "--falloff-factor") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--falloff-factor requires a float argument", argv[i + 1], argv[0]);
            }
            g_falloff_factor_param = atof(argv[++i]);
            if (g_falloff_factor_param <= 0) errorAndExit("--falloff-factor must be positive", NULL, argv[0]);
            g_falloff_factor_param_provided = 1;
        } else if (strcmp(argv[i], "--aniso-beta") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--aniso-beta requires a float argument", argv[i + 1], argv[0]);
            }
            g_anisotropy_beta = atof(argv[++i]);
            // Physically valid range for constant beta Hernquist model
            if (g_anisotropy_beta < -1.0 || g_anisotropy_beta > 0.5) {
                errorAndExit("--aniso-beta must be in range [-1, 0.5] for Hernquist profile", NULL, argv[0]);
            }
            g_anisotropy_beta_provided = 1;
        }
        else if (strcmp(argv[i], "--aniso-factor") == 0) {
            if (g_om_aniso_factor_provided) {
                errorAndExit("Cannot use both --aniso-factor and --aniso-betascale", NULL, argv[0]);
            }
            if (i + 1 >= argc) {
                errorAndExit("--aniso-factor requires an argument", NULL, argv[0]);
            }
            
            // Check for infinity values (hidden feature for numerical isotropic)
            const char* factor_arg = argv[++i];
            if (strcmp(factor_arg, "inf") == 0 || strcmp(factor_arg, "Inf") == 0 ||
                strcmp(factor_arg, "infinity") == 0 || strcmp(factor_arg, "Infinity") == 0) {
                // Special case: trigger numerical Hernquist with true f(E) pathway
                g_om_ra_scale_factor = 1e10;  // Set large value (not actually used)
                g_use_om_profile = 1;  // Trigger numerical Hernquist
                g_use_numerical_isotropic = 1;  // But use f(E) not f(Q)
            } else {
                // Normal numeric value
                if (!isFloat(factor_arg)) {
                    errorAndExit("--aniso-factor requires a float argument", factor_arg, argv[0]);
                }
                g_om_ra_scale_factor = atof(factor_arg);
                if (g_om_ra_scale_factor <= 0) {
                    errorAndExit("--aniso-factor must be positive", NULL, argv[0]);
                }
                g_use_om_profile = 1; // Enable Osipkov-Merritt model
            }
            g_om_aniso_factor_provided = 1;
        }
        else if (strcmp(argv[i], "--aniso-betascale") == 0) {
            if (g_om_aniso_factor_provided) {
                errorAndExit("Cannot use both --aniso-factor and --aniso-betascale", NULL, argv[0]);
            }
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--aniso-betascale requires a float argument", argv[i + 1], argv[0]);
            }
            double beta_scale = atof(argv[++i]);
            // Validate beta_scale is in valid range (0, 1)
            if (beta_scale <= 0.0 || beta_scale >= 1.0) {
                errorAndExit("--aniso-betascale must be in range (0, 1)", NULL, argv[0]);
            }
            // Convert to equivalent aniso-factor: r_a/r_s = sqrt(1/beta_s - 1)
            g_om_ra_scale_factor = sqrt(1.0/beta_scale - 1.0);
            g_use_om_profile = 1; // Enable Osipkov-Merritt model
            g_om_aniso_factor_provided = 1;
        }
        else if (strcmp(argv[i], "--lvals-target") == 0)
        {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--lvals-target requires a float argument", argv[i + 1], argv[0]);
            }
            g_l_target_value = atof(argv[++i]);
            if (g_l_target_value < 0) {
                errorAndExit("--lvals-target must be non-negative", NULL, argv[0]);
            }
        }
        else if (strncmp(argv[i], "--", 2) == 0)
        {
            errorAndExit("unrecognized option", argv[i], argv[0]);
        }
        else
        {
            errorAndExit("unrecognized argument", argv[i], argv[0]);
        }
    }

    /** @note Convert user-facing method number (1-9) to internal identifier (0-8) and get description string. */
    int display_method = method_select; // Store original user input for display
    char *method_verbose_name;

    // Post-parsing validation for interdependent arguments
    if (g_anisotropy_beta_provided && strcmp(g_profile_type_str, "hernquist") != 0) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), 
                 "--aniso-beta is only compatible with '--profile hernquist', but profile is '%s'.", 
                 g_profile_type_str);
        errorAndExit(err_msg, NULL, argv[0]);
    }

    // Post-parsing validation for Osipkov-Merritt parameters
    if (g_use_om_profile) {
        // Hernquist with OM will use numerical implementation
        if (strcmp(g_profile_type_str, "hernquist") == 0 && g_anisotropy_beta_provided) {
            errorAndExit("Cannot use both --aniso-beta and --aniso-factor with Hernquist", NULL, argv[0]);
        }
        // Calculate the physical anisotropy radius r_a = factor * r_scale
        g_om_anisotropy_radius = g_om_ra_scale_factor * g_scale_radius_param;
        log_message("INFO", "Osipkov-Merritt profile enabled with r_a = %.2f * %.2f kpc = %.2f kpc",
                    g_om_ra_scale_factor, g_scale_radius_param, g_om_anisotropy_radius);
    }

    // Convert method number to internal index and set descriptive name.
    switch (method_select)
    {
    case 1:
        method_select = 5;
        method_verbose_name = "Adaptive Leapfrog with Adaptive Levi-Civita";
        break;
    case 2:
        method_select = 4;
        method_verbose_name = "Full-Step Adaptive Leapfrog + Levi-Civita";
        break;
    case 3:
        method_select = 3;
        method_verbose_name = "Full-Step Adaptive Leapfrog";
        break;
    case 4:
        method_select = 6;
        method_verbose_name = "Yoshida 4th-Order";
        break;
    case 5:
        method_select = 8;
        method_verbose_name = "Adams-Bashforth 3rd-Order";
        break;
    case 6:
        method_select = 2;
        method_verbose_name = "Leapfrog (Vel Half-Step)";
        break;
    case 7:
        method_select = 1;
        method_verbose_name = "Leapfrog (Pos Half-Step)";
        break;
    case 8:
        method_select = 7;
        method_verbose_name = "Classic RK4";
        break;
    case 9:
        method_select = 0;
        method_verbose_name = "Euler";
        break;
    default:
        method_verbose_name = "Unknown Method";
        break;
    }

    /** @note Generate internal method string identifier for filenames. */
    char method_str[32];
    switch (method_select)
    {
    case 0:
        strcpy(method_str, "euler");
        break;
    case 1:
        strcpy(method_str, "pos.leap");
        break;
    case 2:
        strcpy(method_str, "vel.leap");
        break;
    case 3:
        strcpy(method_str, "adp.leap");
        break;
    case 4:
        strcpy(method_str, "adp.leap.levi");
        break;
    case 5:
        strcpy(method_str, "adp.leap.adp.levi");
        break;
    case 6:
        strcpy(method_str, "fr4.yoshi");
        break;
    case 7:
        strcpy(method_str, "rk4");
        break;
    case 8:
        strcpy(method_str, "ab3");
        break;
    default:
        strcpy(method_str, "unknown");
        break;
    }

    // Determine active profile type (NFW is default)
    if (g_profile_type_str_provided) {
        if (strcmp(g_profile_type_str, "nfw") == 0) {
            g_use_nfw_profile = 1;
        } else if (strcmp(g_profile_type_str, "cored") == 0) {
            g_use_nfw_profile = 0;
        } else if (strcmp(g_profile_type_str, "hernquist") == 0) {
            // Use numerical Hernquist if OM is requested, otherwise analytical
            if (g_use_om_profile) {
                g_use_hernquist_numerical = 1;
                g_use_hernquist_aniso_profile = 0;
            } else {
                g_use_hernquist_aniso_profile = 1;
                g_use_hernquist_numerical = 0;
            }
            g_use_nfw_profile = 0; // Ensure other profile flags are off
        } else {
            // Parser validation fallback for unknown profile type
            log_message("WARNING", "Unknown profile type '%s', defaulting to NFW.", g_profile_type_str);
            g_use_nfw_profile = 1;
        }
    } else {
        // Default to NFW if --profile flag was not provided
        g_use_nfw_profile = 1;
        strcpy(g_profile_type_str, "nfw"); // Update string for consistency in printouts
    }

    // Set up profile-specific parameters based on generalized flags and profile defaults
    if (g_use_nfw_profile) {
        // NFW Profile Path
        // Halo Mass for NFW
        if (g_halo_mass_param_provided) { // --halo-mass overrides NFW default
            g_nfw_profile_halo_mass = g_halo_mass_param;
        } else { // No --halo-mass, NFW uses its own default
            g_nfw_profile_halo_mass = HALO_MASS_NFW;
            g_halo_mass_param = g_nfw_profile_halo_mass; // Update general param to reflect NFW's choice
        }
        // Scale Radius for NFW
        if (g_scale_radius_param_provided) { // --scale-radius overrides NFW default
            g_nfw_profile_rc = g_scale_radius_param;
        } else { // No --scale-radius, NFW uses its own default
            g_nfw_profile_rc = RC_NFW_DEFAULT;
            g_scale_radius_param = g_nfw_profile_rc; // Update general param to reflect NFW's choice
        }
        // Cutoff Factor for NFW
        if (g_cutoff_factor_param_provided) { // --cutoff-factor overrides NFW default
            g_nfw_profile_rmax_norm_factor = g_cutoff_factor_param;
        } else { // No --cutoff-factor, NFW uses its own default
            g_nfw_profile_rmax_norm_factor = CUTOFF_FACTOR_NFW_DEFAULT;
            // g_cutoff_factor_param is NOT updated here by NFW default; it keeps its own (Cored's) default or user value.
        }
        // Falloff Factor for NFW
        if (g_falloff_factor_param_provided) { // --falloff-factor overrides NFW default
            g_nfw_profile_falloff_factor = g_falloff_factor_param;
        } else { // No --falloff-factor, NFW uses its own default
            g_nfw_profile_falloff_factor = FALLOFF_FACTOR_NFW_DEFAULT;
            // Optionally, update g_falloff_factor_param if NFW is the overall default and no flag given
            // Preserve g_falloff_factor_param default unless explicitly set by user
        }

    } else if (g_use_hernquist_aniso_profile) {
        // Hernquist Profile Path
        // Use generalized parameters directly (no separate profile-specific variables)
        // Parameters g_halo_mass_param and g_scale_radius_param passed to IC generator
        // Cutoff factor also uses generalized parameter
    } else {
        // Cored Profile Path
        // Halo Mass for Cored (already defaults to HALO_MASS or takes from --halo-mass via g_halo_mass_param)
        g_cored_profile_halo_mass = g_halo_mass_param;
        // Scale Radius for Cored (already defaults to RC or takes from --scale-radius via g_scale_radius_param)
        g_cored_profile_rc = g_scale_radius_param;
        // Cutoff Factor for Cored (directly uses generalized or its (Cored's) default)
        g_cored_profile_rmax_factor = g_cutoff_factor_param;
    }

    // Set the single g_active_halo_mass for N-body forces and tdyn from the finalized g_halo_mass_param
    g_active_halo_mass = g_halo_mass_param;
    
    // Validate --sim-extend requirements
    if (g_doSimExtend) {
        if (g_extend_file_source == NULL) {
            errorAndExit("ERROR: --sim-extend requires --extend-file to specify the source file", NULL, argv[0]);
        }
        
        // Check if the extend-file exists
        FILE *check_extend = fopen(g_extend_file_source, "rb");
        if (check_extend == NULL) {
            char error_msg[512];
            snprintf(error_msg, sizeof(error_msg),
                     "ERROR: --extend-file source '%s' does not exist or cannot be opened",
                     g_extend_file_source);
            errorAndExit(error_msg, NULL, argv[0]);
        }
        fclose(check_extend);

        // Validate filename format early (fail fast if wrong format)
        // Extend mode requires parseable filenames to locate auxiliary files
        int early_src_N, early_src_Ntimes;
        double early_src_tfinal;
        if (!parse_nsphere_filename(g_extend_file_source, &early_src_N, &early_src_Ntimes, &early_src_tfinal)) {
            char error_msg[512];
            snprintf(error_msg, sizeof(error_msg),
                     "ERROR: --extend-file '%s' does not match required format.\n"
                     "       Required: data/all_particle_data_<tag>_<N>_<Ntimes>_<tfinal>.dat\n"
                     "       Example:  data/all_particle_data_run1_5000_10001_2.dat\n"
                     "\n"
                     "       Extend mode requires properly formatted filenames to locate:\n"
                     "         - Auxiliary files (IDs, phi, chosen_particles)\n"
                     "         - Double-precision buffers (for reproducibility)",
                     g_extend_file_source);
            errorAndExit(error_msg, NULL, argv[0]);
        }

        printf("Parsed extend-file: N=%d, Ntimes=%d, tfinal=%.5g\n",
               early_src_N, early_src_Ntimes, early_src_tfinal);

        // Ensure --sim-extend is not used with conflicting options
        if (g_doSimRestart) {
            errorAndExit("ERROR: Cannot use --sim-extend and --sim-restart together", NULL, argv[0]);
        }
        if (g_doRestart) {
            errorAndExit("ERROR: Cannot use --sim-extend and --restart together", NULL, argv[0]);
        }
        if (doReadInit) {
            errorAndExit("ERROR: Cannot use --sim-extend and --readinit together", NULL, argv[0]);
        }
        if (tidal_fraction > 0.0) {
            errorAndExit("ERROR: Cannot use --sim-extend with --ftidal (tidal stripping)", NULL, argv[0]);
        }
    }
    
    // Validate that --extend-file is only used with --sim-extend
    if (g_extend_file_source != NULL && !g_doSimExtend) {
        errorAndExit("ERROR: --extend-file can only be used with --sim-extend", NULL, argv[0]);
    }
    
    // Validate that tidal stripping is not used with sim-restart
    if (g_doSimRestart && tidal_fraction > 0.0) {
        errorAndExit("ERROR: Cannot use --sim-restart with --ftidal (tidal stripping)", NULL, argv[0]);
    }
    
    /** @note Display final parameter values used for the simulation run. */
    printf("Parameter values requested:\n\n");
    
    printf("  Number of Particles:          %d\n", npts);
    printf("  Number of Time Steps:         %d\n", Ntimes);
    printf("  Number of Dynamical Times:    %d\n", tfinal_factor);
    printf("  Number of Output Snapshots:   %d\n", nout);
    printf("  Steps Between Writes:         %d\n", dtwrite);
    printf("  Snapshot Buffer Size:         %d\n", snapshot_block_size);
    printf("  Tidal Stripping Fraction:     %.5f\n", tidal_fraction);
    printf("  Integration Method:           %d (%s)\n", display_method, method_verbose_name);
    printf("  Sorting Algorithm:            %d (%s)\n", display_sort, get_sort_description(g_defaultSortAlg));
    /** @note Build the filename tag string based on options for display purposes. */
    char filename_tag[512] = "";

    if (custom_tag[0] != '\0')
    {
        strcat(filename_tag, custom_tag);
    }

    if (include_method_in_suffix)
    {
        if (filename_tag[0] != '\0')
        {
            strcat(filename_tag, "_");
        }
        strcat(filename_tag, method_str);
    }

    printf("  Filename Tag:                 %s\n", filename_tag[0] ? filename_tag : "[none]");
    printf("  SIDM Scattering:              %s\n", g_enable_sidm_scattering ? "Enabled via --sidm" : "Disabled (Default)");
    printf("  SIDM Execution Mode:          %s\n", g_sidm_execution_mode == 1 ? "Parallel (Default)" : "Serial");
    printf("  SIDM Opacity Kappa:           %.1f cm^2/g (Default: 50.0, User set: %s)\n", g_sidm_kappa, g_sidm_kappa_provided ? "Yes" : "No");

    if (g_use_om_profile) {
        if (g_use_numerical_isotropic) {
            printf("  Anisotropy Model:             Isotropic (using numerical Eddington pathway)\n");
        } else {
            printf("  Anisotropy Model:             Osipkov-Merritt\n");
            printf("    Anisotropy Radius (r_a):    %.2f kpc (%.2f x scale radius)\n", g_om_anisotropy_radius, g_om_ra_scale_factor);
            double beta_at_scale = 1.0 / (1.0 + g_om_ra_scale_factor * g_om_ra_scale_factor);
            printf("    β at scale radius:          %.4f\n", beta_at_scale);
        }
    }
    
    const char* profile_name = g_use_hernquist_numerical ? "Numerical Hernquist (OM-compatible)" : 
                              (g_use_hernquist_aniso_profile ? "Anisotropic Hernquist" : 
                              (g_use_nfw_profile ? "NFW-like with Cutoff" : "Cored Plummer-like"));
    printf("  Initial Conditions Profile:   %s\n", profile_name);
    if (g_use_nfw_profile) {
        printf("    NFW Profile Scale Radius (IC): %.3f kpc (NFW Default: %.2f, User set via --scale-radius: %s)\n", g_nfw_profile_rc, RC_NFW_DEFAULT, g_scale_radius_param_provided ? "Yes" : "No");
    } else if (g_use_hernquist_aniso_profile) {
        printf("    Hernquist Profile Scale Radius (IC): %.3f kpc (Default: %.2f, User set via --scale-radius: %s)\n", g_scale_radius_param, RC, g_scale_radius_param_provided ? "Yes" : "No");
        printf("    Hernquist Anisotropy Beta:    %.3f (Default: 0.0, User set via --aniso-beta: %s)\n", g_anisotropy_beta, g_anisotropy_beta_provided ? "Yes" : "No");
    } else if (g_use_hernquist_numerical) {
        printf("    Hernquist Profile Scale Radius (IC): %.3f kpc (Default: %.2f, User set via --scale-radius: %s)\n", g_scale_radius_param, RC, g_scale_radius_param_provided ? "Yes" : "No");
    } else {
        printf("    Cored Profile Scale Radius (IC): %.3f kpc (Cored Default: %.2f, User set via --scale-radius: %s)\n", g_cored_profile_rc, RC, g_scale_radius_param_provided ? "Yes" : "No");
    }
    if (g_use_nfw_profile) {
        printf("    NFW Profile Halo Mass (IC): %.3e Msun (NFW Default: %.2e, User set via --halo-mass: %s)\n", g_nfw_profile_halo_mass, HALO_MASS_NFW, g_halo_mass_param_provided ? "Yes" : "No");
    } else if (g_use_hernquist_aniso_profile) {
        printf("    Hernquist Profile Halo Mass (IC): %.3e Msun (User set via --halo-mass: %s)\n", g_halo_mass_param, g_halo_mass_param_provided ? "Yes" : "No");
    } else {
        printf("    Cored Profile Halo Mass (IC): %.3e Msun (Cored Default: %.2e, User set via --halo-mass: %s)\n", g_cored_profile_halo_mass, HALO_MASS, g_halo_mass_param_provided ? "Yes" : "No");
    }
    printf("    Profile Cutoff Factor:      %.1f (CmdLine/Default: %.1f, User set: %s)\n", g_cutoff_factor_param, (g_use_nfw_profile ? CUTOFF_FACTOR_NFW_DEFAULT : (g_use_hernquist_aniso_profile ? CUTOFF_FACTOR_NFW_DEFAULT : CUTOFF_FACTOR_CORED_DEFAULT)), g_cutoff_factor_param_provided ? "Yes" : "No");
    
    if (g_use_nfw_profile) {
        printf("    NFW Profile Falloff Factor (C): %.1f (NFW Default: %.1f, User set via --falloff-factor: %s)\n", g_nfw_profile_falloff_factor, FALLOFF_FACTOR_NFW_DEFAULT, g_falloff_factor_param_provided ? "Yes" : "No");
    }
    
    // Display Osipkov-Merritt parameters if enabled (but not for numerical isotropic)
    if (g_use_om_profile && !g_use_numerical_isotropic) {
        printf("    Osipkov-Merritt Model:      ENABLED\n");
        printf("    Anisotropy Radius (r_a):    %.2f kpc (%.2f × scale radius)\n", 
               g_om_anisotropy_radius, g_om_ra_scale_factor);
    }
    
    // This g_active_halo_mass is now correctly set from g_halo_mass_param which reflects the chosen profile's mass
    printf("  N-body Active Halo Mass (tdyn): %.3e Msun\n", g_active_halo_mass);
    
    /** @note Display logging status based on g_enable_logging flag. */
    if (g_enable_logging)
    {
        printf("  Logging:                      Enabled (log/nsphere.log)\n\n");
        log_message("INFO", "Simulation started with %d particles, %d timesteps, %d dynamical times",
                    npts, Ntimes, tfinal_factor);
    }
    else
    {
        printf("  Logging:                      Disabled\n\n");
    }

    // Check for SIDM + parallel mode without OpenMP
    if (g_enable_sidm_scattering && g_sidm_execution_mode == 1) {
        #ifndef _OPENMP
            printf("Warning: SIDM parallel mode is enabled by default, but OpenMP is not available in this build.\n");
            printf("         SIDM will run serially. Use '--sidm-mode serial' to suppress this warning.\n\n");
            log_message("WARNING", "SIDM parallel mode requested but OpenMP not available, will run serially.");
            g_sidm_execution_mode = 0; // Force serial if no OpenMP
        #endif
    }

    // Validation occurs earlier in the argument parsing loop.

    /**
     * @brief Oversample initial conditions based on tidal fraction.
     * @details Calculate the number of initial particles (`npts_initial`) needed
     *          before tidal stripping to ensure `npts` particles remain afterwards.
     *          If `tidal_fraction` is 0, `npts_initial` equals `npts`.
     */
    int npts_initial;
    if (tidal_fraction > 0.0)
    {
        // Use ceiling to ensure enough particles remain after stripping
        npts_initial = ceil(npts / (1.0 - tidal_fraction));
    }
    else
    {
        npts_initial = npts;
    }

    /** @note Calculate number of snapshots (`noutsnaps`) based on desired intervals (`nout`). */
    nout = nout + 1; // nout specifies intervals, noutsnaps is number of points (intervals + 1)
    noutsnaps = nout;

    // Allocate arrays for total energy diagnostics if full data saving is enabled.
    if (g_doAllParticleData) {
        g_time_snapshots = (double *)malloc(noutsnaps * sizeof(double));
        g_total_KE = (double *)malloc(noutsnaps * sizeof(double));
        g_total_PE = (double *)malloc(noutsnaps * sizeof(double));
        g_total_E = (double *)malloc(noutsnaps * sizeof(double));

        if (!g_time_snapshots || !g_total_KE || !g_total_PE || !g_total_E) {
            fprintf(stderr, "Error: Failed to allocate memory for total energy diagnostic arrays.\n");
            // Free any that were successfully allocated before exiting
            free(g_time_snapshots);
            free(g_total_KE);
            free(g_total_PE);
            free(g_total_E);
            CLEAN_EXIT(1);
        }
        
        // Note: Existing energy data loaded after restart variable initialization
    }

    /**
     * @brief Adjust Ntimes using adjust_ntimesteps to align with output schedule.
     * @details Ensures (Ntimes - 1) is a multiple of (noutsnaps - 1) * dtwrite.
     * @see adjust_ntimesteps
     */
    int oldN = Ntimes;
    Ntimes = adjust_ntimesteps(Ntimes, noutsnaps, dtwrite); // Use noutsnaps here
    if (Ntimes != oldN)
    {
        printf("Adjusted Number of Time Steps to %d to satisfy parameter constraints.\n", Ntimes);
    }
    /** @brief Calculate total number of write events and steps between major snapshots. */
    int total_writes = ((Ntimes - 1) / dtwrite) + 1; // Total potential write points

    // Restart-related variables (declared early for use in IC generation)
    int restart_completed_snapshots = 0;  // Number of snapshots already completed
    double restart_initial_time = 0.0;    // Initial time for restarted simulation
    int restart_initial_nwrite = 0;       // Initial write count for restarted simulation
    char restart_source_file[512] = "";   // File to read restart data from
    int original_Ntimes = 0;              // Original Ntimes before restart adjustment
    /**
     * @brief EARLY RESTART/EXTEND DETECTION
     * @details This block must execute before IC generation to ensure restart flags are set.
     *          It performs minimal detection to set g_restart_mode_active and restart parameters,
     *          while deferring actual file operations to later.
     */
    if (g_doSimExtend && g_extend_file_source != NULL) {
        // Early detection for --sim-extend
        FILE *source_check = fopen(g_extend_file_source, "rb");
        if (source_check) {
            // Get file size to determine snapshot count
            fseek(source_check, 0, SEEK_END);
            long source_size = ftell(source_check);
            fclose(source_check);
            
            // Parse filename to get N value
            int src_N = 0, src_Ntimes = 0;
            double src_tfinal = 0;
            int parsed_ok = parse_nsphere_filename(g_extend_file_source, &src_N, &src_Ntimes, &src_tfinal);
            
            if (!parsed_ok) {
                // Try to infer N from command line since filename parsing failed
                src_N = npts;  // Assume same N as current run
            }
            
            if (src_N == npts) {
                // Calculate snapshots in source file
                long long source_records = source_size / 16;
                long long source_snapshots = source_records / npts;
                
                if (source_records % npts == 0) {
                    // Valid file structure - set restart parameters NOW
                    restart_completed_snapshots = (int)source_snapshots;
                    restart_initial_nwrite = (int)source_snapshots - 1;
                    
                    // CRITICAL: Set restart_source_file to SOURCE file for IC loading
                    // Points to target filename for continued simulation
                    strncpy(restart_source_file, g_extend_file_source, sizeof(restart_source_file) - 1);
                    restart_source_file[sizeof(restart_source_file) - 1] = '\0';
                    
                    // Mark restart mode active so IC generation loads from file
                    g_restart_mode_active = 1;
                    
                }
            }
        }
    }
    
    // Early detection for --sim-restart
    if (g_doSimRestart) {
        // Build the expected apd_filename early (same logic as later)
        char predicted_suffix[512];
        predicted_suffix[0] = '\0';
        
        // Add custom tag if provided
        if (custom_tag[0] != '\0') {
            snprintf(predicted_suffix, sizeof(predicted_suffix), "_%s", custom_tag);
        }
        
        // Add method/parameter tag
        char temp[256];
        if (include_method_in_suffix) {
            snprintf(temp, sizeof(temp), "_%s_%d_%d_%d", method_str, npts, Ntimes, tfinal_factor);
        } else {
            snprintf(temp, sizeof(temp), "_%d_%d_%d", npts, Ntimes, tfinal_factor);
        }
        strncat(predicted_suffix, temp, sizeof(predicted_suffix) - strlen(predicted_suffix) - 1);
        
        // Build predicted filename
        char predicted_apd[512];
        snprintf(predicted_apd, sizeof(predicted_apd), "data/all_particle_data%s.dat", predicted_suffix);
        
        // Check if this file exists and is incomplete
        FILE *check = fopen(predicted_apd, "rb");
        if (check) {
            // Get file size
            fseek(check, 0, SEEK_END);
            long actual_size = ftell(check);
            fclose(check);
            
            // Calculate completion
            long long expected_complete_size = (long long)total_writes * (long long)npts * 16LL;
            long long completed_records = actual_size / 16;
            long long completed_snapshots = completed_records / npts;
            
            if (actual_size > 0 && actual_size < expected_complete_size) {
                // Incomplete simulation found - set restart parameters
                // In early detection: completed_snapshots is a COUNT
                restart_completed_snapshots = (int)completed_snapshots;
                restart_initial_nwrite = restart_completed_snapshots - 1;  // Convert COUNT to INDEX
                strncpy(restart_source_file, predicted_apd, sizeof(restart_source_file) - 1);
                restart_source_file[sizeof(restart_source_file) - 1] = '\0';
                g_restart_mode_active = 1;
                g_restart_snapshots_is_count = 1;  // Flag that restart_completed_snapshots is a COUNT, not INDEX

                log_message("DEBUG", "Early detection: completed_snapshots=%lld (COUNT)", completed_snapshots);
                log_message("DEBUG", "Early detection: restart_completed_snapshots=%d (COUNT)", restart_completed_snapshots);
                log_message("DEBUG", "Early detection: restart_initial_nwrite=%d (INDEX)", restart_initial_nwrite);
                log_message("DEBUG", "Early detection: g_restart_snapshots_is_count=%d", g_restart_snapshots_is_count);
                printf("Restart mode: Loading snapshot %d (%.1f%% complete)\n",
                       restart_completed_snapshots - 1,
                       (actual_size * 100.0) / expected_complete_size);
            } else if (actual_size >= expected_complete_size) {
                printf("Simulation already complete. Nothing to restart.\n");
                // Could exit here or let it continue to late detection
            }
        }
        // File absence handled by late detection logic
    }
    
    int stepBetweenSnaps = (int)floor(
        (double)(total_writes - 1) / (double)(noutsnaps - 1) + 0.5); // Steps between major snapshots

    /** @brief Initialize the global file suffix string `g_file_suffix` based on cmd line args. */
    g_file_suffix[0] = '\0';

    /** @brief Add custom tag to suffix if provided via `--tag`. */
    if (custom_tag[0] != '\0')
    {
        snprintf(g_file_suffix, sizeof(g_file_suffix), "_%s", custom_tag);
    }

    /** @brief Add method/parameter tag to suffix. */
    char temp[256];
    if (include_method_in_suffix)
    {
        snprintf(temp, sizeof(temp), "_%s_%d_%d_%d", method_str, npts, Ntimes, tfinal_factor);
    }
    else
    {
        snprintf(temp, sizeof(temp), "_%d_%d_%d", npts, Ntimes, tfinal_factor);
    }

    /** @brief Append the parameter tag to the global suffix. */
    strcat(g_file_suffix, temp); // Append temp to g_file_suffix

    /** @brief Set flag `skip_file_writes=1` if in restart mode (`g_doRestart`). */
    if (g_doRestart)
    {
        printf("Restart mode: Skipping simulation phase and file writes, proceeding directly to data product generation.\n\n");
        skip_file_writes = 1;
    }

    /** @brief Create the output 'data' directory if it does not exist. */
    {
        struct stat st = {0};
        if (stat("data", &st) == -1)
        {
            mkdir("data", 0755); // POSIX standard, works on most systems including MinGW/Cygwin
        }
    }

    /** @brief Create the 'init' directory if it does not exist. */
    {
        struct stat st_init = {0};
        if (stat("init", &st_init) == -1)
        {
            #if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
                if (mkdir("init") != 0) {
                     perror("Error creating init directory");
                     // Non-fatal if only writing to init directory
                } else {
                     log_message("INFO", "Created init/ directory.");
                }
            #else
                if (mkdir("init", 0755) != 0) { // POSIX standard
                     perror("Error creating init directory");
                     // Decide if this is fatal
                } else {
                     log_message("INFO", "Created init/ directory.");
                }
            #endif
        }
    }

    /** @brief Write current run parameters to `data/lastparams<suffix>.dat` and create a standard link `data/lastparams.dat`. */
    {
        char file_tag[512] = "";
        if (custom_tag[0] != '\0')
        {
            strcat(file_tag, custom_tag);
            if (include_method_in_suffix)
            {
                strcat(file_tag, "_");
                strcat(file_tag, method_str);
            }
        }
        else if (include_method_in_suffix)
        {
            strcat(file_tag, method_str);
        }

        /** @note Create filename with suffix for the specific run parameters file. */
        char filename[512]; // Holds suffixed filename, e.g., data/lastparams_run1_100k_10k_5.dat
        get_suffixed_filename("data/lastparams.dat", 1, filename, sizeof(filename));
        printf("Saving parameters: %s\n", filename);

        FILE *fp_params = fopen(filename, "w"); // Text mode for regular fprintf
        if (!fp_params)
        {
            printf("Error: cannot open %s\n", filename);
            return 1;
        }

        fprintf(fp_params, "%d %d %d %s\n", npts, Ntimes, tfinal_factor, file_tag);
        fclose(fp_params);

        /** @note Create a standard-named link `data/lastparams.dat` pointing to the
                 suffixed version for compatibility with scripts. Uses copy on Windows. */
        char linkname[512] = "data/lastparams.dat"; // Standard name

        /* Platform detection using standard predefined macros */
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
        /** @note Windows: Copy file content as symlinks can be unreliable or require special privileges. */
        // Windows or Windows-like environment: create a direct file copy.
            // Ensure compatibility with various Windows environments.

        // Use lower-level file operations instead of system commands for better compatibility.
        FILE *source, *dest;
        source = fopen(filename, "rb");
        if (!source)
        {
            printf("Warning: Failed to open source file %s for copying\n", filename);
        }
        else
        {
            dest = fopen(linkname, "wb");
            if (!dest)
            {
                printf("Warning: Failed to create destination file %s\n", linkname);
                fclose(source);
            }
            else
            {
                // Copy file content.
                char buffer[4096];
                size_t bytes_read;

                while ((bytes_read = fread(buffer, 1, sizeof(buffer), source)) > 0)
                {
                    fwrite(buffer, 1, bytes_read, dest);
                }

                fclose(dest);
                fclose(source);

                // Extract the basename for display purposes
                const char *basename = strrchr(filename, '/');
                basename = basename ? basename + 1 : filename; // Skip the '/' or use full name if no '/'

                printf("Created link: %s -> %s\n\n", basename, linkname);
            }
        }
#else
        /** @note Unix: Create symbolic link from basename(filename) to 'linkname', fallback to copy. */
            // Unix-like systems (Linux, macOS, etc.) and fallback for other platforms: use symbolic links.
        char command[1024];

        // First, remove any existing link or file.
        snprintf(command, sizeof(command), "rm -f \"%s\" 2>/dev/null", linkname);
        system(command);

        // Then create the symbolic link - use the basename of the file, not the full path
        // Extract the basename from filename
        const char *basename = strrchr(filename, '/');
        basename = basename ? basename + 1 : filename; // Skip the '/' or use full name if no '/'

        snprintf(command, sizeof(command), "ln -s \"%s\" \"%s\"", basename, linkname);
        if (system(command) != 0)
        {
            // If symbolic link fails, fall back to copying the file.
            snprintf(command, sizeof(command), "cp \"%s\" \"%s\"", filename, linkname);
            if (system(command) != 0)
            {
                printf("Warning: Failed to create link or copy %s to %s\n", filename, linkname);
            }
            else
            {
                printf("Created link: %s -> %s\n\n", filename, linkname);
            }
        }
        else
        {
            printf("Created link: %s -> %s\n\n", filename, linkname);
        }
#endif
    }

    /**
     * @brief Common IC generation variables shared between profile pathways.
     * @details Variables declared before profile selection block;
     *          populated by the selected profile pathway (NFW, Hernquist, or Cored).
     */
    double **particles = NULL;           ///< Main particle data array
    int i = 0;                          ///< Loop counter
    double result, error;               ///< GSL integration results
    double calE;                        ///< Energy value for calculations
    gsl_integration_workspace *w = NULL; ///< GSL integration workspace
    
    // Common spline objects and accelerators
    gsl_spline *splinemass = NULL;      ///< Spline for mass profile M(r)
    gsl_interp_accel *enclosedmass = NULL; ///< Accelerator for mass spline
    gsl_spline *splinePsi = NULL;       ///< Spline for potential profile Psi(r)
    gsl_interp_accel *Psiinterp = NULL; ///< Accelerator for potential spline
    gsl_spline *splinerofPsi = NULL;    ///< Spline for inverse potential r(Psi)
    gsl_interp_accel *rofPsiinterp = NULL; ///< Accelerator for r(Psi) spline
    gsl_interp *g_main_fofEinterp = NULL;  ///< Main f(E) interpolator
    gsl_interp_accel *g_main_fofEacc = NULL; ///< Accelerator for f(E)
    
    // Common data arrays
    double *radius = NULL;              ///< Radial grid points (may be sorted for Hernquist r(Psi) spline)
    double *radius_unsorted = NULL;     ///< Unsorted radial grid (for Hernquist file output)
    double *mass = NULL;                ///< Mass values at radial points
    double *Psivalues = NULL;           ///< Potential values at radial points
    double *nPsivalues = NULL;          ///< Negative potential values (for r(Psi) spline)
    double *Evalues = NULL;             ///< Energy grid points
    double *innerintegrandvalues = NULL; ///< f(E) integrand values
    double *radius_monotonic_grid_nfw = NULL; ///< Monotonic radial grid for NFW calculations
    
    // Key scalar values
    double Psimin = 0.0;                ///< Minimum potential (at \f$r_{\text{max}}\f$)
    double Psimax = 0.0;                ///< Maximum potential (at r=0)
    double rmax = 0.0;                  ///< Maximum radius for profile calculations
    int num_points = 0;                 ///< Number of points for spline interpolation
    
    // File handling
    char fname[256];                    ///< Buffer for file names
    FILE *fp;                           ///< File pointer for data output
    
    /**
     * @brief Allocate main particle data array before profile selection.
     * @details This ensures both NFW and Cored pathways use the same particles array.
     */
    particles = (double **)malloc(7 * sizeof(double *));  // Rows 5,6 for cos(phi), sin(phi)
    if (particles == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed for particle array pointer\n");
        CLEAN_EXIT(1);
    }
    for (i = 0; i < 7; i++) {  // 7 rows for extended particle state
        particles[i] = (double *)malloc(npts_initial * sizeof(double));
        if (particles[i] == NULL) {
            fprintf(stderr, "ERROR: Memory allocation failed for particles[%d]\n", i);
            CLEAN_EXIT(1);
        }
    }

    // Restart-related variables (declared early for use in IC generation)
    
    /**
     * @brief Initialize random number generator for particle generation.
     * @details Sets up the GSL Random Number Generator environment and allocates
     *          the global GSL RNG state used throughout the simulation.
     *          Needed for the Sample Generator if not reading ICs from file.
     */
    gsl_rng_env_setup();                           // Setup GSL RNG environment
    const gsl_rng_type * T_rng = gsl_rng_default;  // Use default RNG type
    g_rng = gsl_rng_alloc(T_rng);                  // Allocate global GSL RNG state
    if (g_rng == NULL) {                           // Check allocation success
        fprintf(stderr, "Error allocating GSL RNG.\n");
        CLEAN_EXIT(1);
    }
    
    /**
     * Seed determination logic (moved here to happen BEFORE RNG seeding):
     * 1. If a specific seed (`--initial-cond-seed` or `--sidm-seed`) is provided, use it.
     * 2. Else if `--master-seed` is provided, derive specific seeds from it.
     * 3. Else if `--load-seeds` is specified, try to load from last_X_seed_{suffix}.dat.
     * 4. Else (no seeds provided, no load requested), generate new seeds from time/pid.
     * Finally, save the seeds actually used to last_X_seed_{suffix}.dat and link last_X_seed.dat.
     */
    
    // Generate a more unique seed using time, PID, and microseconds
    struct timeval tv;
    gettimeofday(&tv, NULL);
    unsigned long int current_time_pid_seed = ((unsigned long int)tv.tv_sec * 1000000UL + (unsigned long int)tv.tv_usec) ^ (unsigned long int)getpid();
    char seed_filepath[512];
    FILE *fp_seed;
    
    // Determine Initial Conditions Seed
    if (!g_initial_cond_seed_provided) {
        if (g_master_seed_provided) {
            g_initial_cond_seed = g_master_seed + 1; // Deterministic offset
        } else if (g_attempt_load_seeds) {
            get_suffixed_filename(g_initial_cond_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
            fp_seed = fopen(seed_filepath, "r");
            if (fp_seed) {
                if (fscanf(fp_seed, "%lu", &g_initial_cond_seed) == 1) {
                    log_message("INFO", "Loaded initial conditions seed %lu from %s", g_initial_cond_seed, seed_filepath);
                } else {
                    g_initial_cond_seed = current_time_pid_seed + 100; // Fallback if read fails
                    log_message("WARNING", "Failed to read IC seed from %s, generating new: %lu", seed_filepath, g_initial_cond_seed);
                }
                fclose(fp_seed);
            } else {
                g_initial_cond_seed = current_time_pid_seed + 100; // File not found, generate
                log_message("INFO", "No IC seed file found, generating new: %lu", g_initial_cond_seed);
            }
        } else {
            g_initial_cond_seed = current_time_pid_seed + 100; // Default generation
            log_message("INFO", "Generating new IC seed from time+PID: %lu (no --load-seeds, no explicit seed)", g_initial_cond_seed);
            printf("Generated new IC seed from time+PID: %lu\n", g_initial_cond_seed);
        }
    } else {
        log_message("INFO", "Using user-provided IC seed: %lu", g_initial_cond_seed);
    }
    
    // Determine SIDM Seed
    if (!g_sidm_seed_provided) {
        if (g_master_seed_provided) {
            g_sidm_seed = g_master_seed + 2; // Deterministic offset, different from IC seed
        } else if (g_attempt_load_seeds) {
            get_suffixed_filename(g_sidm_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
            fp_seed = fopen(seed_filepath, "r");
            if (fp_seed) {
                if (fscanf(fp_seed, "%lu", &g_sidm_seed) == 1) {
                    log_message("INFO", "Loaded SIDM seed %lu from %s", g_sidm_seed, seed_filepath);
                } else {
                    g_sidm_seed = current_time_pid_seed + 200; // Fallback
                    log_message("WARNING", "Failed to read SIDM seed from %s, generating new: %lu", seed_filepath, g_sidm_seed);
                }
                fclose(fp_seed);
            } else {
                g_sidm_seed = current_time_pid_seed + 200; // File not found, generate
                log_message("INFO", "No SIDM seed file found, generating new: %lu", g_sidm_seed);
            }
        } else {
            g_sidm_seed = current_time_pid_seed + 200; // Default generation
            log_message("INFO", "Generating new SIDM seed: %lu", g_sidm_seed);
        }
    } else {
        log_message("INFO", "Using user-provided SIDM seed: %lu", g_sidm_seed);
    }
    
    // Save the seeds that will actually be used
    // Save Initial Conditions Seed
    get_suffixed_filename(g_initial_cond_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
    fp_seed = fopen(seed_filepath, "w");
    if (fp_seed) {
        fprintf(fp_seed, "%lu\n", g_initial_cond_seed);
        fclose(fp_seed);
        log_message("INFO", "Saved initial conditions seed %lu to %s", g_initial_cond_seed, seed_filepath);
    }
    
    // Save SIDM Seed  
    get_suffixed_filename(g_sidm_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
    fp_seed = fopen(seed_filepath, "w");
    if (fp_seed) {
        fprintf(fp_seed, "%lu\n", g_sidm_seed);
        fclose(fp_seed);
        log_message("INFO", "Saved SIDM seed %lu to %s", g_sidm_seed, seed_filepath);
    }
    
    // Seed the global g_rng (used for ICs and Serial SIDM)
    gsl_rng_set(g_rng, g_initial_cond_seed);       // Use the determined IC seed for g_rng
    log_message("INFO", "Global g_rng (intended primarily for IC generation) seeded with %lu", g_initial_cond_seed);

    // Initialize persistent cos(phi) and sin(phi) orientation angles for SIDM scattering.
    // Azimuthal angles represent perpendicular velocity direction in spherical coordinates.
    // Random initialization in [0, 2π]; updated during SIDM scattering events.
    if (!doReadInit && !g_restart_mode_active) {
        log_message("INFO", "Initializing persistent cos/sin(phi) angles for %d particles.", npts_initial);
        for (i = 0; i < npts_initial; i++) {
            double phi_val = 2.0 * PI * gsl_rng_uniform(g_rng);  // Random phi in [0, 2π]
            particles[5][i] = cos(phi_val);  // Store cos(phi)
            particles[6][i] = sin(phi_val);  // Store sin(phi)
        }
    }

    // Initialize per-thread GSL RNGs if OpenMP is enabled
    #ifdef _OPENMP
        g_max_omp_threads_for_rng = omp_get_max_threads();
        if (g_max_omp_threads_for_rng <= 0) g_max_omp_threads_for_rng = 1; // Safety
    #else
        g_max_omp_threads_for_rng = 1;
    #endif

    g_rng_per_thread = (gsl_rng **)malloc(g_max_omp_threads_for_rng * sizeof(gsl_rng *));
    if (g_rng_per_thread == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for per-thread RNG array.\n");
        CLEAN_EXIT(1);
    }

    const gsl_rng_type *T_rng_thread = gsl_rng_default;
    
    for (int i_rng = 0; i_rng < g_max_omp_threads_for_rng; ++i_rng) {
        g_rng_per_thread[i_rng] = gsl_rng_alloc(T_rng_thread);
        if (g_rng_per_thread[i_rng] == NULL) {
            fprintf(stderr, "Error: Failed to allocate GSL RNG for thread %d.\n", i_rng);
            for (int k_rng = 0; k_rng < i_rng; ++k_rng) gsl_rng_free(g_rng_per_thread[k_rng]);
            free(g_rng_per_thread);
            CLEAN_EXIT(1);
        }
        gsl_rng_set(g_rng_per_thread[i_rng], g_sidm_seed + (unsigned long int)i_rng);
    }
    log_message("INFO", "Initialized %d per-thread GSL RNGs (for SIDM) using base SIDM seed %lu", g_max_omp_threads_for_rng, g_sidm_seed);

    if (g_use_nfw_profile) {
        log_message("INFO", "Starting IC generation using NFW-like profile pathway.");
        log_message("INFO", "Generating Initial Conditions using NFW-like profile with its specific numerics...");

        // Configure density derivative function for Eddington inversion.
        // Osipkov-Merritt model requires augmented density derivative.
        if (g_use_om_profile) {
            g_density_derivative_nfw_func = &density_derivative_om_nfw;
            log_message("INFO", "NFW: Using Osipkov-Merritt augmented density derivative");
        } else {
            g_density_derivative_nfw_func = &drhodr_profile_nfwcutoff;
            log_message("INFO", "NFW: Using standard isotropic density derivative");
        }

        // Diagnostic loop for NFW (similar to Cored's diagnostic)
        if (g_doDebug) {
            log_message("INFO", "NFW DIAGNOSTIC LOOP: Starting convergence tests for NFW profile.");
            int diag_integration_points_array[2] = {1000, 10000};
            int diag_spline_points_array[2] = {1000, 10000};

            for (int ii_ip_nfw = 0; ii_ip_nfw < 2; ii_ip_nfw++) {
                for (int ii_sp_nfw = 0; ii_sp_nfw < 2; ii_sp_nfw++) {
                    int Nintegration_diag = diag_integration_points_array[ii_ip_nfw];
                    int Nspline_diag_base = diag_spline_points_array[ii_sp_nfw];
                    int num_points_diag = Nspline_diag_base * 10;

                    log_message("DEBUG", "Diagnostic iteration %d, spline_base=%d (points=%d)",
                                Nintegration_diag, Nspline_diag_base, num_points_diag);

                    // Use the main NFW profile parameters for this diagnostic run
                    double current_diag_rc = g_nfw_profile_rc;
                    double current_diag_halo_mass = g_nfw_profile_halo_mass;
                    double current_diag_rmax_factor = g_nfw_profile_rmax_norm_factor;
                    double current_diag_falloff_C = g_nfw_profile_falloff_factor;
                    double rmax_diag = current_diag_rmax_factor * current_diag_rc;

                    // Local GSL workspace and variables for this diagnostic iteration
                    gsl_integration_workspace *w_diag = gsl_integration_workspace_alloc(Nintegration_diag);
                    if (!w_diag) {
                        log_message("ERROR", "Failed to allocate GSL workspace for NFW diagnostic");
                        continue;
                    }
                    
                    double result_diag, error_diag;
                    double normalization_diag;

                    // Declare all splines and accelerators locally for the diagnostic loop
                    gsl_spline *splinemass_diag = NULL;
                    gsl_interp_accel *enclosedmass_diag = NULL;
                    gsl_spline *splinePsi_diag = NULL;
                    gsl_interp_accel *Psiinterp_diag = NULL;
                    gsl_spline *splinerofPsi_diag = NULL;
                    gsl_interp_accel *rofPsiinterp_diag = NULL;
                    gsl_interp *fofEinterp_diag = NULL;
                    gsl_interp_accel *fofEacc_diag = NULL;

                    double *mass_diag_arr = NULL;
                    double *radius_diag_arr = NULL;
                    double *radius_for_rofPsi_diag_arr = NULL;
                    double *Psivalues_diag_arr = NULL;
                    double *nPsivalues_diag_arr = NULL;
                    double *Evalues_diag_arr = NULL;
                    double *innerintegrandvalues_diag_arr = NULL;

                    // Prepare NFW params for this diagnostic iteration's integrands
                    double nfw_params_diag[4];
                    nfw_params_diag[0] = current_diag_rc;
                    nfw_params_diag[1] = current_diag_halo_mass;
                    nfw_params_diag[2] = 1.0; // Initial nt_nfw guess
                    nfw_params_diag[3] = current_diag_falloff_C;

                    gsl_function F_nfw_diag;
                    F_nfw_diag.function = &massintegrand_profile_nfwcutoff;
                    F_nfw_diag.params = nfw_params_diag;

                    // --- 1. Normalization for NFW Diagnostic ---
                    gsl_integration_qag(&F_nfw_diag, 0.0, rmax_diag, 1e-12, 1e-12, Nintegration_diag, GSL_INTEG_GAUSS51,
                                        w_diag, &result_diag, &error_diag);
                    normalization_diag = result_diag;
                    if (normalization_diag <= 1e-30) {
                        log_message("ERROR", "NFW diagnostic normalization too small: %e", normalization_diag);
                        gsl_integration_workspace_free(w_diag);
                        continue;
                    }
                    nfw_params_diag[2] = current_diag_halo_mass / (4.0 * M_PI * normalization_diag);

                    // --- 2. M(r) spline for NFW Diagnostic ---
                    mass_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    radius_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    if (!mass_diag_arr || !radius_diag_arr) {
                        log_message("ERROR", "Failed to allocate arrays for NFW diagnostic M(r)");
                        gsl_integration_workspace_free(w_diag);
                        free(mass_diag_arr);
                        free(radius_diag_arr);
                        continue;
                    }
                    
                    // Use adaptive minimum radius for diagnostic grid too
                    nfw_potential_params nfw_diag_params = {current_diag_halo_mass, current_diag_rc, 
                                                           current_diag_falloff_C, 0.0, 0.0};
                    double rmin_diag = find_minimum_useful_radius(nfw_potential_wrapper, &nfw_diag_params, 
                                                                current_diag_rc, 1e-12);
                    
                    double log_rmin_diag = log10(rmin_diag);
                    double log_rmax_diag = log10(rmax_diag);
                    
                    for (int k = 0; k < num_points_diag; k++) {
                        double r_k;
                        if (k == num_points_diag - 1) {
                            r_k = rmax_diag;
                        } else {
                            double log_r = log_rmin_diag + (log_rmax_diag - log_rmin_diag) * (double)k / (double)(num_points_diag - 1);
                            r_k = pow(10.0, log_r);
                        }
                        radius_diag_arr[k] = r_k;
                        gsl_integration_qag(&F_nfw_diag, 0.0, r_k, 1e-12, 1e-12, Nintegration_diag, GSL_INTEG_GAUSS51,
                                            w_diag, &result_diag, &error_diag);
                        mass_diag_arr[k] = 4.0 * M_PI * result_diag;
                    }
                    enclosedmass_diag = gsl_interp_accel_alloc();
                    splinemass_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    gsl_spline_init(splinemass_diag, radius_diag_arr, mass_diag_arr, num_points_diag);


                    // Write mass profile diagnostic file
                    char diag_fname_mass[256];
                    char diag_base_mass[128];
                    snprintf(diag_base_mass, sizeof(diag_base_mass), "data/massprofile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_mass, 1, diag_fname_mass, sizeof(diag_fname_mass));
                    FILE *fp_diag_mass = fopen(diag_fname_mass, "wb");
                    if (fp_diag_mass) {
                        for (double r_write_diag = 0.0; r_write_diag < radius_diag_arr[num_points_diag - 1]; r_write_diag += rmax_diag / 900.0) {
                            if (r_write_diag >= radius_diag_arr[0]) {
                                 fprintf_bin(fp_diag_mass, "%f %f\n", r_write_diag, gsl_spline_eval(splinemass_diag, r_write_diag, enclosedmass_diag));
                            }
                        }
                        fclose(fp_diag_mass);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_mass);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_mass);
                    }

                    // --- 3. Psi(r) spline for NFW Diagnostic ---
                    Psivalues_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    nPsivalues_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    radius_for_rofPsi_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    if (!Psivalues_diag_arr || !nPsivalues_diag_arr || !radius_for_rofPsi_diag_arr) {
                        log_message("ERROR", "Failed to allocate arrays for NFW diagnostic Psi(r)");
                        goto cleanup_diag_iteration;
                    }

                    Psiintegrand_params psi_params_nfw_diag;
                    psi_params_nfw_diag.massintegrand_func = &massintegrand_profile_nfwcutoff;
                    psi_params_nfw_diag.params_for_massintegrand = nfw_params_diag;
                    gsl_function F_psi_nfw_diag;
                    F_psi_nfw_diag.function = &Psiintegrand;
                    F_psi_nfw_diag.params = &psi_params_nfw_diag;

                    // Calculate Psi values, handling numerical precision at small radii
                    int first_valid_index = 0;
                    
                    for (int k = 0; k < num_points_diag; k++) {
                        double r_k = radius_diag_arr[k];
                        double r1_psi_k = fmax(r_k, current_diag_rc / 1000000.0);
                        gsl_integration_qagiu(&F_psi_nfw_diag, r1_psi_k, 1e-12, 1e-12, Nintegration_diag,
                                              w_diag, &result_diag, &error_diag);
                        double M_at_r1_k = gsl_spline_eval(splinemass_diag, r1_psi_k, enclosedmass_diag);
                        double first_term_psi_k = (r1_psi_k > 1e-9) ? (G_CONST * M_at_r1_k / r1_psi_k) : 0.0;
                        double second_term_psi_k = G_CONST * 4.0 * M_PI * result_diag;
                        Psivalues_diag_arr[k] = first_term_psi_k + second_term_psi_k;
                        nPsivalues_diag_arr[k] = -Psivalues_diag_arr[k];
                        radius_for_rofPsi_diag_arr[k] = r_k;
                        
                        // Check for constant potential region
                        if (k > 0 && fabs(nPsivalues_diag_arr[k] - nPsivalues_diag_arr[k-1]) < 1e-10 * fabs(nPsivalues_diag_arr[k])) {
                            // Still in constant region
                        } else if (first_valid_index == 0 && k > 0) {
                            first_valid_index = k;
                        }
                    }
                    
                    // If constant region exists at small radii, adjust the arrays
                    if (first_valid_index > 0) {
                        // Shift arrays to start from first distinguishable value
                        int new_num_points = num_points_diag - first_valid_index;
                        for (int k = 0; k < new_num_points; k++) {
                            nPsivalues_diag_arr[k] = nPsivalues_diag_arr[k + first_valid_index];
                            radius_for_rofPsi_diag_arr[k] = radius_for_rofPsi_diag_arr[k + first_valid_index];
                            Psivalues_diag_arr[k] = Psivalues_diag_arr[k + first_valid_index];
                            radius_diag_arr[k] = radius_diag_arr[k + first_valid_index];
                        }
                        num_points_diag = new_num_points;
                    }
                    Psiinterp_diag = gsl_interp_accel_alloc();
                    splinePsi_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    gsl_spline_init(splinePsi_diag, radius_diag_arr, Psivalues_diag_arr, num_points_diag);
                    
                    
                    // Write potential profile diagnostic file
                    char diag_fname_psi[256];
                    char diag_base_psi[128];
                    snprintf(diag_base_psi, sizeof(diag_base_psi), "data/Psiprofile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_psi, 1, diag_fname_psi, sizeof(diag_fname_psi));
                    FILE *fp_diag_psi = fopen(diag_fname_psi, "wb");
                    if (fp_diag_psi) {
                        for (double r_write_diag = 0.0; r_write_diag < radius_diag_arr[num_points_diag - 1]; r_write_diag += rmax_diag / 900.0) {
                             if (r_write_diag >= radius_diag_arr[0]) {
                                fprintf_bin(fp_diag_psi, "%f %f\n", r_write_diag, evaluatespline(splinePsi_diag, Psiinterp_diag, r_write_diag));
                             }
                        }
                        fclose(fp_diag_psi);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_psi);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_psi);
                    }
                    
                    // --- 4. r(Psi) spline for NFW Diagnostic ---
                    struct RrPsiPair *temp_pairs_npsi_nfw_diag = (struct RrPsiPair *)malloc(num_points_diag * sizeof(struct RrPsiPair));
                    if(!temp_pairs_npsi_nfw_diag) {
                        log_message("ERROR", "Failed to allocate sorting pairs for NFW diagnostic r(Psi)");
                        goto cleanup_diag_iteration;
                    }
                    for(int k_sort = 0; k_sort < num_points_diag; ++k_sort) {
                        temp_pairs_npsi_nfw_diag[k_sort].rr = nPsivalues_diag_arr[k_sort];
                        temp_pairs_npsi_nfw_diag[k_sort].psi = radius_for_rofPsi_diag_arr[k_sort];
                    }
                    qsort(temp_pairs_npsi_nfw_diag, num_points_diag, sizeof(struct RrPsiPair), compare_pair_by_first_element);
                    for(int k_sort = 0; k_sort < num_points_diag; ++k_sort) {
                        nPsivalues_diag_arr[k_sort] = temp_pairs_npsi_nfw_diag[k_sort].rr;
                        radius_for_rofPsi_diag_arr[k_sort] = temp_pairs_npsi_nfw_diag[k_sort].psi;
                    }
                    free(temp_pairs_npsi_nfw_diag);

                    rofPsiinterp_diag = gsl_interp_accel_alloc();
                    splinerofPsi_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    if(!check_strict_monotonicity(nPsivalues_diag_arr, num_points_diag, "nPsivalues_diag_arr (NFW_DIAG)")) {
                        log_message("ERROR", "NFW diagnostic nPsivalues not monotonic");
                        goto cleanup_diag_iteration;
                    }
                    gsl_spline_init(splinerofPsi_diag, nPsivalues_diag_arr, radius_for_rofPsi_diag_arr, num_points_diag);


                    // --- 5. I(E) spline (fofEinterp_diag) for NFW Diagnostic ---
                    double Psimin_diag = Psivalues_diag_arr[num_points_diag - 1];
                    double Psimax_diag = Psivalues_diag_arr[0];
                    if (Psimax_diag <= Psimin_diag) {
                        log_message("ERROR", "NFW diagnostic potential not monotonic: Psimax=%e <= Psimin=%e", Psimax_diag, Psimin_diag);
                        goto cleanup_diag_iteration;
                    }

                    Evalues_diag_arr = (double *)malloc((num_points_diag + 1) * sizeof(double));
                    innerintegrandvalues_diag_arr = (double *)malloc((num_points_diag + 1) * sizeof(double));
                    if(!Evalues_diag_arr || !innerintegrandvalues_diag_arr) {
                        log_message("ERROR", "Failed to allocate arrays for NFW diagnostic I(E)");
                        goto cleanup_diag_iteration;
                    }

                    Evalues_diag_arr[0] = Psimin_diag; 
                    innerintegrandvalues_diag_arr[0] = 0.0;
                    gsl_function F_fE_nfw_diag; 
                    F_fE_nfw_diag.function = &fEintegrand_nfw;

                    for (int k = 1; k <= num_points_diag; k++) {
                        double calE_diag = Psimin_diag + (Psimax_diag - Psimin_diag) * ((double)k) / ((double)num_points_diag);
                        if (k > 0 && calE_diag <= Evalues_diag_arr[k-1]) {
                            calE_diag = Evalues_diag_arr[k-1] + DBL_EPSILON * fabs(Evalues_diag_arr[k-1]) + DBL_MIN;
                        }
                        Evalues_diag_arr[k] = calE_diag;

                        fE_integrand_params_NFW_t params_fE_nfw_diag = {
                            calE_diag, splinerofPsi_diag, rofPsiinterp_diag,
                            splinemass_diag, enclosedmass_diag,
                            G_CONST, current_diag_rc, nfw_params_diag[2], current_diag_falloff_C,
                            Psimin_diag, Psimax_diag, g_use_om_profile
                        };
                        F_fE_nfw_diag.params = &params_fE_nfw_diag;
                        double t_upper_diag = sqrt(fmax(0.0, calE_diag - Psimin_diag));
                        double t_lower_diag = (t_upper_diag > 1e-9) ? t_upper_diag / 1.0e4 : 0.0;
                        if (t_lower_diag >= t_upper_diag - 1e-12) { 
                            result_diag = 0.0; 
                        } else {
                            gsl_integration_qag(&F_fE_nfw_diag, t_lower_diag, t_upper_diag, 1e-8, 1e-8,
                                                Nintegration_diag, GSL_INTEG_GAUSS61, w_diag, &result_diag, &error_diag);
                        }
                        innerintegrandvalues_diag_arr[k] = result_diag;
                    }

                    fofEacc_diag = gsl_interp_accel_alloc();
                    fofEinterp_diag = gsl_interp_alloc(gsl_interp_linear, num_points_diag + 1);
                    if(!check_strict_monotonicity(Evalues_diag_arr, num_points_diag + 1, "Evalues_diag_arr (NFW_DIAG)")) {
                        log_message("ERROR", "NFW diagnostic Evalues not monotonic");
                        goto cleanup_diag_iteration;
                    }
                    
                    // Debug: print first few E values to check monotonicity
                    if (g_doDebug) {
                        log_message("DEBUG", "First few Evalues_diag_arr values:");
                        for (int k = 0; k < 5 && k <= num_points_diag; k++) {
                            log_message("DEBUG", "  Evalues_diag_arr[%d] = %.17e", k, Evalues_diag_arr[k]);
                        }
                    }
                    
                    gsl_interp_init(fofEinterp_diag, Evalues_diag_arr, innerintegrandvalues_diag_arr, num_points_diag + 1);

                    // Check and correct negative f(E) in diagnostic spline
                    int num_points_diag_plus_one = num_points_diag + 1;
                    int should_abort_diag = check_and_warn_negative_fQ(
                        &fofEinterp_diag,
                        &fofEacc_diag,
                        Evalues_diag_arr,
                        innerintegrandvalues_diag_arr,
                        &num_points_diag_plus_one,
                        g_use_om_profile ? "f(Q)" : "f(E)",
                        "NFW Diagnostic",
                        0  // verbose=0 for diagnostic (silent correction)
                    );
                    if (should_abort_diag) {
                        log_message("INFO", "Skipping NFW diagnostic output due to negative f(E)");
                        gsl_interp_free(fofEinterp_diag);
                        gsl_interp_accel_free(fofEacc_diag);
                        continue; // Skip this diagnostic iteration
                    }

                    // Write f(E) diagnostic file
                    char diag_fname_fofe[256];
                    char diag_base_fofe[128];
                    snprintf(diag_base_fofe, sizeof(diag_base_fofe), "data/f_of_E_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_fofe, 1, diag_fname_fofe, sizeof(diag_fname_fofe));
                    FILE *fp_diag_fofe = fopen(diag_fname_fofe, "wb");
                    if (fp_diag_fofe) {
                        for (int k = 0; k <= num_points_diag; k++) {
                            double E_diag = Evalues_diag_arr[k];
                            double deriv_diag = 0.0;
                            if (E_diag > Evalues_diag_arr[0] && E_diag < Evalues_diag_arr[num_points_diag]) {
                                 deriv_diag = gsl_interp_eval_deriv(fofEinterp_diag, Evalues_diag_arr, innerintegrandvalues_diag_arr, E_diag, fofEacc_diag);
                            }
                            double fE_val_diag = fabs(deriv_diag) / (sqrt(8.0) * PI * PI);
                            if (!isfinite(fE_val_diag)) fE_val_diag = 0.0;
                            fprintf_bin(fp_diag_fofe, "%f %f\n", E_diag, fE_val_diag);
                        }
                        fclose(fp_diag_fofe);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_fofe);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_fofe);
                    }

                    // Write NFW diagnostic integrand file
                    char diag_fname_integrand[256];
                    char diag_base_integrand[128];
                    snprintf(diag_base_integrand, sizeof(diag_base_integrand), "data/integrand_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_integrand, 1, diag_fname_integrand, sizeof(diag_fname_integrand));
                    FILE *fp_diag_int = fopen(diag_fname_integrand, "wb");
                    if (fp_diag_int) {
                        // Simple integrand convergence test (like Cored profile)
                        double calE_for_integrand = Psimax_diag;
                        fE_integrand_params_NFW_t params_int_nfw = {
                            calE_for_integrand, splinerofPsi_diag, rofPsiinterp_diag,
                            splinemass_diag, enclosedmass_diag,
                            G_CONST, current_diag_rc, nfw_params_diag[2], current_diag_falloff_C,
                            Psimin_diag, Psimax_diag, g_use_om_profile
                        };
                        
                        for (int k_int = 0; k_int < num_points_diag; k_int++) {
                            double t_diag_int = sqrt(fmax(0.0, calE_for_integrand - Psimin_diag)) * ((double)k_int) / ((double)num_points_diag);
                            fprintf_bin(fp_diag_int, "%f %f\n", t_diag_int, fEintegrand_nfw(t_diag_int, &params_int_nfw));
                        }
                        
                        fclose(fp_diag_int);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_integrand);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_integrand);
                    }

                    // Write NFW diagnostic density profile
                    char diag_fname_dens[256];
                    char diag_base_dens[128];
                    snprintf(diag_base_dens, sizeof(diag_base_dens), "data/density_profile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_dens, 1, diag_fname_dens, sizeof(diag_fname_dens));
                    FILE *fp_diag_dens = fopen(diag_fname_dens, "wb");
                    if (fp_diag_dens) {
                        for (int k = 0; k < num_points_diag; k++) {
                            double rr_k = radius_diag_arr[k];
                            double rs_k = rr_k / current_diag_rc;
                            double term_s_k = rs_k + 0.01;
                            if (term_s_k <= 1e-9) term_s_k = 1e-9;
                            double term_n_k = (1.0 + rs_k) * (1.0 + rs_k);
                            double term_c_base_k = rs_k / current_diag_falloff_C;
                            double term_c_k = 1.0 + pow(term_c_base_k, 10.0);
                            double rho_shape_k = (term_s_k < 1e-9 || term_n_k < 1e-9 || term_c_k < 1e-9) ? 0.0 : (1.0 / (term_s_k * term_n_k * term_c_k));
                            if (rr_k < 1e-6 && term_s_k < 1e-3 && rho_shape_k == 0.0) {
                               rho_shape_k = 1.0 / (term_s_k * term_n_k * term_c_k);
                            }
                            double rho_r_k = nfw_params_diag[2] * rho_shape_k;
                            fprintf_bin(fp_diag_dens, "%f %f\n", rr_k, rho_r_k);
                        }
                        fclose(fp_diag_dens);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_dens);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_dens);
                    }

                    // Write NFW diagnostic dPsi/dr
                    char diag_fname_dpsi[256];
                    char diag_base_dpsi[128];
                    snprintf(diag_base_dpsi, sizeof(diag_base_dpsi), "data/dpsi_dr_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_dpsi, 1, diag_fname_dpsi, sizeof(diag_fname_dpsi));
                    FILE *fp_diag_dpsi = fopen(diag_fname_dpsi, "wb");
                    if (fp_diag_dpsi) {
                        for (int k = 0; k < num_points_diag; k++) {
                            double rr_k = radius_diag_arr[k];
                            if (rr_k > 1e-9) {
                                double Menc_k = gsl_spline_eval(splinemass_diag, rr_k, enclosedmass_diag);
                                double dpsidr_k = -(G_CONST * Menc_k) / (rr_k * rr_k);
                                fprintf_bin(fp_diag_dpsi, "%f %f\n", rr_k, dpsidr_k);
                            }
                        }
                        fclose(fp_diag_dpsi);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_dpsi);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_dpsi);
                    }

                    // Write NFW diagnostic drho/dPsi
                    char diag_fname_drhodpsi[256];
                    char diag_base_drhodpsi[128];
                    snprintf(diag_base_drhodpsi, sizeof(diag_base_drhodpsi), "data/drho_dpsi_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_drhodpsi, 1, diag_fname_drhodpsi, sizeof(diag_fname_drhodpsi));
                    FILE *fp_diag_drhodpsi = fopen(diag_fname_drhodpsi, "wb");
                    if (fp_diag_drhodpsi) {
                        for (int k = 1; k < num_points_diag - 1; k++) {
                            double rr_k = radius_diag_arr[k];
                            if (rr_k <= 1e-9) continue;

                            double drhodr_val_k = drhodr_profile_nfwcutoff(rr_k, current_diag_rc, nfw_params_diag[2], current_diag_falloff_C);
                            double Menc_k = gsl_spline_eval(splinemass_diag, rr_k, enclosedmass_diag);
                            double dPsidr_k = -(G_CONST * Menc_k) / (rr_k * rr_k);

                            if (fabs(dPsidr_k) > 1e-30) {
                                double Psi_val_k = evaluatespline(splinePsi_diag, Psiinterp_diag, rr_k);
                                double drho_dPsi_val_k = drhodr_val_k / dPsidr_k;
                                fprintf_bin(fp_diag_drhodpsi, "%f %f\n", Psi_val_k, drho_dPsi_val_k);
                            }
                        }
                        fclose(fp_diag_drhodpsi);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_drhodpsi);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_drhodpsi);
                    }
                    
                    log_message("DEBUG", "Finished diagnostic iteration %d, spline_base=%d", Nintegration_diag, Nspline_diag_base);

                    // --- Cleanup for NFW Diagnostic Iteration ---
cleanup_diag_iteration:
                    gsl_integration_workspace_free(w_diag);
                    if(splinemass_diag) gsl_spline_free(splinemass_diag);
                    if(enclosedmass_diag) gsl_interp_accel_free(enclosedmass_diag);
                    if(splinePsi_diag) gsl_spline_free(splinePsi_diag);
                    if(Psiinterp_diag) gsl_interp_accel_free(Psiinterp_diag);
                    if(splinerofPsi_diag) gsl_spline_free(splinerofPsi_diag);
                    if(rofPsiinterp_diag) gsl_interp_accel_free(rofPsiinterp_diag);
                    if(fofEinterp_diag) gsl_interp_free(fofEinterp_diag);
                    if(fofEacc_diag) gsl_interp_accel_free(fofEacc_diag);
                    free(mass_diag_arr); 
                    free(radius_diag_arr); 
                    free(radius_for_rofPsi_diag_arr);
                    free(Psivalues_diag_arr); 
                    free(nPsivalues_diag_arr);
                    free(Evalues_diag_arr); 
                    free(innerintegrandvalues_diag_arr);
                } // end Nspline_diag_base loop
            } // end Nintegration_diag loop
            log_message("INFO", "NFW DIAGNOSTIC LOOP: Completed convergence tests for NFW profile.");
        } // end if(g_doDebug) for NFW diagnostic loop

        // NFW PROFILE IC GENERATION PATHWAY
        
        /**
         * @brief NFW-specific theoretical calculation for initial conditions.
         * @details Calculates mass profile, potential, and distribution function
         *          for the NFW-like profile with power-law cutoff.
         */
        
        // Re-establish main NFW calc parameters for the main calculation
        double current_profile_rc = g_nfw_profile_rc;
        double current_profile_halo_mass = g_nfw_profile_halo_mass;
        double current_profile_rmax_norm_factor = g_nfw_profile_rmax_norm_factor;
        double current_profile_falloff_C = g_nfw_profile_falloff_factor;

        num_points = 100000; // Main NFW calculation's num_points
        int num_maxv2f = 1000; // Resolution for velocity envelope calculation only
        rmax = current_profile_rmax_norm_factor * current_profile_rc; // Main NFW rmax
        
        // Local variables for NFW pathway
        double nfw_result, nfw_error;
        double nfw_calE;
        int i_nfw;
        
        // Set NFW-specific parameters from generalized profile parameters
        g_nfw_profile_rc = g_scale_radius_param;
        g_nfw_profile_halo_mass = g_halo_mass_param;
        g_nfw_profile_rmax_norm_factor = g_cutoff_factor_param;
        
        // Update current_profile_* variables with new values
        current_profile_rc = g_nfw_profile_rc;
        current_profile_halo_mass = g_nfw_profile_halo_mass;
        current_profile_rmax_norm_factor = g_nfw_profile_rmax_norm_factor;
        current_profile_falloff_C = g_nfw_profile_falloff_factor;
        
        // Set numerical parameters for NFW
        num_points = 100000;  // Conservative default for NFW
        rmax = current_profile_rmax_norm_factor * current_profile_rc;
        
        // Allocate GSL workspace
        w = gsl_integration_workspace_alloc(1000);
        if (!w) {
            fprintf(stderr, "NFW_PATH: Failed to allocate GSL workspace\n");
            CLEAN_EXIT(1);
        }
        
        // Prepare mass integrand function
        gsl_function F_nfw_calc;
        F_nfw_calc.function = &massintegrand_profile_nfwcutoff;
        
        // Prepare parameters for NFW mass integrand: [rc, halo_mass, nt_nfw, falloff_factor]
        double nfw_params[4]; // Parameters for NFW mass integrand function
        nfw_params[0] = current_profile_rc;
        nfw_params[1] = current_profile_halo_mass; // Target total halo mass
        nfw_params[2] = 1.0; // Initial guess for nt_nfw normalization scaler  
        nfw_params[3] = current_profile_falloff_C; // Falloff transition factor C
        F_nfw_calc.params = nfw_params;
        
        // Calculate normalization for NFW profile
        int status_norm = gsl_integration_qag(&F_nfw_calc, 0.0, rmax, 1e-12, 1e-12, 1000, 
                            GSL_INTEG_GAUSS51, w, &nfw_result, &nfw_error);
        normalization = nfw_result;
        
        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            log_message("DEBUG", "Initial normalization integral (int r^2 * rho_guess dr):");
            log_message("DEBUG", "  rmax_for_norm_integral = %.3e kpc (factor=%.1f * rc=%.3f)", rmax, current_profile_rmax_norm_factor, current_profile_rc);
            log_message("DEBUG", "  GSL QAG status for norm_integral: %s", gsl_strerror(status_norm));
            log_message("DEBUG", "  Raw norm_integral_result (nfw_result for norm) = %.6e", nfw_result);
            log_message("DEBUG", "  Raw norm_integral_error_est = %.6e", nfw_error);
            log_message("DEBUG", "  Final 'normalization' variable = %.6e", normalization);
        }

        if (normalization <= 1e-30) {
            fprintf(stderr, "NFW_PATH: Normalization is zero or negative (%.3e). Exiting.\n", normalization);
            CLEAN_EXIT(1);
        }

        // Update nt_nfw with proper normalization
        nfw_params[2] = current_profile_halo_mass / (4.0 * M_PI * normalization);

        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            log_message("DEBUG", "Calculated nt_nfw (density scale factor):");
            log_message("DEBUG", "  current_profile_halo_mass (target M_total) = %.3e Msun", current_profile_halo_mass);
            log_message("DEBUG", "  nt_nfw = %.3e / (4pi * %.3e) = %.6e", current_profile_halo_mass, normalization, nfw_params[2]);
            if (!isfinite(nfw_params[2]) || (fabs(nfw_params[2]) < 1e-100 && fabs(nfw_params[2]) > 0)) {
                log_message("WARNING", "nt_nfw is NaN, Inf, or extremely small/large: %.6e", nfw_params[2]);
            }
        }
        
        log_message("INFO", "NFW Profile: RC=%.3f kpc, Halo Mass=%.3e Msun, Rmax_norm_calc=%.3f kpc, nt_nfw_scaler=%.6e",
               current_profile_rc, current_profile_halo_mass, rmax, nfw_params[2]);
        
        /**
         * @brief Calculate mass profile \f$M(r)\f$ for NFW.
         */
        mass = (double *)malloc(num_points * sizeof(double));
        radius = (double *)malloc(num_points * sizeof(double));
        radius_monotonic_grid_nfw = (double *)malloc(num_points * sizeof(double));
        if (!mass || !radius || !radius_monotonic_grid_nfw) {
            fprintf(stderr, "NFW_PATH: Failed to allocate mass/radius arrays\n");
            CLEAN_EXIT(1);
        }
        
        // Find minimum useful radius adaptively
        nfw_potential_params nfw_pot_params = {current_profile_halo_mass, current_profile_rc,
                                              current_profile_falloff_C, 0.0, 0.0};
        double rmin_nfw = find_minimum_useful_radius(nfw_potential_wrapper, &nfw_pot_params,
                                                    current_profile_rc, 1e-12);

        log_message("INFO", "NFW: Using adaptive rmin = %.3e kpc (%.3e × scale radius)",
                    rmin_nfw, rmin_nfw/current_profile_rc);

        // Use logarithmic grid starting from adaptive minimum radius
        double log_rmin_nfw = log10(rmin_nfw);
        double log_rmax_nfw = log10(rmax);

        for (i_nfw = 0; i_nfw < num_points; i_nfw++) {
            double r_current;
            if (i_nfw == num_points - 1) {
                r_current = rmax; // Ensure exact endpoint
            } else {
                // Logarithmic spacing from rmin to rmax
                double log_r = log_rmin_nfw + (log_rmax_nfw - log_rmin_nfw) * (double)i_nfw / (double)(num_points - 1);
                r_current = pow(10.0, log_r);
            }
            
            gsl_integration_qag(&F_nfw_calc, 0.0, r_current, 1e-12, 1e-12,
                                1000, GSL_INTEG_GAUSS51, w, &nfw_result, &nfw_error);
            mass[i_nfw] = 4.0 * M_PI * nfw_result;
            radius[i_nfw] = r_current; // radius array later sorted for r(Psi) spline
            radius_monotonic_grid_nfw[i_nfw] = r_current; // This 'radius_monotonic_grid_nfw' stays sorted by r
        }
        
        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            log_message("DEBUG", "M(r) spline data summary (num_points=%d):", num_points);
            log_message("DEBUG", "  Target M_total for sampling = %.3e Msun", current_profile_halo_mass);
            log_message("DEBUG", "  nt_nfw used for M(r) calcs = %.3e", nfw_params[2]);
            log_message("DEBUG", "  rmax for M(r) array = %.3e kpc", rmax);
            if (num_points > 0) {
                log_message("DEBUG", "  Final Mass at rmax (radius[num_points-1]=%.3e kpc): %.3e Msun", radius[num_points-1], mass[num_points-1]);
                if (fabs(mass[num_points-1] - current_profile_halo_mass) / current_profile_halo_mass > 0.1) {
                    log_message("WARNING", "Mass at rmax (%.3e) differs significantly from target halo mass (%.3e)!", mass[num_points-1], current_profile_halo_mass);
                }
            }
            log_message("DEBUG", "End of M(r) data summary.");
        }
        
        // Create mass spline
        enclosedmass = gsl_interp_accel_alloc();
        splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
        if (!enclosedmass || !splinemass) {
            fprintf(stderr, "NFW_PATH: Failed to allocate mass spline\n");
            CLEAN_EXIT(1);
        }
        gsl_spline_init(splinemass, radius_monotonic_grid_nfw, mass, num_points);
        
        // Write generic mass profile for plotting script
        if (g_doDebug) {
            fp = fopen("data/massprofile.dat", "wb");
            if (fp) {
                double r_min_spline = radius_monotonic_grid_nfw[0];
                double r_max_spline = radius_monotonic_grid_nfw[num_points-1];
                double mass_at_rmin = mass[0];  // Mass at minimum radius
                double mass_at_rmax = mass[num_points-1];  // Mass at maximum radius
                
                for (double r_write = 0.0; r_write < r_max_spline; r_write += rmax / 900.0) {
                    double mass_at_r = evaluatespline_with_boundary(splinemass, enclosedmass, r_write,
                                                                   r_min_spline, r_max_spline,
                                                                   mass_at_rmin, mass_at_rmax);
                    fprintf(fp, "%e %e\n", r_write, mass_at_r);
                }
                fclose(fp);
            }
        }
        
        /**
         * @brief Calculate gravitational potential Psi(r) for NFW.
         */
        Psivalues = (double *)malloc(num_points * sizeof(double));
        nPsivalues = (double *)malloc(num_points * sizeof(double));
        if (!Psivalues || !nPsivalues) {
            fprintf(stderr, "NFW_PATH: Failed to allocate Psi arrays\n");
            CLEAN_EXIT(1);
        }
        
        // Prepare Psiintegrand parameters for NFW
        Psiintegrand_params psi_params_nfw;
        psi_params_nfw.massintegrand_func = &massintegrand_profile_nfwcutoff;
        psi_params_nfw.params_for_massintegrand = nfw_params;
        
        gsl_function F_for_psi_nfw;
        F_for_psi_nfw.function = &Psiintegrand;
        F_for_psi_nfw.params = &psi_params_nfw;
        
        for (i_nfw = 0; i_nfw < num_points; i_nfw++) {
            double r_current = radius[i_nfw];
            double r1_psi = fmax(r_current, current_profile_rc / 1000000.0);
            
            gsl_integration_qagiu(&F_for_psi_nfw, r1_psi, 1e-12, 1e-12,
                                  1000, w, &nfw_result, &nfw_error);
            double first_term_psi = G_CONST * gsl_spline_eval(splinemass, r1_psi, enclosedmass) / r1_psi;
            double second_term_psi = G_CONST * 4.0 * M_PI * nfw_result;
            if (r1_psi < 1e-9) first_term_psi = 0; // Avoid division by zero
            Psivalues[i_nfw] = (first_term_psi + second_term_psi);
            nPsivalues[i_nfw] = -Psivalues[i_nfw];
        }
        
        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            log_message("DEBUG", "Psi(r) and r(Psi) spline data summary (num_points=%d):", num_points);
            if (num_points > 1) {
                log_message("DEBUG", "  Psivalues[0] (Psimax candidate) = %.6e", Psivalues[0]);
                log_message("DEBUG", "  Psivalues[num_points-1] (Psimin candidate) = %.6e", Psivalues[num_points-1]);
                if (Psivalues[0] <= Psivalues[num_points-1]) {
                    log_message("WARNING", "Psi(r) may not be monotonic decreasing (Psivalues[0]=%.3e <= Psivalues[end]=%.3e)!", Psivalues[0], Psivalues[num_points-1]);
                }
            }
            log_message("DEBUG", "End of Psi(r) data summary.");
        }


        // Create Psi splines
        Psiinterp = gsl_interp_accel_alloc();
        splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
        if (!Psiinterp || !splinePsi) {
            fprintf(stderr, "NFW_PATH: Failed to allocate Psi spline\n");
            CLEAN_EXIT(1);
        }
        gsl_spline_init(splinePsi, radius_monotonic_grid_nfw, Psivalues, num_points);
        
        // Write generic Psi profile for plotting script
        if (g_doDebug) {
            fp = fopen("data/Psiprofile.dat", "wb");
            if (fp) {
                double r_min_spline = radius_monotonic_grid_nfw[0];
                double r_max_spline = radius_monotonic_grid_nfw[num_points-1];
                double psi_at_rmin = Psivalues[0];  // Psi at minimum radius
                double psi_at_rmax = Psivalues[num_points-1];  // Psi at maximum radius
                
                for (double r_write = 0.0; r_write < r_max_spline; r_write += rmax / 900.0) {
                    double psi_at_r = evaluatespline_with_boundary(splinePsi, Psiinterp, r_write,
                                                                   r_min_spline, r_max_spline,
                                                                   psi_at_rmin, psi_at_rmax);
                    fprintf(fp, "%e %e\n", r_write, psi_at_r);
                }
                fclose(fp);
            }
        }
        
        // Create r(Psi) spline accelerator
        rofPsiinterp = gsl_interp_accel_alloc();
        if (!rofPsiinterp) {
            fprintf(stderr, "NFW_PATH: Failed to allocate r(Psi) accelerator\n");
            CLEAN_EXIT(1);
        }
        
        // Use temporary copies for r(Psi) spline to preserve the original radius grid
        double *nPsivalues_for_rPsi_spline = (double *)malloc(num_points * sizeof(double));
        double *radius_values_for_rPsi_spline = (double *)malloc(num_points * sizeof(double));

        if (!nPsivalues_for_rPsi_spline || !radius_values_for_rPsi_spline) {
            fprintf(stderr, "NFW_PATH: Failed to allocate temp arrays for r(Psi) spline data.\n");
            if (nPsivalues_for_rPsi_spline) free(nPsivalues_for_rPsi_spline);
            if (radius_values_for_rPsi_spline) free(radius_values_for_rPsi_spline);
            CLEAN_EXIT(1);
        }

        // Copy nPsivalues and the corresponding radius_monotonic_grid_nfw values
        memcpy(nPsivalues_for_rPsi_spline, nPsivalues, num_points * sizeof(double));
        memcpy(radius_values_for_rPsi_spline, radius_monotonic_grid_nfw, num_points * sizeof(double));

        // Sort nPsivalues_for_rPsi_spline and apply identical swaps to radius_values_for_rPsi_spline
        for (int k_sort = 0; k_sort < num_points - 1; k_sort++) {
            for (int j_sort = k_sort + 1; j_sort < num_points; j_sort++) {
                if (nPsivalues_for_rPsi_spline[k_sort] > nPsivalues_for_rPsi_spline[j_sort]) {
                    // Swap nPsivalues_for_rPsi_spline
                    double temp_npsi = nPsivalues_for_rPsi_spline[k_sort];
                    nPsivalues_for_rPsi_spline[k_sort] = nPsivalues_for_rPsi_spline[j_sort];
                    nPsivalues_for_rPsi_spline[j_sort] = temp_npsi;

                    // Swap corresponding radius_values_for_rPsi_spline
                    double temp_rad = radius_values_for_rPsi_spline[k_sort];
                    radius_values_for_rPsi_spline[k_sort] = radius_values_for_rPsi_spline[j_sort];
                    radius_values_for_rPsi_spline[j_sort] = temp_rad;
                }
            }
        }
        
        // Check for nPsivalues_for_rPsi_spline monotonicity and remove duplicates
        int unique_count = 1;  // Start with first element
        for (int k = 1; k < num_points; k++) {
            // Keep only values that are strictly greater than the previous
            if (nPsivalues_for_rPsi_spline[k] > nPsivalues_for_rPsi_spline[unique_count-1]) {
                if (k != unique_count) {
                    nPsivalues_for_rPsi_spline[unique_count] = nPsivalues_for_rPsi_spline[k];
                    radius_values_for_rPsi_spline[unique_count] = radius_values_for_rPsi_spline[k];
                }
                unique_count++;
            }
        }
        
        if (unique_count < num_points) {
            log_message("WARNING", "Removed %d duplicate nPsi values from r(Psi) spline (keeping %d unique values)", 
                        num_points - unique_count, unique_count);
        }
        
        if (unique_count < 3) {
            fprintf(stderr, "NFW_PATH: Too few unique Psi values (%d) for spline creation\n", unique_count);
            CLEAN_EXIT(1);
        }

        // Allocate spline with the correct number of unique points
        splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, unique_count);
        if (!splinerofPsi) {
            fprintf(stderr, "NFW_PATH: Failed to allocate r(Psi) spline\n");
            CLEAN_EXIT(1);
        }

        // Initialize splinerofPsi with the unique values only
        gsl_spline_init(splinerofPsi, nPsivalues_for_rPsi_spline, radius_values_for_rPsi_spline, unique_count);

        // Free the temporary sorted copies
        free(nPsivalues_for_rPsi_spline);
        free(radius_values_for_rPsi_spline);

        /**
         * @brief Calculate \f$f(E)\f$ distribution function for NFW using Eddington's formula.
         */
        Psimin = Psivalues[num_points - 1];
        Psimax = Psivalues[0];
        
        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            log_message("DEBUG", "Potential range for I(E) calculation: Psimin=%.6e, Psimax=%.6e", Psimin, Psimax);
            log_message("DEBUG", "Continuing with I(E) calculation.");
        }
        
        if (Psimax <= Psimin) {
            fprintf(stderr, "NFW_PATH: Potential not monotonic (Psimax=%.3e <= Psimin=%.3e)\n",
                    Psimax, Psimin);
            CLEAN_EXIT(1);
        }

        innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
        Evalues = (double *)malloc((num_points + 1) * sizeof(double));
        if (!innerintegrandvalues || !Evalues) {
            fprintf(stderr, "NFW_PATH: Failed to allocate f(E) arrays\n");
            CLEAN_EXIT(1);
        }
        
        // NFW uses conservative tolerance for f(E) integral
        F_nfw_calc.function = &fEintegrand_nfw;
        
        innerintegrandvalues[0] = 0.0;
        Evalues[0] = Psimin;
        
        // GSL integration status tracking
        int status_fE_nfw_local;
        
        for (i_nfw = 1; i_nfw <= num_points; i_nfw++) {
            nfw_calE = Psimin + (Psimax - Psimin) * ((double)i_nfw) / ((double)num_points);
            
            // Enforce strict monotonicity for Evalues
            if (i_nfw > 0 && nfw_calE <= Evalues[i_nfw-1]) {
                // If current nfw_calE is not strictly greater than previous, add a tiny increment.
                // DBL_EPSILON insufficient for large Evalues; use scaled increment
                // A small fraction of the typical step, or a fixed small number relative to Evalues scale.
                double previous_E = Evalues[i_nfw-1];
                double ideal_step = (Psimax - Psimin) / (double)num_points;
                double increment = ideal_step * 1e-6; // Small fraction of an ideal step
                if (increment == 0.0) increment = DBL_MIN * fabs(previous_E) + DBL_MIN; // Absolute minimum if ideal step is zero
                if (increment == 0.0) increment = 1e-20; // Fallback if previous_E is zero

                nfw_calE = previous_E + increment;

                if (g_doDebug && (i_nfw <= 10 || i_nfw > num_points -10 || i_nfw % (num_points/50<1?1:num_points/50) == 0) ) { // Log adjustment sparsely
                     log_message("DEBUG","Adjusted Evalues[%d] from ideal %.17e to %.17e (prev E: %.17e)",
                            i_nfw, Psimin + (Psimax - Psimin) * ((double)i_nfw) / ((double)num_points),
                            nfw_calE, previous_E);
                }
            }
            
            // Create NFW-specific parameter structure
            
            fE_integrand_params_NFW_t params_for_fE_integrand_nfw = {
                nfw_calE,               // E_current_shell
                splinerofPsi,           // spline_r_of_Psi (this is r_of_nPsi from nPsivalues)
                rofPsiinterp,           // accel_r_of_Psi
                splinemass,             // spline_M_of_r (this is M(r) for NFW, from corrected NFMP.1)
                enclosedmass,           // accel_M_of_r
                G_CONST,                // const_G_universal
                current_profile_rc,     // profile_rc_const
                nfw_params[2],          // profile_nt_norm_const (this is the scaled nt_nfw)
                nfw_params[3],          // profile_falloff_C_const (falloff factor C)
                Psimin,                 // Psimin_global
                Psimax,                 // Psimax_global
                g_use_om_profile        // use_om flag
            };
            F_nfw_calc.params = &params_for_fE_integrand_nfw;
            
            double E_current_shell = nfw_calE; // E for which I(E) is being computed

            // Integration variable: t_prime = sqrt(E_shell - Psi_true)
            // As Psi_true goes from Psimin_global to E_shell, t_prime goes from sqrt(E_shell - Psimin_global) down to 0.
            // So, integrate t_prime from 0 to sqrt(E_shell - Psimin_global).

            double t_integration_upper_bound = sqrt(fmax(0.0, E_current_shell - Psimin));
            double t_integration_lower_bound;

            if (t_integration_upper_bound < 1e-9) { // If E_current_shell is very close to Psimin (or below)
                t_integration_lower_bound = 0.0;
                t_integration_upper_bound = 0.0;
            } else {
                // Set a very small, but strictly positive, lower bound relative to the upper bound,
                // or an absolute small number if t_upper_bound is itself very small.
                // Avoid GSL singularity at t=0 where 1/t or 1/sqrt(t) undefined
                // The term 1/sqrt(E-Psi) in d(rho)/d(Psi) / sqrt(E-Psi) becomes 1/t when Psi = E-t^2.
                // The fEintegrand_nfw integrand is 2 * d(rho)/d(Psi), which does not have explicit 1/t term
                // Using t_upper_bound / 1.0e4 scaling for numerical consistency.
                t_integration_lower_bound = t_integration_upper_bound / 1.0e4;
                // Extremely small t_integration_lower_bound (< DBL_MIN) treated as zero by GSL
                // Ensure minimum representable positive value when t_upper_bound is positive
                if (t_integration_lower_bound == 0.0 && t_integration_upper_bound > 0.0) {
                    t_integration_lower_bound = DBL_EPSILON * t_integration_upper_bound; // Use DBL_EPSILON for very small t_upper
                    if (t_integration_lower_bound == 0.0) t_integration_lower_bound = 1e-20; // Absolute floor
                }
            }

            // Ensure lower bound is strictly less than upper bound for GSL
            if (t_integration_lower_bound >= t_integration_upper_bound - 1e-12) { // Adjusted epsilon
                t_integration_upper_bound = 0.0; // Force zero integration range
                t_integration_lower_bound = 0.0;
            }
            
            if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL && (i_nfw <= 5 || i_nfw > num_points - 5 || i_nfw % (num_points/10 < 1 ? 1 : num_points/10) == 0) ) {
                log_message("DEBUG", "I(E) integral setup: E_shell=%.3e, Psimin=%.3e, integrating fEintegrand_nfw(t) from t_low=%.3e to t_high=%.3e",
                       E_current_shell, Psimin, t_integration_lower_bound, t_integration_upper_bound);
            }

            if (t_integration_upper_bound <= t_integration_lower_bound + 1e-10) { // If range is zero or too small
                nfw_result = 0.0;
                status_fE_nfw_local = GSL_SUCCESS;
                 if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL && (i_nfw <= 5 || i_nfw > num_points - 5 || i_nfw % (num_points/10 < 1 ? 1 : num_points/10) == 0) ) {
                    log_message("DEBUG", "NFEFE_INTEGRAL_SETUP: Skipping t-integration, range invalid/tiny (t_high=%.3e, t_low=%.3e)", t_integration_upper_bound, t_integration_lower_bound);
                 }
            } else {
                status_fE_nfw_local = gsl_integration_qag(&F_nfw_calc, t_integration_lower_bound, t_integration_upper_bound,
                                    1e-8, 1e-8, 1000, GSL_INTEG_GAUSS61, // Using conservative GSL tolerances
                                    w, &nfw_result, &nfw_error);

                if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL && (i_nfw <= 5 || i_nfw > num_points - 5 || i_nfw % (num_points/10 < 1 ? 1 : num_points/10) == 0) ) {
                    log_message("DEBUG", "NFEFE_INTEGRAL_RESULT: I(E=%.3e) = %.6e, error=%.3e, status=%s",
                           E_current_shell, nfw_result, nfw_error,
                           (status_fE_nfw_local == GSL_SUCCESS) ? "SUCCESS" : "ERROR");
                }
            }
            innerintegrandvalues[i_nfw] = nfw_result;
            Evalues[i_nfw] = nfw_calE;
        }
        
        
        // Create f(E) interpolation using cspline interpolation for NFW (smoother dI/dE)
        g_main_fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
        g_main_fofEacc = gsl_interp_accel_alloc();
        if (!g_main_fofEinterp || !g_main_fofEacc) {
            fprintf(stderr, "NFW_PATH: Failed to allocate f(E) interpolation\n");
            CLEAN_EXIT(1);
        }
        // Validate Evalues array monotonicity before GSL spline initialization
        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            int monotonicity_violations = 0;
            log_message("DEBUG", "Checking Evalues for strict monotonicity (%d points) before I(E) spline init.", num_points + 1);
            // Evalues has num_points + 1 elements, indexed 0 to num_points.
            for (int chk_e = 0; chk_e < num_points; ++chk_e) { // Loop up to num_points-1 to check Evalues[chk_e+1] vs Evalues[chk_e]
                if (!(Evalues[chk_e+1] > Evalues[chk_e])) {
                    if (monotonicity_violations < 20) { // Print first few violations
                        fprintf(stderr, "  MONOTONICITY_VIOLATION_PRE_SPLINE: Evalues[%d]=%.17e, Evalues[%d]=%.17e (Diff: %.3e)\n",
                               chk_e, Evalues[chk_e], chk_e+1, Evalues[chk_e+1], Evalues[chk_e+1] - Evalues[chk_e]);
                    }
                    monotonicity_violations++;
                }
            }
            if (monotonicity_violations > 0) {
                fprintf(stderr, "  NFW_CRITICAL_SPLINE_INIT: Total Evalues monotonicity violations: %d. GSL interp_init requires strict monotonicity. Exiting.\n", monotonicity_violations);
                CLEAN_EXIT(1);
            } else if (g_doDebug) { // Only log success if in debug mode
                log_message("DEBUG", "Evalues array confirmed strictly monotonic before I(E) spline init.");
            }
        }

        gsl_interp_init(g_main_fofEinterp, Evalues, innerintegrandvalues, num_points + 1);

        // Check for negative f(E) or f(Q) values before particle sampling
        log_message("INFO", "NFW: Checking for negative distribution function values...");
        int num_points_plus_one_nfw = num_points + 1;
        int should_abort_nfw = check_and_warn_negative_fQ(
            &g_main_fofEinterp,
            &g_main_fofEacc,
            Evalues,
            innerintegrandvalues,
            &num_points_plus_one_nfw,
            g_use_om_profile ? "f(Q)" : "f(E)",
            "NFW",
            1  // verbose=1 for main pathway
        );
        if (should_abort_nfw) {
            log_message("INFO", "User aborted NFW IC generation due to negative distribution function");
            CLEAN_EXIT(0);
        }

        gsl_integration_workspace_free(w);
        w = NULL;
        
        log_message("INFO", "NFW theoretical calculation for IC splines complete.");
        
        /**
         * @brief NFW Sample Generator - Generate particle positions and velocities.
         * @details Uses rejection sampling with the NFW density profile and \f$f(E)\f$ distribution.
         */
        if (doReadInit) {
            // Read initial conditions from file
            printf("Reading initial conditions from %s...\n", readInitFilename);
            read_initial_conditions(particles, npts_initial, readInitFilename);
        } else if (!g_doRestart) {
            if (tidal_fraction > 0.0) {
                log_message("INFO", "NFW IC Gen: Initial particle count before stripping: %d", npts_initial);
            }
            log_message("INFO", "NFW IC Gen: Generating %d initial particle positions and velocities...", npts_initial);
            
            // Print overall Psimin, Psimax for NFW path once
            if (npts_initial > 0) { // Avoid printing if no particles
                if (Psimax <= Psimin) {
                }
            }
            
            /**
             * @brief Allocate memory for the particle data array.
             * @details 2D array particles[5][npts_initial] where:
             *          [0] = radius, [1] = velocity, [2] = angular momentum,
             *          [3] = particle ID, [4] = orientation (mu)
             */
            particles = (double **)malloc(7 * sizeof(double *));  // Rows 5,6 for cos(phi), sin(phi)
            if (particles == NULL) {
                fprintf(stderr, "NFW_PATH: Memory allocation failed for particle array\n");
                CLEAN_EXIT(1);
            }
            for (i = 0; i < 7; i++) {  // 7 rows for extended particle state
                particles[i] = (double *)malloc(npts_initial * sizeof(double));
                if (particles[i] == NULL) {
                    fprintf(stderr, "NFW_PATH: Memory allocation failed for particles[%d]\n", i);
                    CLEAN_EXIT(1);
                }
            }
            
            /**
             * @brief Calculate maximum velocity squared at each radius for rejection sampling.
             */
            double *maxv2f_nfw = (double *)malloc(num_maxv2f * sizeof(double));
            double *radius_maxv2f_nfw = (double *)malloc(num_maxv2f * sizeof(double));
            if (!maxv2f_nfw || !radius_maxv2f_nfw) {
                fprintf(stderr, "NFW_PATH: Failed to allocate maxv2f arrays\n");
                CLEAN_EXIT(1);
            }
            
            double nfw_vel, nfw_ratio, nfw_Psir, nfw_mu, nfw_maxv, nfw_maxvalue;
            
            // Create spline for r(M) - radius as function of enclosed mass
            gsl_interp_accel *rofMaccel_nfw = gsl_interp_accel_alloc();
            gsl_spline *splinerofM_nfw = gsl_spline_alloc(gsl_interp_cspline, num_points);
            if (!rofMaccel_nfw || !splinerofM_nfw) {
                fprintf(stderr, "NFW_PATH: Failed to allocate r(M) spline\n");
                CLEAN_EXIT(1);
            }
            // Validate mass array monotonicity before GSL spline initialization
        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            int monotonicity_violations_mass_spline = 0;
            log_message("DEBUG", "Checking mass array for strict monotonicity (size: %d)", num_points);
            // 'mass' array has num_points elements. Loop up to num_points-2 to check mass[chk+1] vs mass[chk].
            if (num_points >= 2) { // Need at least 2 points to check monotonicity
                for (int chk_m = 0; chk_m < num_points - 1; ++chk_m) { 
                    if (!(mass[chk_m+1] > mass[chk_m])) {
                        if (monotonicity_violations_mass_spline < 20) { 
                            fprintf(stderr, "  Mass array monotonicity violation: mass[%d]=%.17e >= mass[%d]=%.17e (Diff: %.3e)\n",
                                   chk_m, mass[chk_m], chk_m+1, mass[chk_m+1], mass[chk_m+1] - mass[chk_m]);
                        }
                        monotonicity_violations_mass_spline++;
                    }
                }
            }
            if (monotonicity_violations_mass_spline > 0) {
                fprintf(stderr, "  NFW_CRITICAL_MONO_CHECK_ROFM: Total 'mass' array monotonicity violations: %d. gsl_spline_init for splinerofM_nfw will fail.\n", monotonicity_violations_mass_spline);
                 if (monotonicity_violations_mass_spline > 20) fprintf(stderr, "  (Further violations suppressed)\n");
            } else {
                log_message("DEBUG", "Mass array confirmed strictly monotonic.");
            }
        }

        gsl_spline_init(splinerofM_nfw, mass, radius, num_points);
            
            // Calculate max v^2 * f(E) at each radius
            radius_maxv2f_nfw[0] = 0.0;
            for (int i_r_nfw = 1; i_r_nfw < num_maxv2f; i_r_nfw++) {
                nfw_maxvalue = 0.0;
                double r_maxv2f = (double)i_r_nfw * rmax / (num_maxv2f - 1);
                radius_maxv2f_nfw[i_r_nfw] = r_maxv2f;
                nfw_Psir = evaluatespline(splinePsi, Psiinterp, r_maxv2f);
                nfw_maxv = sqrt(2.0 * (nfw_Psir - Psimin));
                
                // Find maximum of v^2 * dI/dE over velocity range
                for (int j_v_nfw = 1; j_v_nfw < num_maxv2f - 2; j_v_nfw++) {
                    nfw_vel = nfw_maxv * ((double)j_v_nfw) / ((double)num_maxv2f);
                    double E_test_nfw = nfw_Psir - 0.5 * nfw_vel * nfw_vel;
                    double currentvalue_nfw = 0.0;
                    
                    if (E_test_nfw >= Psimin && E_test_nfw <= Psimax) {
                        currentvalue_nfw = nfw_vel * nfw_vel * 
                            fabs(gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, 
                                                       innerintegrandvalues, E_test_nfw, g_main_fofEacc));
                    }
                    if (isfinite(currentvalue_nfw) && currentvalue_nfw > nfw_maxvalue) {
                        nfw_maxvalue = currentvalue_nfw;
                    }
                }
                maxv2f_nfw[i_r_nfw] = nfw_maxvalue;
            }
            
            // Extrapolate for r=0
            if (num_maxv2f >= 3) {
                maxv2f_nfw[0] = 2.0 * maxv2f_nfw[1] - maxv2f_nfw[2];
                if (maxv2f_nfw[0] < 0) maxv2f_nfw[0] = 0;
            } else {
                maxv2f_nfw[0] = maxv2f_nfw[1];
            }
            
            // Create spline for max v^2 * f(E)
            gsl_interp_accel *maxv2faccel_nfw = gsl_interp_accel_alloc();
            gsl_spline *splinemaxv2f_nfw = gsl_spline_alloc(gsl_interp_cspline, num_maxv2f);
            if (!maxv2faccel_nfw || !splinemaxv2f_nfw) {
                fprintf(stderr, "NFW_PATH: Failed to allocate maxv2f spline\n");
                CLEAN_EXIT(1);
            }
            // Validate radius array monotonicity before GSL spline initialization
        if (g_doDebug && getenv("NSPHERE_VERBOSE_IC") != NULL) {
            int monotonicity_violations_rad_spline = 0;
            log_message("DEBUG", "Checking radius array for strict monotonicity (size: %d)", num_points);
            // 'radius' array has num_points elements. Loop up to num_points-2.
            if (num_points >= 2) {
                for (int chk_r = 0; chk_r < num_points - 1; ++chk_r) { 
                    if (!(radius[chk_r+1] > radius[chk_r])) {
                        if (monotonicity_violations_rad_spline < 20) {
                            fprintf(stderr, "  Radius array monotonicity violation: radius[%d]=%.17e >= radius[%d]=%.17e (Diff: %.3e)\n",
                                   chk_r, radius[chk_r], chk_r+1, radius[chk_r+1], radius[chk_r+1] - radius[chk_r]);
                        }
                        monotonicity_violations_rad_spline++;
                    }
                }
            }
            if (monotonicity_violations_rad_spline > 0) {
                fprintf(stderr, "  NFW_CRITICAL_MONO_CHECK_MAXV2F: Total 'radius' array monotonicity violations: %d. gsl_spline_init for splinemaxv2f_nfw will fail.\n", monotonicity_violations_rad_spline);
                if (monotonicity_violations_rad_spline > 20) fprintf(stderr, "  (Further violations suppressed)\n");
            } else {
                log_message("DEBUG", "Radius array confirmed strictly monotonic.");
            }
        }

        gsl_spline_init(splinemaxv2f_nfw, radius_maxv2f_nfw, maxv2f_nfw, num_maxv2f);
            
            /**
             * @brief Generate particles using rejection sampling or read from restart file.
             * In --sim-restart mode: particle loading deferred until after truncation.
             * In --sim-extend mode: particles loaded from complete source file.
             */
            if (g_restart_mode_active && strlen(restart_source_file) > 0 && !g_doSimRestart) {
                // Load last snapshot from file
                FILE *check_file = fopen(restart_source_file, "rb");
                if (!check_file) {
                    fprintf(stderr, "ERROR: Cannot open restart file %s\n", restart_source_file);
                    CLEAN_EXIT(1);
                }
                fseek(check_file, 0, SEEK_END);
                long file_size = ftell(check_file);
                fclose(check_file);

                int total_snapshots = file_size / (npts * 16);
                int snapshot_index = total_snapshots - 1;  // Last snapshot (0-based index)

                printf("Restart file has %d snapshots, loading last snapshot (index %d)\n",
                       total_snapshots, snapshot_index);

                // Allocate inverse_map before loading (needed for particle ID tracking)
                inverse_map = (int *)malloc(npts * sizeof(int));
                if (!inverse_map) {
                    fprintf(stderr, "ERROR: Failed to allocate inverse_map before NFW restart load\n");
                    CLEAN_EXIT(1);
                }

                load_particles_from_restart(
                    restart_source_file,
                    snapshot_index,
                    particles,
                    npts,
                    snapshot_block_size,
                    inverse_map
                );
            } else {
                // Normal particle generation
                for (int k_nfw = 0; k_nfw < npts_initial; k_nfw++) {
                if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) { // Log for first few & periodically
                    fflush(stdout);
                }
                
                // Declare sampled_scalar_speed at particle loop scope for OM model
                double sampled_scalar_speed = 0.0;
                
                // Sample radius from mass distribution
                double mass_frac_sample_nfw = gsl_rng_uniform(g_rng) * 0.999999;
                double mass_sample_nfw = mass_frac_sample_nfw * current_profile_halo_mass;
                particles[0][k_nfw] = evaluatespline(splinerofM_nfw, rofMaccel_nfw, mass_sample_nfw);
                
                if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                }
                
                nfw_maxvalue = evaluatespline(splinemaxv2f_nfw, maxv2faccel_nfw, particles[0][k_nfw]);
                nfw_Psir = evaluatespline(splinePsi, Psiinterp, particles[0][k_nfw]);
                
                // Check for problematic values
                if (!isfinite(nfw_Psir)) {
                    if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                    }
                    particles[1][k_nfw] = 0.0; // Assign zero velocity
                    sampled_scalar_speed = 0.0; // Set for OM model consistency
                    nfw_mu = (2.0 * gsl_rng_uniform(g_rng) - 1.0);
                    particles[2][k_nfw] = 0.0; // L = 0 since v = 0
                    particles[4][k_nfw] = nfw_mu;
                    particles[3][k_nfw] = (double)k_nfw;
                    continue;
                }
                
                if (nfw_Psir <= Psimin + 1e-9 * fabs(Psimin)) { // Check if Psir is too close to Psimin
                    if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                    }
                    particles[1][k_nfw] = 0.0; // No kinetic energy possible
                    sampled_scalar_speed = 0.0; // Set for OM model consistency
                } else {
                    // Sample velocity using rejection method
                    nfw_maxv = sqrt(fmax(0.0, 2.0 * (nfw_Psir - Psimin)));
                    if (!isfinite(nfw_maxv) || nfw_maxv < 1e-9) {
                        particles[1][k_nfw] = 0.0;
                        sampled_scalar_speed = 0.0; // Set for OM model consistency
                    } else {
                        
                        if (!isfinite(nfw_maxvalue) || nfw_maxvalue <= 1e-30) { // If envelope is effectively zero
                            particles[1][k_nfw] = 0.0;
                            sampled_scalar_speed = 0.0; // Set for OM model consistency
                        } else {
                            // Velocity Rejection Sampling Loop
                            int vflag_nfw = 0;
                            int v_trials_nfw = 0;
                            
                            while (vflag_nfw == 0 && v_trials_nfw < 20000) {
                                v_trials_nfw++;
                                double trial_speed = gsl_rng_uniform(g_rng) * nfw_maxv;
                                double E_test_nfw = nfw_Psir - 0.5 * trial_speed * trial_speed;
                                
                                double deriv_val_dIdE = 0.0;
                                if (E_test_nfw >= Psimin && E_test_nfw <= Psimax) {
                                    deriv_val_dIdE = gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, 
                                                                           innerintegrandvalues, E_test_nfw, g_main_fofEacc);
                                }
                                
                                double target_func_val_nfw = trial_speed * trial_speed * fabs(deriv_val_dIdE);
                                if (!isfinite(target_func_val_nfw) || target_func_val_nfw < 0) target_func_val_nfw = 0.0;
                                
                                nfw_ratio = target_func_val_nfw / nfw_maxvalue;
                                if (nfw_ratio < 0) nfw_ratio = 0;
                                if (nfw_ratio > 1.001) nfw_ratio = 1.0;
                                
                                if (gsl_rng_uniform(g_rng) < nfw_ratio) {
                                    sampled_scalar_speed = trial_speed;
                                    vflag_nfw = 1;
                                }
                            }
                            if (!vflag_nfw) {
                                sampled_scalar_speed = 0.0; // Failed to find velocity, assign 0
                            }
                        }
                    }
                }
    
                if (g_use_om_profile) {
                    // --- Osipkov-Merritt Anisotropic Sampling ---
                    double w = sampled_scalar_speed; // This is the pseudo-speed
                    double mu_w = 2.0 * gsl_rng_uniform(g_rng) - 1.0; // Isotropic in w-space
    
                    // Map back to physical velocities
                    double alpha_r = 1.0 + (particles[0][k_nfw] * particles[0][k_nfw]) / (g_om_anisotropy_radius * g_om_anisotropy_radius);
                    double v_r = mu_w * w;
                    double v_t = sqrt(fmax(0.0, 1.0 - mu_w * mu_w)) * w / sqrt(alpha_r);
                    double v_mag = sqrt(v_r * v_r + v_t * v_t);
                    
                    particles[1][k_nfw] = v_mag; // Store velocity magnitude |v|
                    particles[2][k_nfw] = particles[0][k_nfw] * v_t; // Store angular momentum L
                    particles[4][k_nfw] = (v_mag > 1e-9) ? (v_r / v_mag) : 0.0; // Store physical mu
                } else {
                    // --- Isotropic Sampling (original logic) ---
                    double v = sampled_scalar_speed;
                    nfw_mu = 2.0 * gsl_rng_uniform(g_rng) - 1.0;
                    particles[1][k_nfw] = v; // Store velocity magnitude
                    particles[2][k_nfw] = particles[0][k_nfw] * v * sqrt(fmax(0.0, 1.0 - nfw_mu * nfw_mu));
                    particles[4][k_nfw] = nfw_mu; // Store mu
                }
    
                particles[3][k_nfw] = (double)k_nfw; // Particle ID
            }
            } // End of else block (normal generation vs restart)
            
            // Clean up NFW sample generator allocations
            gsl_spline_free(splinerofM_nfw);
            gsl_interp_accel_free(rofMaccel_nfw);
            gsl_spline_free(splinemaxv2f_nfw);
            gsl_interp_accel_free(maxv2faccel_nfw);
            free(maxv2f_nfw);
            free(radius_maxv2f_nfw);
            
            log_message("INFO", "NFW IC Gen: Successfully generated %d particles.", npts_initial);
        } // End NFW sample generator


    } else if (g_use_hernquist_aniso_profile) {
        // Always build splines first (needed for simulation regardless of IC source)
        generate_ics_hernquist_anisotropic(NULL, 0, g_rng, g_halo_mass_param, g_scale_radius_param,
                                           &splinemass, &enclosedmass,
                                           &splinePsi, &Psiinterp,
                                           &splinerofPsi, &rofPsiinterp,
                                           &g_main_fofEinterp, &g_main_fofEacc,
                                           &radius, &radius_unsorted, &mass, &Psivalues, &num_points,
                                           1);  // splines_only = 1
        
        // Set rmax for consistency with other profiles
        rmax = g_cutoff_factor_param * g_scale_radius_param;

        // Now handle IC reading or generation
        if (g_restart_mode_active && strlen(restart_source_file) > 0 && !g_doSimRestart) {
            // Load last snapshot from file
            FILE *check_file = fopen(restart_source_file, "rb");
            if (!check_file) {
                fprintf(stderr, "ERROR: Cannot open restart file %s\n", restart_source_file);
                CLEAN_EXIT(1);
            }
            fseek(check_file, 0, SEEK_END);
            long file_size = ftell(check_file);
            fclose(check_file);

            int total_snapshots = file_size / (npts * 16);
            int snapshot_index = total_snapshots - 1;  // Last snapshot (0-based index)

            printf("Restart file has %d snapshots, loading last snapshot (index %d)\n",
                   total_snapshots, snapshot_index);

            // Allocate inverse_map before loading (needed for particle ID tracking)
            inverse_map = (int *)malloc(npts * sizeof(int));
            if (!inverse_map) {
                fprintf(stderr, "ERROR: Failed to allocate inverse_map before Hernquist aniso restart load\n");
                CLEAN_EXIT(1);
            }

            load_particles_from_restart(
                restart_source_file,
                snapshot_index,
                particles,
                npts,
                snapshot_block_size,
                inverse_map
            );
        } else if (doReadInit) {
            printf("Reading initial conditions from %s...\n", readInitFilename);
            read_initial_conditions(particles, npts_initial, readInitFilename);
        } else if (!g_doRestart) {
            // Generate particles using newly constructed splines
            generate_ics_hernquist_anisotropic(particles, npts_initial, g_rng, g_halo_mass_param, g_scale_radius_param,
                                               &splinemass, &enclosedmass,
                                               &splinePsi, &Psiinterp,
                                               &splinerofPsi, &rofPsiinterp,
                                               &g_main_fofEinterp, &g_main_fofEacc,
                                               &radius, &radius_unsorted, &mass, &Psivalues, &num_points,
                                               0);  // splines_only = 0
        }

    } else if (g_use_hernquist_numerical) {
        // NUMERICAL HERNQUIST - Always build splines first (needed for simulation)
        log_message("INFO", "Starting numerical Hernquist profile setup.");
        
        // Set up Hernquist parameters
        double a_scale = g_scale_radius_param;  // Hernquist scale radius
        double M_total = g_halo_mass_param;     // Total mass
        double cutoff_radius = g_cutoff_factor_param * a_scale;
        
        // Log selected density derivative function
        if (g_use_numerical_isotropic) {
            log_message("INFO", "Hernquist: Using numerical pathway with true f(E) (isotropic)");
        } else if (g_use_om_profile) {
            log_message("INFO", "Hernquist: Using Osipkov-Merritt augmented density derivative f(Q)");
        } else {
            log_message("INFO", "Hernquist: Using standard isotropic density derivative f(E)");
        }
        
        // Prepare parameters for Hernquist mass integrand: [a_scale, M_total, normalization]
        double hern_params[3];
        hern_params[0] = a_scale;
        hern_params[1] = M_total;
        hern_params[2] = 1.0; // Initial normalization guess
        
        // Calculate mass normalization - use existing w variable from main scope
        w = gsl_integration_workspace_alloc(1000);
        if (!w) {
            log_message("ERROR", "Hernquist: Failed to allocate GSL workspace");
            CLEAN_EXIT(1);
        }
        gsl_function F_hern;
        F_hern.function = &massintegrand_hernquist;
        F_hern.params = hern_params;
        
        double result, error;
        gsl_integration_qag(&F_hern, 0.0, cutoff_radius, 1e-12, 1e-12, 1000,
                           GSL_INTEG_GAUSS51, w, &result, &error);
        double hern_normalization = result;
        
        // Update normalization factor to get correct total mass
        hern_params[2] = M_total / (4.0 * M_PI * hern_normalization);
        
        log_message("INFO", "Hernquist: Mass normalization = %e, scaling factor = %e", 
                    hern_normalization, hern_params[2]);
        
        // Build mass profile M(r)
        num_points = 100000;  // Use same as NFW for consistency
        mass = (double *)malloc(num_points * sizeof(double));
        radius = (double *)malloc(num_points * sizeof(double));
        if (!mass || !radius) {
            log_message("ERROR", "Hernquist: Failed to allocate mass/radius arrays");
            CLEAN_EXIT(1);
        }
        
        for (int i = 0; i < num_points; i++) {
            radius[i] = cutoff_radius * ((double)i) / ((double)(num_points - 1));
            if (i == 0) {
                mass[i] = 0.0;
            } else {
                gsl_integration_qag(&F_hern, 0.0, radius[i], 1e-12, 1e-12, 1000,
                                   GSL_INTEG_GAUSS51, w, &result, &error);
                mass[i] = 4.0 * M_PI * result;
            }
        }
        
        // Create mass spline
        splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
        gsl_spline_init(splinemass, radius, mass, num_points);
        enclosedmass = gsl_interp_accel_alloc();
        
        // Build potential profile Psi(r)
        Psivalues = (double *)malloc(num_points * sizeof(double));
        
        // For Hernquist, the potential has an analytical form: Psi(r) = -GM/(r+a)
        // But for consistency with the numerical approach, we'll compute it numerically
        
        // Build potential profile Psi(r)
        // For numerical consistency with NFW approach, compute potential numerically
        gsl_function F_psi_hern;
        Psiintegrand_params psi_params_hern;
        psi_params_hern.massintegrand_func = &massintegrand_hernquist;
        psi_params_hern.params_for_massintegrand = hern_params;
        F_psi_hern.function = &Psiintegrand;
        F_psi_hern.params = &psi_params_hern;
        
        // Use adaptive minimum radius similar to NFW
        double rmin_hern = a_scale / 1000000.0;  // Start with small fraction of scale radius
        double log_rmin = log10(rmin_hern);
        double log_rmax = log10(cutoff_radius);
        
        // Rebuild radius array with logarithmic spacing for better resolution
        free(radius);
        radius = (double *)malloc(num_points * sizeof(double));
        
        for (int k = 0; k < num_points; k++) {
            if (k == 0) {
                radius[k] = rmin_hern;
            } else if (k == num_points - 1) {
                radius[k] = cutoff_radius;
            } else {
                double log_r = log_rmin + (log_rmax - log_rmin) * ((double)k) / ((double)(num_points - 1));
                radius[k] = pow(10.0, log_r);
            }
        }
        
        // Rebuild mass array with new radius grid
        for (int k = 0; k < num_points; k++) {
            if (radius[k] <= 1e-9) {
                mass[k] = 0.0;
            } else {
                gsl_integration_qag(&F_hern, 0.0, radius[k], 1e-12, 1e-12, 1000,
                                   GSL_INTEG_GAUSS51, w, &result, &error);
                mass[k] = 4.0 * M_PI * result;
            }
        }
        
        // Recreate mass spline with new arrays
        gsl_spline_free(splinemass);
        gsl_interp_accel_free(enclosedmass);
        splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
        gsl_spline_init(splinemass, radius, mass, num_points);
        enclosedmass = gsl_interp_accel_alloc();
        
        // Calculate potential at each radius
        for (int k = 0; k < num_points; k++) {
            double r_k = radius[k];
            double r1_psi = fmax(r_k, a_scale / 1000000.0);
            
            // Integrate from r to infinity
            gsl_integration_qagiu(&F_psi_hern, r1_psi, 1e-12, 1e-12, 1000,
                                 w, &result, &error);
            
            double M_at_r = gsl_spline_eval(splinemass, r1_psi, enclosedmass);
            double first_term = (r1_psi > 1e-9) ? (G_CONST * M_at_r / r1_psi) : 0.0;
            double second_term = G_CONST * 4.0 * M_PI * result;
            Psivalues[k] = first_term + second_term;
        }
        
        // Create potential spline
        splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
        gsl_spline_init(splinePsi, radius, Psivalues, num_points);
        Psiinterp = gsl_interp_accel_alloc();
        
        // Create r(Psi) spline for inverse lookup
        double *nPsivalues = (double *)malloc(num_points * sizeof(double));
        double *radius_for_rofPsi = (double *)malloc(num_points * sizeof(double));
        
        for (int k = 0; k < num_points; k++) {
            nPsivalues[k] = -Psivalues[k];  // Use negative Psi for monotonicity
            radius_for_rofPsi[k] = radius[k];
        }
        
        // Sort by nPsi values to ensure monotonicity
        for (int k = 0; k < num_points - 1; k++) {
            for (int j = k + 1; j < num_points; j++) {
                if (nPsivalues[k] > nPsivalues[j]) {
                    double temp = nPsivalues[k];
                    nPsivalues[k] = nPsivalues[j];
                    nPsivalues[j] = temp;
                    
                    temp = radius_for_rofPsi[k];
                    radius_for_rofPsi[k] = radius_for_rofPsi[j];
                    radius_for_rofPsi[j] = temp;
                }
            }
        }
        
        // Remove duplicates
        int unique_count = 1;
        for (int k = 1; k < num_points; k++) {
            if (nPsivalues[k] > nPsivalues[unique_count-1]) {
                if (k != unique_count) {
                    nPsivalues[unique_count] = nPsivalues[k];
                    radius_for_rofPsi[unique_count] = radius_for_rofPsi[k];
                }
                unique_count++;
            }
        }
        
        if (unique_count < 3) {
            log_message("ERROR", "Hernquist: Too few unique Psi values for spline");
            free(nPsivalues); free(radius_for_rofPsi);
            CLEAN_EXIT(1);
        }
        
        splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, unique_count);
        gsl_spline_init(splinerofPsi, nPsivalues, radius_for_rofPsi, unique_count);
        rofPsiinterp = gsl_interp_accel_alloc();
        
        free(nPsivalues);
        free(radius_for_rofPsi);

        // Calculate f(E) distribution function using Eddington formula
        Psimin = Psivalues[num_points - 1];
        Psimax = Psivalues[0];
        
        log_message("INFO", "Hernquist: Potential range Psimin=%.6e, Psimax=%.6e", Psimin, Psimax);
        
        if (Psimax <= Psimin) {
            log_message("ERROR", "Hernquist: Potential not monotonic");
            CLEAN_EXIT(1);
        }
        
        // Allocate arrays using the global pointers
        innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
        Evalues = (double *)malloc((num_points + 1) * sizeof(double));
        
        // Set up fEintegrand for Hernquist
        gsl_function F_fE_hern;
        F_fE_hern.function = &fEintegrand_hernquist;
        
        innerintegrandvalues[0] = 0.0;
        Evalues[0] = Psimin;
        
        for (int i = 1; i <= num_points; i++) {
            double E_current = Psimin + (Psimax - Psimin) * ((double)i) / ((double)num_points);
            
            // Ensure strict monotonicity
            if (i > 0 && E_current <= Evalues[i-1]) {
                double increment = (Psimax - Psimin) / (double)num_points * 1e-6;
                if (increment == 0.0) increment = 1e-20;
                E_current = Evalues[i-1] + increment;
            }
            
            // Create Hernquist-specific parameter structure
            // Special case: if infinity was used, set use_om to 0 for true f(E)
            int use_om_for_hernquist = g_use_numerical_isotropic ? 0 : g_use_om_profile;
            
            fE_integrand_params_hernquist_t params_fE_hern = {
                E_current,          // E_current_shell
                splinerofPsi,       // spline_r_of_Psi
                rofPsiinterp,       // accel_r_of_Psi
                splinemass,         // spline_M_of_r
                enclosedmass,       // accel_M_of_r
                G_CONST,            // const_G_universal
                a_scale,            // hernquist_a_scale
                hern_params[2],     // hernquist_normalization
                Psimin,             // Psimin_global
                Psimax,             // Psimax_global
                use_om_for_hernquist // use_om flag (0 for true f(E) when infinity used)
            };
            F_fE_hern.params = &params_fE_hern;
            
            // Integration bounds for t = sqrt(E - Psi)
            double t_upper = sqrt(fmax(0.0, E_current - Psimin));
            double t_lower = t_upper / 1.0e4;
            
            if (t_upper <= t_lower + 1e-10) {
                result = 0.0;
            } else {
                gsl_integration_qag(&F_fE_hern, t_lower, t_upper,
                                   1e-8, 1e-8, 1000, GSL_INTEG_GAUSS61,
                                   w, &result, &error);
            }
            
            innerintegrandvalues[i] = result;
            Evalues[i] = E_current;
        }
        
        // Create f(E) interpolation
        g_main_fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
        g_main_fofEacc = gsl_interp_accel_alloc();
        gsl_interp_init(g_main_fofEinterp, Evalues, innerintegrandvalues, num_points + 1);

        // Check for negative f(E) or f(Q) values before particle sampling
        log_message("INFO", "Hernquist Numerical: Checking for negative distribution function values...");
        int num_points_plus_one_hern = num_points + 1;
        int should_abort_hern_num = check_and_warn_negative_fQ(
            &g_main_fofEinterp,
            &g_main_fofEacc,
            Evalues,
            innerintegrandvalues,
            &num_points_plus_one_hern,
            g_use_om_profile ? "f(Q)" : "f(E)",
            "Hernquist Numerical",
            1  // verbose=1 for main pathway
        );
        if (should_abort_hern_num) {
            log_message("INFO", "User aborted Hernquist Numerical IC generation due to negative distribution function");
            CLEAN_EXIT(0);
        }

        // Calculate velocity envelope for rejection sampling
        int num_maxv2f = 1000;
        double *maxv2f = (double *)malloc(num_maxv2f * sizeof(double));
        double *radius_maxv2f = (double *)malloc(num_maxv2f * sizeof(double));
        
        radius_maxv2f[0] = 0.0;
        maxv2f[0] = 0.0;
        
        for (int i_r = 1; i_r < num_maxv2f; i_r++) {
            double r_current = (double)i_r * cutoff_radius / (num_maxv2f - 1);
            radius_maxv2f[i_r] = r_current;
            
            double Psi_r = evaluatespline(splinePsi, Psiinterp, r_current);
            double max_v = sqrt(2.0 * (Psi_r - Psimin));
            double max_value = 0.0;
            
            // Find maximum of v^2 * f(E) over velocity range
            for (int j_v = 1; j_v < num_maxv2f - 2; j_v++) {
                double v_trial = max_v * ((double)j_v) / ((double)num_maxv2f);
                double E_test = Psi_r - 0.5 * v_trial * v_trial;
                double current_value = 0.0;
                
                if (E_test >= Psimin && E_test <= Psimax) {
                    current_value = v_trial * v_trial * 
                        fabs(gsl_interp_eval_deriv(g_main_fofEinterp, Evalues,
                                                   innerintegrandvalues, E_test, g_main_fofEacc));
                }
                if (isfinite(current_value) && current_value > max_value) {
                    max_value = current_value;
                }
            }
            maxv2f[i_r] = max_value;
        }
        
        // Extrapolate for r=0
        if (num_maxv2f >= 3) {
            maxv2f[0] = 2.0 * maxv2f[1] - maxv2f[2];
            if (maxv2f[0] < 0) maxv2f[0] = 0;
        } else {
            maxv2f[0] = maxv2f[1];
        }
        
        // Create spline for max v^2 * f(E)
        gsl_interp_accel *maxv2faccel = gsl_interp_accel_alloc();
        gsl_spline *splinemaxv2f = gsl_spline_alloc(gsl_interp_cspline, num_maxv2f);
        gsl_spline_init(splinemaxv2f, radius_maxv2f, maxv2f, num_maxv2f);
        
        // Create r(M) spline for particle sampling
        gsl_interp_accel *rofMaccel = gsl_interp_accel_alloc();
        gsl_spline *splinerofM = gsl_spline_alloc(gsl_interp_cspline, num_points);
        gsl_spline_init(splinerofM, mass, radius, num_points);

        // Now handle IC reading or generation
        if (g_restart_mode_active && strlen(restart_source_file) > 0 && !g_doSimRestart) {
            // Load last snapshot from file
            FILE *check_file = fopen(restart_source_file, "rb");
            if (!check_file) {
                fprintf(stderr, "ERROR: Cannot open restart file %s\n", restart_source_file);
                CLEAN_EXIT(1);
            }
            fseek(check_file, 0, SEEK_END);
            long file_size = ftell(check_file);
            fclose(check_file);

            int total_snapshots = file_size / (npts * 16);
            int snapshot_index = total_snapshots - 1;  // Last snapshot (0-based index)

            printf("Restart file has %d snapshots, loading last snapshot (index %d)\n",
                   total_snapshots, snapshot_index);

            // Allocate inverse_map before loading (needed for particle ID tracking)
            inverse_map = (int *)malloc(npts * sizeof(int));
            if (!inverse_map) {
                fprintf(stderr, "ERROR: Failed to allocate inverse_map before Hernquist numerical restart load\n");
                CLEAN_EXIT(1);
            }

            load_particles_from_restart(
                restart_source_file,
                snapshot_index,
                particles,
                npts,
                snapshot_block_size,
                inverse_map
            );
        } else if (doReadInit) {
            printf("Reading initial conditions from %s...\n", readInitFilename);
            read_initial_conditions(particles, npts_initial, readInitFilename);
        } else if (!g_doRestart) {
            log_message("INFO", "Hernquist: Starting particle generation for %d particles", npts_initial);
            
            // Generate particles using rejection sampling
            for (int k = 0; k < npts_initial; k++) {
            if (k % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                log_message("INFO", "Hernquist: Generated %d/%d particles", k, npts_initial);
            }
            
            // Sample radius from mass distribution
            double mass_frac = gsl_rng_uniform(g_rng) * 0.999999;
            double mass_sample = mass_frac * M_total;
            particles[0][k] = evaluatespline(splinerofM, rofMaccel, mass_sample);
            
            // Get potential and velocity envelope at this radius
            double Psi_r = evaluatespline(splinePsi, Psiinterp, particles[0][k]);
            double maxvalue = evaluatespline(splinemaxv2f, maxv2faccel, particles[0][k]);
            
            double sampled_scalar_speed = 0.0;
            
            if (!isfinite(Psi_r) || Psi_r <= Psimin + 1e-9 * fabs(Psimin)) {
                sampled_scalar_speed = 0.0;
            } else {
                double max_v = sqrt(fmax(0.0, 2.0 * (Psi_r - Psimin)));
                
                if (!isfinite(max_v) || max_v < 1e-9 || !isfinite(maxvalue) || maxvalue <= 1e-30) {
                    sampled_scalar_speed = 0.0;
                } else {
                    // Velocity rejection sampling
                    int vflag = 0;
                    int v_trials = 0;
                    
                    while (vflag == 0 && v_trials < 20000) {
                        v_trials++;
                        double trial_speed = gsl_rng_uniform(g_rng) * max_v;
                        double E_test = Psi_r - 0.5 * trial_speed * trial_speed;
                        
                        double deriv_dIdE = 0.0;
                        if (E_test >= Psimin && E_test <= Psimax) {
                            deriv_dIdE = gsl_interp_eval_deriv(g_main_fofEinterp, Evalues,
                                                               innerintegrandvalues, E_test, g_main_fofEacc);
                        }
                        
                        double target_func = trial_speed * trial_speed * fabs(deriv_dIdE);
                        if (!isfinite(target_func) || target_func < 0) target_func = 0.0;
                        
                        double ratio = target_func / maxvalue;
                        if (ratio < 0) ratio = 0;
                        if (ratio > 1.001) ratio = 1.0;
                        
                        if (gsl_rng_uniform(g_rng) < ratio) {
                            sampled_scalar_speed = trial_speed;
                            vflag = 1;
                        }
                    }
                    
                    if (!vflag) {
                        sampled_scalar_speed = 0.0;
                    }
                }
            }
            
            // Apply OM transformation if enabled (but not for numerical isotropic)
            if (g_use_om_profile && !g_use_numerical_isotropic) {
                // Osipkov-Merritt anisotropic sampling
                double w = sampled_scalar_speed;  // Pseudo-speed
                double mu_w = 2.0 * gsl_rng_uniform(g_rng) - 1.0;  // Isotropic in w-space
                
                // Map back to physical velocities
                double alpha_r = 1.0 + (particles[0][k] * particles[0][k]) / (g_om_anisotropy_radius * g_om_anisotropy_radius);
                double v_r = mu_w * w;
                double v_t = sqrt(fmax(0.0, 1.0 - mu_w * mu_w)) * w / sqrt(alpha_r);
                double v_mag = sqrt(v_r * v_r + v_t * v_t);
                
                particles[1][k] = v_mag;  // Store velocity magnitude
                particles[2][k] = particles[0][k] * v_t;  // Store angular momentum L
                particles[4][k] = (v_mag > 1e-9) ? (v_r / v_mag) : 0.0;  // Store physical mu
            } else {
                // Isotropic sampling
                double v = sampled_scalar_speed;
                double mu = 2.0 * gsl_rng_uniform(g_rng) - 1.0;
                particles[1][k] = v;  // Store velocity magnitude
                particles[2][k] = particles[0][k] * v * sqrt(fmax(0.0, 1.0 - mu * mu));  // L
                particles[4][k] = mu;  // Store mu
            }
            
            particles[3][k] = (double)k;  // Particle ID
        }
        
            // Clean up temporary arrays (but not global ones)
            free(maxv2f);
            free(radius_maxv2f);
            // DO NOT free innerintegrandvalues and Evalues - they are global variables
            gsl_spline_free(splinerofM);
            gsl_interp_accel_free(rofMaccel);
            gsl_spline_free(splinemaxv2f);
            gsl_interp_accel_free(maxv2faccel);
            // DO NOT free w here - managed at main function level
            
            // Set rmax for consistency
            rmax = cutoff_radius;

            log_message("INFO", "Hernquist: Successfully generated %d particles", npts_initial);
        }

        // Diagnostic loop for numerical Hernquist
        if (g_doDebug) {
            log_message("INFO", "HERNQUIST NUMERICAL DIAGNOSTIC LOOP: Starting convergence tests.");
            int diag_integration_points_array[2] = {1000, 10000};
            int diag_spline_points_array[2] = {1000, 10000};

            for (int ii_ip_hern = 0; ii_ip_hern < 2; ii_ip_hern++) {
                for (int ii_sp_hern = 0; ii_sp_hern < 2; ii_sp_hern++) {
                    int Nintegration_diag = diag_integration_points_array[ii_ip_hern];
                    int Nspline_diag_base = diag_spline_points_array[ii_sp_hern];
                    int num_points_diag = Nspline_diag_base * 10;

                    log_message("DEBUG", "Hernquist Diagnostic iteration Ni=%d, Ns_base=%d (points=%d)",
                                Nintegration_diag, Nspline_diag_base, num_points_diag);

                    double a_scale_diag = g_scale_radius_param;
                    double M_total_diag = g_halo_mass_param;
                    double rmax_diag = g_cutoff_factor_param * a_scale_diag;

                    gsl_integration_workspace *w_diag = gsl_integration_workspace_alloc(Nintegration_diag);
                    if (!w_diag) {
                        log_message("ERROR", "Failed to allocate GSL workspace for Hernquist diagnostic");
                        continue;
                    }

                    // Local splines and arrays for this diagnostic iteration
                    gsl_spline *splinemass_diag = NULL;
                    gsl_interp_accel *enclosedmass_diag = NULL;
                    gsl_spline *splinePsi_diag = NULL;
                    gsl_interp_accel *Psiinterp_diag = NULL;
                    gsl_spline *splinerofPsi_diag = NULL;
                    gsl_interp_accel *rofPsiinterp_diag = NULL;
                    gsl_interp *fofEinterp_diag = NULL;
                    gsl_interp_accel *fofEacc_diag = NULL;

                    double *mass_diag_arr = NULL, *radius_diag_arr = NULL, *Psivalues_diag_arr = NULL;
                    double *nPsivalues_diag_arr = NULL, *Evalues_diag_arr = NULL, *innerintegrandvalues_diag_arr = NULL;

                    // --- 1. Mass Profile M(r) ---
                    double hern_params_diag[3] = {a_scale_diag, M_total_diag, 1.0};
                    gsl_function F_hern_diag;
                    F_hern_diag.function = &massintegrand_hernquist;
                    F_hern_diag.params = hern_params_diag;

                    double result_diag, error_diag;
                    gsl_integration_qag(&F_hern_diag, 0.0, rmax_diag, 1e-10, 1e-10, Nintegration_diag, GSL_INTEG_GAUSS51,
                                        w_diag, &result_diag, &error_diag);
                    hern_params_diag[2] = M_total_diag / (4.0 * M_PI * result_diag);

                    mass_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    radius_diag_arr = (double *)malloc(num_points_diag * sizeof(double));

                    hernquist_potential_params hern_params_for_rmin = {M_total_diag, a_scale_diag};
                    double rmin_diag = find_minimum_useful_radius(hernquist_potential_wrapper, &hern_params_for_rmin, a_scale_diag, 1e-12);
                    double log_rmin_diag = log10(rmin_diag);
                    double log_rmax_diag = log10(rmax_diag);

                    for (int k = 0; k < num_points_diag; k++) {
                        double r_k = (k == num_points_diag - 1) ? rmax_diag : pow(10.0, log_rmin_diag + (log_rmax_diag - log_rmin_diag) * k / (num_points_diag - 1.0));
                        radius_diag_arr[k] = r_k;
                        gsl_integration_qag(&F_hern_diag, 0.0, r_k, 1e-10, 1e-10, Nintegration_diag, GSL_INTEG_GAUSS51,
                                            w_diag, &result_diag, &error_diag);
                        mass_diag_arr[k] = 4.0 * M_PI * result_diag;
                    }
                    enclosedmass_diag = gsl_interp_accel_alloc();
                    splinemass_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    gsl_spline_init(splinemass_diag, radius_diag_arr, mass_diag_arr, num_points_diag);

                    char diag_fname[256], diag_base[128];
                    snprintf(diag_base, sizeof(diag_base), "data/massprofile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base, 1, diag_fname, sizeof(diag_fname));
                    FILE *fp_diag = fopen(diag_fname, "wb");
                    if (fp_diag) {
                        for (double r_write = radius_diag_arr[0]; r_write < rmax_diag; r_write += rmax_diag / 900.0) {
                             fprintf_bin(fp_diag, "%f %f\n", r_write, gsl_spline_eval(splinemass_diag, r_write, enclosedmass_diag));
                        }
                        fclose(fp_diag);
                    }

                    // --- 2. Potential Profile Psi(r) ---
                    Psivalues_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    for (int k = 0; k < num_points_diag; k++) {
                        Psivalues_diag_arr[k] = potential_hernquist(radius_diag_arr[k], M_total_diag, a_scale_diag);
                    }
                    Psiinterp_diag = gsl_interp_accel_alloc();
                    splinePsi_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    gsl_spline_init(splinePsi_diag, radius_diag_arr, Psivalues_diag_arr, num_points_diag);

                    snprintf(diag_base, sizeof(diag_base), "data/Psiprofile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base, 1, diag_fname, sizeof(diag_fname));
                    fp_diag = fopen(diag_fname, "wb");
                    if (fp_diag) {
                        for (double r_write = radius_diag_arr[0]; r_write < rmax_diag; r_write += rmax_diag / 900.0) {
                             fprintf_bin(fp_diag, "%f %f\n", r_write, evaluatespline(splinePsi_diag, Psiinterp_diag, r_write));
                        }
                        fclose(fp_diag);
                    }

                    // --- 3. r(Psi) Spline ---
                    // Save original radius array before sorting (needed for dpsi_dr output later)
                    double *radius_diag_original = (double *)malloc(num_points_diag * sizeof(double));
                    for(int k=0; k<num_points_diag; ++k) radius_diag_original[k] = radius_diag_arr[k];

                    nPsivalues_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    for(int k=0; k<num_points_diag; ++k) nPsivalues_diag_arr[k] = -Psivalues_diag_arr[k];
                    sort_rr_psi_arrays(nPsivalues_diag_arr, radius_diag_arr, num_points_diag);

                    int unique_count_diag = 1;
                    for (int i = 1; i < num_points_diag; i++) {
                        if (nPsivalues_diag_arr[i] > nPsivalues_diag_arr[unique_count_diag-1]) {
                            if (i != unique_count_diag) {
                                nPsivalues_diag_arr[unique_count_diag] = nPsivalues_diag_arr[i];
                                radius_diag_arr[unique_count_diag] = radius_diag_arr[i];
                            }
                            unique_count_diag++;
                        }

                    }

                    rofPsiinterp_diag = gsl_interp_accel_alloc();
                    splinerofPsi_diag = gsl_spline_alloc(gsl_interp_cspline, unique_count_diag);
                    gsl_spline_init(splinerofPsi_diag, nPsivalues_diag_arr, radius_diag_arr, unique_count_diag);

                    // --- 4. I(E) Spline (f(E) precursor) ---
                    double Psimin_diag = Psivalues_diag_arr[num_points_diag-1];
                    double Psimax_diag = Psivalues_diag_arr[0];
                    if (Psimax_diag <= Psimin_diag) Psimax_diag = Psimin_diag + 1e-9;

                    Evalues_diag_arr = (double *)malloc((num_points_diag + 1) * sizeof(double));
                    innerintegrandvalues_diag_arr = (double *)malloc((num_points_diag + 1) * sizeof(double));
                    Evalues_diag_arr[0] = Psimin_diag;
                    innerintegrandvalues_diag_arr[0] = 0.0;

                    gsl_function F_fE_hern_diag;
                    F_fE_hern_diag.function = &fEintegrand_hernquist;

                    for (int k = 1; k <= num_points_diag; k++) {
                        double calE_diag = Psimin_diag + (Psimax_diag - Psimin_diag) * k / (double)num_points_diag;
                        Evalues_diag_arr[k] = calE_diag;

                        fE_integrand_params_hernquist_t params_fE_hern_diag = {
                            calE_diag, splinerofPsi_diag, rofPsiinterp_diag,
                            splinemass_diag, enclosedmass_diag, G_CONST,
                            a_scale_diag, hern_params_diag[2],
                            Psimin_diag, Psimax_diag, g_use_om_profile
                        };
                        F_fE_hern_diag.params = &params_fE_hern_diag;

                        double t_upper_diag = sqrt(fmax(0.0, calE_diag - Psimin_diag));
                        gsl_integration_qag(&F_fE_hern_diag, 0, t_upper_diag, 1e-6, 1e-6,
                                            Nintegration_diag, GSL_INTEG_GAUSS61, w_diag, &result_diag, &error_diag);
                        innerintegrandvalues_diag_arr[k] = result_diag;
                    }

                    fofEacc_diag = gsl_interp_accel_alloc();
                    fofEinterp_diag = gsl_interp_alloc(gsl_interp_cspline, num_points_diag + 1);
                    gsl_interp_init(fofEinterp_diag, Evalues_diag_arr, innerintegrandvalues_diag_arr, num_points_diag + 1);

                    // Check and correct negative f(E) in Hernquist diagnostic spline
                    int num_points_diag_hern_p1 = num_points_diag + 1;
                    int should_abort_diag_hern = check_and_warn_negative_fQ(
                        &fofEinterp_diag,
                        &fofEacc_diag,
                        Evalues_diag_arr,
                        innerintegrandvalues_diag_arr,
                        &num_points_diag_hern_p1,
                        g_use_om_profile ? "f(Q)" : "f(E)",
                        "Hernquist Numerical Diagnostic",
                        0  // verbose=0 for diagnostic (silent correction)
                    );
                    if (should_abort_diag_hern) {
                        log_message("INFO", "Skipping Hernquist diagnostic output due to negative f(E)");
                        gsl_interp_free(fofEinterp_diag);
                        gsl_interp_accel_free(fofEacc_diag);
                        continue;
                    }

                    snprintf(diag_base, sizeof(diag_base), "data/f_of_E_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base, 1, diag_fname, sizeof(diag_fname));
                    fp_diag = fopen(diag_fname, "wb");
                    if (fp_diag) {
                        for (int k = 0; k <= num_points_diag; k++) {
                            double E_diag = Evalues_diag_arr[k];
                            double deriv_diag = (k > 0 && k < num_points_diag) ? gsl_interp_eval_deriv(fofEinterp_diag, Evalues_diag_arr, innerintegrandvalues_diag_arr, E_diag, fofEacc_diag) : 0.0;
                            double fE_val_diag = fabs(deriv_diag) / (sqrt(8.0) * PI * PI);
                            if (!isfinite(fE_val_diag)) fE_val_diag = 0.0;
                            fprintf_bin(fp_diag, "%f %f\n", E_diag, fE_val_diag);
                        }
                        fclose(fp_diag);
                    }

                    // --- 5. Other Diagnostic Files ---
                    snprintf(diag_base, sizeof(diag_base), "data/density_profile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base, 1, diag_fname, sizeof(diag_fname));
                    fp_diag = fopen(diag_fname, "wb");
                    if (fp_diag) {
                        for (int k = 0; k < num_points_diag; k++) {
                            fprintf_bin(fp_diag, "%f %f\n", radius_diag_original[k], density_hernquist(radius_diag_original[k], M_total_diag, a_scale_diag));
                        }
                        fclose(fp_diag);
                    }

                    snprintf(diag_base, sizeof(diag_base), "data/dpsi_dr_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base, 1, diag_fname, sizeof(diag_fname));
                    fp_diag = fopen(diag_fname, "wb");
                    if (fp_diag) {
                        for (int k = 0; k < num_points_diag; k++) {
                            double rr_k = radius_diag_original[k];
                            if (rr_k > 1e-9) {
                                double Menc_k = gsl_spline_eval(splinemass_diag, rr_k, enclosedmass_diag);
                                fprintf_bin(fp_diag, "%f %f\n", rr_k, -(G_CONST * Menc_k) / (rr_k * rr_k));
                            }
                        }
                        fclose(fp_diag);
                    }

                    // --- Cleanup for diagnostic iteration ---
                    gsl_integration_workspace_free(w_diag);
                    if(splinemass_diag) gsl_spline_free(splinemass_diag);
                    if(enclosedmass_diag) gsl_interp_accel_free(enclosedmass_diag);
                    if(splinePsi_diag) gsl_spline_free(splinePsi_diag);
                    if(Psiinterp_diag) gsl_interp_accel_free(Psiinterp_diag);
                    if(splinerofPsi_diag) gsl_spline_free(splinerofPsi_diag);
                    if(rofPsiinterp_diag) gsl_interp_accel_free(rofPsiinterp_diag);
                    if(fofEinterp_diag) gsl_interp_free(fofEinterp_diag);
                    if(fofEacc_diag) gsl_interp_accel_free(fofEacc_diag);
                    free(mass_diag_arr);
                    free(radius_diag_arr);
                    free(radius_diag_original);
                    free(Psivalues_diag_arr);
                    free(nPsivalues_diag_arr);
                    free(Evalues_diag_arr);
                    free(innerintegrandvalues_diag_arr);
                }
            }
        }

    } else { // Default: Cored Plummer-like Profile
        log_message("INFO", "Starting IC generation using Cored Plummer-like profile pathway.");
        log_message("INFO", "Generating Initial Conditions using Cored Plummer-like profile...");

        // Configure density derivative function for Eddington inversion.
        // Osipkov-Merritt model requires augmented density derivative.
        if (g_use_om_profile) {
            g_density_derivative_func = &density_derivative_om_cored;
        } else {
            g_density_derivative_func = &drhodr;
        }

        /** @brief Arrays defining integration and spline point counts for theoretical calculations. */
        int integration_points_array[2] = {1000, 10000};
        int spline_points_array[2] = {1000, 10000};

    // Announce diagnostic loop start for consistency with NFW/Hernquist
    if (g_doDebug) {
        log_message("INFO", "CORED DIAGNOSTIC LOOP: Starting convergence tests for Cored profile.");
    }

    /**
     * @brief Theoretical Calculation Loop (Eddington's Formula - Multiple Params).
     * @details Calculates theoretical profiles (mass, potential, \f$f(E)\f$, density) based on the
     *          assumed initial density profile using Eddington's formula and GSL integration/splines.
     *          This loop iterates through different numbers of integration points (`Nintegration`)
     *          and spline points (`Nspline`) from the arrays above to generate reference files
     *          (e.g., massprofile_NiX_NsY.dat). These files are generated for comparison/validation
     *          but are *not* directly used in the main simulation timestepping or IC generation.
     *          They demonstrate the calculation process with varying numerical precision settings.
     */
    {
        for (int ii_ip = 0; ii_ip < 2; ii_ip++) // Loop over Nintegration values
        {
            for (int ii_sp = 0; ii_sp < 2; ii_sp++) // Loop over Nspline values
            {
                int Nintegration = integration_points_array[ii_ip];
                int Nspline = spline_points_array[ii_sp];

                double result, error, r;
                double calE;
                gsl_integration_workspace *w = gsl_integration_workspace_alloc(Nintegration);
                int i;

                gsl_function F;
                F.function = &massintegrand;
                F.params = NULL;

                double rmax = g_cored_profile_rmax_factor * g_cored_profile_rc;
                /** @note Calculate mass normalization factor for these params. */
                gsl_integration_qag(&F, 0.0, rmax, 0, 1.0e-12, Nintegration, 5, w, &result, &error);
                normalization = result;

                /** @note Calculate \f$M(r)\f$ and create mass spline for these params. */
                int num_points = Nspline * 10; // Use more points for spline data generation than for integration
                double *mass = (double *)malloc(num_points * sizeof(double));
                double *radius = (double *)malloc(num_points * sizeof(double));

                // Find minimum useful radius for cored profile
                cored_potential_params cored_params_diag = {g_cored_profile_halo_mass, g_cored_profile_rc};
                double r_min_adaptive_diag = find_minimum_useful_radius(cored_potential_wrapper, &cored_params_diag, g_cored_profile_rc, 1e-10);
                
                // Use logarithmic grid starting from adaptive minimum radius
                double log_rmin_cored_diag = log10(r_min_adaptive_diag);
                double log_rmax_cored_diag = log10(rmax);
                
                for (int i = 0; i < num_points; i++)
                {
                    double r;
                    if (i == num_points - 1) {
                        r = rmax;
                    } else {
                        double log_r = log_rmin_cored_diag + (log_rmax_cored_diag - log_rmin_cored_diag) * (double)i / (num_points - 1.0);
                        r = pow(10.0, log_r);
                    }
                    gsl_integration_qag(&F, 0.0, r, 0, 1.0e-12, Nintegration, 5, w, &result, &error);
                    mass[i] = result * g_cored_profile_halo_mass / normalization;
                    radius[i] = r;
                }

                gsl_interp_accel *enclosedmass = gsl_interp_accel_alloc();
                gsl_spline *splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
                gsl_spline_init(splinemass, radius, mass, num_points);
                double rlow = radius[0];
                double rhigh = radius[num_points - 1];

                /** @note Write mass profile file for these params (e.g., data/massprofile_Ni1k_Ns1k.dat). */
                char fname[256];
                FILE *fp;
                char base_filename_massprofile[256];
                snprintf(base_filename_massprofile, sizeof(base_filename_massprofile), "data/massprofile_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_massprofile, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (r = rlow; r < rhigh; r += rmax / 900.0)
                {
                    fprintf_bin(fp, "%f %f\n", r, gsl_spline_eval(splinemass, r, enclosedmass));
                }
                fclose(fp);

                /** @note Calculate Psi(r) and create potential spline for these params. */
                double *Psivalues = (double *)malloc(num_points * sizeof(double));
                double *nPsivalues = (double *)malloc(num_points * sizeof(double)); // For inverse spline r(Psi)
                // Prepare Psiintegrand parameters for diagnostic loop
                Psiintegrand_params psi_params_diag;
                psi_params_diag.massintegrand_func = &massintegrand;
                psi_params_diag.params_for_massintegrand = NULL;
                
                gsl_function F_for_psi_diag;
                F_for_psi_diag.function = &Psiintegrand;
                F_for_psi_diag.params = &psi_params_diag;
                
                for (i = 0; i < num_points; i++)
                {
                    double r = radius[i];  // Use the already calculated logarithmic grid
                    double r1 = fmax(r, g_cored_profile_rc / 1000000.0);
                    gsl_integration_qagiu(&F_for_psi_diag, r1, 0, 1e-12, Nintegration, w, &result, &error);
                    double M_at_r1 = gsl_spline_eval(splinemass, r1, enclosedmass);
                    double first_term = G_CONST * M_at_r1 / r1;
                    double second_term = G_CONST * result * g_cored_profile_halo_mass / normalization;
                    Psivalues[i] = (first_term + second_term);
                    nPsivalues[i] = -Psivalues[i];
                }


                gsl_interp_accel *Psiinterp = gsl_interp_accel_alloc();
                gsl_spline *splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
                gsl_spline_init(splinePsi, radius, Psivalues, num_points);

                /** @note Write potential profile file for these params (e.g., data/Psiprofile_Ni1k_Ns1k.dat). */
                char base_filename_psiprofile[256];
                snprintf(base_filename_psiprofile, sizeof(base_filename_psiprofile), "data/Psiprofile_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_psiprofile, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (r = rlow; r < ((double)num_points - 1.0) / ((double)num_points) * rmax; r += rmax / 900.0)
                {
                    fprintf_bin(fp, "%f %f\n", r, evaluatespline(splinePsi, Psiinterp, r));
                }
                fclose(fp);

                /** @note Create inverse spline \f$r(\Psi)\f$ for these params. */
                gsl_interp_accel *rofPsiinterp = gsl_interp_accel_alloc();
                gsl_spline *splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
                gsl_spline_init(splinerofPsi, nPsivalues, radius, num_points);

                /** @note Calculate inner integral for \f$f(E)\f$ and create \f$f(E)\f$ spline for these params. */
                double *innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
                double *Evalues = (double *)malloc((num_points + 1) * sizeof(double));
                double Psimin = Psivalues[num_points - 1];
                double Psimax = Psivalues[0];

                char base_filename_integrand[256];
                snprintf(base_filename_integrand, sizeof(base_filename_integrand), "data/integrand_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_integrand, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                calE = Psivalues[0];
                F.function = &fEintegrand;
                fEintegrand_params params = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass, g_use_om_profile};
                F.params = &params;
                for (i = 0; i < num_points; i++)
                {
                    double t = sqrt(calE - Psimin) * ((double)i) / ((double)num_points);
                    fprintf_bin(fp, "%f %f\n", t, fEintegrand(t, &params));
                }
                fclose(fp);

                if (Psimax <= Psimin) { // Check for diagnostic loop
                    if (g_doDebug) log_message("DEBUG", "Diagnostic: Psimax (%.6e) <= Psimin (%.6e) in diagnostic loop", Psimax, Psimin);
                    // Continue with diagnostic but note the issue
                }

                innerintegrandvalues[0] = 0.0;
                Evalues[0] = Psimin;
                for (i = 1; i <= num_points; i++)
                {
                    calE = Psimin + (Psimax - Psimin) * ((double)i) / ((double)num_points);
                    if (i > 0 && calE <= Evalues[i-1]) { // Adjust if not strictly increasing
                        double prev_E_diag = Evalues[i-1];
                        double ideal_step_diag = (Psimax - Psimin) / (double)num_points;
                        double incr_diag = ideal_step_diag * 1e-6;
                        if(incr_diag == 0.0) incr_diag = DBL_MIN * fabs(prev_E_diag) + DBL_MIN;
                        if(incr_diag == 0.0) incr_diag = 1e-20;
                        calE = prev_E_diag + incr_diag;
                        if (g_doDebug && (i <= 3 || i > num_points - 3)) {
                            log_message("DEBUG", "Evalues[%d] adjusted to %.6e", i, calE);
                        }
                    }
                    Evalues[i] = calE;

                    fEintegrand_params params2 = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass, g_use_om_profile};
                    F.params = &params2;
                    // Ensure sqrt argument is non-negative
                    double sqrt_arg_diag = calE - Psimin;
                    if (sqrt_arg_diag < 0) sqrt_arg_diag = 0.0;
                    // Use looser tolerance for OM due to augmented density numerical challenges
                    double abs_tol_diag = g_use_om_profile ? 1.0e-8 : 1.0e-12;
                    double rel_tol_diag = g_use_om_profile ? 1.0e-8 : 1.0e-12;
                    gsl_integration_qag(&F, sqrt(sqrt_arg_diag) / 1.0e4, sqrt(sqrt_arg_diag), abs_tol_diag, rel_tol_diag, Nintegration, 6, w, &result, &error);
                    innerintegrandvalues[i] = result;
                }

                gsl_interp *fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
                gsl_interp_init(fofEinterp, Evalues, innerintegrandvalues, num_points + 1);
                gsl_interp_accel *fofEacc = gsl_interp_accel_alloc();

                // Check and correct negative f(E) in Cored diagnostic spline
                int num_points_cored_diag_p1 = num_points + 1;
                int should_abort_diag_cored = check_and_warn_negative_fQ(
                    &fofEinterp,
                    &fofEacc,
                    Evalues,
                    innerintegrandvalues,
                    &num_points_cored_diag_p1,
                    g_use_om_profile ? "f(Q)" : "f(E)",
                    "Cored Diagnostic",
                    0  // verbose=0 for diagnostic (silent correction)
                );
                if (should_abort_diag_cored) {
                    log_message("INFO", "Skipping Cored diagnostic output due to negative f(E)");
                    gsl_interp_free(fofEinterp);
                    gsl_interp_accel_free(fofEacc);
                    continue;
                }

                /** @note Write theoretical density profile file for these params (e.g., data/density_profile_NiX_NsY.dat). */
                char base_filename[256];
                snprintf(base_filename, sizeof(base_filename), "data/density_profile_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 0; i < num_points; i++)
                {
                    double rr = radius[i];
                    double rho_r = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(rr / g_cored_profile_rc)));
                    fprintf_bin(fp, "%f %f\n", rr, rho_r);
                }
                fclose(fp);

                /** @note Write dPsi/dr file (data/dpsi_dr<suffix>.dat). Mode: wb (overwrite). */
                get_suffixed_filename("data/dpsi_dr.dat", 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 0; i < num_points; i++)
                {
                    double rr = radius[i];
                    if (rr > 0.0)
                    {
                        double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                        double dpsidr = -(G_CONST * Menc) / (rr * rr);
                        fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                    }
                }
                fclose(fp);

                /** @note Write drho/dPsi file (data/drho_dpsi<suffix>.dat). Mode: wb (overwrite). */
                get_suffixed_filename("data/drho_dpsi.dat", 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 1; i < num_points - 1; i++)
                {
                    double rr = radius[i];
                    double rho_left = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i - 1] / g_cored_profile_rc)));
                    double rho_right = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i + 1] / g_cored_profile_rc)));
                    double drho_dr_num = (rho_right - rho_left) / (radius[i + 1] - radius[i - 1]);
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                    double dPsidr = -(G_CONST * Menc) / (rr * rr);

                    if (dPsidr != 0.0)
                    {
                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        fprintf_bin(fp, "%f %f\n", Psi_val, drho_dr_num / dPsidr);
                    }
                }
                fclose(fp);

                /** @note Write \f$f(E)\f$ = dI/dE / const file for these params (e.g., data/f_of_E_NiX_NsY.dat). */
                char base_filename_fofe[256];
                snprintf(base_filename_fofe, sizeof(base_filename_fofe), "data/f_of_E_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_fofe, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 0; i <= num_points; i++)
                {
                    double E = Evalues[i];
                    double deriv = 0.0;
                    if (i > 0 && i < num_points + 1)
                    {
                        if (i > 0 && i < num_points)
                        {
                            deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i - 1]) / (Evalues[i + 1] - Evalues[i - 1]);
                        }
                        else if (i == 0)
                        {
                            deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i]) / (Evalues[i + 1] - Evalues[i]);
                        }
                        else if (i == num_points)
                        {
                            deriv = (innerintegrandvalues[i] - innerintegrandvalues[i - 1]) / (Evalues[i] - Evalues[i - 1]);
                        }
                    }
                    double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                    if (E == 0.0 || !isfinite(fE))
                        fE = 0.0;
                    fprintf_bin(fp, "%f %f\n", E, fE);
                }
                fclose(fp);

                get_suffixed_filename("data/df_fixed_radius.dat", 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                {
                    double r_fixed = 200.0;
                    double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_fixed);
                    Psi_rf *= VEL_CONV_SQ;
                    int vsteps = 100;
                    for (int vv = 0; vv <= vsteps; vv++)
                    {
                        double vtest = (double)vv * (sqrt(2.0 * Psi_rf) / (vsteps));
                        double Etest = Psi_rf - 0.5 * vtest * vtest;
                        double fEval = 0.0;
                        if (Etest > Psimin && Etest < Psimax)
                        {
                            double dval = gsl_interp_eval_deriv(fofEinterp, Evalues, innerintegrandvalues, Etest, fofEacc);
                            fEval = dval / (sqrt(8.0) * PI * PI) * vtest * vtest * r_fixed * r_fixed;
                        }
                        fprintf_bin(fp, "%f %f\n", vtest, fEval);
                    }
                }
                fclose(fp);

                gsl_spline_free(splinePsi);
                gsl_spline_free(splinerofPsi);
                gsl_interp_accel_free(Psiinterp);
                gsl_interp_accel_free(rofPsiinterp);
                gsl_spline_free(splinemass);
                gsl_interp_accel_free(enclosedmass);
                gsl_interp_free(fofEinterp);
                gsl_interp_accel_free(fofEacc);
                free(mass);
                free(radius);
                if (radius_monotonic_grid_nfw != NULL) {
                    free(radius_monotonic_grid_nfw);
                    radius_monotonic_grid_nfw = NULL;
                }
                free(Psivalues);
                free(nPsivalues);
                free(innerintegrandvalues);
                free(Evalues);
                gsl_integration_workspace_free(w);
                w = NULL;
            }
        }
    }

    // Announce diagnostic loop completion for consistency with NFW/Hernquist
    if (g_doDebug) {
        log_message("INFO", "CORED DIAGNOSTIC LOOP: Completed convergence tests for Cored profile.");
    }

    /**
     * @brief Main Theoretical Calculation (using default Nintegration=1000, Nspline=10000).
     * @details Repeats the theoretical profile calculations using fixed, default parameters
     *          (Nintegration=1000, Nspline=10000). The results from *this* block (splines:
     *          `splinemass`, `splinePsi`, `splinerofPsi`, `fofEinterp` and accelerators)
     *          are the ones used for generating the initial particle distribution and potentially
     *          for comparison during the simulation (e.g., debug energy calculation).
     *          Generates primary output files like `massprofile<suffix>.dat`, `Psiprofile<suffix>.dat`, `f_of_E<suffix>.dat`.
     */
    double r;
    
    /** @brief Allocate workspace for GSL integration operations. */
    w = gsl_integration_workspace_alloc(1000);

    gsl_function F;
    F.function = &massintegrand;
    F.params = NULL;

    rmax = g_cored_profile_rmax_factor * g_cored_profile_rc;
    gsl_integration_qag(&F, 0.0, rmax, 0, 1.0e-12, 1000, 5, w, &result, &error);
    normalization = result;

    num_points = 10000;
    /** @brief Allocate arrays for mass profile calculation. */
    mass = (double *)malloc(num_points * sizeof(double));
    radius = (double *)malloc(num_points * sizeof(double));

    // Find minimum useful radius adaptively
    cored_potential_params cored_params = {g_cored_profile_halo_mass, g_cored_profile_rc};
    double rmin_cored = find_minimum_useful_radius(cored_potential_wrapper, &cored_params, 
                                                  g_cored_profile_rc, 1e-12);
    
    log_message("INFO", "Cored: Using adaptive rmin = %.3e kpc (%.3e × scale radius)", 
                rmin_cored, rmin_cored/g_cored_profile_rc);
    
    // Use logarithmic grid starting from adaptive minimum radius
    double log_rmin_cored = log10(rmin_cored);
    double log_rmax_cored = log10(rmax);
    
    for (int i = 0; i < num_points; i++)
    {
        double r;
        if (i == num_points - 1) {
            r = rmax;  // Ensure exact endpoint
        } else {
            // Logarithmic spacing from rmin to rmax
            double log_r = log_rmin_cored + (log_rmax_cored - log_rmin_cored) * (double)i / (double)(num_points - 1);
            r = pow(10.0, log_r);
        }
        gsl_integration_qag(&F, 0.0, r, 0, 1.0e-12, 1000, 5, w, &result, &error);
        mass[i] = result * g_cored_profile_halo_mass / normalization;
        radius[i] = r;
    }

    /** @brief Create mass interpolation spline for \f$M(r)\f$. */
    enclosedmass = gsl_interp_accel_alloc();
    splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
    if (!check_strict_monotonicity(radius, num_points, "radius (main splinemass)")) { 
        fprintf(stderr, "CRITICAL: radius array not monotonic for main splinemass\n");
        fflush(stderr);
        CLEAN_EXIT(1); 
    }
    gsl_spline_init(splinemass, radius, mass, num_points);
    double rlow = radius[0];
    double rhigh = radius[num_points - 1];

    /** @brief Create 'data' directory again (harmless if exists). */
    {
        struct stat st = {0};
        if (stat("data", &st) == -1)
        {
            mkdir("data", 0755);
        }
    }

    /** @brief Write main mass profile file (data/massprofile<suffix>.dat). */
    get_suffixed_filename("data/massprofile.dat", 1, fname, sizeof(fname));
    fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
    double mass_at_rlow = mass[0];  // Mass at minimum radius
    double mass_at_rhigh = mass[num_points-1];  // Mass at maximum radius
    for (r = 0.0; r < rhigh; r += rmax / 900.0)
    {
        double mass_at_r = evaluatespline_with_boundary(splinemass, enclosedmass, r,
                                                       rlow, rhigh, mass_at_rlow, mass_at_rhigh);
        fprintf_bin(fp, "%f %f\n", r, mass_at_r);
    }
    fclose(fp);

    /** @brief Calculate Psi(r) array (num_points=10000) and create primary `splinePsi`. */
    Psivalues = (double *)malloc(num_points * sizeof(double));
    nPsivalues = (double *)malloc(num_points * sizeof(double));
    // Prepare Psiintegrand parameters for Cored profile
    Psiintegrand_params psi_params_cored;
    psi_params_cored.massintegrand_func = &massintegrand;
    psi_params_cored.params_for_massintegrand = NULL;
    
    gsl_function F_for_psi_cored;
    F_for_psi_cored.function = &Psiintegrand;
    F_for_psi_cored.params = &psi_params_cored;
    
    for (i = 0; i < num_points; i++)
    {
        double r = radius[i];  // Use the already calculated logarithmic grid
        double r1 = fmax(r, g_cored_profile_rc / 1000000.0);
        gsl_integration_qagiu(&F_for_psi_cored, r1, 0, 1e-12, 1000, w, &result, &error);
        double first_term = G_CONST * gsl_spline_eval(splinemass, r1, enclosedmass) / r1;
        double second_term = G_CONST * result * g_cored_profile_halo_mass / normalization;
        Psivalues[i] = (first_term + second_term);
        nPsivalues[i] = -Psivalues[i];
    }

    /** @brief Create potential interpolation spline for Psi(r). */
    Psiinterp = gsl_interp_accel_alloc();
    splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
    if (!check_strict_monotonicity(radius, num_points, "radius (main splinePsi)")) { 
        fprintf(stderr, "CRITICAL: radius array not monotonic for main splinePsi\n");
        fflush(stderr);
        CLEAN_EXIT(1); 
    }
    gsl_spline_init(splinePsi, radius, Psivalues, num_points);


    /** @brief Write main potential profile file (data/Psiprofile<suffix>.dat). */
    get_suffixed_filename("data/Psiprofile.dat", 1, fname, sizeof(fname));
    fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
    double psi_at_rlow = Psivalues[0];  // Psi at minimum radius
    double psi_at_rhigh = Psivalues[num_points-1];  // Psi at maximum radius
    for (r = 0.0; r < ((double)num_points - 1.0) / ((double)num_points) * rmax; r += rmax / 900.0)
    {
        double psi_at_r = evaluatespline_with_boundary(splinePsi, Psiinterp, r,
                                                       rlow, rhigh, psi_at_rlow, psi_at_rhigh);
        double mass_at_r = evaluatespline_with_boundary(splinemass, enclosedmass, r,
                                                       rlow, rhigh, mass_at_rlow, mass_at_rhigh);
        fprintf_bin(fp, "%f %f %f\n", r, psi_at_r, mass_at_r);
    }
    fclose(fp);

    /** @brief Create inverse spline \f$r(\Psi)\f$ for radius lookup from potential. */
    rofPsiinterp = gsl_interp_accel_alloc();
    splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
    
    gsl_spline_init(splinerofPsi, nPsivalues, radius, num_points);

    /** @brief Calculate distribution function \f$f(E)\f$ using Eddington's formula. */
    innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
    Evalues = (double *)malloc((num_points + 1) * sizeof(double));
    Psimin = Psivalues[num_points - 1];
    Psimax = Psivalues[0];
    get_suffixed_filename("data/integrand.dat", 1, fname, sizeof(fname));
    fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
    calE = Psivalues[0];
    F.function = &fEintegrand;
    fEintegrand_params params = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass, g_use_om_profile};
    F.params = &params;
    for (i = 0; i < num_points; i++)
    {
        double t = sqrt(calE - Psimin) * ((double)i) / ((double)num_points);
        fprintf_bin(fp, "%f %f\n", t, fEintegrand(t, &params));
    }
    fclose(fp);


    innerintegrandvalues[0] = 0.0;
    Evalues[0] = Psimin;

    for (i = 1; i <= num_points; i++)
    {
        calE = Psimin + (Psimax - Psimin) * ((double)i) / ((double)num_points);
        fEintegrand_params params2 = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass, g_use_om_profile};
        F.params = &params2;
        // Use looser tolerance for OM due to augmented density numerical challenges
        double abs_tol = g_use_om_profile ? 1.0e-8 : 1.0e-12;
        double rel_tol = g_use_om_profile ? 1.0e-8 : 1.0e-12;
        gsl_integration_qag(&F, sqrt(calE - Psimin) / 1.0e4, sqrt(calE - Psimin), abs_tol, rel_tol, 1000, 6, w, &result, &error);
        
        innerintegrandvalues[i] = result;
        Evalues[i] = calE;
    }

    /** @brief Create \f$f(E)\f$ interpolation for particle generation. */

    g_main_fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
    gsl_interp_init(g_main_fofEinterp, Evalues, innerintegrandvalues, num_points + 1);
    g_main_fofEacc = gsl_interp_accel_alloc();

    // Check for negative f(E) or f(Q) values before particle sampling
    log_message("INFO", "Cored: Checking for negative distribution function values...");
    int num_points_plus_one = num_points + 1;
    int should_abort_cored = check_and_warn_negative_fQ(
        &g_main_fofEinterp,
        &g_main_fofEacc,
        Evalues,
        innerintegrandvalues,
        &num_points_plus_one,
        g_use_om_profile ? "f(Q)" : "f(E)",
        "Cored",
        1  // verbose=1 for main pathway
    );
    if (should_abort_cored) {
        log_message("INFO", "User aborted Cored IC generation due to negative distribution function");
        CLEAN_EXIT(0);
    }

    /**
     * @brief Initialize particle arrays with default values (0.0, ID=index).
     * @details Uses the already allocated `particles` array from outer scope.
     *          Component indices: 0=radius, 1=velocity magnitude (initially), 2=ang. mom.,
     *          3=ID (initial index 0..npts_initial-1), 4=orientation(mu).
     */
    for (i = 0; i < npts_initial; i++)
    {
        particles[0][i] = 0.0;
        particles[1][i] = 0.0;
        particles[2][i] = 0.0;
        particles[3][i] = (double)i; // Initial ID is the index
        particles[4][i] = 0.0;
    }

    // Seed determination has been moved to before RNG initialization in main()
    
    // When using --readinit, prefer specified/loaded SIDM seed over derived values
    // or a newly generated one, NOT one derived from IC seed, as ICs are fixed.
    if (doReadInit && !g_sidm_seed_provided && !g_master_seed_provided && !g_attempt_load_seeds) {
        // If reading ICs and no SIDM/master seed is given and not told to load, ensure SIDM seed is fresh.
        // SIDM seed independence from IC seed ensured by earlier logic
    }

    /**
     * @brief INITIAL CONDITION HANDLING block.
     * @details Determines whether to generate initial conditions or load from file.
     *          - If `doReadInit` is true: Loads from `readInitFilename` via `read_initial_conditions`.
     *          - If `g_doRestart` is true: Skips generation (restart mode loads from checkpoint).
     *          - Otherwise: Generates particles using the Sample Generator.
     */

    if (doReadInit)
    {
        /** @brief Load existing initial conditions from specified file. */
        printf("Reading initial conditions from %s...\n", readInitFilename);
        read_initial_conditions(particles, npts_initial, readInitFilename);
    }
    else if (!g_doRestart) // Only generate if NOT reading init file and NOT restarting
    {
        /**
         * @brief SAMPLE GENERATOR block or RESTART LOADER.
         * @details Either loads particles from restart file or generates the initial particle 
         *          distribution for `npts_initial` particles, based on the theoretical equilibrium 
         *          distribution function `f(E)` derived via Eddington's formula.
         */

        // Check whether to load from restart file instead of generating
        if (g_restart_mode_active && strlen(restart_source_file) > 0 && !g_doSimRestart) {
            // Load last snapshot from file
            FILE *check_file = fopen(restart_source_file, "rb");
            if (!check_file) {
                fprintf(stderr, "ERROR: Cannot open restart file %s\n", restart_source_file);
                CLEAN_EXIT(1);
            }
            fseek(check_file, 0, SEEK_END);
            long file_size = ftell(check_file);
            fclose(check_file);

            int total_snapshots = file_size / (npts * 16);
            int snapshot_index = total_snapshots - 1;  // Last snapshot (0-based index)

            printf("Restart file has %d snapshots, loading last snapshot (index %d)\n",
                   total_snapshots, snapshot_index);

            // Allocate inverse_map before loading (needed for particle ID tracking)
            inverse_map = (int *)malloc(npts * sizeof(int));
            if (!inverse_map) {
                fprintf(stderr, "ERROR: Failed to allocate inverse_map before restart load\n");
                CLEAN_EXIT(1);
            }

            load_particles_from_restart(
                restart_source_file,
                snapshot_index,
                particles,
                npts,
                snapshot_block_size,
                inverse_map
            );
        } else {
            // Normal IC generation
            if (tidal_fraction > 0.0) log_message("INFO", "Cored IC Gen: Initial particle count before stripping: %d", npts_initial);
            log_message("INFO", "Cored IC Gen: Generating %d initial particle positions and velocities...", npts_initial);

        double vel, ratio, Psir, mu, maxv, maxvalue;

        /** @brief Set up GSL spline for radius as a function of enclosed mass: \f$r(M)\f$.
         *         Used for inverse transform sampling of radius. */
        gsl_interp_accel *rofMaccel = gsl_interp_accel_alloc();
        gsl_spline *splinerofM = gsl_spline_alloc(gsl_interp_cspline, num_points);
        if (!rofMaccel || !splinerofM) { /* Handle error */ CLEAN_EXIT(1); }
        gsl_spline_init(splinerofM, mass, radius, num_points);

        /** @brief Set up GSL spline for the maximum of `v^2 * \f$f(E)\f$` envelope at each radius `r`.
         *         Used for rejection sampling efficiency. \f$f(E)\f$ proportional to dI/dE. */
        double *maxv2f = (double *)malloc(num_points * sizeof(double));
        if (!maxv2f) { /* Handle error */ CLEAN_EXIT(1); }

        for (int i_r = 1; i_r < num_points; i_r++)
        {
            maxvalue = 0.0;
            Psir = evaluatespline(splinePsi, Psiinterp, radius[i_r]);
            maxv = sqrt(2.0 * (Psir - Psimin));
            for (int j_v = 1; j_v < num_points - 2; j_v++)
            {
                vel = maxv * ((double)j_v) / ((double)num_points);
                double currentvalue = vel * vel * gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, innerintegrandvalues, Psir - (0.5) * vel * vel, g_main_fofEacc);
                if (currentvalue > maxvalue) maxvalue = currentvalue;
            }
            maxv2f[i_r] = maxvalue;
        }
        // Extrapolate for r=0 assuming linear behavior near origin based on points 1 and 2
        if (num_points >= 3) {
            maxv2f[0] = 2.0 * maxv2f[1] - maxv2f[2];
            if (maxv2f[0] < 0.0) maxv2f[0] = 0.0; // Ensure non-negative
        } else if (num_points == 2) {
             maxv2f[0] = maxv2f[1]; // Simple fallback
        } else {
             maxv2f[0] = 0.0; // Fallback for very few points
        }

        gsl_interp_accel *maxv2faccel = gsl_interp_accel_alloc();
        gsl_spline *splinemaxv2f = gsl_spline_alloc(gsl_interp_cspline, num_points);
        if (!maxv2faccel || !splinemaxv2f) { /* Handle error */ free(maxv2f); CLEAN_EXIT(1); }
        gsl_spline_init(splinemaxv2f, radius, maxv2f, num_points);

        /**
         * @brief Generate `npts_initial` particle samples using sampling methods.
         * @note Uses GSL random number generation. Thread-safe parallel execution requires
         *       thread-specific RNG states rather than the global state.
         */
        for (i = 0; i < npts_initial; i++) // Loop over particles to generate
        {
            /** @note 1. Choose radius `r` using inverse transform sampling on \f$M(r)\f$. */
            double mass_frac_sample = gsl_rng_uniform(g_rng) * 0.999999; // Avoid sampling exactly M_total
            double mass_sample = mass_frac_sample * g_cored_profile_halo_mass;
            particles[0][i] = evaluatespline(splinerofM, rofMaccel, mass_sample);

            /** @note 2. Sample scalar speed and direction. */
            maxvalue = evaluatespline(splinemaxv2f, maxv2faccel, particles[0][i]);
            Psir = evaluatespline(splinePsi, Psiinterp, particles[0][i]);
            maxv = sqrt(2.0 * (Psir - Psimin));
            
            int vflag = 0;
            double sampled_scalar_speed = 0.0;
            while (vflag == 0) {
                double trial_speed = gsl_rng_uniform(g_rng) * maxv;
                // This samples v for isotropic, or pseudo-speed w for OM.
                // The distribution function f(E) or f(Q) is correctly used via g_main_fofEinterp.
                double target_func_val = trial_speed * trial_speed * gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, innerintegrandvalues, Psir - 0.5 * trial_speed * trial_speed, g_main_fofEacc);
                ratio = (maxvalue > 1e-15) ? (target_func_val / maxvalue) : 0.0;
                if (gsl_rng_uniform(g_rng) < ratio) {
                    sampled_scalar_speed = trial_speed;
                    vflag = 1;
                }
            }

            if (g_use_om_profile) {
                /**
                 * Osipkov-Merritt velocity transformation from pseudo-space to physical space.
                 * In pseudo-space, the OM invariant \f$Q = E - L^2/(2r_a^2)\f$ defines spherical surfaces.
                 * Sampling isotropic velocities in Q-space then mapping to physical velocities
                 * produces the desired radially-varying anisotropy \f$\beta(r) = r^2/(r^2 + r_a^2)\f$.
                 * Transformation: \f$v_r = w \cos(\theta_w)\f$, \f$v_t = w \sin(\theta_w)/\sqrt{1 + r^2/r_a^2}\f$
                 */
                double w = sampled_scalar_speed; // Pseudo-speed from f(Q) sampling
                double mu_w = 2.0 * gsl_rng_uniform(g_rng) - 1.0; // cos(θ_w) isotropic in pseudo-space

                // Transform from pseudo-velocity to physical velocity components
                double alpha_r = 1.0 + (particles[0][i] * particles[0][i]) / (g_om_anisotropy_radius * g_om_anisotropy_radius);
                double v_r = mu_w * w;
                double v_t = sqrt(1.0 - mu_w * mu_w) * w / sqrt(alpha_r);
                double v_mag = sqrt(v_r * v_r + v_t * v_t);
                
                particles[1][i] = v_mag; // Store velocity magnitude |v|
                particles[2][i] = particles[0][i] * v_t; // Store angular momentum L
                particles[4][i] = v_r / v_mag; // Store physical mu = v_r/|v|
            } else {
                // --- Isotropic Sampling (original logic) ---
                double v = sampled_scalar_speed;
                mu = 2.0 * gsl_rng_uniform(g_rng) - 1.0;
                particles[1][i] = v; // Store velocity magnitude
                particles[2][i] = particles[0][i] * v * sqrt(1.0 - mu * mu); // L
                particles[4][i] = mu; // Store mu
            }

            /** @note Store initial index as particle ID. */
            particles[3][i] = (double)i;
        } // End particle generation loop

        /** @brief Cleanup Sample Generator resources (splines, accelerators, temp arrays). */
        gsl_spline_free(splinerofM);
        gsl_interp_accel_free(rofMaccel);
        gsl_spline_free(splinemaxv2f);
        gsl_interp_accel_free(maxv2faccel);
        free(maxv2f);
        
        } // End else block for normal IC generation (not restart)

    } // End Sample Generator block (if !doReadInit && !g_doRestart)

    } // End if/else for profile selection for IC generation

    /**
     * @brief Save generated initial conditions to file if requested (`--writeinit`).
     * @details Saves the initial state if `doWriteInit` is true and not in restart mode.
     *          The state saved is *before* tidal stripping and unit conversion, using `write_initial_conditions`.
     * @see write_initial_conditions
     */
    if (doWriteInit && !g_doRestart)
    {
        printf("Saving initial conditions to %s...\n", writeInitFilename);
        write_initial_conditions(particles, npts_initial, writeInitFilename);
    }

    /**
     * @brief TIDAL STRIPPING IMPLEMENTATION block.
     * @details If `tidal_fraction` > 0 and not in restart mode, simulates tidal stripping by removing the outermost
     *          fraction of particles based on radius. It first sorts the `npts_initial`
     *          particles by radius, then keeps only the innermost `npts` particles.
     *          It reallocates the `particles` array to the final size `npts` and remaps
     *          the original indices stored in `particles[3]` to ranks [0, npts-1] using
     *          `reassign_orig_ids_with_rank`.
     * @see reassign_orig_ids_with_rank
     * @see sort_particles_with_alg
     */
    if (!g_doRestart) // Skip stripping if restarting
    {
        /** @note Stripping message shown only when `--ftidal` > 0. */
        if (tidal_fraction > 0.0) printf("Tidal stripping: sorting and retaining inner %.1f%% of particles...\n", (1.0 - tidal_fraction) * 100.0);

        /** @note Sort all `npts_initial` particles by radius using basic quadsort. */
        sort_particles_with_alg(particles, npts_initial, "quadsort"); // Sorts by particles[0]

        /** @note Allocate new smaller arrays (`final_particles`) for the `npts` particles to keep. */
        double **final_particles = (double **)malloc(7 * sizeof(double *));  // 7 rows for extended state
        if (final_particles == NULL)
        {
            fprintf(stderr, "Memory allocation failed for final_particles\n");
            CLEAN_EXIT(1);
        }

        /** @brief Copy innermost `npts` particles to final arrays and replace `particles` pointers. */
        for (int i = 0; i < 7; i++) // Loop over all 7 particle components
        {
            final_particles[i] = (double *)malloc(npts * sizeof(double));
            if (final_particles[i] == NULL)
            {
                fprintf(stderr, "Memory allocation failed for final_particles[%d]\n", i);
                CLEAN_EXIT(1);
            }
            /** @note Copy only the first `npts` elements (innermost after sort). */
            memcpy(final_particles[i], particles[i], npts * sizeof(double));
            
            /** @note Free original oversized array and update `particles[i]` pointer. */
            free(particles[i]);                // Free the original oversized array
            particles[i] = final_particles[i]; // particles[i] now points to the smaller array
        }
        free(final_particles); // Free the temporary ** structure, not the data arrays

        /** @note Stripping completion message shown only when `--ftidal` > 0. */
        if (tidal_fraction > 0.0)
        {
            printf("Tidal stripping complete: %d particles retained.\n\n", npts);
        }

        /**
         * @brief Remap particle IDs (in `particles[3]` after stripping) to sequential ranks [0, npts-1].
         * @details Ensures `particles[3][i]` holds the rank ID (0 to npts-1)
         *          for the particle at index `i` after stripping and sorting.
         */
        reassign_orig_ids_with_rank(particles[3], npts);
    } // End tidal stripping block (!g_doRestart)

    /**
     * @brief VELOCITY UNIT CONVERSION and ORIENTATION block.
     * @details Converts particle velocity magnitude (`particles[1]`) and angular momentum
     *          (`particles[2]`) from simulation generation units (implicitly km/s from `f(E)`)
     *          to physical units used in timestepping (kpc/Myr). It also applies the
     *          orientation parameter \f$\mu = v_{\text{radial}} / v_{\text{total}}\f$ (stored in `particles[4]`)
     *          to `particles[1]` to get the actual radial velocity component for integration.
     *          The original velocity magnitude in `particles[1]` is overwritten.
     *          Skipped in restart mode (`g_doRestart`).
     * @see kmsec_to_kpcmyr
     */
    if (!g_doRestart && !g_restart_mode_active)
    {
        for (i = 0; i < npts; i++) // Loop over final npts particles
        {
            // particles[1] holds velocity magnitude 'v' from Sample Generator
            // particles[4] holds orientation 'mu' from Sample Generator
            particles[1][i] *= particles[4][i]; ///< Apply orientation: v_rad = v * mu
            particles[1][i] *= kmsec_to_kpcmyr; ///< Convert v_rad [km/s] to [kpc/Myr]
            // particles[2] holds angular momentum L = r*v*sqrt(1-mu^2)
            particles[2][i] *= kmsec_to_kpcmyr; ///< Convert L [kpc*km/s] to [kpc\f$^2\f$/Myr]
            // particles[4] (mu) is no longer needed after this step.
        }
    }

    /**
     * @brief PARTICLE DATA EXPORT block (Initial State for Simulation).
     * @details Writes the initial state of all `npts` particles (after potential stripping,
     *          unit conversion, orientation application, and ID remapping) to a file
     *          named `data/particles<suffix>.dat`. This file represents the state at t=0
     *          entering the simulation loop.
     *          Format (binary via fprintf_bin):
     *          radius(kpc, float) v_radial(kpc/Myr, float) ang_mom(kpc\f$^2\f$/Myr, float) final_rank_id(float? check fprintf_bin)
     *          Skipped if `skip_file_writes` is true (restart mode).
     * @see fprintf_bin
     */
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/particles.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        FILE *fpp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin output
        if (fpp == NULL)
        {
            fprintf(stderr, "Error opening file %s for writing initial particles\n", suffixed_filename);
            exit(1); // Use CLEAN_EXIT?
        }

        /** @brief Write each particle's state (using fprintf_bin). */
        for (int i = 0; i < npts; i++) // i is current index (0..npts-1)
        {
            fprintf_bin(fpp, "%f %f %f %f\n", // fprintf_bin uses format types only
                        particles[0][i], // Radius (kpc)
                        particles[1][i], // Radial velocity (kpc/Myr)
                        particles[2][i], // Angular momentum (kpc²/Myr)
                        particles[3][i]);// Particle ID (final rank, written as float)
        }
        fclose(fpp);
        printf("Wrote initial particle state to %s\n", suffixed_filename);
    } // End skip_file_writes block for particles.dat

    /**
     * @brief SIMULATION TIMESTEP CALCULATION block.
     * @details Calculates the characteristic dynamical time (`tdyn`) based on core radius (`RC`)
     *          and total mass (`HALO_MASS`). Uses this to determine the total simulation
     *          duration (`totaltime = tfinal_factor * tdyn`) and the individual timestep
     *          size (`dt = totaltime / Ntimes`) used in the integration loop.
     * @see tdyn
     * @see totaltime
     * @see dt
     */
    double characteristic_radius_for_tdyn;
    if (g_use_nfw_profile) {
        characteristic_radius_for_tdyn = g_nfw_profile_rc; // Set from g_scale_radius_param
    } else if (g_use_hernquist_aniso_profile) {
        characteristic_radius_for_tdyn = g_scale_radius_param; // Use Hernquist scale radius
    } else {
        characteristic_radius_for_tdyn = g_cored_profile_rc; // Set from g_scale_radius_param
    }
    double tdyn = 1.0 / sqrt((VEL_CONV_SQ * G_CONST) * g_active_halo_mass / cube(characteristic_radius_for_tdyn));
    double totaltime = (double)tfinal_factor * tdyn; ///< Total simulation time (Myr)
    double dt = totaltime / ((double)(Ntimes - 1));  ///< Individual timestep size (Myr)
    printf("Dynamical time tdyn = %.4f Myr\n", tdyn);
    printf("Total simulation time = %.4f Myr (%.1f tdyn)\n", totaltime, (double)tfinal_factor);
    printf("Timestep dt = %.6f Myr\n\n", dt);

    // Calculate minimum critical radius for Levi-Civita switching
    double r_crit_min = 0.01 * characteristic_radius_for_tdyn;
    printf("\nDerived a fixed minimum critical radius for LC switching: r_crit_min = %.4e kpc\n", r_crit_min);

    /** @brief Initialize simulation time tracking and progress reporting. */
    double time = 0.0;                   ///< Current simulation time (Myr)
    double start_time = omp_get_wtime(); ///< Wall-clock start time for timing
    /** @brief Setup progress reporting steps (array `print_steps` holding step numbers for 0%, 5%, ..., 100%). */
    int print_steps[21];
    for (int k = 0; k <= 20; k++) print_steps[k] = (int)floor(k * 0.05 * Ntimes); // Calculate steps for progress output

    /** @brief Flag to determine if simulation phase can be skipped. */
    int skip_simulation = 0; 

    /** @brief Set up simulation tracking variables. */

    /**
     * @brief TRAJECTORY TRACKING AND BUFFER SETUP block.
     * @details Initializes the trajectory tracking system. This involves:
     *          - Determining which particles to track.
     *          - Calculating the required buffer size based on the snapshot writing schedule.
     *          - Allocating the fixed-size memory buffers for storing trajectory data
     *            before it is flushed to disk.
     */
    int upper_npts_num_traj = (num_traj_particles < npts) ? num_traj_particles : npts; ///< Actual number tracked

    int *traj_particle_ids = (int *)malloc(num_traj_particles * sizeof(int));
    if (!traj_particle_ids) {
        fprintf(stderr, "Error: Failed to allocate trajectory particle IDs array\n");
        CLEAN_EXIT(1);
    }
    for (int i = 0; i < num_traj_particles; i++) {
        traj_particle_ids[i] = i;  // Default: track particles 0 through num_traj_particles-1
    }

    // Calculate the size of the trajectory buffer. It must be large enough to hold
    // all timesteps that occur between two snapshot block writes.
    int buffer_size = dtwrite * snapshot_block_size;
    allocate_trajectory_buffers(upper_npts_num_traj, nlowest, buffer_size, npts);

    // Allocate double-precision snapshot buffer (silent feature, always enabled)
    allocate_double_precision_buffers(npts);

    /**
     * @brief ENERGY AND ANGULAR MOMENTUM INITIALIZATION block.
     * @details Calculates the initial relative energy `E_rel = Psi - KE` and angular
     *          momentum `L` for *all* particles based on their initial state (after
     *          stripping/conversion/remapping). Stores these initial values in `E_i_arr` and `L_i_arr`,
     *          indexed by the particle's final rank ID (`particles[3]`). Uses the
     *          theoretical potential spline `splinePsi` for the potential energy term.
     *          These arrays `E_i_arr`, `L_i_arr` store the *initial* values for later comparison.
     * @see E_i_arr
     * @see L_i_arr
     */
    // The g_E_init_vals and g_L_init_vals arrays are now used to store initial
    // particle energies and angular momenta. They are allocated in allocate_trajectory_buffers().

    /** @brief Calculate initial E_rel and L for each particle and store by final rank ID. */
    for (i = 0; i < npts; i++) // Loop through particles 0..npts-1 (current index 'i')
    {
        double rr = particles[0][i];            // Radius (at current index i)
        double vrad = particles[1][i];          // Radial velocity (at current index i)
        double ell = particles[2][i];           // Angular momentum (at current index i)
        int final_rank_id = (int)particles[3][i]; // Final rank ID (stored at current index i)

        // Ensure final_rank_id is within bounds [0, npts-1]
        if (final_rank_id < 0 || final_rank_id >= npts)
        {
            log_message("ERROR", "Invalid remapped_id %d encountered at index %d during initial E/L calculation.", final_rank_id, i);
            // Handle error: skip particle or exit on invalid ID
            continue;
        }

        /** @brief Calculate theoretical potential Psi(r) using initial theoretical spline. */
        // Uses Psiinterp accelerator associated with theoretical splinePsi
        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
        Psi_val *= VEL_CONV_SQ; // Convert to physical units (kpc/Myr)^2

        /** @brief Calculate relative energy E = Psi - (1/2)(v_r^2 + L^2/r^2). */
        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

        /** @brief Store initial E_rel and L using the final rank ID as index. */
        g_E_init_vals[final_rank_id] = E_rel;
        g_L_init_vals[final_rank_id] = ell;
    } // End initial E/L calculation loop

    /**
     * @brief LOW ANGULAR MOMENTUM PARTICLE SELECTION block.
     * @details Identifies the `nlowest` particles with either the lowest initial absolute angular
     *          momentum (`use_closest_to_Lcompare` = 0) or initial angular momentum closest
     *          to a reference value `\f$L_{compare}\f$` (`use_closest_to_Lcompare` = 1), based on
     *          the values stored in `L_i_arr` (indexed by final_rank_id).
     *          Stores the final rank IDs of these selected particles in the `chosen` array.
     *          Allocates tracking buffers for these particles.
     * @see LAndIndex
     * @see cmp_LAI
     * @see chosen
     */
    {
        // Check for --lvals-target and update L-selection mode if needed
        if (g_l_target_value >= 0.0) {
            use_closest_to_Lcompare = 1;
            Lcompare = g_l_target_value;
        }

        // For restart/extend: try to load previously chosen particles
        int loaded_chosen = 0;
        if (g_restart_mode_active) {
            // Construct chosen filename from source file
            char chosen_filename[512];
            if (strlen(restart_source_file) > 0) {
                // Extract base name and parameters from source file
                char base_name[512];
                strncpy(base_name, restart_source_file, sizeof(base_name) - 1);
                base_name[sizeof(base_name) - 1] = '\0';

                // Replace all_particle_data with chosen_particles
                char *apd_pos = strstr(base_name, "all_particle_data");
                if (apd_pos) {
                    // Calculate positions
                    size_t prefix_len = apd_pos - base_name;
                    char suffix[256];
                    strcpy(suffix, apd_pos + strlen("all_particle_data"));

                    // Build new filename
                    strncpy(chosen_filename, base_name, prefix_len);
                    chosen_filename[prefix_len] = '\0';
                    strcat(chosen_filename, "chosen_particles");
                    strcat(chosen_filename, suffix);
                } else {
                    // Fallback: append _chosen suffix
                    snprintf(chosen_filename, sizeof(chosen_filename), "%s_chosen", restart_source_file);
                }

                // Try to load chosen particles
                int *loaded_traj_ids = NULL;
                int loaded_n_traj = 0;
                if (load_chosen_particles(chosen_filename, &chosen, &nlowest, &loaded_traj_ids, &loaded_n_traj)) {
                    loaded_chosen = 1;

                    // If trajectory IDs were loaded, use them instead of default
                    if (loaded_traj_ids && loaded_n_traj > 0) {
                        free(traj_particle_ids);  // Free default array
                        traj_particle_ids = loaded_traj_ids;
                        num_traj_particles = loaded_n_traj;
                    }
                } else {
                    printf("No chosen particles file found, will select new particles\n");
                }
            }
        }

        if (!loaded_chosen) {
        /** @brief Create temporary LAndIndex array (size npts) to facilitate sorting by \f$L\f$. */
        LAndIndex *LAI = (LAndIndex *)malloc(npts * sizeof(LAndIndex));
        if (!LAI) { fprintf(stderr, "Error: Failed to allocate LAI array\n"); CLEAN_EXIT(1); }
        for (int i = 0; i < npts; i++) // 'i' here is the final_rank_id
        {
            if (use_closest_to_Lcompare) // Mode 1: Closest to Lcompare
            {
                double L_initial = g_L_init_vals[i];
                LAI[i].L = (L_initial - Lcompare) * (L_initial - Lcompare); // Store squared difference
                // Store the sign of the difference for accurate L reconstruction.
                LAI[i].sign = (L_initial - Lcompare >= 0.0) ? 1 : -1;
            }
            else // Mode 0: Lowest absolute L (using signed L values)
            {
                LAI[i].L = g_L_init_vals[i]; // Store initial L value (signed)
                LAI[i].sign = 0;       // Sign field not used for sorting in this mode
            }
            LAI[i].idx = i; // Store the final_rank_id associated with this L value
        }

        /** @brief Sort LAI array by \f$L\f$ member in ascending order (lowest \f$L\f$ or smallest \f$L\f$ deviation from \f$L_{compare}\f$). */
        qsort(LAI, npts, sizeof(LAndIndex), cmp_LAI);

        /** @brief If using Mode 1 (closest to \f$L_{compare}\f$), reconstruct actual \f$L\f$ values in LAI[].\f$L\f$.
         *         Note: This is primarily for the debug log output below; the primary goal is selecting indices. */
        if (use_closest_to_Lcompare) {
            for (int i = 0; i < npts; i++) {
                // Reconstruct L from squared difference: L = Lcompare ± sqrt(diff²)
                // Sign field determines direction of offset from Lcompare
                LAI[i].L = Lcompare + LAI[i].sign * sqrt(LAI[i].L);
            }
        }

        /** @brief Log the lowest \f$L\f$ particles (\f$L\f$ value and final_rank_id) if not restarting. */
        if (!g_doRestart) {
             log_message("DEBUG", "Selecting %d lowest L particles (mode=%d):", nlowest, use_closest_to_Lcompare);
             for (int i = 0; i < nlowest; i++) {
                 log_message("DEBUG", "  Rank %d: L=%.6f ID=%d", i, LAI[i].L, LAI[i].idx);
             }
        }

        /** @brief Allocate `chosen` array to store the final_rank_ids of the selected particles. */
        chosen = (int *)malloc(nlowest * sizeof(int));
        if (!chosen) { fprintf(stderr, "Error: Failed to allocate chosen array\n"); CLEAN_EXIT(1); }

        /** @brief Store the final rank IDs of the nlowest \f$L\f$ particles into `chosen`. */
        for (int i = 0; i < nlowest; i++) chosen[i] = LAI[i].idx;

        free(LAI);
        } // End if (!loaded_chosen)

        /** @brief Save chosen particles for future restarts (only for new runs). */
        if (!loaded_chosen && !g_doRestart) {
            char chosen_filename[512];
            snprintf(chosen_filename, sizeof(chosen_filename), "data/chosen_particles%s.dat", g_file_suffix);
            save_chosen_particles(chosen_filename, chosen, nlowest, traj_particle_ids, num_traj_particles);
            printf("Saved chosen particles to %s\n", chosen_filename);
        }

        // Buffers for lowest-L particles (g_lowestL_r_buf, etc.) allocated by
        // allocate_trajectory_buffers(); used to store trajectory data during simulation.
    } // End Low-L selection block

    /**
     * @brief SIMULATION DATA TRACKING SETUP block.
     * @details Allocates memory for tracking particle data during the simulation.
     *          Includes the `inverse_map` array (mapping final_rank_id to current array index after sorting),
     *          calculates `deltaM` (mass per particle), and allocates block storage arrays
     *          (`L_block`, `Rank_block`, `R_block`, `Vrad_block`) if `g_doAllParticleData` is enabled.
     *          These blocks store data for `snapshot_block_size` timesteps before being written to disk.
     * @see inverse_map
     * @see deltaM
     * @see L_block
     * @see Rank_block
     * @see R_block
     * @see Vrad_block
     */
    /** @brief Allocate index map: `inverse_map[final_rank_id]` will store the current index `i` after sorting. */
    if (!inverse_map) {  // Only allocate if not already allocated (e.g., by restart load)
        inverse_map = (int *)malloc(npts * sizeof(int));
        if (!inverse_map) { fprintf(stderr, "Error: Failed to allocate inverse_map\n"); CLEAN_EXIT(1); }
    }

    /** @brief Allocate particle scatter state array for AB3 history management after SIDM scattering. */
    g_particle_scatter_state = (int *)calloc(npts, sizeof(int)); // Use calloc to initialize all to 0
    if (!g_particle_scatter_state) {
        fprintf(stderr, "Error: Failed to allocate g_particle_scatter_state array\n");
        CLEAN_EXIT(1);
    }

    /** @brief Allocate current timestep scatter count array. */
    g_current_timestep_scatter_counts = (int *)calloc(npts, sizeof(int)); // Use calloc to initialize all to 0
    if (!g_current_timestep_scatter_counts) {
        fprintf(stderr, "Error: Failed to allocate g_current_timestep_scatter_counts array\n");
        CLEAN_EXIT(1);
    }


    /** @brief Calculate mass per particle based on *initial* particle count before stripping. Used for \f$M(\text{rank})\f$. */
    double deltaM = g_active_halo_mass / (double)npts_initial;

    // snapshot_block_size already declared earlier



    /** @brief Create filename for the main all-particle data output file. */
    char apd_filename[256]; // Filename for data/all_particle_data<suffix>.dat
    
    // Use override file if specified, otherwise generate filename from parameters
    if (g_restart_file_override != NULL) {
        strncpy(apd_filename, g_restart_file_override, sizeof(apd_filename) - 1);
        apd_filename[sizeof(apd_filename) - 1] = '\0';
        printf("Using override restart file: %s\n", apd_filename);
    } else {
        get_suffixed_filename("data/all_particle_data.dat", 1, apd_filename, sizeof(apd_filename));
    }

    // Restart variables already declared earlier at line 5704
    
    /**
     * @brief SIMULATION RESTART DETECTION block (`--sim-restart` mode).
     * @details When `--sim-restart` is enabled, check if the simulation is complete.
     *          If incomplete, prepare parameters for restart but do not read data yet.
     *          The actual particle reading will happen in the IC generation section.
     */
    if (g_doSimRestart)
    {
        // Calculate expected file size for a complete simulation
        long long expected_complete_size = (long long)total_writes * (long long)npts * 16LL;
        
        // Check if the all_particle_data file exists
        FILE *check_file = fopen(apd_filename, "rb");
        if (check_file)
        {
            // Get actual file size
            fseek(check_file, 0, SEEK_END);
            long actual_size = ftell(check_file);
            fclose(check_file);
            
            // Calculate completion status
            long long completed_records = actual_size / 16;  // 16 bytes per record
            long long completed_timesteps = completed_records / npts;
            
            if (actual_size >= expected_complete_size)
            {
                printf("Simulation complete: %lld of %d snapshots\n", completed_timesteps, total_writes);
                
                if (g_doSimRestartCheckOnly) {
                    exit(0);
                }
                g_doSimRestart = 0; // Disable restart (simulation complete)
            }
            else
            {
                printf("Simulation incomplete: %lld of %d snapshots completed\n", 
                       completed_timesteps, total_writes);
                
                if (g_doSimRestartCheckOnly) {
                    printf("[Check-only mode - no restart will be performed]\n");
                    exit(0);
                }
                
                // Prepare for restart (particle reading deferred)

                // Create backup of existing files (comprehensive backup like extend mode)
                printf("Creating backup of existing simulation files...\n");

                // 1. Backup main all_particle_data file
                if (create_backup_file(apd_filename) != 0) {
                    fprintf(stderr, "ERROR: Failed to create backup of all_particle_data. Aborting restart.\n");
                    exit(1);
                }
                printf("  Backed up: %s -> %s.backup\n", apd_filename, apd_filename);

                // 2. Backup trajectories file if it exists
                {
                    char traj_filename[512];
                    strncpy(traj_filename, apd_filename, sizeof(traj_filename) - 1);
                    traj_filename[sizeof(traj_filename) - 1] = '\0';
                    char *traj_pos = strstr(traj_filename, "all_particle_data");
                    if (traj_pos) {
                        char suffix[256];
                        strncpy(suffix, traj_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                        suffix[sizeof(suffix) - 1] = '\0';
                        strcpy(traj_pos, "trajectories");
                        strcat(traj_filename, suffix);

                        FILE *check_traj = fopen(traj_filename, "rb");
                        if (check_traj) {
                            fclose(check_traj);
                            if (create_backup_file(traj_filename) == 0) {
                                printf("  Backed up: %s -> %s.backup\n", traj_filename, traj_filename);
                            } else {
                                fprintf(stderr, "WARNING: Failed to backup trajectories file\n");
                            }
                        }
                    }
                }

                // 3. Backup energy_and_angular_momentum file if it exists
                {
                    char eam_filename[512];
                    strncpy(eam_filename, apd_filename, sizeof(eam_filename) - 1);
                    eam_filename[sizeof(eam_filename) - 1] = '\0';
                    char *eam_pos = strstr(eam_filename, "all_particle_data");
                    if (eam_pos) {
                        char suffix[256];
                        strncpy(suffix, eam_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                        suffix[sizeof(suffix) - 1] = '\0';
                        strcpy(eam_pos, "energy_and_angular_momentum_vs_time");
                        strcat(eam_filename, suffix);

                        FILE *check_eam = fopen(eam_filename, "rb");
                        if (check_eam) {
                            fclose(check_eam);
                            if (create_backup_file(eam_filename) == 0) {
                                printf("  Backed up: %s -> %s.backup\n", eam_filename, eam_filename);
                            } else {
                                fprintf(stderr, "WARNING: Failed to backup energy/angular momentum file\n");
                            }
                        }
                    }
                }

                // 4. Backup phi file if it exists
                {
                    char phi_filename[512];
                    strncpy(phi_filename, apd_filename, sizeof(phi_filename) - 1);
                    phi_filename[sizeof(phi_filename) - 1] = '\0';
                    char *ext_pos = strstr(phi_filename, ".dat");
                    if (ext_pos) {
                        strcpy(ext_pos, "_phi.dat");
                        FILE *check_phi = fopen(phi_filename, "rb");
                        if (check_phi) {
                            fclose(check_phi);
                            if (create_backup_file(phi_filename) == 0) {
                                printf("  Backed up: %s -> %s.backup\n", phi_filename, phi_filename);
                            } else {
                                fprintf(stderr, "WARNING: Failed to backup phi file\n");
                            }
                        }
                    }
                }

                // 5. Backup chosen particles file if it exists
                {
                    char chosen_filename[512];
                    strncpy(chosen_filename, apd_filename, sizeof(chosen_filename) - 1);
                    chosen_filename[sizeof(chosen_filename) - 1] = '\0';
                    char *chosen_pos = strstr(chosen_filename, "all_particle_data");
                    if (chosen_pos) {
                        char suffix[256];
                        strncpy(suffix, chosen_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                        suffix[sizeof(suffix) - 1] = '\0';
                        strcpy(chosen_pos, "chosen_particles");
                        strcat(chosen_filename, suffix);

                        FILE *check_chosen = fopen(chosen_filename, "rb");
                        if (check_chosen) {
                            fclose(check_chosen);
                            if (create_backup_file(chosen_filename) == 0) {
                                printf("  Backed up: %s -> %s.backup\n", chosen_filename, chosen_filename);
                            } else {
                                fprintf(stderr, "WARNING: Failed to backup chosen particles file\n");
                            }
                        }
                    }
                }

                // 6. Backup scatter counts file if SIDM is enabled
                if (g_enable_sidm_scattering) {
                    char scatter_filename[512];
                    strncpy(scatter_filename, apd_filename, sizeof(scatter_filename) - 1);
                    scatter_filename[sizeof(scatter_filename) - 1] = '\0';
                    char *scatter_pos = strstr(scatter_filename, "all_particle_data");
                    if (scatter_pos) {
                        char suffix[256];
                        strncpy(suffix, scatter_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                        suffix[sizeof(suffix) - 1] = '\0';
                        strcpy(scatter_pos, "all_particle_scatter_counts");
                        strcat(scatter_filename, suffix);

                        FILE *check_scatter = fopen(scatter_filename, "rb");
                        if (check_scatter) {
                            fclose(check_scatter);
                            if (create_backup_file(scatter_filename) == 0) {
                                printf("  Backed up: %s -> %s.backup\n", scatter_filename, scatter_filename);
                            } else {
                                fprintf(stderr, "WARNING: Failed to backup scatter counts file\n");
                            }
                        }
                    }
                }

                printf("Backup complete. Proceeding with restart...\n");

                // Check all files for consistency and find the last common complete snapshot
                printf("\nChecking file consistency across all output files...\n");
                int last_common = find_last_common_complete_snapshot(npts, dtwrite,
                                                                    num_traj_particles, nlowest,
                                                                    g_file_suffix);


                if (last_common >= 0) {
                    // Always use the minimum snapshot across all files
                    // This handles cases where all_particle_data has more snapshots than other files
                    if (last_common != (int)completed_timesteps - 1) {
                        printf("INFO: Using last common complete snapshot %d (all_particle_data had %lld)\n",
                               last_common, completed_timesteps - 1);
                    }

                    // Truncate all files to ensure consistency at the minimum snapshot
                    if (truncate_files_to_snapshot(last_common, npts, dtwrite,
                                                  num_traj_particles, nlowest,
                                                  g_file_suffix) != 0) {
                        fprintf(stderr, "ERROR: Failed to truncate files for consistency\n");
                        exit(1);
                    }

                    // Always use the common snapshot count (minimum across all files)
                    completed_timesteps = last_common;
                    g_restart_initial_timestep = last_common * dtwrite;
                } else {
                    fprintf(stderr, "ERROR: Could not determine file consistency. Aborting restart.\n");
                    exit(1);
                }

                // Store restart parameters using the consistent snapshot count
                // After find_last_common_complete_snapshot, completed_timesteps is already an INDEX
                // So restart_completed_snapshots is the INDEX of the last valid snapshot
                log_message("DEBUG", "Find_last_common path: last_common=%d (INDEX from find_last_common)", last_common);
                log_message("DEBUG", "Find_last_common path: completed_timesteps=%lld (set to last_common)", completed_timesteps);
                restart_completed_snapshots = (int)completed_timesteps;
                restart_initial_nwrite = restart_completed_snapshots;  // Already an index, no -1 needed
                g_restart_snapshots_is_count = 0;  // Reset flag: find_last_common returns an INDEX, not a COUNT
                log_message("DEBUG", "Find_last_common path: restart_completed_snapshots=%d (INDEX)", restart_completed_snapshots);
                log_message("DEBUG", "Find_last_common path: restart_initial_nwrite=%d (INDEX)", restart_initial_nwrite);
                log_message("DEBUG", "Find_last_common path: g_restart_snapshots_is_count=%d (should be 0)", g_restart_snapshots_is_count);
                strncpy(restart_source_file, apd_filename, sizeof(restart_source_file) - 1);

                // Adjust simulation parameters for remaining work
                original_Ntimes = Ntimes;  // Store original value for post-processing

                // Calculate actual restart timestep
                // If g_restart_initial_timestep was already set (e.g., from extend mode), keep it
                // Otherwise calculate from the completed snapshots
                int restart_timestep = g_restart_initial_timestep > 0 ?
                                      g_restart_initial_timestep :
                                      restart_completed_snapshots * dtwrite;

                // Update g_restart_initial_timestep to match truncated data
                // Match extend mode: use exact timestep without +1 adjustment
                if (g_restart_initial_timestep == 0 || g_restart_initial_timestep > restart_timestep) {
                    g_restart_initial_timestep = restart_timestep;
                }

                // Calculate restart_initial_time to be the time at the LAST completed timestep
                // This matches extend mode: time starts at last completed, j=0 computes next
                restart_initial_time = g_restart_initial_timestep * dt;

                Ntimes = Ntimes - g_restart_initial_timestep + 1;  // Reduce to remaining timesteps, +1 accounts for skipped j=0

                printf("Continuing from time = %.2f Myr\n", restart_initial_time);

                // Set restart flag
                g_restart_mode_active = 1;

                // NOW load particles from the truncated file (AFTER truncation, not before)
                char apd_filename[512];
                get_suffixed_filename("data/all_particle_data.dat", 1, apd_filename, sizeof(apd_filename));

                FILE *check_file = fopen(apd_filename, "rb");
                if (!check_file) {
                    fprintf(stderr, "ERROR: Cannot open restart file %s after truncation\n", apd_filename);
                    CLEAN_EXIT(1);
                }
                fseek(check_file, 0, SEEK_END);
                long file_size = ftell(check_file);
                fclose(check_file);

                int total_snapshots_after_truncation = file_size / (npts * 16);
                int snapshot_to_load = total_snapshots_after_truncation - 1;  // Last snapshot (0-based)

                printf("Loading particles from truncated file: %d snapshots, loading snapshot %d\n",
                       total_snapshots_after_truncation, snapshot_to_load);

                // Allocate inverse_map before loading (needed for particle ID tracking)
                inverse_map = (int *)malloc(npts * sizeof(int));
                if (!inverse_map) {
                    fprintf(stderr, "ERROR: Failed to allocate inverse_map before sim-restart load\n");
                    CLEAN_EXIT(1);
                }

                load_particles_from_restart(
                    apd_filename,
                    snapshot_to_load,
                    particles,
                    npts,
                    snapshot_block_size,
                    inverse_map
                );

                printf("Loaded snapshot %d from restart file after truncation\n", snapshot_to_load);
            }
        }
        else
        {
            if (g_doSimRestartCheckOnly) {
                printf("No simulation found to check.\n");
                exit(0);
            } else {
                // No checkpoint file exists
                g_doSimRestart = 0;
            }
        }
    }
    
    /**
     * @brief Simulation Extension Parameter Scaling Requirements
     * 
     * @details When extending a simulation by factor k using --sim-extend:
     *          
     *          Required parameter scaling:
     *          - ntimesteps \f$\rightarrow\f$ ntimesteps \f$\times\f$ k
     *          - tfinal \f$\rightarrow\f$ tfinal \f$\times\f$ k
     *          - nout \f$\rightarrow\f$ nout \f$\times\f$ k
     *          - dtwrite remains constant
     *          
     *          This maintains:
     *          - Constant physical timestep: dt = tfinal/ntimesteps
     *          - Constant write frequency: every dtwrite timesteps
     *          - Proportional snapshot count: nout + 1 total snapshots
     *          
     *          Constraint that must be satisfied:
     *          (ntimesteps - 1) % (dtwrite \f$\times\f$ nout) == 0
     *          
     * @example Extending by 1.5x:
     *          Original: --ntimesteps 50001 --tfinal 2 --nout 100 --dtwrite 500
     *          Extended: --ntimesteps 75001 --tfinal 3 --nout 150 --dtwrite 500
     */
    
    /**
     * @brief SIMULATION EXTENSION block (--sim-extend mode).
     * @details When --sim-extend is enabled with --extend-file, copy the source file
     *          to the new filename and prepare to extend it.
     */
    if (g_doSimExtend)
    {
        printf("Simulation extension mode enabled.\n");
        
        // Generate the target filename based on current parameters
        char target_filename[256];
        get_suffixed_filename("data/all_particle_data.dat", 1, target_filename, sizeof(target_filename));
        
        printf("Source file:  %s\n", g_extend_file_source);
        printf("Target file:  %s\n", target_filename);
        
        // Open source file
        FILE *source_file = fopen(g_extend_file_source, "rb");
        if (source_file == NULL) {
            fprintf(stderr, "ERROR: Failed to open source file '%s' for reading\n", g_extend_file_source);
            CLEAN_EXIT(1);
        }
        
        // Get source file size
        fseek(source_file, 0, SEEK_END);
        long source_size = ftell(source_file);
        fseek(source_file, 0, SEEK_SET);
        
        // Calculate source file stats
        long long source_records = source_size / 16;
        long long source_snapshots = source_records / npts;
        
        // Validate that the file size is consistent with the particle count
        if (source_records % npts != 0) {
            fprintf(stderr, "\nERROR: Source file size inconsistent with particle count\n");
            fprintf(stderr, "       File has %lld records, which is not divisible by N=%d\n", 
                    source_records, npts);
            fprintf(stderr, "       This suggests the file was created with a different number of particles\n");
            
            // Try to guess the actual N
            int possible_N[] = {100, 1000, 10000, 50000, 100000, 500000, 1000000};
            int num_possibilities = sizeof(possible_N) / sizeof(possible_N[0]);
            fprintf(stderr, "\n       Checking common particle counts:\n");
            for (int i = 0; i < num_possibilities; i++) {
                if (source_records % possible_N[i] == 0) {
                    long long snapshots_for_N = source_records / possible_N[i];
                    fprintf(stderr, "       N=%d would give %lld snapshots\n", 
                            possible_N[i], snapshots_for_N);
                }
            }
            fclose(source_file);
            CLEAN_EXIT(1);
        }
        
        printf("\nSource file analysis:\n");
        printf("  File size:        %ld bytes\n", source_size);
        printf("  Total records:    %lld\n", source_records);
        printf("  Snapshots:        %lld\n", source_snapshots);
        printf("  Particles (N):    %d (validated from file structure)\n", npts);
        
        // Try to validate dtwrite compatibility
        int src_N, src_Ntimes;
        double src_tfinal;
        int parsed_successfully = parse_nsphere_filename(g_extend_file_source, &src_N, &src_Ntimes, &src_tfinal);
        
        if (parsed_successfully) {
            // Store the actual restart timestep for lowest_l_trajectories indexing
            // src_Ntimes is the count (10001 for timesteps 0-10000)
            // The last timestep in source was src_Ntimes - 1 (e.g., 10000)
            // Extend mode continues FROM that timestep
            // Use the timestep INDEX (not count) as the restart point
            g_restart_initial_timestep = src_Ntimes - 1;
            printf("  Source Ntimes:    %d (from filename)\n", src_Ntimes);
            printf("  Source N:         %d (from filename)\n", src_N);
            printf("\nTarget extension parameters:\n");
            printf("  Target Ntimes:    %d\n", Ntimes);
            printf("  Target tfinal:    %d\n", tfinal_factor);
            printf("  Target dtwrite:   %d\n", dtwrite);
            printf("  Target N:         %d\n", npts);
            printf("\n");
            
            // Validate N from filename matches command line
            if (src_N != npts) {
                fprintf(stderr, "\nERROR: Filename indicates N=%d but command line specifies N=%d\n", 
                        src_N, npts);
                fprintf(stderr, "       Files must have the same number of particles for extension\n");
                
                // Also check if the file structure is consistent with either N
                if (source_records % src_N == 0) {
                    long long snapshots_for_src_N = source_records / src_N;
                    fprintf(stderr, "       File structure IS consistent with filename N=%d (%lld snapshots)\n",
                            src_N, snapshots_for_src_N);
                    fprintf(stderr, "       To extend this file, use: --nparticles %d\n", src_N);
                } else {
                    fprintf(stderr, "       WARNING: File structure is NOT consistent with filename N=%d either!\n", 
                            src_N);
                    fprintf(stderr, "       File may be corrupted or filename may be incorrect\n");
                }
                
                exit(1);
            }
            
            // Additional validation that file structure matches current N
            // Redundant validation for data consistency
            if (source_records % npts != 0) {
                fprintf(stderr, "\nERROR: File structure inconsistent with N=%d\n", npts);
                fprintf(stderr, "       File has %lld records, not divisible by N\n", source_records);
                fprintf(stderr, "       But filename indicates N=%d is correct. File may be corrupted.\n", npts);
                exit(1);
            }

            // Calculate the dtwrite from the source file parameters and actual writes FIRST
            // Calculated dtwrite used for validation below
            int src_dtwrite = (src_Ntimes - 1) / (source_snapshots - 1);

            // Cross-validate actual file size against command-line dtwrite
            // Use command-line dtwrite to detect corruption/truncation
            //       If dtwrite is wrong, the next validator will catch it
            int expected_writes_from_cmdline = ((src_Ntimes - 1) / dtwrite) + 1;
            long long expected_size_from_cmdline = (long long)expected_writes_from_cmdline * (long long)src_N * 16LL;

            if (source_size != expected_size_from_cmdline) {
                // Check if mismatch is due to wrong dtwrite or actual corruption
                int expected_writes_from_file = ((src_Ntimes - 1) / src_dtwrite) + 1;
                long long expected_size_from_file = (long long)expected_writes_from_file * (long long)src_N * 16LL;

                if (source_size == expected_size_from_file && src_dtwrite != dtwrite) {
                    // File matches calculated dtwrite but does not match command-line value
                    // Check if src_dtwrite produces exact division (indicates valid file)
                    int division_remainder = (src_Ntimes - 1) % src_dtwrite;

                    if (division_remainder == 0) {
                        // Exact division → file is valid, dtwrite mismatch is user error
                        // Let the dtwrite validator below handle this
                        printf("  Note:             byte count suggests different dtwrite (continuing to dtwrite validation)\n");
                    } else {
                        // Non-exact division → file is corrupted (truncated/padded)
                        fprintf(stderr, "\nERROR: Source file appears corrupted (inexact snapshot count).\n");
                        fprintf(stderr, "       Filename indicates: N=%d, Ntimes=%d\n",
                                src_N, src_Ntimes);
                        fprintf(stderr, "       File has %lld snapshots (requires exact divisor of Ntimes-1)\n",
                                source_snapshots);
                        fprintf(stderr, "       Calculated dtwrite=%d produces remainder %d when dividing Ntimes-1=%d\n",
                                src_dtwrite, division_remainder, src_Ntimes - 1);
                        fprintf(stderr, "\n");
                        fprintf(stderr, "       This suggests file truncation or corruption.\n");
                        fprintf(stderr, "       A valid file would have dtwrite that divides (Ntimes-1) exactly.\n");
                        exit(1);
                    }
                } else {
                    // File appears genuinely corrupted
                    fprintf(stderr, "\nERROR: Source file size does not match expected size.\n");
                    fprintf(stderr, "       Filename indicates: N=%d, Ntimes=%d\n",
                            src_N, src_Ntimes);
                    fprintf(stderr, "       File has %lld snapshots, implying dtwrite=%d\n",
                            source_snapshots, src_dtwrite);
                    fprintf(stderr, "       Expected (with --dtwrite %d): %lld bytes (%d writes × %d particles × 16)\n",
                            dtwrite, expected_size_from_cmdline, expected_writes_from_cmdline, src_N);
                    fprintf(stderr, "       Actual:   %ld bytes (%lld snapshots)\n",
                            source_size, source_snapshots);
                    fprintf(stderr, "\n");

                    if (source_size < expected_size_from_cmdline) {
                        fprintf(stderr, "       File appears TRUNCATED (missing %lld bytes).\n",
                                expected_size_from_cmdline - source_size);
                    } else {
                        fprintf(stderr, "       File has EXTRA data (%lld bytes).\n",
                                source_size - expected_size_from_cmdline);
                    }

                    fprintf(stderr, "       File may be corrupted.\n");
                    exit(1);
                }
            } else {
                printf("  Validated:        byte count matches expected size (%ld bytes)\n", source_size);
            }
            
            // Check if dtwrite values match
            if (src_dtwrite != dtwrite) {
                fprintf(stderr, "\nERROR: Source file dtwrite (%d) does not match current dtwrite (%d)\n", 
                        src_dtwrite, dtwrite);
                fprintf(stderr, "       The source file was run with: N=%d, Ntimes=%d, tfinal=%d\n", 
                        src_N, src_Ntimes, (int)src_tfinal);
                fprintf(stderr, "       Source file has %lld snapshots, implying dtwrite=%d\n",
                        source_snapshots, src_dtwrite);
                fprintf(stderr, "       Current parameters are: N=%d, Ntimes=%d, tfinal=%d, dtwrite=%d\n", 
                        npts, Ntimes, tfinal_factor, dtwrite);
                fprintf(stderr, "\n");
                fprintf(stderr, "       To extend this file correctly, use:\n");
                fprintf(stderr, "       --dtwrite %d\n", src_dtwrite);
                fprintf(stderr, "\n");
                fprintf(stderr, "       This will maintain the same write frequency as the original simulation.\n");
                exit(1);
            }
            
            printf("  Validated:        dtwrite matches (%d timesteps between writes)\n", src_dtwrite);

            // Validate physical timestep (dt) matches exactly using integer arithmetic
            // dt = tfinal × tdyn / (Ntimes - 1) must be identical
            // Cross multiply to avoid floating point: tfinal_src × (Ntimes_tgt - 1) = tfinal_tgt × (Ntimes_src - 1)
            long long lhs_product = (long long)src_tfinal * (long long)(Ntimes - 1);
            long long rhs_product = (long long)tfinal_factor * (long long)(src_Ntimes - 1);

            if (lhs_product != rhs_product) {
                fprintf(stderr, "\nERROR: Physical timestep (dt) does not match between source and target.\n");
                fprintf(stderr, "       Source: tfinal=%d, Ntimes=%d\n", (int)src_tfinal, src_Ntimes);
                fprintf(stderr, "       Target: tfinal=%d, Ntimes=%d\n", tfinal_factor, Ntimes);
                fprintf(stderr, "       dt must match exactly for extension (tfinal/Ntimes ratio must be identical).\n");
                exit(1);
            }

            printf("  Validated:        dt matches (physical timestep preserved)\n");
        } else {
            // Filename parsing failed - cannot proceed with extend mode
            fprintf(stderr, "\nERROR: Cannot parse --extend-file filename.\n");
            fprintf(stderr, "       Required format: data/all_particle_data_<tag>_<N>_<Ntimes>_<tfinal>.dat\n");
            fprintf(stderr, "       Got: %s\n", g_extend_file_source);
            fprintf(stderr, "\n");
            fprintf(stderr, "       Extend mode requires properly named files to locate:\n");
            fprintf(stderr, "         - all_particle_ids (trajectory tracking)\n");
            fprintf(stderr, "         - all_particle_phi (phi angles)\n");
            fprintf(stderr, "         - chosen_particles (trajectory continuity)\n");
            fprintf(stderr, "         - double_buffer (precision preservation)\n");
            exit(1);
        }
        
        // Check if extension makes sense
        if (source_snapshots >= total_writes) {
            fprintf(stderr, "\nERROR: Source file already contains %lld snapshots\n", source_snapshots);
            fprintf(stderr, "       Current parameters expect %d writes\n", total_writes);
            fprintf(stderr, "       Extension not needed - source already meets or exceeds target\n");
            fclose(source_file);
            CLEAN_EXIT(1);
        }
        
        // Copy source to target
        FILE *target_file = fopen(target_filename, "wb");
        if (target_file == NULL) {
            fprintf(stderr, "ERROR: Failed to create target file '%s'\n", target_filename);
            fclose(source_file);
            CLEAN_EXIT(1);
        }
        
        // Perform the copy
        char *copy_buffer = (char *)malloc(1024 * 1024); // 1MB buffer
        if (copy_buffer == NULL) {
            fprintf(stderr, "ERROR: Failed to allocate copy buffer\n");
            fclose(source_file);
            fclose(target_file);
            CLEAN_EXIT(1);
        }
        
        size_t bytes_read;
        while ((bytes_read = fread(copy_buffer, 1, 1024 * 1024, source_file)) > 0) {
            if (fwrite(copy_buffer, 1, bytes_read, target_file) != bytes_read) {
                fprintf(stderr, "ERROR: Write error during file copy\n");
                free(copy_buffer);
                fclose(source_file);
                fclose(target_file);
                CLEAN_EXIT(1);
            }
        }
        
        free(copy_buffer);
        fclose(source_file);
        fclose(target_file);

        // Copy the corresponding phi file if it exists
        // Replace "all_particle_data" with "all_particle_phi" in filename
        char source_phi_file[512];
        strncpy(source_phi_file, g_extend_file_source, sizeof(source_phi_file) - 1);
        source_phi_file[sizeof(source_phi_file) - 1] = '\0';
        char *data_pos = strstr(source_phi_file, "all_particle_data");
        if (data_pos) {
            // Replace "all_particle_data" with "all_particle_phi"
            char temp[512];
            size_t prefix_len = data_pos - source_phi_file;
            strncpy(temp, source_phi_file, prefix_len);
            temp[prefix_len] = '\0';
            strcat(temp, "all_particle_phi");
            strcat(temp, data_pos + strlen("all_particle_data"));
            strcpy(source_phi_file, temp);
        }

        char target_phi_file[512];
        strncpy(target_phi_file, target_filename, sizeof(target_phi_file) - 1);
        target_phi_file[sizeof(target_phi_file) - 1] = '\0';
        data_pos = strstr(target_phi_file, "all_particle_data");
        if (data_pos) {
            // Replace "all_particle_data" with "all_particle_phi"
            char temp[512];
            size_t prefix_len = data_pos - target_phi_file;
            strncpy(temp, target_phi_file, prefix_len);
            temp[prefix_len] = '\0';
            strcat(temp, "all_particle_phi");
            strcat(temp, data_pos + strlen("all_particle_data"));
            strcpy(target_phi_file, temp);
        }

        FILE *source_phi = fopen(source_phi_file, "rb");
        if (source_phi != NULL) {
            FILE *target_phi = fopen(target_phi_file, "wb");
            if (target_phi != NULL) {
                char *phi_buffer = (char *)malloc(1024 * 1024); // 1MB buffer
                if (phi_buffer != NULL) {
                    size_t phi_bytes;
                    while ((phi_bytes = fread(phi_buffer, 1, 1024 * 1024, source_phi)) > 0) {
                        fwrite(phi_buffer, 1, phi_bytes, target_phi);
                    }
                    free(phi_buffer);
                }
                fclose(target_phi);
            } else {
                fprintf(stderr, "WARNING: Could not create target phi file %s\n", target_phi_file);
            }
            fclose(source_phi);
        } else {
            printf("Note: No corresponding phi data file found at %s. File will be created during simulation if needed.\n", source_phi_file);
        }

        // Copy the total_energy_vs_time file as well if it exists
        if (g_doAllParticleData) {
            // Construct source and target energy filenames
            char source_energy_file[256];
            char target_energy_file[256];
            
            // Extract suffix from source all_particle_data filename
            char *source_base = strdup(g_extend_file_source);
            char *all_particle_str = strstr(source_base, "all_particle_data");
            
            if (all_particle_str != NULL) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, all_particle_str + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                // Construct source energy filename by replacing "all_particle_data" with "total_energy_vs_time"
                strcpy(source_energy_file, "data/total_energy_vs_time");
                strcat(source_energy_file, suffix);
                
                // Construct target energy filename
                get_suffixed_filename("data/total_energy_vs_time.dat", 1, target_energy_file, sizeof(target_energy_file));
                
                // Check if source energy file exists
                FILE *source_energy = fopen(source_energy_file, "rb");
                if (source_energy != NULL) {
                    // Copy the energy file
                    FILE *target_energy = fopen(target_energy_file, "wb");
                    if (target_energy != NULL) {
                        // Reuse the copy buffer approach
                        char *energy_buffer = (char *)malloc(65536); // 64KB buffer (energy files are small)
                        if (energy_buffer != NULL) {
                            size_t energy_bytes;
                            while ((energy_bytes = fread(energy_buffer, 1, 65536, source_energy)) > 0) {
                                if (fwrite(energy_buffer, 1, energy_bytes, target_energy) != energy_bytes) {
                                    fprintf(stderr, "WARNING: Error copying energy file\n");
                                    break;
                                }
                            }
                            free(energy_buffer);

                            // Mark energy data loaded for finalize_energy_diagnostics append mode
                            g_energy_snapshots_loaded = (int)source_snapshots;
                        }
                        fclose(target_energy);
                    } else {
                        fprintf(stderr, "WARNING: Could not create target energy file %s\n", target_energy_file);
                    }
                    fclose(source_energy);
                } else {
                    printf("Note: No existing energy file found at %s. File will be created during simulation.\n", source_energy_file);
                }
            }
            free(source_base);
        }

        // Copy the corresponding all_particle_ids_*.dat file if it exists
        char source_ids_file[512];
        strncpy(source_ids_file, g_extend_file_source, sizeof(source_ids_file) - 1);
        source_ids_file[sizeof(source_ids_file) - 1] = '\0';
        char *ids_ext_pos = strstr(source_ids_file, "all_particle_data");
        if (ids_ext_pos) {
            // Save the suffix that comes after "all_particle_data"
            char suffix[256];
            strncpy(suffix, ids_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
            suffix[sizeof(suffix) - 1] = '\0';
            // Now replace and append
            strcpy(ids_ext_pos, "all_particle_ids");
            strcat(source_ids_file, suffix);
        }

        char target_ids_file[512];
        strncpy(target_ids_file, target_filename, sizeof(target_ids_file) - 1);
        target_ids_file[sizeof(target_ids_file) - 1] = '\0';
        ids_ext_pos = strstr(target_ids_file, "all_particle_data");
        if (ids_ext_pos) {
            // Save the suffix that comes after "all_particle_data"
            char suffix[256];
            strncpy(suffix, ids_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
            suffix[sizeof(suffix) - 1] = '\0';
            // Now replace and append
            strcpy(ids_ext_pos, "all_particle_ids");
            strcat(target_ids_file, suffix);
        }

        FILE *source_ids = fopen(source_ids_file, "rb");
        if (source_ids != NULL) {
            FILE *target_ids = fopen(target_ids_file, "wb");
            if (target_ids != NULL) {
                char *ids_buffer = (char *)malloc(1024 * 1024); // 1MB buffer
                if (ids_buffer != NULL) {
                    size_t ids_bytes;
                    while ((ids_bytes = fread(ids_buffer, 1, 1024 * 1024, source_ids)) > 0) {
                        fwrite(ids_buffer, 1, ids_bytes, target_ids);
                    }
                    free(ids_buffer);
                }
                fclose(target_ids);
            } else {
                fprintf(stderr, "WARNING: Could not create target IDs file %s\n", target_ids_file);
            }
            fclose(source_ids);
        } else {
            printf("Note: No corresponding particle IDs file found at %s. File will be created during simulation.\n", source_ids_file);
        }

        // Copy the corresponding chosen_particles file if it exists
        char source_chosen_file[512];
        char target_chosen_file[512];

        // Build source chosen filename from source file pattern
        char *apd_pos = strstr(g_extend_file_source, "all_particle_data");
        if (apd_pos) {
            // Save the suffix that comes after "all_particle_data"
            char suffix[256];
            strncpy(suffix, apd_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
            suffix[sizeof(suffix) - 1] = '\0';
            size_t prefix_len = apd_pos - g_extend_file_source;
            strncpy(source_chosen_file, g_extend_file_source, prefix_len);
            source_chosen_file[prefix_len] = '\0';
            strcat(source_chosen_file, "chosen_particles");
            strcat(source_chosen_file, suffix);
        } else {
            // Fallback
            snprintf(source_chosen_file, sizeof(source_chosen_file), "%s_chosen", g_extend_file_source);
        }

        // Build target chosen filename - g_file_suffix not yet initialized
        // Use the same pattern as the source but with new tag
        char *apd_pos2 = strstr(target_filename, "all_particle_data");
        if (apd_pos2) {
            // Save the suffix that comes after "all_particle_data"
            char suffix[256];
            strncpy(suffix, apd_pos2 + strlen("all_particle_data"), sizeof(suffix) - 1);
            suffix[sizeof(suffix) - 1] = '\0';
            size_t prefix_len2 = apd_pos2 - target_filename;
            strncpy(target_chosen_file, target_filename, prefix_len2);
            target_chosen_file[prefix_len2] = '\0';
            strcat(target_chosen_file, "chosen_particles");
            strcat(target_chosen_file, suffix);
        } else {
            snprintf(target_chosen_file, sizeof(target_chosen_file), "%s_chosen", target_filename);
        }

        FILE *source_chosen = fopen(source_chosen_file, "rb");
        if (source_chosen != NULL) {
            FILE *target_chosen = fopen(target_chosen_file, "wb");
            if (target_chosen != NULL) {
                char *chosen_buffer = (char *)malloc(1024); // Small buffer for chosen file
                if (chosen_buffer != NULL) {
                    size_t chosen_bytes;
                    while ((chosen_bytes = fread(chosen_buffer, 1, 1024, source_chosen)) > 0) {
                        fwrite(chosen_buffer, 1, chosen_bytes, target_chosen);
                    }
                    free(chosen_buffer);
                }
                fclose(target_chosen);
            } else {
                fprintf(stderr, "WARNING: Could not create target chosen file %s\n", target_chosen_file);
            }
            fclose(source_chosen);
        } else {
            printf("Note: No chosen particles file found at %s\n", source_chosen_file);
        }

        // Copy the corresponding scatter counts file if SIDM is enabled
        if (g_enable_sidm_scattering) {
            char source_scatter_file[512];
            strncpy(source_scatter_file, g_extend_file_source, sizeof(source_scatter_file) - 1);
            source_scatter_file[sizeof(source_scatter_file) - 1] = '\0';
            char *scatter_ext_pos = strstr(source_scatter_file, "all_particle_data");
            if (scatter_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, scatter_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                strcpy(scatter_ext_pos, "all_particle_scatter_counts");
                strcat(source_scatter_file, suffix);
            }

            char target_scatter_file[512];
            strncpy(target_scatter_file, target_filename, sizeof(target_scatter_file) - 1);
            target_scatter_file[sizeof(target_scatter_file) - 1] = '\0';
            scatter_ext_pos = strstr(target_scatter_file, "all_particle_data");
            if (scatter_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, scatter_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                strcpy(scatter_ext_pos, "all_particle_scatter_counts");
                strcat(target_scatter_file, suffix);
            }

            FILE *source_scatter = fopen(source_scatter_file, "rb");
            if (source_scatter != NULL) {
                FILE *target_scatter = fopen(target_scatter_file, "wb");
                if (target_scatter != NULL) {
                    char *scatter_buffer = (char *)malloc(1024 * 1024); // 1MB buffer
                    if (scatter_buffer != NULL) {
                        size_t scatter_bytes;
                        while ((scatter_bytes = fread(scatter_buffer, 1, 1024 * 1024, source_scatter)) > 0) {
                            fwrite(scatter_buffer, 1, scatter_bytes, target_scatter);
                        }
                        free(scatter_buffer);
                    }
                    fclose(target_scatter);
                } else {
                    fprintf(stderr, "WARNING: Could not create target scatter counts file %s\n", target_scatter_file);
                }
                fclose(source_scatter);
            }
        }

        // Copy the energy_and_angular_momentum_vs_time file if it exists
        {
            char source_eam_file[512];
            strncpy(source_eam_file, g_extend_file_source, sizeof(source_eam_file) - 1);
            source_eam_file[sizeof(source_eam_file) - 1] = '\0';
            char *eam_ext_pos = strstr(source_eam_file, "all_particle_data");
            if (eam_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, eam_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                strcpy(eam_ext_pos, "energy_and_angular_momentum_vs_time");
                strcat(source_eam_file, suffix);
            }

            char target_eam_file[512];
            get_suffixed_filename("data/energy_and_angular_momentum_vs_time.dat", 1, target_eam_file, sizeof(target_eam_file));

            FILE *source_eam = fopen(source_eam_file, "rb");
            if (source_eam != NULL) {
                FILE *target_eam = fopen(target_eam_file, "wb");
                if (target_eam != NULL) {
                    char *eam_buffer = (char *)malloc(1024 * 1024); // 1MB buffer
                    if (eam_buffer != NULL) {
                        size_t eam_bytes;
                        while ((eam_bytes = fread(eam_buffer, 1, 1024 * 1024, source_eam)) > 0) {
                            fwrite(eam_buffer, 1, eam_bytes, target_eam);
                        }
                        free(eam_buffer);
                    }
                    fclose(target_eam);
                } else {
                    fprintf(stderr, "WARNING: Could not create target energy/angular momentum file %s\n", target_eam_file);
                }
                fclose(source_eam);
            }
        }

        // Copy the trajectories file if it exists
        {
            char source_traj_file[512];
            strncpy(source_traj_file, g_extend_file_source, sizeof(source_traj_file) - 1);
            source_traj_file[sizeof(source_traj_file) - 1] = '\0';
            char *traj_ext_pos = strstr(source_traj_file, "all_particle_data");
            if (traj_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, traj_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                // Now replace and append
                strcpy(traj_ext_pos, "trajectories");
                strcat(source_traj_file, suffix);
            } else {
                // Fallback: construct from scratch
                snprintf(source_traj_file, sizeof(source_traj_file), "data/trajectories%s",
                        strrchr(g_extend_file_source, '_') ? strrchr(g_extend_file_source, '_') : "");
            }

            char target_traj_file[512];
            strncpy(target_traj_file, target_filename, sizeof(target_traj_file) - 1);
            target_traj_file[sizeof(target_traj_file) - 1] = '\0';
            traj_ext_pos = strstr(target_traj_file, "all_particle_data");
            if (traj_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, traj_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                // Now replace and append
                strcpy(traj_ext_pos, "trajectories");
                strcat(target_traj_file, suffix);
            } else {
                // Fallback: construct from scratch
                snprintf(target_traj_file, sizeof(target_traj_file), "data/trajectories%s",
                        strrchr(target_filename, '_') ? strrchr(target_filename, '_') : "");
            }

            FILE *source_traj = fopen(source_traj_file, "rb");
            if (source_traj) {
                FILE *target_traj = fopen(target_traj_file, "wb");
                if (target_traj) {
                    // Reset to beginning of source file
                    fseek(source_traj, 0, SEEK_SET);

                    // Copy the file
                    char *traj_buffer = (char *)malloc(1048576); // 1MB buffer
                    if (traj_buffer) {
                        size_t traj_bytes;
                        while ((traj_bytes = fread(traj_buffer, 1, 1048576, source_traj)) > 0) {
                            if (fwrite(traj_buffer, 1, traj_bytes, target_traj) != traj_bytes) {
                                fprintf(stderr, "WARNING: Error copying trajectories file\n");
                                break;
                            }
                        }
                        free(traj_buffer);
                    }
                    fclose(target_traj);
                } else {
                    fprintf(stderr, "WARNING: Could not create target trajectories file %s\n", target_traj_file);
                }
                fclose(source_traj);
            }
        }

        // Copy the single_trajectory file if it exists
        {
            char source_single_file[512];
            strncpy(source_single_file, g_extend_file_source, sizeof(source_single_file) - 1);
            source_single_file[sizeof(source_single_file) - 1] = '\0';
            char *single_ext_pos = strstr(source_single_file, "all_particle_data");
            if (single_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, single_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                // Now replace and append
                strcpy(single_ext_pos, "single_trajectory");
                strcat(source_single_file, suffix);
            } else {
                // Fallback: construct from scratch
                snprintf(source_single_file, sizeof(source_single_file), "data/single_trajectory%s",
                        strrchr(g_extend_file_source, '_') ? strrchr(g_extend_file_source, '_') : "");
            }

            char target_single_file[512];
            strncpy(target_single_file, target_filename, sizeof(target_single_file) - 1);
            target_single_file[sizeof(target_single_file) - 1] = '\0';
            single_ext_pos = strstr(target_single_file, "all_particle_data");
            if (single_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, single_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                // Now replace and append
                strcpy(single_ext_pos, "single_trajectory");
                strcat(target_single_file, suffix);
            } else {
                // Fallback: construct from scratch
                snprintf(target_single_file, sizeof(target_single_file), "data/single_trajectory%s",
                        strrchr(target_filename, '_') ? strrchr(target_filename, '_') : "");
            }

            FILE *source_single = fopen(source_single_file, "rb");
            if (source_single) {
                FILE *target_single = fopen(target_single_file, "wb");
                if (target_single) {
                    // Reset to beginning of source file
                    fseek(source_single, 0, SEEK_SET);

                    // Copy the file
                    char *single_buffer = (char *)malloc(1048576); // 1MB buffer
                    if (single_buffer) {
                        size_t single_bytes;
                        while ((single_bytes = fread(single_buffer, 1, 1048576, source_single)) > 0) {
                            if (fwrite(single_buffer, 1, single_bytes, target_single) != single_bytes) {
                                fprintf(stderr, "WARNING: Error copying single_trajectory file\n");
                                break;
                            }
                        }
                        free(single_buffer);
                    }
                    fclose(target_single);
                } else {
                    fprintf(stderr, "WARNING: Could not create target single_trajectory file %s\n", target_single_file);
                }
                fclose(source_single);
            }
        }

        // Copy the lowest_l or chosen_l trajectories file if it exists
        {
            // Determine which trajectory file to look for (chosen_l takes priority)
            const char* traj_basename = (g_l_target_value >= 0.0) ? "chosen_l_trajectories" : "lowest_l_trajectories";

            char source_lowestl_file[512];
            strncpy(source_lowestl_file, g_extend_file_source, sizeof(source_lowestl_file) - 1);
            source_lowestl_file[sizeof(source_lowestl_file) - 1] = '\0';
            char *lowestl_ext_pos = strstr(source_lowestl_file, "all_particle_data");
            if (lowestl_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, lowestl_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                // Now replace and append
                strcpy(lowestl_ext_pos, traj_basename);
                strcat(source_lowestl_file, suffix);
            } else {
                // Fallback: construct from scratch
                snprintf(source_lowestl_file, sizeof(source_lowestl_file), "data/%s%s",
                        traj_basename, strrchr(g_extend_file_source, '_') ? strrchr(g_extend_file_source, '_') : "");
            }

            char target_lowestl_file[512];
            strncpy(target_lowestl_file, target_filename, sizeof(target_lowestl_file) - 1);
            target_lowestl_file[sizeof(target_lowestl_file) - 1] = '\0';
            lowestl_ext_pos = strstr(target_lowestl_file, "all_particle_data");
            if (lowestl_ext_pos) {
                // Save the suffix that comes after "all_particle_data"
                char suffix[256];
                strncpy(suffix, lowestl_ext_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
                // Now replace and append
                strcpy(lowestl_ext_pos, traj_basename);
                strcat(target_lowestl_file, suffix);
            } else {
                // Fallback: construct from scratch
                snprintf(target_lowestl_file, sizeof(target_lowestl_file), "data/%s%s",
                        traj_basename, strrchr(target_filename, '_') ? strrchr(target_filename, '_') : "");
            }

            FILE *source_lowestl = fopen(source_lowestl_file, "rb");
            if (source_lowestl) {
                FILE *target_lowestl = fopen(target_lowestl_file, "wb");
                if (target_lowestl) {
                    // Copy the file
                    char *lowestl_buffer = (char *)malloc(1048576); // 1MB buffer
                    if (lowestl_buffer) {
                        size_t lowestl_bytes;
                        while ((lowestl_bytes = fread(lowestl_buffer, 1, 1048576, source_lowestl)) > 0) {
                            if (fwrite(lowestl_buffer, 1, lowestl_bytes, target_lowestl) != lowestl_bytes) {
                                fprintf(stderr, "WARNING: Error copying %s file\n", traj_basename);
                                break;
                            }
                        }
                        free(lowestl_buffer);
                    }
                    fclose(target_lowestl);
                } else {
                    fprintf(stderr, "WARNING: Could not create target %s file %s\n", traj_basename, target_lowestl_file);
                }
                fclose(source_lowestl);
            }
        }

        // Copy double buffer files if they exist
        {
            // Extract suffix from source file
            char suffix[256] = "";
            const char *data_pos = strstr(g_extend_file_source, "all_particle_data");
            if (data_pos) {
                strncpy(suffix, data_pos + strlen("all_particle_data"), sizeof(suffix) - 1);
                suffix[sizeof(suffix) - 1] = '\0';
            }

            // Construct source and target filenames for double buffer data
            char source_dbl_data[512], target_dbl_data[512];
            char source_dbl_phi[512], target_dbl_phi[512];

            snprintf(source_dbl_data, sizeof(source_dbl_data),
                    "data/double_buffer_all_particle_data%s", suffix);
            snprintf(source_dbl_phi, sizeof(source_dbl_phi),
                    "data/double_buffer_all_particle_phi%s", suffix);

            snprintf(target_dbl_data, sizeof(target_dbl_data),
                    "data/double_buffer_all_particle_data%s.dat", g_file_suffix);
            snprintf(target_dbl_phi, sizeof(target_dbl_phi),
                    "data/double_buffer_all_particle_phi%s.dat", g_file_suffix);

            // Copy double buffer data file
            FILE *source_dbl_d = fopen(source_dbl_data, "rb");
            if (source_dbl_d) {
                FILE *target_dbl_d = fopen(target_dbl_data, "wb");
                if (target_dbl_d) {
                    char *buffer = (char *)malloc(1048576); // 1MB buffer
                    if (buffer) {
                        size_t bytes_read;
                        while ((bytes_read = fread(buffer, 1, 1048576, source_dbl_d)) > 0) {
                            if (fwrite(buffer, 1, bytes_read, target_dbl_d) != bytes_read) {
                                fprintf(stderr, "WARNING: Error copying double buffer data file\n");
                                break;
                            }
                        }
                        free(buffer);
                    }
                    fclose(target_dbl_d);
                } else {
                    fprintf(stderr, "WARNING: Could not create target double buffer data file\n");
                }
                fclose(source_dbl_d);

                // Copy double buffer phi file
                FILE *source_dbl_p = fopen(source_dbl_phi, "rb");
                if (source_dbl_p) {
                    FILE *target_dbl_p = fopen(target_dbl_phi, "wb");
                    if (target_dbl_p) {
                        char *buffer = (char *)malloc(1048576); // 1MB buffer
                        if (buffer) {
                            size_t bytes_read;
                            while ((bytes_read = fread(buffer, 1, 1048576, source_dbl_p)) > 0) {
                                if (fwrite(buffer, 1, bytes_read, target_dbl_p) != bytes_read) {
                                    fprintf(stderr, "WARNING: Error copying double buffer phi file\n");
                                    break;
                                }
                            }
                            free(buffer);
                        }
                        fclose(target_dbl_p);
                    }
                    fclose(source_dbl_p);
                }
            }
        }

        // Now update apd_filename to use the target for the rest of the simulation
        strncpy(apd_filename, target_filename, sizeof(apd_filename) - 1);
        apd_filename[sizeof(apd_filename) - 1] = '\0';
        
        // Switch to sim-restart mode to continue from where source left off
        g_doSimRestart = 1;
        // Keep g_doSimExtend true to preserve extend mode origin flag
        
        // Set up restart parameters
        restart_completed_snapshots = (int)source_snapshots;
        restart_initial_time = (restart_completed_snapshots - 1) * dtwrite * dt;
        restart_initial_nwrite = restart_completed_snapshots - 1;  // Use last snapshot INDEX, not count

        // Note: g_restart_initial_timestep set from src_Ntimes during filename parsing
        // Represents the last completed timestep from the source run

        // Set restart_source_file to the TARGET (which has the copied data)
        strncpy(restart_source_file, target_filename, sizeof(restart_source_file) - 1);
        
        // Adjust Ntimes for remaining work
        original_Ntimes = Ntimes;
        // Use g_restart_initial_timestep which has the actual completed timesteps
        int completed_timesteps_j = g_restart_initial_timestep;
        Ntimes = Ntimes - completed_timesteps_j + 1;  // +1 accounts for skipped j=0

        // Mark restart mode active for IC loading
        g_restart_mode_active = 1;
    }

    // Load existing energy diagnostic data if in restart mode (NOT extend mode - that copies the file)
    if (g_doAllParticleData && g_time_snapshots != NULL && 
        g_restart_mode_active && !g_doSimExtend && restart_completed_snapshots > 0) {
        
        char energy_file[256];
        get_suffixed_filename("data/total_energy_vs_time.dat", 1, energy_file, sizeof(energy_file));
        
        int loaded_snapshots = load_existing_energy_diagnostics(energy_file, restart_completed_snapshots);
        if (loaded_snapshots > 0) {
            g_energy_snapshots_loaded = loaded_snapshots;
            log_message("INFO", "Loaded %d existing energy snapshots for restart continuity", loaded_snapshots);
        } else {
            log_message("WARNING", "Could not load existing energy data from %s, starting fresh", energy_file);
            g_energy_snapshots_loaded = 0;
        }
    }

    /** @brief Allocate primary block storage arrays (float/int for memory efficiency).
     *         These store data for `snapshot_block_size` steps, indexed [step_in_block * npts + final_rank_id].
     *         Freed later via `cleanup_all_particle_data()`. */
    L_block = (float *)malloc((size_t)npts * snapshot_block_size * sizeof(float));    // Angular momentum
    Rank_block = (int *)malloc((size_t)npts * snapshot_block_size * sizeof(int));     // Particle rank (sorted index) at that step
    R_block = (float *)malloc((size_t)npts * snapshot_block_size * sizeof(float));    // Radius
    Vrad_block = (float *)malloc((size_t)npts * snapshot_block_size * sizeof(float)); // Radial velocity
    phi_block = (float *)malloc((size_t)npts * snapshot_block_size * sizeof(float)); // Phi angle
    scatter_count_block = (int *)calloc((size_t)npts * snapshot_block_size, sizeof(int)); // Scatter counts (initialized to 0)
    ID_block = (int *)malloc((size_t)npts * snapshot_block_size * sizeof(int));     // Particle IDs
    // Check allocation results
    if (!L_block || !Rank_block || !R_block || !Vrad_block || !phi_block || !scatter_count_block || !ID_block) {
        fprintf(stderr, "Error: Failed to allocate block storage arrays.\n");
        free(L_block); free(Rank_block); free(R_block); free(Vrad_block); free(phi_block); free(scatter_count_block); free(ID_block); // Free any that were allocated
        L_block = NULL; Rank_block = NULL; R_block = NULL; Vrad_block = NULL; phi_block = NULL; scatter_count_block = NULL; ID_block = NULL; // Prevent double free in cleanup
        CLEAN_EXIT(1);
    }

    /**
     * @brief RESTART MODE DATA VERIFICATION block (`all_particle_data.dat` check).
     * @details If running in restart mode (`g_doRestart`), this block checks if the main particle
     *          data output file (`all_particle_data<suffix>.dat`) already exists and is non-empty.
     *          If it exists and contains data, the flag `skip_simulation` is set to 1. This flag
     *          will cause the main timestepping loop to be bypassed entirely, allowing the program
     *          to proceed directly to the post-simulation snapshot analysis phase.
     * @note This check happens *before* the main loop. A more detailed check using
     *       `find_last_processed_snapshot` occurs *later* if snapshot analysis needs restarting.
     * @see skip_simulation
     * @see find_last_processed_snapshot
     */
    if (g_doRestart)
    {
        /** @brief Check for existence and size of the particle data file. */
        FILE *check_file = fopen(apd_filename, "rb");
        if (check_file)
        {
            /** @brief Determine file size by seeking to end and getting position. */
            fseek(check_file, 0, SEEK_END);
            long file_size = ftell(check_file);
            fclose(check_file);

            if (file_size > 0)
            {
                /** @brief Valid data file found - enable simulation phase bypass. */
                char human_size[32];
                format_file_size(file_size, human_size, sizeof(human_size));
                printf("Restart mode: Found existing all_particle_data file '%s' (%s).\n",
                       apd_filename, human_size);
                skip_simulation = 1;
                
                /** @brief Display appropriate message based on particle count comparison. */
                if (npts == npts_initial) {
                    printf("Skipping simulation phase and proceeding to post-processing. "
                           "Initial condition generation already performed, skipping these steps.\n\n");
                } else {
                    printf("Skipping simulation phase and proceeding to post-processing. "
                           "Initial condition generation and tidal stripping already performed, "
                           "skipping these steps.\n\n");
                }
            }
            else {
                /** @brief File exists but contains no data - simulation required. */
                printf("Restart mode: Found empty all_particle_data file '%s'. Will run simulation.\n", apd_filename);
            }
        }
        else {
            /** @brief File not found - simulation required. */
            printf("Restart mode: all_particle_data file '%s' not found. Will run simulation.\n", apd_filename);
        }
    } // End restart check block

    /**
     * @brief OUTPUT FILE INITIALIZATION block (`all_particle_data.dat`).
     * @details Creates (or overwrites) an empty binary file `all_particle_data<suffix>.dat`
     *          if saving all particle data (`g_doAllParticleData` is true) AND the simulation
     *          is *not* being skipped (`skip_simulation` is false). This file will be appended to
     *          incrementally during the simulation timestepping loop via block writes.
     * @see append_all_particle_data_chunk_to_file
     * @see apd_filename
     * @see g_doAllParticleData
     * @see skip_simulation
     */
    if (g_doAllParticleData && !skip_simulation)
    {
        // Skip file initialization in restart mode (file already exists)
        if (!g_restart_mode_active) {
            // Create/truncate the output file in binary write mode.
            FILE *fapd = fopen(apd_filename, "wb");
            if (!fapd) {
                fprintf(stderr, "Error: cannot create all_particle_data output file %s\n", apd_filename);
                CLEAN_EXIT(1);
            }
            fclose(fapd); // Close immediately, file is now ready for appending.

            // Create/truncate the phi output file in binary write mode
            char phi_filename[1024];
            snprintf(phi_filename, sizeof(phi_filename), "data/all_particle_phi%s.dat", g_file_suffix);
            FILE *fphi = fopen(phi_filename, "wb");
            if (!fphi) {
                fprintf(stderr, "Error: cannot create all_particle_phi output file %s\n", phi_filename);
                CLEAN_EXIT(1);
            }
            fclose(fphi); // Close immediately, file is now ready for appending.

            // Create/truncate the scatter count output file in binary write mode
            char scatter_filename[1024];
            snprintf(scatter_filename, sizeof(scatter_filename), "data/all_particle_scatter_counts%s.dat", g_file_suffix);
            FILE *fscat = fopen(scatter_filename, "wb");
            if (!fscat) {
                fprintf(stderr, "Error: cannot create all_particle_scatter_counts output file %s\n", scatter_filename);
                CLEAN_EXIT(1);
            }
            fclose(fscat); // Close immediately, file is now ready for appending.

            // Create/truncate the particle IDs output file in binary write mode
            char ids_filename[1024];
            snprintf(ids_filename, sizeof(ids_filename), "data/all_particle_ids%s.dat", g_file_suffix);
            FILE *fids = fopen(ids_filename, "wb");
            if (!fids) {
                fprintf(stderr, "Error: cannot create all_particle_ids output file %s\n", ids_filename);
                CLEAN_EXIT(1);
            }
            fclose(fids); // Close immediately, file is now ready for appending.

            printf("Initialized empty file for all particle data: %s\n", apd_filename);
            printf("Initialized empty file for all particle phi data: %s\n", phi_filename);
            printf("Initialized empty file for all particle scatter counts: %s\n", scatter_filename);
            printf("Initialized empty file for all particle IDs: %s\n", ids_filename);
        } else {
            // Existing all_particle_data file appended in restart mode
        }
        
        // Calculate and display expected file size
        long long expected_size = (long long)total_writes * (long long)npts * 16LL; // 16 bytes per particle record
        double size_gb = expected_size / (1024.0 * 1024.0 * 1024.0);
        double size_mb = expected_size / (1024.0 * 1024.0);
        double size_kb = expected_size / 1024.0;
        
        if (size_gb >= 1.0) {
            printf("All particle data file requires: %.1f GB (%lld bytes)\n", size_gb, expected_size);
        } else if (size_mb >= 1.0) {
            printf("All particle data file requires: %.1f MB (%lld bytes)\n", size_mb, expected_size);
        } else if (size_kb >= 1.0) {
            printf("All particle data file requires: %.1f KB (%lld bytes)\n", size_kb, expected_size);
        } else {
            printf("All particle data file requires: %lld bytes\n", expected_size);
        }
        
        // Calculate and display expected snapshot file sizes
        // Each snapshot has 2 files: unsorted (28 bytes/particle) and sorted (32 bytes/particle)
        long long snapshot_size = (long long)npts * (28LL + 32LL); // Total per snapshot pair
        long long total_snapshot_size = snapshot_size * (long long)noutsnaps;
        double snap_size_gb = total_snapshot_size / (1024.0 * 1024.0 * 1024.0);
        double snap_size_mb = total_snapshot_size / (1024.0 * 1024.0);
        double snap_size_kb = total_snapshot_size / 1024.0;
        
        printf("%d time snapshot files will require: ", noutsnaps);
        if (snap_size_gb >= 1.0) {
            printf("%.1f GB (%lld bytes)\n", snap_size_gb, total_snapshot_size);
        } else if (snap_size_mb >= 1.0) {
            printf("%.1f MB (%lld bytes)\n", snap_size_mb, total_snapshot_size);
        } else if (snap_size_kb >= 1.0) {
            printf("%.1f KB (%lld bytes)\n", snap_size_kb, total_snapshot_size);
        } else {
            printf("%lld bytes\n", total_snapshot_size);
        }
        
        // Calculate and display total disk space
        long long total_disk_space = expected_size + total_snapshot_size;
        double total_gb = total_disk_space / (1024.0 * 1024.0 * 1024.0);
        double total_mb = total_disk_space / (1024.0 * 1024.0);
        double total_kb = total_disk_space / 1024.0;
        
        printf("Total disk space required: ");
        if (total_gb >= 1.0) {
            printf("%.1f GB (%lld bytes)\n", total_gb, total_disk_space);
        } else if (total_mb >= 1.0) {
            printf("%.1f MB (%lld bytes)\n", total_mb, total_disk_space);
        } else if (total_kb >= 1.0) {
            printf("%.1f KB (%lld bytes)\n", total_kb, total_disk_space);
        } else {
            printf("%lld bytes\n", total_disk_space);
        }
        
        // Check available disk space
        long long available_space = get_available_disk_space("data/");
        if (available_space > 0) {
            double avail_gb = available_space / (1024.0 * 1024.0 * 1024.0);
            double avail_mb = available_space / (1024.0 * 1024.0);
            double avail_kb = available_space / 1024.0;
            
            printf("Available disk space: ");
            if (avail_gb >= 1.0) {
                printf("%.1f GB (%lld bytes)\n", avail_gb, available_space);
            } else if (avail_mb >= 1.0) {
                printf("%.1f MB (%lld bytes)\n", avail_mb, available_space);
            } else if (avail_kb >= 1.0) {
                printf("%.1f KB (%lld bytes)\n", avail_kb, available_space);
            } else {
                printf("%lld bytes\n", available_space);
            }
            
            // Check if within 5% of capacity or insufficient
            double usage_after = (double)(available_space - total_disk_space) / (double)available_space;
            
            if (available_space < total_disk_space) {
                // Insufficient space
                fprintf(stderr, "\nError: Insufficient disk space!\n");
                fprintf(stderr, "Required: %.1f GB\n", total_gb);
                fprintf(stderr, "Available: %.1f GB\n", avail_gb);
                fprintf(stderr, "Shortfall: %.1f GB\n", total_gb - avail_gb);
                CLEAN_EXIT(1);
            } else if (usage_after < 0.05) {
                // Within 5% of capacity after simulation
                printf("\nWarning: Simulation will use %.1f%% of available disk space!\n", 
                       (100.0 * total_disk_space / available_space));
                printf("After simulation: %.1f GB free (%.1f%% remaining)\n", 
                       (available_space - total_disk_space) / (1024.0 * 1024.0 * 1024.0),
                       usage_after * 100.0);
                
                if (!prompt_yes_no("Continue")) {
                    printf("Aborting simulation.\n");
                    CLEAN_EXIT(0);
                }
            }
        } else {
            fprintf(stderr, "Warning: Could not determine available disk space.\n");
            if (!prompt_yes_no("Continue without disk space check")) {
                printf("Aborting simulation.\n");
                CLEAN_EXIT(0);
            }
        }
        
        printf("\n");
        
        /** @brief Display initial simulation progress. */
        if (g_restart_mode_active) {
            // Restart continues from saved state
        } else {
            printf("0%% complete, timestep 0/%d, time=0.0000 Myr, elapsed=0.00 s\n", Ntimes);
        }
    }

    /**
     * @brief BLOCK STORAGE INITIALIZATION block (Initial L).
     * @details Copies initial angular momentum values (`L_i_arr`, indexed by final_rank_id)
     *          into the first timestep slot (index 0 implicitly) of the `L_block` storage array.
     *          Converts from double precision (`L_i_arr`) to single precision (`L_block`).
     *          This prepares the block storage for the first chunk of simulation data.
     * @note L_block indexing: `[step_in_block * npts + final_rank_id]`.
     */
    for (i = 0; i < npts; i++) // Loop over final_rank_id 'i'
    {
        double l_val = g_L_init_vals[i];
        float lf = (float)l_val;
        // Store initial L in the slot for step 0 for this final_rank_id
        L_block[i] = lf; // Index 'i' corresponds to final_rank_id for the first block slot (step 0)

        // Store initial phi in the slot for step 0
        // Convert cos(phi) and sin(phi) to actual phi angle using atan2
        double phi_val = atan2(particles[6][i], particles[5][i]);
        phi_block[i] = (float)phi_val;
    }

    /**
     * @brief Store initial (t=0) approximate energy for the debug particle.
     * @details If `g_doDebug` is enabled, this block calculates the theoretical
     *          approximate energy `E = Psi - KE` for the particle with final rank ID
     *          `DEBUG_PARTICLE_ID` based on its *initial* state (before any
     *          timesteps) and stores it using `store_debug_approxE` at snapshot index 0.
     * @note `DEBUG_PARTICLE_ID` refers to the final rank ID after stripping and remapping.
     *       Retrieves the initial state using this ID as the index into the `particles`
     *       array before the first sort in the main loop.
     */
    if (g_doDebug)
    {
        int debug_id = DEBUG_PARTICLE_ID;
        if (debug_id >= 0 && debug_id < npts) {
            // Retrieve initial state using final_rank_id as index into initial particles array
            double r0 = particles[0][debug_id];
            double v0 = particles[1][debug_id]; // This is v_rad
            double l0 = particles[2][debug_id];

            /** @brief Evaluate theoretical Psi(r) at initial radius using initial spline. */
            double psi_val = evaluatespline(splinePsi, Psiinterp, r0) * VEL_CONV_SQ;

            /** @brief Calculate initial approximate energy E = Psi - KE. */
            double E_approx0 = psi_val - 0.5 * (v0 * v0 + (l0 * l0) / (r0 * r0));
            double time0 = 0.0;

            /** @brief Store initial energy in debug arrays at snapshot index 0. */
            store_debug_approxE(0, E_approx0, time0);
        } else {
             fprintf(stderr, "Warning: Invalid DEBUG_PARTICLE_ID %d (must be 0 <= ID < %d)\n",
                     debug_id, npts);
        }
    }

    static int nwrite_total = 0; ///< Counter for number of dtwrite-interval writes performed.

    // If restarting, initialize counters with restart values
    if (g_restart_mode_active) {
        nwrite_total = restart_initial_nwrite;
        // Initialize time to the NEXT timestep after loaded state to avoid duplicate time recording
        // The loaded particles are at state corresponding to restart_initial_time
        // First evolution step (j=0) evolves from loaded state; time recorded as restart_initial_time + dt
        // with time = restart_initial_time + dt
        time = restart_initial_time + dt;
    }

    /**
     * @brief Perform initial sort of particles by radius before time integration.
     * @details Uses `quadsort` for this initial sort. Executed only by the master thread
     *          via `#pragma omp single`. After this sort, the index `i` of `particles[:][i]`
     *          corresponds to its rank based on radius.
     */
    #pragma omp single
    {
        sort_particles_with_alg(particles, npts, "quadsort");
    }

    /**
     * @brief Build inverse_map after initial sort to enable particle lookups by original ID.
     * @details After sorting, particles[3][idx] contains the original ID. The mapping
     *          inverse_map[orig_id] = idx enables lookup of particles by their original ID.
     */
    #pragma omp parallel for default(shared) schedule(static)
    for (int idx = 0; idx < npts; idx++) {
        int orig_id = (int)particles[3][idx];
        inverse_map[orig_id] = idx;
    }

    /**
     * @brief Record initial conditions for lowest_l_trajectories at timestep 0.
     * @details Stores the initial radius, energy, and angular momentum values for the selected
     *          lowest-L particles after the initial sort. This ensures that timestep 0 data
     *          matches what gets written to all_particle_data snapshot 0.
     * @note This must be done AFTER the initial sort and AFTER populating inverse_map,
     *       enabling lookup of sorted particle positions via inverse_map[chosen[p]].
     */
    if (!skip_simulation && nlowest > 0 && !g_restart_mode_active) {
        // Record initial time in buffer
        if (g_trajectory_buffer_index == 0) {
            g_time_buf[0] = 0.0;

            // Record initial conditions for lowest-L particles
            // Use inverse_map to find particles by their original ID after sorting
            for (int p = 0; p < nlowest; p++) {
                int particle_id = chosen[p];
                int idx = inverse_map[particle_id];
                double rr = particles[0][idx];
                double vrad = particles[1][idx];
                double ell = particles[2][idx];

                // Clamp radius to valid range
                if (rr < 0.0) rr = 0.0000000001;
                if (rr > rmax) rr = rmax;

                // Calculate initial energy
                double Psi_val = evaluatespline(splinePsi, Psiinterp, rr) * VEL_CONV_SQ;
                double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

                // Store in buffers at index 0
                g_lowestL_r_buf[p][0] = rr;
                g_lowestL_E_buf[p][0] = E_rel;
                g_lowestL_L_buf[p][0] = ell;
            }

            // Buffer index remains at 0 until all trajectory types are recorded
        }
    }

    /**
     * @brief Main Simulation Timestepping Loop.
     * @details This loop iterates from j = 0 to Ntimes + dtwrite - 1.
     *          In each iteration `j` (representing timestep from \f$t_j\f$ to \f$t_{j+1}\f$):
     *          1. The appropriate N-body integration method (selected by `method_select`)
     *             is called to advance all particle positions and velocities over `dt` due
     *             to gravitational forces. This typically involves sorting particles by radius
     *             for rank-based force calculation and updating an inverse map from original
     *             particle ID to current sorted rank.
     *          2. If SIDM is enabled (`g_enable_sidm_scattering`), the `handle_sidm_step()`
     *             function is called to apply stochastic SIDM scattering to particle pairs,
     *             further modifying their velocities and angular momenta.
     *          3. Simulation time `time` is incremented by `dt`.
     *          4. Trajectory and energy data for selected particles (low-ID and low-L) are recorded.
     *          5. If it is a `dtwrite` interval, a block of full particle data (rank, R, Vrad, L)
     *             is stored and potentially appended to `all_particle_data.dat`. Debug energy
     *             for `DEBUG_PARTICLE_ID` is also calculated and stored if it is a snapshot step.
     *          6. Progress is printed to the console at 5% intervals.
     *          This entire loop is skipped if `skip_simulation` is true (e.g., in restart mode
     *          where only post-processing of existing `all_particle_data.dat` is required).
     */

    if (!skip_simulation)
    {
        for (int j = 0; j < Ntimes; j++) // Main time loop
        {
            int current_step;
            // Calculate restart offset for display and write operations
            // In restart mode, restart_offset represents the last completed timestep
            int restart_offset = g_restart_mode_active ? (restart_initial_nwrite * dtwrite) : 0;

            // In restart mode, skip j=0 evolution and trajectory recording
            // Particles loaded at restart timestep; evolution begins from j=1
            if (g_restart_mode_active && j == 0) {
                continue;
            }

            // Method_select = 1; Flag now.
            if (method_select == 0)
            {
/****************************/
// EULER METHOD
/****************************/
#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier

#pragma omp parallel for default(shared) schedule(static)
                {
                    for (int idx = 0; idx < npts; idx++)
                    {
                        int orig_id = (int)particles[3][idx];
                        inverse_map[orig_id] = idx;
                    }
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];
                    double drdt = vrad;

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[0][i] += drdt * dt;
                    particles[1][i] += dvdt * dt;
                }

                // SIDM scattering: profile-aware scale radius selection and execution mode handling
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

                // Record trajectory data at every timestep into buffer
#pragma omp single
                {
                    // Store time for this step and advance buffer index
                    if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                        g_time_buf[g_trajectory_buffer_index] = time;
                    }
                    g_trajectory_buffer_index++;

                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;
                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;
                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                    double l_current = ell;
                    // Buffer index incremented in preceding #pragma omp single block
                    if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                        g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                        g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                        g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                        g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                        g_L_buf[p][g_trajectory_buffer_index - 1] = l_current;
                    }
                }
            }
            else if (method_select == 1)
            {
                /****************************/
                // LEAPFROG METHOD (POSITION HALF STEP)
                /****************************/

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    particles[0][i] += particles[1][i] * (dt / 2.0);
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier
#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] += dvdt * dt;
                    particles[0][i] += particles[1][i] * (dt / 2.0);
                }

                // SIDM scattering after leapfrog drift completion
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

                // Record trajectory data at every timestep into buffer
#pragma omp single
                    {
                        // Store time for this step and advance buffer index
                        if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                            g_time_buf[g_trajectory_buffer_index] = time;
                        }
                        g_trajectory_buffer_index++;
                    }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                    for (int p = 0; p < upper_npts_num_traj; p++)
                    {
                        int idx = inverse_map[p];
                        double rr = particles[0][idx];
                        double vrad = particles[1][idx];
                        double ell = particles[2][idx];
                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        Psi_val *= VEL_CONV_SQ;
                        double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                        double mu_val = vrad / vtot;
                        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                        double l_current = ell;
                        if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                            g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                            g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                            g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                            g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                            g_L_buf[p][g_trajectory_buffer_index - 1] = l_current;
                        }
                    }

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
            }
            else if (method_select == 2)
            {

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] = vrad + 0.5 * dvdt * dt;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * dt;
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }
#pragma omp barrier

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] = vrad + 0.5 * dvdt * dt;
                }

                // SIDM scattering after velocity half-step completion
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

                // Record trajectory data at every timestep into buffer
#pragma omp single
                    {
                        // Store time for this step and advance buffer index
                        if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                            g_time_buf[g_trajectory_buffer_index] = time;
                        }
                        g_trajectory_buffer_index++;
                    }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                    for (int p = 0; p < upper_npts_num_traj; p++)
                    {
                        int idx = inverse_map[p];
                        double rr = particles[0][idx];
                        double vrad = particles[1][idx];
                        double ell = particles[2][idx];
                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        Psi_val *= VEL_CONV_SQ;
                        double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                        double mu_val = vrad / vtot;
                        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                        double l_current = ell;
                        if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                            g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                            g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                            g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                            g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                            g_L_buf[p][g_trajectory_buffer_index - 1] = l_current;
                        }
                    }

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
            }
            else if (method_select == 3)
            {
                /**
                 * @brief Full-step adaptive leapfrog integration.
                 * @details Performs single step integration from (r_n, v_n) to (r_{n+1}, v_{n+1})
                 *          using adaptive timestep control and the doAdaptiveFullLeap function.
                 */

                double velocity_tol = 1.0e-5;
                double radius_tol = 1.0e-5;
                // Int max_subdiv = 8383608;
                int max_subdiv = 1;
                int out_type = 0;

                /**
                 * @brief Radial sorting phase - organize particles by radius.
                 * @details Sorting improves cache locality and force calculation efficiency.
                 */
#pragma omp single
                {
                    sort_particles(particles, npts);
                }
                /**
                 * @brief Main integration loop - adaptive leapfrog update for each particle.
                 * @details Each particle is advanced independently using adaptive timestepping.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double v = particles[1][i];
                    double ell = particles[2][i];

                    // One full step => h = dt.
                    double r_new, v_new;
                    doAdaptiveFullLeap(
                        i, npts,
                        r, v,
                        ell,
                        dt,
                        radius_tol, velocity_tol,
                        max_subdiv,
                        G_CONST,
                        out_type,
                        &r_new, &v_new);

                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

                /**
                 * @note Optional re-sorting after integration disabled for performance.
                 *       Enable via: #pragma omp single { sort_particles(particles, npts); }
                 */

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

                // SIDM scattering after adaptive leapfrog completion
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

                /**
                 * @brief Timestep wrap-up phase - update time and record particle states.
                 * @details Updates global time counter and records trajectory information
                 *          for selected particles.
                 */
                // Record trajectory data at every timestep into buffer
#pragma omp single
                    {
                        // Store time for this step and advance buffer index
                        if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                            g_time_buf[g_trajectory_buffer_index] = time;
                        }
                        g_trajectory_buffer_index++;
                    }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                    for (int p = 0; p < upper_npts_num_traj; p++)
                    {
                        int idx = inverse_map[p];
                        double rr = particles[0][idx];
                        double vrad = particles[1][idx];
                        double ell = particles[2][idx];

                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        Psi_val *= VEL_CONV_SQ;

                        double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                        double mu_val = vrad / vtot;

                        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

                        if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                            g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                            g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                            g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                            g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                            g_L_buf[p][g_trajectory_buffer_index - 1] = ell;
                        }
                    }

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
            }
            else if (method_select == 4)
            {
                /**
                 * @brief Hybrid integration with adaptive method selection.
                 * @details Uses Levi-Civita regularization for close encounters (r < r_crit)
                 *          and standard leapfrog otherwise. Radius threshold r_crit is 
                 *          dynamically calculated for each particle.
                 */

                double velocity_tol = 1.0e-8;
                double radius_tol = 1.0e-8;
                int max_subdiv = 4096 * 4096;
                int out_type = 2;
                int N_taumin = 1000;

                double alpha_param = 0.05;

#pragma omp single
                {
                    sort_particles(particles, npts);
                }
                /**
                 * @brief Main integration loop - method selection based on orbital parameters.
                 * @details Dynamically selects between standard leapfrog and Levi-Civita
                 *          regularization based on particle's radius and angular momentum.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double v = particles[1][i];
                    double ell = particles[2][i];

                    // Define critical radius r_crit for switching integration method:
                    // r_crit = alpha_param * (ell^2) / (G * M(r))
                    double r_crit = 0.0;
                    if (ell != 0.0)
                    {
                        double M_enc = ((double)i / (double)npts) * g_active_halo_mass;
                        double gravPart = (VEL_CONV_SQ * G_CONST) * M_enc;
                        r_crit = (ell * ell) * (alpha_param) / (gravPart);
                    }
                    else
                    {
                        r_crit = 0.0;
                    }

                    double r_new, v_new;

                    if (((r > 1.0e-30) && (r < r_crit)) || (r < r_crit_min))
                    {
                        doLeviCivitaLeapfrog(
                            i, npts,
                            r, v,
                            ell,
                            dt,
                            N_taumin,
                            G_CONST,
                            &r_new, &v_new);
                    }
                    else
                    {
                        doAdaptiveFullLeap(
                            i, npts,
                            r, v,
                            ell,
                            dt,
                            radius_tol, velocity_tol,
                            max_subdiv,
                            G_CONST,
                            out_type,
                            &r_new, &v_new);
                    }

                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

                // SIDM scattering after hybrid integrator completion (Levi-Civita/adaptive leapfrog)
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

                // Record trajectory data at every timestep into buffer
#pragma omp single
                    {
                        // Store time for this step and advance buffer index
                        if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                            g_time_buf[g_trajectory_buffer_index] = time;
                        }
                        g_trajectory_buffer_index++;
                    }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                    for (int p = 0; p < upper_npts_num_traj; p++)
                    {
                        int idx = inverse_map[p];
                        double rr = particles[0][idx];
                        double vrad = particles[1][idx];
                        double ell = particles[2][idx];

                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        Psi_val *= VEL_CONV_SQ;

                        double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                        double mu_val = vrad / vtot;

                        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

                        if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                            g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                            g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                            g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                            g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                            g_L_buf[p][g_trajectory_buffer_index - 1] = ell;
                        }
                    }

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
            }
            else if (method_select == 5)
            {
                double velocity_tol = 1.0e-7;
                double radius_tol = 1.0e-7;
                int max_subdiv = 4096 * 4096 * 16;
                int out_type = 2;
                int N_taumin = 10;
                double alpha_param = 0.05;

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

                // SIDM scattering before adaptive orbital integration with Levi-Civita regularization
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

#pragma omp parallel for default(shared) schedule(static)
                for (int i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double v = particles[1][i];
                    double ell = particles[2][i];

                    // Calculate r_crit, using special handling for particle i=0 (M_enc=0) to avoid division by zero.
                    double r_crit = 0.0;
                    if (fabs(ell) > 1.0e-30)
                    {
                        double M_enc;
                        if (i == 0)
                        {
                            // Special handling for i=0 to avoid zero mass in denominator
                            M_enc = 0.1 * (1.0 / (double)npts) * g_active_halo_mass;
                        }
                        else
                        {
                            M_enc = ((double)i / (double)npts) * g_active_halo_mass;
                        }
                        double gravPart = (VEL_CONV_SQ * G_CONST) * M_enc;
                        r_crit = (ell * ell) * alpha_param / gravPart;
                    }

                    double r_new, v_new;
                    if (r > 1.0e-30 && r < r_crit) // Switch based on critical radius
                    {
                        doAdaptiveFullLeviCivita(
                            i, npts, r, v, ell, dt, N_taumin,
                            radius_tol, velocity_tol, max_subdiv,
                            G_CONST, out_type, &r_new, &v_new);
                    }
                    else
                    {
                        doAdaptiveFullLeap(
                            i, npts, r, v, ell, dt,
                            radius_tol, velocity_tol, max_subdiv,
                            G_CONST, out_type, &r_new, &v_new);
                    }
                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

                // Record trajectory data at every timestep into buffer
#pragma omp single
                    {
                        // Store time for this step and advance buffer index
                        if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                            g_time_buf[g_trajectory_buffer_index] = time;
                        }
                        g_trajectory_buffer_index++;
                    }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                    for (int p = 0; p < upper_npts_num_traj; p++)
                    {
                        int idx = inverse_map[p];
                        double rr = particles[0][idx];
                        double vr = particles[1][idx];
                        double ell = particles[2][idx];

                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        Psi_val *= VEL_CONV_SQ;

                        double vtot = sqrt(vr * vr + (ell * ell) / (rr * rr));
                        double mu_val = vr / vtot;
                        double E_rel = Psi_val - 0.5 * (vr * vr + (ell * ell) / (rr * rr));

                        if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                            g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                            g_velocities_buf[p][g_trajectory_buffer_index - 1] = vr;
                            g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                            g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                            g_L_buf[p][g_trajectory_buffer_index - 1] = ell;
                        }
                    }

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
            }
            else if (method_select == 6)
            {

                // Coefficients for 4th-order Forest-Ruth-Yoshida integrator (c1=c3).
                // Derived from: c1 = 1 / (2 - 2^(1/3)), c2 = 1 - 2*c1
                double c1 = 0.6756035959798289;
                double c2 = -0.3512071919596578; // = 1.0 - 2.0 * c1
                double c3 = c1;

                /**
                 * @brief STEP 1: Kick by (c1 * dt/2).
                 * @details Velocity update using initial position at timestep n.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);
                    particles[1][i] = vrad + 0.5 * c1 * dt * dvdt;
                }

                /**
                 * @brief STEP 2: Drift by (c1 * dt).
                 * @details Position update using the intermediate velocity v^*.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * (c1 * dt);
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier
#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

                /**
                 * @brief STEP 3: Kick by ((c1 + c2) * dt/2).
                 * @details Velocity update using position after first drift substep.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);
                    double coeff = 0.5 * (c1 + c2);
                    particles[1][i] = vrad + coeff * dt * dvdt;
                }

                /**
                 * @brief STEP 4: Drift by (c2 * dt).
                 * @details Position update using the intermediate velocity.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * (c2 * dt);
                }

                /**
                 * @brief STEP 5: Kick by ((c2 + c3) * dt/2).
                 * @details Velocity update using position after second drift substep.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    // Recompute acceleration at position after second drift
                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    // Combination of the remaining half of c2 and half of c3.
                    double coeff = 0.5 * (c2 + c3);
                    particles[1][i] = vrad + coeff * dt * dvdt;
                }

                /**
                 * @brief STEP 6: Drift by (c3 * dt).
                 * @details Final position update in this integration step.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * (c3 * dt);
                }

                /**
                 * @brief STEP 7: Kick by (c3 * dt/2).
                 * @details Final velocity update (half-kick) to complete the integration step.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);
                    particles[1][i] = vrad + 0.5 * c3 * dt * dvdt;
                }

                // SIDM scattering after 4th-order Yoshida symplectic integration
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

                // Record trajectory data at every timestep into buffer
#pragma omp single
                    {
                        // Store time for this step and advance buffer index
                        if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                            g_time_buf[g_trajectory_buffer_index] = time;
                        }
                        g_trajectory_buffer_index++;
                    }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                    for (int p = 0; p < upper_npts_num_traj; p++)
                    {
                        int idx = inverse_map[p];
                        double rr = particles[0][idx];
                        double vrad = particles[1][idx];
                        double ell = particles[2][idx];
                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        Psi_val *= VEL_CONV_SQ;
                        double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                        double mu_val = vrad / vtot;
                        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                        double l_current = ell;
                        if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                            g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                            g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                            g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                            g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                            g_L_buf[p][g_trajectory_buffer_index - 1] = l_current;
                        }
                    }

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
            }
            else if (method_select == 7)
            {
                /****************************/
                // RK4 METHOD (Fourth-Order Runge-Kutta)
                /****************************/

                double *r_orig_by_id = (double *)malloc(npts * sizeof(double));
                double *v_orig_by_id = (double *)malloc(npts * sizeof(double));
                double *k1r_by_id = (double *)malloc(npts * sizeof(double));
                double *k1v_by_id = (double *)malloc(npts * sizeof(double));
                double *k2r_by_id = (double *)malloc(npts * sizeof(double));
                double *k2v_by_id = (double *)malloc(npts * sizeof(double));
                double *k3r_by_id = (double *)malloc(npts * sizeof(double));
                double *k3v_by_id = (double *)malloc(npts * sizeof(double));
                double *k4r_by_id = (double *)malloc(npts * sizeof(double));
                double *k4v_by_id = (double *)malloc(npts * sizeof(double));
                double h = dt;

// Store original state by orig_id.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    r_orig_by_id[orig_id] = particles[0][i];
                    v_orig_by_id[orig_id] = particles[1][i];
                }

// K1 calculation.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = vrad;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    k1r_by_id[orig_id] = drdt;
                    k1v_by_id[orig_id] = dvdt;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r_mid = particles[0][i];
                    double v_mid = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = v_mid;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r_mid, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r_mid, ell);

                    k2r_by_id[orig_id] = drdt;
                    k2v_by_id[orig_id] = dvdt;
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier
#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r_mid = particles[0][i];
                    double v_mid = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = v_mid;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r_mid, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r_mid, ell);

                    k3r_by_id[orig_id] = drdt;
                    k3v_by_id[orig_id] = dvdt;
                }

// K4 calculation.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r_end = particles[0][i];
                    double v_end = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = v_end;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r_end, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r_end, ell);

                    k4r_by_id[orig_id] = drdt;
                    k4v_by_id[orig_id] = dvdt;
                }

// Final RK4 combination.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];

                    double r_new = r_orig_by_id[orig_id] + (h / 6.0) * (k1r_by_id[orig_id] + 2.0 * k2r_by_id[orig_id] + 2.0 * k3r_by_id[orig_id] + k4r_by_id[orig_id]);
                    double v_new = v_orig_by_id[orig_id] + (h / 6.0) * (k1v_by_id[orig_id] + 2.0 * k2v_by_id[orig_id] + 2.0 * k3v_by_id[orig_id] + k4v_by_id[orig_id]);

                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

                // SIDM scattering after RK4 state update completion
                double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0, g_current_timestep_scatter_counts);

                // Record trajectory data at every timestep into buffer
#pragma omp single
                    {
                        // Store time for this step and advance buffer index
                        if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                            g_time_buf[g_trajectory_buffer_index] = time;
                        }
                        g_trajectory_buffer_index++;
                    }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                    for (int p = 0; p < upper_npts_num_traj; p++)
                    {
                        int idx = inverse_map[p];
                        double rr = particles[0][idx];
                        double vrad = particles[1][idx];
                        double ell = particles[2][idx];
                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        Psi_val *= VEL_CONV_SQ;
                        double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                        double mu_val = vrad / vtot;
                        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                        double l_current = ell;
                        if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                            g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                            g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                            g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                            g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                            g_L_buf[p][g_trajectory_buffer_index - 1] = l_current;
                        }
                    }

#pragma omp single
                {
                    current_step = j + 1;
                    time += h;
                }

                // Free RK4 arrays.
                free(r_orig_by_id);
                free(v_orig_by_id);
                free(k1r_by_id);
                free(k1v_by_id);
                free(k2r_by_id);
                free(k2v_by_id);
                free(k3r_by_id);
                free(k3v_by_id);
                free(k4r_by_id);
                free(k4v_by_id);
            }
            else if (method_select == 8)
            {
                // Static variables for Adams-Bashforth 3rd Order (AB3) method
                static int ab3_bootstrap_done = 0;           ///< Flag indicating the completion status of the AB3 bootstrap phase (0=incomplete, 1=complete).
                static double **f_ab3_r = NULL;               ///< History array for \f$dr/dt\f$ derivatives. Indexed by `[history_slot (0..2)][original_particle_id]`. Slot 2 is most recent (\f$f_n\f$).
                static double **f_ab3_v = NULL;               ///< History array for \f$dv_{rad}/dt\f$ derivatives. Indexed by `[history_slot (0..2)][original_particle_id]`. Slot 2 is most recent (\f$f_n\f$).
                static double h_ab3_bootstrap_step;         ///< Timestep size (`dt`) used during the Euler steps of the bootstrap phase.

                // Adams-Bashforth coefficients for different orders
                static int ab3_num[3] = {23, -16, 5};        ///< Numerator coefficients for the AB3 formula: \f$y_{n+1} = y_n + (h/12) \sum (\text{ab3_num}_i \cdot f_{n-i})\f$.
                static int ab2_num[2] = {18, -6};           ///< Numerator coefficients for the AB2 formula (for comparison or fallback).
                static int ab3_den = 12;                     ///< Common denominator for the AB3 formula coefficients.

                static int bootstrap_euler_steps_needed = 2; ///< Number of full Euler steps (each of size \f$h_{bootstrap}\f$) required to generate the initial 3 derivative history points (\f$f_0, f_1, f_2\f$).

                // Allocate AB3 history arrays once.
                if (f_ab3_r == NULL)
                {
                    f_ab3_r = (double **)malloc(3 * sizeof(double *));
                    f_ab3_v = (double **)malloc(3 * sizeof(double *));
                    for (int hh = 0; hh < 3; hh++)
                    {
                        f_ab3_r[hh] = (double *)malloc(npts * sizeof(double));
                        f_ab3_v[hh] = (double *)malloc(npts * sizeof(double));
                    }
                    h_ab3_bootstrap_step = dt; // Step size equals dt.
                }

                // If bootstrap not done yet, do it ONCE to fill the 3-step derivative history.
                if (!ab3_bootstrap_done)
                {

                    // Bootstrap for AB3: Need to compute f0, f1, f2.
                    // This requires 2 Euler steps to get states y1, y2.
                    // sub_step = 0: calc f0 (from y0), store f_ab3_x[0]. Euler y0->y1. state is y1.
                    // sub_step = 1: calc f1 (from y1), store f_ab3_x[1]. Euler y1->y2. state is y2.
                    // sub_step = 2: calc f2 (from y2), store f_ab3_x[2]. NO Euler update. state is y2.
#pragma omp single
                    {
                        for (int sub_step = 0; sub_step <= bootstrap_euler_steps_needed; sub_step++)
                        {
                            sort_particles(particles, npts);
#pragma omp parallel for default(shared) schedule(static)
                            for (int idx = 0; idx < npts; idx++)
                            {
                                int orig_id = (int)particles[3][idx];
                                inverse_map[orig_id] = idx;
                            }
                            // Compute derivatives => store in f_ab3_r[sub_step], f_ab3_v[sub_step].
#pragma omp parallel for default(shared) schedule(static)
                            for (int i_eval = 0; i_eval < npts; i_eval++)
                            {
                                int orig_id = (int)particles[3][i_eval];
                                double rr = particles[0][i_eval];
                                double vrad = particles[1][i_eval];
                                double ell = particles[2][i_eval];

                                double drdt = vrad;
                                // Use the gravitational_force and effective_angular_force functions.
                                double force = gravitational_force(rr, i_eval, npts, G_CONST, g_active_halo_mass);
                                double dvdt = force + effective_angular_force(rr, ell);

                                f_ab3_r[sub_step][orig_id] = drdt;
                                f_ab3_v[sub_step][orig_id] = dvdt;
                            }

                            // If sub_step < bootstrap_euler_steps_needed (i.e., for sub_step 0 and 1),
                            // perform mini-substeps to advance particles to the next state with higher accuracy.
                            if (sub_step < bootstrap_euler_steps_needed)
                            {
                                double dt_mini = h_ab3_bootstrap_step / (double)NUM_MINI_SUBSTEPS_BOOTSTRAP;

#pragma omp parallel for default(shared) schedule(static)
                                for (int i_part = 0; i_part < npts; i_part++)
                                {
                                    // Each particle is evolved independently over NUM_MINI_SUBSTEPS_BOOTSTRAP
                                    double current_r_mini = particles[0][i_part];
                                    double current_vrad_mini = particles[1][i_part];
                                    double current_ell_mini = particles[2][i_part]; // Angular momentum (constant)
                                    
                                    // Perform NUM_MINI_SUBSTEPS_BOOTSTRAP mini-steps
                                    for (int m = 0; m < NUM_MINI_SUBSTEPS_BOOTSTRAP; m++)
                                    {
                                        // Calculate derivatives based on current mini-step state
                                        double drdt_m = current_vrad_mini;
                                        double force_m = gravitational_force(current_r_mini, i_part, npts, G_CONST, g_active_halo_mass);
                                        double dvdt_m = force_m + effective_angular_force(current_r_mini, current_ell_mini);
                                        
                                        // Euler update for this mini-step
                                        current_r_mini += dt_mini * drdt_m;
                                        current_vrad_mini += dt_mini * dvdt_m;
                                    }
                                    
                                    // After all mini-steps, update the main particles array
                                    particles[0][i_part] = current_r_mini;
                                    particles[1][i_part] = current_vrad_mini;
                                }
                            }
                        }
                    }

#pragma omp single
                    {
                        // Mark bootstrap done.
                        ab3_bootstrap_done = 1;

                        // Bootstrap runs at j=0; handle trajectory buffer accordingly
                            // Store time for this step and advance buffer index
                            if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                                g_time_buf[g_trajectory_buffer_index] = time;
                            }
                            g_trajectory_buffer_index++;

                    }
                }
                else
                {
                    /**
                     * Normal AB3 step each iteration
                     */

#pragma omp single
                    {
                        sort_particles(particles, npts);
                    }

#pragma omp barrier
#pragma omp parallel for default(shared) schedule(static)
                    for (int idx = 0; idx < npts; idx++)
                    {
                        int orig_id = (int)particles[3][idx];
                        inverse_map[orig_id] = idx;
                    }

#pragma omp parallel for default(shared) schedule(static)
                    for (int i = 0; i < npts; i++)
                    {
                        // Adams-Bashforth 8th order integration step.
                        int orig_id = (int)particles[3][i];
                        double rr = particles[0][i];
                        double vrad = particles[1][i];

                        double sum_r = 0.0;
                        double sum_v = 0.0;
                        int particle_state = g_particle_scatter_state[orig_id];

                        // AB3 coefficients: b0=23/12, b1=-16/12, b2=5/12. Denom ab3_den=12.
                        // History: f_ab3_[r/v][2] is f_n (latest), [1] is f_{n-1}, [0] is f_{n-2}

                        if (particle_state == 1) { // Just scattered: Use AB1 (Euler-like)
                            // sum = 12 * f_n
                            sum_r = 12.0 * f_ab3_r[2][orig_id];
                            sum_v = 12.0 * f_ab3_v[2][orig_id];
                            if (g_doDebug && i < 5) { // Extremely sparse debug
                                 log_message("DEBUG", "AB3_RESET: Particle %d (orig_id) using AB1 step (state 1)", orig_id);
                            }
                        } else if (particle_state == 2) { // One step after scatter: Use AB2
                            // sum = ab2_num[0] * f_n + ab2_num[1] * f_{n-1}
                            sum_r = ab2_num[0] * f_ab3_r[2][orig_id] + ab2_num[1] * f_ab3_r[1][orig_id];
                            sum_v = ab2_num[0] * f_ab3_v[2][orig_id] + ab2_num[1] * f_ab3_v[1][orig_id];
                            if (g_doDebug && i < 5) {
                                 log_message("DEBUG", "AB3_RESET: Particle %d (orig_id) using AB2 step (state 2)", orig_id);
                            }
                        } else { // Normal AB3 step
                            sum_r = ab3_num[0] * f_ab3_r[2][orig_id] + ab3_num[1] * f_ab3_r[1][orig_id] + ab3_num[2] * f_ab3_r[0][orig_id];
                            sum_v = ab3_num[0] * f_ab3_v[2][orig_id] + ab3_num[1] * f_ab3_v[1][orig_id] + ab3_num[2] * f_ab3_v[0][orig_id];
                        }

                        double r_next = rr + (dt / (double)ab3_den) * sum_r;
                        double v_next = vrad + (dt / (double)ab3_den) * sum_v;

                        particles[0][i] = r_next;
                        particles[1][i] = v_next;
                    }

                    // SIDM scattering after Adams-Bashforth update (skip during bootstrap)
                    double current_active_rc_for_sidm = g_use_hernquist_aniso_profile ? g_scale_radius_param : (g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc);
                    handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, !ab3_bootstrap_done, g_current_timestep_scatter_counts);

                    // Re-sort and compute new derivatives to shift AB3 history
#pragma omp single
                    {
                        sort_particles(particles, npts);
                    }
#pragma omp barrier
                    // #pragma omp single.
                    // {

#pragma omp parallel for default(shared) schedule(static)
                    for (int idx = 0; idx < npts; idx++)
                    {
                        int orig_id = (int)particles[3][idx];
                        inverse_map[orig_id] = idx;
                    }

                    // Recompute the derivatives for the new time => goes into f_ab3_r[2], f_ab3_v[2].
                    double **f_new_r = (double **)malloc(sizeof(double *));
                    double **f_new_v = (double **)malloc(sizeof(double *));
                    f_new_r[0] = (double *)malloc(npts * sizeof(double));
                    f_new_v[0] = (double *)malloc(npts * sizeof(double));

#pragma omp parallel for default(shared) schedule(static)
                    for (int i_dbg = 0; i_dbg < npts; i_dbg++)
                    {
                        int orig_id = (int)particles[3][i_dbg];
                        double rr = particles[0][i_dbg];
                        double vrad = particles[1][i_dbg];
                        double ell = particles[2][i_dbg];

                        double drdt = vrad;
                        // Use the gravitational_force and effective_angular_force functions.
                        double force = gravitational_force(rr, i_dbg, npts, G_CONST, g_active_halo_mass);
                        double dvdt = force + effective_angular_force(rr, ell);

                        f_new_r[0][orig_id] = drdt;
                        f_new_v[0][orig_id] = dvdt;
                    }

#pragma omp single
                    {
                        // SHIFT AB3 HISTORY: f0 ← f1, f1 ← f2
                        for (int i_s = 0; i_s < npts; i_s++)
                        {
                            f_ab3_r[0][i_s] = f_ab3_r[1][i_s]; // f_{n-2} ← f_{n-1} (history shift)
                            f_ab3_v[0][i_s] = f_ab3_v[1][i_s];

                            f_ab3_r[1][i_s] = f_ab3_r[2][i_s]; // f_{n-1} ← f_n (history shift)
                            f_ab3_v[1][i_s] = f_ab3_v[2][i_s];
                        }
                        // Store current derivatives in slot 2 (latest position)
                        for (int i_s = 0; i_s < npts; i_s++)
                        {
                            f_ab3_r[2][i_s] = f_new_r[0][i_s]; // f_n (latest)
                            f_ab3_v[2][i_s] = f_new_v[0][i_s];
                        }

                        free(f_new_r[0]);
                        free(f_new_v[0]);
                        free(f_new_r);
                        free(f_new_v);

                        // Record trajectory data at every timestep into buffer
                            // Store time for this step and advance buffer index
                            if (g_trajectory_buffer_index < g_trajectory_buffer_size) {
                                g_time_buf[g_trajectory_buffer_index] = time;
                            }
                            g_trajectory_buffer_index++;


                        current_step = j + 1;
                        time += dt;

                        // Advance particle scatter states for next AB step
                        if (ab3_bootstrap_done) { // Only advance state if AB is active and past bootstrap
                            for (int k_pstate = 0; k_pstate < npts; k_pstate++) {
                                // k_pstate here is the original_id since g_particle_scatter_state is indexed by orig_id
                                if (g_particle_scatter_state[k_pstate] == 2) {
                                    g_particle_scatter_state[k_pstate] = 0; // Transition from AB2 to full AB3
                                } else if (g_particle_scatter_state[k_pstate] == 1) {
                                    g_particle_scatter_state[k_pstate] = 2; // Transition from AB1 to AB2
                                }
                                // If state is 0, it remains 0 unless SIDM sets it to 1 in the next call to handle_sidm_step
                            }
                        }
                    }
                } // End of the "else" block for normal AB3.
            }
            // Record trajectory data at every timestep into buffer
                // Record trajectory data for selected low-ID particles
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr) * VEL_CONV_SQ;
                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;

                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                    if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size) {
                        g_trajectories_buf[p][g_trajectory_buffer_index - 1] = rr;
                        g_velocities_buf[p][g_trajectory_buffer_index - 1] = vrad;
                        g_mu_buf[p][g_trajectory_buffer_index - 1] = mu_val;
                        g_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                        g_L_buf[p][g_trajectory_buffer_index - 1] = ell;

                }
            }

            // Record trajectory data for selected low-L particles
            int max_threads = omp_get_max_threads();
            gsl_interp_accel **thread_accel = (gsl_interp_accel **)malloc(max_threads * sizeof(gsl_interp_accel *));
            for (int i = 0; i < max_threads; i++)
            {
                thread_accel[i] = gsl_interp_accel_alloc();
            }

#pragma omp parallel for if (nlowest > 1000) schedule(static)
            for (int p = 0; p < nlowest; p++)
            {
                int thread_id = omp_get_thread_num();
                gsl_interp_accel *thread_safe_accel = thread_accel[thread_id];

                int idx_lowest = inverse_map[chosen[p]];

                double rr = particles[0][idx_lowest];
                double vrad = particles[1][idx_lowest];
                double ell = particles[2][idx_lowest];

                if (rr < 0.0)
                {
                    rr = 0.0000000001;
                }
                if (rr > rmax)
                {
                    rr = rmax;
                }

                // Use thread-safe accelerator instead of shared one.
                double Psi_val = evaluatespline(splinePsi, thread_safe_accel, rr) * VEL_CONV_SQ;
                double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

                if (g_trajectory_buffer_index - 1 < g_trajectory_buffer_size && g_trajectory_buffer_index > 0) {
                    g_lowestL_r_buf[p][g_trajectory_buffer_index - 1] = rr;
                    g_lowestL_E_buf[p][g_trajectory_buffer_index - 1] = E_rel;
                    g_lowestL_L_buf[p][g_trajectory_buffer_index - 1] = ell;
                }
            }

            // Clean up thread-local accelerators.
            for (int i = 0; i < max_threads; i++)
            {
                gsl_interp_accel_free(thread_accel[i]);
            }
            free(thread_accel);

#pragma omp single
            {
                // Check whether to write snapshot at this absolute timestep
                // current_step = j + 1 represents the current step in this run
                // restart_offset is the number of timesteps completed before this run
                // Write snapshots based on previous timestep (j), not current
                // So the absolute timestep to check is restart_offset + j
                int absolute_timestep = restart_offset + j;

                // In restart mode, skip j=0 to avoid re-writing the loaded snapshot
                int should_write = ((absolute_timestep % dtwrite) == 0) && !(g_restart_mode_active && j == 0);

                if (should_write)
                {

                    double elapsed = omp_get_wtime() - start_time;
                    // For display, show the actual timestep including restart offset
                    printf("Write data at timestep %d after %.2f s.\n",
                           absolute_timestep, elapsed);

                    // Increment count BEFORE using it
                    nwrite_total++;

                    // Use the persistent nwrite_total counter which tracks the absolute write index
                    int nwrite = nwrite_total - 1;  // Use the pre-increment value

                    if (g_doAllParticleData)
                    {
                        // Calculate buffer index (0 to snapshot_block_size-1) based on writes in this run
                        // On restart, buffer is freshly allocated and must start at position 0
                        int writes_this_run = g_restart_mode_active ? (nwrite - restart_initial_nwrite) : nwrite;
                        int block_index_apd = writes_this_run % snapshot_block_size;
                        
                        for (int pi = 0; pi < npts; pi++)
                        {
                            int orig_id = pi;
                            int rank = inverse_map[orig_id];
                            double par_r = particles[0][rank];
                            double par_vrad = particles[1][rank];
                            double par_ell = particles[2][rank];
                            // Convert cos(phi) and sin(phi) to actual phi angle using atan2
                            double par_phi = atan2(particles[6][rank], particles[5][rank]);

                            Rank_block[block_index_apd * npts + orig_id] = rank;
                            R_block[block_index_apd * npts + orig_id] = (float)par_r;
                            Vrad_block[block_index_apd * npts + orig_id] = (float)par_vrad;
                            L_block[block_index_apd * npts + orig_id] = (float)par_ell;
                            phi_block[block_index_apd * npts + orig_id] = (float)par_phi;
                            scatter_count_block[block_index_apd * npts + orig_id] = g_current_timestep_scatter_counts[orig_id];  // Store scatter count
                            ID_block[block_index_apd * npts + orig_id] = orig_id;  // Store original particle ID

                            if (g_doDebug)
                            {
                                // Check if this write corresponds to a desired snapshot output time
                                if (nwrite % stepBetweenSnaps == 0)
                                {
                                    int snapIndex = nwrite / stepBetweenSnaps;
                                    if (snapIndex < noutsnaps) // Ensure snapIndex is valid
                                    {
                                        int debug_id = DEBUG_PARTICLE_ID;
                                        
                                        // Retrieve current state for debug particle from block arrays
                                        float r_valF = R_block[block_index_apd * npts + debug_id];
                                        float v_valF = Vrad_block[block_index_apd * npts + debug_id];
                                        float l_valF = L_block[block_index_apd * npts + debug_id];
                                        double r_val = (double)r_valF;
                                        double v_val = (double)v_valF;
                                        double l_val = (double)l_valF;
                                        
                                        // Evaluate theoretical potential using original spline
                                        double psi_val = 0.0;
                                        if (r_val >= 0.0 && r_val <= rmax)
                                        {
                                            psi_val = evaluatespline(splinePsi, Psiinterp, r_val) * VEL_CONV_SQ;
                                        }
                                        
                                        // Calculate approximate energy E = Psi - KE
                                        double E_approx = psi_val - 0.5 * (v_val * v_val + (l_val * l_val) / (r_val * r_val));
                                        double sim_time = time;
                                        
                                        // Store the approximate energy for comparison
                                        store_debug_approxE(snapIndex, E_approx, sim_time);
                                    }
                                }
                            }
                        }
                    }
                    if (g_doAllParticleData)
                    {
                        /** @brief Append block to file if block buffer is full.
                         * @details In restart mode, prevents writing incomplete blocks that would
                         *          duplicate data. When nwrite=99 after restart from snapshot 49,
                         *          only 50 new snapshots written (indices 0-49 in buffer).
                         *          Writing a full 100-snapshot block would include 50 uninitialized
                         *          positions, causing duplication when leftovers are flushed.
                         * @note Prevents block duplication in restart mode by tracking writes correctly.
                         */
                        // Calculate whether to write a block
                        // Determine if a full block has been accumulated in this run.
                        // The +1 accounts for checking AFTER filling but BEFORE incrementing nwrite_total
                        int writes_this_run_for_check = nwrite + 1;
                        int should_write_block = (writes_this_run_for_check > 0 && (writes_this_run_for_check % snapshot_block_size) == 0);

                        if (should_write_block)
                        {
                            append_all_particle_data_chunk_to_file(apd_filename,
                                                                   npts,
                                                                   snapshot_block_size,
                                                                   L_block,
                                                                   Rank_block,
                                                                   R_block,
                                                                   Vrad_block);
                            // Block appended to all_particle_data file

                            // Append phi data to separate file
                            char phi_filename[1024];
                            snprintf(phi_filename, sizeof(phi_filename), "data/all_particle_phi%s.dat", g_file_suffix);
                            append_all_particle_phi_data_chunk_to_file(phi_filename,
                                                                       npts,
                                                                       snapshot_block_size,
                                                                       phi_block);

                            // Append scatter count data to separate file
                            char scatter_filename[1024];
                            snprintf(scatter_filename, sizeof(scatter_filename), "data/all_particle_scatter_counts%s.dat", g_file_suffix);
                            append_all_particle_scatter_counts_to_file(scatter_filename,
                                                                       npts,
                                                                       snapshot_block_size,
                                                                       scatter_count_block);

                            // Append particle ID data to separate file
                            char ids_filename[1024];
                            snprintf(ids_filename, sizeof(ids_filename), "data/all_particle_ids%s.dat", g_file_suffix);
                            append_all_particle_ids_to_file(ids_filename,
                                                           npts,
                                                           snapshot_block_size,
                                                           ID_block);

                            // Reset scatter counts for next block
                            memset(scatter_count_block, 0, (size_t)npts * snapshot_block_size * sizeof(int));

                            // Update double-precision buffer with current snapshot (BEFORE float32 conversion!)
                            // Allocate temporary arrays for buffer update using current double precision data
                            double *extracted_R = (double *)malloc(npts * sizeof(double));
                            double *extracted_Vrad = (double *)malloc(npts * sizeof(double));
                            double *extracted_L = (double *)malloc(npts * sizeof(double));
                            int *extracted_Rank = (int *)malloc(npts * sizeof(int));
                            double *extracted_phi = (double *)malloc(npts * sizeof(double));

                            if (!extracted_R || !extracted_Vrad || !extracted_L || !extracted_Rank || !extracted_phi) {
                                fprintf(stderr, "ERROR: Failed to allocate temp arrays for double buffer update\n");
                                CLEAN_EXIT(1);
                            }

                            // Extract CURRENT double-precision particle state (particles array is sorted by radius)
                            for (int pi = 0; pi < npts; pi++) {
                                int orig_id = pi;
                                int rank = inverse_map[orig_id];
                                extracted_R[orig_id] = particles[0][rank];       // Full double precision
                                extracted_Vrad[orig_id] = particles[1][rank];    // Full double precision
                                extracted_L[orig_id] = particles[2][rank];       // Full double precision
                                extracted_Rank[orig_id] = rank;                  // Rank for this particle
                                // Convert cos(phi) and sin(phi) back to phi angle
                                extracted_phi[orig_id] = atan2(particles[6][rank], particles[5][rank]);
                            }

                            // Create particles array format for add_snapshot_to_double_buffer
                            double *particles_for_buffer[7];
                            particles_for_buffer[0] = extracted_R;      // R
                            particles_for_buffer[1] = extracted_Vrad;   // Vrad
                            particles_for_buffer[2] = extracted_L;      // L

                            // Add to double buffer (preserves full double precision)
                            add_snapshot_to_double_buffer(particles_for_buffer, extracted_phi,
                                                         extracted_Rank, npts);

                            // Flush buffer to disk
                            flush_double_buffer_to_disk(g_file_suffix);

                            // Free temporary arrays
                            free(extracted_R);
                            free(extracted_Vrad);
                            free(extracted_L);
                            free(extracted_Rank);
                            free(extracted_phi);
                        }

                        // Check if this write corresponds to a snapshot for energy diagnostics
                        if (nwrite % stepBetweenSnaps == 0) {
                            int snapIndex = nwrite / stepBetweenSnaps;
                            if (snapIndex < noutsnaps) {
                                // The particles array is sorted at this point in the loop
                                double current_KE, current_PE;
                                calculate_system_energies(particles, npts, deltaM, &current_KE, &current_PE);

                                // Store the calculated values in the global arrays
                                g_time_snapshots[snapIndex] = time;
                                g_total_KE[snapIndex] = current_KE;
                                g_total_PE[snapIndex] = current_PE;
                                g_total_E[snapIndex] = current_KE + current_PE;
                            }
                        }
                    }


                    // Flush trajectory buffers if a full block has been written. This call is
                    // synchronized with the main particle data block write to ensure that all
                    // output files correspond to the same set of timesteps.
                    // Use nwrite_total (already incremented at line 13449) for accurate count
                    int writes_this_run_for_check = nwrite_total;
                    int should_flush_traj_block = (writes_this_run_for_check > 0 && (writes_this_run_for_check % snapshot_block_size) == 0);

                    if (should_flush_traj_block)
                    {
                        flush_trajectory_buffers(g_trajectory_buffer_index, upper_npts_num_traj, nlowest);
                    }

                    // Reset scatter counts for next timestep
                    memset(g_current_timestep_scatter_counts, 0, npts * sizeof(int));
                }
            }

            for (int k = 0; k <= 20; k++)
            {
                if (current_step == print_steps[k])
                {
                    double elapsed = omp_get_wtime() - start_time;
                    // Calculate actual percentage based on total progress
                    int display_current = current_step + restart_offset;
                    int display_total = g_restart_mode_active ? original_Ntimes : Ntimes;
                    int percent = g_restart_mode_active ? 
                        (int)(100.0 * display_current / display_total + 0.5) : 
                        k * 5;
                    printf("%d%% complete, timestep %d/%d, time=%f Myr, elapsed=%.2f s\n",
                           percent, display_current, display_total, time, elapsed);
                    break;
                }
            }
        }
        // After the main simulation loop, flush any remaining data in the last partial block.
        {
            if (g_doAllParticleData)
            {
                // On restart, count only writes since restart for correct buffer indexing
                int leftover;
                if (g_restart_mode_active && restart_initial_nwrite > 0) {
                    // Count only writes performed since restart
                    int writes_since_restart = nwrite_total - restart_initial_nwrite;
                    leftover = writes_since_restart % snapshot_block_size;
                } else {
                    // Normal operation
                    leftover = nwrite_total % snapshot_block_size;
                }
                if (leftover > 0)
                {
                    printf("Flushing leftover %d steps from block storage...\n", leftover);
                    
                    append_all_particle_data_chunk_to_file(apd_filename,
                                                           npts,
                                                           leftover, // Number of steps in this partial block
                                                           L_block,
                                                           Rank_block,
                                                           R_block,
                                                           Vrad_block);

                    // Also flush phi data
                    char phi_filename[1024];
                    snprintf(phi_filename, sizeof(phi_filename), "data/all_particle_phi%s.dat", g_file_suffix);
                    append_all_particle_phi_data_chunk_to_file(phi_filename,
                                                               npts,
                                                               leftover,
                                                               phi_block);

                    // Also flush scatter count data
                    char scatter_filename[1024];
                    snprintf(scatter_filename, sizeof(scatter_filename), "data/all_particle_scatter_counts%s.dat", g_file_suffix);
                    append_all_particle_scatter_counts_to_file(scatter_filename,
                                                               npts,
                                                               leftover,
                                                               scatter_count_block);

                    // Also flush particle ID data
                    char ids_filename[1024];
                    snprintf(ids_filename, sizeof(ids_filename), "data/all_particle_ids%s.dat", g_file_suffix);
                    append_all_particle_ids_to_file(ids_filename,
                                                   npts,
                                                   leftover,
                                                   ID_block);
                }
            }
        }
    } // Close the skip_simulation if block.

    // Write final particle state (skipped if skip_file_writes active)
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/particlesfinal.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        if (fp == NULL)
        {
            fprintf(stderr, "Error opening file %s for writing\n", suffixed_filename);
            exit(1);
        }

        for (i = 0; i < npts; i++)
        {
            fprintf_bin(fp, "%f %f %f  %f\n", particles[0][i], particles[1][i], particles[2][i], particles[3][i]);
        }
        fclose(fp);
    }

    // Write theoretical profiles (profile-specific formulas)

    if (g_use_nfw_profile) {
        /**
         * @brief Write final theoretical NFW profile characteristics to .dat files.
         * @details This block outputs several files (massprofile, Psiprofile, density_profile,
         *          dpsi_dr, drho_dpsi, f_of_E, df_fixed_radius) using splines and parameters
         *          from NFW initial condition generation (splinemass, splinePsi, g_main_fofEinterp,
         *          num_points, radius, normalization, g_nfw_profile_rc).
         *          Analytical formulas for NFW density and its derivatives are used where appropriate.
         */
        log_message("INFO", "Writing NFW theoretical profiles to final .dat files...");

        // Write NFW theoretical mass profile
        get_suffixed_filename("data/massprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                    fprintf_bin(fp, "%f %f\n", r_plot, gsl_spline_eval(splinemass, r_plot, enclosedmass));
                }
            }
            if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], gsl_spline_eval(splinemass, radius[num_points-1], enclosedmass));
            }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final NFW mass profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping NFW mass profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write NFW theoretical potential profile
        get_suffixed_filename("data/Psiprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinePsi && Psiinterp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                     fprintf_bin(fp, "%f %f\n", r_plot, evaluatespline(splinePsi, Psiinterp, r_plot));
                }
            }
             if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], evaluatespline(splinePsi, Psiinterp, radius[num_points-1]));
             }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final NFW Psi profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping NFW Psi profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write NFW theoretical density profile
        get_suffixed_filename("data/density_profile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            double nt_nfw_scaler_final = g_nfw_profile_halo_mass / (4.0 * M_PI * normalization);
            for (i = 0; i < num_points; i++) {
                double rr = radius[i]; 
                double rs_k = rr / g_nfw_profile_rc;
                double term_s_k = rs_k + 0.01; 
                if (term_s_k <= 1e-9) term_s_k = 1e-9;
                double term_n_k = (1.0 + rs_k) * (1.0 + rs_k);
                double term_c_base_k = rs_k / g_nfw_profile_falloff_factor;
                double term_c_k = 1.0 + pow(term_c_base_k, 10.0);
                double rho_shape_k = (term_s_k < 1e-9 || term_n_k < 1e-9 || term_c_k < 1e-9) ? 0.0 : (1.0 / (term_s_k * term_n_k * term_c_k));
                if (rr < 1e-6 && term_s_k < 1e-3 && rho_shape_k == 0.0) { 
                   rho_shape_k = 1.0 / (term_s_k * term_n_k * term_c_k);
                }
                double rho_r_k = nt_nfw_scaler_final * rho_shape_k;
                fprintf_bin(fp, "%f %f\n", rr, rho_r_k);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW density profile", suffixed_filename); 
        }

        // Write NFW theoretical dPsi/dr profile
        get_suffixed_filename("data/dpsi_dr.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                if (rr > 0.0) {
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass); 
                    double dpsidr = -(G_CONST * Menc) / (rr * rr);
                    fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                }
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW dpsi/dr profile", suffixed_filename); 
        }

        // Write NFW theoretical drho/dPsi profile
        get_suffixed_filename("data/drho_dpsi.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            double nt_nfw_scaler_final = g_nfw_profile_halo_mass / (4.0 * M_PI * normalization);
            for (i = 1; i < num_points - 1; i++) {
                double rr = radius[i];
                if (rr <= 1e-9) continue;

                // Use NFW derivative function
                double drhodr_val_k = drhodr_profile_nfwcutoff(rr, g_nfw_profile_rc, nt_nfw_scaler_final, g_nfw_profile_falloff_factor);

                double Menc_k = gsl_spline_eval(splinemass, rr, enclosedmass);
                double dPsidr_k = -(G_CONST * Menc_k) / (rr * rr);

                if (fabs(dPsidr_k) > 1e-30) {
                    double Psi_val_k = evaluatespline(splinePsi, Psiinterp, rr);
                    double drho_dPsi_val_k = drhodr_val_k / dPsidr_k;
                    fprintf_bin(fp, "%f %f\n", Psi_val_k, drho_dPsi_val_k);
                }
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW drho/dpsi profile", suffixed_filename); 
        }

        // Write NFW theoretical f(E) profile
        get_suffixed_filename("data/f_of_E.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i <= num_points; i++) {
                double E = Evalues[i];
                double deriv = 0.0;
                if (i > 0 && i < num_points + 1) {
                    if (i > 0 && i < num_points) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i - 1]) / (Evalues[i + 1] - Evalues[i - 1]);
                    }
                    else if (i == 0) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i]) / (Evalues[i + 1] - Evalues[i]);
                    }
                    else if (i == num_points) {
                        deriv = (innerintegrandvalues[i] - innerintegrandvalues[i - 1]) / (Evalues[i] - Evalues[i - 1]);
                    }
                }
                double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                if (E == 0.0 || !isfinite(fE))
                    fE = 0.0;
                fprintf_bin(fp, "%f %f\n", E, fE);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW f(E) profile", suffixed_filename); 
        }

        // Write NFW distribution function at fixed radius (skipped if skip_file_writes active)
        if (!skip_file_writes) {
            get_suffixed_filename("data/df_fixed_radius.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            fp = fopen(suffixed_filename, "wb");
            if (fp) {
                double r_F = 2.0 * g_nfw_profile_rc;  // r_F = 2 × NFW scale radius
                double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_F);
                Psi_rf *= VEL_CONV_SQ;
                double Psimin_test = VEL_CONV_SQ * Psimin; // Convert to (km/s)² for velocity calculation

                int vsteps = 10000;
                int reduce_vsteps = 300;
                for (int vv = 0; vv <= vsteps - reduce_vsteps; vv++) {
                    double sqrt_arg_v = Psi_rf - Psimin_test;
                    if (sqrt_arg_v < 0) sqrt_arg_v = 0;
                    double vtest = (double)vv * (sqrt(2.0 * sqrt_arg_v) / (vsteps));
                    double vtest_codes = vtest / kmsec_to_kpcmyr;
                    double fEval = 0.0;

                    if (g_use_om_profile) {
                        // OM: integrate f(Q) over μ using GSL adaptive quadrature
                        double Psi_rf_codes = Psi_rf / VEL_CONV_SQ;
                        double v_sq = vtest_codes * vtest_codes;
                        double r_over_ra_sq = (r_F / g_om_anisotropy_radius) * (r_F / g_om_anisotropy_radius);

                        om_mu_integral_params params_mu = {v_sq, Psi_rf_codes, r_over_ra_sq,
                                                            Psimin, Psimax, g_main_fofEinterp,
                                                            Evalues, innerintegrandvalues, g_main_fofEacc};
                        gsl_function F_mu;
                        F_mu.function = &om_mu_integrand;
                        F_mu.params = &params_mu;

                        double integral_result, error;
                        gsl_integration_workspace *w_mu = gsl_integration_workspace_alloc(10000);
                        int status = gsl_integration_qag(&F_mu, 0.0, 1.0, 1e-7, 1e-7, 10000,
                                                         GSL_INTEG_GAUSS61, w_mu, &integral_result, &error);
                        gsl_integration_workspace_free(w_mu);

                        if (status == GSL_SUCCESS) {
                            fEval = 2.0 * vtest_codes * vtest_codes * r_F * r_F * integral_result;
                        }
                    } else {
                        // Isotropic: standard evaluation
                        double Etest = Psi_rf - 0.5 * vtest * vtest;
                        double Etest_codes = Etest / VEL_CONV_SQ;
                        if (Etest_codes >= Psimin && Etest_codes <= Psimax) {
                            double derivative;
                            int status = gsl_interp_eval_deriv_e(g_main_fofEinterp, Evalues, innerintegrandvalues,
                                                                 Etest_codes, g_main_fofEacc, &derivative);
                            if (status == GSL_SUCCESS) {
                                fEval = derivative / (sqrt(8.0) * PI * PI) * vtest * vtest * r_F * r_F;
                            }
                        }
                    }
                    if (!isfinite(fEval)) fEval = 0.0;
                    fprintf_bin(fp, "%f %f\n", vtest, fEval);
                }
                fclose(fp);
            } else {
                log_message("ERROR", "Failed to open %s for final NFW df_fixed_radius", suffixed_filename); 
            }
        }

    } else if (g_use_hernquist_aniso_profile) {
        /**
         * @brief Write final theoretical Hernquist anisotropic profile characteristics to .dat files.
         * @details This block outputs several files (massprofile, Psiprofile, density_profile,
         *          dpsi_dr, drho_dpsi, f_of_E, df_fixed_radius) using splines and parameters
         *          from Hernquist initial condition generation (splinemass, splinePsi, g_main_fofEinterp).
         *          Analytical formulas for Hernquist density and its derivatives are used where appropriate.
         */
        log_message("INFO", "Writing Hernquist anisotropic theoretical profiles to final .dat files...");
        
        // Write Hernquist theoretical mass profile
        get_suffixed_filename("data/massprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                    fprintf_bin(fp, "%f %f\n", r_plot, gsl_spline_eval(splinemass, r_plot, enclosedmass));
                }
            }
            if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], gsl_spline_eval(splinemass, radius[num_points-1], enclosedmass));
            }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist mass profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist mass profile output - splines not initialized");
                fclose(fp);
            }
        }
        
        // Write Hernquist theoretical potential profile
        get_suffixed_filename("data/Psiprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinePsi && Psiinterp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                     fprintf_bin(fp, "%f %f\n", r_plot, evaluatespline(splinePsi, Psiinterp, r_plot));
                }
            }
             if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], evaluatespline(splinePsi, Psiinterp, radius[num_points-1]));
             }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist Psi profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist Psi profile output - splines not initialized");
                fclose(fp);
            }
        }
        
        // Write Hernquist theoretical density profile
        get_suffixed_filename("data/density_profile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i < num_points; i++) {
                double rr = radius_unsorted[i];
                double rho_r = density_hernquist(rr, g_halo_mass_param, g_scale_radius_param);
                fprintf_bin(fp, "%f %f\n", rr, rho_r);
            }
            fclose(fp);
        } else {
            log_message("ERROR", "Failed to open %s for final Hernquist density profile", suffixed_filename);
        }
        
        // Write Hernquist theoretical dPsi/dr profile
        get_suffixed_filename("data/dpsi_dr.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass) {
            for (i = 0; i < num_points; i++) {
                double rr = radius_unsorted[i];
                if (rr > 0.0) {
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                    double dpsidr = -(G_CONST * Menc) / (rr * rr);
                    fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                }
            }
            fclose(fp);
        } else {
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist dpsi/dr profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist dpsi/dr profile output - splines not initialized");
                fclose(fp);
            }
        }
        
        // Write Hernquist theoretical drho/dPsi profile
        get_suffixed_filename("data/drho_dpsi.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass && splinePsi && Psiinterp) {
            for (i = 1; i < num_points - 1; i++) {
                double rr = radius[i];
                if (rr <= 1e-9) continue;
                
                // Analytical derivative of Hernquist density
                double a = g_scale_radius_param;
                double M = g_halo_mass_param;
                double r_plus_a = rr + a;
                double drhodr_val = -(M * a / (2.0 * M_PI)) * (1.0/(rr*rr) * 1.0/(r_plus_a*r_plus_a*r_plus_a) + 3.0/(rr) * 1.0/(r_plus_a*r_plus_a*r_plus_a*r_plus_a));

                double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                double dPsidr = -(G_CONST * Menc) / (rr * rr);

                if (fabs(dPsidr) > 1e-30) {
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    double drho_dPsi_val = drhodr_val / dPsidr;
                    fprintf_bin(fp, "%f %f\n", Psi_val, drho_dPsi_val);
                }
            }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist drho/dpsi profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist drho/dpsi profile output - splines not initialized");
                fclose(fp);
            }
        }
        
        // Write Hernquist theoretical f(E) profile
        get_suffixed_filename("data/f_of_E.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && Evalues && innerintegrandvalues) {
            // Use the numerically derived data from IC generation
            for (i = 0; i <= num_points; i++) {
                double E = Evalues[i];
                double deriv = 0.0;
                if (i > 0 && i < num_points) {
                    deriv = gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, innerintegrandvalues, E, g_main_fofEacc);
                }
                double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                if (!isfinite(fE)) fE = 0.0;
                fprintf_bin(fp, "%f %f\n", E, fE);
            }
            fclose(fp);
        } else {
            log_message("ERROR", "Failed to open %s for final Hernquist aniso f(E) profile", suffixed_filename);
        }
        
        // Write Hernquist distribution function at a fixed radius
        if (!skip_file_writes) {
            get_suffixed_filename("data/df_fixed_radius.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            fp = fopen(suffixed_filename, "wb");
            if (fp && splinePsi && Psiinterp && Psivalues && num_points > 0) {
                double r_F = 2.0 * g_scale_radius_param;
                double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_F);
                Psi_rf *= VEL_CONV_SQ;
                double Psimin_local = Psivalues[num_points - 1]; // Potential at rmax
                double Psimin_test = VEL_CONV_SQ * Psimin_local;
                
                int vsteps = 1000;
                for (int vv = 0; vv <= vsteps; vv++) {
                    double sqrt_arg_v = Psi_rf - Psimin_test;
                    if (sqrt_arg_v < 0) sqrt_arg_v = 0;
                    double vtest = (double)vv * (sqrt(2.0 * sqrt_arg_v) / vsteps);
                    double Etest = -Psi_rf + 0.5 * vtest * vtest;

                    // Anisotropic distribution function for Hernquist profile at fixed radius
                    double fEval = 0.0;
                    double E_bind_phys = -Etest;
                    if (E_bind_phys > 0 && r_F > 0) {
                        double E_bind_codes = E_bind_phys / VEL_CONV_SQ;
                        double vtest_codes = vtest / kmsec_to_kpcmyr;
                        double L_codes = r_F * vtest_codes * 0.5; // Approximate L in code units

                        if (L_codes > 0) {
                            double f_val = df_hernquist_aniso(E_bind_codes, L_codes, g_halo_mass_param, g_scale_radius_param);
                            fEval = f_val * vtest_codes * vtest_codes * r_F * r_F;
                        }
                    }
                    if (!isfinite(fEval)) fEval = 0.0;
                    fprintf_bin(fp, "%f %f\n", vtest, fEval);
                }
                fclose(fp);
            } else { 
                if (!fp) {
                    log_message("ERROR", "Failed to open %s for final Hernquist df_fixed_radius", suffixed_filename);
                } else {
                    log_message("WARNING", "Skipping Hernquist df_fixed_radius output - required data not initialized");
                    if (fp) fclose(fp);
                }
            }
        }

    } else if (g_use_hernquist_numerical) {
        /**
         * @brief Write final theoretical Hernquist numerical profile characteristics to .dat files.
         * @details This block outputs files using splines generated during the numerical
         *          Hernquist IC setup. It uses analytical formulas where appropriate.
         */
        log_message("INFO", "Writing Hernquist numerical theoretical profiles to final .dat files...");

        // Write Hernquist theoretical mass profile from spline
        get_suffixed_filename("data/massprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                    fprintf_bin(fp, "%f %f\n", r_plot, gsl_spline_eval(splinemass, r_plot, enclosedmass));
                }
            }
            if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], gsl_spline_eval(splinemass, radius[num_points-1], enclosedmass));
            }
            fclose(fp);
        } else {
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist (num) mass profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist (num) mass profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write Hernquist theoretical potential profile from spline
        get_suffixed_filename("data/Psiprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinePsi && Psiinterp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                     fprintf_bin(fp, "%f %f\n", r_plot, evaluatespline(splinePsi, Psiinterp, r_plot));
                }
            }
             if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], evaluatespline(splinePsi, Psiinterp, radius[num_points-1]));
             }
            fclose(fp);
        } else {
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist (num) Psi profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist (num) Psi profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write Hernquist theoretical density profile using analytical formula
        get_suffixed_filename("data/density_profile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                double rho_r = density_hernquist(rr, g_halo_mass_param, g_scale_radius_param);
                fprintf_bin(fp, "%f %f\n", rr, rho_r);
            }
            fclose(fp);
        } else {
            log_message("ERROR", "Failed to open %s for final Hernquist (num) density profile", suffixed_filename);
        }

        // Write Hernquist theoretical dPsi/dr profile using spline
        get_suffixed_filename("data/dpsi_dr.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                if (rr > 0.0) {
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                    double dpsidr = -(G_CONST * Menc) / (rr * rr);
                    fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                }
            }
            fclose(fp);
        } else {
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist (num) dpsi/dr profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist (num) dpsi/dr profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write Hernquist theoretical drho/dPsi profile
        get_suffixed_filename("data/drho_dpsi.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass && splinePsi && Psiinterp) {
            for (i = 1; i < num_points - 1; i++) {
                double rr = radius[i];
                if (rr <= 1e-9) continue;

                double drhodr_val = drhodr_hernquist(rr, g_scale_radius_param, g_halo_mass_param);

                double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                double dPsidr = -(G_CONST * Menc) / (rr * rr);

                if (fabs(dPsidr) > 1e-30) {
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    double drho_dPsi_val = drhodr_val / dPsidr;
                    fprintf_bin(fp, "%f %f\n", Psi_val, drho_dPsi_val);
                }
            }
            fclose(fp);
        } else {
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final Hernquist (num) drho/dpsi profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping Hernquist (num) drho/dpsi profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write Hernquist theoretical f(E) profile from numerical spline
        get_suffixed_filename("data/f_of_E.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && Evalues && innerintegrandvalues) {
            for (i = 0; i <= num_points; i++) {
                double E = Evalues[i];
                double deriv = 0.0;
                if (i > 0 && i < num_points) {
                    deriv = gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, innerintegrandvalues, E, g_main_fofEacc);
                }
                double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                if (!isfinite(fE)) fE = 0.0;
                fprintf_bin(fp, "%f %f\n", E, fE);
            }
            fclose(fp);
        } else {
            log_message("ERROR", "Failed to open %s for final Hernquist (num) f(E) profile", suffixed_filename);
        }

        // Write Hernquist distribution function at a fixed radius using numerical f(E) spline
        if (!skip_file_writes) {
            get_suffixed_filename("data/df_fixed_radius.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            fp = fopen(suffixed_filename, "wb");
            if (fp && splinePsi && Psiinterp && Psivalues && num_points > 0 && g_main_fofEinterp && g_main_fofEacc) {
                double r_F = 2.0 * g_scale_radius_param;
                double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_F);
                Psi_rf *= VEL_CONV_SQ;
                double Psimin_local = Psivalues[num_points - 1];
                double Psimin_test = VEL_CONV_SQ * Psimin_local;

                int vsteps = 1000;
                for (int vv = 0; vv <= vsteps; vv++) {
                    double sqrt_arg_v = Psi_rf - Psimin_test;
                    if (sqrt_arg_v < 0) sqrt_arg_v = 0;
                    double vtest = (double)vv * (sqrt(2.0 * sqrt_arg_v) / vsteps);
                    double vtest_codes = vtest / kmsec_to_kpcmyr;
                    double fEval = 0.0;

                    if (g_use_om_profile && !g_use_numerical_isotropic) {
                        // OM: integrate f(Q) over μ using GSL adaptive quadrature
                        double Psi_rf_codes = Psi_rf / VEL_CONV_SQ;
                        double v_sq = vtest_codes * vtest_codes;
                        double r_over_ra_sq = (r_F / g_om_anisotropy_radius) * (r_F / g_om_anisotropy_radius);

                        om_mu_integral_params params_mu = {v_sq, Psi_rf_codes, r_over_ra_sq,
                                                            Psimin, Psimax, g_main_fofEinterp,
                                                            Evalues, innerintegrandvalues, g_main_fofEacc};
                        gsl_function F_mu;
                        F_mu.function = &om_mu_integrand;
                        F_mu.params = &params_mu;

                        double integral_result, error;
                        gsl_integration_workspace *w_mu = gsl_integration_workspace_alloc(10000);
                        int status = gsl_integration_qag(&F_mu, 0.0, 1.0, 1e-12, 1e-12, 10000,
                                                         GSL_INTEG_GAUSS61, w_mu, &integral_result, &error);
                        gsl_integration_workspace_free(w_mu);

                        if (status == GSL_SUCCESS) {
                            fEval = 2.0 * vtest_codes * vtest_codes * r_F * r_F * integral_result;
                        }
                    } else {
                        // Isotropic: standard evaluation
                        double Etest = Psi_rf - 0.5 * vtest * vtest;
                        double Etest_codes = Etest / VEL_CONV_SQ;
                        if (Etest_codes >= Psimin && Etest_codes <= Psimax) {
                            double derivative;
                            int status = gsl_interp_eval_deriv_e(g_main_fofEinterp, Evalues, innerintegrandvalues,
                                                                 Etest_codes, g_main_fofEacc, &derivative);
                            if (status == GSL_SUCCESS) {
                                fEval = derivative / (sqrt(8.0) * PI * PI) * vtest_codes * vtest_codes * r_F * r_F;
                            }
                        }
                    }
                    if (!isfinite(fEval)) fEval = 0.0;
                    fprintf_bin(fp, "%f %f\n", vtest, fEval);
                }
                fclose(fp);
            } else {
                if (!fp) {
                    log_message("ERROR", "Failed to open %s for final Hernquist (num) df_fixed_radius", suffixed_filename);
                } else {
                    log_message("WARNING", "Skipping Hernquist (num) df_fixed_radius output - required data not initialized");
                    if (fp) fclose(fp);
                }
            }
        }
    } else {
        /**
         * @brief Write final theoretical Cored Plummer-like profile characteristics to .dat files.
         * @details This block outputs several files (massprofile, Psiprofile, density_profile,
         *          dpsi_dr, drho_dpsi, f_of_E, df_fixed_radius) using splines and parameters
         *          from Cored Plummer initial condition generation (splinemass, splinePsi,
         *          g_main_fofEinterp, num_points, radius, normalization, g_cored_profile_rc).
         *          Analytical formulas for the Cored density and its derivatives are used where appropriate.
         */
        log_message("INFO", "Writing Cored theoretical profiles to final .dat files...");

        // Write Cored theoretical mass profile
        get_suffixed_filename("data/massprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                    fprintf_bin(fp, "%f %f\n", r_plot, gsl_spline_eval(splinemass, r_plot, enclosedmass));
                }
            }
            if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], gsl_spline_eval(splinemass, radius[num_points-1], enclosedmass));
            }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final cored mass profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping cored mass profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write Cored theoretical potential profile
        get_suffixed_filename("data/Psiprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinePsi && Psiinterp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                     fprintf_bin(fp, "%f %f\n", r_plot, evaluatespline(splinePsi, Psiinterp, r_plot));
                }
            }
             if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], evaluatespline(splinePsi, Psiinterp, radius[num_points-1]));
             }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final cored Psi profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping cored Psi profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write Cored theoretical density profile
        get_suffixed_filename("data/density_profile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                double rho_r = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(rr / g_cored_profile_rc)));
                fprintf_bin(fp, "%f %f\n", rr, rho_r);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final cored density profile", suffixed_filename); 
        }

        // Write Cored theoretical dPsi/dr profile
        get_suffixed_filename("data/dpsi_dr.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                if (rr > 0.0) {
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                    double dpsidr = -(G_CONST * Menc) / (rr * rr);
                    fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                }
            }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final cored dpsi/dr profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping cored dpsi/dr profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write Cored theoretical drho/dPsi profile
        get_suffixed_filename("data/drho_dpsi.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp && splinemass && enclosedmass && splinePsi && Psiinterp) {
            for (i = 1; i < num_points - 1; i++) {
                double rr = radius[i];
                double rho_left = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i - 1] / g_cored_profile_rc)));
                double rho_right = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i + 1] / g_cored_profile_rc)));
                double drho_dr_num = (rho_right - rho_left) / (radius[i + 1] - radius[i - 1]);
                double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                double dPsidr = -(G_CONST * Menc) / (rr * rr);
                if (dPsidr != 0.0) {
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    fprintf_bin(fp, "%f %f\n", Psi_val, drho_dr_num / dPsidr);
                }
            }
            fclose(fp);
        } else { 
            if (!fp) {
                log_message("ERROR", "Failed to open %s for final cored drho/dpsi profile", suffixed_filename);
            } else {
                log_message("WARNING", "Skipping cored drho/dpsi profile output - splines not initialized");
                fclose(fp);
            }
        }

        // Write theoretical f(E) profile
        get_suffixed_filename("data/f_of_E.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i <= num_points; i++) {
                double E = Evalues[i];
                double deriv = 0.0;
                if (i > 0 && i < num_points + 1) {
                    if (i > 0 && i < num_points) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i - 1]) / (Evalues[i + 1] - Evalues[i - 1]);
                    }
                    else if (i == 0) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i]) / (Evalues[i + 1] - Evalues[i]);
                    }
                    else if (i == num_points) {
                        deriv = (innerintegrandvalues[i] - innerintegrandvalues[i - 1]) / (Evalues[i] - Evalues[i - 1]);
                    }
                }
                double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                if (E == 0.0 || !isfinite(fE))
                    fE = 0.0;
                fprintf_bin(fp, "%f %f\n", E, fE);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final f(E) profile (%s)", suffixed_filename, g_use_nfw_profile ? "NFW" : "Cored"); 
        }

        // Write distribution function at fixed radius (conditional on skip_file_writes and spline availability)
        if (!skip_file_writes && splinePsi && Psiinterp && g_main_fofEinterp && g_main_fofEacc) {
            get_suffixed_filename("data/df_fixed_radius.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            fp = fopen(suffixed_filename, "wb");
            if (fp) {
                double r_F = 2.0 * g_cored_profile_rc;  // r_F = 2 × Cored scale radius
                double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_F);
                Psi_rf *= VEL_CONV_SQ;
                double Psimin_test = VEL_CONV_SQ * Psimin; // Convert to (km/s)² for velocity calculation

                int vsteps = 10000;
                int reduce_vsteps = 300;
                for (int vv = 0; vv <= vsteps - reduce_vsteps; vv++) {
                    double sqrt_arg_v = Psi_rf - Psimin_test;
                    if (sqrt_arg_v < 0) sqrt_arg_v = 0;
                    double vtest = (double)vv * (sqrt(2.0 * sqrt_arg_v) / (vsteps));
                    double vtest_codes = vtest / kmsec_to_kpcmyr;
                    double fEval = 0.0;

                    if (g_use_om_profile) {
                        // OM: integrate f(Q) over μ using GSL adaptive quadrature
                        double Psi_rf_codes = Psi_rf / VEL_CONV_SQ;
                        double v_sq = vtest_codes * vtest_codes;
                        double r_over_ra_sq = (r_F / g_om_anisotropy_radius) * (r_F / g_om_anisotropy_radius);

                        om_mu_integral_params params_mu = {v_sq, Psi_rf_codes, r_over_ra_sq,
                                                            Psimin, Psimax, g_main_fofEinterp,
                                                            Evalues, innerintegrandvalues, g_main_fofEacc};
                        gsl_function F_mu;
                        F_mu.function = &om_mu_integrand;
                        F_mu.params = &params_mu;

                        double integral_result, error;
                        gsl_integration_workspace *w_mu = gsl_integration_workspace_alloc(10000);
                        int status = gsl_integration_qag(&F_mu, 0.0, 1.0, 1e-12, 1e-12, 10000,
                                                         GSL_INTEG_GAUSS61, w_mu, &integral_result, &error);
                        gsl_integration_workspace_free(w_mu);

                        if (status == GSL_SUCCESS) {
                            fEval = 2.0 * vtest_codes * vtest_codes * r_F * r_F * integral_result;
                        }
                    } else {
                        // Isotropic: standard evaluation
                        double Etest = Psi_rf - 0.5 * vtest * vtest;
                        double Etest_codes = Etest / VEL_CONV_SQ;
                        if (Etest_codes >= Psimin && Etest_codes <= Psimax) {
                            double derivative;
                            int status = gsl_interp_eval_deriv_e(g_main_fofEinterp, Evalues, innerintegrandvalues,
                                                                 Etest_codes, g_main_fofEacc, &derivative);
                            if (status == GSL_SUCCESS) {
                                fEval = derivative / (sqrt(8.0) * PI * PI) * vtest * vtest * r_F * r_F;
                            }
                        }
                    }
                    if (!isfinite(fEval)) fEval = 0.0;
                    fprintf_bin(fp, "%f %f\n", vtest, fEval);
                }
                fclose(fp);
            } else { 
                log_message("ERROR", "Failed to open %s for final df_fixed_radius (%s)", suffixed_filename, g_use_nfw_profile ? "NFW" : "Cored"); 
            }
        } else if (!skip_file_writes) {
            log_message("WARNING", "Skipping df_fixed_radius output - splines not initialized");
        }
    }

    get_suffixed_filename("data/particles.dat", 1, suffixed_filename, sizeof(suffixed_filename));
    FILE *finit = fopen(suffixed_filename, "rb"); // Binary mode for fscanf_bin
    if (!finit)
    {
        printf("Error: can't open %s\n", suffixed_filename);
        CLEAN_EXIT(1);
    }
    double *r_initial = (double *)malloc(npts * sizeof(double));
    double *rv_initial = (double *)malloc(npts * sizeof(double));
    double *v_initial = (double *)malloc(npts * sizeof(double));
    double *l_initial = (double *)malloc(npts * sizeof(double));
    double rank_id;
    for (i = 0; i < npts; i++)
    {
        fscanf_bin(finit, "%f %f %f %f\n", &r_initial[i], &rv_initial[i], &l_initial[i], &rank_id);
    }
    fclose(finit);

    for (i = 0; i < npts; i++)
    {
        v_initial[i] = sqrt(rv_initial[i] * rv_initial[i] + l_initial[i] * l_initial[i] / (r_initial[i] * r_initial[i]));
    }

    double *r_final = (double *)malloc(npts * sizeof(double));
    double *v_final = (double *)malloc(npts * sizeof(double));
    for (i = 0; i < npts; i++)
    {
        r_final[i] = particles[0][i];
        v_final[i] = sqrt(particles[1][i] * particles[1][i] + particles[2][i] * particles[2][i] / (particles[0][i] * particles[0][i]));
    }

    /**
     * @brief Calculate percentile-based ranges for dynamic histogram binning.
     * @details Combines initial and final particle distributions to determine
     * appropriate histogram ranges using the 99th percentile with 1.2x padding.
     * This ensures virtually all particles are captured while avoiding outliers.
     */
    double *r_all_sorted = (double *)malloc(2 * npts * sizeof(double));
    double *v_all_sorted = (double *)malloc(2 * npts * sizeof(double));

    // Combine initial and final data for percentile calculation
    for (i = 0; i < npts; i++) {
        r_all_sorted[i] = r_initial[i];
        r_all_sorted[npts + i] = r_final[i];
        v_all_sorted[i] = fabs(v_initial[i]) * (1.0 / kmsec_to_kpcmyr); // Convert to km/s
        v_all_sorted[npts + i] = fabs(v_final[i]) * (1.0 / kmsec_to_kpcmyr); // Convert to km/s
    }

    // Sort arrays to find percentiles
    qsort(r_all_sorted, 2 * npts, sizeof(double), double_cmp);
    qsort(v_all_sorted, 2 * npts, sizeof(double), double_cmp);

    // Calculate 99th percentile with 1.2x multiplier for padding
    int p99_index = (int)(0.99 * (2 * npts - 1));
    double max_r_all = r_all_sorted[p99_index] * 1.2;
    double max_v_all = v_all_sorted[p99_index] * 1.2;

    // Ensure minimum ranges for very concentrated distributions
    if (max_r_all < 50.0) max_r_all = 50.0;   // Minimum 50 kpc
    if (max_v_all < 50.0) max_v_all = 50.0;   // Minimum 50 km/s

    // Log the calculated ranges
    if (g_enable_logging) {
        log_message("INFO", "2D Histogram dynamic ranges: r=[0, %.1f] kpc, v=[0, %.1f] km/s", 
                    max_r_all, max_v_all);
    }

    free(r_all_sorted);
    free(v_all_sorted);

    // Histogram bin widths calculated from dynamic ranges
    double rbin_width = max_r_all / HIST_NBINS;
    double vbin_width = max_v_all / HIST_NBINS;

    double bin_width = max_r_all / HIST_NBINS;  // For 1D histogram

    /**
     * @brief Generate and write 2D phase-space histograms (radius vs. velocity magnitude).
     * @details Calculates 2D histograms of particle counts in (r, |v|) bins for both the
     *          initial (t=0, after IC generation and any stripping/conversion) and final
     *          (end of simulation) particle distributions. The bin ranges are dynamically
     *          determined based on the 99th percentile of the combined initial and final
     *          radial positions and velocity magnitudes, with a 1.2x padding factor.
     *          Outputs `2d_hist_initial.dat` and `2d_hist_final.dat`.
     */
    int hist_initial[HIST_NBINS][HIST_NBINS];
    memset(hist_initial, 0, sizeof(hist_initial));
    int hist_final[HIST_NBINS][HIST_NBINS];
    memset(hist_final, 0, sizeof(hist_final));

    for (i = 0; i < npts; i++)
    {
        int rbini = (int)(r_initial[i] / rbin_width);
        int vbini = (int)(fabs(v_initial[i]) * (1.0 / kmsec_to_kpcmyr) / vbin_width);
        if (rbini < HIST_NBINS && vbini < HIST_NBINS && rbini >= 0 && vbini >= 0)
            hist_initial[rbini][vbini]++;

        rbini = (int)(r_final[i] / rbin_width);
        vbini = (int)(fabs(v_final[i]) * (1.0 / kmsec_to_kpcmyr) / vbin_width);
        if (rbini < HIST_NBINS && vbini < HIST_NBINS && rbini >= 0 && vbini >= 0)
            hist_final[rbini][vbini]++;
    }

    // Write initial 2D histogram (skipped if skip_file_writes active)
    if (!skip_file_writes)
    {
        char suffixed_filename[256];
        get_suffixed_filename("data/2d_hist_initial.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int rr = 0; rr < HIST_NBINS; rr++)
        {
            for (int vv = 0; vv < HIST_NBINS; vv++)
            {
                fprintf_bin(fp, "%f %f %d\n", (rr + 0.5) * rbin_width, (vv + 0.5) * vbin_width, hist_initial[rr][vv]);
            }
            fprintf_bin(fp, "\n");
        }
        fclose(fp);
    } // End skip_file_writes block.

    // Write final 2D histogram (skipped if skip_file_writes active)
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/2d_hist_final.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int rr = 0; rr < HIST_NBINS; rr++)
        {
            for (int vv = 0; vv < HIST_NBINS; vv++)
            {
                fprintf_bin(fp, "%f %f %d\n", (rr + 0.5) * rbin_width, (vv + 0.5) * vbin_width, hist_final[rr][vv]);
            }
            fprintf_bin(fp, "\n");
        }
        fclose(fp);
    } // End skip_file_writes block.

    /**
     * @brief Generate and write 1D radial distribution histograms.
     * @details Calculates 1D histograms of particle counts in radial bins for both the
     *          initial and final particle distributions. Uses the same dynamically determined
     *          radial binning as the 2D histograms.
     *          Outputs `combined_histogram.dat` with columns: r_bin_center, count_initial, count_final.
     */
    int hist_i[HIST_NBINS];
    memset(hist_i, 0, sizeof(hist_i));
    int hist_f[HIST_NBINS];
    memset(hist_f, 0, sizeof(hist_f));

    for (i = 0; i < npts; i++)
    {
        int b = (int)(r_initial[i] / bin_width);
        if (b < HIST_NBINS && b >= 0)
            hist_i[b]++;
        b = (int)(r_final[i] / bin_width);
        if (b < HIST_NBINS && b >= 0)
            hist_f[b]++;
    }

    // Write combined 1D radius histogram (skipped if skip_file_writes active)
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/combined_histogram.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (i = 0; i < HIST_NBINS; i++)
        {
            double bin_center = (i + 0.5) * bin_width;
            fprintf_bin(fp, "%f %d %d\n", bin_center, hist_i[i], hist_f[i]);
        }
        fclose(fp);
    } // End skip_file_writes block.

    // Trajectory data is now written incrementally via the flush_trajectory_buffers() function,
    // which is called during the main simulation loop. The final flush and file closing
    // are handled by the cleanup_trajectory_buffers() function.


    // When restarting, adjust total_writes to include ALL data (pre-restart + post-restart)
    if (g_restart_mode_active && original_Ntimes > 0) {
        // Recalculate total_writes based on original simulation parameters
        int original_total_writes = ((original_Ntimes - 1) / dtwrite) + 1;
        printf("\nAdjusting post-processing for complete dataset:\n");
        printf("  - Original total writes: %d\n", original_total_writes);
        printf("  - Current total_writes: %d\n", total_writes);
        printf("  - Using original value for complete data processing\n\n");
        total_writes = original_total_writes;
    }

    int snapshot_steps[noutsnaps];
    // Determine the timesteps corresponding to the desired snapshot outputs
    for (int s = 0; s < noutsnaps; s++)
    {
        snapshot_steps[s] = (int)floor(
            s * (total_writes - 1) / (double)(noutsnaps - 1));
        log_message("DEBUG", "snapshot_steps[%d] = %d", s, snapshot_steps[s]);
    }

    if (g_doAllParticleData)
    {
        size_t total_outsnaps_size = (size_t)npts * noutsnaps; // Use size_t for large allocations
        log_message("INFO", "Allocating arrays for %zu particles across %d snapshots (total: %zu elements)",
                    (size_t)npts, noutsnaps, total_outsnaps_size);

        int *Rank_partdata_outsnaps = (int *)malloc(total_outsnaps_size * sizeof(int));
        float *R_partdata_outsnaps = (float *)malloc(total_outsnaps_size * sizeof(float));
        float *Vrad_partdata_outsnaps = (float *)malloc(total_outsnaps_size * sizeof(float));

        if (!Rank_partdata_outsnaps || !R_partdata_outsnaps || !Vrad_partdata_outsnaps)
        {
            printf("[ERROR] Failed to allocate memory for output arrays. Aborting.\n");
            fflush(stdout);
            CLEAN_EXIT(1);
        }

        char apd_filename_for_read[256];
        // If restart-file was specified, use it for reading, otherwise use the normal all_particle_data file
        if (g_restart_file_override != NULL) {
            strncpy(apd_filename_for_read, g_restart_file_override, sizeof(apd_filename_for_read) - 1);
            apd_filename_for_read[sizeof(apd_filename_for_read) - 1] = '\0';
            printf("Reading particle data from restart file: %s\n", apd_filename_for_read);
        } else {
            get_suffixed_filename("data/all_particle_data.dat", 1, apd_filename_for_read, sizeof(apd_filename_for_read));
        }

        double fixed_bin_width = 0.0; // Initialized, set during first snapshot processing

        int start_index = 0; // Default start index for snapshot processing loop
        if (g_doRestart)
        {
            if (g_doRestartForce) {
                // Force mode: always start from 0 and regenerate all snapshots
                start_index = 0;
                printf("FORCE mode: Regenerating ALL snapshots regardless of existing files.\n");
                log_message("INFO", "Restart FORCE mode: Regenerating all snapshots from index 0.");
            } else {
                // Normal restart mode: check for existing snapshots
                start_index = find_last_processed_snapshot(snapshot_steps, noutsnaps);
                if (start_index == -2) // Indicates all snapshots already processed
                {
                    log_message("INFO", "All Rank files already exist for suffix '%s'. Skipping snapshot regeneration.", g_file_suffix);
                    goto cleanup_partdata_outsnaps; // Jump past the processing loop
                }
                else if (start_index < 0) // Indicates error or no files found
                {
                    start_index = 0; // Default to starting from the beginning
                }
                // Otherwise, start_index is the index of the first snapshot *to be* processed
                log_message("INFO", "Restart: Starting snapshot processing from index %d.", start_index);
            }
        }

        printf("Starting parallel processing of %d snapshots (from %d to %d) using %d threads\n\n",
               noutsnaps - start_index, start_index, noutsnaps - 1, omp_get_max_threads());
        log_message("INFO", "Starting parallel processing with %d threads for %d snapshots (from index %d to %d)",
                    omp_get_max_threads(), noutsnaps - start_index, start_index, noutsnaps - 1);

#pragma omp parallel for schedule(static, 1) ordered
        for (int s = start_index; s < noutsnaps; s++)
        {
            int snap = snapshot_steps[s];

            float *tmpL_partdata_snap = (float *)malloc(npts * sizeof(float));
            int *tmpRank_partdata_snap = (int *)malloc(npts * sizeof(int));
            float *tmpR_partdata_snap = (float *)malloc(npts * sizeof(float));
            float *tmpV_partdata_snap = (float *)malloc(npts * sizeof(float));

            if (!tmpL_partdata_snap || !tmpRank_partdata_snap ||
                !tmpR_partdata_snap || !tmpV_partdata_snap)
            {
                fprintf(stderr, "Error: out of memory in parallel loop.\n");
                CLEAN_EXIT(1);
            }

            retrieve_all_particle_snapshot(
                apd_filename_for_read,
                snap,       // Which snap to read.
                npts,       // #particles
                snapshot_block_size, // Block size.
                tmpL_partdata_snap,
                tmpRank_partdata_snap,
                tmpR_partdata_snap,
                tmpV_partdata_snap);

/**
 * Process snapshot data in order.
 * The #pragma omp ordered block ensures the copying into output arrays
 * (Rank_partdata_outsnaps, etc.) and status messages happen sequentially.
 */
#pragma omp ordered
            {
                // Display status message sequentially.
                printf("Processing snapshot %d...\n", snap);

                // Check array validity before copying
                if (!Rank_partdata_outsnaps || !R_partdata_outsnaps || !Vrad_partdata_outsnaps)
                {
                    log_message("ERROR", "Thread %d: Output arrays not properly allocated for snapshot %d",
                                omp_get_thread_num(), snap);
                }
                else if (!tmpRank_partdata_snap || !tmpR_partdata_snap || !tmpV_partdata_snap)
                {
                    log_message("ERROR", "Thread %d: Input arrays not properly allocated for snapshot %d",
                                omp_get_thread_num(), snap);
                }
                else
                {
                    // Copy data from thread-local buffers to the large shared output arrays.
                    for (int ii = 0; ii < npts; ii++)
                    {
                        if ((size_t)(s * npts + ii) < total_outsnaps_size)
                        {
                            Rank_partdata_outsnaps[s * npts + ii] = tmpRank_partdata_snap[ii];
                            R_partdata_outsnaps[s * npts + ii] = tmpR_partdata_snap[ii];
                            Vrad_partdata_outsnaps[s * npts + ii] = tmpV_partdata_snap[ii];
                        }
                        else
                        {
                            log_message("ERROR", "Thread %d: Index %zu out of bounds (%zu) in ordered copy for snapshot %d", 
                                        omp_get_thread_num(), (size_t)(s * npts + ii), total_outsnaps_size, snap);
                            break;
                        }
                    }
                }
            }

            /** @brief Build "unsorted" arrays (Rank, Mass, R, Vrad, L) from local snapshot data. */
            log_message("INFO", "Thread %d: Allocating arrays for snapshot %d",
                        omp_get_thread_num(), snap);

            int *Rank_unsorted = (int *)malloc(npts * sizeof(int));
            double *Mass_unsorted = (double *)malloc(npts * sizeof(double));
            double *R_unsorted = (double *)malloc(npts * sizeof(double));
            double *Vrad_unsorted = (double *)malloc(npts * sizeof(double));
            double *L_unsorted = (double *)malloc(npts * sizeof(double));

            if (!Rank_unsorted || !Mass_unsorted || !R_unsorted || !Vrad_unsorted || !L_unsorted)
            {
                printf("[ERROR] Thread %d: Memory allocation failed for snapshot %d\n",
                       omp_get_thread_num(), snap);
                // Free any memory that was allocated.
                if (Rank_unsorted)
                    free(Rank_unsorted);
                if (Mass_unsorted)
                    free(Mass_unsorted);
                if (R_unsorted)
                    free(R_unsorted);
                if (Vrad_unsorted)
                    free(Vrad_unsorted);
                if (L_unsorted)
                    free(L_unsorted);
                continue; // Skip to next snapshot.
            }

            for (int ii = 0; ii < npts; ii++)
            {
                int rankval = tmpRank_partdata_snap[ii];
                double massv = (rankval + 1) * deltaM;

                Rank_unsorted[ii] = rankval;
                Mass_unsorted[ii] = massv;
                R_unsorted[ii] = (double)tmpR_partdata_snap[ii];
                Vrad_unsorted[ii] = (double)tmpV_partdata_snap[ii];
                L_unsorted[ii] = (double)tmpL_partdata_snap[ii];
            }

            // Sort by Rank
            // --- Prepare data structure for sorting by radius ---
            /** @brief Allocate `partarr` (array of struct PartData) for sorting snapshot data. */
            struct PartData *partarr = malloc(npts * sizeof(struct PartData));
            if (!partarr)
            {
                log_message("ERROR", "Thread %d: Failed to allocate partarr for snapshot %d",
                            omp_get_thread_num(), snap);
                // Free allocated memory.
                free(Rank_unsorted);
                free(Mass_unsorted);
                free(R_unsorted);
                free(Vrad_unsorted);
                free(L_unsorted);
                continue;
            }

            struct PartData *local_partarr = (struct PartData *)malloc(npts * sizeof(struct PartData));
            if (!local_partarr)
            {
                log_message("ERROR", "Thread %d: Failed to allocate local_partarr for snapshot %d",
                            omp_get_thread_num(), snap);
                free(partarr);
                continue;
            }

            for (int ii = 0; ii < npts; ii++)
            {
                // Check for invalid values to prevent sort issues.
                float rad_val = tmpR_partdata_snap[ii];
                float vrad_val = tmpV_partdata_snap[ii];
                float angmom_val = tmpL_partdata_snap[ii];

                // Handle potential NaN values before assigning to struct
                if (rad_val != rad_val) { // Check for NaN (safe with fast-math)
                    log_message("WARNING", "Thread %d: NaN radius at index %d for snapshot %d, replaced with 0", 
                                omp_get_thread_num(), ii, snap);
                    rad_val = 0.0f;
                }
                if (vrad_val != vrad_val) vrad_val = 0.0f; // Check for NaN (safe with fast-math)
                if (angmom_val != angmom_val) angmom_val = 0.0f; // Check for NaN (safe with fast-math)

                local_partarr[ii].rank = tmpRank_partdata_snap[ii];
                local_partarr[ii].rad = rad_val;
                local_partarr[ii].vrad = vrad_val;
                local_partarr[ii].angmom = angmom_val;
                local_partarr[ii].original_index = ii; // Store original index before sort
            }

            memcpy(partarr, local_partarr, npts * sizeof(struct PartData));
            free(local_partarr);

            log_message("INFO", "Thread %d: Sorting partarr by radius for snapshot %d", 
                        omp_get_thread_num(), snap);

            // Sort particle data by radius for snapshot processing
            sort_by_rad(partarr, npts);

            // If processing the first snapshot (index start_index), write lowest radius IDs
            if (s == start_index)
            {
                // Skip writing if file writes are disabled (e.g., restart post-processing only)
                if (!skip_file_writes)
                {
                    char fname_lowest_ids[256];
                    get_suffixed_filename("data/lowest_radius_ids.dat", 1, fname_lowest_ids, sizeof(fname_lowest_ids));
                    log_message("INFO", "Thread %d: Writing lowest radius IDs to %s",
                                omp_get_thread_num(), fname_lowest_ids);

                    FILE *id_file = fopen(fname_lowest_ids, "wb"); // Binary mode for fprintf_bin
                    if (id_file)
                    {
                        fprintf_bin(id_file, "# ID  Initial_Radius\n"); // Header (binary float for radius)

                        int num_tracked = (npts < 1000) ? npts : 1000; // Track up to 1000 particles

                        for (int i = 0; i < num_tracked; i++)
                        {
                            int original_id = partarr[i].original_index;
                            float initial_radius = partarr[i].rad;
                            fprintf_bin(id_file, "%d %f\n", original_id, initial_radius);
                        }
                        fclose(id_file);
                        log_message("INFO", "Thread %d: Successfully wrote %d lowest-radius particle IDs to %s",
                                    omp_get_thread_num(), num_tracked, fname_lowest_ids);
                    }
                    else
                    {
                        log_message("ERROR", "Thread %d: Failed to open %s for writing",
                                    omp_get_thread_num(), fname_lowest_ids);
                    }
                }
                else
                {
                    log_message("INFO", "Thread %d: Skipping lowest radius ID file creation in restart mode",
                                omp_get_thread_num());
                }
            }

            // Allocate arrays to store the data sorted by radius
            int *Rank_sorted = (int *)malloc(npts * sizeof(int));
            double *Mass_sorted = (double *)malloc(npts * sizeof(double));
            double *R_sorted = (double *)malloc(npts * sizeof(double));
            double *Vrad_sorted = (double *)malloc(npts * sizeof(double));
            double *L_sorted = (double *)malloc(npts * sizeof(double));

            if (!Rank_sorted || !Mass_sorted || !R_sorted || !Vrad_sorted || !L_sorted)
            {
                log_message("ERROR", "Thread %d: Failed to allocate sorted arrays for snapshot %d",
                            omp_get_thread_num(), snap);
                // Free allocated memory.
                if (Rank_sorted)
                    free(Rank_sorted);
                if (Mass_sorted)
                    free(Mass_sorted);
                if (R_sorted)
                    free(R_sorted);
                if (Vrad_sorted)
                    free(Vrad_sorted);
                if (L_sorted)
                    free(L_sorted);
                free(partarr);
                free(Rank_unsorted);
                free(Mass_unsorted);
                free(R_unsorted);
                free(Vrad_unsorted);
                free(L_unsorted);
                continue;
            }

            // Populate the sorted arrays using the sorted partarr
            for (int ii = 0; ii < npts; ii++)
            {
                partarr[ii].rank = ii; // Update rank based on sorted position
                int r_val = partarr[ii].rank; // Rank equals index after sort
                Rank_sorted[ii] = r_val;
                Mass_sorted[ii] = (r_val + 1) * deltaM; // Mass enclosed up to this rank
                R_sorted[ii] = (double)partarr[ii].rad;
                Vrad_sorted[ii] = (double)partarr[ii].vrad;
                // Retrieve L from the unsorted array using the original index stored in partarr
                L_sorted[ii] = L_unsorted[partarr[ii].original_index];
            }

            /** @brief Set fixed bin width for density calculation using the first processed snapshot. */
            log_message("INFO", "Thread %d: Checking fixed bin width for snapshot %d", 
                        omp_get_thread_num(), snap);
            if (s == start_index) // Only calculate on the first snapshot processed in this run
            {
                // Determine number of bins based on particle count (e.g., proportional to npts^(1/3))
                int num_bins_fixed = ceil(100 * pow(npts / 10000.0, 0.3333333333));
                if (num_bins_fixed < 1) num_bins_fixed = 1;
                // Use the maximum radius found in this first snapshot
                double r_max_first = R_sorted[npts - 1];
                if (r_max_first <= 0 || num_bins_fixed <= 0) {
                    log_message("ERROR", "Thread %d: Invalid r_max (%f) or bins (%d) for fixed_bin_width calc", 
                                omp_get_thread_num(), r_max_first, num_bins_fixed);
                    // Set a default or handle error
                    fixed_bin_width = 1.0;
                } else {
                    fixed_bin_width = r_max_first / num_bins_fixed;
                }
                log_message("INFO", "Thread %d: Calculated fixed_bin_width=%f for snapshot %d (r_max=%f, bins=%d)",
                            omp_get_thread_num(), fixed_bin_width, snap, r_max_first, num_bins_fixed);
            }

            double *density_sorted = NULL;

            log_message("INFO", "Thread %d: Starting density calculation for snapshot %d",
                        omp_get_thread_num(), snap);
            {
                // Free previous allocation if it exists (safety measure)
                if (density_sorted != NULL)
                {
                    free(density_sorted);
                    density_sorted = NULL; // Avoid dangling pointer
                    log_message("INFO", "Thread %d: Freed previous density_sorted for snapshot %d",
                                omp_get_thread_num(), snap);
                }

                log_message("INFO", "Thread %d: Allocating density_sorted for snapshot %d",
                            omp_get_thread_num(), snap);
                density_sorted = malloc(npts * sizeof(double));
                if (!density_sorted)
                {
                    fprintf(stderr, "Error: Failed to allocate memory for density_sorted\n");
                    // Cleanup memory allocated within this snapshot's loop iteration
                    free(tmpL_partdata_snap); free(tmpRank_partdata_snap); free(tmpR_partdata_snap); free(tmpV_partdata_snap);
                    free(Rank_unsorted); free(Mass_unsorted); free(R_unsorted); free(Vrad_unsorted); free(L_unsorted);
                    free(partarr);
                    free(Rank_sorted); free(Mass_sorted); free(R_sorted); free(Vrad_sorted); free(L_sorted);
                    continue; // Proceed to the next snapshot
                }

                double bandwidth_factor = 0.08;
                log_message("INFO", "Thread %d: Checking R_sorted monotonicity for snapshot %d",
                            omp_get_thread_num(), snap);
                int r_violations = 0;
                for (int i = 1; i < npts; i++)
                {
                    if (R_sorted[i] <= R_sorted[i - 1])
                        r_violations++;
                }
                log_message("INFO", "Thread %d: Found %d radius violations for snapshot %d",
                            omp_get_thread_num(), r_violations, snap);

                double *R_filtered = NULL;
                double *Mass_filtered = NULL;
                int filtered_count = 0;

                if (r_violations > 0)
                {
                    // Allocate arrays to hold strictly monotonic radius/mass data
                    R_filtered = malloc(npts * sizeof(double));
                    Mass_filtered = malloc(npts * sizeof(double));
                    if (!R_filtered || !Mass_filtered) { 
                        log_message("ERROR", "Thread %d: Failed to allocate filtered arrays for snapshot %d", 
                            omp_get_thread_num(), snap);
                        continue; 
                    }

                    // Copy only the points that maintain strict monotonicity
                    R_filtered[0] = R_sorted[0];
                    Mass_filtered[0] = Mass_sorted[0];
                    filtered_count = 1;

                    for (int i = 1; i < npts; i++)
                    {
                        if (R_sorted[i] > R_filtered[filtered_count - 1])
                        {
                            R_filtered[filtered_count] = R_sorted[i];
                            Mass_filtered[filtered_count] = Mass_sorted[i];
                            filtered_count++;
                        }
                    }
                }
                else
                {
                    // If no violations, point directly to the sorted data (no copy needed)
                    R_filtered = R_sorted;
                    Mass_filtered = Mass_sorted;
                    filtered_count = npts;
                }

                // Decimate the filtered data for spline interpolation efficiency
                int decimated_size = imin((int)ceil(npts / 100.0), 30000); // Limit max size
                if (decimated_size <= 0) decimated_size = 1; // Ensure at least one point
                double *R_decimated = malloc(decimated_size * sizeof(double));
                double *Mass_decimated = malloc(decimated_size * sizeof(double));
                if (!R_decimated || !Mass_decimated) { 
                    log_message("ERROR", "Thread %d: Failed to allocate decimated arrays for snapshot %d", 
                        omp_get_thread_num(), snap);
                    if (r_violations > 0) { free(R_filtered); free(Mass_filtered); }
                    continue; 
                }

                if (filtered_count <= decimated_size)
                {
                    // Use all filtered points if fewer than desired decimated size
                    decimated_size = filtered_count;
                    memcpy(R_decimated, R_filtered, filtered_count * sizeof(double));
                    memcpy(Mass_decimated, Mass_filtered, filtered_count * sizeof(double));
                }
                else
                {
                    // Sample evenly from the filtered data
                    double step = (double)(filtered_count - 1) / (decimated_size - 1);
                    for (int i = 0; i < decimated_size; i++)
                    {
                        int idx = (int)round(i * step); // Use round for potentially better sampling
                        if (idx >= filtered_count)
                            idx = filtered_count - 1;
                        R_decimated[i] = R_filtered[idx];
                        Mass_decimated[i] = Mass_filtered[idx];
                    }
                }

                // Validate the range and finiteness of the decimated radius data for spline
                double min_r_check = (decimated_size > 0) ? R_decimated[0] : -1.0;
                double max_r_check = (decimated_size > 0) ? R_decimated[decimated_size - 1] : -1.0;
                // Check for invalid size, non-positive min, inverted range, or non-finite values (NaN or Inf)
                if (decimated_size <= 0 || min_r_check <= 0.0 || max_r_check <= min_r_check ||
                    ((min_r_check != min_r_check) || !(fabs(min_r_check) <= DBL_MAX)) || // Check !isfinite(min_r_check)
                    ((max_r_check != max_r_check) || !(fabs(max_r_check) <= DBL_MAX))    // Check !isfinite(max_r_check)
                   ) {
                    log_message("ERROR", "Thread %d: Invalid radius range [%f, %f] with %d points after decimation for snapshot %d", 
                                omp_get_thread_num(), min_r_check, max_r_check, decimated_size, snap);
                    free(R_decimated); free(Mass_decimated);
                    if (r_violations > 0) { free(R_filtered); free(Mass_filtered); } // Free if allocated
                    free(density_sorted); density_sorted = NULL;
                    continue;
                }

                // Define the range and parameters for the uniform log-spaced grid
                double min_r = R_decimated[0];
                double max_r = R_decimated[decimated_size - 1] * 1.04; // Extend range slightly
                double log_min_r = log10(min_r);
                double log_max_r = log10(max_r);

                // Define size for the uniform grid (used for convolution)
                int grid_size = 131072; // Power of 2 often good for FFT, but direct used here
                if (grid_size <= 1) grid_size = 2; // Ensure at least 2 points
                double dlog = (log_max_r - log_min_r) / (grid_size - 1);

                // Allocate arrays for the uniform log-spaced grid
                double *r_grid = malloc(grid_size * sizeof(double));
                double *log_r_grid = malloc(grid_size * sizeof(double));
                double *mass_grid = malloc(grid_size * sizeof(double));
                double *density_grid = malloc(grid_size * sizeof(double));
                if (!r_grid || !log_r_grid || !mass_grid || !density_grid) { 
                    log_message("ERROR", "Thread %d: Failed to allocate grid arrays for snapshot %d", 
                        omp_get_thread_num(), snap);
                    // Free previously allocated resources
                    free(R_decimated); free(Mass_decimated);
                    if (r_violations > 0) { free(R_filtered); free(Mass_filtered); }
                    if (r_grid) free(r_grid);
                    if (log_r_grid) free(log_r_grid);
                    if (mass_grid) free(mass_grid);
                    if (density_grid) free(density_grid);
                    free(density_sorted); density_sorted = NULL;
                    continue; 
                }

                // Populate the log-spaced grid coordinates
                for (int i = 0; i < grid_size; i++)
                {
                    log_r_grid[i] = log_min_r + i * dlog;
                    r_grid[i] = pow(10.0, log_r_grid[i]);
                }

                /** @brief Ensure r_grid is strictly monotonic for GSL spline init. */
                int rgrid_corrections_made = 0;
                for (int i = 1; i < grid_size; i++) {
                    if (r_grid[i] <= r_grid[i - 1]) {
                        // Ensure strict monotonicity with multiplicative increment
                        r_grid[i] = r_grid[i - 1] * (1.0 + 1e-12);
                        rgrid_corrections_made++;
                    }
                }
                if (rgrid_corrections_made > 0) {
                     log_message("WARNING", "Thread %d made %d corrections to r_grid for monotonicity in snapshot %d", omp_get_thread_num(), rgrid_corrections_made, snap);
                }

                // Interpolate mass from decimated data onto the uniform log-spaced grid
                gsl_interp_accel *acc = NULL;
                gsl_spline *mass_spline = NULL;

#pragma omp critical(gsl_accel)
                {
                    acc = gsl_interp_accel_alloc();
                }

                if (!acc)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate GSL interp accel for snapshot %d",
                                omp_get_thread_num(), snap);
                    continue;
                }

                /** @brief Ensure R_decimated is strictly monotonic for GSL spline init. */
                int corrections_made = 0;
                for (int i = 1; i < decimated_size; i++) {
                    if (R_decimated[i] <= R_decimated[i - 1]) {
                        // Ensure a strictly larger value using multiplicative increment
                        R_decimated[i] = R_decimated[i - 1] * (1.0 + 1e-12);
                        corrections_made++;
                    }
                }
                if (corrections_made > 0) {
                     log_message("WARNING", "Thread %d made %d corrections to R_decimated for monotonicity in snapshot %d", omp_get_thread_num(), corrections_made, snap);
                }

#pragma omp critical(gsl_mass_spline)
                {
                    mass_spline = gsl_spline_alloc(gsl_interp_cspline, decimated_size);
                    if (mass_spline)
                    {
                        gsl_spline_init(mass_spline, R_decimated, Mass_decimated, decimated_size);
                    }
                }

                if (!mass_spline)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate or initialize mass spline for snapshot %d",
                                omp_get_thread_num(), snap);
                    gsl_interp_accel_free(acc);
                    continue;
                }

                for (int i = 0; i < grid_size; i++)
                {
                    if (r_grid[i] < R_decimated[0])
                    {
                        mass_grid[i] = Mass_decimated[0]; // Extrapolate using first point
                    }
                    else if (r_grid[i] > R_decimated[decimated_size - 1])
                    {
                        mass_grid[i] = Mass_decimated[decimated_size - 1]; // Extrapolate using last point
                    }
                    else
                    {
#pragma omp critical(gsl_mass_eval)
                        {
                            // Use _e version for error checking if needed
                            mass_grid[i] = gsl_spline_eval(mass_spline, r_grid[i], acc);
                        }
                    }
                }

                // Calculate density = dM/dr on the uniform grid using central differences
                for (int i = 1; i < grid_size - 1; i++)
                {
                    // Avoid division by zero if grid points coincide
                    double dr = r_grid[i + 1] - r_grid[i - 1];
                    if (dr > 1e-15) {
                        double dM = mass_grid[i + 1] - mass_grid[i - 1];
                        density_grid[i] = dM / dr;
                    } else {
                        // Fallback for coincident points (e.g., use forward/backward difference or average)
                        density_grid[i] = (i > 1) ? density_grid[i-1] : 0.0; // Simple fallback
                    }

                    // Ensure non-negative density, apply floor
                    if (density_grid[i] < 1e-10)
                    {
                        density_grid[i] = 1e-10;
                    }
                }

                // Handle endpoints using one-sided difference or extrapolation
                if (grid_size >= 2) {
                   density_grid[0] = density_grid[1]; // Simple extrapolation
                   density_grid[grid_size - 1] = density_grid[grid_size - 2]; // Simple extrapolation
                } else if (grid_size == 1) {
                   density_grid[0] = 0.0; // Or some default
                }

                log_message("INFO", "Thread %d: Allocating memory for density smoothing arrays for snapshot %d",
                            omp_get_thread_num(), snap);
                double *density_smoothed_direct = malloc(npts * sizeof(double));
                double *direct_grid_result = malloc(grid_size * sizeof(double));

                if (!density_smoothed_direct || !direct_grid_result)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate memory for smoothing arrays for snapshot %d",
                                omp_get_thread_num(), snap);
                    if (density_smoothed_direct)
                        free(density_smoothed_direct);
                    if (direct_grid_result)
                        free(direct_grid_result);
                    continue;
                }

                // Calculate smoothing kernel width (sigma) based on particle count
                double sigma = bandwidth_factor * pow(npts / 10000.0, -0.40);
                double sigma_log = sigma; // Apply smoothing in log-space

                // Perform Gaussian convolution on the uniform density grid
                if (!density_grid || !log_r_grid || !direct_grid_result)
                {
                    log_message("ERROR", "Thread %d: NULL arrays detected before gaussian_convolution for snapshot %d", 
                                omp_get_thread_num(), snap);
                    // Clean up resources
                    if (density_grid) free(density_grid);
                    if (log_r_grid) free(log_r_grid);
                    if (direct_grid_result) free(direct_grid_result);
                    continue;
                }

                // Use critical section for FFTW thread safety if FFT is used internally
#pragma omp critical(gaussian_convolution)
                {
                    gaussian_convolution(density_grid, grid_size, log_r_grid, sigma_log, direct_grid_result);
                }

                // Create a GSL spline from the smoothed density on the uniform grid
                gsl_spline *density_spline = NULL;

#pragma omp critical(gsl_alloc)
                {
                    density_spline = gsl_spline_alloc(gsl_interp_cspline, grid_size);
                }

                if (!density_spline)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate GSL spline for snapshot %d",
                                omp_get_thread_num(), snap);
                    // Clean up and skip.
                    free(density_smoothed_direct);
                    free(direct_grid_result);
                    continue;
                }

#pragma omp critical(gsl_init)
                {
                    gsl_spline_init(density_spline, r_grid, direct_grid_result, grid_size);
                }

                // Temporarily disable GSL default error handler to manage errors locally
                gsl_error_handler_t *old_handler = gsl_set_error_handler_off();

                // Interpolate smoothed density back onto the original sorted particle radii (R_sorted)
                for (int i = 0; i < npts; i++)
                {
                    double r_val = R_sorted[i]; // Target radius

                    if (r_val <= r_grid[0])
                    {
                        density_smoothed_direct[i] = direct_grid_result[0]; // Extrapolate below range
                    }
                    else if (r_val >= r_grid[grid_size - 1])
                    {
                        density_smoothed_direct[i] = direct_grid_result[grid_size - 1]; // Extrapolate above range
                    }
                    else
                    {
                        // For values within range, use spline interpolation with error checking.
                        int status = 0;
                        double result = 0.0;

// Try to evaluate with error checking.
#pragma omp critical(gsl_eval)
                        {
                            status = gsl_spline_eval_e(density_spline, r_val, acc, &result);
                        }

                        if (status != GSL_SUCCESS)
                        {
                            // Spline evaluation failed; fall back to linear interpolation.
                            int idx_low = 0;

                            // Binary search to find the lower index.
                            int left = 0;
                            int right = grid_size - 1;

                            while (left <= right)
                            {
                                int mid = left + (right - left) / 2;

                                if (r_grid[mid] <= r_val && (mid == grid_size - 1 || r_grid[mid + 1] > r_val))
                                {
                                    idx_low = mid;
                                    break;
                                }
                                else if (r_grid[mid] > r_val)
                                {
                                    right = mid - 1;
                                }
                                else
                                {
                                    left = mid + 1;
                                }
                            }

                            int idx_high = idx_low + 1;

                            // Safety check.
                            if (idx_high >= grid_size)
                            {
                                idx_high = grid_size - 1;
                                idx_low = idx_high - 1;
                            }

                            // Linear interpolation.
                            double t = (r_val - r_grid[idx_low]) / (r_grid[idx_high] - r_grid[idx_low]);
                            result = direct_grid_result[idx_low] * (1.0 - t) + direct_grid_result[idx_high] * t;
                        }

                        density_smoothed_direct[i] = result;
                    }
                }

                // Restore the previous GSL error handler.
                gsl_set_error_handler(old_handler);

                // Copy the smoothed density values to the output array.
                memcpy(density_sorted, density_smoothed_direct, npts * sizeof(double));

                // Free resources.
                gsl_spline_free(mass_spline);
                gsl_spline_free(density_spline);
                gsl_interp_accel_free(acc);
                free(r_grid);
                free(log_r_grid);
                free(mass_grid);
                free(density_grid);
                free(direct_grid_result);
                free(density_smoothed_direct);
                free(R_decimated);
                free(Mass_decimated);

                // Free filtered arrays if allocated
                if (r_violations > 0)
                {
                    free(R_filtered);
                    free(Mass_filtered);
                }
            }
            // =============================================================================
            // END DENSITY CALCULATION
            // =============================================================================

            // Density calculation complete using fixed binning from R_sorted min to max with (0,0) anchor point

            if (g_doDynPsi)
            {

                double *rrA = (double *)malloc(npts * sizeof(double));
                double *M_rA = (double *)malloc(npts * sizeof(double));

                for (int ii = 0; ii < npts; ii++)
                {
                    M_rA[ii] = Mass_sorted[ii];
                    rrA[ii] = R_sorted[ii];
                }

                double *inv_rA = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    inv_rA[ii] = 1.0 / rrA[ii];
                }
                double *suffix_sum_invrA = (double *)malloc(npts * sizeof(double));
                suffix_sum_invrA[npts - 1] = inv_rA[npts - 1];
                for (int jj = npts - 2; jj >= 0; jj--)
                {
                    suffix_sum_invrA[jj] = suffix_sum_invrA[jj + 1] + inv_rA[jj];
                }

                double *psiAarr = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = rrA[ii];
                    double M_r = M_rA[ii];
                    double outer_sum = 0.0;
                    if (ii < npts - 1)
                        outer_sum = (deltaM)*suffix_sum_invrA[ii + 1];
                    double psiA = G_CONST * ((M_r / rr) + outer_sum);
                    psiA *= VEL_CONV_SQ;
                    psiAarr[ii] = psiA;
                }

                // Compute PsiA(0).
                double sum_invr = 0.0;
                for (int ii = 0; ii < npts; ii++)
                {
                    sum_invr += inv_rA[ii];
                }
                double PsiA0 = G_CONST * deltaM * sum_invr * VEL_CONV_SQ; // PsiA at r=0.

                // Write the calculated potential profile to a file.
                char fname_PsiA[256];
                char base_filename[256];
                snprintf(base_filename, sizeof(base_filename), "data/Psi_methodA_t%05d.dat", snap);
                get_suffixed_filename(base_filename, 1, fname_PsiA, sizeof(fname_PsiA));

                FILE *fA = fopen(fname_PsiA, "wb"); // Binary mode for fprintf_bin
                // Write the zero point first.
                double rr_zero = 0.0;
                fprintf_bin(fA, "%f %f\n", rr_zero, PsiA0);
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = rrA[ii];
                    fprintf_bin(fA, "%f %f\n", rr, psiAarr[ii]);
                }
                fclose(fA);

                // Allocate extended array (npts+1 points) including origin point (0, PsiA0) for spline
                double *rrA_spline = (double *)malloc((npts + 1) * sizeof(double));
                double *psiAarr_spline = (double *)malloc((npts + 1) * sizeof(double));
                rrA_spline[0] = 0.0;
                psiAarr_spline[0] = PsiA0;
                for (int ii = 0; ii < npts; ii++)
                {
                    rrA_spline[ii + 1] = rrA[ii];
                    psiAarr_spline[ii + 1] = psiAarr[ii];
                }

                // Enforce strict monotonicity for GSL spline initialization
                for (int i = 1; i <= npts; i++)
                {
                    if (rrA_spline[i] <= rrA_spline[i - 1])
                    {
                        rrA_spline[i] = rrA_spline[i - 1] * (1.0 + 1e-12);
                    }
                }

                // Create a spline for PsiA(r) including the (0,PsiA0) point
                gsl_interp_accel *PsiAinterp = gsl_interp_accel_alloc();
                gsl_spline *splinePsiA = gsl_spline_alloc(gsl_interp_linear, npts + 1);

                gsl_spline_init(splinePsiA, rrA_spline, psiAarr_spline, npts + 1);

                // Compute PsiA for all unsorted and sorted radii using this spline
                double *PsiA_unsorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr_uf = R_unsorted[ii];
                    PsiA_unsorted[ii] = gsl_spline_eval(splinePsiA, rr_uf, PsiAinterp);
                }

                double *PsiA_sorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr_sf = R_sorted[ii];
                    PsiA_sorted[ii] = gsl_spline_eval(splinePsiA, rr_sf, PsiAinterp);
                }

                // =============================================================================
                // ENERGY CALCULATION FOR PARTICLE DISTRIBUTIONS
                // =============================================================================
                //
                // Calculate total energy for each particle using the formula:
                // E = Ψ - L²/(2r²) - v²/2
                //
                // Components:
                // - Ψ: Gravitational potential (from spline interpolation)
                // - L: Angular momentum
                // - r: Radial position
                // - v: Radial velocity
                //
                // Computed separately for both unsorted and sorted distributions
                double *E_unsorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = R_unsorted[ii];      // Radius
                    double vrad = Vrad_unsorted[ii]; // Radial velocity
                    double l = L_unsorted[ii];       // Angular momentum

                    // Calculate total energy using the orbital energy equation
                    E_unsorted[ii] = PsiA_unsorted[ii]           // Potential energy (Ψ)
                                     - (l * l / (2.0 * rr * rr)) // Rotational energy (L²/2r²)
                                     - 0.5 * vrad * vrad;        // Kinetic energy (v²/2)
                }

                double *E_sorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = R_sorted[ii];      // Radius
                    double vrad = Vrad_sorted[ii]; // Radial velocity
                    double l = L_sorted[ii];       // Angular momentum

                    // Same energy formula applied to sorted distribution
                    E_sorted[ii] = PsiA_sorted[ii]             // Potential energy (Ψ)
                                   - (l * l / (2.0 * rr * rr)) // Rotational energy (L²/2r²)
                                   - 0.5 * vrad * vrad;        // Kinetic energy (v²/2)
                }

                if (g_doDebug)
                {
                    // =============================================================================
                    // DEBUG ENERGY COMPUTATION - DYNAMIC ANALYSIS
                    // =============================================================================
                    //
                    // Computes dynamic energy for tracked particle before rewriting files.
                    // Uses PsiA_unsorted to calculate energy for the specific debug ID.
                    // This information is used to validate energy conservation during simulation.

                    {
                        /**
                         * TRACKED PARTICLE ENERGY ANALYSIS
                         *
                         * Extract and compute dynamic energy components for a single tracked particle
                         * identified by DEBUG_PARTICLE_ID. Used to validate energy conservation
                         * and compare with theoretical predictions. The components are stored for
                         * post-simulation analysis and visualization.
                         */

                        // Retrieve the particle ID to track from the global constant
                        int debug_id = DEBUG_PARTICLE_ID;

                        // Extract physical properties from the unsorted arrays (original ordering)
                        double r_val = R_unsorted[debug_id];       // Radius
                        double v_val = Vrad_unsorted[debug_id];    // Radial velocity
                        double l_val = L_unsorted[debug_id];       // Angular momentum
                        double psiA_val = PsiA_unsorted[debug_id]; // Potential (newly computed)

                        // Calculate energy components:
                        // Total energy = potential - kinetic
                        double E_dyn = psiA_val - 0.5 * (v_val * v_val + (l_val * l_val) / (r_val * r_val));

                        // Kinetic energy = radial + rotational components
                        double K_dyn = 0.5 * (v_val * v_val + (l_val * l_val) / (r_val * r_val));

                        // Calculate simulation time corresponding to this snapshot
                        // Ensure consistent time basis with approximate energy calculation
                        double sim_time = (double)(snapshot_steps[s]) * dtwrite * dt;

                        // Store energy components in global arrays for later analysis
                        store_debug_dynE_components(
                            s,        // Snapshot index
                            E_dyn,    // Total energy
                            K_dyn,    // Kinetic energy
                            psiA_val, // Potential energy
                            sim_time, // Simulation time
                            r_val     // Radius
                        );

                        // =============================================================================
                        // END DEBUG ENERGY COMPUTATION
                        // =============================================================================
                    }
                }

                log_message("INFO", "Thread %d: Starting to write rank files for snapshot %d",
                            omp_get_thread_num(), snap);

                if (g_doDynRank)
                {
                    log_message("INFO", "Thread %d: About to write unsorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                    // Unsorted file write
                    char fname_unsorted[256];
                    char base_filename[256];
                    snprintf(base_filename, sizeof(base_filename), "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
                    get_suffixed_filename(base_filename, 1, fname_unsorted, sizeof(fname_unsorted));

                    log_message("INFO", "Thread %d: Opening %s for writing",
                                omp_get_thread_num(), fname_unsorted);

                    FILE *fun_final = fopen(fname_unsorted, "wb"); // Binary mode for fprintf_bin

                    if (!fun_final)
                    {
                        log_message("ERROR", "Thread %d: Failed to open %s for writing",
                                    omp_get_thread_num(), fname_unsorted);
                        // Skip this file but continue with other operations.
                        goto skip_unsorted_write;
                    }
                    for (int ii = 0; ii < npts; ii++)
                    {
                        fprintf_bin(fun_final, "%d %f %f %f %f %f %f\n",
                                    Rank_unsorted[ii],
                                    Mass_unsorted[ii],
                                    R_unsorted[ii],
                                    Vrad_unsorted[ii],
                                    PsiA_unsorted[ii],
                                    E_unsorted[ii],
                                    L_unsorted[ii]);
                    }
                    fclose(fun_final);
                    log_message("INFO", "Thread %d: Successfully wrote unsorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                skip_unsorted_write:

                    // Sorted file write
                    log_message("INFO", "Thread %d: About to write sorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                    char fname_sorted[256];
                    snprintf(base_filename, sizeof(base_filename), "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
                    get_suffixed_filename(base_filename, 1, fname_sorted, sizeof(fname_sorted));

                    log_message("INFO", "Thread %d: Opening %s for writing",
                                omp_get_thread_num(), fname_sorted);

                    FILE *fsort_final = fopen(fname_sorted, "wb"); // Binary mode for fprintf_bin

                    if (!fsort_final)
                    {
                        log_message("ERROR", "Thread %d: Failed to open %s for writing",
                                    omp_get_thread_num(), fname_sorted);
                        // Skip this file but continue.
                        goto skip_sorted_write;
                    }

                    for (int ii = 0; ii < npts; ii++)
                    {
                        fprintf_bin(fsort_final, "%d %f %f %f %f %f %f %f\n",
                                    Rank_sorted[ii],
                                    Mass_sorted[ii],
                                    R_sorted[ii],
                                    Vrad_sorted[ii],
                                    PsiA_sorted[ii],
                                    E_sorted[ii],
                                    L_sorted[ii],
                                    density_sorted[ii]);
                    }
                    fclose(fsort_final);
                    log_message("INFO", "Thread %d: Successfully wrote sorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                skip_sorted_write:; // Empty statement needed after label.
                } // END if (g_doDynRank).

                log_message("INFO", "Thread %d: Completed writing rank files for snapshot %d, freeing memory",
                            omp_get_thread_num(), snap);

                free(inv_rA);
                free(suffix_sum_invrA);
                free(rrA);
                free(M_rA);
                free(psiAarr);
                free(rrA_spline);
                free(psiAarr_spline);
                free(PsiA_unsorted);
                free(PsiA_sorted);
                free(E_unsorted);
                free(E_sorted);
                gsl_spline_free(splinePsiA);
                gsl_interp_accel_free(PsiAinterp);
            }
            // Free temporary arrays

            // =============================================================================
            // CLEANUP SECTION - PARTICLE DATA PROCESSING COMPLETE
            // =============================================================================
            //
            // Release all memory allocated during particle data processing phase.
            // Memory is freed in the reverse order of allocation to prevent memory leaks.
            // This includes thread-local arrays and shared memory structures.
            // Free thread-local arrays
            // Free all local arrays
            free(Rank_unsorted);
            free(Mass_unsorted);
            free(R_unsorted);
            free(Vrad_unsorted);
            free(L_unsorted);

            free(Rank_sorted);
            free(Mass_sorted);
            free(R_sorted);
            free(Vrad_sorted);
            free(L_sorted);

            free(partarr);

            // Free density_sorted array which was allocated during density calculation
            if (density_sorted != NULL) {
                free(density_sorted);
                log_message("INFO", "Thread %d: Freed density_sorted for snapshot %d",
                           omp_get_thread_num(), snap);
            }

            // Free the initial local buffers
            log_message("INFO", "Thread %d: FINAL cleanup for snapshot %d",
                        omp_get_thread_num(), snap);

            free(tmpL_partdata_snap);
            free(tmpRank_partdata_snap);
            free(tmpR_partdata_snap);
            free(tmpV_partdata_snap);

            log_message("INFO", "Thread %d COMPLETED processing of snapshot index %d (snapshot number %d)",
                        omp_get_thread_num(), s, snap);
        }

        log_message("INFO", "All snapshot processing completed. Cleaning up.");

    cleanup_partdata_outsnaps:
        // Free the memory for particle data arrays
        free(Rank_partdata_outsnaps);
        free(R_partdata_outsnaps);
        free(Vrad_partdata_outsnaps);
    }
    {
        log_message("INFO", "Final operations using file suffix: %s", g_file_suffix);

        for (int s = 0; s < noutsnaps; s++)
        {
            snapshot_steps[s] = (int)floor(
                s * (total_writes - 1) / (double)(noutsnaps - 1));
        }
    }

    // Final cleanup of trajectory buffers
    cleanup_trajectory_buffers(upper_npts_num_traj, nlowest);

    free(r_initial);
    free(rv_initial);
    free(v_initial);
    free(l_initial);
    free(r_final);
    free(v_final);


    free(mass);
    free(radius);
    if (radius_monotonic_grid_nfw != NULL) {
        free(radius_monotonic_grid_nfw);
        radius_monotonic_grid_nfw = NULL;
    }
    free(Psivalues);
    free(nPsivalues);
    free(innerintegrandvalues);
    free(Evalues);
    if (w != NULL) {
        gsl_integration_workspace_free(w);
    }
    // Free GSL RNG resources
    if (g_rng != NULL) {
        gsl_rng_free(g_rng);
        g_rng = NULL;
    }

    // End method_select == 3 block.

    for (i = 0; i < 7; i++)  // 7 rows for extended particle state
    {
        free(particles[i]);
    }
    free(particles);

    // Write total energy diagnostics to file if data was generated
    if (g_doAllParticleData && g_time_snapshots != NULL) {
        int num_energy_snaps = nwrite_total / stepBetweenSnaps;
        if (num_energy_snaps > noutsnaps) num_energy_snaps = noutsnaps;
        finalize_energy_diagnostics(num_energy_snaps);
    }

    if (g_doDebug)
    {
        finalize_debug_energy_output(); // Ensures all data is collected first
    }

    // Free splines and accelerators if allocated
    if (splinemass) gsl_spline_free(splinemass);
    if (splinePsi) gsl_spline_free(splinePsi);
    if (splinerofPsi) gsl_spline_free(splinerofPsi);
    if (enclosedmass) gsl_interp_accel_free(enclosedmass);
    if (Psiinterp) gsl_interp_accel_free(Psiinterp);
    if (rofPsiinterp) gsl_interp_accel_free(rofPsiinterp);
    if (g_main_fofEinterp) gsl_interp_free(g_main_fofEinterp);
    if (g_main_fofEacc) gsl_interp_accel_free(g_main_fofEacc);

    // Free total energy diagnostic arrays
    free(g_time_snapshots);
    free(g_total_KE);
    free(g_total_PE);
    free(g_total_E);

    cleanup_all_particle_data();

#ifdef _OPENMP
    // Clean up FFTW threads if initialized
    fftw_cleanup_threads();

    if (g_enable_sidm_scattering) {
        printf("Total SIDM scattering events during simulation: %lld\n", g_total_sidm_scatters);
        log_message("INFO", "Total SIDM scattering events: %lld", g_total_sidm_scatters);
    }
#endif

    // Free per-thread GSL RNG resources
    if (g_rng_per_thread != NULL) {
        for (int i_rng = 0; i_rng < g_max_omp_threads_for_rng; ++i_rng) {
            if (g_rng_per_thread[i_rng] != NULL) {
                gsl_rng_free(g_rng_per_thread[i_rng]);
            }
        }
        free(g_rng_per_thread);
        g_rng_per_thread = NULL;
        log_message("INFO", "Freed per-thread GSL RNGs.");
    }

    // Cleanup for the conditionally declared persistent sort buffer.
    if (g_sort_columns_buffer != NULL) {
        for (int i = 0; i < g_sort_columns_buffer_npts; i++) {
            if (g_sort_columns_buffer[i]) free(g_sort_columns_buffer[i]);
        }
        free(g_sort_columns_buffer);
        g_sort_columns_buffer = NULL; // Mark as freed.
        g_sort_columns_buffer_npts = 0; // Reset size.
    }

    // Free particle scatter state array
    free(g_particle_scatter_state);
    free(g_current_timestep_scatter_counts);  // Free scatter count array

    // Report sorting benchmark statistics if benchmarking mode was used
    if (strcmp(g_defaultSortAlg, "benchmark_mode") == 0 && g_benchmark_count > 0) {
        printf("\n");
        printf("================================================================================\n");
        printf("ADAPTIVE SORTING SUMMARY (Option 6)\n");
        printf("================================================================================\n");
        printf("Total sort operations:       %d\n", g_sort_call_count);
        printf("Benchmark comparisons made:  %d (every 1000 sorts)\n", g_benchmark_count);
        printf("Final algorithm in use:      %s\n", g_current_best_sort);
        printf("\n");
        printf("Algorithm Win Distribution:\n");
        printf("  Insertion wins:   %d (%.1f%%)\n",
               g_insertion_wins,
               100.0 * g_insertion_wins / g_benchmark_count);
        printf("  Quadsort wins:    %d (%.1f%%)\n",
               g_quadsort_wins,
               100.0 * g_quadsort_wins / g_benchmark_count);
        printf("  Radix wins:       %d (%.1f%%)\n",
               g_radix_wins,
               100.0 * g_radix_wins / g_benchmark_count);
        printf("\n");
        printf("Average times per benchmark:\n");
        printf("  Insertion:        %.3f ms\n", g_total_insertion_time / g_benchmark_count);
        printf("  Quadsort:         %.3f ms\n", g_total_quadsort_time / g_benchmark_count);
        printf("  Radix:            %.3f ms\n", g_total_radix_time / g_benchmark_count);

        // Find fastest overall
        double min_avg = g_total_insertion_time / g_benchmark_count;
        const char *fastest = "Insertion";
        if (g_total_quadsort_time / g_benchmark_count < min_avg) {
            min_avg = g_total_quadsort_time / g_benchmark_count;
            fastest = "Quadsort";
        }
        if (g_total_radix_time / g_benchmark_count < min_avg) {
            min_avg = g_total_radix_time / g_benchmark_count;
            fastest = "Radix";
        }
        printf("\n");
        printf("Fastest algorithm overall:   %s (%.3f ms average)\n", fastest, min_avg);
        printf("================================================================================\n");
    }

    return 0;
} // End main function.

/**
 * @brief GSL integrand \f$r^2 \rho_{shape}(r)\f$ for Cored Plummer-like profile mass calculation.
 * @details Computes the term \f$r^2 \rho_{shape}(r)\f$ for the Cored Plummer-like density profile,
 *          where \f$\rho_{shape}(r) = (1 + (r/r_c)^2)^{-3}\f$. This integrand is used in
 *          GSL numerical integration routines (e.g., `gsl_integration_qag`) to calculate
 *          the normalization factor or the enclosed mass \f$M(<r) = 4\pi \int_0^r r'^2 \rho_{physical}(r') dr'\f$.
 *          It directly uses the `RC` macro for the scale radius.
 *
 * @param r      [in] Radial coordinate \f$r\f$ (kpc).
 * @param params [in] Void pointer to parameters (unused).
 * @return double The value of the mass integrand \f$r^2 \rho_{shape}(r)\f$ at radius `r`.
 */
double massintegrand(double r, void *params)
{
    (void)params; // Unused parameter
    double startingprofile = 1.0 / cube((1.0 + sqr(r / g_cored_profile_rc)));
    return r * r * startingprofile;
}

/**
 * @brief Calculates \f$d\rho/dr\f$ for the Cored Plummer-like density profile.
 * @details The density profile is \f$\rho(r) \propto (1 + (r/r_c)^2)^{-3}\f$.
 *          This function computes its analytical derivative with respect to \f$r\f$.
 *          It directly uses the `RC` macro for the scale radius.
 *
 * @param r [in] Radial coordinate (kpc) at which to evaluate the derivative.
 * @return double The value of \f$d\rho/dr\f$ at radius `r`.
 */
double drhodr(double r)
{
    return -6.0 * r / (g_cored_profile_rc * g_cored_profile_rc) / pow(1.0 + sqr(r / g_cored_profile_rc), 4.0);
}

/**
 * @brief GSL integrand \f$r^2 \rho(r)\f$ for NFW-like profile mass calculation.
 * @details Computes \f$r^2 \rho(r)\f$ where \f$\rho(r)\f$ is an NFW-like density profile
 *          with an inner softening term and an outer power-law cutoff.
 *          The density \f$\rho(r) = p[2] \times [ (r_s + \epsilon)(1+r_s)^2 (1 + (r_s/C)^N) ]^{-1}\f$,
 *          where \f$r_s = r/p[0]\f$, \f$\epsilon=0.01\f$, \f$C=p[3]\f$, \f$N=10.0\f$.
 *          This integrand is used in GSL routines to calculate the normalization factor
 *          or enclosed mass for the NFW-like profile.
 *
 * @param r      [in] Radial coordinate \f$r\f$ (kpc).
 * @param params [in] Void pointer to a `double` array `p` of size 4:
 *                    - `p[0]` (rc_param): Scale radius (kpc).
 *                    - `p[1]` (halo_mass): Target total halo mass (\f$M_{\odot}\f$) - used to derive nt_nfw_scaler.
 *                    - `p[2]` (nt_nfw_scaler): Density normalization constant \f$nt_{NFW}\f$.
 *                    - `p[3]` (falloff_C_param): Falloff transition factor \f$C\f$.
 * @return double The value of the mass integrand \f$r^2 \rho(r)\f$. Returns 0.0 if `rc_param` (p[0]) is non-positive.
 */
double massintegrand_profile_nfwcutoff(double r, void *params) {
    double *p = (double *)params;
    double rc_param = p[0];                 // Scale radius RC from parameters
    // p[1] is current_profile_halo_mass, not used directly in density formula
    double nt_nfw_scaler = p[2];            // Density scaling factor nt_nfw

    // Parameters for the NFW-like profile shape
    const double epsilon_softening = 0.01;  // Softening parameter for r/rc term
    double C_cutoff_factor = p[3];         // Falloff factor from params
    if (C_cutoff_factor <= 0) C_cutoff_factor = 19.0; // Safety default if param is bad
    const double N_cutoff_power = 10.0;     // Power for the cutoff term

    if (rc_param <= 0) { // Avoid division by zero if rc is invalid
        return 0.0;
    }
    
    double rs = r / rc_param; // r normalized by scale radius

    // Calculate the structural part of the density profile (unscaled by nt_nfw)
    double term_softening = rs + epsilon_softening;
    if (term_softening <= 1e-9) term_softening = 1e-9; // Avoid division by zero from softening

    double term_nfw_slope = (1.0 + rs) * (1.0 + rs); // (1 + r/rc)^2

    double cutoff_rs = rs / C_cutoff_factor;
    double term_cutoff = 1.0 + pow(cutoff_rs, N_cutoff_power);

    double density_shape;
    if (term_softening < 1e-9 || term_nfw_slope < 1e-9 || term_cutoff < 1e-9) { // Denominator terms too small
         density_shape = 0.0; // Or handle as very large if r is very small
         if (r < 1e-6 && term_softening < 1e-3) { // special handling for very small r to match NFW cusp
             density_shape = 1.0 / (term_softening * term_nfw_slope * term_cutoff); // Let it be large
         }
    } else {
         density_shape = 1.0 / (term_softening * term_nfw_slope * term_cutoff);
    }

    // Apply the overall density scaling factor
    double physical_density = nt_nfw_scaler * density_shape;
    
    return r * r * physical_density;
}

/**
 * @brief Calculates \f$d\rho/dr\f$ for the NFW-like profile with a power-law cutoff.
 * @details The NFW-like density profile used is:
 *          \f$\rho(r) = \text{nt_nfw_scaler} \times [ (r_s + \epsilon)(1+r_s)^2 (1 + (r_s/C)^N) ]^{-1}\f$
 *          where \f$r_s = r / \text{rc_param}\f$, \f$\epsilon\f$ is a softening parameter (0.01),
 *          \f$C\f$ is the `falloff_C_param`, and \f$N\f$ is a power-law index (10.0).
 *          This function computes the analytical derivative of this \f$\rho(r)\f$ with respect to \f$r\f$.
 *
 * @param r               [in] Radial coordinate (kpc) at which to evaluate the derivative.
 * @param rc_param        [in] Scale radius \f$r_c\f$ of the NFW-like profile (kpc).
 * @param nt_nfw_scaler   [in] Density normalization constant (nt_nfw) for the profile.
 * @param falloff_C_param [in] Falloff transition factor \f$C\f$ for the power-law cutoff.
 * @return double The value of \f$d\rho/dr\f$ at radius `r`. Returns 0.0 if `rc_param` is non-positive.
 */
double drhodr_profile_nfwcutoff(double r, double rc_param, double nt_nfw_scaler, double falloff_C_param) { // Added falloff_C_param

    // Parameters for the NFW-like profile shape
    const double epsilon_softening = 0.01;
    double C_cutoff_factor = falloff_C_param;
    if (C_cutoff_factor <= 0) C_cutoff_factor = 19.0; // Safety default
    const double N_cutoff_power = 10.0;

    if (rc_param <= 0) return 0.0;
    
    double rs = r / rc_param;

    //rho_shape(rs) = 1.0 / ( (rs+eps) * (1+rs)^2 * (1+(rs/C)^N) )
    double term_s = rs + epsilon_softening;
    if (term_s <= 1e-9) term_s = 1e-9; // Avoid division by zero for denominator
    double term_n = (1.0 + rs) * (1.0 + rs);
    double term_c_base = rs / C_cutoff_factor;
    double term_c_pow_N = pow(term_c_base, N_cutoff_power);
    double term_c = 1.0 + term_c_pow_N;

    double density_shape_val;
    if (term_s < 1e-9 || term_n < 1e-9 || term_c < 1e-9) {
        density_shape_val = 0.0; // Or very large if r is tiny
        if (r < 1e-6 && term_s < 1e-3) {
            density_shape_val = 1.0 / (term_s * term_n * term_c);
        }
    } else {
        density_shape_val = 1.0 / (term_s * term_n * term_c);
    }

    // Derivative of each term in the denominator w.r.t rs (d/d(rs)):
    // d/drs (rs+eps) = 1
    // d/drs (1+rs)^2 = 2*(1+rs)
    // d/drs (1+(rs/C)^N) = N * (rs/C)^(N-1) * (1/C)
    
    double d_log_term_s_d_rs = 1.0 / term_s;
    double d_log_term_n_d_rs = 2.0 / (1.0 + rs);
    double d_log_term_c_d_rs = (N_cutoff_power / C_cutoff_factor) * pow(term_c_base, N_cutoff_power - 1.0) / term_c;
    if (!isfinite(term_c_base) || (term_c_base < 1e-9 && N_cutoff_power -1 < 0)) { // Avoid pow(small_negative_base)
         d_log_term_c_d_rs = 0; // If rs/C is zero and N-1 is negative
    }

    // d(rho_shape)/dr = d(rho_shape)/d(rs) * d(rs)/dr = d(rho_shape)/d(rs) * (1/rc_param)
    // d(log(rho_shape))/d(rs) = - (d_log_term_s_d_rs + d_log_term_n_d_rs + d_log_term_c_d_rs)
    // d(rho_shape)/d(rs) = rho_shape * d(log(rho_shape))/d(rs)
    
    double d_rho_shape_d_rs = -density_shape_val * (d_log_term_s_d_rs + d_log_term_n_d_rs + d_log_term_c_d_rs);
    double drho_dr = nt_nfw_scaler * d_rho_shape_d_rs / rc_param;

    
    return drho_dr;
}

/**
 * @brief GSL integrand \f$r^2 \rho(r)\f$ for Hernquist profile mass calculation.
 * @details Computes \f$r^2 \rho(r)\f$ where \f$\rho(r) = M_{\text{total}} a / [2\pi r (r+a)^3]\f$
 *          is the Hernquist density profile.
 *
 * @param r      [in] Radial coordinate (kpc).
 * @param params [in] Void pointer to a double array: [a_scale, M_total, normalization_factor].
 * @return double The value of \f$r^2 \rho(r)\f$ at radius `r`.
 */
double massintegrand_hernquist(double r, void *params) {
    double *p = (double *)params;
    double a_scale = p[0];           // Scale radius a
    double M_norm = p[2];            // Normalization factor (will be M_total/(4π) after normalization)
    
    if (r <= 0 || a_scale <= 0) return 0.0;
    
    // Hernquist density shape: ρ(r) ∝ 1/(r(r+a)³)
    double r_plus_a = r + a_scale;
    double density_shape = 1.0 / (r * r_plus_a * r_plus_a * r_plus_a);
    
    // Return r²ρ(r) for mass integration
    return M_norm * r * r * density_shape;
}

/**
 * @brief Calculates \f$d\rho/dr\f$ for the Hernquist profile.
 * @details Computes the radial derivative of the Hernquist density profile
 *          \f$\rho(r) = M_{\text{total}} a / [2\pi r (r+a)^3]\f$.
 *          The derivative is: \f$d\rho/dr = -(M a / 2\pi) (4r + a) / [r^2(r+a)^4]\f$
 *
 * @param r       [in] Radial coordinate (kpc).
 * @param a_scale [in] Scale radius \f$a\f$ of the Hernquist profile (kpc).
 * @param M_total [in] Total mass normalization (not used in shape calculation).
 * @return double The value of \f$d\rho/dr\f$ at radius `r`.
 */
double drhodr_hernquist(double r, double a_scale, double M_total) {
    (void)M_total; // Unused parameter, kept for consistency with other drhodr functions
    if (r <= 0 || a_scale <= 0) return 0.0;
    
    double r_plus_a = r + a_scale;
    // d/dr[1/(r(r+a)³)] = -(4r+a)/(r²(r+a)⁴)
    double numerator = -(4.0 * r + a_scale);
    double denominator = r * r * r_plus_a * r_plus_a * r_plus_a * r_plus_a;
    
    return numerator / denominator;
}

// =============================================================================
// OSIPKOV-MERRITT ANISOTROPY MODEL FUNCTIONS
// =============================================================================
/**
 * @brief Osipkov-Merritt augmented density functions for anisotropic models.
 * @details The OM model uses radially-varying anisotropy \f$\beta(r) = r^2/(r^2 + r_a^2)\f$
 *          implemented via augmented density \f$\rho_Q(r) = \rho(r)(1 + r^2/r_a^2)\f$.
 *          These functions compute derivatives of the augmented density for
 *          use in Eddington-like inversion to obtain the distribution function \f$f(Q)\f$.
 */

/**
 * @brief Calculates \f$d\rho_Q/dr\f$ for the Osipkov-Merritt Cored Plummer-like profile.
 * @details Computes the radial derivative of the OM augmented density \f$\rho_Q(r) = \rho(r)(1 + r^2/r_a^2)\f$.
 *          Uses the product rule: \f$d\rho_Q/dr = (d\rho/dr)(1+r^2/r_a^2) + \rho(r)(2r/r_a^2)\f$.
 *          This is used in the Eddington-like inversion to find \f$f(Q)\f$.
 *
 * @param r [in] Radial coordinate (kpc).
 * @return double The value of \f$d\rho_Q/dr\f$ at radius `r`.
 */
double density_derivative_om_cored(double r)
{
    if (g_om_anisotropy_radius <= 0) return drhodr(r); // Isotropic case when anisotropy radius approaches infinity

    // First term: rho(r) * d/dr(1 + r^2/r_a^2)
    // Density shape function without mass normalization factor.
    // Mass normalization cancels in the ratio d(rho)/d(psi).
    double rho_shape = 1.0 / cube(1.0 + sqr(r / g_cored_profile_rc));
    double term1 = rho_shape * (2.0 * r / (g_om_anisotropy_radius * g_om_anisotropy_radius));

    // Second term: (d(rho)/dr) * (1 + r^2/r_a^2)
    double drhodr_val = drhodr(r);
    double term2 = drhodr_val * (1.0 + r * r / (g_om_anisotropy_radius * g_om_anisotropy_radius));
    
    return term1 + term2;
}

/**
 * @brief Calculates \f$d\rho_Q/dr\f$ for the Osipkov-Merritt Hernquist profile.
 * @details Computes the radial derivative of the OM augmented density
 *          \f$\rho_Q(r) = \rho(r)(1 + r^2/r_a^2)\f$ for the Hernquist profile.
 *          Uses the product rule: \f$d\rho_Q/dr = (d\rho/dr)(1+r^2/r_a^2) + \rho(r)(2r/r_a^2)\f$.
 *
 * @param r       [in] Radial coordinate (kpc).
 * @param a_scale [in] Scale radius \f$a\f$ of the Hernquist profile (kpc).
 * @param M_total [in] Total mass normalization (not used in shape calculation).
 * @return double The value of \f$d\rho_Q/dr\f$ at radius `r`.
 */
double drhodr_om_hernquist(double r, double a_scale, double M_total) {
    if (g_om_anisotropy_radius <= 0) return drhodr_hernquist(r, a_scale, M_total);
    
    // First term: rho(r) * d/dr(1 + r^2/r_a^2)
    // Hernquist density shape (unnormalized): 1/(r(r+a)³)
    double r_plus_a = r + a_scale;
    double rho_shape = 1.0 / (r * r_plus_a * r_plus_a * r_plus_a);
    double term1 = rho_shape * (2.0 * r / (g_om_anisotropy_radius * g_om_anisotropy_radius));
    
    // Second term: (d(rho)/dr) * (1 + r^2/r_a^2)
    double drhodr_val = drhodr_hernquist(r, a_scale, M_total);
    double term2 = drhodr_val * (1.0 + r * r / (g_om_anisotropy_radius * g_om_anisotropy_radius));
    
    return term1 + term2;
}

/**
 * @brief Calculates \f$d\rho_Q/dr\f$ for the Osipkov-Merritt NFW-like profile.
 * @details Computes the radial derivative of the OM augmented density \f$\rho_Q(r) = \rho(r)(1 + r^2/r_a^2)\f$.
 *          Uses the product rule: \f$d\rho_Q/dr = (d\rho/dr)(1+r^2/r_a^2) + \rho(r)(2r/r_a^2)\f$.
 *          This is used in the Eddington-like inversion to find \f$f(Q)\f$.
 *
 * @param r [in] Radial coordinate (kpc).
 * @param rc_param [in] Scale radius \f$r_c\f$ of the NFW-like profile (kpc).
 * @param nt_nfw_scaler [in] Density normalization constant (nt_nfw) for the profile.
 * @param falloff_C_param [in] Falloff transition factor C for the power-law cutoff.
 * @return double The value of \f$d\rho_Q/dr\f$ at radius `r`.
 */
double density_derivative_om_nfw(double r, double rc_param, double nt_nfw_scaler, double falloff_C_param)
{
    if (g_om_anisotropy_radius <= 0) {
        return drhodr_profile_nfwcutoff(r, rc_param, nt_nfw_scaler, falloff_C_param);
    }

    // --- Calculate rho(r) for the NFW profile ---
    const double epsilon_softening = 0.01;
    double C_cutoff_factor = falloff_C_param;
    if (C_cutoff_factor <= 0) C_cutoff_factor = 19.0;
    const double N_cutoff_power = 10.0;
    if (rc_param <= 0) return 0.0;
    double rs = r / rc_param;
    double term_softening = rs + epsilon_softening;
    if (term_softening <= 1e-9) term_softening = 1e-9;
    double term_nfw_slope = (1.0 + rs) * (1.0 + rs);
    double cutoff_rs = rs / C_cutoff_factor;
    double term_cutoff = 1.0 + pow(cutoff_rs, N_cutoff_power);
    double density_shape;
    if (term_softening < 1e-9 || term_nfw_slope < 1e-9 || term_cutoff < 1e-9) {
         density_shape = (r < 1e-6 && term_softening < 1e-3) ? (1.0 / (term_softening * term_nfw_slope * term_cutoff)) : 0.0;
    } else {
         density_shape = 1.0 / (term_softening * term_nfw_slope * term_cutoff);
    }
    double rho_val = nt_nfw_scaler * density_shape;

    // --- Product Rule Term 1: rho(r) * d/dr(1 + r^2/r_a^2) ---
    double term1 = rho_val * (2.0 * r / (g_om_anisotropy_radius * g_om_anisotropy_radius));

    // --- Product Rule Term 2: (d(rho)/dr) * (1 + r^2/r_a^2) ---
    double drhodr_val = drhodr_profile_nfwcutoff(r, rc_param, nt_nfw_scaler, falloff_C_param);
    double term2 = drhodr_val * (1.0 + r * r / (g_om_anisotropy_radius * g_om_anisotropy_radius));

    return term1 + term2;
}

/**
 * @brief Integrand for calculating gravitational potential.
 * @details Computes the integrand required for the integral term in the
 *          potential calculation, \f$\Psi(r)\f$. This function acts as a dispatcher,
 *          calling the appropriate profile-specific mass integrand via a function
 *          pointer provided in the `params` struct.
 *
 * @param rp The radial integration variable \f$r'\f$ (kpc).
 * @param params Void pointer to a `Psiintegrand_params` struct, which contains
 *               the function pointer to the profile's mass integrand.
 * @return The value of the potential integrand at `rp`.
 *
 * @see Psiintegrand_params
 * @see massintegrand
 * @see massintegrand_profile_nfwcutoff
 */
double Psiintegrand(double rp, void *params)
{
    Psiintegrand_params *p_psi = (Psiintegrand_params *)params;
    if (rp <= 0.0)
    {
        printf("rp out of range\n");
        CLEAN_EXIT(1);
    }
    // Call the profile-specific mass integrand function
    return p_psi->massintegrand_func(rp, p_psi->params_for_massintegrand) / rp;
}


/**
 * @brief Safely evaluates a GSL spline with robust bounds checking.
 * @details Evaluates the spline at the given value. If the value is outside the
 *          spline's defined domain, this function clamps it to the nearest valid
 *          boundary to prevent a GSL domain error. Handles NULL pointers for the
 *          spline or accelerator by logging an error and returning 0.0.
 *
 * @param spline Pointer to the initialized GSL spline object.
 * @param acc Pointer to the GSL interpolation accelerator.
 * @param value The value at which to evaluate the spline.
 *
 * @return The interpolated value from the spline. If `value` is out of bounds,
 *         returns the value at the nearest boundary. If `spline` or `acc`
 *         is NULL, prints an error to stderr and returns 0.0.
 */
double evaluatespline(gsl_spline *spline, gsl_interp_accel *acc, double value)
{
    // NULL pointer safety check.
    if (spline == NULL || acc == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to evaluatespline (spline=%p, acc=%p)\n",
                (void *)spline, (void *)acc);
        return 0.0; // Return a default value instead of crashing.
    }

    // Get the actual min and max ranges of the spline from its data directly.
    double x_min = spline->x[0];
    double x_max = spline->x[spline->size - 1];

    // Ensure the value is within the valid interpolation range with a small safety margin.
    const double MARGIN = 1e-10; // Small safety margin.

    if (value < x_min)
    {
        if (getenv("DEBUG_SPLINE") != NULL) {
            fprintf(stderr, "Warning: Spline interpolation value %g below minimum %g, clamping\n",
                    value, x_min);
        }
        // Clamp to minimum with a tiny margin to stay inside the valid range.
        return gsl_spline_eval(spline, x_min + MARGIN, acc);
    }
    else if (value > x_max)
    {
        if (getenv("DEBUG_SPLINE") != NULL) {
            fprintf(stderr, "Warning: Spline interpolation value %g above maximum %g, clamping\n",
                    value, x_max);
        }
        // Clamp to maximum with a tiny margin to stay inside the valid range.
        return gsl_spline_eval(spline, x_max - MARGIN, acc);
    }

    // Normal case - value is within range.
    return gsl_spline_eval(spline, value, acc);
}

/**
 * Calculates the integrand for the distribution function calculation.
 *
 * Uses splines to evaluate the potential and mass at a given energy and radius,
 * and computes the distribution function integrand according to Eddington's formula.
 *
 * @param t Integration variable
 * @param params Structure containing necessary splines and energy value
 * @return Value of the distribution function integrand at the specified point
 */
double fEintegrand(double t, void *params)
{
    fEintegrand_params *p = (fEintegrand_params *)params;
    gsl_spline *splinePsi = p->splinePsi;
    gsl_spline *splinemass = p->splinemass;
    gsl_interp_accel *rofPsiarray = p->rofPsiarray;
    gsl_interp_accel *massarray = p->massarray;
    double E_or_Q = p->E;  // This is E for isotropic, Q for OM
    
    // For OM: Q = Psi - t^2 where t = sqrt(Q - Psi)
    // For isotropic: E = Psi - t^2 where t = sqrt(E - Psi)
    // In both cases: Psi = E_or_Q - t^2
    double Psi = E_or_Q - t * t;
    double r = evaluatespline(splinePsi, rofPsiarray, -Psi);
    
    double drhodr_val = g_density_derivative_func(r);
    double Menc = evaluatespline(splinemass, massarray, r);
    double dPsidr = -(G_CONST * Menc) / (r * r);
    double drhodpsi = drhodr_val / dPsidr;
    return 2.0 * drhodpsi;
}

/**
 * @brief Integrand for the NFW distribution function \f$I(E)\f$ calculation.
 * @details The function computes the integrand \f$2 \cdot (d\rho/d\Psi)\f$ for Eddington's formula,
 *          using the transformed integration variable \f$t = \sqrt{E_{shell} - \Psi(r)}\f$.
 *          It determines the radius `r` corresponding to a given `t` by first finding
 *          \f$\Psi(r) = E_{shell} - t^2\f$ and then using a spline for \f$r(\Psi)\f$.
 *          The derivatives \f$d\rho/dr\f$ and \f$d\Psi/dr\f$ are then calculated at that radius.
 *
 * @param t_integration_var The integration variable, \f$t = \sqrt{E_{shell} - \Psi(r)}\f$.
 * @param params Pointer to a `fE_integrand_params_NFW_t` struct containing the
 *               energy shell, splines, and profile parameters.
 * @return The value of the integrand \f$2 \cdot (d\rho/d\Psi)\f$. Returns 0.0 if the
 *         potential is outside the physical range, the radius is non-physical,
 *         \f$d\Psi/dr\f$ is near zero, or the result is non-finite.
 */
/**
 * @brief Integrand of Hernquist-specific Eddington inversion for computing \f$I(E)\f$ or \f$I(Q)\f$.
 * @details The integrand is \f$2(d\rho/d\Psi)\f$ which is positive due to sign convention \f$d\Psi/dr = -GM/r^2\f$.
 *          Uses transformed variable t_integration_var = sqrt(E_shell - Psi_true) to avoid
 *          singularities. For OM model, E_shell represents \f$Q = E - L^2/(2r_a^2)\f$.
 *
 * @param t_integration_var [in] The t variable where t = sqrt(E_shell - Psi_true).
 *                               Integration bounds are [0, sqrt(E_shell - Psimin_global)].
 * @param params [in] Pointer to fE_integrand_params_hernquist_t containing E_shell, splines,
 *                    profile parameters, and physical range limits.
 * @return double The value of the integrand \f$2(d\rho/d\Psi)\f$. Returns 0.0 if invalid.
 */
double fEintegrand_hernquist(double t_integration_var, void *params) {
    fE_integrand_params_hernquist_t *p_hern = (fE_integrand_params_hernquist_t *)params;
    double E_shell = p_hern->E_current_shell;
    
    // Calculate Psi_true from t: Psi_true = E_shell - t²
    double Psi_true_at_r = E_shell - t_integration_var * t_integration_var;
    
    // Ensure Psi_true is within valid physical range
    if (Psi_true_at_r < p_hern->Psimin_global - 1e-7*fabs(p_hern->Psimin_global) || 
        Psi_true_at_r > p_hern->Psimax_global + 1e-7*fabs(p_hern->Psimax_global)) {
        return 0.0;
    }
    
    // Get radius r from Psi_true. Spline expects -Psi_true as input.
    double spline_x_input = -Psi_true_at_r;
    double spline_x_min = p_hern->spline_r_of_Psi->x[0];
    double spline_x_max = p_hern->spline_r_of_Psi->x[p_hern->spline_r_of_Psi->size - 1];
    
    if (spline_x_input < spline_x_min) spline_x_input = spline_x_min;
    if (spline_x_input > spline_x_max) spline_x_input = spline_x_max;
    
    double r_val = gsl_spline_eval(p_hern->spline_r_of_Psi, spline_x_input, p_hern->accel_r_of_Psi);
    
    // Check for non-physical radius
    if (r_val <= 1e-10) {
        return 0.0;
    }
    
    // Calculate drho/dr at r_val (use OM or isotropic based on use_om flag)
    double drho_dr_val;
    if (p_hern->use_om) {
        drho_dr_val = drhodr_om_hernquist(r_val, p_hern->hernquist_a_scale, 
                                          p_hern->hernquist_normalization * 4.0 * M_PI);
    } else {
        drho_dr_val = drhodr_hernquist(r_val, p_hern->hernquist_a_scale, 
                                       p_hern->hernquist_normalization * 4.0 * M_PI);
    }
    
    double M_at_r = gsl_spline_eval(p_hern->spline_M_of_r, r_val, p_hern->accel_M_of_r);
    if (M_at_r < 0) M_at_r = 0;

    double dPsi_dr = -(p_hern->const_G_universal * M_at_r) / (r_val * r_val);

    if (fabs(dPsi_dr) < 1e-30) {
        return 0.0;
    }

    double drho_dPsi_val = drho_dr_val / dPsi_dr;
    double integrand_value = 2.0 * drho_dPsi_val;
    
    if (!isfinite(integrand_value)) {
        return 0.0;
    }
    
    return integrand_value;
}

/**
 * @brief Integrand for NFW distribution function \f$I(E)\f$ calculation.
 * @param t_integration_var Integration variable \f$t = \sqrt{E_{shell} - \Psi(r)}\f$.
 * @param params Pointer to fE_integrand_params_NFW_t struct.
 * @return Integrand value \f$2(d\rho/d\Psi)\f$.
 */
double fEintegrand_nfw(double t_integration_var, void *params) {
    fE_integrand_params_NFW_t *p_nfw = (fE_integrand_params_NFW_t *)params;

    double E_shell = p_nfw->E_current_shell; // Energy of the current shell for I(E)

    // Calculate Psi_true from t_integration_var: Psi_true = E_shell - t_integration_var^2
    // This means as t_integration_var goes from 0 to sqrt(E_shell - Psimin_global),
    // Psi_true goes from E_shell down to Psimin_global.
    double Psi_true_at_r = E_shell - t_integration_var * t_integration_var;


    // Ensure Psi_true_at_r is within the valid physical range [Psimin_global, Psimax_global]
    if (Psi_true_at_r < p_nfw->Psimin_global - 1e-7*fabs(p_nfw->Psimin_global) || Psi_true_at_r > p_nfw->Psimax_global + 1e-7*fabs(p_nfw->Psimax_global)) {
         return 0.0;
    }
    
    // Get radius r from Psi_true_at_r. Spline p->spline_r_of_Psi expects -Psi_true as input.
    double r_val;
    double spline_x_input_rPsi = -Psi_true_at_r; // Input for r_of_Psi spline
    double spline_rPsi_x_min = p_nfw->spline_r_of_Psi->x[0];
    double spline_rPsi_x_max = p_nfw->spline_r_of_Psi->x[p_nfw->spline_r_of_Psi->size - 1];

    if (spline_x_input_rPsi < spline_rPsi_x_min) spline_x_input_rPsi = spline_rPsi_x_min;
    if (spline_x_input_rPsi > spline_rPsi_x_max) spline_x_input_rPsi = spline_rPsi_x_max;
        
    r_val = gsl_spline_eval(p_nfw->spline_r_of_Psi, spline_x_input_rPsi, p_nfw->accel_r_of_Psi); 


    // If radius is non-physical (negative or zero), the integrand is ill-defined or zero.
    if (r_val <= 1e-10) { // Using a slightly larger epsilon than machine precision for safety
        return 0.0; 
    }
    // Use r_val as computed; return 0 if non-physical (negative or zero)

    // Calculate drho/dr at r_val
    double drho_dr_val = g_density_derivative_nfw_func(r_val, p_nfw->profile_rc_const, p_nfw->profile_nt_norm_const, p_nfw->profile_falloff_C_const);

    // Calculate dPsi/dr with proper sign convention
    double M_at_r_val = gsl_spline_eval(p_nfw->spline_M_of_r, r_val, p_nfw->accel_M_of_r);
    if (M_at_r_val < 0) M_at_r_val = 0; // Mass must be non-negative

    // dPsi/dr = -GM(r)/r² (negative because Psi becomes less negative as r increases)
    double dPsi_dr = -(p_nfw->const_G_universal * M_at_r_val) / (r_val * r_val);

    if (fabs(dPsi_dr) < 1e-30) { // If dPsi/dr is effectively zero (e.g. M(r)=0 at r=0)
        return 0.0; // drho/dPsi would be undefined or infinite
    }

    // drho/dPsi = (drho/dr) / (dPsi/dr) = (negative) / (negative) = positive
    double drho_dPsi_val = drho_dr_val / dPsi_dr;

    // Integrand is 2 * drho/dPsi (positive)
    double integrand_value = 2.0 * drho_dPsi_val;


    if (!isfinite(integrand_value)) {
        if (g_doDebug) fprintf(stderr, "Warning: NFW fEintegrand (refactored) returning non-finite value for t_in=%.3e, E_shell=%.3e\n", t_integration_var, E_shell);
        return 0.0; // Return 0 for non-finite cases
    }

    return integrand_value;
}

/**
 * @brief Integrand for OM \f$\mu\f$-integral in df_fixed_radius.
 */
double om_mu_integrand(double mu, void *params) {
    om_mu_integral_params *p = (om_mu_integral_params *)params;
    double one_minus_mu_sq = 1.0 - mu * mu;
    double bracket = 1.0 + p->r_over_ra_sq * one_minus_mu_sq;
    double Q_val = p->Psi_rf_codes - 0.5 * p->v_sq * bracket;

    if (Q_val < p->Psimin || Q_val > p->Psimax) return 0.0;

    double derivative;
    int status = gsl_interp_eval_deriv_e(p->fofQ_interp, p->Q_array, p->I_array,
                                         Q_val, p->fofQ_accel, &derivative);
    if (status != GSL_SUCCESS || !isfinite(derivative)) return 0.0;

    return derivative / (sqrt(8.0) * PI * PI);
}

// =============================================================================
// PARALLEL SORTING ALGORITHM FUNCTIONS
// =============================================================================
//
// Functions implementing parallel sorting algorithms.
// The sorting implementation uses a parallel chunk-based approach with overlapping
// regions between chunks to ensure correct ordering at chunk boundaries.
// Constants controlling behavior are defined at the top of this file.

/**
 * @brief Compares two particle entries for sorting based on the first column value.
 *
 * @details Used as a comparison function for qsort, quadsort, and other sorting algorithms.
 *          Particles with smaller values in column 0 will be sorted before those with larger values.
 *
 * @param a Pointer to the first particle entry (as void pointer, expected double**).
 * @param b Pointer to the second particle entry (as void pointer, expected double**).
 * @return -1 if a<b, 1 if a>b, 0 if equal based on the first column value (radius).
 *
 * @note Particle data structure: `columns[particle_index][component_index]`.
 *       Comparison based on `columns[i][0]` (radial position).
 */
int compare_particles(const void *a, const void *b)
{
    double *col_a = *(double **)a;
    double *col_b = *(double **)b;
    if (col_a[0] < col_b[0])
        return -1;
    if (col_a[0] > col_b[0])
        return 1;
    return 0;
}

/**
 * @brief Implementation of classic insertion sort algorithm for particle data.
 *
 * @details Sorts an array of particle data using the insertion sort algorithm.
 *          While not the fastest algorithm for large datasets, it is stable and works well
 *          for small arrays or nearly sorted data.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
static void insertion_sort(double **columns, int n)
{
    for (int i = 1; i < n; i++)
    {
        double *temp = columns[i];
        int j = i - 1;
        while (j >= 0 && compare_particles(&columns[j], &temp) > 0)
        {
            columns[j + 1] = columns[j];
            j--;
        }
        columns[j + 1] = temp;
    }
}

/**
 * @brief Wrapper for the standard C library quicksort function.
 *
 * @details Provides a consistent interface to the standard library `qsort` function
 *          using the `compare_particles` function as the comparison callback.
 *
 * @param columns 2D array of particle data to be sorted (passed as `double**`).
 * @param n Number of elements (particles) to sort.
 */
static void stdlib_qsort_wrapper(double **columns, int n)
{
    qsort(columns, n, sizeof(double *), compare_particles);
}

/**
 * Wrapper for the external quadsort algorithm.
 *
 * Provides a consistent interface to the external quadsort function,
 * which is typically faster than standard quicksort for many distributions.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
static void quadsort_wrapper(double **columns, int n)
{
    quadsort(columns, n, sizeof(double *), compare_particles);
}

/**
 * @brief Performs insertion sort on a subarray of particle data.
 *
 * @details Operates on range `[start..end]` inclusive within the columns array.
 *          Used by parallel sorting algorithms to sort individual chunks and seam regions.
 *
 * @param columns 2D array of particle data containing the target subarray (`double**`).
 * @param start Starting index of the subarray (inclusive).
 * @param end Ending index of the subarray (inclusive).
 */
static void insertion_sort_sub(double **columns, int start, int end)
{
    for (int i = start + 1; i <= end; i++)
    {
        double *temp = columns[i];
        int j = i - 1;
        while (j >= start && compare_particles(&columns[j], &temp) > 0)
        {
            columns[j + 1] = columns[j];
            j--;
        }
        columns[j + 1] = temp;
    }
}

/**
 * Parallel insertion sort implementation using chunk-based approach with overlap.
 *
 * Divides the data into num_sort_sections chunks, sorts each chunk in parallel,
 * then fixes the boundaries between chunks by re-sorting overlap regions.
 * This approach balances parallelism with the need to ensure properly sorted output.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
/**
 * @brief Parallel implementation of insertion sort algorithm optimized for particle sorting.
 *
 * @details Uses multiple OpenMP threads to sort sections of particle data in parallel, 
 * followed by seam-fixing operations to ensure global ordering. Includes dynamic section
 * calculation and optimized overlap sizing based on chunk characteristics.
 *
 * @param columns Column-major data array [particle][component]
 * @param n Number of particles to sort
 */
void insertion_parallel_sort(double **columns, int n)
{
    // Dynamically determine number of sections based on runtime threads and constants.
    int active_num_sort_sections;
    #ifdef _OPENMP
        int n_runtime_threads = omp_get_max_threads(); 
        if (n_runtime_threads <= 0) n_runtime_threads = 1;
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD; 
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD;
        if (active_num_sort_sections <= 0) active_num_sort_sections = PARALLEL_SORT_DEFAULT_SECTIONS;
    #else
        active_num_sort_sections = 1; // Force serial behavior if OpenMP is not compiled in
    #endif

    // Ensure a reasonable number of sections
    if (active_num_sort_sections < 1) active_num_sort_sections = 1;
    if (n > 0 && active_num_sort_sections > n) active_num_sort_sections = n; 
    // Optional: Add a hard cap for maximum sections if desired, e.g.:
    // if (active_num_sort_sections > 96) active_num_sort_sections = 96;

    // Fallback to serial sort for small N or if chunks would be too small
    int estimated_avg_chunk_size = (n > 0 && active_num_sort_sections > 0) ? (n / active_num_sort_sections) : n;
    if (n < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD || \
        active_num_sort_sections <= 1 || \
        estimated_avg_chunk_size < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD) {
        // If PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD is set carefully (e.g. >= 2 * PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP),
        // this also helps ensure chunks are large enough for meaningful overlap.
        insertion_sort(columns, n); // Call serial insertion sort
        return;
    }

    // Determine chunk boundaries, distributing remainder
    int base_chunk_size = n / active_num_sort_sections;
    int remainder = n % active_num_sort_sections;
    int *startIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    int *endIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    if (!startIdx || !endIdx) { /* Handle error */ CLEAN_EXIT(1); }

    int offset = 0;
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        int size_c = base_chunk_size + (c < remainder ? 1 : 0);
        startIdx[c] = offset;
        endIdx[c] = offset + size_c - 1;
        offset += size_c;
    }

    // Sort each chunk in parallel using insertion sort
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        insertion_sort_sub(columns, startIdx[c], endIdx[c]);
    }

    // Calculate minChunkSize based on actual chunk distribution using active_num_sort_sections
    int minChunkSize = n; 
    if (active_num_sort_sections > 0 && n > 0 && endIdx && startIdx) { // Check endIdx/startIdx validity
        minChunkSize = (endIdx[0] - startIdx[0] + 1);
        for (int c = 1; c < active_num_sort_sections; c++) {
            int csize = endIdx[c] - startIdx[c] + 1;
            if (csize < minChunkSize) minChunkSize = csize;
        }
    }
    if (minChunkSize <= 0 && n > 0) minChunkSize = 1; // Safety for valid n

    int overlapSize;
    if (n <= 1 || active_num_sort_sections <= 1 || minChunkSize <= 0) {
        overlapSize = 0; 
    } else {
        int proportional_overlap = minChunkSize / PARALLEL_SORT_OVERLAP_DIVISOR;
        if (proportional_overlap == 0 && minChunkSize > 0) {
            proportional_overlap = 1; 
        }

        // Ensure overlap is at least the minimum required for correctness,
        // but only if that minimum does not make overlap excessive for chunk size
        if (PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP > 0 && proportional_overlap < PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP) {
            overlapSize = PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP;
        } else {
            overlapSize = proportional_overlap;
        }

        // Cap overlap at 50% of smallest chunk size
        // Handles cases where MIN_CORRECTNESS_OVERLAP exceeds chunk capacity
        int max_permissible_relative_overlap = minChunkSize / 2; // Example: Cap at 50% of chunk
        if (max_permissible_relative_overlap < 1 && minChunkSize > 0) max_permissible_relative_overlap = 1; // Ensure cap is at least 1 if chunk exists

        if (overlapSize > max_permissible_relative_overlap && minChunkSize > 1) {
            overlapSize = max_permissible_relative_overlap;
        }
        
        // Ensure minimum overlap when using multiple sections with valid data
        if (overlapSize == 0 && minChunkSize > 0 && active_num_sort_sections > 1) {
             overlapSize = 1; 
        }
    }
    if (overlapSize < 0) overlapSize = 0; // Final safety check
    // Additional absolute cap based on total N, mostly for sanity with very few sections.
    if (n > 1 && overlapSize > n / 2) overlapSize = n / 2;


    // Optional debug print (controlled by DEBUG_SORT_PARAMS environment variable)
    if (getenv("DEBUG_SORT_PARAMS") != NULL) {
        #ifdef _OPENMP
        if (omp_get_thread_num() == 0) // Print only from one thread
        #endif
        {
            printf("[NSPHERE_IS_PARALLEL_DEBUG] N=%d, Sections=%d, MinChunkSz=%d, OverlapSize=%d (Using DIV:%d, MIN_CORRECT:%d)\n",
                   n, active_num_sort_sections, minChunkSize, overlapSize,
                   PARALLEL_SORT_OVERLAP_DIVISOR, PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP);
            fflush(stdout);
        }
    }

    // Merge/fix the seams in parallel
    int nSeams = active_num_sort_sections - 1;
    if (overlapSize > 0 && nSeams > 0) {
#pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < nSeams; s++)
        {
            int c_left = s;
            int c_right = s + 1;

            // Robust boundary calculations for seam sorting
            int seam_sort_start = endIdx[c_left] - overlapSize + 1;
            if (seam_sort_start < startIdx[c_left]) seam_sort_start = startIdx[c_left];

            int seam_sort_end = startIdx[c_right] + overlapSize - 1;
            if (seam_sort_end > endIdx[c_right]) seam_sort_end = endIdx[c_right];

            // Sort the combined overlap region
            if (seam_sort_start <= seam_sort_end)
            {
                insertion_sort_sub(columns, seam_sort_start, seam_sort_end);
            }
        }
    }

    free(startIdx);
    free(endIdx);
}

/**
 * @brief CPU-based parallel radix sort optimized for particle data
 *
 * @details High-performance radix sort that operates directly on double-precision
 * radius values by converting them to sortable integer representations. Automatically
 * chooses between serial and parallel implementations based on array size.
 * Typically outperforms GPU sorting for arrays < 50M elements due to lower overhead.
 *
 * @param columns 2D array of particle data to be sorted [particle][component]
 * @param n Number of particles to sort
 */
static void parallel_radix_sort(double **columns, int n)
{
    if (n <= 1) return;

    // For small arrays, use insertion sort (better cache behavior)
    if (n < 1000) {
        insertion_sort(columns, n);
        return;
    }

    // Allocate working memory for keys and indices
    uint64_t *keys = (uint64_t*)malloc(n * sizeof(uint64_t));
    uint64_t *keys_temp = (uint64_t*)malloc(n * sizeof(uint64_t));
    int *indices = (int*)malloc(n * sizeof(int));
    int *indices_temp = (int*)malloc(n * sizeof(int));

    if (!keys || !keys_temp || !indices || !indices_temp) {
        fprintf(stderr, "ERROR: Failed to allocate memory for parallel radix sort\n");
        // Fall back to quadsort on allocation failure
        quadsort_wrapper(columns, n);
        goto cleanup;
    }

    // Convert doubles to sortable uint64 representation
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        indices[i] = i;
        double radius = columns[i][0];

        // Handle special cases
        if (isnan(radius)) {
            keys[i] = UINT64_MAX;  // NaN goes to end
        } else {
            // Convert to sortable integer - preserves ordering
            union { double d; uint64_t u; } val;
            val.d = radius;

            if (val.u & 0x8000000000000000ULL) {
                // Negative: flip all bits
                keys[i] = ~val.u;
            } else {
                // Positive: flip sign bit only
                keys[i] = val.u | 0x8000000000000000ULL;
            }
        }
    }

    // Select parallel or serial sorting based on array size
    int use_parallel = (n >= 100000);

    #ifdef _OPENMP
    int num_threads = use_parallel ? omp_get_max_threads() : 1;
    if (n < 1000000 && num_threads > 4) num_threads = 4;  // Limit threads for medium arrays
    #else
    int num_threads = 1;
    use_parallel = 0;
    #endif

    // Radix sort main loop - process 8 bits at a time
    for (int shift = 0; shift < 64; shift += RADIX_BITS) {

        if (use_parallel && num_threads > 1) {
            #ifdef _OPENMP
            // Parallel radix sort pass
            int global_hist[RADIX_SIZE] = {0};

            // Parallel histogram computation
            #pragma omp parallel num_threads(num_threads)
            {
                int local_hist[RADIX_SIZE] = {0};

                #pragma omp for nowait
                for (int i = 0; i < n; i++) {
                    int bucket = (keys[i] >> shift) & RADIX_MASK;
                    local_hist[bucket]++;
                }

                // Combine local histograms
                #pragma omp critical
                {
                    for (int b = 0; b < RADIX_SIZE; b++) {
                        global_hist[b] += local_hist[b];
                    }
                }
            }

            // Compute prefix sums (serial - fast for 256 bins)
            int offsets[RADIX_SIZE];
            offsets[0] = 0;
            for (int b = 1; b < RADIX_SIZE; b++) {
                offsets[b] = offsets[b-1] + global_hist[b-1];
            }

            // Reset histogram for position tracking
            memcpy(global_hist, offsets, RADIX_SIZE * sizeof(int));

            // Scatter to output positions (serial to maintain order)
            for (int i = 0; i < n; i++) {
                int bucket = (keys[i] >> shift) & RADIX_MASK;
                int pos = global_hist[bucket]++;
                keys_temp[pos] = keys[i];
                indices_temp[pos] = indices[i];
            }
            #endif
        } else {
            // Serial radix sort pass
            int hist[RADIX_SIZE] = {0};

            // Build histogram
            for (int i = 0; i < n; i++) {
                int bucket = (keys[i] >> shift) & RADIX_MASK;
                hist[bucket]++;
            }

            // Convert to offsets
            int offsets[RADIX_SIZE];
            offsets[0] = 0;
            for (int b = 1; b < RADIX_SIZE; b++) {
                offsets[b] = offsets[b-1] + hist[b-1];
            }

            // Scatter to sorted positions
            for (int i = 0; i < n; i++) {
                int bucket = (keys[i] >> shift) & RADIX_MASK;
                int pos = offsets[bucket]++;
                keys_temp[pos] = keys[i];
                indices_temp[pos] = indices[i];
            }
        }

        // Swap arrays for next iteration
        uint64_t *swap_keys = keys; keys = keys_temp; keys_temp = swap_keys;
        int *swap_indices = indices; indices = indices_temp; indices_temp = swap_indices;
    }

    // Rearrange columns array based on sorted indices
    double **sorted_columns = (double**)malloc(n * sizeof(double*));
    if (sorted_columns) {
        for (int i = 0; i < n; i++) {
            sorted_columns[i] = columns[indices[i]];
        }
        memcpy(columns, sorted_columns, n * sizeof(double*));
        free(sorted_columns);
    }

cleanup:
    if (keys) free(keys);
    if (keys_temp) free(keys_temp);
    if (indices) free(indices);
    if (indices_temp) free(indices_temp);
}

/**
 * Parallel quadsort implementation using chunk-based approach with overlap.
 *
 * Uses quadsort algorithm for both initial chunk sorting and seam fixing,
 * providing better performance than insertion sort for large datasets
 * while maintaining sort stability.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
/**
 * @brief Parallel implementation of quadsort algorithm optimized for particle sorting.
 *
 * @details Uses multiple OpenMP threads to sort sections of particle data in parallel, 
 * followed by seam-fixing operations to ensure global ordering. Includes dynamic section
 * calculation and optimized overlap sizing based on chunk characteristics.
 *
 * @param columns Column-major data array [particle][component]
 * @param n Number of particles to sort
 */
void quadsort_parallel_sort(double **columns, int n)
{
    // Dynamically determine number of sections based on runtime threads and constants.
    int active_num_sort_sections;
    #ifdef _OPENMP
        int n_runtime_threads = omp_get_max_threads(); 
        if (n_runtime_threads <= 0) n_runtime_threads = 1;
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD; 
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD;
        if (active_num_sort_sections <= 0) active_num_sort_sections = PARALLEL_SORT_DEFAULT_SECTIONS;
    #else
        active_num_sort_sections = 1; // Force serial behavior if OpenMP is not compiled in
    #endif

    // Ensure a reasonable number of sections
    if (active_num_sort_sections < 1) active_num_sort_sections = 1;
    if (n > 0 && active_num_sort_sections > n) active_num_sort_sections = n; 
    // Optional: Add a hard cap for maximum sections if desired, e.g.:
    // if (active_num_sort_sections > 96) active_num_sort_sections = 96;

    // Fallback to serial sort for small N or if chunks would be too small
    int estimated_avg_chunk_size = (n > 0 && active_num_sort_sections > 0) ? (n / active_num_sort_sections) : n;
    if (n < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD || \
        active_num_sort_sections <= 1 || \
        estimated_avg_chunk_size < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD) {
        // If PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD is set carefully (e.g. >= 2 * PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP),
        // this also helps ensure chunks are large enough for meaningful overlap.
        quadsort_wrapper(columns, n); // Call serial quadsort wrapper
        return;
    }

    // Determine chunk boundaries
    int base_chunk_size = n / active_num_sort_sections;
    int remainder = n % active_num_sort_sections;
    int *startIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    int *endIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    if (!startIdx || !endIdx) { /* Handle error */ CLEAN_EXIT(1); }

    int offset = 0;
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        int size_c = base_chunk_size + (c < remainder ? 1 : 0);
        startIdx[c] = offset;
        endIdx[c] = offset + size_c - 1;
        offset += size_c;
    }

    // Sort each chunk in parallel using quadsort
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        quadsort(&columns[startIdx[c]], endIdx[c] - startIdx[c] + 1, sizeof(double *), compare_particles);
    }

    // Calculate minChunkSize based on actual chunk distribution using active_num_sort_sections
    int minChunkSize = n; 
    if (active_num_sort_sections > 0 && n > 0 && endIdx && startIdx) { // Check endIdx/startIdx validity
        minChunkSize = (endIdx[0] - startIdx[0] + 1);
        for (int c = 1; c < active_num_sort_sections; c++) {
            int csize = endIdx[c] - startIdx[c] + 1;
            if (csize < minChunkSize) minChunkSize = csize;
        }
    }
    if (minChunkSize <= 0 && n > 0) minChunkSize = 1; // Safety for valid n

    int overlapSize;
    if (n <= 1 || active_num_sort_sections <= 1 || minChunkSize <= 0) {
        overlapSize = 0; 
    } else {
        int proportional_overlap = minChunkSize / PARALLEL_SORT_OVERLAP_DIVISOR;
        if (proportional_overlap == 0 && minChunkSize > 0) {
            proportional_overlap = 1; 
        }

        // Ensure overlap is at least the minimum required for correctness,
        // but only if that minimum does not make overlap excessive for chunk size
        if (PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP > 0 && proportional_overlap < PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP) {
            overlapSize = PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP;
        } else {
            overlapSize = proportional_overlap;
        }

        // Cap overlap at 50% of smallest chunk size
        // Handles cases where MIN_CORRECTNESS_OVERLAP exceeds chunk capacity
        int max_permissible_relative_overlap = minChunkSize / 2; // Example: Cap at 50% of chunk
        if (max_permissible_relative_overlap < 1 && minChunkSize > 0) max_permissible_relative_overlap = 1; // Ensure cap is at least 1 if chunk exists

        if (overlapSize > max_permissible_relative_overlap && minChunkSize > 1) {
            overlapSize = max_permissible_relative_overlap;
        }
        
        // Ensure minimum overlap when using multiple sections with valid data
        if (overlapSize == 0 && minChunkSize > 0 && active_num_sort_sections > 1) {
             overlapSize = 1; 
        }
    }
    if (overlapSize < 0) overlapSize = 0; // Final safety check
    // Additional absolute cap based on total N, mostly for sanity with very few sections.
    if (n > 1 && overlapSize > n / 2) overlapSize = n / 2;


    // Optional debug print (controlled by DEBUG_SORT_PARAMS environment variable)
    if (getenv("DEBUG_SORT_PARAMS") != NULL) {
        #ifdef _OPENMP
        if (omp_get_thread_num() == 0) // Print only from one thread
        #endif
        {
            printf("[NSPHERE_QS_PARALLEL_DEBUG] N=%d, Sections=%d, MinChunkSz=%d, OverlapSize=%d (Using DIV:%d, MIN_CORRECT:%d)\n",
                   n, active_num_sort_sections, minChunkSize, overlapSize,
                   PARALLEL_SORT_OVERLAP_DIVISOR, PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP);
            fflush(stdout);
        }
    }

    // Merge/fix the seams in parallel using quadsort
    int nSeams = active_num_sort_sections - 1;
    if (overlapSize > 0 && nSeams > 0) {
#pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < nSeams; s++)
        {
            int c_left = s;
            int c_right = s + 1;

            // Robust boundary calculations for seam sorting
            int seam_sort_start = endIdx[c_left] - overlapSize + 1;
            if (seam_sort_start < startIdx[c_left]) seam_sort_start = startIdx[c_left];

            int seam_sort_end = startIdx[c_right] + overlapSize - 1;
            if (seam_sort_end > endIdx[c_right]) seam_sort_end = endIdx[c_right];

            // Sort the combined overlap region
            if (seam_sort_start <= seam_sort_end)
            {
                int seam_len = seam_sort_end - seam_sort_start + 1;
                if (seam_len > 1) {  // Only sort if multiple elements exist
                    quadsort(&columns[seam_sort_start], seam_len, sizeof(double *), compare_particles);
                }
            }
        }
    }

    free(startIdx);
    free(endIdx);
}

/**
 * Validates sorting results by comparing with standard qsort.
 *
 * Creates a copy of the input array, sorts it with standard qsort,
 * then compares the results element by element to verify that the
 * sorting algorithm produced the expected ordering.
 *
 * @param columns 2D array of particle data that has been sorted
 * @param n Number of elements in the array
 * @param label Name of the sorting algorithm for diagnostic output
 */
void verify_sort_results(double **columns, int n, const char *label)
{
    // Create a temporary array of pointers to the columns
    double **tempCopy = (double **)malloc(n * sizeof(double *));
    if (!tempCopy) { fprintf(stderr, "Malloc failed in verify_sort_results\n"); return; }
    for (int i = 0; i < n; i++)
    {
        tempCopy[i] = columns[i];
    }

    // Sort the temporary pointer array using standard qsort
    stdlib_qsort_wrapper(tempCopy, n);

    // Compare the original sorted array with the qsort-ed copy
    long mismatches = 0;
    for (int i = 0; i < n; i++)
    {
        // Compare based on the actual data pointed to
        if (compare_particles(&columns[i], &tempCopy[i]) != 0)
        {
            mismatches++;
        }
    }

    if (mismatches == 0)
    {
        fprintf(stderr, "[DEBUG] SortAlg='%s': Verified => Results match standard qsort.\n", label);
    }
    else
    {
        fprintf(stderr, "[DEBUG] SortAlg='%s': *** MISMATCH *** => %ld rows differ from qsort.\n",
                label, mismatches);
    }

    free(tempCopy);
}

/**
 * Main function for sorting particle data using a specified algorithm.
 *
 * Transposes the data format from particles[component][particle] to
 * columns[particle][component], applies the specified sorting algorithm,
 * and then transposes back to the original format.
 *
 * @param particles 2D array of particle data to be sorted [component][particle]
 * @param npts Number of particles to sort
 * @param sortAlg String identifier of the sorting algorithm to use ("quadsort", "quadsort_parallel", or default to "insertion_parallel")
 */
/**
 * @brief Sorts particle data using the specified sorting algorithm.
 * @details Performs a three-phase particle sorting operation:
 *   1. Data transposition to a persistent, column-major buffer.
 *   2. Application of the selected sorting algorithm on the buffer.
 *   3. Reverse transposition of the sorted data back to the original array.
 *
 * The function uses a persistent global buffer (`g_sort_columns_buffer`) to
 * minimize memory allocation/deallocation overhead across repeated sort calls.
 *
 * @param particles 2D array of particle data to be sorted [component][particle]
 * @param npts Number of particles to sort
 * @param sortAlg Sorting algorithm to use ("quadsort", "quadsort_parallel",
 *               "insertion", or "insertion_parallel")
 */
void sort_particles_with_alg(double **particles, int npts, const char *sortAlg)
{

    /**
     * Phase 1: Memory allocation and data transposition
     * Prepares the column-major data format required for efficient sorting.
     */

    double **columns_to_sort_on; // Will point to the buffer used for sorting

    /**
     * Persistent buffer allocation strategy.
     * Uses a global buffer to reduce allocation overhead across multiple sort operations.
     */
    
    // Allocate or reallocate only if needed
    if (g_sort_columns_buffer == NULL || g_sort_columns_buffer_npts != npts) {
        // Free existing buffer if size has changed
        if (g_sort_columns_buffer != NULL) {
            for (int i = 0; i < g_sort_columns_buffer_npts; i++) {
                if (g_sort_columns_buffer[i]) free(g_sort_columns_buffer[i]);
            }
            free(g_sort_columns_buffer);
        }

        // Allocate new buffer with the required size
        g_sort_columns_buffer = (double **)malloc(npts * sizeof(double *));
        if (!g_sort_columns_buffer) {
            fprintf(stderr, "ERROR: Malloc failed for g_sort_columns_buffer in sort_particles_with_alg\n");
            CLEAN_EXIT(1);
        }
        
        // Allocate sub-arrays for each particle's components
        for (int i = 0; i < npts; i++) {
            g_sort_columns_buffer[i] = (double *)malloc(7 * sizeof(double));  // All 7 components including cos(phi) and sin(phi)
            if (!g_sort_columns_buffer[i]) {
                fprintf(stderr, "ERROR: Malloc failed for g_sort_columns_buffer[%d] in sort_particles_with_alg\n", i);
                // Clean up partial allocation
                for(int k=0; k<i; ++k) free(g_sort_columns_buffer[k]);
                free(g_sort_columns_buffer);
                g_sort_columns_buffer = NULL;
                CLEAN_EXIT(1);
            }
        }
        g_sort_columns_buffer_npts = npts;
    }

    // Transpose data from row-major (particles) to column-major (g_sort_columns_buffer)
    #pragma omp parallel for
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < 7; j++) {  // All 7 components including cos(phi) and sin(phi)
            g_sort_columns_buffer[i][j] = particles[j][i];
        }
    }
    columns_to_sort_on = g_sort_columns_buffer;


    /**
     * Phase 2: Apply the selected sorting algorithm
     * Uses one of several sorting algorithms based on input parameter or default.
     * Available algorithms include quadsort (sequential), quadsort_parallel,
     * insertion sort (sequential), and insertion_parallel (default).
     */

    // Select and apply the sorting algorithm
    const char *method = (sortAlg ? sortAlg : "insertion_parallel"); // Default if NULL

    if (strcmp(method, "benchmark_mode") == 0) {
        // Adaptive benchmarking mode: Start with insertion, benchmark every 1000 sorts,
        // and switch to the winning algorithm
        g_sort_call_count++;

        if (g_sort_call_count % 1000 == 0) {
            // Create copies for benchmarking
            double **copy1 = (double **)malloc(npts * sizeof(double *));
            double **copy2 = (double **)malloc(npts * sizeof(double *));
            if (!copy1 || !copy2) {
                fprintf(stderr, "ERROR: Failed to allocate memory for benchmark copies\n");
                if (copy1) free(copy1);
                if (copy2) free(copy2);
                // Fallback to current best
                if (strcmp(g_current_best_sort, "quadsort_parallel") == 0) {
                    quadsort_parallel_sort(columns_to_sort_on, npts);
                } else if (strcmp(g_current_best_sort, "parallel_radix") == 0) {
                    parallel_radix_sort(columns_to_sort_on, npts);
                } else {
                    insertion_parallel_sort(columns_to_sort_on, npts);
                }
                return;
            }

            // Allocate and copy data
            int alloc_failed = 0;
            for (int i = 0; i < npts && !alloc_failed; i++) {
                copy1[i] = (double *)malloc(7 * sizeof(double));
                copy2[i] = (double *)malloc(7 * sizeof(double));
                if (!copy1[i] || !copy2[i]) {
                    alloc_failed = 1;
                    // Cleanup
                    for (int k = 0; k <= i; k++) {
                        if (copy1[k]) free(copy1[k]);
                        if (copy2[k]) free(copy2[k]);
                    }
                }
                if (!alloc_failed) {
                    for (int j = 0; j < 7; j++) {
                        copy1[i][j] = columns_to_sort_on[i][j];
                        copy2[i][j] = columns_to_sort_on[i][j];
                    }
                }
            }

            if (alloc_failed) {
                free(copy1);
                free(copy2);
                // Fallback to current best
                if (strcmp(g_current_best_sort, "quadsort_parallel") == 0) {
                    quadsort_parallel_sort(columns_to_sort_on, npts);
                } else if (strcmp(g_current_best_sort, "parallel_radix") == 0) {
                    parallel_radix_sort(columns_to_sort_on, npts);
                } else {
                    insertion_parallel_sort(columns_to_sort_on, npts);
                }
                return;
            }

            struct timeval start, end;

            // Time insertion sort
            gettimeofday(&start, NULL);
            insertion_parallel_sort(columns_to_sort_on, npts);
            gettimeofday(&end, NULL);
            double time_insertion = ((end.tv_sec - start.tv_sec) * 1000.0) +
                                   ((end.tv_usec - start.tv_usec) / 1000.0);

            // Time quadsort on copy1
            gettimeofday(&start, NULL);
            quadsort_parallel_sort(copy1, npts);
            gettimeofday(&end, NULL);
            double time_quadsort = ((end.tv_sec - start.tv_sec) * 1000.0) +
                                  ((end.tv_usec - start.tv_usec) / 1000.0);

            // Time radix sort on copy2
            gettimeofday(&start, NULL);
            parallel_radix_sort(copy2, npts);
            gettimeofday(&end, NULL);
            double time_radix = ((end.tv_sec - start.tv_sec) * 1000.0) +
                               ((end.tv_usec - start.tv_usec) / 1000.0);

            // Verify all produce same result
            int results_match = 1;
            for (int i = 0; i < npts && results_match; i++) {
                if (fabs(columns_to_sort_on[i][0] - copy1[i][0]) > 1e-12 ||
                    fabs(columns_to_sort_on[i][0] - copy2[i][0]) > 1e-12) {
                    results_match = 0;
                }
            }

            // Update statistics
            g_benchmark_count++;
            g_total_insertion_time += time_insertion;
            g_total_quadsort_time += time_quadsort;
            g_total_radix_time += time_radix;

            // Determine winner and update
            double min_time = time_insertion;
            strcpy(g_current_best_sort, "insertion_parallel");
            g_insertion_wins++;

            if (time_quadsort < min_time) {
                min_time = time_quadsort;
                strcpy(g_current_best_sort, "quadsort_parallel");
                g_insertion_wins--;  // Correct the increment
                g_quadsort_wins++;
            } else if (time_radix < min_time) {
                min_time = time_radix;
                strcpy(g_current_best_sort, "parallel_radix");
                g_insertion_wins--;  // Correct the increment
                g_radix_wins++;
            }

            // Cleanup copies
            for (int i = 0; i < npts; i++) {
                free(copy1[i]);
                free(copy2[i]);
            }
            free(copy1);
            free(copy2);
        } else {
            // Regular sort with current best algorithm
            if (strcmp(g_current_best_sort, "quadsort_parallel") == 0) {
                quadsort_parallel_sort(columns_to_sort_on, npts);
            } else if (strcmp(g_current_best_sort, "parallel_radix") == 0) {
                parallel_radix_sort(columns_to_sort_on, npts);
            } else {
                insertion_parallel_sort(columns_to_sort_on, npts);
            }
        }
    }
    else if (strcmp(method, "quadsort") == 0) {
        quadsort_wrapper(columns_to_sort_on, npts);
    }
    else if (strcmp(method, "quadsort_parallel") == 0) {
        quadsort_parallel_sort(columns_to_sort_on, npts);
    }
    else if (strcmp(method, "insertion") == 0) {
        insertion_sort(columns_to_sort_on, npts);
    }
    else if (strcmp(method, "parallel_radix") == 0) {
        parallel_radix_sort(columns_to_sort_on, npts);
    }
    else { // Default to parallel insertion sort
        insertion_parallel_sort(columns_to_sort_on, npts);
    }


    /**
     * Phase 3: Data transposition and memory cleanup
     * Restores sorted data to original format and performs appropriate cleanup.
     */

    // Transpose sorted data back to the original format
    #pragma omp parallel for
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < 7; j++) {  // All 7 components including cos(phi) and sin(phi)
            particles[j][i] = columns_to_sort_on[i][j];
        }
    }
    // Note: Persistent buffer g_sort_columns_buffer remains allocated for reuse
    //       across subsequent sort operations; freed only at program exit.


}

/**
 * @brief Convenience wrapper function for sorting particles with the default algorithm.
 * @details Calls sort_particles_with_alg using the default sorting algorithm specified in g_defaultSortAlg.
 *
 * @param particles [in,out] 2D array of particle data to be sorted [component][particle].
 * @param npts [in] Number of particles to sort.
 */
void sort_particles(double **particles, int npts)
{
    sort_particles_with_alg(particles, npts, g_defaultSortAlg);
}

/**
 * @brief Comparison function for qsort to sort PartData structures by radius.
 * @details Compares two PartData structures based on their `rad` (radius) member
 *          for sorting in ascending order. Handles NaN values by placing them
 *          consistently at the beginning (NaN values sorted first).
 *
 * @param a Pointer to the first PartData structure.
 * @param b Pointer to the second PartData structure.
 * @return int -1 if pa->rad < pb->rad, 1 if pa->rad > pb->rad, 0 otherwise.
 *             Specific return for NaNs ensures consistent ordering.
 */
int compare_partdata_by_rad(const void *a, const void *b)
{
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;

    const struct PartData *pa = (const struct PartData *)a;
    const struct PartData *pb = (const struct PartData *)b;

    int pa_is_nan = (pa->rad != pa->rad); // Check for NaN (safe with fast-math)
    int pb_is_nan = (pb->rad != pb->rad); // Check for NaN (safe with fast-math)

    if (pa_is_nan && pb_is_nan) return 0;
    if (pa_is_nan) return -1;
    if (pb_is_nan) return 1;

    if (pa->rad < pb->rad) return -1;
    if (pa->rad > pb->rad) return 1;
    return 0;
}

/**
 * @brief Sorts an array of PartData structures by their radial position (`rad`).
 * @details Uses the standard library `qsort` function with `compare_partdata_by_rad`
 *          as the comparison function. Includes basic safety checks for NULL array
 *          or non-positive `npts`. The sort is performed in-place.
 *
 * @param array [in,out] Array of PartData structures to be sorted.
 * @param npts  [in] Number of elements in the array.
 */
void sort_by_rad(struct PartData *array, int npts)
{
    if (!array)
    {
        fprintf(stderr, "ERROR: sort_by_rad called with NULL array\n");
        return;
    }
    if (npts <= 0)
    {
        // Sorting an empty or negatively sized array is meaningless or an error.
        // fprintf(stderr, "Warning: sort_by_rad called with npts <= 0: %d\n", npts);
        return; // Nothing to sort
    }

    qsort(array, (size_t)npts, sizeof(struct PartData), compare_partdata_by_rad);
}

// =============================================================================
// SPLINE DATA SORTING UTILITIES
// =============================================================================
//
// Utility functions and structures for sorting spline data arrays.
// Provides mechanisms to sort arrays used for GSL spline creation (like radius `r`
// and potential `Psi`) while maintaining the correct correspondence between
// paired values after sorting based on one of the arrays (typically radius).


/**
 * @brief Comparison function for qsort to sort RrPsiPair structures by the 'rr' member.
 * @details Used to sort an array of RrPsiPair structures in ascending order
 *          based on their radial (`rr`) component. This is primarily used when
 *          preparing data for splines where the x-axis (e.g., radius or -Psi)
 *          must be strictly monotonic.
 *
 * @param a Pointer to the first RrPsiPair structure.
 * @param b Pointer to the second RrPsiPair structure.
 * @return int -1 if pa->rr < pb->rr, 1 if pa->rr > pb->rr, 0 otherwise.
 */
int compare_pair_by_first_element(const void *a, const void *b)
{
    const struct RrPsiPair *pa = (const struct RrPsiPair *)a;
    const struct RrPsiPair *pb = (const struct RrPsiPair *)b;
    if (pa->rr < pb->rr)
        return -1;
    if (pa->rr > pb->rr)
        return 1;
    return 0;
}

/**
 * @brief Sorts radius and potential arrays in tandem, maintaining their correspondence.
 * @details This function takes an array of radial coordinates (`rrA_spline`) and an
 *          array of corresponding potential values (`psiAarr_spline`). It sorts
 *          `rrA_spline` in ascending order and applies the identical swaps to
 *          `psiAarr_spline`, ensuring that `psiAarr_spline[i]` still corresponds to
 *          `rrA_spline[i]` after sorting. This is crucial for creating GSL splines
 *          where the x-array must be strictly monotonic and the y-array must maintain
 *          its pairing with the x-values. The arrays are assumed to have `npts` elements.
 *
 * @param rrA_spline    [in,out] Array of radial coordinates to be sorted. Modified in-place.
 * @param psiAarr_spline [in,out] Array of corresponding Psi values. Modified in-place in tandem with `rrA_spline`.
 * @param npts          The number of points in the arrays (arrays are of size `npts`).
 */
void sort_rr_psi_arrays(double *rrA_spline, double *psiAarr_spline, int npts)
{
    // Allocate temporary array of pairs
    struct RrPsiPair *pairs = (struct RrPsiPair *)malloc(npts * sizeof(struct RrPsiPair));
    if (!pairs)
    {
        perror("malloc failed in sort_rr_psi_arrays");
        CLEAN_EXIT(EXIT_FAILURE);
    }

    // Populate the pairs array
    for (int i = 0; i < npts; i++)
    {
        pairs[i].rr = rrA_spline[i];
        pairs[i].psi = psiAarr_spline[i];
    }

    // Sort the pairs based on the radius value
    qsort(pairs, npts, sizeof(struct RrPsiPair), compare_pair_by_first_element);

    // Copy the sorted data back into the original arrays
    for (int i = 0; i < npts; i++)
    {
        rrA_spline[i] = pairs[i].rr;
        psiAarr_spline[i] = pairs[i].psi;
    }

    free(pairs);
}

/**
 * @brief Constructs a three-dimensional vector from its Cartesian components.
 * @details This utility function initializes a `threevector` structure with the
 *          provided x, y, and z components. It serves as a convenient constructor.
 *
 * @param x [in] The x-component of the vector.
 * @param y [in] The y-component of the vector.
 * @param z [in] The z-component of the vector.
 * @return threevector An initialized `threevector` structure.
 */
threevector make_threevector(double x, double y, double z) {
    return (threevector){x, y, z};
}

/**
 * @brief Computes the scalar dot product of two three-dimensional vectors.
 * @details Calculates \f$X \cdot Y = X_x Y_x + X_y Y_y + X_z Y_z\f$.
 *          The dot product is a measure of the projection of one vector onto another
 *          and is used in various physics calculations, such as determining the
 *          magnitude squared of a vector (\f$V \cdot V = |V|^2\f$) or the angle between vectors.
 *
 * @param X [in] The first threevector operand.
 * @param Y [in] The second threevector operand.
 * @return double The scalar result of the dot product \f$X \cdot Y\f$.
 */
double dotproduct(threevector X, threevector Y) {
  return X.x * Y.x + X.y * Y.y + X.z * Y.z;
}

/**
 * @brief Computes the vector cross product of two three-dimensional vectors.
 * @details Calculates \f$Z = X \times Y\f$, where \f$X = (X_x, X_y, X_z)\f$ and \f$Y = (Y_x, Y_y, Y_z)\f$.
 *          The components of the resulting vector \f$Z\f$ are determined by:
 *          \f$Z_x = X_y Y_z - X_z Y_y\f$
 *          \f$Z_y = X_z Y_x - X_x Y_z\f$
 *          \f$Z_z = X_x Y_y - X_y Y_x\f$
 *          This follows the standard right-hand rule for vector cross products.
 *
 * @param X [in] The first threevector operand.
 * @param Y [in] The second threevector operand.
 * @return threevector The resulting vector \f$Z = X \times Y\f$.
 */
threevector crossproduct(threevector X, threevector Y) {
    threevector Z;
    Z.x = X.y * Y.z - X.z * Y.y;
    Z.y = X.z * Y.x - X.x * Y.z;
    Z.z = X.x * Y.y - X.y * Y.x;
    return Z;
}

/**
 * @brief Calculates the total kinetic and potential energy of the N-body system.
 * @details This function iterates over all particles to compute the system's total
 *          kinetic energy (KE) and potential energy (PE) at a specific snapshot in time.
 *          The `particles` array must be sorted by radius before calling this function
 *          to ensure the rank-based potential energy calculation is correct.
 *
 * @param particles         [in] The main particle data array, sorted by radius.
 * @param npts              [in] The total number of particles in the simulation.
 * @param deltaM            [in] The mass of a single particle (\f$M_{\odot}\f$).
 * @param total_KE_out      [out] Pointer to store the calculated total kinetic energy.
 * @param total_PE_out      [out] Pointer to store the calculated total potential energy.
 */
static void calculate_system_energies(double** particles, int npts, double deltaM,
                                      double* total_KE_out, double* total_PE_out)
{
    double KE_sum = 0.0;
    double PE_sum = 0.0;

    // This loop is parallelized with a reduction for summing energy components.
    #pragma omp parallel for reduction(+:KE_sum, PE_sum)
    for (int i = 0; i < npts; i++) {
        double r = particles[0][i];
        double v_rad = particles[1][i];
        double L = particles[2][i];

        // Kinetic Energy per unit mass for particle i: 0.5 * (v_r^2 + v_t^2)
        // where v_t = L / r.
        if (r > 1e-12) { // Avoid division by zero for particles at the origin
            double ke_per_mass = 0.5 * (v_rad * v_rad + (L * L) / (r * r));
            KE_sum += ke_per_mass;
        }

        // Potential Energy per unit mass for particle i.
        // The potential is due to the mass enclosed within its rank-ordered position.
        double M_enclosed = (double)(i + 1) * deltaM;
        if (r > 1e-12) { // Avoid division by zero
            double pe_per_mass = -(G_CONST * VEL_CONV_SQ) * M_enclosed / r;
            PE_sum += pe_per_mass;
        }
    }

    // Scale the summed energies-per-unit-mass by the particle mass to get total energies.
    *total_KE_out = KE_sum * deltaM;
    *total_PE_out = PE_sum * deltaM;
}

/**
 * @brief Calculates the self-interacting dark matter (SIDM) scattering cross-section.
 * @details Implements a velocity-independent cross-section model where the opacity
 *          \f$\sigma/m = \kappa\f$ is constant. The function uses the global
 *          variable `g_sidm_kappa` for the opacity \f$\kappa\f$. The total cross-section
 *          \f$\sigma\f$ scales with the individual particle mass, \f$m_{particle}\f$, derived
 *          from the total `halo_mass_for_calc` and the number of particles `npts`.
 *          The function includes unit conversions to return the cross-section
 *          in simulation units (kpc\f$^2\f$).
 *
 *          Unit Conversion Detail:
 *          \f$\kappa\f$ (cm²/g) \f$\times m_{particle}\f$ (\f$M_{\odot}\f$) \f$\rightarrow \sigma\f$ (kpc²)
 *          The conversion factor is \f$2.089 \times 10^{-10} \text{ (kpc}^2 \text{ } M_{\odot}^{-1}) / (\text{cm}^2 \text{ g}^{-1})\f$.
 *
 * @param vrel                 [in] Relative velocity between particles (kpc/Myr). This
 *                             parameter is unused in the velocity-independent model.
 * @param npts                 [in] Total number of simulation particles, used for calculating \f$m_{particle}\f$.
 * @param halo_mass_for_calc [in] Total halo mass (\f$M_{\odot}\f$) for the active profile, used for \f$m_{particle}\f$.
 * @param rc_for_calc          [in] Scale radius (kpc) of the active profile. This
 *                             parameter is unused in the velocity-independent model.
 * @return double Total scattering cross-section \f$\sigma\f$ in kpc\f$^2\f$. Returns 0.0 if `npts` or
 *                the calculated `particle_mass_Msun` is non-positive.
 */
double sigmatotal(double vrel, int npts, double halo_mass_for_calc, double rc_for_calc) {
  (void)vrel; // Unused in velocity-independent model
  (void)rc_for_calc; // Unused in velocity-independent model
  double kappa = g_sidm_kappa; // Self-interaction opacity parameter (cm²/g)
  // Ensure npts is positive to prevent division by zero or negative particle mass
  if (npts <= 0) {
    return 0.0;
  }
  double particle_mass_Msun = halo_mass_for_calc / ((double)npts);
  if (particle_mass_Msun <= 0) {
    return 0.0;
  }
  return 2.089e-10 * kappa * particle_mass_Msun; // Cross-section (kpc²)
}

/**
 * @brief Executes self-interacting dark matter (SIDM) scattering for one simulation timestep (Serial version).
 * @details Implements a serial SIDM scattering algorithm with persistent azimuthal angle tracking.
 *          For each particle `i` (the primary scatterer), it considers up to `nscat` (typically 10)
 *          subsequent particles in the array as potential scattering partners. The `particles` array
 *          is sorted by radius, making these neighbors spatially proximate.
 *
 *          **Azimuthal Angle Treatment:**
 *          The azimuthal angle φ for each particle's perpendicular velocity is stored persistently
 *          as cos(φ) and sin(φ) in `particles[5]` and `particles[6]`. These values are initialized
 *          randomly at simulation start and updated only during scattering events. Between scatters,
 *          φ remains constant (no orbital precession tracking). This approximation is valid because:
 *          - In high-scatter regimes, φ evolves minimally between frequent scattering events.
 *          - In low-scatter regimes, stochastic partner selection and orbital dynamics erase
 *            correlations from prior scattering history.
 *          This approach maintains physical accuracy while avoiding costly random number generation
 *          at every timestep.
 *
 *          **Algorithm Steps:**
 *          1. Loads persistent φ orientation (cos(φ), sin(φ)) from `particles[5]` and `particles[6]`.
 *          2. For each of the `nscat` potential partners `m`:
 *             a. Loads partner's φ orientation and calculates relative azimuthal separation.
 *             b. Computes 3D relative velocity \f$v_{rel}\f$ using Law of Cosines.
 *             c. Calculates interaction rate \f$\Gamma_m = \sigma(v_{rel}) v_{rel}\f$.
 *             d. Accumulates total interaction rate \f$\Gamma_{tot} = \sum \Gamma_m\f$.
 *          3. Estimates local particle number density using shell volume approximation.
 *          4. Calculates scattering probability: \f$P_{scatter} = 1 - \exp(-\Gamma_{tot} \times dt / (2V_{shell}))\f$.
 *          5. If scatter occurs (stochastic decision):
 *             a. Selects partner weighted by individual \f$\Gamma_m\f$ contributions.
 *             b. Performs isotropic scattering in center-of-mass frame.
 *             c. Updates both particles' velocities (radial and perpendicular components).
 *             d. Updates persistent φ storage (cos(φ), sin(φ)) for both particles.
 *             e. Sets `g_particle_scatter_state` flags for AB3 integrator history management.
 *
 * @param particles             [in,out] Main particle data array: `particles[component][current_sorted_index]`.
 *                              Modified in-place with post-scattering velocities/angular momenta.
 * @param npts                  [in] Total number of simulation particles.
 * @param dt                    [in] Integration timestep (Myr).
 * @param current_time          [in] Current simulation time (Myr). Marked `unused` but available.
 * @param rng                   [in] GSL random number generator instance for all stochastic processes.
 * @param Nscatter_total_step   [out] Pointer to a long long to accumulate total scattering events this timestep.
 * @param halo_mass_for_sidm    [in] Total halo mass (\f$M_{\odot}\f$) for the active profile, passed to `sigmatotal`.
 * @param rc_for_sidm           [in] Scale radius (kpc) for the active profile, passed to `sigmatotal`.
 * @param current_scatter_counts [in,out] Per-particle scatter count array for tracking (can be NULL).
 */

void perform_sidm_scattering_serial(double **particles, int npts, double dt, double current_time, gsl_rng *rng, long long *Nscatter_total_step, double halo_mass_for_sidm, double rc_for_sidm, int *current_scatter_counts) {
    (void)current_time; // Unused parameter
    (void)halo_mass_for_sidm; // Unused parameter
    (void)rc_for_sidm; // Unused parameter
    long long Nscatters_this_call = 0;
    int i;

    // Note: scatter counts are accumulated between dtwrite intervals
    // and reset by the main loop after writing, not here

    extern int g_sidm_max_interaction_range;

    for (i = 0; i < npts - 1; i++) {
        int nscat = g_sidm_max_interaction_range;
        if (npts - 1 - i < nscat) nscat = npts - 1 - i;
        if (nscat <= 0) continue;

        double partialprobability[nscat + 1];
        double probability_sum_term = 0.0;

        double v_rad_i = particles[1][i];
        double v_ti = particles[2][i] / particles[0][i];

        // Load persistent phi orientation angles
        double cos_phi_i = particles[5][i];
        double sin_phi_i = particles[6][i];

        // Calculate interaction rates with neighboring particles using correct 3D relative velocity
        for (int m = 1; m <= nscat; m++) {
            int partner_idx = i + m;

            double v_rad_m = particles[1][partner_idx];
            double v_tm = particles[2][partner_idx] / particles[0][partner_idx];

            // Load neighbor's phi and calculate cos(phi_i - phi_m)
            double cos_phi_m = particles[5][partner_idx];
            double sin_phi_m = particles[6][partner_idx];
            double cos_alpha = cos_phi_i * cos_phi_m + sin_phi_i * sin_phi_m;

            // Calculate v_rel^2 using the Law of Cosines (no 3D vectors needed)
            double v_rel_rad_sq = sqr(v_rad_i - v_rad_m);
            double v_rel_tan_sq = sqr(v_ti) + sqr(v_tm) - 2.0 * v_ti * v_tm * cos_alpha;
            double vrel_val = sqrt(v_rel_rad_sq + v_rel_tan_sq);

            partialprobability[m] = sigmatotal(vrel_val, npts, halo_mass_for_sidm, rc_for_sidm) * vrel_val;
            probability_sum_term += partialprobability[m];
        }

        // Determine the outer radius of the shell containing these nscat neighbors
        // Shell outer radius defined by the (nscat+1)-th particle
        int use_nscat_plus_1_for_shell = 1;
        int outer_shell_particle_idx_for_vol;

        if (use_nscat_plus_1_for_shell) {
            outer_shell_particle_idx_for_vol = i + nscat + 1;
        } else {
            outer_shell_particle_idx_for_vol = i + nscat;
        }

        // Ensure the chosen outer index is within bounds
        if (outer_shell_particle_idx_for_vol >= npts) {
            // If out of bounds, try to use the last available particle as the boundary
            if (i + nscat < npts) {
                outer_shell_particle_idx_for_vol = i + nscat;
            } else {
                // No valid shell can be formed
                probability_sum_term = 0.0; // Force no scatter, skip probability calculation
            }
        }


        double radius_diff = 0.0;
        // Calculate radius_diff only if scattering probability exists and shell is valid
        if (probability_sum_term > 1e-30 && (outer_shell_particle_idx_for_vol > i)) {
            radius_diff = particles[0][outer_shell_particle_idx_for_vol] - particles[0][i];
        } else {
            probability_sum_term = 0.0; // Ensure no scatter if shell is invalid
        }

        double probability = 0.0;
        if (radius_diff > 1e-15 && particles[0][i] > 1e-15 && probability_sum_term > 1e-30) {
            // Calculate scattering rate using shell volume approximation
            double rate_dt = probability_sum_term * (0.5) * dt / (4.0 * PI * sqr(particles[0][i]) * radius_diff);

            // Apply proper Poisson process transformation: P' = 1 - exp(-rate*dt)
            // For small rate*dt, this approximates to rate*dt
            // For large rate*dt, this correctly saturates at 1.0
            probability = 1.0 - exp(-rate_dt);
        }

        // Stochastic scattering determination with conditional RNG calls
        if (gsl_rng_uniform(rng) < probability) {
            Nscatters_this_call++;
            int m_scatter = 1; // Default to first neighbor

            // Weighted selection among multiple neighbors
            if (nscat > 1 && probability_sum_term > 1e-15) {
                double cumulative_prob[nscat + 1];
                cumulative_prob[0] = 0.0;
                for (int k = 1; k <= nscat; k++) {
                    cumulative_prob[k] = (k > 1 ? cumulative_prob[k - 1] : 0.0) + partialprobability[k] / probability_sum_term;
                }
                if (nscat > 0) cumulative_prob[nscat] = 1.0;

                // Simple uniform random for partner selection
                double random_select = gsl_rng_uniform(rng);
                m_scatter = 1;
                // Select partner based on cumulative probability distribution
                while (m_scatter < nscat && random_select > cumulative_prob[m_scatter]) {
                    m_scatter++;
                }
            }

            int actual_partner_idx = i + m_scatter;
            if (actual_partner_idx >= npts) {
                Nscatters_this_call--;
                continue;
            }

            // --- Reconstruct state for the actual scatter partner ---
            double v_rad_m = particles[1][actual_partner_idx];
            double v_tm = particles[2][actual_partner_idx] / particles[0][actual_partner_idx];

            // Load stored phi orientation angles for both particles
            double cos_phi_i = particles[5][i];
            double sin_phi_i = particles[6][i];
            double cos_phi_m = particles[5][actual_partner_idx];
            double sin_phi_m = particles[6][actual_partner_idx];

            // --- Phase 1: Forward Transformation (Rotate to Align) ---
            double cos_alpha = cos_phi_i * cos_phi_m + sin_phi_i * sin_phi_m;
            double sin_alpha = sin_phi_i * cos_phi_m - cos_phi_i * sin_phi_m; // sin(a-b)

            // Construct 3D vectors in the rotated frame (m is aligned with x')
            threevector Vi_prime = make_threevector(v_ti * cos_alpha, v_ti * sin_alpha, v_rad_i);
            threevector Vm_prime = make_threevector(v_tm, 0.0, v_rad_m);

            threevector Vrel_prime = make_threevector(Vi_prime.x - Vm_prime.x, Vi_prime.y - Vm_prime.y, Vi_prime.z - Vm_prime.z);
            double vrel_val = sqrt(dotproduct(Vrel_prime, Vrel_prime));

            if (vrel_val > 1e-15) {
                // Phase 2: Scattering Physics with simple RNG calls
                threevector V_cm_prime = make_threevector((Vi_prime.x + Vm_prime.x)/2.0, (Vi_prime.y + Vm_prime.y)/2.0, (Vi_prime.z + Vm_prime.z)/2.0);
                double costheta = 2.0 * gsl_rng_uniform(rng) - 1.0;  // Simple uniform in [-1, 1]
                double sintheta = sqrt(fmax(0.0, 1.0 - costheta*costheta));
                double phif_scatter = 2.0 * PI * gsl_rng_uniform(rng);  // Simple uniform in [0, 2π]
                double cf = cos(phif_scatter);
                double sf = sin(phif_scatter);
                threevector nhat0,nhat1,nhat2,nhatref;
                nhat0=make_threevector(Vrel_prime.x/vrel_val,Vrel_prime.y/vrel_val,Vrel_prime.z/vrel_val);
                if(fabs(nhat0.z)<0.999)nhatref=make_threevector(0.0,0.0,1.0);else nhatref=make_threevector(1.0,0.0,0.0);
                nhat1=crossproduct(nhat0,nhatref);double normnhat1=sqrt(dotproduct(nhat1,nhat1));
                if(normnhat1<1e-15){if(fabs(nhat0.x)<0.999)nhatref=make_threevector(1.0,0.0,0.0);else nhatref=make_threevector(0.0,1.0,0.0);
                nhat1=crossproduct(nhat0,nhatref);normnhat1=sqrt(dotproduct(nhat1,nhat1));if(normnhat1<1e-15){Nscatters_this_call--;continue;}}
                nhat1=make_threevector(nhat1.x/normnhat1,nhat1.y/normnhat1,nhat1.z/normnhat1); nhat2=crossproduct(nhat0,nhat1);
                threevector nhat_perp_rotated=make_threevector(nhat1.x*cf+nhat2.x*sf,nhat1.y*cf+nhat2.y*sf,nhat1.z*cf+nhat2.z*sf);
                threevector V_rel_final_half=make_threevector((vrel_val/2.0)*(costheta*nhat0.x+sintheta*nhat_perp_rotated.x),(vrel_val/2.0)*(costheta*nhat0.y+sintheta*nhat_perp_rotated.y),(vrel_val/2.0)*(costheta*nhat0.z+sintheta*nhat_perp_rotated.z));

                threevector Vifinal_prime = make_threevector(V_cm_prime.x+V_rel_final_half.x,V_cm_prime.y+V_rel_final_half.y,V_cm_prime.z+V_rel_final_half.z);
                threevector Vmfinal_prime = make_threevector(V_cm_prime.x-V_rel_final_half.x,V_cm_prime.y-V_rel_final_half.y,V_cm_prime.z-V_rel_final_half.z);

                // Phase 3: Inverse Transformation (Decompose and Un-rotate)
                particles[1][i] = Vifinal_prime.z;
                particles[1][actual_partner_idx] = Vmfinal_prime.z;

                double Vperp_i_final_mag = sqrt(sqr(Vifinal_prime.x)+sqr(Vifinal_prime.y));
                double Vperp_m_final_mag = sqrt(sqr(Vmfinal_prime.x)+sqr(Vmfinal_prime.y));

                particles[2][i] = particles[0][i] * Vperp_i_final_mag;
                particles[2][actual_partner_idx] = particles[0][actual_partner_idx] * Vperp_m_final_mag;

                // STORED PHI METHOD: Extract and update phi values after scattering
                // Extract new cos/sin in rotated frame (where m was aligned with x-axis)
                double cos_phi_i_final_prime = (Vperp_i_final_mag > 1e-15) ? Vifinal_prime.x / Vperp_i_final_mag : 1.0;
                double sin_phi_i_final_prime = (Vperp_i_final_mag > 1e-15) ? Vifinal_prime.y / Vperp_i_final_mag : 0.0;
                double cos_phi_m_final_prime = (Vperp_m_final_mag > 1e-15) ? Vmfinal_prime.x / Vperp_m_final_mag : 1.0;
                double sin_phi_m_final_prime = (Vperp_m_final_mag > 1e-15) ? Vmfinal_prime.y / Vperp_m_final_mag : 0.0;

                // Un-rotate back to global frame using angle addition formulas: cos(a+b) and sin(a+b)
                // These stored values are used in the next timestep for any subsequent scatters
                particles[5][i] = cos_phi_i_final_prime * cos_phi_m - sin_phi_i_final_prime * sin_phi_m;
                particles[6][i] = sin_phi_i_final_prime * cos_phi_m + cos_phi_i_final_prime * sin_phi_m;
                particles[5][actual_partner_idx] = cos_phi_m_final_prime * cos_phi_m - sin_phi_m_final_prime * sin_phi_m;
                particles[6][actual_partner_idx] = sin_phi_m_final_prime * cos_phi_m + cos_phi_m_final_prime * sin_phi_m;

                // Mark both scattered particles in case it is needed elsewhere
                int orig_id1 = (int)particles[3][i];
                int orig_id2 = (int)particles[3][actual_partner_idx];
                if (orig_id1 >= 0 && orig_id1 < npts) g_particle_scatter_state[orig_id1] = 1;
                if (orig_id2 >= 0 && orig_id2 < npts) g_particle_scatter_state[orig_id2] = 1;

                // Increment scatter counts for both particles if buffer is provided
                if (current_scatter_counts != NULL) {
                    if (orig_id1 >= 0 && orig_id1 < npts) current_scatter_counts[orig_id1]++;
                    if (orig_id2 >= 0 && orig_id2 < npts) current_scatter_counts[orig_id2]++;
                }
            } else {
                Nscatters_this_call--;
            }
        }
    }

    *Nscatter_total_step = Nscatters_this_call;
}


// =============================================================================
// GRAPH COLORING PARALLEL SIDM IMPLEMENTATION
// =============================================================================

/**
 * @brief Helper function to perform scatter update for two particles
 * @details Updates velocities and angular components for both particles after a
 *          scattering event. The graph coloring algorithm uses this function to
 *          directly update particle states, avoiding intermediate buffering.
 *
 * @param particles Particle data arrays
 * @param i First particle index
 * @param j Second particle index
 * @param rng Random number generator for scattering angles
 */
static inline void perform_scatter_update(
    double **particles,
    int i,
    int j,
    gsl_rng *rng
) {
    // Get current velocities
    double cos_phi_i = particles[5][i];
    double sin_phi_i = particles[6][i];
    double Viperp = particles[2][i] / particles[0][i];
    threevector Vi = make_threevector(Viperp * cos_phi_i, Viperp * sin_phi_i, particles[1][i]);

    double cos_phi_j = particles[5][j];
    double sin_phi_j = particles[6][j];
    double Vjperp = particles[2][j] / particles[0][j];
    threevector Vj = make_threevector(Vjperp * cos_phi_j, Vjperp * sin_phi_j, particles[1][j]);

    // Calculate relative velocity
    threevector Vrel = make_threevector(Vi.x - Vj.x, Vi.y - Vj.y, Vi.z - Vj.z);
    double vrel = sqrt(dotproduct(Vrel, Vrel));

    if (vrel < 1e-15) return;  // No scattering for zero relative velocity

    // Generate random scattering angles
    double costheta = 2.0 * gsl_rng_uniform(rng) - 1.0;
    double sintheta = sqrt(fmax(0.0, 1.0 - costheta * costheta));
    double phif = 2.0 * PI * gsl_rng_uniform(rng);
    double cf = cos(phif);
    double sf = sin(phif);

    // Build orthonormal basis for rotation
    threevector nhat0, nhat1, nhat2, nhatref;
    nhat0 = make_threevector(Vrel.x/vrel, Vrel.y/vrel, Vrel.z/vrel);

    if (fabs(nhat0.z) < 0.999) {
        nhatref = make_threevector(0.0, 0.0, 1.0);
    } else {
        nhatref = make_threevector(1.0, 0.0, 0.0);
    }

    nhat1 = crossproduct(nhat0, nhatref);
    double normnhat1 = sqrt(dotproduct(nhat1, nhat1));

    if (normnhat1 < 1e-15) {
        if (fabs(nhat0.x) < 0.999) {
            nhatref = make_threevector(1.0, 0.0, 0.0);
        } else {
            nhatref = make_threevector(0.0, 1.0, 0.0);
        }
        nhat1 = crossproduct(nhat0, nhatref);
        normnhat1 = sqrt(dotproduct(nhat1, nhat1));
        if (normnhat1 < 1e-15) return;  // Degenerate case
    }

    nhat1 = make_threevector(nhat1.x/normnhat1, nhat1.y/normnhat1, nhat1.z/normnhat1);
    nhat2 = crossproduct(nhat0, nhat1);

    // Rotate relative velocity vector
    threevector nperp_rot = make_threevector(
        nhat1.x * cf + nhat2.x * sf,
        nhat1.y * cf + nhat2.y * sf,
        nhat1.z * cf + nhat2.z * sf
    );

    // New relative velocity (half for each particle)
    threevector Vrel_f_half = make_threevector(
        (vrel/2.0) * (costheta * nhat0.x + sintheta * nperp_rot.x),
        (vrel/2.0) * (costheta * nhat0.y + sintheta * nperp_rot.y),
        (vrel/2.0) * (costheta * nhat0.z + sintheta * nperp_rot.z)
    );

    // Calculate center of mass velocity
    threevector Vcm = make_threevector((Vi.x + Vj.x)/2.0, (Vi.y + Vj.y)/2.0, (Vi.z + Vj.z)/2.0);

    // Calculate final velocities
    threevector Vi_final = make_threevector(
        Vcm.x + Vrel_f_half.x,
        Vcm.y + Vrel_f_half.y,
        Vcm.z + Vrel_f_half.z
    );
    threevector Vj_final = make_threevector(
        Vcm.x - Vrel_f_half.x,
        Vcm.y - Vrel_f_half.y,
        Vcm.z - Vrel_f_half.z
    );

    // Update particle i
    particles[1][i] = Vi_final.z;  // Vz
    double Viperp_new = sqrt(Vi_final.x * Vi_final.x + Vi_final.y * Vi_final.y);
    particles[2][i] = particles[0][i] * Viperp_new;  // r*Vperp
    if (Viperp_new > 1e-15) {
        particles[5][i] = Vi_final.x / Viperp_new;  // cos(phi)
        particles[6][i] = Vi_final.y / Viperp_new;  // sin(phi)
    } else {
        particles[5][i] = 1.0;
        particles[6][i] = 0.0;
    }

    // Update particle j
    particles[1][j] = Vj_final.z;  // Vz
    double Vjperp_new = sqrt(Vj_final.x * Vj_final.x + Vj_final.y * Vj_final.y);
    particles[2][j] = particles[0][j] * Vjperp_new;  // r*Vperp
    if (Vjperp_new > 1e-15) {
        particles[5][j] = Vj_final.x / Vjperp_new;  // cos(phi)
        particles[6][j] = Vj_final.y / Vjperp_new;  // sin(phi)
    } else {
        particles[5][j] = 1.0;
        particles[6][j] = 0.0;
    }
}

/**
 * @brief Performs SIDM scattering calculations for one timestep using graph coloring parallel algorithm.
 * @details Implements a parallel SIDM scattering algorithm using graph coloring to eliminate race conditions.
 *          Particles are divided into 11 color groups where particles of the same color are separated
 *          by more than the interaction range (>10 positions), ensuring no conflicts. Each color is
 *          processed sequentially, but particles within a color are processed in parallel with direct
 *          state updates and no synchronization overhead.
 *
 *          **Azimuthal Angle Treatment:**
 *          Uses persistent azimuthal angle storage (cos(φ), sin(φ)) in `particles[5]` and `particles[6]`.
 *          These values are initialized randomly at simulation start and updated only during scattering
 *          events. Between scatters, φ remains constant (no orbital precession tracking). This
 *          approximation is physically valid because in high-scatter regimes φ evolves minimally
 *          between events, while in low-scatter regimes stochastic partner selection and orbital
 *          dynamics erase correlations from prior scatters. This approach maintains accuracy while
 *          avoiding costly per-timestep random number generation.
 *
 *          **Graph Coloring Algorithm:**
 *          1. Divides particles into 11 colors (modulo arithmetic: color = i % 11).
 *          2. Processes each color sequentially to prevent neighbor conflicts.
 *          3. Within each color, processes particles in parallel with per-thread RNGs.
 *          4. For each particle:
 *             a. Loads persistent φ orientation from stored cos(φ), sin(φ).
 *             b. Evaluates scattering probability with neighbors using 3D relative velocities.
 *             c. If scatter occurs: performs isotropic scattering, updates velocities and φ storage.
 *             d. Updates `g_particle_scatter_state` for AB3 integrator compatibility.
 *
 *          This design achieves high parallel efficiency without race conditions or atomic operations.
 *
 * @param particles             [in,out] Main particle data array: `particles[component][current_sorted_index]`.
 *                              Modified in-place with post-scattering velocities/angular momenta.
 * @param npts                  [in] Total number of particles.
 * @param dt                    [in] Simulation timestep (Myr), used in probability calculation.
 * @param current_time          [in] Current simulation time (Myr). Marked `unused` but available for future use.
 * @param rng_per_thread_list   [in] Array of GSL RNG states, one for each OpenMP thread.
 * @param num_threads_for_rng   [in] The number of allocated RNGs in `rng_per_thread_list` (equals max threads).
 * @param Nscatter_total_step   [out] Pointer to a long long to accumulate the total number of scatter events
 *                              that occurred in this timestep.
 * @param halo_mass_for_sidm    [in] Total halo mass (\f$M_{\odot}\f$) for the active profile, passed to `sigmatotal`.
 * @param rc_for_sidm           [in] Scale radius (kpc) for the active profile, passed to `sigmatotal`.
 * @param current_scatter_counts [in,out] Per-particle scatter count array for tracking (can be NULL).
 */
// Structure for reorganized particle data
void perform_sidm_scattering_parallel_graphcolor(
    double **particles,
    int npts,
    double dt,
    double current_time,
    gsl_rng **rng_per_thread_list,
    int num_threads_for_rng,
    long long *Nscatter_total_step,
    double halo_mass_for_sidm,
    double rc_for_sidm,
    int *current_scatter_counts
) {
    (void)current_time; // Unused parameter
    long long total_scatters = 0;
    extern int g_sidm_max_interaction_range;
    const int NUM_COLORS = g_sidm_max_interaction_range + 1;  // Maximum interaction distance + 1

    // Process each color sequentially
    for (int color = 0; color < NUM_COLORS; color++) {
        long long color_scatters = 0;

        // Process all particles of this color in parallel
        #pragma omp parallel reduction(+:color_scatters)
        {
            // Get thread-local RNG
            gsl_rng *local_rng = NULL;
            int thread_id = 0;
            #ifdef _OPENMP
                thread_id = omp_get_thread_num();
            #endif

            if (rng_per_thread_list != NULL && thread_id < num_threads_for_rng &&
                rng_per_thread_list[thread_id] != NULL) {
                local_rng = rng_per_thread_list[thread_id];
            } else {
                // Thread has no RNG - skip SIDM for this thread
                #pragma omp critical (rng_error_graphcolor)
                {
                    fprintf(stderr, "WARNING: Thread %d has no valid RNG in graph coloring SIDM\n", thread_id);
                }
                local_rng = NULL;
            }

            // Process particles with current color
            #pragma omp for schedule(guided)
            for (int i = color; i < npts; i += NUM_COLORS) {
                // Skip if no RNG available
                if (local_rng == NULL) continue;

                // Skip if too close to end
                if (i >= npts - 1) continue;

                // Determine how many neighbors to check
                int nscat = g_sidm_max_interaction_range;
                if (npts - 1 - i < nscat) nscat = npts - 1 - i;
                if (nscat <= 0) continue;

                // Arrays for probability calculations (VLA for flexibility)
                double partialprobability[nscat + 1];
                double probability_sum = 0.0;

                // Get particle i's current state
                double cos_phi_i = particles[5][i];
                double sin_phi_i = particles[6][i];
                double Viperp = particles[2][i] / particles[0][i];
                threevector Vi = make_threevector(Viperp * cos_phi_i, Viperp * sin_phi_i, particles[1][i]);

                // Check all possible scatter partners
                for (int m = 1; m <= nscat; m++) {
                    int j = i + m;

                    // Get neighbor's state
                    double cos_phi_j = particles[5][j];
                    double sin_phi_j = particles[6][j];
                    double Vjperp = particles[2][j] / particles[0][j];
                    threevector Vj = make_threevector(Vjperp * cos_phi_j, Vjperp * sin_phi_j, particles[1][j]);

                    // Calculate relative velocity
                    threevector Vrel = make_threevector(Vi.x - Vj.x, Vi.y - Vj.y, Vi.z - Vj.z);
                    double vrel = sqrt(dotproduct(Vrel, Vrel));

                    // Calculate partial probability (cross section * relative velocity)
                    partialprobability[m] = sigmatotal(vrel, npts, halo_mass_for_sidm, rc_for_sidm) * vrel;
                    probability_sum += partialprobability[m];
                }

                // Calculate shell volume for probability normalization
                int outer_shell_idx = (i + nscat + 1 < npts) ? i + nscat + 1 : i + nscat;
                double radius_diff = 0.0;
                if (outer_shell_idx < npts && outer_shell_idx > i) {
                    radius_diff = particles[0][outer_shell_idx] - particles[0][i];
                }

                // Calculate total scatter probability
                double probability = 0.0;
                if (radius_diff > 1e-15 && particles[0][i] > 1e-15 && probability_sum > 1e-30) {
                    // Scattering rate using shell volume approximation
                    double rate_dt = probability_sum * 0.5 * dt / (4.0 * PI * sqr(particles[0][i]) * radius_diff);
                    // Proper Poisson process: P = 1 - exp(-rate*dt)
                    probability = 1.0 - exp(-rate_dt);
                }

                // Decide if scattering occurs
                if (gsl_rng_uniform(local_rng) < probability) {
                    // Select which neighbor to scatter with
                    int m_scatter = 1;
                    if (nscat > 1 && probability_sum > 1e-15) {
                        // Build cumulative probability distribution
                        double cumulative_prob[nscat + 1];
                        cumulative_prob[0] = 0.0;
                        for (int k = 1; k <= nscat; k++) {
                            cumulative_prob[k] = cumulative_prob[k-1] + partialprobability[k] / probability_sum;
                        }
                        cumulative_prob[nscat] = 1.0;  // Ensure normalization

                        // Select partner based on weighted probability
                        double random_select = gsl_rng_uniform(local_rng);
                        while (m_scatter < nscat && random_select > cumulative_prob[m_scatter]) {
                            m_scatter++;
                        }
                    }

                    int j = i + m_scatter;
                    if (j >= npts) continue;  // Safety check

                    // Perform the scatter update using the existing function
                    perform_scatter_update(particles, i, j, local_rng);

                    // Update scatter counts
                    color_scatters++;

                    // Update per-particle scatter counts if tracking
                    if (current_scatter_counts != NULL) {
                        // These increments are safe - no other thread touches these particles
                        // during this color's processing
                        current_scatter_counts[i]++;
                        current_scatter_counts[j]++;
                    }
                }
            }
        } // End parallel region for this color

        total_scatters += color_scatters;

        // Optional: Progress reporting for debugging (controlled by environment variable)
        if (getenv("SIDM_DEBUG_GRAPHCOLOR") != NULL) {
            printf("  Color %2d: %lld scatters\n", color, color_scatters);
        }
    } // End color loop

    *Nscatter_total_step = total_scatters;
}

// =============================================================================
// HERNQUIST PROFILE ANALYTICAL FUNCTIONS
// =============================================================================

/**
 * @brief Analytical density for the Hernquist profile.
 * @param r Radius (kpc).
 * @param M Total halo mass (\f$M_{\odot}\f$).
 * @param a Scale radius \f$a\f$ of the Hernquist profile (kpc).
 * @return Physical density (\f$M_{\odot}\f$ / kpc\f$^3\f$).
 */
static inline double density_hernquist(double r, double M, double a) {
    if (r < 1e-9) r = 1e-9; // Avoid singularity at r=0 for numerical stability
    return (M * a) / (2.0 * PI * r * pow(r + a, 3.0));
}

/**
 * @brief Analytical potential for the Hernquist profile.
 * @param r Radius (kpc).
 * @param M Total halo mass (\f$M_{\odot}\f$).
 * @param a Scale radius \f$a\f$ of the Hernquist profile (kpc).
 * @return Gravitational potential in code units of (kpc/Myr)\f$^2\f$.
 */
static inline double potential_hernquist(double r, double M, double a) {
    // Returns the POSITIVE relative potential Psi = -Phi
    return (G_CONST * M) / (r + a);
}

/**
 * @brief Anisotropic distribution function for Hernquist profile with arbitrary beta.
 * @param E_bind Binding energy, E_bind = -E_total (must be positive). Units are (kpc/Myr)\f$^2\f$.
 * @param L Angular momentum magnitude (kpc\f$^2\f$/Myr).
 * @param M Total halo mass (\f$M_{\odot}\f$).
 * @param a Scale radius \f$a\f$ of the Hernquist profile (kpc).
 * @return Value of the distribution function f(E, L).
 */
static inline double df_hernquist_aniso(double E_bind, double L, double M, double a) {
    if (E_bind <= 0 || L <= 1e-15) {
        return 0.0;
    }
    
    double beta = g_anisotropy_beta;
    
    // Distribution function shape determines relative probabilities for rejection sampling.
    // Absolute normalization factors cancel in the acceptance ratio calculation.
    
    // Normalize energy to dimensionless units for hypergeometric evaluation
    double GM = G_CONST * M;
    double E_normalized = E_bind / (GM / a);  // Normalize to dimensionless units
    
    // Check for hypergeometric convergence (typically need |E| < 1)
    if (fabs(E_normalized) >= 1.0) {
        return 0.0;  // Hypergeometric function may not converge
    }
    
    // Gamma function singularities do not affect rejection sampling 
    // as only relative probability ratios matter.
    
    // Calculate the power terms
    double power_term = pow(E_bind, 2.5 - beta) / pow(L, 2.0*beta);
    
    // Calculate hypergeometric function using safe wrapper
    double hyperg = hyperg_2F1_safe(beta, E_normalized);

    if (hyperg == 0.0) {
        return 0.0;  // Invalid result from hypergeometric
    }

    // Return unnormalized distribution (shape only)
    return power_term * hyperg;
}
