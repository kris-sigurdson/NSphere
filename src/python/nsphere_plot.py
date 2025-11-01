#!/usr/bin/env python3
# Copyright 2025 Kris Sigurdson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Check for --help flag at the very beginning
import sys
showing_help = '--help' in sys.argv or '-h' in sys.argv

# Import other modules
import numpy as np
import matplotlib
matplotlib.use('Agg', force=True)
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import energy_distance
from scipy.interpolate import UnivariateSpline
import math
import os
import glob
import re
import gc
import psutil
import struct
import imageio.v2 as imageio # Use imageio v2 for get_writer
from io import BytesIO
import multiprocessing as mp
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
from scipy.stats import binned_statistic
import argparse
import signal
import contextlib
import time
import atexit
from tqdm import tqdm
import logging
import shutil
import traceback
import threading

def find_project_root():
    """
    Determines the project root directory.

    Returns
    -------
    str
        Path to the project root directory.

    Notes
    -----
    The project root is identified by the presence of a 'src' directory.
    Uses path normalization for cross-platform compatibility.
    """
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    
    if os.path.basename(script_dir) == 'python':
        potential_root = os.path.dirname(os.path.dirname(script_dir))
    else:
        potential_root = script_dir
    
    potential_root = os.path.normpath(potential_root)
    
    if os.path.isdir(os.path.join(potential_root, 'src')):
        return potential_root
    
    parent_dir = os.path.dirname(potential_root)
    if os.path.isdir(os.path.join(parent_dir, 'src')):
        return parent_dir
        
    grandparent_dir = os.path.dirname(parent_dir)
    if os.path.isdir(os.path.join(grandparent_dir, 'src')):
        return grandparent_dir
    
    print(f"Warning: Could not verify project root via 'src' directory. Using derived directory: {potential_root}", file=sys.stderr)
    return potential_root

# Set up project paths
PROJECT_ROOT = find_project_root()
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

# Add the project src directory to the module search path if needed
src_dir = os.path.join(PROJECT_ROOT, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Define global variables
suffix = ""  # Will be set in main()
start_snap = 0
end_snap = 0
step_snap = 1
duration = 100.0  # Default frame duration in ms (for fps=10)
section_delay = 0.0  # No delay by default (will be set to 5.0 if paced_mode is enabled)
progress_delay = 0.0  # No delay by default (will be set to 2.0 if paced_mode is enabled)
enable_logging = False  # Default: don't log info messages unless --log is specified
paced_mode = False  # When true, enables delays between sections with timers

def show_section_delay(delay_seconds):
    """
    Display a visual timer with dots for section transitions.
    
    Parameters
    ----------
    delay_seconds : float
        The number of seconds to delay/pace
    
    Notes
    -----
    Shows an upward counting timer with dots accumulating each second.
    Displays the full delay_seconds at the end to ensure consistent timing.
    Allows the user to skip the delay with spacebar or pause/resume with 'p'.
    """
    try:
        import msvcrt  # Windows
        def kbhit():
            return msvcrt.kbhit()
        def getch():
            return msvcrt.getch().decode('utf-8').lower()
        is_windows = True
    except ImportError:
        try:
            import termios, fcntl, os, select  # Unix/Linux/MacOS
            is_windows = False
            # Save the terminal settings
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            # Setup non-blocking input
            def setup_nonblocking():
                new = termios.tcgetattr(fd)
                new[3] = new[3] & ~termios.ICANON
                new[3] = new[3] & ~termios.ECHO
                termios.tcsetattr(fd, termios.TCSANOW, new)
                oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
                fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
            def restore_terminal():
                termios.tcsetattr(fd, termios.TCSAFLUSH, old_settings)
            def kbhit():
                dr, dw, de = select.select([sys.stdin], [], [], 0)
                return len(dr) > 0
            def getch():
                if kbhit():
                    ch = sys.stdin.read(1)
                    return ch.lower()
                return None
            setup_nonblocking()
        except ImportError:
            # If neither input method is available, fallback to basic behavior
            def kbhit():
                return False
            def getch():
                return None
            def restore_terminal():
                pass
            is_windows = False
    
    start_time = time.time()
    dots = ""
    paused = False
    pause_start_time = 0
    pause_elapsed_time = 0  # Store the elapsed time when entering pause
    total_paused_time = 0
    
    # Helper to format display string
    def get_display_text(elapsed_time, dots_str, pause_status):
        # Fix the position of the message by padding the dots to a consistent length
        max_dots = int(delay_seconds) + 1  # Maximum possible dots
        dots_padding = " " * (max_dots - len(dots_str))
        
        # Format base display string - time goes right after "for"
        base_display = f"\rPacing output for {elapsed_time:.2f}s "
        
        # Calculate message length and ensure padding is sufficient
        pause_msg = "Press Spacebar to Skip, P to Resume"
        run_msg = "Press Spacebar to Skip, P to Pause"
        
        # Ensure extra padding is present for the longer message
        max_msg_length = max(len(pause_msg), len(run_msg)) + 4  # +4 for brackets and buffer
        
        # Define spacing constants
        status_text = "[PAUSED] "
        status_padding = " " * len(status_text)
        
        # Add appropriate status or padding after the time
        if pause_status:
            # When paused, show [PAUSED] after the time
            display_with_status = base_display + status_text
            msg = pause_msg
        else:
            # When not paused, add padding to keep elements position consistent
            display_with_status = base_display + status_padding
            msg = run_msg  # No extra space here - it will be added after the bracket
        
        # Place message 14 spaces to the right and ensure it stays in place
        guidance = "              [" + msg + "] "  # Added a space after the closing bracket for padding
        
        # Complete the display string with dots and guidance
        return f"{display_with_status}{dots_str}{dots_padding}{guidance}"
    
    # Initial display with empty dots and consistent padding
    initial_dots = "."  # Start with one dot
    sys.stdout.write("\n")
    # Always start in the unpaused state for the initial display
    sys.stdout.write(get_display_text(0.0, initial_dots, False))
    sys.stdout.flush()
    
    try:
        # Keep running until the delay duration is reached or exceeded
        while True:
            if kbhit():
                key = getch()
                if key == ' ':  # Spacebar always skips entirely, even when paused
                    # Clear the entire line and return immediately
                    sys.stdout.write("\r\033[2K")
                    sys.stdout.flush()
                    return
                elif key == 'p':  # 'p' toggles pause/resume
                    if paused:
                        # Resuming - add elapsed pause time to total
                        total_paused_time += time.time() - pause_start_time
                        paused = False
                    else:
                        # Pausing - record the time pause began and save current elapsed time
                        pause_start_time = time.time()
                        # Calculate and store the current elapsed time to display during pause
                        pause_elapsed_time = time.time() - start_time - total_paused_time
                        paused = True
            
            if not paused:
                # Only update time when not paused
                current_time = time.time()
                elapsed = current_time - start_time - total_paused_time
                
                if elapsed >= delay_seconds:
                    # The end of the delay has been reached
                    # Instead of showing a final display, clear the line completely (like spacebar skip)
                    sys.stdout.write("\r\033[2K")
                    sys.stdout.flush()
                    break
                
                # Add a dot every second
                new_dots = "." * (int(elapsed) + 1)
                if new_dots != dots:
                    dots = new_dots
                
                # Update display
                display_content = get_display_text(elapsed, dots, paused)
                sys.stdout.write("\r\033[2K" + truncate_and_pad_string(display_content))
                sys.stdout.flush()
            else:
                # When paused, display the stored time and dots from when we entered pause
                # Use the stored pause_elapsed_time which is frozen at the moment of pause
                display_content_paused = get_display_text(pause_elapsed_time, dots, paused)
                sys.stdout.write("\r\033[2K" + truncate_and_pad_string(display_content_paused))
                sys.stdout.flush()
            
            time.sleep(0.1)
    
        # No newline after completing - line is already cleared
    finally:
        # Ensure terminal is restored if using Unix-style input
        if not is_windows and 'restore_terminal' in locals():
            restore_terminal()

def show_progress_delay(delay_seconds):
    """
    Display a minimal visual indicator during progress delay.
    
    Parameters
    ----------
    delay_seconds : float
        The number of seconds to delay/pace
    
    Notes
    -----
    Shows only dots accumulating with each second, without text.
    """
    start_time = time.time()
    dots = ""
    
    while (time.time() - start_time) < delay_seconds:
        elapsed = time.time() - start_time
        # Add a dot every second
        new_dots = "." * (int(elapsed) + 1)
        if new_dots != dots:
            dots = new_dots
        # Truncate the dots string (unlikely to be needed but consistent)
        sys.stdout.write(f"\r\033[2K{truncate_and_pad_string(dots)}")
        sys.stdout.flush()
        time.sleep(0.1)
    
    sys.stdout.write("\r\033[2K")  # Clear the dots
    sys.stdout.flush()

# Global variables for animation data
mass_snapshots = []
density_snapshots = []
psi_snapshots = []

# Global variables for histogram data tracking
particles_original_count = 0
particles_final_original_count = 0

# Configure tqdm to work properly in all environments
tqdm.monitor_interval = 0  # Disable monitor thread to avoid issues

# --- TQDM Dynamic Formatting Constants ---

# Format Strings (No Bar, No Percentage for counter_tqdm line)
FMT_FULL = '{desc}: {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
FMT_NO_REM = '{desc}: {n_fmt}/{total_fmt} [{elapsed}, {rate_fmt}]'
FMT_NO_RATE = '{desc}: {n_fmt}/{total_fmt} [{elapsed}]'
FMT_COUNTS = '{desc}: {n_fmt}/{total_fmt}'
FMT_DESC = '{desc}'

# Descriptions (Full and Short)
DESCRIPTIONS = {
    'proc_anim_frames': {
        'full': "Processing animation frames", 'short': "Proc. anim frames"},
    'encod_frames': {
        'full': "Encoding frames", 'short': "Encod. frames"},
    'preproc_phase': {
        'full': "Preprocessing phase space data", 'short': "Preproc. phase data"},
    'render_phase': {
        'full': "Rendering phase space frames", 'short': "Render phase frames"},
    'proc_sorted_snaps': {
        'full': "Processing rank sorted snapshot files", 'short': "Proc. sorted snaps"},
    'proc_unsorted_snaps': {
        'full': "Processing unsorted snapshot files", 'short': "Proc. unsorted snaps"},
    'proc_energy_series': {
        'full': "Processing energy data for time series", 'short': "Proc. energy series"},
    'gen_mass_frames': {
        'full': "Generating mass profile frames", 'short': "Gen. mass frames"},
    'gen_dens_frames': {
        'full': "Generating density profile frames", 'short': "Gen. density frames"},
    'gen_psi_frames': {
        'full': "Generating psi profile frames", 'short': "Gen. psi frames"},
}

# Pre-calculated Thresholds (Min width needed for format with specific desc length)
# Based on: Counts=11, Elapsed=9, Remaining=9, Rate=14, Separators
TQDM_THRESHOLDS = {
    'proc_anim_frames': { # Len 28/18
        'full': {'full': 80, 'no_rem': 70, 'no_rate': 52, 'counts': 41, 'desc': 28},
        'short': {'full': 70, 'no_rem': 60, 'no_rate': 42, 'counts': 31, 'desc': 18}},
    'encod_frames': { # Len 15/13
        'full': {'full': 67, 'no_rem': 57, 'no_rate': 39, 'counts': 28, 'desc': 15},
        'short': {'full': 65, 'no_rem': 55, 'no_rate': 37, 'counts': 26, 'desc': 13}},
    'preproc_phase': { # Len 30/20
        'full': {'full': 82, 'no_rem': 72, 'no_rate': 54, 'counts': 43, 'desc': 30},
        'short': {'full': 72, 'no_rem': 62, 'no_rate': 44, 'counts': 33, 'desc': 20}},
    'render_phase': { # Len 28/19
        'full': {'full': 80, 'no_rem': 70, 'no_rate': 52, 'counts': 41, 'desc': 28},
        'short': {'full': 71, 'no_rem': 61, 'no_rate': 43, 'counts': 32, 'desc': 19}},
    'proc_sorted_snaps': { # Len 36/18
        'full': {'full': 88, 'no_rem': 78, 'no_rate': 60, 'counts': 49, 'desc': 36},
        'short': {'full': 70, 'no_rem': 60, 'no_rate': 42, 'counts': 31, 'desc': 18}},
    'proc_unsorted_snaps': { # Len 34/20
        'full': {'full': 86, 'no_rem': 76, 'no_rate': 58, 'counts': 47, 'desc': 34},
        'short': {'full': 72, 'no_rem': 62, 'no_rate': 44, 'counts': 33, 'desc': 20}},
    'proc_energy_series': { # Len 38/20
        'full': {'full': 90, 'no_rem': 80, 'no_rate': 62, 'counts': 51, 'desc': 38},
        'short': {'full': 72, 'no_rem': 62, 'no_rate': 44, 'counts': 33, 'desc': 20}},
    'gen_mass_frames': { # Len 30/16
        'full': {'full': 82, 'no_rem': 72, 'no_rate': 54, 'counts': 43, 'desc': 30},
        'short': {'full': 68, 'no_rem': 58, 'no_rate': 40, 'counts': 29, 'desc': 16}},
    'gen_dens_frames': { # Len 32/19
        'full': {'full': 84, 'no_rem': 74, 'no_rate': 56, 'counts': 45, 'desc': 32},
        'short': {'full': 71, 'no_rem': 61, 'no_rate': 43, 'counts': 32, 'desc': 19}},
    'gen_psi_frames': { # Len 29/15
        'full': {'full': 81, 'no_rem': 71, 'no_rate': 53, 'counts': 42, 'desc': 29},
        'short': {'full': 67, 'no_rem': 57, 'no_rate': 39, 'counts': 28, 'desc': 15}},
}

# Helper function to select format and description
def select_tqdm_format(desc_key, term_width):
    """Selects tqdm description and format based on terminal width."""
    if desc_key not in TQDM_THRESHOLDS or desc_key not in DESCRIPTIONS:
        # Fallback if key is unknown
        return desc_key, FMT_NO_RATE # Default to a reasonable format

    thresholds = TQDM_THRESHOLDS[desc_key]
    full_desc = DESCRIPTIONS[desc_key]['full']
    short_desc = DESCRIPTIONS[desc_key]['short']

    # Determine which description and format to use
    if term_width >= thresholds['full']['full']:
        return full_desc, FMT_FULL
    elif term_width >= thresholds['full']['no_rem']:
        return full_desc, FMT_NO_REM
    elif term_width >= thresholds['full']['no_rate']:
        return full_desc, FMT_NO_RATE
    # Switch to short description if full doesn't fit even minimal stats
    elif term_width >= thresholds['short']['no_rate']:
        return short_desc, FMT_NO_RATE
    elif term_width >= thresholds['short']['counts']:
        return short_desc, FMT_COUNTS
    else:
        return short_desc, FMT_DESC
# --- End TQDM Dynamic Formatting Constants ---

# Configure tqdm to match the custom progress bar format
tqdm_kwargs = {
    'ascii': False,              # Use Unicode characters for progress bars
    'position': 0,               # Always at position 0
    'leave': True,               # Leave the progress bar after completion
    'miniters': 5,               # Update every 5 iterations to reduce flickering
    'dynamic_ncols': True,       # Enable dynamic width adaptation
    'ncols': None,               # No fixed width override
    'smoothing': 0.3,            # Smoother progress updates
    # 'bar_format': MUST BE ABSENT or None
}

# Handle keyboard interrupts gracefully
original_sigint_handler = signal.getsignal(signal.SIGINT)

def signal_handler(sig, frame):
    """
    Handle keyboard interrupts gracefully.

    Parameters
    ----------
    sig : int
        Signal number
    frame : frame
        Current stack frame

    Notes
    -----
    Restores the original signal handler to allow a forced exit
    with a second Ctrl+C if needed.
    """
    print("\nInterrupted by user. Cleaning up...")
    # Restore original handler to allow a second Ctrl+C to force exit
    signal.signal(signal.SIGINT, original_sigint_handler)
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

# Context manager to suppress stdout during progress bar updates
@contextlib.contextmanager
def suppress_stdout():
    """
    Context manager that suppresses stdout output.

    Temporarily redirects stdout to a dummy file object that discards output.
    Used to prevent unwanted console output during progress bar updates.

    Yields
    ------
    None
        Control returns to the caller with stdout suppressed

    Notes
    -----
    The original stdout is always restored when leaving the context,
    even if an exception occurs.

    Examples
    --------
    >>> with suppress_stdout():
    ...     print("This won't be displayed")
    """
    original_stdout = sys.stdout
    class DummyFile:
        def write(self, x): pass
        def flush(self): pass
    sys.stdout = DummyFile()
    try:
        yield
    finally:
        sys.stdout = original_stdout

# Helper function for processing animation frames (must be at module level for multiprocessing)
def process_animation_frame(frame):
    """
    Process a single animation frame for GIF output.

    Parameters
    ----------
    frame : numpy.ndarray
        The frame image data

    Returns
    -------
    numpy.ndarray
        Processed frame ready for GIF encoding
    """
    # Create BytesIO buffer for this frame
    buf = BytesIO()

    # Write frame to BytesIO buffer with explicit format (needed in v3)
    imageio.imwrite(buf, frame, extension='.png', plugin='pillow')

    # Reset buffer position to start
    buf.seek(0)

    # Read the frame back as an image
    frame_img = imageio.imread(buf, index=0)

    # Clean up buffer
    buf.close()

    return frame_img

# ANSI escape codes for cursor manipulation
HIDE_CURSOR = "\033[?25l"  # Hide the cursor
SHOW_CURSOR = "\033[?25h"  # Show the cursor

def truncate_and_pad_string(text, fallback_width=100):
    """
    Truncates a string to the terminal width and pads with spaces.

    Parameters
    ----------
    text : str
        The string to potentially truncate and pad.
    fallback_width : int, optional
        The width to use if terminal size can't be determined, by default 100.

    Returns
    -------
    str
        The string, truncated and padded with spaces to the terminal width.
    """
    try:
        # Attempt to get the terminal size
        columns = shutil.get_terminal_size().columns
        width = columns
    except OSError:
        # If output is redirected or terminal size unavailable, use fallback
        width = fallback_width
    # Truncate the string first
    truncated = text[:width]
    # Pad the truncated string with spaces to the full width
    return truncated.ljust(width)

def get_separator_line(char='-', fallback_width=100):
    """
    Generates a separator line spanning the terminal width.

    Parameters
    ----------
    char : str, optional
        The character to repeat for the line, by default '-'.
    fallback_width : int, optional
        The width to use if terminal size can't be determined, by default 100.

    Returns
    -------
    str
        A string consisting of 'char' repeated to fill the terminal width.
    """
    try:
        # Attempt to get the terminal size
        columns = shutil.get_terminal_size().columns
        # Use the determined width, but ensure it's at least fallback_width
        # to avoid overly short separators on very narrow terminals.
        width = max(columns, fallback_width // 2) # Ensure a minimum reasonable width
    except OSError:
        # If output is redirected or terminal size unavailable, use fallback
        width = fallback_width
    return char * width

# Function to clear the current line before printing
def clear_line():
    """
    Clear the current terminal line for clean console output using ANSI escape codes.

    Notes
    -----
    Uses carriage return and the ANSI sequence `\033[2K` to erase the entire line.
    This is more reliable than printing spaces across different terminal widths.
    """
    sys.stdout.write("\r\033[2K") # Move to beginning, erase entire line
    sys.stdout.flush()

# Hide/show cursor functions for progress displays
def hide_cursor():
    """
    Hide the terminal cursor.

    Notes
    -----
    Uses ANSI escape code to hide the cursor for cleaner progress bar display.
    """
    sys.stdout.write(HIDE_CURSOR)
    sys.stdout.flush()

def show_cursor():
    """
    Show the terminal cursor.

    Notes
    -----
    Uses ANSI escape code to restore cursor visibility after it was hidden.
    """
    sys.stdout.write(SHOW_CURSOR)
    sys.stdout.flush()

# Safety mechanism to restore cursor if process is terminated unexpectedly
def ensure_cursor_visible():
    """
    Restore cursor visibility when program exits.

    Notes
    -----
    Safety mechanism registered with atexit to ensure the terminal cursor
    remains visible even if the program terminates unexpectedly.
    """
    show_cursor()

# Register the safety function to run on exit
atexit.register(ensure_cursor_visible)

# Set up the logger
def setup_logging():
    """
    Configure and initialize the logging system.

    Returns
    -------
    logging.Logger
        Configured logger instance for nsphere_plot.py

    Notes
    -----
    Creates a log directory if it doesn't exist and configures
    a file-based logger that writes to log/nsphere_plot.log with
    timestamp, log level, and message formatting.
    """
    # Create log directory if it doesn't exist
    os.makedirs("log", exist_ok=True)

    # Configure the logger
    log_file = "log/nsphere_plot.log"
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger('nsphere_plot')

def log_message(message, level="info"):
    """
    Log a message to the log file based on level and settings.

    Parameters
    ----------
    message : str
        The message to log
    level : str, optional
        The log level (info, warning, error, debug), by default "info"

    Notes
    -----
    Warning and error messages are always logged regardless of enable_logging flag.
    Info and debug messages are only logged if enable_logging is True (--log specified).
    """
    global enable_logging

    # Always log warnings and errors regardless of enable_logging setting
    if level.lower() == "warning":
        logger.warning(message)
        return
    elif level.lower() == "error":
        logger.error(message)
        return

    # For info and debug, only log if enable_logging is True
    if not enable_logging:
        return

    if level.lower() == "info":
        logger.info(message)
    elif level.lower() == "debug":
        logger.debug(message)
    else:
        # Default to info level
        logger.info(message)

# Initialize the logger
logger = setup_logging()

# Log plot saving to file, display on console cycling through filenames
def log_plot_saved(output_file, current=0, total=1, section_type="plots"):
    """
    Log a saved plot to log file and update status on console.

    Parameters
    ----------
    output_file : str
        Full path to the saved file
    current : int, optional
        Current plot number, by default 0
    total : int, optional
        Total plots to save, by default 1
    section_type : str, optional
        Type of plots being saved, by default "plots"

    Notes
    -----
    Displays a progress bar and updates the console with the current plot
    being saved. Includes timing information once enough plots have been
    processed to calculate a reasonable rate.
    """
    # Log the full path to the log file
    logger.info(f"Plot saved: {output_file}")

    # First plot initialization is handled globally
    if current == 1:
        pass

    # Show shorter message on console with cycling display
    prefix = get_file_prefix(output_file)
    clear_line()

    # Create progress display
    progress = (current / total) * 100 if total > 0 else 100
    bar_length = 20
    filled_length = int(bar_length * current // total)
    bar = '█' * filled_length + ' ' * (bar_length - filled_length)

    # Calculate time estimates for multiple plots
    time_info = ""
    if hasattr(log_plot_saved, 'start_time') and total > 1:
        elapsed = time.time() - log_plot_saved.start_time
        # For display purposes, ensure elapsed time is at least 0.01 seconds
        displayed_elapsed = max(elapsed, 0.01)  # Minimum displayed time
        # Apply a minimum nominal time to prevent unrealistically high rates
        nominal_elapsed = max(elapsed, 0.02)  # Minimum 0.02 seconds

        if current > 1:  # Need at least 2 plots to calculate rate
            rate = current / nominal_elapsed  # plots per second
            # Cap the rate at a reasonable maximum for display
            rate = min(rate, 99.9)

            # Only show remaining time once there's a reasonable rate calculation
            if current >= max(3, total * 0.1):  # At least 3 plots or 10% complete
                remaining = (total - current) / rate if rate > 0 else 0
                time_info = f" [{displayed_elapsed:.2f}<{remaining:.2f}, {rate:.1f}file/s]"
    else:
        # Initialize the start time on first call
        log_plot_saved.start_time = time.time()

    # Print the progress bar and current plot name inline with extra padding
    # Truncate the content part of the string
    content_string = f"Save: {bar} {progress:.1f}% | File: {prefix}{time_info}"
    sys.stdout.write(f"\r{truncate_and_pad_string(content_string)}")
    sys.stdout.flush()

    # Print newline if this is the last plot
    if current == total:
        sys.stdout.write("\n")
        # Add a separator line after a completed progress bar
        sys.stdout.write(get_separator_line(char='-') + "\n")
        sys.stdout.flush()

        # Add delay after completing progress section if enabled
        if progress_delay > 0 and paced_mode:
            show_progress_delay(progress_delay)

        # Reset start time for next batch
        if hasattr(log_plot_saved, 'start_time'):
            delattr(log_plot_saved, 'start_time')

# Global state for combined progress tracking
_combined_plot_trackers = {}

def start_combined_progress(section_key, total_plots):
    """
    Initialize a combined progress tracker for a section of related plots.

    Parameters
    ----------
    section_key : str
        Unique identifier for this plotting section
    total_plots : int
        Total number of plots expected in this section

    Notes
    -----
    Sets up a progress tracker in the global _combined_plot_trackers dictionary
    to monitor progress across multiple related operations. Initializes timing
    information for rate calculation.
    """
    _combined_plot_trackers[section_key] = {
        'current': 0,
        'total': total_plots,
        'plots': [],
        'start_time': time.time()  # Initialize timing information
    }
    logger.info(f"Starting combined progress tracking for {section_key} with {total_plots} total plots")

def update_combined_progress(section_key, output_file):
    """
    Update the combined progress tracker for a specific plot section.

    Parameters
    ----------
    section_key : str
        Identifier for the plotting section
    output_file : str
        Path to the plot file that was saved

    Returns
    -------
    tuple
        (current, total) counts for the section, indicating progress

    Notes
    -----
    Updates the progress tracker for the specified section, increments
    the counter, and displays a progress bar showing completion status.
    Includes timing information and estimated completion time once
    enough data points are available.
    """
    if section_key not in _combined_plot_trackers:
        logger.warning(f"No active progress tracker for {section_key}")
        return (0, 0)

    tracker = _combined_plot_trackers[section_key]
    tracker['current'] += 1
    tracker['plots'].append(output_file)

    logger.info(f"Progress update: {output_file}")

    prefix = get_file_prefix(output_file)
    clear_line()

    progress = (tracker['current'] / tracker['total']) * 100 if tracker['total'] > 0 else 100
    bar_length = 20
    filled_length = int(bar_length * tracker['current'] // tracker['total'])
    bar = '█' * filled_length + ' ' * (bar_length - filled_length)

    # Calculate time estimates
    time_info = ""
    if 'start_time' in tracker:
        elapsed = time.time() - tracker['start_time']
        # For display purposes, ensure elapsed time is at least 0.01 seconds
        displayed_elapsed = max(elapsed, 0.01)  # Minimum displayed time
        # Apply a minimum nominal time to prevent unrealistically high rates
        nominal_elapsed = max(elapsed, 0.02)  # Minimum 0.02 seconds

        # Always show timing information, even for single items
        # Calculate a reasonable rate based on progress so far
        rate = tracker['current'] / nominal_elapsed  # items per second

        # Cap the rate at a reasonable maximum for display
        rate = min(rate, 99.9)

        # Calculate remaining time based on current rate
        remaining = (tracker['total'] - tracker['current']) / rate if rate > 0 else 0

        # Always show timing information
        time_info = f" [{displayed_elapsed:.2f}<{remaining:.2f}, {rate:.1f}file/s]"
    else:
        # Initialize the start time on first update
        tracker['start_time'] = time.time()

    # Determine if this is a loading progress or saving progress
    action = "Loading data" if section_key.endswith("_data_loading") else "Saving plots"

    # Print the progress bar and current plot name inline with extra padding
    # For progress bars, use shorter Read/Save labels
    display_action = "Read:" if section_key.endswith("_data_loading") else "Save:"
    # Truncate the content part of the string
    content_string = f"{display_action} {bar} {progress:.1f}% | File: {prefix}{time_info}"
    sys.stdout.write(f"\r{truncate_and_pad_string(content_string)}")
    sys.stdout.flush()

    # Print newline if this is the last plot
    if tracker['current'] >= tracker['total']:
        sys.stdout.write("\n")
        # Add a separator line after a completed progress bar
        sys.stdout.write(get_separator_line(char='-') + "\n")
        sys.stdout.flush()

        # Add delay after completing progress section if enabled
        if progress_delay > 0 and paced_mode:
            show_progress_delay(progress_delay)

        # Clear the tracker as it's complete
        _combined_plot_trackers.pop(section_key, None)

    return (tracker['current'], tracker['total'])

# Custom print functions for consistent output formatting
def print_status(message):
    """
    Print a message to the console and also log it if detailed logging is enabled.

    Parameters
    ----------
    message : str
        The message to print and potentially log.
    """
    # Always print to console (if not showing help)
    if not showing_help:
        clear_line()
        # Truncate the message before writing
        sys.stdout.write(truncate_and_pad_string(message) + "\n")

    # Also log the message if detailed logging is enabled
    if enable_logging:
        logger.info(f"[CONSOLE] {message}")

def print_header(title, add_newline=True):
    """
    Print a consistent section header with configurable delay and log it if detailed logging is enabled.

    Parameters
    ----------
    title : str
        The section title to display.
    add_newline : bool, optional
        Whether to add a newline before the header. Default is True.
    """
    if not showing_help:
        # Add delay between sections for better visual separation, but only if this is not the first header
        global section_delay, enable_logging, paced_mode
        # Use a static variable to track if this is the first header
        if not hasattr(print_header, 'first_call'):
            print_header.first_call = True
        elif section_delay > 0 and paced_mode:
            show_section_delay(section_delay)

        clear_line()
        # Only add newline if requested (for energy plots we want to control this)
        if add_newline:
            sys.stdout.write("\n")
        sys.stdout.write(get_separator_line(char='=') + "\n")
        # Truncate the title before writing
        sys.stdout.write(truncate_and_pad_string(title) + "\n")
        sys.stdout.write(get_separator_line(char='=') + "\n")

        # Also log the header if detailed logging is enabled
        if enable_logging:
            logger.info(f"[SECTION] {title}")

def print_footer(message=""):
    """
    Print a footer for a section and log it if detailed logging is enabled.

    Parameters
    ----------
    message : str, optional
        The footer message to display. Default is an empty string.
    """
    if not showing_help and message:
        clear_line()
        # Truncate the message before writing
        sys.stdout.write(truncate_and_pad_string(f"{message}") + "\n")
        sys.stdout.flush()

        # Also log the footer if detailed logging is enabled
        if enable_logging:
            logger.info(f"[FOOTER] {message}")

def read_lastparams(filename="data/lastparams.dat", return_suffix=True, user_suffix=None):
    """
    Read parameters from lastparams.dat file.

    Parameters
    ----------
    filename : str, optional
        Base path to the lastparams.dat file.
    return_suffix : bool, optional
        If True, return a suffix string. If False, return individual parameters.
    user_suffix : str, optional
        If provided, look for lastparams<user_suffix>.dat instead of lastparams.dat.

    Returns
    -------
    str or tuple
        If return_suffix is True, returns a string like "_[filetag]_npts_Ntimes_tfinal_factor".
        If return_suffix is False, returns a tuple of (npts, Ntimes, tfinal_factor, file_tag).
    """
    global showing_help

    default_suffix = ""
    default_params = (30000, 1001, 5, "")  # Default values

    if showing_help:
        return default_suffix if return_suffix else default_params

    # Determine which lastparams file to read
    if user_suffix:
        # If user supplied a suffix, use lastparams<suffix>.dat
        filename = f"data/lastparams{user_suffix}.dat"
        logger.info(f"Attempting to read parameters from user-specified file: {filename}")
    else:
        # Otherwise, use the default lastparams.dat which points to the latest run
        filename = "data/lastparams.dat"
        logger.info(f"Reading parameters from default file: {filename}")

    if not os.path.exists(filename):
        print(truncate_and_pad_string(f"Error: Could not find parameter file: {filename}"))
        print(truncate_and_pad_string("Please run the simulation or specify a valid --suffix."))
        sys.exit(1)

    try:
        with open(filename, "r") as f:
            line = f.readline().strip()
        if not line:
            return default_suffix if return_suffix else default_params

        parts = line.split(maxsplit=3)
        if len(parts) < 3:
            return default_suffix if return_suffix else default_params

        npts, Ntimes, tfinal_factor = map(int, parts[:3])
        file_tag = parts[3] if len(parts) > 3 else ""

        logger.info(f"Read parameters: npts={npts}, Ntimes={Ntimes}, tfinal={tfinal_factor}, tag='{file_tag}'")

        if return_suffix:
            # Reconstruct the suffix from the read parameters
            if file_tag:
                return f"_{file_tag}_{npts}_{Ntimes}_{tfinal_factor}"
            else:
                return f"_{npts}_{Ntimes}_{tfinal_factor}"
        else:
            return (npts, Ntimes, tfinal_factor, file_tag)

    except Exception as e:
        print(truncate_and_pad_string(f"Warning: Error reading {filename}: {e}"))
        return default_suffix if return_suffix else default_params

# Default values for module-level constants
# These will be properly initialized in main() via Configuration
npts, Ntimes, tfinal_factor, file_tag = 30000, 1001, 5, ""

# Define column counts for various data structures
ncol_traj_particles = 10
nlowest = 5

ncol_convergence = 2
ncol_debug_energy_compare = 8
ncol_particles_dat = 5
ncol_particlesfinal = 4
ncol_density_profile = 2
ncol_mass_profile = 2
ncol_psi_profile = 2
ncol_psi_theory = 3
ncol_dpsi_dr = 2
ncol_drho_dpsi = 2
ncol_f_of_E = 2
ncol_df_fixed_radius = 2
ncol_combined_histogram = 3
ncol_integrand = 2
ncol_particles_initial = 4
ncol_Rank_Mass_Rad_VRad_unsorted = 7

# Constants for unit conversions
kmsec_to_kpcmyr = 1.02271e-3  # Conversion factor from km/s to kpc/Myr

def _calculate_robust_range(data_array, percentile=95.0, percentile_multiplier=1.2,
                           min_abs_extent=1.0,
                           default_if_empty=(0, 1.0),
                           can_be_negative=False,
                           force_symmetric_around_zero=False,
                           axis_name="axis"):
    """
    Calculates a robust plot range based on percentiles.

    Handles positive-only data (e.g., radius, speed magnitude) and data
    that can be positive or negative (e.g., radial velocity).

    Parameters
    ----------
    data_array : np.ndarray
        1D array of data points.
    percentile : float
        Percentile to use (0-100) for range calculation.
    percentile_multiplier : float
        Factor to multiply the percentile value by to determine the range extent.
    min_abs_extent : float, optional
        A minimum absolute value for the extent of the range (e.g., if range is
        [0, X], then X must be at least `min_abs_extent`). Ensures a non-zero
        range. Default is 1.0.
    default_if_empty : tuple, optional
        Default (min_val, max_val) tuple to return if `data_array` is empty
        or contains no finite values. Default is (0, 1.0).
    can_be_negative : bool, optional
        If True, the data can have negative values, and the range calculation
        will consider both positive and negative parts independently.
        If False, data is assumed to be non-negative, and the range will
        start from 0. Default is False.
    force_symmetric_around_zero : bool, optional
        If True (and `can_be_negative` is True), makes the range symmetric
        around zero, e.g., (-max_abs_val, max_abs_val). Default is False.
    percentile_fallback : float, optional
        Percentile (0-100) to use for range estimation if the median is close
        to zero, to avoid an overly small range. Default is 95.0.
    axis_name : str, optional
        Name of the axis for logging purposes (e.g., "radius", "velocity").
        Default is "axis".

    Returns
    -------
    tuple
        (min_val, max_val) for the plot range.
    """
    if data_array is None or data_array.size == 0:
        logger.debug(f"_calculate_robust_range for {axis_name}: Data array empty. Returning default {default_if_empty}.")
        return default_if_empty

    finite_data = data_array[np.isfinite(data_array)]
    if finite_data.size == 0:
        logger.debug(f"_calculate_robust_range for {axis_name}: No finite data. Returning default {default_if_empty}.")
        return default_if_empty

    min_lim_final = 0.0
    max_lim_final = min_abs_extent # Initialize with a minimal positive extent

    if can_be_negative:
        pos_data = finite_data[finite_data > 0]
        neg_data = finite_data[finite_data < 0] # Keep negative values for percentile calculation

        # --- Positive side ---
        if pos_data.size > 0:
            # Direct index calculation for percentile (data may not be sorted)
            pos_data_sorted = np.sort(pos_data)
            p_index = int(percentile / 100.0 * (len(pos_data_sorted) - 1))
            p_val_pos = pos_data_sorted[p_index]
            max_lim_final = max(p_val_pos * percentile_multiplier, min_abs_extent)
            logger.debug(f"_calc_robust for {axis_name} (pos): {percentile}th percentile ({p_val_pos:.2e}) * {percentile_multiplier} -> max_lim_final={max_lim_final:.2e}")
        else: # No positive data
            max_lim_final = default_if_empty[1] if default_if_empty[1] > 0 else min_abs_extent # Ensure some positive extent

        # --- Negative side ---
        if neg_data.size > 0:
            # Direct index calculation for negative data percentile
            neg_data_abs_sorted = np.sort(np.abs(neg_data))
            p_index = int(percentile / 100.0 * (len(neg_data_abs_sorted) - 1))
            p_val_neg_abs = neg_data_abs_sorted[p_index]
            min_lim_final = -max(p_val_neg_abs * percentile_multiplier, min_abs_extent)
            logger.debug(f"_calc_robust for {axis_name} (neg): {percentile}th percentile of abs ({p_val_neg_abs:.2e}) * {percentile_multiplier} -> min_lim_final={min_lim_final:.2e}")
        else: # No negative data
            min_lim_final = default_if_empty[0] if default_if_empty[0] < 0 else -min_abs_extent # Ensure some negative extent if expected

        if force_symmetric_around_zero:
            max_abs_val = max(np.abs(min_lim_final), np.abs(max_lim_final))
            min_lim_final = -max_abs_val
            max_lim_final = max_abs_val

    else: # Data is purely positive (or zero)
        # For positive-only data, range starts at 0.
        min_lim_final = 0.0
        abs_finite_data = np.abs(finite_data) # Ensure all positive for percentile calculation
        if abs_finite_data.size > 0:
            # Direct index calculation for percentile
            abs_data_sorted = np.sort(abs_finite_data)
            p_index = int(percentile / 100.0 * (len(abs_data_sorted) - 1))
            p_val_abs = abs_data_sorted[p_index]
            max_lim_final = max(p_val_abs * percentile_multiplier, min_abs_extent)
            logger.debug(f"_calc_robust for {axis_name} (pos-only): {percentile}th percentile ({p_val_abs:.2e}) * {percentile_multiplier} -> max_lim_final={max_lim_final:.2e}")
        else: # Should not happen due to earlier check, but as a fallback
            max_lim_final = default_if_empty[1]

    # Ensure min_lim is less than max_lim
    if min_lim_final >= max_lim_final:
        if can_be_negative and not force_symmetric_around_zero:
             # Could happen if e.g. only negative data, min_lim_final calculated, max_lim_final is default_if_empty[1]
             # or only positive data. Try to make a sensible small range.
            if min_lim_final == 0 and max_lim_final == 0: # Both ended up zero
                max_lim_final = min_abs_extent
            else: # If one side is zero, and other side also becomes zero or crosses over
                if np.abs(min_lim_final) > np.abs(max_lim_final):
                    max_lim_final = np.abs(min_lim_final) / 2.0 # Arbitrary small positive
                else:
                    min_lim_final = -np.abs(max_lim_final) / 2.0 # Arbitrary small negative
        elif force_symmetric_around_zero : # Should be covered by logic but as safety
            max_lim_final = max(np.abs(min_lim_final), np.abs(max_lim_final), min_abs_extent)
            min_lim_final = -max_lim_final
        else: # Positive only data
            max_lim_final = max(min_lim_final, max_lim_final, min_abs_extent) # Ensure max_lim is at least min_abs_extent
        logger.debug(f"_calc_robust for {axis_name}: Adjusted min/max due to overlap or zero: min={min_lim_final:.2e}, max={max_lim_final:.2e}")


    logger.debug(f"_calculate_robust_range for {axis_name}: Result ({min_lim_final:.2e}, {max_lim_final:.2e})")
    return (min_lim_final, max_lim_final)


def _calculate_global_animation_ranges(rank_files_subset,
                                       r_min_abs_extent, v_min_abs_extent,
                                       r_default_if_empty, v_default_if_empty,
                                       kmsec_to_kpcmyr,
                                       current_suffix_for_logging,
                                       axis_name_prefix="anim_global",
                                       config=None):
    """
    Calculates global X (radius) and Y (velocity) ranges for phase space animation.

    Processes only the first and last snapshot files, calculates 95th percentile
    for both radius and velocity, then uses the maximum of these two ranges.
    This ensures consistent axis scaling across all animation frames.

    Parameters
    ----------
    rank_files_subset : list[str]
        A list of Rank snapshot filenames to sample for range calculation.
    r_min_abs_extent : float
        Minimum absolute extent for the radius range.
    v_min_abs_extent : float
        Minimum absolute extent for the velocity range.
    r_default_if_empty : tuple
        Default (min, max) for radius range if data is empty.
    v_default_if_empty : tuple
        Default (min, max) for velocity range if data is empty.
    kmsec_to_kpcmyr : float
        Conversion factor from km/s to kpc/Myr (or its inverse depending on usage).
        Here, it's used to convert simulation velocity units to km/s.
    current_suffix_for_logging : str
        The current simulation suffix, used for logging messages.
    axis_name_prefix : str, optional
        Prefix for axis names in logging messages. Default is "anim_global".

    Returns
    -------
    tuple
        A tuple ((overall_min_r, overall_max_r), (overall_min_v, overall_max_v))
        representing the global ranges for radius and velocity. Returns default
        ranges if no valid data is found in the subset.
    """
    if not rank_files_subset:
        logger.warning(f"{axis_name_prefix}: No rank files provided for global range calculation. Returning defaults.")
        return (r_default_if_empty, v_default_if_empty)

    # Use config parameters if available
    percentile = 95.0
    x_percentile_multiplier = 1.2
    y_percentile_multiplier = 1.2
    if config:
        percentile = getattr(config, 'x_percentile', 95.0)
        x_percentile_multiplier = getattr(config, 'x_percentile_multiplier', 1.2)
        y_percentile_multiplier = getattr(config, 'y_percentile_multiplier', 1.2)
    
    # Process only first and last frames
    frames_to_check = []
    if len(rank_files_subset) > 0:
        frames_to_check.append(rank_files_subset[0])  # First frame
        if len(rank_files_subset) > 1:
            frames_to_check.append(rank_files_subset[-1])  # Last frame
    
    max_r = 0
    max_v = 0
    any_valid_data_found = False
    
    # Define dtype list for efficient loading of specific columns
    # Rank(0), Mass(1), Radius(2), Vrad(3), Psi(4), Energy(5), L(6), Density(7)
    # We need Radius (2), Vrad (3), L (6)
    cols_to_load_indices = sorted([2, 3, 6])
    # Dtype for *all* 8 columns of Rank_Mass_Rad_VRad_sorted*.dat
    full_rank_sorted_dtype_list = [
        np.int32, np.float32, np.float32, np.float32,
        np.float32, np.float32, np.float32, np.float32
    ]
    
    # Map loaded column index back to conceptual meaning (R, Vrad, L)
    idx_map_radius = 0
    idx_map_vrad = 1
    idx_map_l = 2

    logger.info(f"{axis_name_prefix}: Calculating 95th percentile ranges for first and last frames (2 files max).")

    for fname in frames_to_check:
        loaded_cols = load_specific_columns_bin(
            fname,
            ncols_total=ncol_Rank_Mass_Rad_VRad_sorted,
            cols_to_load=cols_to_load_indices,
            dtype_list=full_rank_sorted_dtype_list
        )

        if loaded_cols is None or len(loaded_cols) != len(cols_to_load_indices):
            continue

        radii_snap_raw = loaded_cols[idx_map_radius]
        vrad_snap_raw = loaded_cols[idx_map_vrad]
        l_snap_raw = loaded_cols[idx_map_l]
        del loaded_cols
        gc.collect()

        # Filter out non-positive radii
        valid_radius_mask = (radii_snap_raw > 1e-9) & np.isfinite(radii_snap_raw)
        if not np.any(valid_radius_mask):
            continue

        radii_snap = radii_snap_raw[valid_radius_mask]
        vrad_snap = vrad_snap_raw[valid_radius_mask]
        l_snap = l_snap_raw[valid_radius_mask]

        # Calculate total velocity in km/s
        with np.errstate(divide='ignore', invalid='ignore'):
            tangential_velocity_sim = l_snap / radii_snap
        total_velocity_sim = np.sqrt(vrad_snap**2 + tangential_velocity_sim**2)
        total_velocity_kms_snap = total_velocity_sim / kmsec_to_kpcmyr

        # Filter velocities for finite values
        valid_velocity_mask = np.isfinite(total_velocity_kms_snap)
        if not np.any(valid_velocity_mask):
             continue
        
        final_radii_snap = radii_snap[valid_velocity_mask]
        final_total_velocity_kms_snap = total_velocity_kms_snap[valid_velocity_mask]

        if final_radii_snap.size == 0:
            continue
        
        any_valid_data_found = True

        # Calculate 95th percentile for this frame
        r_95 = np.percentile(final_radii_snap, percentile) * x_percentile_multiplier
        v_95 = np.percentile(final_total_velocity_kms_snap, percentile) * y_percentile_multiplier
        
        # Update maximum values
        max_r = max(max_r, r_95)
        max_v = max(max_v, v_95)

    if not any_valid_data_found:
        logger.warning(f"{axis_name_prefix}: No valid data found in any snapshot files. Returning defaults.")
        return (r_default_if_empty, v_default_if_empty)
    
    # Apply minimum extents if needed
    max_r = max(max_r, r_min_abs_extent)
    max_v = max(max_v, v_min_abs_extent)
    
    final_r_range = (0, max_r)
    final_v_range = (0, max_v)

    logger.info(f"{axis_name_prefix}: 95th percentile-based ranges (first/last frames): R={final_r_range}, V={final_v_range}")
    return (final_r_range, final_v_range)



def _calculate_profile_animation_ranges(snapshots, config, data_index=2, 
                                       x_default_max=300.0, y_multiplier=1.1,
                                       animation_name=""):
    """
    Calculate X and Y axis ranges for profile animations using smart x-axis calculation
    based on where curves peak/decay or reach asymptotic values.
    
    Parameters
    ----------
    snapshots : list
        List of snapshot tuples (snap, x_data, y_data)
    config : Configuration
        Configuration object (checked for use_median_ranges flag)
    data_index : int, optional
        Index of the data value in snapshot tuple (2 for most animations)
    x_default_max : float, optional
        Default maximum X value if calculation fails
    y_multiplier : float, optional
        Multiplier for Y-axis maximum (default 1.1 = 10% headroom)
    animation_name : str
        Name of the animation (used to determine calculation method)
        
    Returns
    -------
    tuple
        ((x_min, x_max), y_max) where x_max is calculated based on profile behavior
        and y_max is multiplier × maximum y value
    """
    if not config or not hasattr(config, 'use_median_ranges') or not config.use_median_ranges:
        return None, None  # Use original behavior
    
    # Determine animation type based on name
    is_mass_animation = "mass" in animation_name.lower()
    
    # Process only first and last snapshots for X-axis calculation
    snapshots_to_check = []
    if len(snapshots) > 0:
        snapshots_to_check.append(snapshots[0])  # First frame
        if len(snapshots) > 1:
            snapshots_to_check.append(snapshots[-1])  # Last frame
    
    max_x_candidate = 0
    max_y = 0  # Y-axis still uses simple max across all frames
    
    # Calculate smart X-axis range from first/last frames
    for snapshot in snapshots_to_check:
        x_data = np.array(snapshot[1])  # radius data
        y_data = np.array(snapshot[data_index])  # profile data
        
        if len(x_data) > 0 and len(y_data) > 0:
            # Sort by radius to ensure proper ordering
            sorted_indices = np.argsort(x_data)
            x_sorted = x_data[sorted_indices]
            y_sorted = y_data[sorted_indices]
            
            if is_mass_animation:
                # For mass: find where curve rises above 97% of maximum
                y_max = np.max(y_sorted)
                if y_max > 0:
                    threshold = 0.97 * y_max
                    # Find first radius where mass exceeds threshold
                    above_threshold = np.where(y_sorted >= threshold)[0]
                    if len(above_threshold) > 0:
                        radius_candidate = x_sorted[above_threshold[0]]
                        max_x_candidate = max(max_x_candidate, radius_candidate * 1.33)
            else:
                # For density/psi: find where curve drops below threshold of maximum
                y_max = np.max(y_sorted)
                if y_max > 0:
                    # Use 5% for psi, 3% for density
                    if "psi" in animation_name.lower():
                        threshold = 0.05 * y_max
                    else:
                        threshold = 0.03 * y_max
                    # Find the peak location first
                    peak_idx = np.argmax(y_sorted)
                    
                    # Look for drop below threshold after the peak
                    if peak_idx < len(y_sorted) - 1:
                        below_threshold = np.where(y_sorted[peak_idx:] < threshold)[0]
                        if len(below_threshold) > 0:
                            # Add peak_idx to get correct index in original array
                            radius_candidate = x_sorted[peak_idx + below_threshold[0]]
                            max_x_candidate = max(max_x_candidate, radius_candidate * 1.33)
    
    # Calculate max Y across all frames (not just first/last)
    for snapshot in snapshots:
        y_data = np.array(snapshot[data_index])  # profile data
        if len(y_data) > 0:
            max_y = max(max_y, np.max(y_data))
    
    # Use defaults if no valid data found
    final_x_max = max_x_candidate if max_x_candidate > 0 else x_default_max
    final_y_max = max_y * y_multiplier if max_y > 0 else 1.0
    
    # Log the calculated ranges
    if is_mass_animation:
        calculation_method = "97% of max"
    elif "psi" in animation_name.lower():
        calculation_method = "5% of max after peak"
    else:
        calculation_method = "3% of max after peak"
    logger.info(f"{animation_name} animation: X range = [0, {final_x_max:.3f}] kpc ({calculation_method}), Y max = {final_y_max:.3e}")
    
    return (0, final_x_max), final_y_max


def _calculate_robust_range_from_histogram(bin_centers, bin_counts, 
                                          percentile=95.0, percentile_multiplier=1.2,
                                          min_abs_extent=1.0,
                                          default_if_empty=(0, 1.0), can_be_negative=False,
                                          axis_name="axis"):
    """
    Calculate robust plot range from pre-binned histogram data using percentile-based scaling.
    
    This function reconstructs the underlying distribution from histogram bins to calculate
    a percentile-based range that captures the bulk of the data while avoiding outliers.
    
    Parameters
    ----------
    bin_centers : array-like
        The center values of histogram bins (e.g., radius or velocity values).
    bin_counts : array-like
        The count/frequency in each bin.
    percentile : float
        Percentile to use for determining the range extent (e.g., 95.0).
    percentile_multiplier : float
        Factor to multiply the percentile value by when determining the range extent.
        Typical value: 1.2 for 20% padding beyond the percentile.
    min_abs_extent : float, optional
        Minimum absolute extent for the range. Default is 1.0.
        Ensures the range is at least this wide even for very concentrated data.
    default_if_empty : tuple, optional
        Default (min, max) range to use if no valid data. Default is (0, 1.0).
    can_be_negative : bool, optional
        Whether the data can have negative values. Default is False.
        If False, the minimum is clamped to 0.
    axis_name : str, optional
        Name of the axis for logging purposes. Default is "axis".
        
    Returns
    -------
    tuple
        (min_value, max_value) representing the calculated range.
        
    Notes
    -----
    The algorithm works by:
    1. Creating a weighted array where each bin center appears N times (N = bin count)
    2. Calculating the specified percentile of this reconstructed distribution
    3. Setting range as [0, percentile * multiplier] for non-negative data
    4. Applying minimum extent constraints
    5. Handling edge cases with appropriate defaults
    
    This is particularly useful for histogram data from nsphere.c where we have
    pre-binned data but still want to apply intelligent axis scaling.
    """
    # Convert to numpy arrays and flatten
    bin_centers = np.asarray(bin_centers).flatten()
    bin_counts = np.asarray(bin_counts).flatten()
    
    # Validate inputs
    if len(bin_centers) != len(bin_counts):
        logger.warning(f"{axis_name}: Bin centers and counts have different lengths. Using defaults.")
        return default_if_empty
        
    # Filter out bins with zero or negative counts
    valid_mask = bin_counts > 0
    valid_centers = bin_centers[valid_mask]
    valid_counts = bin_counts[valid_mask].astype(int)
    
    if len(valid_centers) == 0:
        logger.debug(f"{axis_name}: No bins with positive counts. Using defaults.")
        return default_if_empty
        
    # Reconstruct the distribution by repeating each bin center by its count
    # This gives us an array that represents the underlying distribution
    try:
        reconstructed_data = np.repeat(valid_centers, valid_counts)
    except MemoryError:
        # If too much data, sample instead
        logger.warning(f"{axis_name}: Too much data for full reconstruction. Sampling instead.")
        # Use weighted random sampling
        total_counts = np.sum(valid_counts)
        sample_size = min(1000000, total_counts)  # Sample up to 1M points
        probabilities = valid_counts / total_counts
        reconstructed_data = np.random.choice(valid_centers, size=sample_size, p=probabilities)
    
    if reconstructed_data.size == 0:
        logger.debug(f"{axis_name}: No valid data after reconstruction. Using defaults.")
        return default_if_empty
        
    # Sort the reconstructed data - necessary because filtered bin centers may not be contiguous
    reconstructed_data = np.sort(reconstructed_data)
    
    # Calculate the specified percentile using direct index method
    p_index = int(percentile / 100.0 * (len(reconstructed_data) - 1))
    data_percentile = reconstructed_data[p_index]
    
    # Calculate additional statistics for logging context
    # Now that data is sorted, we can calculate percentiles correctly
    median_index = int(0.5 * (len(reconstructed_data) - 1))
    data_median = reconstructed_data[median_index]
    
    # Calculate 25th and 75th percentiles for IQR
    p25_index = int(0.25 * (len(reconstructed_data) - 1))
    p75_index = int(0.75 * (len(reconstructed_data) - 1))
    p25 = reconstructed_data[p25_index]
    p75 = reconstructed_data[p75_index]
    
    # Calculate the range based on percentile
    if can_be_negative:
        # For data that can be negative, use percentile for both sides
        max_abs_val = data_percentile * percentile_multiplier
        min_val = -max_abs_val
        max_val = max_abs_val
    else:
        # For non-negative data (like radius), start from 0
        min_val = 0
        max_val = max(data_percentile * percentile_multiplier, min_abs_extent)
        
    # Log the statistics
    logger.info(f"{axis_name} histogram statistics: {percentile:.0f}th percentile={data_percentile:.3f}, "
                f"median={data_median:.3f}, IQR=[{p25:.3f}, {p75:.3f}], range=[{min_val:.3f}, {max_val:.3f}]")
    
    return (min_val, max_val)


def filter_finite_rows(*arrs):
    """
    Filter out rows containing NaN or Inf values across multiple arrays.

    Parameters
    ----------
    *arrs : array-like
        Variable number of arrays to filter. All must have the same length.

    Returns
    -------
    list
        New arrays with invalid rows removed from all input arrays.

    Notes
    -----
    This function removes any row that contains NaN or Inf in ANY of the input arrays.
    All arrays must have the same length for proper row-wise filtering.
    """
    if not arrs:
        return arrs
    length = len(arrs[0])
    mask = np.ones(length, dtype=bool)
    for arr in arrs:
        if len(arr) != length:
            raise ValueError(
                "Arrays must have the same length to apply filter_finite_rows.")
        mask &= np.isfinite(arr)
    return [arr[mask] for arr in arrs]

# Additional column counts derived from the base counts
ncol_trajectories = 1 + 3 * ncol_traj_particles
ncol_single_trajectory = 4
ncol_energy_and_angular_momentum_vs_time = 1 + 4 * ncol_traj_particles
ncol_lowest_l_trajectories = 1 + 3 * nlowest
ncol_2d_hist_initial = 3
ncol_2d_hist_final = 3
ncol_all_particle_data_snapshot = 3
ncol_Rank_Mass_Rad_VRad_unsorted = 7
ncol_Rank_Mass_Rad_VRad_sorted = 8

# Parameter information is displayed in the main function header

def get_mem_usage_mb():
    """
    Get current memory usage of the Python process.

    Returns
    -------
    float
        Current Resident Set Size (RSS) memory usage in megabytes.

    Notes
    -----
    Provides the actual physical memory being used by the Python process
    using the psutil library to access system resource information.
    """
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024.0 * 1024.0)

def update_timer_energy_plots(stop_timer, energy_plot_start_time, paced_mode):
    """
    Continuously update the timer display for energy plot processing.

    Parameters
    ----------
    stop_timer : threading.Event
        Event to signal when to stop the timer thread
    energy_plot_start_time : float
        Time when energy plot processing started
    paced_mode : bool
        Whether the visualization is in paced mode (with deliberate delays)

    Notes
    -----
    Runs in a separate thread and updates the console with elapsed time,
    estimated remaining time, and processing rate. Updates every 0.1 seconds
    until stop_timer is set.
    """
    # Only wait briefly to let the initial display settle
    if paced_mode:
        # In paced mode, use longer delay for better visualization
        time.sleep(0.5)
    else:
        # In fast mode (default), use minimal delay
        time.sleep(0.01)

    while not stop_timer.is_set():
        cur_elapsed = time.time() - energy_plot_start_time
        # Apply a minimum nominal time to prevent unrealistically high rates
        # For display purposes, ensure elapsed time is at least 0.01 seconds
        displayed_elapsed = max(cur_elapsed, 0.01)  # Minimum displayed time
        nominal_elapsed = max(cur_elapsed, 0.02)  # Minimum for rate calculation
        est_remaining = nominal_elapsed  # Estimate remaining time based on elapsed
        cur_rate = 0.5 / nominal_elapsed  # 0.5 plots in the elapsed time
        # Cap the rate at a reasonable maximum for display
        cur_rate = min(cur_rate, 99.9)

        # Format the current timing info with extra padding spaces
        time_info = f" [{displayed_elapsed:.2f}<{est_remaining:.2f}, {cur_rate:.1f}file/s]"

        # Print the progress bar and current plot name inline with extra padding
        bar_length = 20
        half_filled = int(bar_length * 0.5)
        half_bar = '█' * half_filled + ' ' * (bar_length - half_filled)

        # Get current file prefix
        unsorted_name = "Energy_vs_timestep_unsorted"  # Base name without suffix
        prefix = get_file_prefix(unsorted_name)

        # Use ANSI escape sequences to update progress in-place
        # Truncate the content part of the string
        content_string = f"Save: {half_bar} 50.0% | File: {prefix}{time_info}"
        sys.stdout.write(f"\r\033[2K{truncate_and_pad_string(content_string)}")
        sys.stdout.flush()

        # Sleep briefly to avoid high CPU usage
        if paced_mode:
            # In paced mode, use longer delay for better visualization
            time.sleep(0.2)
        else:
            # In fast mode (default), use minimal delay
            time.sleep(0.02)

def get_snapshot_number(filename, pattern=None):
    """
    Extract the snapshot number from a filename containing a timestamp pattern.

    Parameters
    ----------
    filename : str
        Filename to extract snapshot number from
    pattern : re.Pattern, optional
        Compiled regex pattern to use. If None, defaults to Rank_Mass_Rad_VRad_sorted_t pattern.

    Returns
    -------
    int
        Snapshot number extracted from filename, or a very large number if no match found

    Notes
    -----
    Uses regular expression matching to extract the snapshot number from filenames
    that follow a pattern like "Rank_Mass_Rad_VRad_sorted_t00012". Returns a large
    value for non-matching files to ensure they sort after valid snapshot files.
    """
    global suffix
    if pattern is None:
        # Build a pattern that correctly handles the suffix
        pattern_str = r'Rank_Mass_Rad_VRad_sorted_t(\d+)' + re.escape(suffix) + r'\.dat$'
        pattern = re.compile(pattern_str)

    match = pattern.search(filename)
    if match:
        return int(match.group(1))
    return 999999999

def get_file_prefix(filepath):
    """
    Extract the filename prefix without path, extension or suffix.

    Parameters
    ----------
    filepath : str
        Full path or filename to process

    Returns
    -------
    str
        The extracted filename prefix

    Examples
    --------
    >>> get_file_prefix('results/phase_space_initial_kris_40000_1001_5.png')
    'phase_space_initial'
    >>> get_file_prefix('loading_combined_histogram')
    'combined_histogram'

    Notes
    -----
    Handles special cases including loading prefixes and various suffix
    patterns. For files with a standard suffix pattern (_tag_nnn_nnn_n),
    removes these components to extract the core filename.
    """
    # Make sure filepath is a string (handle None or other types)
    if not isinstance(filepath, str):
        return "unknown"

    # Special handling for "loading_" identifiers
    if filepath.startswith("loading_"):
        return filepath[8:]  # Remove "loading_" prefix

    # Get the filename without the path
    filename = os.path.basename(filepath)
    # Remove the extension
    prefix = os.path.splitext(filename)[0]
    # For files with the suffix, remove the suffix part
    parts = prefix.split('_')

    # Handle potential energy_compare filename change
    if prefix.startswith("energy_compare") and not prefix.startswith("debug_energy_compare"):
        prefix = prefix.replace("energy_compare", "debug_energy_compare", 1)

    # If the filename has a suffix pattern _tag_nnn_nnn_n
    if len(parts) > 3 and all(part.isdigit() for part in parts[-3:]):
        # Remove the last parts that match the suffix pattern
        # For files with the pattern name_tag_npts_Ntimes_tfinal_factor
        if len(parts) > 3:
            # Keep everything before the suffix
            # Re-check prefix after potential rename
            prefix_parts = prefix.split('_')
            return '_'.join(prefix_parts[:-4] if prefix_parts[-4].isalpha() else prefix_parts[:-3])

    # If the file path looks like "loading_something_convergence_data" or other non-filename format
    # (used for progress updates), just return the basename without special processing
    if "loading_" in prefix or "processing_" in prefix:
        return prefix

    return prefix

def load_partial_file(filename, dtype=np.float32):
    """
    Load data from file, handling incomplete or partially-written lines.

    Parameters
    ----------
    filename : str
        Path to the file to read
    dtype : numpy.dtype, optional
        Data type to use for the array, by default np.float32

    Returns
    -------
    numpy.ndarray or None
        Array containing the data successfully read from the file, or None if
        the file doesn't exist or contains no valid data

    Notes
    -----
    The function stops reading if it encounters a line it cannot parse,
    returning all data successfully read up to that point. This is useful
    for handling files that might be incomplete or partially written.

    Skips blank lines and stops reading at the first parsing error.
    """
    # Skip if file doesn't exist
    if not os.path.exists(filename):
        print(truncate_and_pad_string(f"WARNING: {filename} not found. Skipping."))
        return None

    valid_rows = []
    with open(filename, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                # skip blank lines
                continue
            try:
                row = np.fromstring(stripped, sep=' ', dtype=dtype)
            except ValueError:
                # If there's a parse error (e.g., incomplete line),
                # Stop reading further
                break
            if len(row) == 0:
                # If line was empty or couldn't parse anything, stop
                continue
            valid_rows.append(row)
    return np.array(valid_rows, dtype=dtype)

def load_partial_file_bin(filename, ncols, dtype=np.float32):
    """
    Load data from binary file with support for different data types.

    Parameters
    ----------
    filename : str
        Path to the binary file to read
    ncols : int
        Number of columns expected in the data
    dtype : numpy.dtype or list, optional
        Data type(s) to use for the array, by default np.float32

    Returns
    -------
    numpy.ndarray or None
        Array containing the data from the file, or None if the file
        is missing or contains no valid data

    Notes
    -----
    Supports two different modes of operation:

    1. Single dtype (e.g., np.float32):
       Returns array of shape (N, ncols)

    2. List/tuple of dtypes:
       Returns structured array with fields named 'f0', 'f1', ..., 'f{ncols-1}',
       each with its corresponding dtype from the list

    Uses align=False to ensure tight packing of structured arrays.
    Reads as many complete rows as possible from the file.
    """
    if not os.path.exists(filename):
        print(truncate_and_pad_string(f"WARNING: {filename} not found. Skipping."))
        return None

    # Get file size for diagnostics
    file_size = os.path.getsize(filename)
    # Log file size for very small files (<1000 bytes) to help debug issues
    if file_size < 1000:
        logger.info(f"Small binary file detected: {filename}, size: {file_size} bytes")

    with open(filename, 'rb') as f:
        raw = f.read()
    if not raw:
        logger.warning(f"Empty binary file: {filename}, size: {file_size} bytes")
        return None

    # Check if dtype is a list of dtypes => build structured dtype
    if isinstance(dtype, (list, tuple)):
        if len(dtype) != ncols:
            logger.error(f"ERROR: dtype list length {len(dtype)} != ncols={ncols}")
            return None

        # Build fields: [("f0", dtype[0]), ("f1", dtype[1]), ...]
        fields = []
        for i in range(ncols):
            fields.append((f"f{i}", dtype[i]))
        structured = np.dtype(fields, align=False)

        itemsize = structured.itemsize
        nrows = len(raw) // itemsize

        # Log extra diagnostics for small files
        if nrows < 100:
            logger.info(f"Binary file has few rows: {filename}, raw bytes: {len(raw)}, itemsize: {itemsize}, estimated rows: {nrows}")

        if nrows == 0:
            logger.warning(f"No complete rows found in binary file: {filename}, raw bytes: {len(raw)}, itemsize: {itemsize}")
            # For very small files, log hex dump of raw bytes to help debug
            if len(raw) < 100:
                logger.info(f"Raw binary data (hex): {' '.join(f'{b:02x}' for b in raw[:100])}")
            return None

        # Drop leftover bytes
        bytes_to_use = nrows * itemsize
        if len(raw) != bytes_to_use:
            logger.warning(f"Partial final row detected: {filename}, using {bytes_to_use} of {len(raw)} bytes")
        raw = raw[:bytes_to_use]

        arr = np.frombuffer(raw, dtype=structured)
        gc.collect()

        # For debugging small datasets, log some content
        if nrows < 10:
            logger.info(f"Structured array content (first {min(nrows, 5)} rows): {arr[:5] if nrows > 0 else 'empty'}")

        return arr

    else:
        # Normal single-dtype path
        arr = np.frombuffer(raw, dtype=dtype)
        full_count = arr.size // ncols

        # Log extra diagnostics for small arrays
        if full_count < 100:
            logger.info(f"Binary file has few rows: {filename}, array size: {arr.size}, columns: {ncols}, complete rows: {full_count}")

        if full_count == 0:
            logger.warning(f"No complete rows found in binary file: {filename}, array size: {arr.size}, columns: {ncols}")
            # For very small arrays, log raw values to help debug
            if arr.size < 100:
                logger.info(f"Raw array values: {arr}")
            return None

        arr = arr[:full_count * ncols]
        arr = arr.reshape((full_count, ncols))
        gc.collect()

        # For debugging small datasets, log some content
        if full_count < 10:
            logger.info(f"Array content (first {min(full_count, 5)} rows): {arr[:5] if full_count > 0 else 'empty'}")

        return arr

def safe_load_and_filter(filename, dtype=np.float32):
    """
    Load and filter data from file with error handling.

    Parameters
    ----------
    filename : str
        Path to the file to read
    dtype : numpy.dtype, optional
        Data type to use for the array, by default np.float32

    Returns
    -------
    numpy.ndarray or None
        Filtered array containing only valid data, or None if
        the file doesn't exist or contains no valid data

    Notes
    -----
    This function performs three steps:

    1. Verifies the file exists
    2. Loads as much data as possible using load_partial_file
    3. Filters out rows containing NaN or Inf values

    Returns None with appropriate warnings if the file is missing or
    contains no valid data after filtering.
    """
    if not os.path.exists(filename):
        print(truncate_and_pad_string(f"WARNING: {filename} does not exist. Skipping."))
        return None

    data = load_partial_file(filename, dtype=dtype)
    gc.collect()

    if data is None or data.size == 0:
        # Means either the file was missing or empty or partial lines only
        print(truncate_and_pad_string(f"WARNING: {filename} had no valid data. Skipping."))
        return None

    # Filter out any invalid (NaN/Inf) rows
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]

    if data.size == 0:
        print(
            truncate_and_pad_string(f"WARNING: After filtering, {filename} had no valid rows. Skipping."))
        return None

    return data

def safe_load_and_filter_bin(filename, ncols, dtype=np.float32):
    """
    Load and filter binary data with support for structured arrays.

    Parameters
    ----------
    filename : str
        Path to the binary file to read
    ncols : int
        Number of columns expected in the data
    dtype : numpy.dtype or list, optional
        Data type(s) to use for the array, by default np.float32

    Returns
    -------
    numpy.ndarray or None
        2D float array containing only valid data, or None if
        the file doesn't exist or contains no valid data

    Notes
    -----
    This function performs four steps:

    1. Verifies the file exists
    2. Loads binary data using load_partial_file_bin
    3. Converts structured arrays to standard floating-point arrays if needed
    4. Filters out rows containing NaN or Inf values

    For structured arrays (when dtype is a list), each field is extracted and
    converted to float32 before filtering. This ensures consistent handling
    regardless of the original data types.
    """
    if not os.path.exists(filename):
        print(truncate_and_pad_string(f"WARNING: {filename} does not exist. Skipping."))
        return None

    data = load_partial_file_bin(filename, ncols=ncols, dtype=dtype)
    gc.collect()

    if data is None or data.size == 0:
        print(truncate_and_pad_string(f"WARNING: {filename} had no valid data. Skipping."))
        return None

    # Handle structured arrays (from list dtype) differently
    if isinstance(dtype, (list, tuple)):
        # Convert structured array to a regular 2D array with all fields as float32
        nrows = data.shape[0]
        float_data = np.zeros((nrows, ncols), dtype=np.float32)

        # Extract each field and convert to float
        for i in range(ncols):
            field_name = f"f{i}"
            float_data[:, i] = data[field_name].astype(np.float32, copy=False)

        # Replace structured array with regular float array
        data = float_data

        # Apply isfinite to the standard array
        mask = np.all(np.isfinite(data), axis=1)
        data = data[mask]
    else:
        # For regular arrays, just filter normally
        mask = np.all(np.isfinite(data), axis=1)
        data = data[mask]

    if data.size == 0:
        print(
            truncate_and_pad_string(f"WARNING: After filtering, {filename} had no valid rows. Skipping."))
        return None

    return data

def load_specific_columns_bin(filename, ncols_total, cols_to_load, dtype_list):
    """
    Loads specific columns from a flat binary file using memory mapping.

    This version uses numpy.memmap for efficiency with large files, reading
    only necessary data pages from disk into memory when columns are accessed.

    Parameters
    ----------
    filename : str
        Path to the binary file.
    ncols_total : int
        Total number of columns (fields) in each row of the file.
    cols_to_load : list[int]
        List of 0-based column indices to load and return.
    dtype_list : list
        List of numpy dtypes corresponding to *all* columns in the file,
        in order. E.g., [np.int32, np.float32, np.float32, ...].

    Returns
    -------
    list[np.ndarray] or None
        A list containing 1D numpy arrays (as float32) for each requested
        column, filtered for finite values based on the loaded columns.
        Returns None if the file doesn't exist, cannot be mapped,
        or contains no valid data in requested columns.

    Notes
    -----
    Requires the `dtype_list` specifying the type for *all* columns
    to correctly interpret the binary structure for memmap.
    Returned arrays are converted to float32 for consistency.
    """
    if not os.path.exists(filename):
        logger.warning(f"Memmap loader: File not found {filename}")
        return None
    if not isinstance(dtype_list, list) or len(dtype_list) != ncols_total:
         logger.error(f"Memmap loader: Invalid dtype_list provided for {filename}. Expected list of length {ncols_total}.")
         return None

    try:
        # Define the structured dtype for the entire row
        row_fields = [(f'f{i}', dtype) for i, dtype in enumerate(dtype_list)]
        row_dtype = np.dtype(row_fields)
        item_size = row_dtype.itemsize

        file_size = os.path.getsize(filename)
        if file_size < item_size:
            logger.warning(f"Memmap loader: File {filename} is smaller than one row.")
            return None

        # Calculate number of rows based on file size and item size
        num_rows = file_size // item_size
        if num_rows == 0:
             logger.warning(f"Memmap loader: No complete rows found in {filename}")
             return None

        # Memory-map the file
        # mode='r' is read-only
        mmap_array = np.memmap(filename, dtype=row_dtype, mode='r', shape=(num_rows,))

        # Extract required columns using field names
        # Create copies to bring data into memory and ensure float32
        extracted_cols = []
        col_names_to_load = [f'f{i}' for i in cols_to_load]

        for col_name in col_names_to_load:
            # Access column via field name, convert to float32 and copy into RAM
            # Using copy() is crucial when the memmap will be closed.
            extracted_cols.append(mmap_array[col_name].astype(np.float32).copy())

        # Filter based on finiteness of all loaded columns *together*
        # Stack columns temporarily for efficient masking
        valid_cols_data = np.column_stack(extracted_cols)
        finite_mask = np.all(np.isfinite(valid_cols_data), axis=1)
        del valid_cols_data # Free temporary stack
        gc.collect()

        if not np.any(finite_mask):
             logger.warning(f"Memmap loader: No finite rows for requested columns in {filename}")
             # Clean up memmap object before returning
             del mmap_array
             gc.collect()
             return None

        # Apply mask to the list of extracted columns
        filtered_cols = [col[finite_mask] for col in extracted_cols]

        # Clean up memmap object explicitly (important!)
        del mmap_array
        gc.collect()

        return filtered_cols

    except Exception as e:
        logger.error(f"Memmap loader: Error processing {filename}: {e}")
        logger.error(traceback.format_exc())
        # Ensure memmap is cleaned up if partially created
        if 'mmap_array' in locals() and mmap_array._mmap is not None:
            del mmap_array
        gc.collect()
        return None

def load_partial_file_10(filename, dtype=np.float32):
    """
    Load the first 10 lines from a text file.

    Parameters
    ----------
    filename : str
        Path to the file to read
    dtype : numpy.dtype, optional
        Data type to use for the array, by default np.float32

    Returns
    -------
    numpy.ndarray or None
        Array containing the data from the first 10 lines of the file,
        or None if the file doesn't exist or contains no valid data

    Notes
    -----
    Useful for quickly previewing or analyzing the beginning of a large file
    without loading the entire dataset into memory. Only reads the first 10
    lines that contain valid data.
    """
    if not os.path.exists(filename):
        return None

    with open(filename, 'r') as f:
        lines = []
        for i, line in enumerate(f):
            if i >= 10:  # Only read 10 lines
                break
            lines.append(line)

    if not lines:
        return None

    # Convert the lines to a numpy array
    data = np.array([np.fromstring(line.strip(), sep=' ', dtype=dtype) for line in lines if line.strip()])
    return data

def safe_load_and_filter_10(filename, dtype=np.float32):
    """
    Load and filter the first 10 lines from a text file.

    Parameters
    ----------
    filename : str
        Path to the file to read
    dtype : numpy.dtype, optional
        Data type to use for the array, by default np.float32

    Returns
    -------
    numpy.ndarray or None
        Filtered array containing only valid data from the first 10 lines,
        or None if the file doesn't exist or contains no valid data

    Notes
    -----
    Similar to safe_load_and_filter but only reads the first 10 lines.
    Useful for analyzing the beginning of large files efficiently.
    """

    if not os.path.exists(filename):
        print(truncate_and_pad_string(f"WARNING: {filename} does not exist. Skipping."))
        return None

    data = load_partial_file_10(filename, dtype=dtype)
    gc.collect()

    if data is None or data.size == 0:
        print(truncate_and_pad_string(f"WARNING: {filename} had no valid data. Skipping."))
        return None

    # Filter out any invalid (NaN/Inf) rows
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]

    if data.size == 0:
        print(
            truncate_and_pad_string(f"WARNING: After filtering, {filename} had no valid rows. Skipping."))
        return None

    return data

def safe_load_and_filter_10_bin(filename, ncols, dtype=np.float32):
    """
    Load and filter the first 10 rows from a binary file.

    Parameters
    ----------
    filename : str
        Path to the binary file to read
    ncols : int
        Number of columns expected in the data
    dtype : numpy.dtype or list, optional
        Data type(s) to use for the array, by default np.float32

    Returns
    -------
    numpy.ndarray or None
        Filtered array containing only valid data from the first 10 rows,
        or None if the file doesn't exist or contains no valid data

    Notes
    -----
    Performs the following steps:

    1. Checks if file exists
    2. Loads entire binary file via load_partial_file_bin
    3. Slices off the first 10 rows (if available)
    4. Filters out invalid rows (NaN/Inf)
    5. Returns the filtered array, or None if empty/no valid rows

    Useful for analyzing the beginning of large binary files efficiently.
    """
    if not os.path.exists(filename):
        print(truncate_and_pad_string(f"WARNING: {filename} does not exist. Skipping."))
        return None

    data = load_partial_file_bin(filename, ncols=ncols, dtype=dtype)
    gc.collect()

    if data is None or data.size == 0:
        print(
            truncate_and_pad_string(f"WARNING: {filename} had no valid data (or is empty). Skipping."))
        return None

    # Keep only the first 10 rows (or fewer if file has <10 rows)
    max_needed = min(10, data.shape[0])
    data = data[:max_needed, :]

    # Filter out any invalid (NaN/Inf) rows
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]

    if data.size == 0:
        print(
            truncate_and_pad_string(f"WARNING: After filtering, {filename} had no valid rows. Skipping."))
        return None

    return data

def safe_load_particle_ids_bin(filename, ncols, particle_ids, dtype=np.float32):
    """
    Load data for specific particle IDs from a flat binary file.

    Uses efficient file seeking for basic numpy dtypes (e.g., np.float32)
    to read only requested rows. Falls back to loading the full file via
    `load_partial_file_bin` for complex/structured dtypes.

    Parameters
    ----------
    filename : str
        Path to the binary file (assumed flat row data).
    ncols : int
        Number of columns per row.
    particle_ids : array-like
        Array of particle IDs (0-based row indices) to extract.
    dtype : numpy.dtype or list, optional
        Data type of file contents. Basic numpy dtypes trigger optimized
        seek-based reading. Lists/tuples trigger a full read and fallback
        extraction. Default is np.float32.

    Returns
    -------
    numpy.ndarray or None
        Numpy array (float32) containing only the rows for the specified
        particle IDs (NaNs for missing/invalid IDs), or None if file
        doesn't exist or a fatal error occurs. Returns float32 array
        even if input dtype has different precision.
    """
    # Basic validation
    if not os.path.exists(filename):
        return None

    # Check if we should use the optimized path
    use_optimized_path = isinstance(dtype, type) and hasattr(dtype, 'itemsize')
    
    if not use_optimized_path:
         data = load_partial_file_bin(filename, ncols=ncols, dtype=dtype) # Existing full load
         if data is None: 
             return None
         # Convert structured to float32 if needed (logic from safe_load_and_filter_bin)
         if isinstance(dtype, (list, tuple)):
            nrows = data.shape[0]
            float_data = np.zeros((nrows, ncols), dtype=np.float32)
            for i in range(ncols):
                field_name = f"f{i}"
                float_data[:, i] = data[field_name].astype(np.float32, copy=False)
            data = float_data
         # Now extract rows using boolean indexing (less efficient than direct seek)
         num_particles = len(particle_ids)
         result_array = np.full((num_particles, ncols), np.nan, dtype=np.float32)
         # Need unique IDs and sort order for efficient lookup if using fallback
         unique_ids, original_indices = np.unique(particle_ids, return_inverse=True)
         valid_mask = (unique_ids >= 0) & (unique_ids < data.shape[0])
         found_ids = unique_ids[valid_mask]
         if len(found_ids) > 0:
            extracted_data = data[found_ids, :]
            # Map back to original particle_ids order
            mapping = {pid: i for i, pid in enumerate(found_ids)}
            original_mask_for_found = valid_mask[original_indices]
            result_indices = np.where(original_mask_for_found)[0]
            source_indices = [mapping[pid] for pid in np.array(particle_ids)[original_mask_for_found]]
            if len(result_indices) > 0:
                 result_array[result_indices] = extracted_data[source_indices]
         return result_array

    # --- Start: Optimized path for basic dtypes like float32 ---
    try:
        item_size = np.dtype(dtype).itemsize
        row_size = ncols * item_size
        file_size = os.path.getsize(filename)
        max_rows = file_size // row_size

        if row_size == 0 or max_rows == 0:
            return None

        num_requested = len(particle_ids)
        # Ensure result array is float32, regardless of input dtype's precision
        result_array = np.full((num_requested, ncols), np.nan, dtype=np.float32)
        found_count = 0

        with open(filename, 'rb') as f:
            for i, pid in enumerate(particle_ids):
                # Check if pid is a valid row index for this file
                if 0 <= pid < max_rows:
                    offset = pid * row_size
                    f.seek(offset)
                    row_bytes = f.read(row_size)
                    if len(row_bytes) == row_size:
                        # Convert bytes to numpy array of the specified dtype
                        row_data = np.frombuffer(row_bytes, dtype=dtype, count=ncols)
                        # Store as float32
                        result_array[i, :] = row_data.astype(np.float32, copy=False)
                        found_count += 1

        if found_count == 0:
            return None

        return result_array
    except Exception as e:
        return None
    finally:
        gc.collect()

def plot_density(r_values, rho_values, output_file=None):
    """
    Plot density profile as a function of radius and save to file.

    Parameters
    ----------
    r_values : array-like
        Radius values in kpc
    rho_values : array-like
        Density values (rho) at each radius
    output_file : str, optional
        Path to save the output plot, by default "density_profile.png"

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    Filters out any non-finite values before plotting.
    Creates a figure showing density as a function of radius with
    appropriate labels and grid.
    """
    r_values, rho_values = filter_finite_rows(r_values, rho_values)
    
    # Filter for positive values only for log scale
    positive_mask = (r_values > 0) & (rho_values > 0)
    if np.any(positive_mask):
        r_pos = r_values[positive_mask]
        rho_pos = rho_values[positive_mask]
        
        plt.figure(figsize=(10, 6))
        plt.loglog(r_pos, rho_pos, linewidth=2, label=r'$\rho(r)$')
        plt.xlabel(r'$r$ (kpc)', fontsize=12)
        plt.ylabel(r'$\rho(r)$ (M$_{\odot}$/kpc$^3$)', fontsize=12)
        plt.title(r'Radial Density Profile $\rho(r)$ (Log-Log)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        if output_file is None:
            output_file = f"results/density_profile{suffix}.png"
        plt.savefig(output_file, dpi=200)
        plt.close()
    else:
        # Fallback to empty plot if no valid data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No valid data for log-log plot', 
                 transform=plt.gca().transAxes, ha='center', va='center')
        plt.title(r'Radial Density Profile $\rho(r)$ (Log-Log)', fontsize=14)
        if output_file is None:
            output_file = f"results/density_profile{suffix}.png"
        plt.savefig(output_file, dpi=200)
        plt.close()

    
    log_plot_saved(output_file)
    return output_file

def plot_mass_enclosed(r_values, mass_values, output_file=None):
    """
    Plot enclosed mass as a function of radius and save to file.

    Parameters
    ----------
    r_values : array-like
        Radius values in kpc
    mass_values : array-like
        Enclosed mass values at each radius in Msun
    output_file : str, optional
        Path to save the output plot, by default "mass_enclosed.png"

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    Filters out any non-finite values before plotting. Creates a figure
    showing the mass enclosed within each radius with appropriate labels and grid.
    """
    r_values, mass_values = filter_finite_rows(r_values, mass_values)
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, mass_values, label=r'$M(r)$')
    plt.xlabel(r'$r$ (kpc)', fontsize=12)
    plt.ylabel(r'$M(r)$ (M$_\odot$)', fontsize=12)
    plt.title(r'Enclosed Mass Profile $M(r)$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if output_file is None:
        output_file = f"results/mass_enclosed{suffix}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()


    log_plot_saved(output_file)
    return output_file

def plot_psi(r_values, psi_values, output_file=None):
    """
    Plot gravitational potential (Psi) as a function of radius and save to file.

    Parameters
    ----------
    r_values : array-like
        Radius values in kpc
    psi_values : array-like
        Gravitational potential values at each radius (in km^2/s^2)
    output_file : str, optional
        Path to save the output plot, by default "psi_profile.png"

    Returns
    -------
    None
        Function does not return a value, but saves plot to the specified path

    Notes
    -----
    Filters out any non-finite values before plotting.
    Creates a figure showing gravitational potential as a function of radius
    with appropriate labels and grid. Assumes psi_values are already scaled.
    """

    r_values, psi_values = filter_finite_rows(r_values, psi_values)
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, psi_values, label=r'$\Psi(r)$')
    plt.xlabel(r'$r$ (kpc)', fontsize=12)
    plt.ylabel(r'$\Psi(r)$ (km$^2$/s$^2$)', fontsize=12)
    plt.title(r'Gravitational Potential Profile $\Psi(r)$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    if output_file is None:
        output_file = f"results/psi_profile{suffix}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_dpsi_dr(r_values, dpsi_values, output_file="dpsi_dr.png"):
    """
    Plot gravitational acceleration (dPsi/dr) as a function of radius and save to file.

    Parameters
    ----------
    r_values : array-like
        Radius values in kpc
    dpsi_values : array-like
        Gravitational acceleration values at each radius
    output_file : str, optional
        Path to save the output plot, by default "dpsi_dr.png"

    Returns
    -------
    None
        Function does not return a value, but saves plot to the specified path

    Notes
    -----
    Filters out any non-finite values before plotting.
    Creates a figure showing gravitational acceleration (dPsi/dr) as a
    function of radius with appropriate labels and grid.
    Units of dPsi/dr are typically (km/s)^2 / kpc.
    """

    r_values, dpsi_values = filter_finite_rows(r_values, dpsi_values)
    plt.figure(figsize=(10, 6))
    plt.plot(r_values, dpsi_values, label=r'$d\Psi/dr$')
    plt.xlabel(r'$r$ (kpc)', fontsize=12)
    plt.ylabel(r'$d\Psi/dr$ ((km/s)$^2$/kpc)', fontsize=12)
    plt.title(r'Gravitational Acceleration Profile $d\Psi/dr$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()

def plot_drho_dpsi(psi_values, drho_dpsi, output_file="drho_dpsi.png"):
    """
    Plot derivative of density with respect to potential and save to file.

    Parameters
    ----------
    psi_values : array-like
        Gravitational potential values (in km^2/s^2)
    drho_dpsi : array-like
        Derivative of density with respect to potential
    output_file : str, optional
        Path to save the output plot, by default "drho_dpsi.png"

    Returns
    -------
    None
        Function does not return a value, but saves plot to the specified path

    Notes
    -----
    Filters out any non-finite values before plotting.
    Creates a figure showing the derivative of density with respect to
    gravitational potential with appropriate labels and grid.
    """

    psi_values, drho_dpsi = filter_finite_rows(psi_values, drho_dpsi)

    # Filter for positive values only for log scale
    positive_mask = (psi_values > 0) & (drho_dpsi > 0)
    if np.any(positive_mask):
        psi_pos = psi_values[positive_mask]
        drho_dpsi_pos = drho_dpsi[positive_mask]

        plt.figure(figsize=(10, 6))
        plt.loglog(psi_pos, drho_dpsi_pos, label=r'$d\rho/d\Psi$')
        plt.xlabel(r'$\Psi$ (km$^2$/s$^2$)', fontsize=12)
        plt.ylabel(r'$d\rho/d\Psi$ ((M$_\odot$/kpc$^3$)/(km$^2$/s$^2$))', fontsize=12)
        plt.title(r'Density Derivative $d\rho/d\Psi$ (Log-Log)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()
    else:
        # Fallback to empty plot if no valid data
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, 'No valid data for log-log plot',
                 transform=plt.gca().transAxes, ha='center', va='center')
        plt.title(r'Density Derivative $d\rho/d\Psi$ (Log-Log)', fontsize=14)
        plt.savefig(output_file, dpi=150)
        plt.close()

def plot_f_of_E(E_values, f_values, output_file="f_of_E.png"):
    """
    Plot distribution function as a function of energy and save to file.

    Parameters
    ----------
    E_values : array-like
        Energy values
    f_values : array-like
        Distribution function values at each energy
    output_file : str, optional
        Path to save the output plot, by default "f_of_E.png"

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    Filters out any non-finite values before plotting.
    Creates a figure showing the distribution function (f(E)) as a
    function of energy with appropriate labels and grid.
    Uses log-log scale for better visualization.
    """

    E_values, f_values = filter_finite_rows(E_values, f_values)
    
    # Filter out zero or negative values for log-log plot
    mask = (E_values > 0) & (f_values > 0)
    E_values = E_values[mask]
    f_values = f_values[mask]
    
    if len(E_values) == 0:
        logger.warning("No positive values to plot for f_of_E")
        return output_file
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(E_values, f_values, linewidth=2, label=r'$f(\mathcal{E})$')
    ax.set_xlabel(r'$\mathcal{E}$ (km$^2$/s$^2$)', fontsize=12)
    ax.set_ylabel(r'$f(\mathcal{E})$ ((km/s)$^{-3}$ kpc$^{-3}$)', fontsize=12)
    ax.set_title(r'Distribution Function $f(\mathcal{E})$', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7, which='both')
    
    # Set log-log scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    
    log_plot_saved(output_file)
    return output_file

def plot_df_at_fixed_radius(v_values, df_values, r_fixed, output_file="df_fixed_radius.png"):
    """
    Plot the distribution function at a fixed radius.

    Parameters
    ----------
    v_values : array-like
        Velocity values
    df_values : array-like
        Distribution function values
    r_fixed : float or str
        The fixed radius value in kpc, or a string like "2r_s" for r_F = 2×scale_radius
    output_file : str, optional
        Path to save the output plot, by default "df_fixed_radius.png"

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    Filters out any non-finite values before plotting.
    Creates a figure showing the distribution function at a specific
    fixed radius as a function of velocity.
    """
    # Only log warnings for critical issues
    if np.allclose(df_values, 0.0):
        logger.warning("df_fixed_radius: All distribution function values are near zero")
        # Only if all values are exactly 0, set a small value to prevent empty plot
        if np.max(df_values) == 0:
            df_values = np.maximum(df_values, 1e-10)

    # Filter out non-finite values
    v_values, df_values = filter_finite_rows(v_values, df_values)

    # Check if arrays are empty after filtering
    if len(v_values) == 0 or len(df_values) == 0:
        logger.error("df_fixed_radius: No valid data points after filtering")
        return None

    # Convert velocity from simulation units (kpc/Myr) to km/s
    kmsec_to_kpcmyr = 1.02271e-3
    v_values_kms = v_values / kmsec_to_kpcmyr

    # Calculate speed distribution P(v|r=r_F) by normalizing
    # Integrate over velocity to get normalization factor
    # Use trapezoidal rule for integration
    if len(v_values_kms) > 1:
        # Sort by velocity for proper integration
        sort_indices = np.argsort(v_values_kms)
        v_sorted = v_values_kms[sort_indices]
        f_sorted = df_values[sort_indices]
        
        # Integrate using trapezoidal rule: ∫ f(v) dv
        normalization = np.trapezoid(f_sorted, v_sorted)
        
        if normalization > 0:
            # Normalize to get speed distribution P(v|r_F) with units km/s^(-1)
            speed_dist = df_values / normalization
        else:
            # Fallback if normalization fails
            speed_dist = df_values
            logger.warning("Normalization failed for speed distribution, using raw values")
    else:
        speed_dist = df_values
        logger.warning("Insufficient data points for normalization")

    # Use a simpler single-panel plot
    plt.figure(figsize=(10, 6))

    
    # Format the radius label based on whether it's numeric or string
    if isinstance(r_fixed, str):
        r_label = r_fixed.replace('r_s', r'r_s')  # Format for LaTeX
        plt.plot(v_values_kms, speed_dist, linewidth=2, label=r'$P(v|r_F = ' + r_label + r')$')
        plt.title(r'Speed Distribution at Fixed Radius ($r_F = ' + r_label + r'$)', fontsize=14)
    else:
        plt.plot(v_values_kms, speed_dist, linewidth=2, label=r'$P(v|r_F = ' + f'{r_fixed:.1f}' + r'$ kpc)')
        plt.title(r'Speed Distribution at Fixed Radius ($r_F = ' + f'{r_fixed:.1f}' + r'$ kpc)', fontsize=14)
    
    plt.xlabel(r'$v$ (km/s)', fontsize=12)
    plt.ylabel(r'$P(v|r_F)$ (km/s)$^{-1}$', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()

    
    plt.savefig(output_file, dpi=200)
    plt.close()

    
    log_plot_saved(output_file)
    return output_file

def plot_tangential_velocity_histogram(v_perp_data_initial, v_perp_data_final, output_file="tangential_velocity_histogram_compare.png", plot_range=None):
    """
    Plot a comparison of initial and final tangential velocity distributions.

    Parameters
    ----------
    v_perp_data_initial : array-like
        Array of initial tangential velocity magnitudes (km/s).
    v_perp_data_final : array-like
        Array of final tangential velocity magnitudes (km/s).
    output_file : str, optional
        Path to save the output plot.
    plot_range : tuple, optional
        Tuple (min, max) for the x-axis range of the histogram. Auto-ranged if None.

    Returns
    -------
    str
        Path to the saved plot file.
    """
    plt.figure(figsize=(10, 6))
    bins = 100  # Or another appropriate number

    # Filter NaN/Inf values independently for initial and final, as they come from different processing steps
    v_perp_data_initial = v_perp_data_initial[np.isfinite(v_perp_data_initial)]
    v_perp_data_final = v_perp_data_final[np.isfinite(v_perp_data_final)]

    if len(v_perp_data_initial) == 0 and len(v_perp_data_final) == 0:
        logger.warning(f"No valid data for tangential velocity histogram. Skipping plot: {output_file}")
        plt.close()
        return None

    # Determine common range if not specified, considering both datasets
    if plot_range is None:
        combined_data = np.array([])
        if len(v_perp_data_initial) > 0:
            combined_data = np.concatenate((combined_data, v_perp_data_initial))
        if len(v_perp_data_final) > 0:
            combined_data = np.concatenate((combined_data, v_perp_data_final))
        
        if len(combined_data) > 0:
            plot_range = (0, np.percentile(combined_data, 99.5)) # Start from 0, go to 99.5th percentile
            if plot_range[1] <= plot_range[0]: # Ensure max > min
                plot_range = (0, plot_range[1] + 1 if plot_range[1] > 0 else 1)
        else:
            plot_range = (0, 100) # Fallback range

    if len(v_perp_data_initial) > 0:
        plt.hist(v_perp_data_initial, bins=bins, range=plot_range, alpha=0.6, color='blue', label=r'Initial $v_{\perp}$', density=True)
    if len(v_perp_data_final) > 0:
        plt.hist(v_perp_data_final, bins=bins, range=plot_range, alpha=0.6, color='red', label=r'Final $v_{\perp}$', density=True)

    plt.xlabel(r'$v_{\perp}$ (km/s)', fontsize=12)
    plt.ylabel('Normalized Frequency', fontsize=12)
    plt.title(r'Comparison of Tangential Velocity Magnitude Distribution $N(v_{\perp})$', fontsize=14)
    if len(v_perp_data_initial) > 0 or len(v_perp_data_final) > 0: # Only add legend if there's data
        plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    
    # Note: log_plot_saved will be called by the calling function (process_variable_histograms)
    return output_file

def plot_combined_histogram_from_file(input_file, output_file, config=None):
    """
    Create an overlay histogram comparing initial and final radial distributions.

    Parameters
    ----------
    input_file : str
        Path to the input data file containing combined histogram data
    output_file : str
        Path to save the output plot
    config : Configuration, optional
        Configuration object containing plot settings. If provided,
        enables robust dynamic ranging for axes.

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    The input file should contain three columns:
    - bin_centers: The center values of histogram bins
    - initial counts: Counts in each bin for the initial distribution
    - final counts: Counts in each bin for the final distribution

    The plot shows three categories using different colors:
    - Overlap between initial and final (purple)
    - Initial > Final (blue)
    - Final > Initial (red)

    This visualization helps identify which parts of the distribution
    remained stable and which parts changed during the simulation.
    """
    data = safe_load_and_filter_bin(input_file, ncol_combined_histogram, dtype=[
                                    np.float32, np.int32, np.int32])
    if data is None:
        return None
    gc.collect()
    bin_centers = data[:, 0]
    hist_iradii = data[:, 1]
    hist_fradii = data[:, 2]

    overlap = np.minimum(hist_iradii, hist_fradii)
    initial_excess = hist_iradii - overlap
    final_excess = hist_fradii - overlap

    bin_width = bin_centers[1]-bin_centers[0] if len(bin_centers) > 1 else 1.0

    # Calculate robust range for X-axis if config enables it
    if config and hasattr(config, 'use_median_ranges') and config.use_median_ranges:
        # Reconstruct radius distribution from histogram data
        combined_counts = hist_iradii + hist_fradii
        r_min, r_max = _calculate_robust_range_from_histogram(
            bin_centers, combined_counts,
            percentile=config.x_percentile,
            percentile_multiplier=config.x_percentile_multiplier,
            min_abs_extent=config.min_x_range_abs,
            default_if_empty=(0, 250.0),
            can_be_negative=False,
            axis_name="combined radial histogram"
        )
        x_range = [r_min, r_max]
    else:
        x_range = None  # Auto-range

    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, overlap, width=bin_width,
            color='purple', label='Overlap', align='center')
    plt.bar(bin_centers, initial_excess, width=bin_width, bottom=overlap,
            color='blue', label='Initial > Final', align='center')
    plt.bar(bin_centers, final_excess, width=bin_width, bottom=overlap,
            color='red', label='Final > Initial', align='center')

    plt.xlabel(r'$r$ (kpc)', fontsize=12)
    plt.ylabel(r'$N(r)$', fontsize=12)
    plt.title(r'Comparison of Initial and Final Radial Distributions $N(r)$', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Apply robust ranging if calculated
    if x_range is not None:
        plt.xlim(x_range[0], x_range[1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    
    log_plot_saved(output_file)
    return output_file

def plot_trajectories(input_file, output_file):
    """
    Plot particle trajectories (radius vs time) for multiple particles.

    Parameters
    ----------
    input_file : str
        Path to the input data file containing trajectory data.
    output_file : str
        Path to save the output plot.

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    The input file should contain column groups with time and radial position
    for multiple particles over the simulation period.
    """
    data = safe_load_and_filter_bin(
        input_file, ncol_trajectories, dtype=np.float32)
    if data is None:
        return None
    gc.collect()
    time = data[:, 0]
    plt.figure(figsize=(10, 6))
    ncols = data.shape[1]
    nparticles = (ncols - 1)//3
    for p in range(nparticles):
        r_col = 1 + 3*p
        plt.plot(time, data[:, r_col], linewidth=1.5)
    plt.xlabel(r'$t$ (Myr)', fontsize=12)
    plt.ylabel(r'$r(t)$ (kpc)', fontsize=12)
    plt.title(r'Particle Trajectories', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    
    log_plot_saved(output_file)
    return output_file

def plot_single_trajectory(input_file, output_file):
    """
    Plot the orbital trajectory (radius vs time) for a single test particle.

    Parameters
    ----------
    input_file : str
        Path to the input data file containing trajectory data.
    output_file : str
        Path to save the output plot.

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    The input file should contain columns with time and radial position.
    """
    data = safe_load_and_filter_bin(
        input_file, ncol_single_trajectory, dtype=np.float32)
    if data is None:
        return None
    gc.collect()

    time = data[:, 0]
    radius = data[:, 1]
    plt.figure(figsize=(10, 6))
    plt.plot(time, radius, linewidth=2)
    plt.xlabel(r'$t$ (Myr)', fontsize=12)
    plt.ylabel(r'$r(t)$ (kpc)', fontsize=12)
    plt.title(r'Single Particle Trajectory', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    
    log_plot_saved(output_file)
    return output_file

def plot_energy_time(input_file, output_file):
    """
    Plot energy vs. time for multiple particles, illustrating integration stability.
    NOTE: 'Current' energy uses the initial static potential for evaluation.

    Parameters
    ----------
    input_file : str
        Path to the input data file containing energy data.
    output_file : str
        Path to save the output plot.

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    The input file should contain time in the first column, followed by repeated groups
    of energy values (current energy evaluated with static potential, initial energy)
    for each particle. Loads full data.
    """
    # Load full data using the binary safe loader
    data = safe_load_and_filter_bin(
        input_file, ncol_energy_and_angular_momentum_vs_time, dtype=np.float32)
    if data is None:
        # Log warning if data loading failed
        logger.warning(f"Failed to load or filter data from {input_file} for energy plot.")
        return None
    gc.collect()

    time = data[:, 0]
    ncols = data.shape[1]
    nparticles = (ncols - 1) // 4 # Calculate number of particles based on columns

    # Check if there are enough columns for at least one particle
    if nparticles <= 0:
        logger.warning(f"No particle data found in {input_file} after calculating columns.")
        return None

    fig, ax = plt.subplots(figsize=(10, 6)) # Get axis handle

    for p in range(nparticles):
        E_col = 1 + 4*p   # Column index for current energy (using static Psi)
        E_i_col = 2 + 4*p # Column index for initial energy

        # Ensure column indices are valid
        if E_i_col >= ncols:
            logger.warning(f"Column index {E_i_col} out of bounds for particle {p+1} in {input_file}.")
            continue # Skip this particle if indices are invalid

        # Plot current energy (using static Psi) and initial energy
        # Updated label to clarify E(t) uses static potential
        ax.plot(time, data[:, E_col], linewidth=1.5, label=fr'$\mathcal{{E}}(t)$ (Static $\Psi_0$) P{p+1}')
        ax.plot(time, data[:, E_i_col], '--', linewidth=1.5, label=fr'$\mathcal{{E}}_0$ P{p+1}')

    # Use LaTeX for axis labels
    ax.set_xlabel(r'$t$ (Myr)', fontsize=12)
    ax.set_ylabel(r'$\mathcal{E}$ (km$^2$/s$^2$)', fontsize=12)


    ax.set_title(r'Integration Stability: Energy Conservation Test', fontsize=14)

    # Add legend, adjust size and location as needed
    # Reduced font size slightly to avoid potential overlap with text box
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add explanatory text
    explanation = (
        "Tests numerical integration stability.\n"
        "Solid: Energy evaluated at current state (r, v, L) using the *initial static potential*.\n"
        "Dashed: Initial energy.\n"
        "Note: This is not true energy conservation if the potential evolves over time."
    )
    # Place text box in bottom left, slightly offset from axes
    ax.text(0.02, 0.02, explanation, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', fc='aliceblue', alpha=0.8))

    # Adjust layout slightly to prevent xlabel overlap with the text box
    plt.tight_layout(rect=[0, 0.05, 1, 1]) # Added bottom margin (rect=[left, bottom, right, top])
    plt.savefig(output_file, dpi=200)
    plt.close(fig) # Close the specific figure

    
    log_plot_saved(output_file)
    return output_file

def plot_angular_momentum_time(input_file, output_file):
    """
    Plot angular momentum vs. time for multiple particles, showing both current and initial values.

    Parameters
    ----------
    input_file : str
        Path to the input data file containing angular momentum data.
    output_file : str
        Path to save the output plot.

    Returns
    -------
    str
        Path to the saved plot file

    Notes
    -----
    The input file should contain time in the first column, followed by repeated groups
    of values (current energy, initial energy, current angular momentum, initial angular momentum)
    for each particle.
    Loads full data.
    """
    data = safe_load_and_filter_bin(
        input_file, ncol_energy_and_angular_momentum_vs_time, dtype=np.float32)
    if data is None:
        return None
    gc.collect()

    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    time = data[:, 0]
    ncols = data.shape[1]
    nparticles = (ncols - 1)//4
    plt.figure(figsize=(10, 6))
    for p in range(nparticles):
        L_col = 3 + 4*p
        L_i_col = 4 + 4*p
        # Use LaTeX in legend (using \ell)
        plt.plot(time, data[:, L_col], linewidth=1.5, label=fr'$\ell(t)$ P{p+1}')
        plt.plot(time, data[:, L_i_col], '--', linewidth=1.5, label=fr'$\ell_i$ P{p+1}')
    # Use LaTeX for axis labels
    plt.xlabel(r'$t$ (Myr)', fontsize=12)
    plt.ylabel(r'$\ell$ (kpc$\cdot$km/s)', fontsize=12)
    plt.title(r'Angular Momentum Conservation $\ell(t)$ vs $\ell_i$', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_file, dpi=200)
    plt.close()

    
    log_plot_saved(output_file)
    return output_file

def plot_lowestL_trajectories_3panel(input_file="lowest_l_trajectories.dat", output_file="lowestl_3panel.png", file_prefix="lowestl"):
    """
    Create a 3-panel plot showing trajectories of selected particles.

    Parameters
    ----------
    input_file : str, optional
        Path to the input data file containing trajectory data.
        Default is "lowest_l_trajectories.dat".
    output_file : str, optional
        Path to save the output plot. Default is "lowestl_3panel.png".
    file_prefix : str, optional
        Prefix to use for labeling ("lowestl" or "chosenl").

    Returns
    -------
    None

    Notes
    -----
    The input file should have columns in the format:
    time  r1  E1  L1   r2  E2  L2   ...   rN  EN  LN

    The function creates a 3-panel plot showing:
    1) Radii vs. time for each particle
    2) Energies vs. time for each particle
    3) Percentage deviation from average energy vs. time for each particle

    Label changes based on file type:
    - "Lowest" for lowest_l files
    - "Chosen" for chosen_l files
    """

    # Load the data
    data = safe_load_and_filter_bin(
        input_file, ncol_lowest_l_trajectories, dtype=np.float32)
    if data is None:
        return
    gc.collect()

    # Filter out rows with NaN/Inf
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]

    # time = first column
    time = data[:, 0]
    ncols = data.shape[1]
    # each lowest-L particle contributes 3 columns: (r, E, L)
    # so number of lowest-L particles = (ncols - 1)//3
    nlowest = (ncols - 1) // 3

    # Prepare the figure with 3 side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax_r = axes[0]
    ax_e = axes[1]
    ax_dev = axes[2]

    # Determine label text based on file prefix
    label_descriptor = "Chosen" if file_prefix == "chosenl" else "Lowest"

    # Extract initial L values from first timestep for legend
    initial_L_values = []
    for p in range(nlowest):
        L_col = 3 + 3*p  # L is the third of each triplet
        initial_L = data[0, L_col] if data.shape[0] > 0 else 0.0
        initial_L_values.append(initial_L)

    # 1) Radii vs. time
    for p in range(nlowest):
        r_col = 1 + 3*p  # r is the first of each triplet
        ax_r.plot(time, data[:, r_col], label=f"P{p+1} ($\\ell_0$={initial_L_values[p]:.3f})")
    ax_r.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_r.set_ylabel(r"$r$ (kpc)", fontsize=12)
    ax_r.set_title(rf"Radius vs. Time ({label_descriptor} $\ell$)", fontsize=14)
    ax_r.legend(fontsize=9)
    ax_r.grid(True)

    # 2) Energies vs. time
    #   second of each triplet is the energy
    for p in range(nlowest):
        e_col = 2 + 3*p
        ax_e.plot(time, data[:, e_col], label=f"P{p+1} ($\\ell_0$={initial_L_values[p]:.3f})")
    ax_e.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_e.set_ylabel(r"$\mathcal{E}$ (km$^2$/s$^2$)", fontsize=12)
    ax_e.set_title(rf"Energy vs. Time ({label_descriptor} $\ell$)", fontsize=14)
    ax_e.legend(fontsize=9)
    ax_e.grid(True)

    # 3) % deviation from average energy
    #   We'll build an array (nsteps x nlowest) of energies
    nsteps = data.shape[0]
    energies = np.zeros((nsteps, nlowest), dtype=np.float32)
    for p in range(nlowest):
        e_col = 2 + 3*p
        energies[:, p] = data[:, e_col]
        # compute each particle's time-averaged energy
    for p in range(nlowest):
        # average of that particle's energy over time
        single_mean = np.mean(energies[:, p])
        pct_dev = (energies[:, p] - single_mean) / single_mean * 100.0
        ax_dev.plot(time, pct_dev, label=f"P{p+1} ($\\ell_0$={initial_L_values[p]:.3f})")
    ax_dev.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_dev.set_ylabel(r"$\Delta\mathcal{E}/\langle\mathcal{E}\rangle$ (%)", fontsize=12)
    ax_dev.set_title(rf"Energy Deviation ({label_descriptor} $\ell$)", fontsize=14)
    ax_dev.legend(fontsize=9)
    ax_dev.grid(True)

    # Save & close figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    gc.collect()

def plot_debug_energy_compare(input_file, output_file="results/energy_compare.png"):
    """
    Create multi-panel diagnostic plot comparing energy calculation methods.

    Parameters
    ----------
    input_file : str
        Path to the input data file containing energy comparison data.
    output_file : str, optional
        Path to save the output plot. Default is "results/energy_compare.png".

    Returns
    -------
    None

    Notes
    -----
    Input file should have 8 columns:
    
    - 0: snapIdx - Snapshot index
    - 1: time(Myr) - Time in megayears
    - 2: radius(kpc) - Radius in kiloparsecs
    - 3: approxE - Approximated energy
    - 4: dynE - Dynamically calculated energy
    - 5: (dynE - approxE) - Difference between calculation methods
    - 6: KE - Kinetic Energy
    - 7: PE - Potential Energy

    Creates a 5-panel figure in a 2x3 layout:
    
    - Top row (3 panels):
      [0,0]: Radius vs. time
      [0,1]: Dynamic energy vs. time
      [0,2]: Energy deviation from mean (%) vs. time
    - Bottom row (2 panels + 1 blank):
      [1,0]: Kinetic energy vs. time
      [1,1]: Potential energy vs. time
      [1,2]: (empty placeholder)
    """
    data = safe_load_and_filter_bin(
        input_file, ncol_debug_energy_compare, dtype=np.float32)
    if data is None:
        print(
            truncate_and_pad_string(f"WARNING: {input_file} not found or no valid data. Skipping energy-compare plot."))
        return

    # Filter out any invalid (NaN/Inf) rows
    mask = np.all(np.isfinite(data), axis=1)
    data = data[mask]
    if data.size == 0:
        print(
            truncate_and_pad_string(f"Warning: After filtering, {input_file} had no valid rows. Skipping."))
        return

    # Extract data columns from the energy comparison file
    snap_idx = data[:, 0]   # Snapshot index
    time_myr = data[:, 1]   # Time in megayears
    radius_kpc = data[:, 2] # Radius in kiloparsecs
    approx_e = data[:, 3]   # Approximated energy
    dyn_e = data[:, 4]      # Dynamically calculated energy
    diff_e = data[:, 5]     # Difference between methods (dynE - approxE)
    ke_vals = data[:, 6]    # Kinetic energy
    pe_vals = data[:, 7]    # Potential energy

    # Create a 2-row by 3-column figure
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    # Top row subplots
    ax_r = axes[0, 0]  # radius vs time
    ax_e = axes[0, 1]  # dynE vs time
    ax_diff = axes[0, 2]  # energy deviation from mean

    # Bottom row subplots
    ax_ke = axes[1, 0]  # KE vs time
    ax_pe = axes[1, 1]  # PE vs time
    # The last panel [1,2] we turn off or leave blank
    axes[1, 2].axis('off')

    # 1) Radius vs Time
    ax_r.plot(time_myr, radius_kpc, 'b-', label=r"$r$")
    ax_r.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_r.set_ylabel(r"$r$ (kpc)", fontsize=12)
    ax_r.set_title(r"Radius vs. Time", fontsize=14)
    ax_r.grid(True)
    ax_r.legend(fontsize=10)

    # 2) dynE vs. Time (Using \mathcal{E}) - negated
    # Plot the negated energy values first to get auto-scaling
    neg_dyn_e = -dyn_e
    ax_e.plot(time_myr, neg_dyn_e, 'r-', label=r"$-\mathcal{E}_{\rm dyn}$")
    ax_e.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_e.set_ylabel(r"$-\mathcal{E}$ (km$^2$/s$^2$)", fontsize=12)
    ax_e.set_title(r"Energy vs. Time ($-\mathcal{E}_{\rm dyn}$)", fontsize=14)
    
    # Get the auto-scaled range after plotting
    epsilon_range_min, epsilon_range_max = ax_e.get_ylim()
    
    # Calculate appropriate variation scale based on potential energy variation
    psi_range = np.abs(np.max(pe_vals) - np.min(pe_vals))
    delta = psi_range / 3.0  # Characteristic energy variation scale
    neg_mean_e = np.mean(neg_dyn_e)  # Mean of negated energy
    
    # Calculate new limits that preserve both the auto-scaled range and our calculated range
    new_min = min(epsilon_range_min, neg_mean_e - 0.5*delta)
    new_max = max(epsilon_range_max, neg_mean_e + 0.5*delta)
    
    # Set the expanded y-axis limits
    ax_e.set_ylim(new_min, new_max)
    
    ax_e.grid(True)
    ax_e.legend(fontsize=10)

    # 3) Energy deviation from mean (Using \mathcal{E})
    # Calculate average deviation from time average
    mean_dyn = np.mean(dyn_e)

    # Prevent division by zero by using a minimal denominator value
    eps_d = mean_dyn if abs(mean_dyn) > 1e-30 else 1e-30

    dev_dyn_pct = 100.0 * (dyn_e - mean_dyn) / eps_d

    # Plot shows the deviation percentage without negation
    # as this represents relative change which is independent of sign
    ax_diff.plot(time_myr, dev_dyn_pct, 'r-',
                 label=r"$(\mathcal{E}_{\rm dyn} - \langle\mathcal{E}_{\rm dyn}\rangle)/\langle\mathcal{E}_{\rm dyn}\rangle$ (%)")
    ax_diff.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_diff.set_ylabel(r"$\Delta\mathcal{E}/\langle\mathcal{E}\rangle$ (%)", fontsize=12)
    ax_diff.set_title(r"Energy Deviation from Mean", fontsize=14)
    ax_diff.grid(True)
    ax_diff.legend(fontsize=10)

    # 4) Kinetic Energy vs. Time
    ax_ke.plot(time_myr, ke_vals, 'c-', label=r"$K$")
    ax_ke.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_ke.set_ylabel(r"$K$ (km$^2$/s$^2$)", fontsize=12)
    ax_ke.set_title(r"Kinetic Energy vs. Time", fontsize=14)
    ax_ke.grid(True)
    ax_ke.legend(fontsize=10)

    # 5) Potential Energy vs. Time (negated)
    ax_pe.plot(time_myr, -pe_vals, 'y-', label=r"$-\Psi$")
    ax_pe.set_xlabel(r"$t$ (Myr)", fontsize=12)
    ax_pe.set_ylabel(r"$-\Psi$ (km$^2$/s$^2$)", fontsize=12)
    ax_pe.set_title(r"Potential Energy vs. Time", fontsize=14)
    ax_pe.grid(True)
    ax_pe.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    gc.collect()

    # Log to file only, console output handled by progress tracker
    logger.info(f"Plot saved: {output_file}")

    # The calling function should use update_combined_progress instead
    # of this function directly updating the progress bar

class Configuration:
    """
    Configuration class that encapsulates all parameters and settings for the nsphere_plot.py script.
    This reduces the reliance on global variables and improves code maintainability and testability.
    """
    def __init__(self, args=None):
        """
        Initialize the configuration with command line arguments.

        Parameters
        ----------
        args : argparse.Namespace, optional
            The parsed command line arguments.
            If None, arguments will be parsed from sys.argv.
        """
        # Command line arguments
        self.args = args if args else self._parse_arguments()

        # Set parameters from arguments, with validation
        self._setup_parameters()

    def _parse_arguments(self):
        """
        Parse command line arguments for configuring the visualization options.

        Returns
        -------
        argparse.Namespace
            The parsed command line arguments with visualization settings.

        Notes
        -----
        This method creates an ArgumentParser with options for controlling which
        visualizations to generate and sets appropriate defaults based on lastparams.dat.
        """
        # Note: showing_help flag is already set at the beginning of the script

        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description='Generate plots and animations from simulation data.',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        # Get default suffix from lastparams.dat
        default_suffix = read_lastparams(return_suffix=True)

        parser.add_argument('--suffix', type=str, default=default_suffix,
                            help='Suffix for data files (e.g., _adp.leap.adp.levi_30000_1001_5)')
        parser.add_argument('--start', type=int, default=0,
                            help='Starting snapshot number')
        parser.add_argument('--end', type=int, default=0,
                            help='Ending snapshot number (0 means use all available)')
        parser.add_argument('--step', type=int, default=1,
                            help='Step size between snapshots')
        parser.add_argument('--fps', type=int, default=10,
                            help='Frames per second in the output animations')
        # Flags for only generating specific visualization groups
        parser.add_argument('--phase-space', action='store_true',
                            help='When used, ONLY generate phase space plots and animations')
        parser.add_argument('--phase-comparison', action='store_true',
                            help='When used, ONLY generate initial vs. final phase space comparison')
        parser.add_argument('--profile-plots', action='store_true',
                            help='When used, ONLY generate profile plots (density, mass, psi, etc.)')
        parser.add_argument('--trajectory-plots', action='store_true',
                            help='When used, ONLY generate trajectory and diagnostic plots')
        parser.add_argument('--2d-histograms', action='store_true',
                            help='When used, ONLY generate 2D histograms')
        parser.add_argument('--convergence-tests', action='store_true',
                            help='When used, ONLY generate convergence test plots')
        parser.add_argument('--animations', action='store_true',
                            help='When used, ONLY generate animations')
        parser.add_argument('--energy-plots', action='store_true',
                            help='When used, ONLY generate energy plots')

        # Flags for skipping specific visualization groups
        # Flag for generating 1D variable distributions
        parser.add_argument('--distributions', action='store_true',
                            help='When used, ONLY generate 1D variable distributions (Initial vs Final)')

        parser.add_argument('--no-phase-space', action='store_true',
                            help='Skip phase space plots and animations')
        parser.add_argument('--no-phase-comparison', action='store_true',
                            help='Skip initial vs. final phase space comparison')
        parser.add_argument('--no-profile-plots', action='store_true',
                            help='Skip profile plots (density, mass, psi, etc.)')
        parser.add_argument('--no-trajectory-plots', action='store_true',
                            help='Skip trajectory and diagnostic plots')
        parser.add_argument('--no-histograms', action='store_true',
                            help='Skip 2D histograms')
        parser.add_argument('--no-convergence-tests', action='store_true',
                            help='Skip convergence test plots')
        parser.add_argument('--no-animations', action='store_true',
                            help='Skip animations')
        parser.add_argument('--no-energy-plots', action='store_true',
                            help='Skip energy plots')
        parser.add_argument('--no-distributions', action='store_true',
                            help='Skip 1D variable distributions (Initial vs Final)')
        parser.add_argument('--log', action='store_true',
                            help='Enable detailed logging to file (default: only errors and warnings)')
        parser.add_argument('--paced', action='store_true',
                            help='Paced mode: enable delays between sections (default: fast mode with no delays)')

        return parser.parse_args()

    def _setup_parameters(self):
        """
        Configure parameters based on command line arguments and validate settings.

        This method initializes the following configuration properties:
        - suffix: File suffix for identifying data files
        - fps: Frames per second for animations
        - duration: Frame duration in milliseconds
        - start_snap, end_snap, step_snap: Snapshot filtering parameters

        It also handles global flags like enable_logging and paced_mode, and extracts
        additional parameters from the suffix (npts, Ntimes, tfinal_factor, file_tag).
        """
        global showing_help, enable_logging

        # Set the suffix for data files
        self.suffix = self.args.suffix

        # Calculate frame duration from fps
        self.fps = self.args.fps
        self.duration = 1000.0 / self.fps

        # Set the start, end, and step parameters for filtering Rank files
        self.start_snap = self.args.start
        self.end_snap = self.args.end
        self.step_snap = self.args.step

        # Handle logging flags (--log)
        global enable_logging, paced_mode
        if self.args.log:
            enable_logging = True

        # Handle paced mode flag (--paced)
        global section_delay, progress_delay
        if self.args.paced:
            paced_mode = True
            section_delay = 5.0  # Delay between different sections in seconds when paced mode is enabled
            progress_delay = 2.0  # Delay between progress bars in seconds when paced mode is enabled

        # Parameter display is centralized in the main function banner

        # Configure percentile-based robust ranging system
        self.use_median_ranges = True  # Enable by default (now uses percentiles)
        self.x_percentile = 95.0        # Use 95th percentile for X-axis
        self.y_percentile = 95.0        # Use 95th percentile for Y-axis
        self.x_percentile_multiplier = 1.2  # Multiplier for percentile value
        self.y_percentile_multiplier = 1.65  # Multiplier for percentile value
        self.min_x_range_abs = 10.0     # Minimum radius range extent (kpc)
        self.min_y_range_abs = 20.0     # Minimum velocity range extent (km/s)

        # Extract parameters from suffix for compatibility with existing code
        self._extract_params_from_suffix()

    def _extract_params_from_suffix(self):
        """
        Parse the suffix string to extract simulation parameters.

        This method parses the suffix pattern "_[file_tag]_npts_Ntimes_tfinal_factor"
        to extract the following parameters:
        - npts: Number of particles in the simulation
        - Ntimes: Number of time steps in the simulation
        - tfinal_factor: Final time factor of the simulation
        - file_tag: Optional tag identifying the simulation run

        If parsing fails, it falls back to reading from lastparams.dat.

        Notes
        -----
        These parameters are used throughout the code for file paths,
        plot labels, and determining output filenames.
        """
        # First, try to read parameters from the corresponding lastparams file
        # This is the most reliable source of truth for the parameters.
        try:
            self.npts, self.Ntimes, self.tfinal_factor, self.file_tag = read_lastparams(
                user_suffix=self.suffix, return_suffix=False
            )
            # Reconstruct the suffix from the authoritative source to ensure consistency
            if self.file_tag:
                self.suffix = f"_{self.file_tag}_{self.npts}_{self.Ntimes}_{self.tfinal_factor}"
            else:
                self.suffix = f"_{self.npts}_{self.Ntimes}_{self.tfinal_factor}"

            logger.info(f"Successfully loaded parameters for suffix '{self.suffix}' from its lastparams file.")

        except SystemExit:
            # Fallback to parsing the suffix string itself if lastparams<suffix>.dat doesn't exist
            logger.warning(f"Could not find lastparams file for suffix '{self.suffix}'. Falling back to parsing suffix string.")
            parts = self.suffix.strip('_').rsplit('_', 3)

            try:
                if len(parts) == 4: # Format: _tag_npts_ntimes_tfinal
                    self.file_tag, npts_str, ntimes_str, tfinal_str = parts
                elif len(parts) == 3: # Format: _npts_ntimes_tfinal
                    self.file_tag = ""
                    npts_str, ntimes_str, tfinal_str = parts
                else:
                    raise ValueError("Suffix does not match expected format.")

                self.npts = int(npts_str)
                self.Ntimes = int(ntimes_str)
                self.tfinal_factor = int(tfinal_str)
                logger.info(f"Parsed parameters from suffix string: npts={self.npts}, tag='{self.file_tag}'")
            except (ValueError, IndexError):
                print(f"Error: Suffix '{self.suffix}' is malformed and its lastparams file was not found.")
                sys.exit(1)

    def setup_file_paths(self):
        """
        Generate a dictionary mapping file types to their full file paths.
        All filenames now correctly include the suffix.
        """
        return {
            "particles": f"data/particles{self.suffix}.dat",
            "particlesfinal": f"data/particlesfinal{self.suffix}.dat",
            "integrand": f"data/integrand{self.suffix}.dat",
            "density_profile": f"data/density_profile{self.suffix}.dat",
            "massprofile": f"data/massprofile{self.suffix}.dat",
            "psiprofile": f"data/Psiprofile{self.suffix}.dat",
            "dpsi_dr": f"data/dpsi_dr{self.suffix}.dat",
            "drho_dpsi": f"data/drho_dpsi{self.suffix}.dat",
            "f_of_E": f"data/f_of_E{self.suffix}.dat",
            "df_fixed_radius": f"data/df_fixed_radius{self.suffix}.dat",
            "combined_histogram": f"data/combined_histogram{self.suffix}.dat",
            "trajectories": f"data/trajectories{self.suffix}.dat",
            "single_trajectory": f"data/single_trajectory{self.suffix}.dat",
            "energy_and_angular_momentum": f"data/energy_and_angular_momentum_vs_time{self.suffix}.dat",
            "hist_init": f"data/2d_hist_initial{self.suffix}.dat",
            "hist_final": f"data/2d_hist_final{self.suffix}.dat",
            "lowest_l_trajectories": f"data/lowest_l_trajectories{self.suffix}.dat",
            "debug_energy_compare": f"data/debug_energy_compare{self.suffix}.dat",
            "total_energy_vs_time": f"data/total_energy_vs_time{self.suffix}.dat",
            "lowest_radius_ids": f"data/lowest_radius_ids{self.suffix}.dat" # Added for energy plots
        }

    def only_specific_visualizations(self):
        """
        Determine if the user requested specific visualization types only.

        Returns
        -------
        bool
            True if any "only" flag is specified (e.g., --phase-space, --profile-plots),
            False if running in normal mode where all visualizations are generated.

        Notes
        -----
        When "only" flags are active, the script will generate only those specific
        visualization types and skip all others, regardless of --no-* flags.
        """
        return (self.args.phase_space or self.args.phase_comparison or
                self.args.profile_plots or self.args.trajectory_plots or
                getattr(self.args, '2d_histograms', False) or self.args.convergence_tests or
                self.args.animations or self.args.energy_plots or
                self.args.distributions)

    def need_to_process_rank_files(self):
        """
        Determine if the visualization plan requires processing rank data files.

        Returns
        -------
        bool
            True if rank files need to be processed for animations or energy plots,
            False if they can be skipped based on user-selected visualization options.

        Notes
        -----
        Rank files are needed for animations and energy plots. This method checks
        if either of these visualization types are specifically requested (via "only" flags)
        or if they haven't been explicitly excluded (via "no-" flags) in normal mode.
        """
        only_flags_active = self.only_specific_visualizations()

        return (self.args.animations or self.args.energy_plots or
                (not only_flags_active and (not self.args.no_animations or not self.args.no_energy_plots)))

def plot_total_energy_diagnostics(input_file, output_file="results/total_energy_diagnostics.png"):
    """
    Create a three-panel diagnostic plot for total system energy conservation.
    
    Panel 1: Absolute values of KE, PE, and Total Energy vs time
    Panel 2: Percent change from initial values for all three quantities (auto-scaled)
    Panel 3: Percent change of total energy only (zoomed in for detail)
    
    Parameters
    ----------
    input_file : str
        Path to the total_energy_vs_time binary file from nsphere
    output_file : str
        Path to save the output plot (PNG format)
    
    Returns
    -------
    str or None
        Path to the saved plot file, or None if plotting failed
    """
    import struct
    
    if not os.path.exists(input_file):
        logger.warning(f"Total energy file {input_file} does not exist")
        return None
    
    try:
        # Read the binary file
        with open(input_file, 'rb') as f:
            data = f.read()
        
        # The file is pure binary - fprintf_bin ignores the header string
        # because it has no format specifiers (%f, %g, etc.)
        # Data starts at byte 0: 5 floats per row (time, KE, PE, total_E, frac_change)
        
        # Parse binary data
        num_floats = len(data) // 4
        if num_floats % 5 != 0:
            logger.warning(f"Data size mismatch in {input_file}: {num_floats} floats, expected multiple of 5")
            # Try to proceed with what we have
            num_floats = (num_floats // 5) * 5
        
        values = struct.unpack(f'{num_floats}f', data[:num_floats*4])
        
        # Reshape into rows
        nrows = num_floats // 5
        if nrows == 0:
            logger.warning(f"No valid data rows in {input_file}")
            return None
        
        # Extract columns
        time = np.array([values[i*5] for i in range(nrows)])
        total_ke = np.array([values[i*5+1] for i in range(nrows)])
        total_pe = np.array([values[i*5+2] for i in range(nrows)])
        total_e = np.array([values[i*5+3] for i in range(nrows)])
        
        # Calculate mean values for percent change calculation
        ke_mean = np.mean(total_ke)
        pe_mean = np.mean(total_pe)
        e_mean = np.mean(total_e)
        
        # Calculate percent changes from mean
        ke_percent = 100.0 * (total_ke - ke_mean) / np.abs(ke_mean) if ke_mean != 0 else np.zeros_like(total_ke)
        pe_percent = 100.0 * (total_pe - pe_mean) / np.abs(pe_mean) if pe_mean != 0 else np.zeros_like(total_pe)
        e_percent = 100.0 * (total_e - e_mean) / np.abs(e_mean) if e_mean != 0 else np.zeros_like(total_e)
        
        # Create the three-panel figure
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Panel 1: True energy values (not absolute)
        ax1 = axes[0]
        ax1.plot(time, total_ke, 'r-', linewidth=2, label='Kinetic Energy', alpha=0.8)
        ax1.plot(time, total_pe, 'b-', linewidth=2, label='Potential Energy', alpha=0.8)
        ax1.plot(time, total_e, 'k-', linewidth=2.5, label='Total Energy')
        
        ax1.set_xlabel('Time (Myr)', fontsize=12)
        ax1.set_ylabel('Energy (code units)', fontsize=12)
        ax1.set_title('System Energy Evolution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        
        # Panel 2: Percent change of all quantities from mean (auto-scaled)
        ax2 = axes[1]
        ax2.plot(time, ke_percent, 'r-', linewidth=2, label='KE % from mean', alpha=0.8)
        ax2.plot(time, pe_percent, 'b-', linewidth=2, label='PE % from mean', alpha=0.8)
        ax2.plot(time, e_percent, 'k-', linewidth=2.5, label='Total E % from mean')
        
        ax2.set_xlabel('Time (Myr)', fontsize=12)
        ax2.set_ylabel('% Change from Mean', fontsize=12)
        ax2.set_title('Relative Energy Changes from Mean', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        # Panel 3: Total energy percent change from mean (zoomed)
        ax3 = axes[2]
        ax3.plot(time, e_percent, 'k-', linewidth=2.5, label='Total Energy % from mean')
        
        ax3.set_xlabel('Time (Myr)', fontsize=12)
        ax3.set_ylabel('% Change in Total Energy from Mean', fontsize=12)
        ax3.set_title('Total Energy Conservation Detail', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # Set y-axis limits for better visibility
        max_e_var = np.max(np.abs(e_percent))
        y_margin = max(0.001, max_e_var * 1.2)  # At least 0.001% margin
        ax3.set_ylim(-y_margin, y_margin)
        
        ax3.legend(loc='best', fontsize=9)
        
        # Overall figure adjustments
        fig.suptitle('Total System Energy Diagnostics', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # Save the plot
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.info(f"Total energy diagnostics plot saved to {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error creating total energy diagnostics plot: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def parse_arguments():
    """
    Parse command line arguments for the nsphere_plot.py script.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    config = Configuration()
    return config.args

def setup_global_parameters(args):
    """
    Setup global parameters based on command line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        The parsed command line arguments.

    Returns
    -------
    tuple
        The calculated parameters (suffix, start_snap, end_snap, step_snap, duration).
    """
    global suffix, start_snap, end_snap, step_snap, duration

    config = Configuration(args)

    suffix = config.suffix
    start_snap = config.start_snap
    end_snap = config.end_snap
    step_snap = config.step_snap
    duration = config.duration

    return suffix, start_snap, end_snap, step_snap, duration

def setup_file_paths(suffix):
    """
    Setup file paths for all data files using the given suffix.

    Parameters
    ----------
    suffix : str
        The suffix to use for all file names.

    Returns
    -------
    dict
        Dictionary containing all file paths.
    """
    temp_config = Configuration()
    temp_config.suffix = suffix

    return temp_config.setup_file_paths()

def process_profile_plots(file_paths, config=None):
    """
    Process various profile data files and generate plots.

    Parameters
    ----------
    file_paths : dict
        Dictionary of file paths for various data files.
    config : Configuration, optional
        Configuration object containing plot settings. If provided,
        enables robust dynamic ranging for plots.
    """
    global suffix

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Calculate total number of data files to load (for progress tracking)
    # Potential data files to load: density, mass, psi, dpsi, drho, f_of_E, df_fixed, combined
    total_data_files = 8

    # Start progress tracking for data loading
    start_combined_progress("profile_data_loading", total_data_files)

    # Track plots to be generated
    plots_to_generate = []
    output_files = []

    # First, prepare all the data that needs to be plotted
    prepared_plots = []

    # Plot density profile (now using rho, not 4*pi*r^2*rho)
    logger.info(f"Loading density profile data from {file_paths['density_profile']}")
    update_combined_progress("profile_data_loading", "loading_density_profile")
    dens_data = safe_load_and_filter_bin(
        file_paths["density_profile"], ncol_density_profile, dtype=np.float32)
    if dens_data is not None:
        gc.collect()
        mask = np.all(np.isfinite(dens_data), axis=1)
        dens_data = dens_data[mask]
        output_file = f"results/density_profile{suffix}.png"
        prepared_plots.append(("density", dens_data[:, 0], dens_data[:, 1], output_file))

    # Plot mass profile
    logger.info(f"Loading mass profile data from {file_paths['massprofile']}")
    update_combined_progress("profile_data_loading", "loading_mass_profile")
    mass_data = safe_load_and_filter_bin(
        file_paths["massprofile"], ncol_mass_profile, dtype=np.float32)
    if mass_data is not None:
        gc.collect()
        mask = np.all(np.isfinite(mass_data), axis=1)
        mass_data = mass_data[mask]
        output_file = f"results/mass_enclosed{suffix}.png"
        prepared_plots.append(("mass", mass_data[:, 0], mass_data[:, 1], output_file))

    # Plot psi profile
    logger.info(f"Loading psi profile data from {file_paths['psiprofile']}")
    update_combined_progress("profile_data_loading", "loading_psi_profile")
    psi_data = safe_load_and_filter_bin(
        file_paths["psiprofile"], ncol_psi_profile, dtype=np.float32)
    if psi_data is not None:
        gc.collect()
        mask = np.all(np.isfinite(psi_data), axis=1)
        psi_data = psi_data[mask]
        VEL_CONV_SQ = (1.02271e-3*1.02271e-3)
        psi_data_scaled = psi_data[:, 1]*VEL_CONV_SQ # Scale to km^2/s^2
        output_file = f"results/psi_profile{suffix}.png"
        prepared_plots.append(("psi", psi_data[:, 0], psi_data_scaled, output_file))

    # Plot dpsi/dr profile
    logger.info(f"Loading dpsi/dr profile data from {file_paths['dpsi_dr']}")
    update_combined_progress("profile_data_loading", "loading_dpsi_dr_profile")
    dpsi_data = safe_load_and_filter_bin(
        file_paths["dpsi_dr"], ncol_dpsi_dr, dtype=np.float32)
    if dpsi_data is not None:
        gc.collect()
        mask = np.all(np.isfinite(dpsi_data), axis=1)
        dpsi_data = dpsi_data[mask]
        output_file = f"results/dpsi_dr{suffix}.png"
        prepared_plots.append(("dpsi", dpsi_data[:, 0], dpsi_data[:, 1], output_file))

    # Plot drho/dpsi profile
    logger.info(f"Loading drho/dpsi profile data from {file_paths['drho_dpsi']}")
    update_combined_progress("profile_data_loading", "loading_drho_dpsi_profile")
    drho_data = safe_load_and_filter_bin(
        file_paths["drho_dpsi"], ncol_drho_dpsi, dtype=np.float32)
    if drho_data is not None:
        gc.collect()
        mask = np.all(np.isfinite(drho_data), axis=1)
        drho_data = drho_data[mask]
        output_file = f"results/drho_dpsi{suffix}.png"
        prepared_plots.append(("drho", drho_data[:, 0], drho_data[:, 1], output_file))

    # Plot f(E) profile
    logger.info(f"Loading f(E) profile data from {file_paths['f_of_E']}")
    update_combined_progress("profile_data_loading", "loading_f_of_E_profile")
    fE_data = safe_load_and_filter_bin(
        file_paths["f_of_E"], ncol_f_of_E, dtype=np.float32)
    if fE_data is not None:
        gc.collect()
        mask = np.all(np.isfinite(fE_data), axis=1)
        fE_data = fE_data[mask]
        output_file = f"results/f_of_E{suffix}.png"
        prepared_plots.append(("f_of_E", fE_data[:, 0], fE_data[:, 1], output_file))

    # Plot df at fixed radius
    logger.info(f"Loading df at fixed radius data from {file_paths['df_fixed_radius']}")
    update_combined_progress("profile_data_loading", "loading_df_fixed_radius_profile")

    # Load directly as float32 array - each row has two 4-byte floats
    try:
        # Load binary data
        with open(file_paths["df_fixed_radius"], 'rb') as f:
            raw_data = np.fromfile(f, dtype=np.float32)

        # Reshape into rows of 2 columns
        if len(raw_data) >= 2:  # Make sure we have at least one complete row
            rows = len(raw_data) // 2
            df_fixed_data = raw_data.reshape(rows, 2)
            logger.info(f"Successfully loaded df_fixed_radius data: {rows} rows of velocity/distribution data")
        else:
            logger.warning(f"Not enough data in {file_paths['df_fixed_radius']}")
            df_fixed_data = None

    except Exception as e:
        logger.error(f"Error loading df_fixed_radius data: {e}")
        # Fall back to standard loading method
        df_fixed_data = safe_load_and_filter_bin(
            file_paths["df_fixed_radius"], ncol_df_fixed_radius, dtype=np.float32)

    if df_fixed_data is not None:
        gc.collect()
        # Filter out any NaN or Inf values
        mask = np.all(np.isfinite(df_fixed_data), axis=1)
        df_fixed_data = df_fixed_data[mask]

        # Log detailed information about the data
        logger.info(f"df_fixed_radius data loaded successfully, shape: {df_fixed_data.shape}")
        if df_fixed_data.size > 0:
            logger.info(f"df_fixed_radius data range - v: [{np.min(df_fixed_data[:,0])}, {np.max(df_fixed_data[:,0])}], " +
                       f"f: [{np.min(df_fixed_data[:,1])}, {np.max(df_fixed_data[:,1])}]")
            logger.info(f"First 5 rows: {df_fixed_data[:5]}")

        output_file = f"results/df_fixed_radius{suffix}.png"
        prepared_plots.append(("df_fixed", df_fixed_data[:, 0], df_fixed_data[:, 1], output_file))

    # Combined histogram
    combined_histogram_file = file_paths['combined_histogram']
    logger.info(f"Loading combined histogram data from {combined_histogram_file}")
    update_combined_progress("profile_data_loading", "loading_combined_histogram")
    output_file = f"results/combined_radial_distribution{suffix}.png"
    prepared_plots.append(("combined", combined_histogram_file, None, output_file))

    # Now generate all the plots with progress tracking
    total_plots = len(prepared_plots)
    for i, (plot_type, x_data, y_data, output_file) in enumerate(prepared_plots, 1):
        try:
            if plot_type == "density":
                plot_density(x_data, y_data, output_file)
            elif plot_type == "mass":
                plot_mass_enclosed(x_data, y_data, output_file)
            elif plot_type == "psi":
                plot_psi(x_data, y_data, output_file)
            elif plot_type == "dpsi":
                plot_dpsi_dr(x_data, y_data, output_file)
            elif plot_type == "drho":
                plot_drho_dpsi(x_data, y_data, output_file)
            elif plot_type == "f_of_E":
                plot_f_of_E(x_data, y_data, output_file)
            elif plot_type == "df_fixed":
                # Use 2×scale_radius (the actual radius is determined by C code)
                plot_df_at_fixed_radius(x_data, y_data, "2r_s", output_file)
            elif plot_type == "combined":
                plot_combined_histogram_from_file(x_data, output_file, config)

            # Log to file and update console display with progress
            log_plot_saved(output_file, current=i, total=total_plots)

        except Exception as e:
            logger.error(f"Error generating {plot_type} plot: {str(e)}")
            print_status(f"Error generating {plot_type} plot: {str(e)}")
            continue

def process_trajectory_energy_plots(file_paths, include_angular_momentum=True):
    """
    Process energy plots that are part of the trajectory plots but should also
    be included in the energy plots category.

    Parameters
    ----------
    file_paths : dict
        Dictionary of file paths for various data files.
    include_angular_momentum : bool, optional
        Whether to include angular momentum plots. Default is True.
    """
    global suffix

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Prepare plots to be generated
    prepared_plots = []

    # Energy vs time plot
    energy_output_file = f"results/energy_vs_time{suffix}.png"
    prepared_plots.append(("energy", file_paths["energy_and_angular_momentum"], energy_output_file))

    # Angular momentum vs time (if requested)
    if include_angular_momentum:
        angular_output_file = f"results/angular_momentum_vs_time{suffix}.png"
        prepared_plots.append(("angular", file_paths["energy_and_angular_momentum"], angular_output_file))

    # Debug energy comparison (if available)
    debug_file = file_paths["debug_energy_compare"]
    if os.path.exists(debug_file):
        # Define output filename for the energy comparison plot
        debug_output_file = f"results/energy_compare{suffix}.png"
        prepared_plots.append(("debug", debug_file, debug_output_file))
    else:
        logger.warning(f"Energy compare file {debug_file} not found; skipping energy compare plot.")

    # Define a unique section key for this set of plots
    section_key = "additional_energy_plots"
    total_plots = len(prepared_plots)

    # Start combined progress tracking for these plots
    if total_plots > 0:
        start_combined_progress(section_key, total_plots)

        # Generate all plots with progress tracking
        for plot_type, input_file, output_file in prepared_plots:
            try:
                if plot_type == "energy":
                    plot_energy_time(input_file, output_file)
                elif plot_type == "angular":
                    plot_angular_momentum_time(input_file, output_file)
                elif plot_type == "debug":
                    plot_debug_energy_compare(input_file, output_file=output_file)

                # When called from trajectory_plots, update the combined progress tracker
                if "trajectory_plots" in _combined_plot_trackers:
                    update_combined_progress("trajectory_plots", output_file)
                else:
                    # Otherwise, use our section tracker
                    update_combined_progress(section_key, output_file)

            except Exception as e:
                logger.error(f"Error generating {plot_type} plot: {str(e)}")
                print_status(f"Error generating {plot_type} plot: {str(e)}")
                continue
    else:
        # If no plots to generate, just log and continue
        logger.info("No additional energy plots to generate.")

def process_trajectory_plots(file_paths):
    """
    Process trajectory and diagnostic plots.

    Parameters
    ----------
    file_paths : dict
        Dictionary of file paths for various data files.
    """
    global suffix

    # Create the results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Calculate total number of data files to load (for progress tracking)
    # Four different data files to potentially load: trajectories, single_trajectory, lowest_l_trajectories, energy_and_angular_momentum
    total_data_files = 4

    # Start progress tracking for data loading
    start_combined_progress("trajectory_data_loading", total_data_files)

    # Prepare plots to be generated
    prepared_plots = []

    # Trajectory plots - check and load
    logger.info(f"Loading trajectory data from {file_paths['trajectories']}")
    update_combined_progress("trajectory_data_loading", "loading_trajectories")
    if os.path.exists(file_paths["trajectories"]):
        trajectories_output = f"results/trajectories{suffix}.png"
        prepared_plots.append(("trajectories", file_paths["trajectories"], trajectories_output))
    else:
        logger.warning(f"Trajectory file {file_paths['trajectories']} not found, skipping.")

    # Single trajectory - check and load
    logger.info(f"Loading single trajectory data from {file_paths['single_trajectory']}")
    update_combined_progress("trajectory_data_loading", "loading_single_trajectory")
    if os.path.exists(file_paths["single_trajectory"]):
        single_trajectory_output = f"results/single_trajectory{suffix}.png"
        prepared_plots.append(("single", file_paths["single_trajectory"], single_trajectory_output))
    else:
        logger.warning(f"Single trajectory file {file_paths['single_trajectory']} not found, skipping.")

    # Lowest L or Chosen L 3-panel - check for both, prioritize chosen_l
    chosen_l_file = f"data/chosen_l_trajectories{suffix}.dat"
    lowest_l_file = file_paths["lowest_l_trajectories"]

    actual_input_file = None
    file_prefix_for_plot = "lowestl"

    if os.path.exists(chosen_l_file):
        actual_input_file = chosen_l_file
        file_prefix_for_plot = "chosenl"
        logger.info(f"Loading chosen L trajectories data from {chosen_l_file}")
    elif os.path.exists(lowest_l_file):
        actual_input_file = lowest_l_file
        file_prefix_for_plot = "lowestl"
        logger.info(f"Loading lowest L trajectories data from {lowest_l_file}")

    update_combined_progress("trajectory_data_loading", "loading_lowest_l_trajectories")

    if actual_input_file:
        output_filename = f"results/{file_prefix_for_plot}_3panel{suffix}.png"
        prepared_plots.append(("lowest_l", actual_input_file, output_filename, file_prefix_for_plot))
    else:
        logger.warning(f"Neither chosen_l nor lowest_l trajectories file found, skipping.")

    # Energy and angular momentum data - check and load
    logger.info(f"Loading energy and angular momentum data from {file_paths['energy_and_angular_momentum']}")
    update_combined_progress("trajectory_data_loading", "loading_energy_angular_momentum")

    # Generate trajectory plots using the combined progress tracker
    for item in prepared_plots:
        plot_type = item[0]
        input_file = item[1]
        output_file = item[2]
        prefix = item[3] if len(item) > 3 else None

        try:
            if plot_type == "trajectories":
                plot_trajectories(input_file, output_file)
            elif plot_type == "single":
                plot_single_trajectory(input_file, output_file)
            elif plot_type == "lowest_l":
                plot_lowestL_trajectories_3panel(input_file, output_file, file_prefix=prefix)

            # Update the combined progress tracker (shared with energy plots)
            update_combined_progress("trajectory_plots", output_file)

        except Exception as e:
            logger.error(f"Error generating {plot_type} plot: {str(e)}")
            print_status(f"Error generating {plot_type} plot: {str(e)}")
            continue

    # Also generate the energy plots from this category
    process_trajectory_energy_plots(file_paths)

# Global variables to store snapshot data for animation rendering
mass_snapshots = []
density_snapshots = []
psi_snapshots = []

# Global variables to store max values for consistent scaling across frames
mass_max_value = 0.0
density_max_value = 0.0
psi_max_value = 0.0

def calculate_global_max_values():
    """
    Calculate the maximum values of mass, density, and potential across all snapshots.
    This ensures consistent scaling in animations.
    
    Returns
    -------
    tuple
        (mass_max, density_max, psi_max) - Maximum values for each quantity
    """
    global mass_snapshots, density_snapshots, psi_snapshots
    global mass_max_value, density_max_value, psi_max_value
    
    # Find max mass value
    if mass_snapshots:
        mass_max_value = max(np.max(mass_data) for _, _, mass_data in mass_snapshots)
        # Add 5% margin
        mass_max_value *= 1.05
    
    # Find max density value (4*pi*r^2*rho)
    if density_snapshots:
        density_max_value = max(np.max(density_data) for _, _, density_data in density_snapshots)
        # Add 5% margin
        density_max_value *= 1.05
    
    # Find max psi value
    if psi_snapshots:
        psi_max_value = max(np.max(psi_data) for _, _, psi_data in psi_snapshots)
        # Add 5% margin
        psi_max_value *= 1.05
    
    logger.info(f"Global max values: Mass={mass_max_value:.3e}, Density={density_max_value:.3e}, Psi={psi_max_value:.3e}")
    
    return (mass_max_value, density_max_value, psi_max_value)

# Frame rendering functions for animation
def render_mass_frame(frame_data):
    """
    Render a single frame of mass profile animation.

    Parameters
    ----------
    frame_data : tuple 
        Tuple containing (snapshot_data, tfinal_factor, total_frames, r_range, m_range, project_root_path) where:
        - snapshot_data is the (snap, radius, mass) tuple from mass_snapshots
        - r_range is (r_min, r_max) tuple for x-axis
        - m_range is (m_min, m_max) tuple for y-axis

    Returns
    -------
    numpy.ndarray
        Image data for the rendered frame.
    """
    # No longer need global mass_max_value as we pass ranges directly

    # Unpack the passed data tuple
    snapshot_data, tfinal_factor, total_snapshots, r_range, m_range, project_root_path = frame_data
    # Unpack the actual data from the snapshot tuple
    snap, radius, mass = snapshot_data
    # Unpack the range tuples
    r_min, r_max = r_range
    m_min, m_max = m_range

    # Calculate time in dynamical times using the passed total_snapshots
    if total_snapshots and total_snapshots > 1:
        t_dyn_fraction = snap / (total_snapshots - 1) * tfinal_factor
    else:
        # Fallback if total_snapshots is 0 or 1
        t_dyn_fraction = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Mass Profile Evolution", fontsize=14, pad=20)
    ax.set_xlabel(r"$r$ (kpc)", fontsize=12)
    ax.set_ylabel(r"$M(r)$ (M$_\odot$)", fontsize=12)
    
    
    # Use the passed range values
    ax.set_xlim(r_min, r_max)
    
    # Use the passed mass range
    ax.set_ylim(m_min, m_max)
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.plot(radius, mass, lw=2)

    # Add text in upper right corner showing time in dynamical times
    # Regular black text on semi-transparent white background for mass profile
    ax.text(0.98, 0.95, f"$t = {t_dyn_fraction:.2f}\\,t_{{\\rm dyn}}$",
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    return img

def render_density_frame(frame_data):
    """
    Render a single frame of density profile animation.
    Note: Assumes density_snapshots contains 4*pi*r^2*rho.

    Parameters
    ----------
    frame_data : tuple
        Tuple containing (snapshot_data, tfinal_factor, total_frames, r_range, d_range, project_root_path) where
        snapshot_data is the (snap, radius, density) tuple from density_snapshots,
        r_range is (r_min, r_max) tuple for radius axis limits,
        d_range is (d_min, d_max) tuple for density axis limits.

    Returns
    -------
    numpy.ndarray
        Image data for the rendered frame.
    """
    # No longer need global density_max_value as it's passed in d_range

    # Unpack the passed data tuple
    snapshot_data, tfinal_factor, total_snapshots, r_range, d_range, project_root_path = frame_data
    # Unpack the actual data from the snapshot tuple
    snap, radius, density = snapshot_data # Assumed this is 4*pi*r^2*rho

    # Calculate time in dynamical times using the passed total_snapshots
    if total_snapshots and total_snapshots > 1:
        t_dyn_fraction = snap / (total_snapshots - 1) * tfinal_factor
    else:
        # Fallback if total_snapshots is 0 or 1
        t_dyn_fraction = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Density Profile Evolution", fontsize=14, pad=20)
    ax.set_xlabel(r"$r$ (kpc)", fontsize=12)
    ax.set_ylabel(r"$4\pi r^2 \rho(r)$ (M$_\odot$/kpc)", fontsize=12) # Label matches data
    
    
    # Use the passed range values
    ax.set_xlim(r_range[0], r_range[1])
    
    # Use the passed density range
    ax.set_ylim(d_range[0], d_range[1])
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    # Directly plot the 'density' variable, as it's assumed to be 4*pi*r^2*rho already
    ax.plot(radius, density, lw=2)

    # Add text in upper right corner showing time in dynamical times
    # Regular black text on semi-transparent white background for density profile
    ax.text(0.98, 0.95, f"$t = {t_dyn_fraction:.2f}\\,t_{{\\rm dyn}}$",
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    return img

def render_psi_frame(frame_data):
    """
    Render a single frame of psi profile animation.

    Parameters
    ----------
    frame_data : tuple
        Tuple containing (snapshot_data, tfinal_factor, total_frames, r_range, p_range, project_root_path) where
        snapshot_data is the (snap, radius, psi) tuple from psi_snapshots,
        r_range is (r_min, r_max) tuple for radius axis limits,
        p_range is (p_min, p_max) tuple for psi axis limits.

    Returns
    -------
    numpy.ndarray
        Image data for the rendered frame.
    """
    # No longer need global psi_max_value as it's passed in p_range

    # Unpack the passed data tuple
    snapshot_data, tfinal_factor, total_snapshots, r_range, p_range, project_root_path = frame_data
    # Unpack the actual data from the snapshot tuple
    snap, radius, psi = snapshot_data

    # Calculate time in dynamical times using the passed total_snapshots
    if total_snapshots and total_snapshots > 1:
        t_dyn_fraction = snap / (total_snapshots - 1) * tfinal_factor
    else:
        # Fallback if total_snapshots is 0 or 1
        t_dyn_fraction = 0.0

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Potential Profile Evolution", fontsize=14, pad=20)
    ax.set_xlabel(r"$r$ (kpc)", fontsize=12)
    ax.set_ylabel(r"$\Psi(r)$ (km$^2$/s$^2$)", fontsize=12)
    
    
    # Use the passed range values
    ax.set_xlim(r_range[0], r_range[1])
    
    # Use the passed psi range
    ax.set_ylim(p_range[0], p_range[1])
    
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    ax.plot(radius, psi, lw=2)

    # Add text in upper right corner showing time in dynamical times
    # Regular black text on semi-transparent white background for potential profile
    ax.text(0.98, 0.95, f"$t = {t_dyn_fraction:.2f}\\,t_{{\\rm dyn}}$",
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=3))

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    buf.close()
    return img

def process_particle_data(data, n_cols):
    """
    Extracts key quantities and calculates total velocity.
    
    Parameters
    ----------
    data : ndarray
        Particle data array
    n_cols : int
        Number of columns in the data
        
    Returns
    -------
    tuple
        (radius, radial_velocity, angular_momentum, total_velocity) for valid particles
    """
    # These use global constants defined at the top of the file
    
    if data is None or data.shape[0] == 0: 
        return None, None, None, None, None

    if n_cols == ncol_particles_initial:
        radii, radial_velocity, angular_momentum = data[:, 0], data[:, 1], data[:, 2]
    elif n_cols == ncol_Rank_Mass_Rad_VRad_unsorted:
        radii, radial_velocity, angular_momentum = data[:, 2], data[:, 3], data[:, 6]
    else:
        log_message(f"Error: Unsupported column count ({n_cols})", level="error")
        return None, None, None, None, None

    nonzero_mask = radii > 0
    if not np.any(nonzero_mask):
        log_message("Warning: No particles with positive radius found.", level="warning")
        return None, None, None, None, None

    radii_nz = radii[nonzero_mask]
    radial_velocity_nz = radial_velocity[nonzero_mask]
    angular_momentum_nz = angular_momentum[nonzero_mask]

    if np.any(radii_nz <= 0):
        log_message("Warning: Zero or negative radius found after mask?", level="warning")
        return None, None, None, None, None

    with np.errstate(divide='ignore', invalid='ignore'):
        tangential_velocity_sim = angular_momentum_nz / radii_nz
        total_velocity_sim = np.sqrt(tangential_velocity_sim**2 + radial_velocity_nz**2)
    
    kmsec_to_kpcmyr = 1.02271e-3
    total_velocity_kms = total_velocity_sim / kmsec_to_kpcmyr
    tangential_velocity_kms = tangential_velocity_sim / kmsec_to_kpcmyr # Convert v_perp to km/s

    valid_mask = np.isfinite(radii_nz) & np.isfinite(total_velocity_kms) & \
                 np.isfinite(radial_velocity_nz) & np.isfinite(angular_momentum_nz) & \
                 np.isfinite(tangential_velocity_kms) & \
                 (total_velocity_kms >= 0) & (radii_nz > 0) # v_total can be 0 if at rest

    if not np.any(valid_mask):
        log_message("Warning: No valid particles after filtering infinities/NaNs.", level="warning")
        return None, None, None, None, None

    final_radii = radii_nz[valid_mask]
    final_vr_kms = (radial_velocity_nz[valid_mask]) / kmsec_to_kpcmyr
    final_L = angular_momentum_nz[valid_mask]
    final_vtotal_kms = total_velocity_kms[valid_mask]
    final_vperp_kms = tangential_velocity_kms[valid_mask]

    log_message(f"Processed data: {len(final_radii)} valid particles.", level="debug")
    return final_radii, final_vr_kms, final_L, final_vtotal_kms, final_vperp_kms

def plot_particles_histograms(suffix, progress_callback=None, config=None):
    """
    Plots 2D histograms from the particles files.

    • Reads the initial particles file: data/particles{suffix}.dat (expected 4 columns)
      and plots a histogram with plt.hist2d.

    • Reads the final particles file: data/particlesfinal{suffix}.dat (expected 4 columns);
      computes a derived velocity value and then plots a 2D histogram.

    Resulting images are saved to the "results" folder.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
    progress_callback : callable, optional
        Function to call after each plot is saved,
        with the output file path as argument.
    config : dict, optional
        Configuration dictionary with plot settings
    """
    # Constants for expected number of columns
    ncol_particles_initial = 4
    ncol_particles_final = 4

    # Track plots we'll generate
    output_files = []

    # Start progress tracking for data loading (4 steps: check initial, load initial, check final, load final)
    start_combined_progress("particles_histograms_data_loading", 4)

    # ---------------------------------------
    # Plot the initial particles file histogram
    particles_file = f"data/particles{suffix}.dat"

    # Update progress - checking initial file
    update_combined_progress("particles_histograms_data_loading", "checking_initial_file")

    if os.path.exists(particles_file):
        # Use the appropriate column count for initial particles
        logger.info(f"Loading initial particles data from {particles_file}")

        # Update progress - loading initial data
        update_combined_progress("particles_histograms_data_loading", "loading_initial_data")

        data = safe_load_and_filter_bin(
            particles_file, ncol_particles_initial, np.float32)
        if data is None or data.shape[0] == 0:
            logger.warning(f"No valid data in {particles_file}")
            print_status(f"No valid data in {particles_file}")
        else:
            iradii = data[:, 0]
            radialvelocities = data[:, 1]
            ell = data[:, 2]
            # Avoid division by zero by filtering out non-positive radii.
            nonzero_mask = iradii > 0
            iradii = iradii[nonzero_mask]
            radialvelocities = radialvelocities[nonzero_mask]
            ell = ell[nonzero_mask]

            # Log data processing details
            logger.info(f"Processing initial particles data, shape before filtering: {data.shape}")
            logger.info(f"Non-zero radii: {np.sum(nonzero_mask)}/{len(nonzero_mask)}")

            # Record original data size for debugging
            global particles_original_count
            particles_original_count = data.shape[0]
            logger.info(f"Particles 2D initial histogram - Original data size: {particles_original_count}")

            # Log original array size before filtering
            logger.info(f"Original data array size (before filtering): {len(iradii)}")

            # Compute the derived (total) velocity as given by:
            # ivelocities = sqrt(ell^2 + (radialvelocities^2 * iradii^2)) / iradii / 1.02271e-3
            ivelocities = np.sqrt(ell**2 + (radialvelocities**2)
                                  * (iradii**2)) / iradii / 1.02271e-3
            # Filter out extreme values for visualization
            valid_mask = (ivelocities > 0) & (ivelocities < 1e8)
            iradii = iradii[valid_mask]
            ivelocities = ivelocities[valid_mask]

            logger.info(f"Valid velocities after filtering: {np.sum(valid_mask)}/{len(valid_mask)}")

            if len(iradii) == 0:
                logger.warning("All initial radii/velocities invalid or empty.")
                print_status("All initial radii/velocities invalid or empty.")
            else:
                # Calculate dynamic ranges using the robust range helper
                r_min, r_max = _calculate_robust_range(
                    iradii, percentile=config.x_percentile, 
                    percentile_multiplier=config.x_percentile_multiplier,
                    min_abs_extent=config.min_x_range_abs,
                    default_if_empty=(0, 250.0), can_be_negative=False,
                    axis_name="radius"
                )
                
                v_min, v_max = _calculate_robust_range(
                    ivelocities, percentile=config.y_percentile,
                    percentile_multiplier=config.y_percentile_multiplier,
                    min_abs_extent=config.min_y_range_abs, 
                    default_if_empty=(0, 320.0), can_be_negative=False,
                    axis_name="velocity"
                )
                
                plt.figure(figsize=(8, 6))
                hist_counts, _, _ = np.histogram2d(iradii, ivelocities, bins=250, 
                                                   range=[[r_min, r_max], [v_min, v_max]])
                
                # Calculate color scale limits based on statistics of non-zero bins
                nonzero_counts = hist_counts[hist_counts > 0]
                if len(nonzero_counts) > 0:
                    # Use 2.5x the 75th percentile for color scale maximum
                    vmax = np.percentile(nonzero_counts, 75) * 2.5
                    vmin = 0
                else:
                    vmin, vmax = 0, 1
                
                plt.hist2d(iradii, ivelocities, bins=250, range=[
                           [r_min, r_max], [v_min, v_max]], cmap='viridis', vmin=vmin, vmax=vmax)
                plt.colorbar(label='Counts')
                plt.xlabel(r'$r$ (kpc)', fontsize=12)
                plt.ylabel(r'$v$ (km/s)', fontsize=12)
                plt.title(r'Initial Phase Space Distribution', fontsize=14)
                plt.xlim(r_min, r_max)
                plt.ylim(v_min, v_max)
                output_file = f"results/2d_histogram_initial{suffix}.png"

                # Save original data size before any filtering
                original_size = data.shape[0]
                # Calculate and store histogram statistics
                hist_counts, _, _ = np.histogram2d(iradii, ivelocities, bins=250, range=[[r_min, r_max], [v_min, v_max]])
                initial_max = np.max(hist_counts)
                initial_nonzero = np.count_nonzero(hist_counts)
                initial_mean = np.sum(hist_counts)/initial_nonzero if initial_nonzero > 0 else 0
                initial_total_particles = len(iradii)
                initial_histogram_sum = np.sum(hist_counts)

                logger.info(f"Original particle count: {original_size}")
                logger.info(f"Filtered particle count: {initial_total_particles}")
                logger.info(f"Histogram sum: {initial_histogram_sum}")

                # Log detailed statistics
                logger.info(f"Initial histogram statistics: Max={initial_max}, Mean={initial_mean:.2f}, Non-Zero Bins={initial_nonzero}, Total Particles={initial_total_particles}")

                plt.savefig(output_file, dpi=150)
                plt.close()
                output_files.append(output_file)
    else:
        logger.warning(f"File {particles_file} not found.")
        print_status(f"File {particles_file} not found.")

    # ---------------------------------------
    # Plot final particles histogram
    particlesfinal_file = f"data/particlesfinal{suffix}.dat"

    # Update progress - checking final file
    update_combined_progress("particles_histograms_data_loading", "checking_final_file")

    if os.path.exists(particlesfinal_file):
        logger.info(f"Loading final particles data from {particlesfinal_file}")

        # Update progress - loading final data
        update_combined_progress("particles_histograms_data_loading", "loading_final_data")

        data = safe_load_and_filter_bin(
            particlesfinal_file, ncol_particles_final, np.float32)
        if data is None or data.shape[0] == 0:
            logger.warning(f"No valid data in {particlesfinal_file}")
            print_status(f"No valid data in {particlesfinal_file}")
        else:
            fradii = data[:, 0]
            radialvelocities = data[:, 1]
            ell = data[:, 2]

            # Log data processing details
            logger.info(f"Processing final particles data, shape before filtering: {data.shape}")

            # Record original data size for debugging
            global particles_final_original_count
            particles_final_original_count = data.shape[0]
            logger.info(f"Particles 2D final histogram - Original data size: {particles_final_original_count}")

            # Avoid division by zero by filtering out non-positive radii.
            nonzero_mask = fradii > 0
            fradii = fradii[nonzero_mask]
            radialvelocities = radialvelocities[nonzero_mask]
            ell = ell[nonzero_mask]

            logger.info(f"Non-zero radii: {np.sum(nonzero_mask)}/{len(nonzero_mask)}")

            # Log original array size before filtering
            logger.info(f"Original data array size (before filtering): {len(fradii)}")

            # Compute the derived velocity as given by:
            # fvelocities = sqrt(ell^2 + (radialvelocities^2 * fradii^2)) / fradii / 1.02271e-3
            fvelocities = np.sqrt(
                ell**2 + (radialvelocities**2) * (fradii**2)) / fradii / 1.02271e-3
            # Filter out extreme values for visualization
            valid_mask = (fvelocities > 0) & (fvelocities < 1e8)
            fradii = fradii[valid_mask]
            fvelocities = fvelocities[valid_mask]

            logger.info(f"Valid velocities after filtering: {np.sum(valid_mask)}/{len(valid_mask)}")

            if len(fradii) == 0:
                logger.warning("All final radii/velocities invalid or empty.")
                print_status("All final radii/velocities invalid or empty.")
            else:
                # Calculate dynamic ranges using the robust range helper
                r_min, r_max = _calculate_robust_range(
                    fradii, percentile=config.x_percentile,
                    percentile_multiplier=config.x_percentile_multiplier,
                    min_abs_extent=config.min_x_range_abs,
                    default_if_empty=(0, 250.0), can_be_negative=False,
                    axis_name="radius"
                )
                
                v_min, v_max = _calculate_robust_range(
                    fvelocities, percentile=config.y_percentile,
                    percentile_multiplier=config.y_percentile_multiplier,
                    min_abs_extent=config.min_y_range_abs, 
                    default_if_empty=(0, 320.0), can_be_negative=False,
                    axis_name="velocity"
                )
                
                plt.figure(figsize=(8, 6))
                hist_counts_temp, _, _ = np.histogram2d(fradii, fvelocities, bins=250, 
                                                        range=[[r_min, r_max], [v_min, v_max]])
                
                # Calculate color scale limits based on statistics of non-zero bins
                nonzero_counts_temp = hist_counts_temp[hist_counts_temp > 0]
                if len(nonzero_counts_temp) > 0:
                    # Use 2.5x the 75th percentile for color scale maximum
                    vmax_color = np.percentile(nonzero_counts_temp, 75) * 2.5
                    vmin_color = 0
                else:
                    vmin_color, vmax_color = 0, 1
                
                plt.hist2d(fradii, fvelocities, bins=250, range=[
                           [r_min, r_max], [v_min, v_max]], cmap='viridis', vmin=vmin_color, vmax=vmax_color)
                plt.colorbar(label='Counts')
                plt.xlabel(r'$r$ (kpc)', fontsize=12)
                plt.ylabel(r'$v$ (km/s)', fontsize=12)
                plt.title(r'Final Phase Space Distribution', fontsize=14)
                plt.xlim(r_min, r_max)
                plt.ylim(v_min, v_max)
                output_file = f"results/2d_histogram_final{suffix}.png"

                # Save original data size before any filtering
                original_size = data.shape[0]
                # Calculate and store histogram statistics
                hist_counts, _, _ = np.histogram2d(fradii, fvelocities, bins=250, range=[[r_min, r_max], [v_min, v_max]])
                final_max = np.max(hist_counts)
                final_nonzero = np.count_nonzero(hist_counts)
                final_mean = np.sum(hist_counts)/final_nonzero if final_nonzero > 0 else 0
                final_total_particles = len(fradii)
                final_histogram_sum = np.sum(hist_counts)

                logger.info(f"Original particle count: {original_size}")
                logger.info(f"Filtered particle count: {final_total_particles}")
                logger.info(f"Histogram sum: {final_histogram_sum}")

                # Log detailed statistics
                logger.info(f"Final histogram statistics: Max={final_max}, Mean={final_mean:.2f}, Non-Zero Bins={final_nonzero}, Total Particles={final_total_particles}")

                plt.savefig(output_file, dpi=150)
                plt.close()
                output_files.append(output_file)
    else:
        logger.warning(f"File {particlesfinal_file} not found.")
        print_status(f"File {particlesfinal_file} not found.")

    # Print histogram statistics to console for valid data
    if len(output_files) > 0:
        print_status("Particles histogram statistics:")

        # Display initial histogram statistics
        if 'initial_max' in locals() and 'initial_nonzero' in locals() and 'initial_mean' in locals():
            print_status(f"Initial Data: Max. Value = {initial_max}, Non-Zero Bins = {initial_nonzero}, Mean Value (Non-Zero) = {initial_mean:.2f}")
            if 'initial_total_particles' in locals() and 'initial_histogram_sum' in locals():
                # Only log the particle counts and histogram total to the log file
                if enable_logging:
                    logger.info(f"Initial Data: Original Size = {particles_original_count}, Particles = {initial_total_particles}, Histogram Total = {initial_histogram_sum:.0f}")
        else:
            # Find initial file if we don't have statistics
            initial_file = next((f for f in output_files if "initial" in f), None)
            if initial_file:
                print_status(f"Initial Data: Generated histogram saved to {initial_file}")

        # Display final histogram statistics
        if 'final_max' in locals() and 'final_nonzero' in locals() and 'final_mean' in locals():
            print_status(f"Final Data:   Max. Value = {final_max}, Non-Zero Bins = {final_nonzero}, Mean Value (Non-Zero) = {final_mean:.2f}")
            if 'final_total_particles' in locals() and 'final_histogram_sum' in locals():
                # Only log the particle counts and histogram total to the log file
                if enable_logging:
                    logger.info(f"Final Data: Original Size = {particles_final_original_count}, Particles = {final_total_particles}, Histogram Total = {final_histogram_sum:.0f}")
        else:
            # Find final file if we don't have statistics
            final_file = next((f for f in output_files if "final" in f), None)
            if final_file:
                print_status(f"Final Data:   Generated histogram saved to {final_file}")

        # Add separator line after statistics
        sys.stdout.write(get_separator_line(char='-') + "\n")

    # Once data loading is complete, start plot saving progress if needed
    if output_files:
        # Start a new progress tracker for saving plots
        start_combined_progress("particles_histograms_plots", len(output_files))

    # Log generated plots with progress indication
    if progress_callback:
        # Use progress callback for unified progress tracking
        for output_file in output_files:
            progress_callback(output_file)
    else:
        # Fallback to standard progress tracking
        for output_file in output_files:
            update_combined_progress("particles_histograms_plots", output_file)

def plot_nsphere_histograms(suffix, progress_callback=None, config=None):
    """
    Plots 2D histograms from the nsphere.c-produced binary histogram files.

    Expects each file to contain 3 columns and 160000 records (reshaped to 400 × 400).

    • data/2d_hist_initial{suffix}.dat for the initial histogram.
    • data/2d_hist_final{suffix}.dat for the final histogram.

    The plots are created using plt.pcolormesh and saved into the "results" folder.
    
    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
    progress_callback : callable, optional
        Function to call after each plot is saved,
        with the output file path as argument.
    config : dict, optional
        Configuration dictionary with plot settings
    """
    ncols_hist = 3
    hist_dtype = [np.float32, np.float32, np.int32]

    # Track plots we'll generate
    output_files = []

    # Start progress tracking for data loading (4 steps: check initial, load initial, check final, load final)
    start_combined_progress("nsphere_histograms_data_loading", 4)

    # ---------------------------------------
    # Plot nsphere initial histogram
    hist_initial_file = f"data/2d_hist_initial{suffix}.dat"

    # Update progress - checking initial file
    update_combined_progress("nsphere_histograms_data_loading", "checking_initial_file")

    if os.path.exists(hist_initial_file):
        logger.info(f"Loading initial histogram data from {hist_initial_file}")

        # Update progress - loading initial data
        update_combined_progress("nsphere_histograms_data_loading", "loading_initial_histogram")

        data_init = safe_load_and_filter_bin(
            hist_initial_file, ncols_hist, hist_dtype)
        if data_init is None:
            logger.warning(f"No valid data in {hist_initial_file}")
            print_status(f"No valid data in {hist_initial_file}")
        else:
            if data_init.shape[0] != 160000:
                logger.warning(
                    f"Expected 160000 entries in {hist_initial_file}, got {data_init.shape[0]}")
                print_status(
                    f"Warning: Expected 160000 entries in {hist_initial_file}, got {data_init.shape[0]}")
            else:
                try:
                    logger.info(f"Processing initial histogram data, shape: {data_init.shape}")
                    X_init = data_init[:, 0].reshape((400, 400))
                    Y_init = data_init[:, 1].reshape((400, 400))
                    C_init = data_init[:, 2].reshape((400, 400))

                    # Calculate statistics for histogram
                    hist_max = np.max(C_init)
                    hist_nonzero = np.count_nonzero(C_init)
                    hist_mean = np.sum(C_init)/hist_nonzero if hist_nonzero > 0 else 0
                    # Get total particles (sum of all counts)
                    hist_total_particles = np.sum(C_init)

                    # Record original data size for debugging
                    hist_original_particles = data_init.shape[0]
                    logger.info(f"NSphere initial histogram - Original data size: {hist_original_particles}, Histogram sum: {hist_total_particles:.0f}")

                    logger.info(f"Initial histogram statistics: Min={np.min(C_init)}, Max={hist_max}, Mean={hist_mean:.2f}, Non-Zero Bins={hist_nonzero}")

                    # Calculate robust ranges from histogram data if config available
                    if config and hasattr(config, 'use_median_ranges') and config.use_median_ranges:
                        # Get unique bin centers - the histogram is a meshgrid
                        r_centers = np.unique(X_init)  # Get unique radius values
                        v_centers = np.unique(Y_init)  # Get unique velocity values
                        
                        
                        # Sum histogram counts along axes to get 1D distributions
                        # From debug: X varies down rows (axis 0), Y varies across columns (axis 1)
                        # So C[i,j] represents counts at (r_centers[i], v_centers[j])
                        r_counts = np.sum(C_init, axis=1)  # Sum across columns (v) to get r distribution
                        v_counts = np.sum(C_init, axis=0)  # Sum across rows (r) to get v distribution
                        
                        # Calculate robust ranges
                        r_min, r_max = _calculate_robust_range_from_histogram(
                            r_centers, r_counts, 
                            percentile=config.x_percentile,
                            percentile_multiplier=config.x_percentile_multiplier,
                            min_abs_extent=config.min_x_range_abs,
                            default_if_empty=(0, 250.0), can_be_negative=False,
                            axis_name="radius (initial nsphere)"
                        )
                        
                        v_min, v_max = _calculate_robust_range_from_histogram(
                            v_centers, v_counts,
                            percentile=config.y_percentile,
                            percentile_multiplier=config.y_percentile_multiplier,
                            min_abs_extent=config.min_y_range_abs,
                            default_if_empty=(0, 320.0), can_be_negative=False,
                            axis_name="velocity (initial nsphere)"
                        )
                        
                        # For nsphere histograms, cap the range at the actual maximum populated bin
                        # to avoid showing excessive empty space from the fixed 320 km/s C code range
                        nonzero_v_indices = np.where(v_counts > 0)[0]
                        if len(nonzero_v_indices) > 0:
                            max_populated_v = v_centers[nonzero_v_indices[-1]]
                            v_max = min(v_max, max_populated_v * 1.05)  # Allow 5% padding above max populated bin
                    else:
                        # Use full data range
                        r_min, r_max = np.min(X_init), np.max(X_init)
                        v_min, v_max = np.min(Y_init), np.max(Y_init)
                    
                    # Find indices within the calculated range limits
                    r_vals = X_init[0, :]  # Radius values from first row
                    v_vals = Y_init[:, 0]  # Velocity values from first column
                    
                    # Find indices that fall within our calculated ranges
                    r_mask = (r_vals >= r_min) & (r_vals <= r_max)
                    v_mask = (v_vals >= v_min) & (v_vals <= v_max)
                    
                    # Trim the data to only include bins within range
                    X_trimmed = X_init[np.ix_(v_mask, r_mask)]
                    Y_trimmed = Y_init[np.ix_(v_mask, r_mask)]
                    C_trimmed = C_init[np.ix_(v_mask, r_mask)]
                    
                    plt.figure(figsize=(8, 6))
                    pcm = plt.pcolormesh(X_trimmed, Y_trimmed, C_trimmed,
                                         shading='auto', cmap='viridis')
                    cbar = plt.colorbar(pcm, label='Counts')
                    plt.xlabel(r'$r$ (kpc)', fontsize=12)
                    plt.ylabel(r'$v$ (km/s)', fontsize=12)
                    plt.title(r'Initial Phase Space Distribution (nsphere.c)', fontsize=14)
                    plt.xlim(r_min, r_max)
                    plt.ylim(v_min, v_max)
                    output_file = f"results/2d_hist_nsphere_initial{suffix}.png"
                    plt.savefig(output_file, dpi=150)
                    plt.close()
                    output_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error reshaping data from {hist_initial_file}: {e}")
                    print_status(f"Error reshaping data from {hist_initial_file}: {e}")
    else:
        logger.warning(f"File {hist_initial_file} not found.")
        print_status(f"File {hist_initial_file} not found.")

    # ---------------------------------------
    # Plot nsphere final histogram
    hist_final_file = f"data/2d_hist_final{suffix}.dat"

    # Update progress - checking final file
    update_combined_progress("nsphere_histograms_data_loading", "checking_final_file")

    if os.path.exists(hist_final_file):
        logger.info(f"Loading final histogram data from {hist_final_file}")

        # Update progress - loading final data
        update_combined_progress("nsphere_histograms_data_loading", "loading_final_histogram")

        data_final = safe_load_and_filter_bin(
            hist_final_file, ncols_hist, hist_dtype)
        if data_final is None:
            logger.warning(f"No valid data in {hist_final_file}")
            print_status(f"No valid data in {hist_final_file}")
        else:
            if data_final.shape[0] != 160000:
                logger.warning(
                    f"Expected 160000 entries in {hist_final_file}, got {data_final.shape[0]}")
                print_status(
                    f"Warning: Expected 160000 entries in {hist_final_file}, got {data_final.shape[0]}")
            else:
                try:
                    logger.info(f"Processing final histogram data, shape: {data_final.shape}")
                    X_final = data_final[:, 0].reshape((400, 400))
                    Y_final = data_final[:, 1].reshape((400, 400))
                    C_final = data_final[:, 2].reshape((400, 400))

                    # Calculate statistics for histogram
                    # Use different variable names for final histogram to avoid overwriting initial values
                    final_max = np.max(C_final)
                    final_nonzero = np.count_nonzero(C_final)
                    final_mean = np.sum(C_final)/final_nonzero if final_nonzero > 0 else 0
                    # Get total particles (sum of all counts)
                    final_total_particles = np.sum(C_final)

                    # Record original data size for debugging
                    final_original_particles = data_final.shape[0]
                    logger.info(f"NSphere final histogram - Original data size: {final_original_particles}, Histogram sum: {final_total_particles:.0f}")

                    logger.info(f"Final histogram statistics: Min={np.min(C_final)}, Max={final_max}, Mean={final_mean:.2f}, Non-Zero Bins={final_nonzero}")

                    # Calculate robust ranges from histogram data if config available
                    if config and hasattr(config, 'use_median_ranges') and config.use_median_ranges:
                        # Get unique bin centers - the histogram is a meshgrid
                        r_centers = np.unique(X_final)  # Get unique radius values
                        v_centers = np.unique(Y_final)  # Get unique velocity values
                        
                        
                        # Sum histogram counts along axes to get 1D distributions
                        # From debug: X varies down rows (axis 0), Y varies across columns (axis 1)
                        # So C[i,j] represents counts at (r_centers[i], v_centers[j])
                        r_counts = np.sum(C_final, axis=1)  # Sum across columns (v) to get r distribution
                        v_counts = np.sum(C_final, axis=0)  # Sum across rows (r) to get v distribution
                        
                        # Debug: Check if high-v particles are in C_final
                        total_in_2d = np.sum(C_final)
                        high_v_indices = np.where(v_centers > 210)[0]
                        particles_above_210_in_2d = np.sum(C_final[high_v_indices, :])
                        logger.info(f"NSphere final: Total particles in 2D histogram = {total_in_2d}")
                        logger.info(f"NSphere final: Particles with v>210 in 2D histogram = {particles_above_210_in_2d}")
                        
                        # Find the actual maximum populated velocity bin
                        nonzero_v_indices = np.where(v_counts > 0)[0]
                        if len(nonzero_v_indices) > 0:
                            max_populated_v_bin = nonzero_v_indices[-1]
                            max_populated_v = v_centers[max_populated_v_bin]
                            total_counts = np.sum(v_counts)
                            high_v_counts_200 = np.sum(v_counts[v_centers > 200])  # Count particles above 200 km/s
                            high_v_counts_210 = np.sum(v_counts[v_centers > 210])  # Count particles above 210 km/s
                            high_v_counts_250 = np.sum(v_counts[v_centers > 250])  # Count particles above 250 km/s
                            logger.info(f"NSphere final: Maximum populated velocity bin at {max_populated_v:.1f} km/s")
                            logger.info(f"NSphere final: Total particles = {total_counts}")
                            logger.info(f"NSphere final: particles > 200 km/s = {high_v_counts_200} ({100*high_v_counts_200/total_counts:.1f}%)")
                            logger.info(f"NSphere final: particles > 210 km/s = {high_v_counts_210} ({100*high_v_counts_210/total_counts:.1f}%)")
                            logger.info(f"NSphere final: particles > 250 km/s = {high_v_counts_250} ({100*high_v_counts_250/total_counts:.1f}%)")
                            
                            # Check WHERE the high-velocity particles are in radius
                            high_v_mask = v_centers > 210
                            # Check within displayed range (r < 120 kpc)
                            displayed_r_mask = r_centers < 120
                            high_v_displayed_count = 0
                            high_v_outside_count = 0
                            
                            for r_idx in range(len(r_centers)):
                                for v_idx in np.where(high_v_mask)[0]:
                                    if C_final[v_idx, r_idx] > 0:
                                        if displayed_r_mask[r_idx]:
                                            high_v_displayed_count += C_final[v_idx, r_idx]
                                        else:
                                            high_v_outside_count += C_final[v_idx, r_idx]
                            
                            logger.info(f"NSphere final: High-v particles (>210 km/s) within r<120 kpc = {high_v_displayed_count}")
                            logger.info(f"NSphere final: High-v particles (>210 km/s) at r>120 kpc = {high_v_outside_count}")
                            
                            # Find which radius bins contain the high-v particles
                            high_v_by_radius = []
                            for r_idx in range(len(r_centers)):
                                count_at_r = 0
                                for v_idx in np.where(high_v_mask)[0]:
                                    count_at_r += C_final[v_idx, r_idx]
                                if count_at_r > 0:
                                    high_v_by_radius.append((r_centers[r_idx], count_at_r))
                            
                            # Sort by count and show top 5
                            high_v_by_radius.sort(key=lambda x: x[1], reverse=True)
                            logger.info("NSphere final: Top 5 radius bins with high-v particles (>210 km/s):")
                            for i, (r, count) in enumerate(high_v_by_radius[:5]):
                                logger.info(f"  r={r:.1f} kpc: {count:.0f} particles")
                            
                            # Check velocity distribution at r~55 kpc
                            r_55_idx = np.argmin(np.abs(r_centers - 55.0))
                            v_dist_at_55 = C_final[:, r_55_idx]
                            nonzero_v_at_55 = v_centers[v_dist_at_55 > 0]
                            counts_at_55 = v_dist_at_55[v_dist_at_55 > 0]
                            
                            if len(nonzero_v_at_55) > 0:
                                # Reconstruct distribution for percentile calculation
                                v_values_at_55 = np.repeat(nonzero_v_at_55, counts_at_55.astype(int))
                                v_95_at_55 = np.percentile(v_values_at_55, 95)
                                v_max_at_55 = np.max(nonzero_v_at_55)
                                logger.info(f"NSphere final: At r={r_centers[r_55_idx]:.1f} kpc:")
                                logger.info(f"  Total particles: {np.sum(counts_at_55):.0f}")
                                logger.info(f"  95th percentile velocity: {v_95_at_55:.1f} km/s")
                                logger.info(f"  Maximum velocity: {v_max_at_55:.1f} km/s")
                        
                        # Calculate robust ranges
                        r_min, r_max = _calculate_robust_range_from_histogram(
                            r_centers, r_counts, 
                            percentile=config.x_percentile,
                            percentile_multiplier=config.x_percentile_multiplier,
                            min_abs_extent=config.min_x_range_abs,
                            default_if_empty=(0, 250.0), can_be_negative=False,
                            axis_name="radius (final nsphere)"
                        )
                        
                        v_min, v_max = _calculate_robust_range_from_histogram(
                            v_centers, v_counts,
                            percentile=config.y_percentile,
                            percentile_multiplier=config.y_percentile_multiplier,
                            min_abs_extent=config.min_y_range_abs,
                            default_if_empty=(0, 320.0), can_be_negative=False,
                            axis_name="velocity (final nsphere)"
                        )
                        
                        # For nsphere histograms, cap the range at the actual maximum populated bin
                        # to avoid showing excessive empty space from the fixed 320 km/s C code range
                        if len(nonzero_v_indices) > 0:
                            max_populated_v = v_centers[nonzero_v_indices[-1]]
                            v_max = min(v_max, max_populated_v * 1.05)  # Allow 5% padding above max populated bin
                        
                        logger.info(f"NSphere final histogram velocity range: [{v_min:.1f}, {v_max:.1f}] km/s")
                    else:
                        # Use full data range
                        r_min, r_max = np.min(X_final), np.max(X_final)
                        v_min, v_max = np.min(Y_final), np.max(Y_final)
                    
                    # Find indices within the calculated range limits
                    r_vals = X_final[0, :]  # Radius values from first row
                    v_vals = Y_final[:, 0]  # Velocity values from first column
                    
                    # Find indices that fall within our calculated ranges
                    r_mask = (r_vals >= r_min) & (r_vals <= r_max)
                    v_mask = (v_vals >= v_min) & (v_vals <= v_max)
                    
                    # Trim the data to only include bins within range
                    X_trimmed = X_final[np.ix_(v_mask, r_mask)]
                    Y_trimmed = Y_final[np.ix_(v_mask, r_mask)]
                    C_trimmed = C_final[np.ix_(v_mask, r_mask)]
                    
                    plt.figure(figsize=(8, 6))
                    pcm = plt.pcolormesh(X_trimmed, Y_trimmed, C_trimmed,
                                         shading='auto', cmap='viridis')
                    cbar = plt.colorbar(pcm, label='Counts')
                    plt.xlabel(r'$r$ (kpc)', fontsize=12)
                    plt.ylabel(r'$v$ (km/s)', fontsize=12)
                    plt.title(r'Final Phase Space Distribution (nsphere.c)', fontsize=14)
                    plt.xlim(r_min, r_max)
                    plt.ylim(v_min, v_max)
                    output_file = f"results/2d_hist_nsphere_final{suffix}.png"
                    plt.savefig(output_file, dpi=150)
                    plt.close()
                    output_files.append(output_file)
                except Exception as e:
                    logger.error(f"Error reshaping data from {hist_final_file}: {e}")
                    print_status(f"Error reshaping data from {hist_final_file}: {e}")
    else:
        logger.warning(f"File {hist_final_file} not found.")
        print_status(f"File {hist_final_file} not found.")

    # Print histogram statistics to console for valid data
    if 'hist_max' in locals() and 'hist_nonzero' in locals() and 'hist_mean' in locals():
        print_status("NSphere histogram statistics:")
        print_status(f"Initial Data: Max. Value = {hist_max}, Non-Zero Bins = {hist_nonzero}, Mean Value (Non-Zero) = {hist_mean:.2f}")
        # Only log the particle counts to the log file
        if enable_logging and 'hist_total_particles' in locals() and 'hist_original_particles' in locals():
            logger.info(f"NSphere Initial: Original Size = {hist_original_particles}, Total Particles = {hist_total_particles:.0f}")

        if 'final_max' in locals() and 'final_nonzero' in locals() and 'final_mean' in locals():
            print_status(f"Final Data:   Max. Value = {final_max}, Non-Zero Bins = {final_nonzero}, Mean Value (Non-Zero) = {final_mean:.2f}")
            # Only log the particle counts to the log file
            if enable_logging and 'final_total_particles' in locals() and 'final_original_particles' in locals():
                logger.info(f"NSphere Final: Original Size = {final_original_particles}, Total Particles = {final_total_particles:.0f}")

        # Add separator line after statistics
        sys.stdout.write(get_separator_line(char='-') + "\n")

    # The particles histogram statistics are now displayed in the main histogram statistics section

    # Once data loading is complete, start plot saving progress if needed
    if output_files:
        # Start a new progress tracker for saving plots
        start_combined_progress("nsphere_histograms_plots", len(output_files))

    # Log generated plots with progress indication
    if progress_callback:
        # Use progress callback for unified progress tracking
        for output_file in output_files:
            progress_callback(output_file)
    else:
        # Fallback to standard progress tracking
        for output_file in output_files:
            update_combined_progress("nsphere_histograms_plots", output_file)


# Storage for tracked particle IDs used in energy analysis
tracked_ids_for_energy = []

# Helper function to process each file with tracked particle IDs for energy-time plot
def process_sorted_energy_file(task_data):
    """
    Process an unsorted rank file for energy-time plot, extracting data for specific tracked particle IDs.
    
    Parameters
    ----------
    task_data : tuple
        A tuple containing (fname, local_suffix) where:
        - fname is the path to the unsorted Rank data file
        - local_suffix is the suffix for file identification

    Returns
    -------
    tuple or None
        (snapshot_number, energy_data) if successful or None if there was an error.
    """
    # Unpack arguments
    fname, local_suffix = task_data
    global ncol_Rank_Mass_Rad_VRad_unsorted

    # Use the suffix-aware helper
    snap = _extract_Rank_snapnum(fname, local_suffix)
    if snap == 999999999:
        logger.warning(f"Regex failed for sorted energy file {fname} with suffix '{local_suffix}'")
        return None

    # Reload tracked IDs inside the worker
    lowest_radius_ids_file = f"data/lowest_radius_ids{local_suffix}.dat"
    id_data = safe_load_and_filter_bin(lowest_radius_ids_file, ncols=2, dtype=[np.int32, np.float32])
    if id_data is None or len(id_data) < 1:
         logger.warning(f"Worker failed to load IDs from {lowest_radius_ids_file}")
         return None
    max_particles = min(10, id_data.shape[0])
    local_tracked_ids = id_data[:max_particles, 0].astype(int)

    # Use helper function to load data for specific particle IDs
    tracked_energy_data = safe_load_particle_ids_bin(
        fname, ncols=ncol_Rank_Mass_Rad_VRad_unsorted, particle_ids=local_tracked_ids, dtype=np.float32
    )

    if tracked_energy_data is not None and tracked_energy_data.shape[0] > 0:
        return (snap, tracked_energy_data)
    else:
        return None

# Helper function for phase space plotting
def phase_space_process_rank_file(fname):
    """
    Process a single sorted Rank file for phase space plotting.

    Extracts snapshot number and reads data using the appropriate
    data types and safe loader.

    Parameters
    ----------
    fname : str
        The filename of the sorted Rank file to process.

    Returns
    -------
    tuple or None
        (snapshot_number, data) if successful, otherwise None.
        'data' is a structured numpy array containing the file contents.
    """
    # Use the suffix-aware helper
    snap = _extract_Rank_snapnum(fname, suffix)
    if snap == 999999999:
        return None

    # Constants for the expected number of columns
    ncol_Rank_Mass_Rad_VRad_sorted = 8

    # Read the sorted file using safe loader
    data = safe_load_and_filter_bin(fname, ncol_Rank_Mass_Rad_VRad_sorted, dtype=[
        np.int32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32])

    if data is None:
        return None

    return (snap, data)

def process_unsorted_rank_file(task_data_with_suffix):
    """
    Process an unsorted rank file and return the snapshot number and energy values.
    
    Parameters
    ----------
    task_data_with_suffix : tuple
        A tuple containing (fname, project_root_path, local_suffix) where:
        - fname is the path to the unsorted Rank data file
        - project_root_path is the explicit path to the project root directory
        - local_suffix is the suffix for file identification

    Returns
    -------
    tuple or None
        (snapshot_number, energy_values) if successful or None if there was an error.
    """
    # Unpack arguments
    fname, project_root_path, local_suffix = task_data_with_suffix
    
    # Use the suffix-aware helper
    snap = _extract_Rank_snapnum(fname, local_suffix)
    if snap == 999999999:
        logger.warning(f"Regex failed for unsorted file {fname} with suffix '{local_suffix}'")
        return None

    # Constants for the expected number of columns
    ncol = ncol_Rank_Mass_Rad_VRad_unsorted # Should be 7

    # --- Start: Optimized Seek-Based Read Logic ---
    # Get IDs for the first 10 particles
    num_particles_to_get = 10
    particle_ids_to_get = list(range(num_particles_to_get))

    # Use safe_load_particle_ids_bin which uses seek for basic dtypes
    # Pass np.float32 to ensure it takes the optimized path
    data_subset = safe_load_particle_ids_bin(
        fname,
        ncols=ncol,
        particle_ids=particle_ids_to_get,
        dtype=np.float32 # IMPORTANT: Use basic dtype for seek path
    )

    if data_subset is None or data_subset.shape[0] == 0:
        return None

    # Extract Energy (column 5) from the loaded subset
    # Handle cases where fewer than 10 particles were actually found (rows might contain NaN)
    # We only want valid energy values
    valid_energy_mask = ~np.isnan(data_subset[:, 5])
    E_values = data_subset[valid_energy_mask, 5]

    if E_values.size == 0:
        return None

    # --- End: Optimized Seek-Based Read Logic ---

    return (snap, E_values)

def process_sorted_energy(fname):
    """
    Process a rank file for energy plots. This function extracts energy values for
    specific particles based on their IDs.

    Parameters
    ----------
    fname : str
        The filename to process.

    Returns
    -------
    tuple or None
        (snapshot_number, energy_values) if successful or None if there was an error.
    """
    # Use the suffix-aware helper
    snap = _extract_Rank_snapnum(fname, suffix)
    if snap == 999999999:
        return None

    # Load the particle IDs to track (particles with lowest initial radius)
    lowest_radius_ids_file = f"data/lowest_radius_ids{suffix}.dat"

    # Load the lowest_radius_ids file using the binary file handler
    id_data = safe_load_and_filter_bin(lowest_radius_ids_file, ncols=2, dtype=[np.int32, np.float32])
    if id_data is None or len(id_data) < 1:
        print_status(f"Failed to load particle IDs from {lowest_radius_ids_file}")
        return None

    # Extract the first 10 particles to track (or fewer if less available)
    max_particles = min(10, id_data.shape[0])
    tracked_ids = id_data[:max_particles, 0].astype(int)  # Convert to integers for use as indices

    # Now find the corresponding unsorted file for this snapshot
    unsorted_fname = f"data/Rank_Mass_Rad_VRad_unsorted_t{snap:05d}{suffix}.dat"
    if not os.path.exists(unsorted_fname):
        return None

    # Use helper function to load data for specific particle IDs
    tracked_energy_data = safe_load_particle_ids_bin(
        unsorted_fname, ncols=ncol_Rank_Mass_Rad_VRad_unsorted, particle_ids=tracked_ids, dtype=np.float32
    )

    if tracked_energy_data is None or tracked_energy_data.shape[0] == 0:
        return None

    # Extract energy values (column 5)
    energy_values = tracked_energy_data[:, 5]

    return (snap, energy_values)

def process_rank_file(fname):
    """
    Process a rank file and return the snapshot number and decimated data.

    Parameters
    ----------
    fname : str
        The filename to process.

    Returns
    -------
    tuple or None
        (snapshot_number, decimated_data) if successful or None if there was an error.
    """
    # Use the suffix-aware helper
    snap = _extract_Rank_snapnum(fname, suffix)
    if snap == 999999999:
        return None

    # Read the file using the safe loader
    data = safe_load_and_filter_bin(fname, ncol_Rank_Mass_Rad_VRad_sorted,
                                  dtype=[np.int32, np.float32, np.float32, np.float32,
                                         np.float32, np.float32, np.float32, np.float32])
    if data is None or data.shape[0] == 0:
        print_status(f"No data to process in {fname}")
        return None
    # Process the data
    decimated = data
    del data  # Free full-resolution data
    gc.collect()
    return (snap, decimated)

def process_rank_file_for_1d_anim(task_data_with_suffix):
    """
    Processes a single sorted Rank snapshot file for 1D animations.
    Now accepts suffix explicitly.

    Loads only the required columns (Mass, Radius, Psi, Density), sorts by
    radius, filters invalid data, fits linear splines to Mass, Psi and Density,
    performs downsampling to a common radius grid, and returns the processed data
    ready for animation.

    Parameters
    ----------
    task_data_with_suffix : tuple
        A tuple containing (fname, project_root_path, local_suffix) where:
        - fname is the path to the sorted Rank data file
        - project_root_path is the explicit path to the project root directory
        - local_suffix is the suffix for file identification

    Returns
    -------
    tuple or None
        On success, returns a tuple:
        (snap_num, sampled_radii, sampled_mass, sampled_density, sampled_psi)
        where arrays have the downsampled length.
        Returns None if loading, processing, or spline fitting fails.
    """
    # Unpack arguments including the suffix
    fname, project_root_path, local_suffix = task_data_with_suffix
    # No longer need: global suffix
    global ncol_Rank_Mass_Rad_VRad_sorted # Keep this global if needed elsewhere in function
    
    # Define the dtype list for the *entire* row of the sorted file
    sorted_rank_dtype_list = [np.int32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]

    # Columns needed: Radius (2, sort key), Mass (1), Psi (4), Density (7)
    # Indices passed to loader must be sorted for some selection methods
    cols_to_load_indices = sorted([1, 2, 4, 7])
    # Map back to original meaning after loading
    load_idx_map = {1: 0, 2: 1, 4: 2, 7: 3} # Original Col -> Index in loaded data
    radius_load_idx = load_idx_map[2] # Index of Radius within loaded data

    # Extract snapshot number from filename
    snap_num = _extract_Rank_snapnum(fname, local_suffix)
    if snap_num == 999999999:
        logger.warning(f"Could not extract valid snapshot number from {fname} using suffix '{local_suffix}' for 1D anim. Skipping.")
        return None

    try:
        # Load only the necessary columns
        loaded_cols = load_specific_columns_bin(
            fname,
            ncols_total=ncol_Rank_Mass_Rad_VRad_sorted, # Total cols in file (8)
            cols_to_load=cols_to_load_indices,
            dtype_list=sorted_rank_dtype_list
        )

        if loaded_cols is None or len(loaded_cols) != len(cols_to_load_indices):
            logger.warning(f"Failed to load required columns for snap {snap_num} from {fname}")
            return None

        # Combine into a temporary array for sorting/filtering
        # Order: Mass, Radius, Psi, Density (based on sorted cols_to_load_indices)
        temp_data = np.column_stack(loaded_cols)
        del loaded_cols # Free memory
        gc.collect()

        # Sort by Radius (which is at index radius_load_idx)
        sort_indices = np.argsort(temp_data[:, radius_load_idx])
        temp_data_sorted = temp_data[sort_indices, :]
        del temp_data, sort_indices # Free memory
        gc.collect()

        # Separate columns after sorting
        radius_sorted = temp_data_sorted[:, radius_load_idx]
        mass_sorted = temp_data_sorted[:, load_idx_map[1]]
        psi_sorted = temp_data_sorted[:, load_idx_map[4]]
        density_sorted = temp_data_sorted[:, load_idx_map[7]]
        del temp_data_sorted # Free memory
        gc.collect()

        # Filter out duplicates in radius (needed for spline) and non-finite values
        unique_radii, unique_indices = np.unique(radius_sorted, return_index=True)
        # Keep only data corresponding to unique radii
        radius_unique = unique_radii
        mass_unique = mass_sorted[unique_indices]
        psi_unique = psi_sorted[unique_indices]
        density_unique = density_sorted[unique_indices]

        # Filter for finite values in all arrays
        finite_mask = ( np.isfinite(radius_unique) &
                        np.isfinite(mass_unique) &
                        np.isfinite(psi_unique) &
                        np.isfinite(density_unique) )

        if not np.any(finite_mask):
            logger.warning(f"No finite data after filtering for snap {snap_num}")
            return None

        radius_final = radius_unique[finite_mask]
        mass_final = mass_unique[finite_mask]
        psi_final = psi_unique[finite_mask]
        density_final = density_unique[finite_mask]

        if len(radius_final) < 10: # Need enough points for spline
            logger.warning(f"Too few points ({len(radius_final)}) for spline fitting snap {snap_num}")
            return None

        # --- Spline Fitting ---
        try:
            # Use k=1 for linear interpolation, s=0 to force interpolation
            spl_mass = UnivariateSpline(radius_final, mass_final, k=1, s=0, ext='raise')
            spl_psi = UnivariateSpline(radius_final, psi_final, k=1, s=0, ext='raise')
            spl_density = UnivariateSpline(radius_final, density_final, k=1, s=0, ext='raise')
        except Exception as spline_e:
             logger.error(f"Spline fitting failed for snap {snap_num}: {spline_e}")
             return None


        # --- Downsampling ---
        r_min, r_max = radius_final.min(), radius_final.max()
        # Ensure min/max reasonable
        if r_min <= 0 or r_max <= r_min:
             logger.warning(f"Invalid radius range [{r_min}, {r_max}] for snap {snap_num}")
             return None

        n_particles_approx = len(radius_final) # Use length after filtering
        n_samples = max(int(n_particles_approx / 100), 10000) # Downsample factor 100, min 10k points

        sampled_radii = np.linspace(r_min, r_max, n_samples)

        # --- Evaluate Splines ---
        sampled_mass = spl_mass(sampled_radii)
        sampled_psi = spl_psi(sampled_radii)
        sampled_density = spl_density(sampled_radii)

        # Return the downsampled data
        return (snap_num, sampled_radii, sampled_mass, sampled_density, sampled_psi)

    except Exception as e:
        logger.error(f"Error processing {fname} for 1D anim: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        gc.collect() # Cleanup worker memory

def preprocess_phase_space_file(rank_file_data_with_suffix):
    """
    Preprocess a single Rank file for phase space animation.

    Parameters
    ----------
    rank_file_data_with_suffix : tuple
        Tuple containing (rank_file, placeholder, ncol, max_r_all, max_v_all, nbins, 
        kmsec_to_kpcmyr, project_root_path, local_suffix).

    Returns
    -------
    tuple or None
        (snap_num, H, frame_vmax) if successful or None if processing failed.
    """
    # Unpack arguments including the suffix
    rank_file, _, ncol, max_r_all, max_v_all, nbins, kmsec_to_kpcmyr, project_root_path, local_suffix = rank_file_data_with_suffix

    # Extract snapshot number for labeling
    snap_num = _extract_Rank_snapnum(rank_file, local_suffix)
    if snap_num == 999999999:
        logger.warning(f"Could not extract valid snapshot number from {rank_file} using suffix '{local_suffix}' for phase space. Skipping.")
        return None

    # Load the snapshot data
    data = safe_load_and_filter_bin(
        rank_file,
        ncol,
        dtype=[np.int32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]
    )

    if data is None or data.shape[0] == 0:
        return None

    # Extract radius and velocity data from sorted data with columns:
    # Rank(0), Mass(1), Radius(2), Vrad(3), Psi(4), Energy(5), L(6), Density(7)
    radii = data[:, 2]          # Column index 2 holds radius (r) - simulation units
    radial_velocity = data[:, 3]  # Column index 3 holds radial velocity (v_r) - simulation units
    angular_momentum = data[:, 6]  # Column index 6 holds angular momentum (L) - simulation units

    # Filter out non-positive radii to avoid division by zero
    nonzero_mask = radii > 0
    radii = radii[nonzero_mask]
    radial_velocity = radial_velocity[nonzero_mask]
    angular_momentum = angular_momentum[nonzero_mask]

    # Skip if not enough data points
    if len(radii) < 10:
        return None

    # Compute the total velocity - ALL IN SIMULATION UNITS
    # v_total_sim = sqrt((L/r)^2 + v_r^2)
    tangential_velocity_sim = angular_momentum / radii  # Tangential velocity in simulation units
    total_velocity_sim = np.sqrt(tangential_velocity_sim**2 + radial_velocity**2)  # Total velocity in simulation units

    # Convert to km/s at the end
    total_velocity = total_velocity_sim / kmsec_to_kpcmyr  # Convert to km/s

    # Filter out extreme values for visualization
    valid_mask = (total_velocity > 0) & (total_velocity < 1e8)
    radii = radii[valid_mask]
    total_velocity = total_velocity[valid_mask]

    # Skip if not enough valid data points
    if len(radii) < 10:
        return None

    # Create the 2D histogram
    H, xedges, yedges = np.histogram2d(
        radii,
        total_velocity,
        bins=[nbins, nbins],
        range=[[0, max_r_all], [0, max_v_all]]
    )

    # Calculate a reasonable vmax for the frame
    frame_vmax = calculate_reasonable_vmax(H)

    # Clean up memory before returning
    del radii, total_velocity, radial_velocity, angular_momentum, data
    gc.collect()

    return (snap_num, H, frame_vmax)

def generate_phase_space_animation(config, fps=10):
    """
    Creates an animation showing the evolution of the phase space distribution over time.
    
    Parameters
    ----------
    config : object
        Configuration object containing suffix and plot settings.
    fps : int, optional
        Frames per second for the animation, by default 10.
        
    Returns
    -------
    bool
        True if animation was created successfully, False otherwise.
        
    Notes
    -----
    Uses parallel processing for preprocessing histogram data and rendering frames.
    Saves the animation incrementally using `imageio.get_writer` to reduce
    peak memory usage compared to collecting all frames first.
    Requires imageio v2 (`pip install imageio==2.*`).
    """
    # Find all Rank sorted files with the exact suffix
    rank_files = glob.glob(f"data/Rank_Mass_Rad_VRad_sorted_t*{config.suffix}.dat")

    # Filter to keep ONLY files that match the exact pattern:
    # data/Rank_Mass_Rad_VRad_sorted_t00001_40000_1001_5.dat
    # without any extra characters between "t00001" and the suffix
    correct_pattern = re.compile(r'data/Rank_Mass_Rad_VRad_sorted_t\d+' + re.escape(suffix) + r'\.dat$')
    rank_files = [f for f in rank_files if correct_pattern.match(f)]

    # Debug output
    global enable_logging
    if enable_logging:
        log_message(f"Found {len(rank_files)} rank files with pattern: data/Rank_Mass_Rad_VRad_sorted_t*{config.suffix}.dat (after filtering)")

    if not rank_files:
        print_status("No Rank sorted files found. Cannot create phase space animation.")
        return False

    # Sort files by snapshot number to ensure correct animation sequence
    rank_files.sort(key=lambda fname: _extract_Rank_snapnum(fname, config.suffix))

    # Log the exact file count (to log file only)
    file_count = len(rank_files)
    log_message(f"Found {file_count} rank files for phase space animation.")

    # For debugging - list the first few files
    if file_count > 0:
        log_message(f"Example files (first 3): {', '.join(rank_files[:3])}")

    # Constants
    ncol_Rank_Mass_Rad_VRad_sorted = 8 # This remains a file structure constant
    kmsec_to_kpcmyr_local = kmsec_to_kpcmyr # Use the global kmsec_to_kpcmyr

    # Default histogram parameters (used if not median-based or if median calc fails)
    default_max_r_anim = 250.0
    default_max_v_anim = 320.0
    nbins = 200 # Keep nbins fixed for now

    # Initialize ranges to defaults
    current_plot_range_r_anim = (0, default_max_r_anim)
    current_plot_range_v_anim = (0, default_max_v_anim)

    if hasattr(config, 'use_median_ranges') and config.use_median_ranges:
        # Determine a subset of files for range calculation if too many
        # For example, take a sample or first/last N files.
        # Here, using all filtered rank_files, but could be refined.
        files_for_range_calc = rank_files 
        if len(files_for_range_calc) > 50: # Heuristic: if more than 50 files, sample ~20-30%
             sample_size = max(20, int(len(files_for_range_calc) * 0.3))
             files_for_range_calc = sorted(np.random.choice(files_for_range_calc, size=sample_size, replace=False))


        r_range_tuple, v_range_tuple = _calculate_global_animation_ranges(
            rank_files_subset=files_for_range_calc,
            r_min_abs_extent=config.min_x_range_abs,
            v_min_abs_extent=config.min_y_range_abs,
            r_default_if_empty=(0, default_max_r_anim),
            v_default_if_empty=(0, default_max_v_anim),
            kmsec_to_kpcmyr=kmsec_to_kpcmyr_local,
            current_suffix_for_logging=config.suffix,
            axis_name_prefix="phase_anim_global",
            config=config
        )
        current_plot_range_r_anim = r_range_tuple
        current_plot_range_v_anim = v_range_tuple
        logger.info(f"Phase Anim Global Ranges: R={current_plot_range_r_anim}, V={current_plot_range_v_anim}")
    else:
        logger.info(f"Phase Anim Global Ranges: Using fixed R={(0, default_max_r_anim)}, V={(0, default_max_v_anim)}")
        # Defaults current_plot_range_r_anim and current_plot_range_v_anim are already set

    # Prepare data for parallel preprocessing
    # Log to file only, keep console output minimal
    log_message(f"Processing {len(rank_files)} files for phase space animation...")

    # Prepare data for parallel preprocessing, ensuring suffix is explicitly passed
    preprocess_args_with_suffix = [(
        rank_file,
        None,  # Placeholder
        ncol_Rank_Mass_Rad_VRad_sorted,
        current_plot_range_r_anim[1], # Pass determined max R for hist range
        current_plot_range_v_anim[1], # Pass determined max V for hist range
        nbins,
        kmsec_to_kpcmyr_local, # Use local var
        PROJECT_ROOT,
        config.suffix # Use config.suffix
    ) for rank_file in rank_files]

    # Process all files in parallel
    with mp.Pool() as pool:
        # Use a custom tqdm instance with two progress displays
        # First, create a standard tqdm for the counter on the first line
        # Determine description and format dynamically
        try:
            term_width = shutil.get_terminal_size().columns
        except OSError:
            term_width = 100 # Fallback width
        selected_desc, selected_bar_format = select_tqdm_format('preproc_phase', term_width)
        
        counter_tqdm = tqdm(
            total=len(rank_files),
            desc=selected_desc, # Use dynamic description
            unit="file",
            position=0,
            leave=True,
            miniters=1,
            dynamic_ncols=True,
            ncols=None,
            bar_format=selected_bar_format
        )


        bar_tqdm = tqdm(
            total=len(rank_files),
            position=1,
            leave=True,
            dynamic_ncols=True,
            bar_format="{bar} {percentage:3.1f}%",
            ascii=False
        )

        # Process the files with a custom callback to update both progress bars
        results = []
        # Use the modified arguments list
        for result in pool.imap(preprocess_phase_space_file, preprocess_args_with_suffix):
            results.append(result)
            counter_tqdm.update(1)
            bar_tqdm.update(1)


        counter_tqdm.close()
        bar_tqdm.close()

        # Add delay after progress bars if enabled
        if progress_delay > 0 and paced_mode:
            show_progress_delay(progress_delay)

    # Log to file only
    log_message(f"Processed {len([r for r in results if r is not None])} phase space files successfully")

    # Filter out None results and calculate global vmax
    frame_data_list = [r for r in results if r is not None]
    if not frame_data_list:
        print_status("No valid frames found. Cannot create phase space animation.")
        return False

    # Find the global maximum value for consistent color scaling
    initial_vmax = max(vmax for _, _, vmax in frame_data_list)

    # Extract tfinal_factor from suffix if possible
    tfinal_factor = 5  # Default value
    parts = config.suffix.strip('_').split('_') # Use config.suffix
    if len(parts) >= 3:
        try:
            tfinal_factor = int(parts[-1])
        except (ValueError, IndexError):
            pass

    total_frames = len(frame_data_list)
    # Augment frame_data_list to include the determined plot ranges for rendering
    frame_data_list = [(
        snap_num, H, initial_vmax, # initial_vmax is for color scale
        tfinal_factor, total_frames,
        current_plot_range_r_anim,    # Pass tuple (min_r, max_r)
        current_plot_range_v_anim     # Pass tuple (min_v, max_v)
    ) for snap_num, H, _ in frame_data_list] # Assuming frame_data_list was populated from preprocess_phase_space_file results

    # Render frames in parallel
    # Still set the global for backward compatibility, though workers won't use it
    global total_snapshots
    total_snapshots = total_frames
    # Log detailed info to file only
    log_message(f"Generating phase space frames for {total_frames} snapshots...")

    # Set up imageio writer
    phase_anim_output = f"results/Phase_Space_Animation{config.suffix}.gif"
    # Use seconds per frame for imageio v2 duration
    frame_duration_sec_v2 = 1.0 / fps  # fps is frames per second, duration is seconds per frame
    try:
        # Use mode='I' for multiple images, loop=0 for infinite loop
        writer = imageio.get_writer(
            phase_anim_output,
            format='GIF-PIL',        # Explicitly use Pillow
            mode='I',
            # quantizer='nq',          # Temporarily removed
            palettesize=256,         # Ensure full palette
            duration=frame_duration_sec_v2,
            loop=0
        )
    except Exception as e:
        print_status(f"Error creating GIF writer: {e}")
        return False # Cannot proceed without writer

    # Use multiple processes to render frames
    with mp.Pool() as pool:
        # Use a custom tqdm instance with two progress displays
        # First, create a standard tqdm for the counter on the first line
        # Determine description and format dynamically
        try:
            term_width = shutil.get_terminal_size().columns
        except OSError:
            term_width = 100 # Fallback width
        selected_desc, selected_bar_format = select_tqdm_format('render_phase', term_width)
        
        counter_tqdm = tqdm(
            total=total_frames,
            desc=selected_desc, # Use dynamic description
            unit="frame",
            position=0,
            leave=True,
            miniters=1,
            dynamic_ncols=True,
            ncols=None,
            bar_format=selected_bar_format
        )


        bar_tqdm = tqdm(
            total=total_frames,
            position=1,
            leave=True,
            dynamic_ncols=True,
            bar_format="{bar} {percentage:3.1f}%",
            ascii=False
        )

        # Process the frames with a custom callback to update both progress bars
        frame_count = 0
        for frame_image in pool.imap(render_phase_frame, frame_data_list):
            if frame_image is not None:
                try:
                    writer.append_data(frame_image) # Append to writer
                    frame_count += 1
                except Exception as e:
                    log_message(f"Error appending frame {frame_count+1}: {e}", level="error")
                    # Decide whether to break or continue
                    break
            # Update progress bars
            counter_tqdm.update(1)
            bar_tqdm.update(1)


        counter_tqdm.close()
        bar_tqdm.close()

        # Close the writer
        try:
            writer.close()
        except Exception as e:
            log_message(f"Error closing GIF writer: {e}", level="error")

        # Add delay after progress bars if enabled
        if progress_delay > 0 and paced_mode:
            show_progress_delay(progress_delay)

    # Log to file only
    log_message(f"Generated and saved {frame_count}/{total_frames} phase space frames successfully to {phase_anim_output}")
    sys.stdout.write(get_separator_line(char='-') + "\n")

    if frame_count == total_frames:
        print_status(f"Animation saved: {get_file_prefix(phase_anim_output)}")
        # Clean up
        del frame_data_list
        gc.collect()
        return True
    else:
        print_status(f"Animation partially saved ({frame_count}/{total_frames}): {get_file_prefix(phase_anim_output)}")
        # Clean up
        del frame_data_list
        gc.collect()
        return False

def generate_all_1D_animations(suffix, duration, config=None):
    """
    Generate all three 1D profile animations (mass, density, psi) with optimized parallel processing.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
    duration : float
        Duration of each frame in milliseconds.
    config : object, optional
        Configuration object containing plot settings. If provided and has
        use_median_ranges=True, will use robust ranging for axes.

    Notes
    -----
    This function runs each animation in sequence to ensure clean console output,
    but each animation internally uses parallel processing for generating and encoding
    frames. This gets the best performance while maintaining readable output.
    """
    # Generate 1D profile animations sequentially for clean console output
    # Each animation creation function internally uses parallel processing for frames
    animations = [
        ("mass", create_mass_animation),
        ("density", create_density_animation),
        ("psi", create_psi_animation)
    ]

    # Track animation success
    success_count = 0

    # Process each animation in sequence
    for name, create_func in animations:
        try:
            log_message(f"Starting {name} animation generation")
            create_func(suffix, duration, config)
            success_count += 1
            log_message(f"Completed {name} animation successfully")
        except Exception as e:
            log_message(f"Error generating {name} animation: {str(e)}", level="error")
            print_status(f"Error generating {name} animation: {str(e)}")
            # Continue with the next animation despite errors

    # Return success status
    return success_count == len(animations)

def _extract_Rank_snapnum(fname, suffix):
    """
    Extract snapshot number from a rank file name.
    Now handles both sorted and unsorted file patterns.
    """
    # Try sorted pattern first
    pattern_sorted = r'Rank_Mass_Rad_VRad_sorted_t(\d+)' + re.escape(suffix) + r'\.dat$'
    mo = re.search(pattern_sorted, fname)
    if mo:
        return int(mo.group(1))

    # Fallback to unsorted pattern
    pattern_unsorted = r'Rank_Mass_Rad_VRad_unsorted_t(\d+)' + re.escape(suffix) + r'\.dat$'
    mo = re.search(pattern_unsorted, fname)
    if mo:
        return int(mo.group(1))

    return 999999999

def calculate_reasonable_vmax(H):
    """
    Calculate an appropriate maximum value for colorbar scaling based on histogram data.

    Parameters
    ----------
    H : ndarray
        The 2D histogram data array.

    Returns
    -------
    float
        A reasonable maximum value for the colorbar, based on the 99.5th percentile
        of non-zero values, rounded to a visually pleasing number.

    Notes
    -----
    This function helps avoid having too much empty space in the colorbar while still
    maintaining a consistent scale across different frames in animations.
    """
    # Get non-zero values from the histogram
    nonzero_values = H[H > 0]

    if len(nonzero_values) == 0:
        return 100  # Default if no non-zero values

    # Calculate the 99.5th percentile of non-zero values
    vmax = np.percentile(nonzero_values, 99.5)

    # Round up to a nice number
    if vmax < 10:
        vmax = 10
    elif vmax < 100:
        vmax = np.ceil(vmax / 10) * 10
    else:
        vmax = np.ceil(vmax / 50) * 50

    return vmax

def render_phase_frame(frame_data):
    """
    Renders a single frame for the phase space animation.
    This function needs to be at the module level for multiprocessing to work.

    Parameters
    ----------
    frame_data : tuple
        A tuple containing (snap_num, H, vmax, tfinal_factor, total_snapshots)

        snap_num : int
            The snapshot number for labeling.
        H : ndarray
            The 2D histogram data.
        vmax : float
            The maximum value for the colorbar (consistent across all frames).
        tfinal_factor : int
            Factor relating simulation time steps to dynamical times.
        total_snapshots : int
            Total number of frames/snapshots for calculating normalized time.

    Returns
    -------
    numpy.ndarray
        Image data for the rendered frame.
    """
    snap_num, H, vmax_colorscale, tfinal_factor, total_snapshots, plot_range_r_tuple, plot_range_v_tuple = frame_data

    # Set up histogram parameters using passed full ranges
    nbins = H.shape[0]  # H was binned using these ranges in preprocess

    # Recreate the meshgrid for pcolormesh using the exact min/max from tuples
    xedges = np.linspace(plot_range_r_tuple[0], plot_range_r_tuple[1], nbins + 1)
    yedges = np.linspace(plot_range_v_tuple[0], plot_range_v_tuple[1], nbins + 1)
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    C = H.T  # Transpose for correct orientation

    # Create a figure for this frame
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()

    # Plot the phase space - Get the colormap from this call to use consistently
    cmap = plt.cm.viridis  # Default colormap
    pcm = plt.pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0, vmax=vmax_colorscale)
    cbar = plt.colorbar(pcm)
    cbar.set_label('Counts', fontsize=12)
    plt.xlabel(r'$r$ (kpc)', fontsize=12)
    plt.ylabel(r'$v$ (km/s)', fontsize=12)
    plt.title('Phase Space Distribution Evolution', fontsize=14, pad=20)
    plt.xlim(plot_range_r_tuple) # Pass the tuple directly
    plt.ylim(plot_range_v_tuple) # Pass the tuple directly

    # Calculate time in dynamical times using the tfinal_factor and passed total_snapshots
    if total_snapshots and total_snapshots > 1:
        normalized_snap = snap_num / (total_snapshots - 1)
    else:
        # Approximate by assuming snapshots range from 0-100
        # (snapshots often use a zero-padded 5-digit format with max around 00100)
        normalized_snap = min(1.0, snap_num / 100)

    # Scale by tfinal_factor to get time in dynamical units
    t_dyn_fraction = normalized_snap * tfinal_factor

    # Add text in upper right corner showing time in dynamical times
    # Use the same colormap as the plot for consistency
    # Get colors directly from the current colormap
    # Map min value to background (usually dark blue/purple in viridis)
    # Map max value to text (usually yellow in viridis)
    bg_color = pcm.cmap(0.0)  # Background color = min value in colormap
    text_color = pcm.cmap(1.0)  # Text color = max value in colormap

    ax.text(0.98, 0.95, f"$t = {t_dyn_fraction:.2f}\\,t_{{\\rm dyn}}$",
            transform=ax.transAxes, ha='right', va='top', fontsize=12,
            color=text_color,
            bbox=dict(facecolor=bg_color, alpha=1.0, edgecolor='none', pad=3))

    # Convert figure to image in memory
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)

    # Clean up the figure to avoid memory leaks
    plt.close()

    # Return the image
    return imageio.imread(buf)

def generate_initial_phase_histogram(suffix, config=None):
    """
    Creates a phase space histogram from the initial particles.dat file.
    This represents the true initial distribution.
    
    Parameters
    ----------
    suffix : str
        Suffix for file naming
    config : dict, optional
        Configuration dictionary with plot settings
    """
    # Constants for expected number of columns
    ncol_particles_initial = 4

    # Path to the initial particles file
    particles_file = f"data/particles{suffix}.dat"

    if not os.path.exists(particles_file):
        print_status(f"Initial particles file {particles_file} not found")
        return False

    # Start loading the initial data with timing information
    # Log to file only
    logger.info(f"Loading initial phase space histogram: {get_file_prefix(particles_file)}")
    start_time = time.time()

    # Start progress tracking for data loading
    start_combined_progress("phase_space_data_loading", 1)

    # Update progress - loading initial data (use filename prefix format)
    update_combined_progress("phase_space_data_loading", "particles_data")

    # Load the data
    data = safe_load_and_filter_bin(
        particles_file, ncol_particles_initial, np.float32)

    elapsed = time.time() - start_time

    if data is None or data.shape[0] == 0:
        print_status(f"No valid data in {particles_file}")
        return False

    # Log to file only
    logger.info(f"Initial histogram loaded successfully [{elapsed:.2f}s]")

    # Extract data columns - using first 3 relevant columns
    iradii = data[:, 0]
    radialvelocities = data[:, 1]
    ell = data[:, 2]
    # Avoid division by zero by filtering out non-positive radii
    nonzero_mask = iradii > 0
    iradii = iradii[nonzero_mask]
    radialvelocities = radialvelocities[nonzero_mask]
    ell = ell[nonzero_mask]

    # Compute the derived (total) velocity - ALL IN SIMULATION UNITS
    # v_total_sim = sqrt((L/r)^2 + v_r^2)
    tangential_velocity_sim = ell / iradii  # Tangential velocity in simulation units
    total_velocity_sim = np.sqrt(tangential_velocity_sim**2 + radialvelocities**2)  # Total velocity in simulation units

    # Convert to km/s ONLY at the end
    kmsec_to_kpcmyr = 1.02271e-3  # Conversion factor
    ivelocities = total_velocity_sim / kmsec_to_kpcmyr  # Convert to km/s

    # Filter out invalid values
    valid_mask = (iradii >= 0) & (iradii < 1e8) & (ivelocities > 0) & (ivelocities < 1e8)
    iradii = iradii[valid_mask]
    ivelocities = ivelocities[valid_mask]

    if len(iradii) == 0:
        print_status("All initial radii/velocities invalid or empty.")
        return False

    # Log statistics to file only
    logger.info(f"Velocity range: {np.min(ivelocities)} to {np.max(ivelocities)} km/s")
    logger.info(f"Radius range: {np.min(iradii)} to {np.max(iradii)} kpc")

    # Set up output path
    output_file = f"results/phase_space_initial{suffix}.png"

    # Start progress tracking for plot saving
    start_combined_progress("phase_space_plots", 1)

    # Set up histogram parameters using robust ranging
    nbins = 200
    
    # Calculate dynamic ranges using the robust range helper
    r_min, r_max = _calculate_robust_range(
        iradii, percentile=config.x_percentile,
        percentile_multiplier=config.x_percentile_multiplier,
        min_abs_extent=config.min_x_range_abs,
        default_if_empty=(0, 250.0), can_be_negative=False,
        axis_name="radius"
    )
    
    v_min, v_max = _calculate_robust_range(
        ivelocities, percentile=config.y_percentile,
        percentile_multiplier=config.y_percentile_multiplier,
        min_abs_extent=config.min_y_range_abs, 
        default_if_empty=(0, 320.0), can_be_negative=False,
        axis_name="velocity"
    )

    # Create the 2D histogram
    H, xedges, yedges = np.histogram2d(
        iradii,
        ivelocities,
        bins=[nbins, nbins],
        range=[[r_min, r_max], [v_min, v_max]]
    )

    # Create meshgrid for pcolormesh
    X, Y = np.meshgrid(xedges[:-1], yedges[:-1])
    C = H.T  # Transpose for correct orientation


    plt.figure(figsize=(8, 6))

    # Use pcolormesh with shading='auto' like in nsphere plot
    # Calculate a reasonable vmax based on the histogram data
    vmax = calculate_reasonable_vmax(H)
    pcm = plt.pcolormesh(X, Y, C, shading='auto', cmap='viridis', vmin=0, vmax=vmax)

    cbar = plt.colorbar(pcm)
    cbar.set_label('Counts', fontsize=12)
    plt.xlabel(r'$r$ (kpc)', fontsize=12)
    plt.ylabel(r'$v$ (km/s)', fontsize=12)
    plt.title(r'Initial Phase Space Distribution', fontsize=14)

    # Set consistent limits using calculated ranges
    plt.xlim(r_min, r_max)
    plt.ylim(v_min, v_max)


    os.makedirs("results", exist_ok=True)
    start_time = time.time()
    plt.savefig(output_file, dpi=150)
    plt.close()
    elapsed = time.time() - start_time

    # Update progress for saving plot
    update_combined_progress("phase_space_plots", output_file)

    # Log completion message to file only, not to console
    log_message(f"Phase space histogram saved: {get_file_prefix(output_file)} [{elapsed:.2f}s]", "info")


    logger.info(f"Initial phase space histogram saved to {output_file}")

    return True

def generate_comparison_plot(suffix, config=None):
    """
    Creates a side-by-side comparison of the initial and last available snapshot phase space histograms.
    
    Parameters
    ----------
    suffix : str
        Suffix for file naming
    config : dict, optional
        Configuration dictionary with plot settings
    """
    # Constants for expected number of columns
    ncol_particles_initial = 4
    ncol_Rank_Mass_Rad_VRad_sorted = 8

    # Path to the initial particles file
    particles_file = f"data/particles{suffix}.dat"

    # Find the last available snapshot file
    rank_files = glob.glob(f"data/Rank_Mass_Rad_VRad_sorted_t*{config.suffix}.dat")
    if not rank_files:
        print_status("No snapshot files found.")
        return False

    # Sort files by snapshot number and get the last one
    rank_files.sort(key=lambda fname: _extract_Rank_snapnum(fname, config.suffix))
    last_snap_file = rank_files[-1]
    last_snap_num = _extract_Rank_snapnum(last_snap_file, suffix)

    # Log the file search details to the log file, not the console
    logger.info(f"Looking for initial file: {particles_file}")
    logger.info(f"Looking for last snapshot file: {last_snap_file} (snapshot {last_snap_num})")

    if not os.path.exists(particles_file):
        print_status(f"Initial particles file {particles_file} not found.")
        return False

    if not os.path.exists(last_snap_file):
        print_status(f"Last snapshot file {last_snap_file} not found.")
        return False

    # Start progress tracking for data loading (5 steps: 2 for loading, 3 for processing/histograms)
    start_combined_progress("phase_space_loading", 5)

    # Update progress - both files found, starting to load (use filename prefix)
    update_combined_progress("phase_space_loading", "particles_data")

    # Load the initial data
    initial_data = safe_load_and_filter_bin(particles_file, ncol_particles_initial, np.float32)

    if initial_data is None or initial_data.shape[0] == 0:
        print_status(f"No valid data in {particles_file}")
        return False

    # Log the data shape details to the log file
    logger.info(f"Initial data loaded, shape: {initial_data.shape}")

    # Update progress - loaded initial data (use filename prefix)
    update_combined_progress("phase_space_loading", "particles_done")

    # Load the last snapshot data
    data = safe_load_and_filter_bin(
        last_snap_file,
        ncol_Rank_Mass_Rad_VRad_sorted,
        dtype=[np.int32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32, np.float32]
    )

    if data is None or data.shape[0] == 0:
        print_status(f"No valid data in {last_snap_file}")
        return False

    # Log the data shape details to the log file
    logger.info(f"Final snapshot data loaded, shape: {data.shape}")

    # Update progress - loaded last snapshot data (use filename prefix)
    update_combined_progress("phase_space_loading", "snapshot_data")

    # Update progress for processing initial data - use file prefix format
    update_combined_progress("phase_space_loading", "initial_data")
    # Extract data columns - using first 3 relevant columns
    iradii = initial_data[:, 0]
    radialvelocities = initial_data[:, 1]
    ell = initial_data[:, 2]

    # Log detailed processing steps to log file
    logger.info("Processing initial data - extracting columns and calculating velocities")

    # Avoid division by zero by filtering out non-positive radii
    nonzero_mask = iradii > 0
    iradii = iradii[nonzero_mask]
    radialvelocities = radialvelocities[nonzero_mask]
    ell = ell[nonzero_mask]

    # Compute the derived (total) velocity - ALL IN SIMULATION UNITS
    # v_total_sim = sqrt((L/r)^2 + v_r^2)
    tangential_velocity_sim = ell / iradii  # Tangential velocity in simulation units
    total_velocity_sim = np.sqrt(tangential_velocity_sim**2 + radialvelocities**2)  # Total velocity in simulation units

    # Convert to km/s ONLY at the end
    kmsec_to_kpcmyr = 1.02271e-3  # Conversion factor
    ivelocities = total_velocity_sim / kmsec_to_kpcmyr  # Convert to km/s

    # Filter out invalid values
    valid_mask = (iradii >= 0) & (iradii < 1e8) & (ivelocities > 0) & (ivelocities < 1e8)
    iradii = iradii[valid_mask]
    ivelocities = ivelocities[valid_mask]

    if len(iradii) == 0:
        print_status("All initial radii/velocities invalid or empty.")
        return False

    # Update progress after processing initial data - use file prefix format
    update_combined_progress("phase_space_loading", "final_data")

    # Process last snapshot data
    # Extract radius and velocity data from sorted data with columns:
    # Rank(0), Mass(1), Radius(2), Vrad(3), Psi(4), Energy(5), L(6), Density(7)
    radii = data[:, 2]          # Column index 2 holds radius (r) - simulation units
    radial_velocity = data[:, 3]  # Column index 3 holds radial velocity (v_r) - simulation units
    angular_momentum = data[:, 6]  # Column index 6 holds angular momentum (L) - simulation units


    logger.info("Processing final snapshot data - extracting columns and calculating velocities")

    # Filter out non-positive radii to avoid division by zero
    nonzero_mask = radii > 0
    radii = radii[nonzero_mask]
    radial_velocity = radial_velocity[nonzero_mask]
    angular_momentum = angular_momentum[nonzero_mask]

    # Compute the total velocity - ALL IN SIMULATION UNITS
    # v_total_sim = sqrt((L/r)^2 + v_r^2)
    tangential_velocity_sim = angular_momentum / radii  # Tangential velocity in simulation units
    total_velocity_sim = np.sqrt(tangential_velocity_sim**2 + radial_velocity**2)  # Total velocity in simulation units

    # Convert to km/s ONLY at the end
    total_velocity = total_velocity_sim / kmsec_to_kpcmyr  # Convert to km/s

    # Filter out invalid values
    valid_mask = (radii >= 0) & (radii < 1e8) & (total_velocity > 0) & (total_velocity < 1e8)
    radii = radii[valid_mask]
    total_velocity = total_velocity[valid_mask]

    if len(radii) == 0:
        print_status("All final snapshot radii/velocities invalid or empty.")
        return False

    # Update progress after processing both datasets - use a data file format for display
    histogram_file = f"data/creating_histograms{suffix}.dat"
    update_combined_progress("phase_space_loading", histogram_file)

    # Set up histogram parameters using robust ranging
    nbins = 200
    
    # Calculate global ranges covering both datasets
    combined_radii = np.concatenate([iradii, radii])
    combined_velocities = np.concatenate([ivelocities, total_velocity])
    
    # Calculate dynamic ranges using the robust range helper
    r_min, r_max = _calculate_robust_range(
        combined_radii, percentile=config.x_percentile,
        percentile_multiplier=config.x_percentile_multiplier,
        min_abs_extent=config.min_x_range_abs,
        default_if_empty=(0, 250.0), can_be_negative=False,
        axis_name="radius"
    )
    
    v_min, v_max = _calculate_robust_range(
        combined_velocities, percentile=config.y_percentile,
        percentile_multiplier=config.y_percentile_multiplier,
        min_abs_extent=config.min_y_range_abs, 
        default_if_empty=(0, 320.0), can_be_negative=False,
        axis_name="velocity"
    )

    # Create the 2D histograms
    H_initial, xedges_initial, yedges_initial = np.histogram2d(
        iradii,
        ivelocities,
        bins=[nbins, nbins],
        range=[[r_min, r_max], [v_min, v_max]]
    )

    H_last_snap, xedges_last_snap, yedges_last_snap = np.histogram2d(
        radii,
        total_velocity,
        bins=[nbins, nbins],
        range=[[r_min, r_max], [v_min, v_max]]
    )

    # Print histogram statistics for comparison
    print_status("Comparison of histogram statistics:")
    print_status(f"Initial Data: Max. Value = {np.max(H_initial)}, Non-Zero Bins = {np.count_nonzero(H_initial)}, Mean Value (Non-Zero) = {np.sum(H_initial)/np.count_nonzero(H_initial):.2f}")
    print_status(f"Final Data:   Max. Value = {np.max(H_last_snap)}, Non-Zero Bins = {np.count_nonzero(H_last_snap)}, Mean Value (Non-Zero) = {np.sum(H_last_snap)/np.count_nonzero(H_last_snap):.2f}")
    # Add separator line after statistics
    sys.stdout.write(get_separator_line(char='-') + "\n")

    # Create meshgrids for pcolormesh
    X_initial, Y_initial = np.meshgrid(xedges_initial[:-1], yedges_initial[:-1])
    C_initial = H_initial.T  # Transpose for correct orientation

    X_last_snap, Y_last_snap = np.meshgrid(xedges_last_snap[:-1], yedges_last_snap[:-1])
    C_last_snap = H_last_snap.T  # Transpose for correct orientation

    # Create the side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)

    # Set vmin and vmax to use a consistent color scale across both plots
    vmax = calculate_reasonable_vmax(H_initial)

    
    pcm1 = ax1.pcolormesh(X_initial, Y_initial, C_initial, shading='auto', cmap='viridis', vmin=0, vmax=vmax)
    ax1.set_xlabel(r'$r$ (kpc)', fontsize=12)
    ax1.set_ylabel(r'$v$ (km/s)', fontsize=12)
    ax1.set_title(r'Initial Phase Space', fontsize=14)
    ax1.set_xlim(r_min, r_max)
    ax1.set_ylim(v_min, v_max)

    
    pcm2 = ax2.pcolormesh(X_last_snap, Y_last_snap, C_last_snap, shading='auto', cmap='viridis', vmin=0, vmax=vmax)
    ax2.set_xlabel(r'$r$ (kpc)', fontsize=12)
    ax2.set_title(fr'Final Phase Space (Snapshot $t={last_snap_num}$)', fontsize=14)
    ax2.set_xlim(r_min, r_max)
    ax2.set_ylim(v_min, v_max)

    
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(pcm2, cax=cbar_ax)
    cbar.set_label('Counts', fontsize=12)

    
    fig.suptitle('Comparison of Phase Space Distributions: Initial vs Final', fontsize=16)
    # Use fig.subplots_adjust instead of tight_layout for better control
    fig.subplots_adjust(left=0.08, right=0.9, top=0.9, bottom=0.1)

    # Ensure the results directory exists
    os.makedirs("results", exist_ok=True)

    
    output_files = []

    
    output_file = f"results/phase_space_comparison{suffix}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()
    output_files.append(output_file)

    # Create a difference plot to highlight changes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Calculate the difference (last_snap - initial)
    diff = C_last_snap - C_initial

    # Use a diverging colormap for the difference plot with adjusted scale
    # Calculate a reasonable scale based on the data
    abs_max = max(abs(np.min(diff)), abs(np.max(diff)))
    # Ensure we have a non-zero scale and use a slightly larger value to avoid clipping
    vmax_diff = max(abs_max * 1.1, 1.0)

    pcm_diff = ax.pcolormesh(X_initial, Y_initial, diff, shading='auto', cmap='RdBu_r',
                           vmin=-vmax_diff, vmax=vmax_diff)  # Dynamic symmetric scale for differences

    ax.set_xlabel(r'$r$ (kpc)', fontsize=12)
    ax.set_ylabel(r'$v$ (km/s)', fontsize=12)
    ax.set_title(fr'Phase Space Difference (Final - Initial)', fontsize=14)
    ax.set_xlim(r_min, r_max)
    ax.set_ylim(v_min, v_max)

    
    cbar = fig.colorbar(pcm_diff)
    cbar.set_label('Difference in Counts', fontsize=12)

    
    diff_output_file = f"results/phase_space_difference{suffix}.png"
    plt.savefig(diff_output_file, dpi=150)
    plt.close()
    output_files.append(diff_output_file)

    # Start combined progress tracking for the plots
    total_plots = len(output_files)
    start_combined_progress("phase_space_plots", total_plots)

    # Log both plots with progress indication
    for i, output_file in enumerate(output_files, 1):
        update_combined_progress("phase_space_plots", output_file)

    return True

def plot_convergence_test(Nint_arr, Nspl_arr, basefile, suffix, xlabel, ylabel, title, output_file):
    """
    Generate convergence test plots comparing different integration and spline parameters.

    Parameters
    ----------
    Nint_arr : list
        List of integration parameter values.
    Nspl_arr : list
        List of spline parameter values.
    basefile : str
        Base filename to use.
    suffix : str
        Suffix for input/output files.
    xlabel : str
        X-axis label (already formatted LaTeX string).
    ylabel : str
        Y-axis label (already formatted LaTeX string).
    title : str
        Plot title (already formatted LaTeX string).
    output_file : str
        Output file path.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Determine number of columns based on basefile
    ncols = ncol_convergence # Default for most convergence files
    if basefile == "density_profile":
        ncols = ncol_density_profile
    elif basefile == "massprofile":
        ncols = ncol_mass_profile
    elif basefile == "Psiprofile":
        ncols = ncol_psi_profile
    elif basefile == "f_of_E":
        ncols = ncol_f_of_E
    elif basefile == "integrand":
        ncols = ncol_integrand

    has_data = False
    for Nint in Nint_arr:
        for Nspl in Nspl_arr:
            filepath = f"data/{basefile}_Ni{Nint}_Ns{Nspl}{suffix}.dat"
            if not os.path.exists(filepath):
                continue

            data = safe_load_and_filter_bin(filepath, ncols, dtype=np.float32)
            if data is None:
                continue

            # Filter out rows with NaN or Inf values
            mask = np.all(np.isfinite(data), axis=1)
            filtered_data = data[mask]

            if len(filtered_data) == 0:
                continue

            # Additional filtering for log-log plots: remove zero or negative values
            log_mask = (filtered_data[:, 0] > 0) & (filtered_data[:, 1] > 0)
            filtered_data = filtered_data[log_mask]
            
            if len(filtered_data) == 0:
                continue

            ax.plot(
                filtered_data[:, 0],
                filtered_data[:, 1],
                label=r"$N_{\rm int}=%d$, $N_{\rm spl}=%d$" % (Nint, Nspl)
            )
            has_data = True

    if has_data:
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7, which='both')
        
        # Set log-log scale
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        plt.close()

        # Log to file only, no console output
        logger.info(f"Plot saved: {output_file}")
    else:
        plt.close()
        logger.warning(f"No valid data found for convergence test plot: {output_file}")

    # Progress tracking is handled by the caller

def generate_convergence_test_plots(suffix):
    """
    Generate all convergence test plots.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
    """
    # Define parameter arrays for testing
    Nint_arr = [1000, 10000]
    Nspl_arr = [1000, 10000]

    # Define plot specifications (basefile, labels, title, output)
    plot_specs = [
        # Psi profile convergence test
        ("Psiprofile", r"$r$ (kpc)", r"$\Psi$ (km$^2$/s$^2$)", r"$\Psi(r)$ Profile Convergence Test",
         f"results/Psiprofile_convergence_test{suffix}.png"),
        # Integrand convergence test - updated labels based on C code analysis
        ("integrand", r'$\sqrt{\mathcal{E}_{max} - \Psi}$', r'$2 \, d\rho/d\Psi$', r"Integrand Convergence Test",
         f"results/integrand_convergence_test{suffix}.png"),
        # Distribution function convergence test - corrected units and symbol
        ("f_of_E", r"$\mathcal{E}$ (km$^2$/s$^2$)", r"$f(\mathcal{E})$ ((km/s)$^{-3}$ kpc$^{-3}$)", r"$f(\mathcal{E})$ Convergence Test",
         f"results/f_of_E_convergence_test{suffix}.png"),
        # Mass profile convergence test
        ("massprofile", r"$r$ (kpc)", r"$M(r)$ (M$_\odot$)", r"$M(r)$ Profile Convergence Test",
         f"results/massprofile_convergence_test{suffix}.png"),
        # Density profile convergence test - uses rho, not 4*pi*r^2*rho
        ("density_profile", r"$r$ (kpc)", r"$\rho(r)$ (M$_\odot$/kpc$^3$)", r"Density Profile $\rho(r)$ Convergence Test",
         f"results/density_profile_convergence_test{suffix}.png")
    ]

    # Start progress tracking for data loading
    total_data_files = len(plot_specs)
    start_combined_progress("convergence_data_loading", total_data_files)

    # First check for data files and log loading progress
    for i, (basefile, xlabel, ylabel, title, output_file) in enumerate(plot_specs, 1):
        # Check for different Nint/Nspl combinations to see if any files exist
        found_files = False
        for Nint in Nint_arr:
            for Nspl in Nspl_arr:
                filepath = f"data/{basefile}_Ni{Nint}_Ns{Nspl}{suffix}.dat"
                if os.path.exists(filepath):
                    found_files = True
                    logger.info(f"Found convergence test file: {filepath}")
                    break
            if found_files:
                break

        if found_files:
            logger.info(f"Loading data for {basefile} convergence test")
        else:
            logger.warning(f"No data files found for {basefile} convergence test")

        # Update progress with a more filename-like format
        if found_files:
            # Use the first found file as the reference
            for Nint in Nint_arr:
                for Nspl in Nspl_arr:
                    filepath = f"data/{basefile}_Ni{Nint}_Ns{Nspl}{suffix}.dat"
                    if os.path.exists(filepath):
                        update_combined_progress("convergence_data_loading", filepath)
                        break
                if os.path.exists(filepath):
                    break
        else:
            # If no files were found, still use a filename-like format
            dummy_filepath = f"data/{basefile}_convergence{suffix}.dat"
            update_combined_progress("convergence_data_loading", dummy_filepath)

    # Initialize combined progress tracking for plot generation
    total_plots = len(plot_specs)
    start_combined_progress("convergence_tests", total_plots)

    # Generate all plots with combined progress tracking
    for i, (basefile, xlabel, ylabel, title, output_file) in enumerate(plot_specs, 1):
        try:

            plot_convergence_test(Nint_arr, Nspl_arr, basefile, suffix, xlabel, ylabel, title, output_file)

            # Update the combined progress bar
            update_combined_progress("convergence_tests", output_file)

        except Exception as e:
            logger.error(f"Error generating {basefile} convergence test: {str(e)}")
            print_status(f"Error generating {basefile} convergence test: {str(e)}")
            continue

def process_rank_files(suffix, start_snap, end_snap, step_snap):
    """
    Processes sorted and unsorted Rank snapshot files using multiprocessing.

    - **Sorted Files:** Calls worker `process_rank_file_for_1d_anim` for each
      filtered sorted file. This worker loads necessary columns, uses
      linear interpolation/downsampling, and returns processed arrays. The
      results are used to populate the global `mass_snapshots`,
      `density_snapshots`, and `psi_snapshots` lists directly.
    - **Unsorted Files:** Calls worker `process_unsorted_rank_file` for each
      unsorted file. This worker uses optimized seeking to load energy
      data for the first few particles.
    - Progress is displayed using tqdm.

    Parameters
    ----------
    suffix : str
        Suffix for data files.
    start_snap : int
        First snapshot number to include.
    end_snap : int
        Last snapshot number to include (0 means use max available).
    step_snap : int
        Step size between snapshots.

    Returns
    -------
    tuple
        (None, unsorted_energy_list): The first element is always None as
        sorted data is placed directly into global lists. The second element
        is a list of (snap, energy_values) tuples from unsorted files.
    """
    global mass_snapshots, density_snapshots, psi_snapshots

    # Clear global snapshot data first to ensure we start fresh
    mass_snapshots.clear()
    density_snapshots.clear()
    psi_snapshots.clear()

    # Find and process sorted rank files
    Rank_pattern = f"data/Rank_Mass_Rad_VRad_sorted_t*{suffix}.dat"
    all_rank_files = glob.glob(Rank_pattern)

    # Filter to keep ONLY files that match the exact pattern:
    # data/Rank_Mass_Rad_VRad_sorted_t00001_40000_1001_5.dat
    # without any extra characters between "t00001" and the suffix
    correct_pattern = re.compile(r'data/Rank_Mass_Rad_VRad_sorted_t\d+' + re.escape(suffix) + r'\.dat$')
    all_rank_files = [f for f in all_rank_files if correct_pattern.match(f)]

    # Debug output
    global enable_logging
    if enable_logging:
        log_message(f"Found {len(all_rank_files)} rank files after filtering.")

    # Find and process unsorted rank files
    unsorted_pattern = f"data/Rank_Mass_Rad_VRad_unsorted_t*{suffix}.dat"
    unsorted_files = glob.glob(unsorted_pattern)

    # Filter unsorted files as well
    correct_unsorted_pattern = re.compile(r'data/Rank_Mass_Rad_VRad_unsorted_t\d+' + re.escape(suffix) + r'\.dat$')
    unsorted_files = [f for f in unsorted_files if correct_unsorted_pattern.match(f)]
    correct_unsorted_pattern = re.compile(r'data/Rank_Mass_Rad_VRad_unsorted_t\d+' + re.escape(suffix) + r'\.dat$')
    unsorted_files = [f for f in unsorted_files if correct_unsorted_pattern.match(f)]
    logger.info(f"Found {len(all_rank_files)} rank sorted snapshot files and {len(unsorted_files)} unsorted snapshot files.")

    # Process sorted files for 1D Animations
    if not all_rank_files:
        logger.warning("No rank sorted snapshot files found for animation.")
        print_status("No rank sorted snapshot files found for animation.")
    else:
        # Sort files by snapshot number
        pattern = re.compile(r'Rank_Mass_Rad_VRad_sorted_t(\d+)')
        all_rank_files.sort(key=lambda filename: get_snapshot_number(filename, pattern))

        # Filter files based on start/end/step if specified
        filtered_rank_files = []
        for fname in all_rank_files:
            snap = get_snapshot_number(fname, pattern)
            # Skip files outside the requested range
            if start_snap > 0 and snap < start_snap:
                continue
            if end_snap > 0 and snap > end_snap:
                continue
            # Skip files not on the requested step
            if step_snap > 1 and (snap - start_snap) % step_snap != 0:
                continue
            filtered_rank_files.append(fname)

        logger.info(f"Processing {len(filtered_rank_files)} sorted files for 1D animations...")

        with mp.Pool(mp.cpu_count()) as pool:
            # Setup tqdm bars
            try:
                term_width = shutil.get_terminal_size().columns
            except OSError:
                term_width = 100 # Fallback width
            selected_desc, selected_bar_format = select_tqdm_format('proc_sorted_snaps', term_width)
            
            counter_tqdm = tqdm(
                total=len(filtered_rank_files),
                desc=selected_desc, # Use dynamic description
                unit="file",
                position=0,
                leave=True,
                miniters=1,
                dynamic_ncols=True, # Allow adapting to terminal width
                ncols=None,
                bar_format=selected_bar_format
            )

            # For alignment with the counter line
            bar_tqdm = tqdm(
                total=len(filtered_rank_files),
                position=1,
                leave=True,
                dynamic_ncols=True,
                bar_format="{bar} {percentage:3.1f}%",
                ascii=False  # Use Unicode block characters
            )

            # Process files and handle results as they arrive
            processed_count_1d = 0
            # Prepare arguments including the suffix for the 1D animation worker
            args_for_1d_anim = [(fname, PROJECT_ROOT, suffix) for fname in filtered_rank_files]
            # Pass the modified arguments list to imap_unordered
            for result in pool.imap_unordered(process_rank_file_for_1d_anim, args_for_1d_anim, chunksize=2):
                if result is not None:
                    snap, radii, mass, density, psi = result
                    # Append directly to global lists
                    mass_snapshots.append((snap, radii, mass))
                    density_snapshots.append((snap, radii, density))
                    psi_snapshots.append((snap, radii, psi))
                    processed_count_1d += 1
                # Update progress bars
                counter_tqdm.update(1)
                bar_tqdm.update(1)
                if counter_tqdm.n % 50 == 0: gc.collect() # Periodic GC in main

            counter_tqdm.close()
            bar_tqdm.close()
            gc.collect() # Collect after pool

        # Add separator line after tqdm progress bar
        sys.stdout.write(get_separator_line(char='-') + "\n")

        # Sort the snapshot lists by snap number AFTER collecting all results
        mass_snapshots.sort(key=lambda x: x[0])
        density_snapshots.sort(key=lambda x: x[0])
        psi_snapshots.sort(key=lambda x: x[0])

        logger.info(f"Finished processing {processed_count_1d} sorted files for 1D animations.")

    # Process unsorted files
    unsorted_energy_list = []
    if unsorted_files:
        # Process unsorted files for basic energy plot in parallel
        logger.info(f"Processing {len(unsorted_files)} unsorted snapshot files...")

        with mp.Pool(mp.cpu_count()) as pool:
            # Use a custom tqdm instance with two progress displays
            # First, create a standard tqdm for the counter on the first line
            # Determine description and format dynamically
            try:
                term_width = shutil.get_terminal_size().columns
            except OSError:
                term_width = 100 # Fallback width
            selected_desc, selected_bar_format = select_tqdm_format('proc_unsorted_snaps', term_width)
            
            counter_tqdm = tqdm(
                total=len(unsorted_files),
                desc=selected_desc, # Use dynamic description
                unit="file",
                position=0,
                leave=True,
                miniters=1,
                dynamic_ncols=True, # Allow adapting to terminal width
                ncols=None,
                bar_format=selected_bar_format
            )

            # For alignment with the counter line, we'll use a fixed width
            bar_tqdm = tqdm(
                total=len(unsorted_files),
                position=1,
                leave=True,
                dynamic_ncols=True,
                bar_format="{bar} {percentage:3.1f}%",
                ascii=False  # Use Unicode block characters
            )

            # Process the files with a custom callback to update both progress bars
            results = []
            # Prepare arguments including suffix for the unsorted worker
            args_for_unsorted = [(fname, PROJECT_ROOT, suffix) for fname in unsorted_files]
            # Pass modified args to imap
            for result in pool.imap(process_unsorted_rank_file, args_for_unsorted):
                results.append(result)
                counter_tqdm.update(1)
                bar_tqdm.update(1)


            counter_tqdm.close()
            bar_tqdm.close()

            # Add delay after progress bars if enabled
            if progress_delay > 0 and paced_mode:
                show_progress_delay(progress_delay)

            unsorted_energy_list = results

        # Add separator line after tqdm progress bar
        sys.stdout.write(get_separator_line(char='-') + "\n")

        # Remove any files that failed to process
        unsorted_energy_list = [x for x in unsorted_energy_list if x is not None]
        if not unsorted_energy_list:
            logger.warning("No valid unsorted energy data found.")
            print_status("No valid unsorted energy data found.")
    else:
        logger.warning("No unsorted snapshot files found for energy plots.")
        print_status("No unsorted snapshot files found for energy plots.")

    # Get the maximum available snapshot number
    max_available_snap = 0
    if all_rank_files:
        match = pattern.search(all_rank_files[-1])
        if match:
            max_available_snap = int(match.group(1))

    # If end_snap is 0, use the maximum available snapshot
    local_end_snap = end_snap
    if local_end_snap == 0:
        local_end_snap = max_available_snap

    logger.info(f"Using snapshot range: {start_snap} to {local_end_snap} with step {step_snap}")

    # Filter files based on start, end, and step parameters
    filtered_rank_files = []
    for f in all_rank_files:
        match = pattern.search(f)
        if match:
            snap_num = int(match.group(1))
            if start_snap <= snap_num <= local_end_snap and (snap_num - start_snap) % step_snap == 0:
                filtered_rank_files.append(f)

    logger.info(f"Found {len(filtered_rank_files)} rank sorted snapshot files in the specified range.")

    # Sort unsorted energy data by snapshot number
    if unsorted_energy_list:
        unsorted_energy_list.sort(key=lambda x: x[0])


    print_status("Particle snapshot data processing complete.")
    # Add blank line after completion message
    print()

    # Return the processed data
    return None, unsorted_energy_list

def generate_unsorted_energy_plot(unsorted_energy_list, suffix):
    """
    Generate energy plot from unsorted rank files.

    Parameters
    ----------
    unsorted_energy_list : list
        Pre-processed unsorted energy data.
    suffix : str
        Suffix for input/output files.

    Returns
    -------
    str or None
        Path to the generated plot file, or None if no plot was generated.
    """
    if not unsorted_energy_list:
        logger.info("No unsorted energy data available for plotting.")
        return None

    # Get snapshot numbers
    unsorted_snaps = [snap for (snap, data) in unsorted_energy_list]

    # Extract tfinal_factor from suffix if possible
    # Format is typically _[file_tag]_npts_Ntimes_tfinal_factor
    tfinal_factor = 5  # Default value
    parts = suffix.strip('_').split('_')
    if len(parts) >= 3:
        try:
            tfinal_factor = int(parts[-1])
        except (ValueError, IndexError):
            # Use default if parsing fails
            pass

    # Convert snapshot numbers to dynamical times
    total_snapshots = len(unsorted_snaps)
    if total_snapshots > 1:
        dyn_times = [snap / (total_snapshots - 1) * tfinal_factor for snap in unsorted_snaps]
    else:
        dyn_times = [0]  # Default if only one snapshot

    
    unsorted_E_matrix = np.array([data for (snap, data) in unsorted_energy_list])
    if unsorted_E_matrix.ndim >= 2 and unsorted_E_matrix.shape[0] > 0 and unsorted_E_matrix.shape[1] > 0:
        plt.figure(figsize=(10, 6))
        for row_idx in range(min(10, unsorted_E_matrix.shape[1])):
            plt.plot(dyn_times, unsorted_E_matrix[:, row_idx]) # Labels removed for clarity
        plt.xlabel(r'$t$ ($t_{\rm dyn}$)', fontsize=12)
        plt.ylabel(r'$\mathcal{E}$ (km$^2$/s$^2$)', fontsize=12)
        plt.title(r'Energy vs. Time (Random Sample)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        # Legend removed
        output_file = f"results/Energy_vs_timestep_unsorted{suffix}.png"
        plt.savefig(output_file, dpi=150)
        plt.close()


        logger.info(f"Saved unsorted energy plot: {output_file}")

        # Return the output file path for progress tracking
        return output_file

    return None

def generate_sorted_energy_plot(suffix):
    """
    Generate energy plot from sorted rank files (particles with lowest initial radius).

    Uses the efficient `process_sorted_energy_file` worker which employs 
    seek-based loading to extract energy data only for specific particles. 
    Processes results iteratively to build the final matrix, avoiding 
    intermediate data accumulation.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
        
    Returns
    -------
    str or None
        Path to the generated plot file, or None if no plot was generated.
        
    Notes
    -----
    This function uses memory-efficient file reading by:
    1. Identifying specific particle IDs to track
    2. Using optimized file reading that seeks to specific rows
    3. Processing data incrementally to avoid large intermediate arrays
    """
    try:
        # Load the particle IDs to track (particles with lowest initial radius)
        lowest_radius_ids_file = f"data/lowest_radius_ids{suffix}.dat"

        # Load the lowest_radius_ids file using the binary file handler
        id_data = safe_load_and_filter_bin(lowest_radius_ids_file, ncols=2, dtype=[np.int32, np.float32])
        if id_data is None or len(id_data) < 1:
            print_status(f"Failed to load particle IDs from {lowest_radius_ids_file}")
            return None

        # Extract the first 10 particles to track (or fewer if less available)
        max_particles = min(10, id_data.shape[0])
        tracked_radii = id_data[:max_particles, 1]

        # Update the global tracked_ids_for_energy
        global tracked_ids_for_energy
        tracked_ids_for_energy = id_data[:max_particles, 0].astype(int)

        # Collect unsorted data for these specific particles in parallel
        unsorted_pattern = f"data/Rank_Mass_Rad_VRad_unsorted_t*{suffix}.dat"
        unsorted_files_for_sort = glob.glob(unsorted_pattern)

        # Filter to keep ONLY files that match the exact pattern without any extra characters between "t00001" and the suffix
        correct_unsorted_pattern = re.compile(r'data/Rank_Mass_Rad_VRad_unsorted_t\d+' + re.escape(suffix) + r'\.dat$')
        unsorted_files_for_sort = [f for f in unsorted_files_for_sort if correct_unsorted_pattern.match(f)]

        # Log the number of files found after filtering
        logger.info(f"Processing {len(unsorted_files_for_sort)} files for sorted energy plot after filtering.")

        if not unsorted_files_for_sort:
            print_status("No unsorted files found for sorted energy plot.")
            return None

        # Process files in parallel using the global function
        # Log detailed info to file but show condensed version on console
        logger.info(f"Processing {len(unsorted_files_for_sort)} files for energy-time plot...")

        with mp.Pool(mp.cpu_count()) as pool:
            # Use a custom tqdm instance with two progress displays
            # First, create a standard tqdm for the counter on the first line
            # Determine description and format dynamically
            try:
                term_width = shutil.get_terminal_size().columns
            except OSError:
                term_width = 100 # Fallback width
            selected_desc, selected_bar_format = select_tqdm_format('proc_energy_series', term_width)
            
            counter_tqdm = tqdm(
                total=len(unsorted_files_for_sort),
                desc=selected_desc,
                unit="file",
                position=0,  # Position 0 to be the first bar shown
                leave=True,  # Keep the progress bar visible after completion
                miniters=1,
                dynamic_ncols=True, # Allow adapting to terminal width
                ncols=None,
                bar_format=selected_bar_format
            )


            bar_tqdm = tqdm(
                total=len(unsorted_files_for_sort),
                position=1,  # Position 1 to appear below the counter
                leave=True,  # Keep the progress bar visible after completion
                dynamic_ncols=True,
                bar_format="{bar} {percentage:3.1f}%",
                ascii=False  # Use Unicode block characters
            )

            # Process the files with a custom callback to update both progress bars
            results = []
            # Prepare arguments for the worker, including the suffix
            args_for_sorted_energy = [(fname, suffix) for fname in unsorted_files_for_sort]
            # Pass the modified arguments
            for result in pool.imap(process_sorted_energy_file, args_for_sorted_energy):
                results.append(result)
                counter_tqdm.update(1)
                bar_tqdm.update(1)


            counter_tqdm.close()
            bar_tqdm.close()

            # Add delay after progress bars if enabled
            if progress_delay > 0 and paced_mode:
                show_progress_delay(progress_delay)

            # Add a separator line after processing
            print(get_separator_line(char='-'))

            sorted_energy_list = results

        # Remove any files that failed to process
        sorted_energy_list = [x for x in sorted_energy_list if x is not None]

        if not sorted_energy_list:
            print_status("No valid data found for energy-time sorted plot.")
            return None

        sorted_energy_list.sort(key=lambda x: x[0])
        sorted_snaps = [snap for (snap, data) in sorted_energy_list]

        # Preallocate a matrix of NaN values with shape (num_snapshots, num_particles)
        num_snapshots = len(sorted_energy_list)
        expected_particles = max_particles

        # Create a matrix filled with NaN values initially
        logger.info(f"Creating energy matrix for {num_snapshots} snapshots and {expected_particles} particles")
        sorted_E_matrix = np.full((num_snapshots, expected_particles), np.nan)

        # Fill in the actual values from each snapshot
        for i, (snap, data) in enumerate(sorted_energy_list):
            # Log useful information about the data shape
            particle_count = min(data.shape[0], expected_particles)
            logger.info(f"Snapshot {snap}: Found {particle_count} of {expected_particles} expected particles")

            # Fill in available data (may be fewer than expected_particles)
            for j in range(min(data.shape[0], expected_particles)):
                if j < data.shape[0]:  # Make sure we don't exceed the data bounds
                    sorted_E_matrix[i, j] = data[j, 5]  # Column 5 is energy

        # Log the final data shape
        logger.info(f"Final energy matrix shape: {sorted_E_matrix.shape}")

        # Extract tfinal_factor from suffix for time conversion
        tfinal_factor = 5  # Default value
        parts = suffix.strip('_').split('_')
        if len(parts) >= 3:
            try:
                tfinal_factor = int(parts[-1])
            except (ValueError, IndexError):
                # Use default if parsing fails
                pass

        # Convert snapshot numbers to dynamical times once (for all particles)
        total_snapshots = len(sorted_snaps)
        if total_snapshots > 1:
            dyn_times = [snap / (total_snapshots - 1) * tfinal_factor for snap in sorted_snaps]
        else:
            dyn_times = [0]  # Default if only one snapshot

        # Extract initial energies from the first snapshot
        initial_energies = {}
        if sorted_energy_list and len(sorted_energy_list) > 0:
            first_snap, first_data = sorted_energy_list[0]
            for i, row in enumerate(first_data):
                if i < len(tracked_ids_for_energy):
                    initial_energies[tracked_ids_for_energy[i]] = row[5]  # Energy is in column 5

        if sorted_E_matrix.ndim >= 2 and sorted_E_matrix.shape[0] > 0 and sorted_E_matrix.shape[1] > 0:
            plt.figure(figsize=(10, 6))
            for row_idx in range(min(max_particles, sorted_E_matrix.shape[1])):
                particle_id = tracked_ids_for_energy[row_idx]
                init_e = initial_energies.get(particle_id, "N/A")

                # Format initial energy with scientific notation if it's a number
                if init_e != "N/A":
                    init_e_str = f"{init_e:.2e}"
                else:
                    init_e_str = "N/A"

                # Get the data for this particle
                particle_data = sorted_E_matrix[:, row_idx]

                # Check if we have any non-NaN data for this particle
                if np.any(~np.isnan(particle_data)):
                    # Use masked array to ignore NaN values
                    valid_indices = ~np.isnan(particle_data)
                    if np.any(valid_indices):
                        valid_times = np.array(dyn_times)[valid_indices]
                        valid_data = particle_data[valid_indices]

                        # Log how much data was found for this particle
                        data_percentage = np.sum(valid_indices) / len(valid_indices) * 100
                        logger.info(f"Particle {particle_id}: Found {np.sum(valid_indices)}/{len(valid_indices)} data points ({data_percentage:.1f}%)")

                        # Only plot if at least some valid data points exist
                        if len(valid_data) > 0:
                             # Use LaTeX for legend
                            plt.plot(valid_times, valid_data,
                                     label=fr"ID {particle_id} ($r_0$={tracked_radii[row_idx]:.4f}, $\mathcal{{E}}_0$={init_e_str})")
                    else:
                        logger.warning(f"No valid data for particle ID {particle_id}")
                else:
                    logger.warning(f"No data available for particle ID {particle_id}")


            plt.xlabel(r'$t$ ($t_{\rm dyn}$)', fontsize=12)
            plt.ylabel(r'$\mathcal{E}$ (km$^2$/s$^2$)', fontsize=12)
            plt.title(r'Energy vs. Time (Lowest Initial Radius)', fontsize=14)
            # Legend removed for clarity
            plt.grid(True, linestyle='--', alpha=0.7)
            output_file = f"results/Energy_vs_timestep_sorted{suffix}.png"
            plt.savefig(output_file, dpi=150)
            plt.close()


            logger.info(f"Saved sorted energy plot: {output_file}")

            # Return the output file path for progress tracking
            return output_file
    except Exception as e:
        logger.error(f"Failed to generate energy-time sorted plot: {e}")
        logger.error(traceback.format_exc())
        return None

def create_mass_animation(suffix, duration, config=None):
    """
    Create mass profile animation with optional robust ranging.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
    duration : float
        Duration of each frame in milliseconds.
    config : object, optional
        Configuration object containing plot settings. If provided and has
        use_median_ranges=True, will use robust ranging for axes.
        
    Returns
    -------
    None
        Function does not return a value, but saves the animation to a file.
        
    Notes
    -----
    Uses parallel processing for rendering animation frames.
    Saves the animation incrementally using `imageio.get_writer` to reduce
    peak memory usage compared to collecting all frames first.
    Requires imageio v2 (`pip install imageio==2.*`).
    With robust ranging enabled, calculates data-driven axis limits based
    on the median of the data distribution.
    """
    global mass_snapshots, mass_max_value
    
    if not mass_snapshots:
        print_status("No mass data available for animation.")
        return
        
    # Calculate global maximum values for consistent scaling
    calculate_global_max_values()

    total_frames = len(mass_snapshots)
    # Log detailed info to file only
    log_message(f"Generating mass profile frames for {total_frames} snapshots...")

    # Extract tfinal_factor from suffix if possible
    # Format is typically _[file_tag]_npts_Ntimes_tfinal_factor
    tfinal_factor = 5  # Default value
    parts = suffix.strip('_').split('_')
    if len(parts) >= 3:
        try:
            tfinal_factor = int(parts[-1])
        except (ValueError, IndexError):
            # Use default if parsing fails
            pass
            
    # Calculate axis ranges for consistent plotting
    x_range, y_max = _calculate_profile_animation_ranges(
        mass_snapshots, config, data_index=2, x_default_max=200.0,
        animation_name="Mass"
    )
    
    if x_range is not None and y_max is not None:
        # Use calculated ranges
        r_min, r_max = x_range
        m_min = 0  # Mass is always non-negative
        mass_max_value = y_max
    else:
        # Original hardcoded logic
        r_max = 1.1 * np.max([np.max(r) for _, r, _ in mass_snapshots if len(r) > 0])
        r_max = min(200, r_max)  # Cap at 200 kpc for reasonable display
        r_min = 0  # Mass profiles always start at r=0
        m_min = 0  # Mass is always non-negative

    # Create frame_data tuples with the actual snapshot data and range information
    if not mass_snapshots:
        print_status("Error: mass_snapshots list is empty before pool creation.")
        return # Handle error appropriately
    frame_data_list = [(mass_snapshots[i], tfinal_factor, total_frames, 
                        (r_min, r_max), (m_min, mass_max_value), PROJECT_ROOT) 
                       for i in range(total_frames)]

    # Set up imageio writer
    mass_anim_output = f"results/Mass_Profile_Animation{suffix}.gif"
    # Use seconds per frame for imageio v2 duration
    frame_duration_sec_v2 = duration / 1000.0 # Convert ms to seconds
    try:
        # Use mode='I' for multiple images, loop=0 for infinite loop
        writer = imageio.get_writer(
            mass_anim_output,
            format='GIF-PIL',        # Explicitly use Pillow
            mode='I',
            # quantizer='nq',          # Temporarily removed
            palettesize=256,         # Ensure full palette
            duration=frame_duration_sec_v2,
            loop=0
        )
    except Exception as e:
        print_status(f"Error creating GIF writer: {e}")
        return # Cannot proceed without writer

    with mp.Pool(mp.cpu_count()) as pool:
        # Use a custom tqdm instance with two progress displays
        # First, create a standard tqdm for the counter on the first line
        # Determine description and format dynamically
        try:
            term_width = shutil.get_terminal_size().columns
        except OSError:
            term_width = 100 # Fallback width
        selected_desc, selected_bar_format = select_tqdm_format('gen_mass_frames', term_width)
        
        counter_tqdm = tqdm(
            total=total_frames,
            desc=selected_desc, # Use dynamic description
            unit="frame",
            position=0,
            leave=True,
            miniters=1,
            dynamic_ncols=True,
            ncols=None,
            bar_format=selected_bar_format
        )


        bar_tqdm = tqdm(
            total=total_frames,
            position=1,
            leave=True,
            dynamic_ncols=True,
            ncols=None,
            bar_format="{bar} {percentage:3.1f}%",
            ascii=False
        )

        # Process the frames and append directly to the writer
        frame_count = 0
        for frame_image in pool.imap(render_mass_frame, frame_data_list):
            if frame_image is not None:
                try:
                    writer.append_data(frame_image) # Append directly to writer
                    frame_count += 1
                except Exception as e:
                    log_message(f"Error appending frame {frame_count+1}: {e}", level="error")
                    break
            # Update progress bars
            counter_tqdm.update(1)
            bar_tqdm.update(1)


        counter_tqdm.close()
        bar_tqdm.close()
        
        # Close the writer
        try:
            writer.close()
        except Exception as e:
            log_message(f"Error closing GIF writer: {e}", level="error")

        # No delay between frame generation and animation saving to improve fluidity

    # Log to file
    log_message(f"Generated and saved {frame_count}/{total_frames} mass profile frames successfully to {mass_anim_output}")

    if frame_count > 0:
        print_status(f"Animation saved: {get_file_prefix(mass_anim_output)}")
        # Add separator line after animation completion
        sys.stdout.write(get_separator_line(char='-') + "\n")
    else:
        print_status("Failed to generate any mass profile frames.")
    
    # Encourage garbage collection
    gc.collect()

def create_density_animation(suffix, duration, config=None):
    """
    Create density profile animation.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
    duration : float
        Duration of each frame in milliseconds.
    config : Configuration, optional
        Configuration object containing plot settings. If provided,
        enables robust dynamic ranging for axes.
        
    Returns
    -------
    None
        Function does not return a value, but saves the animation to a file.
        
    Notes
    -----
    Uses parallel processing for rendering animation frames.
    Saves the animation incrementally using `imageio.get_writer` to reduce
    peak memory usage compared to collecting all frames first.
    Requires imageio v2 (`pip install imageio==2.*`).
    """
    global density_snapshots, density_max_value

    if not density_snapshots:
        print_status("No density data available for animation.")
        return
        
    # Calculate global maximum values for consistent scaling
    x_range, y_max = _calculate_profile_animation_ranges(
        density_snapshots, config, data_index=2, x_default_max=300.0,
        animation_name="Density"
    )
    
    if y_max is not None:
        density_max_value = y_max
    else:
        calculate_global_max_values()

    total_frames = len(density_snapshots)
    # Log detailed info to file only
    log_message(f"Generating density profile frames for {total_frames} snapshots...")

    # Extract tfinal_factor from suffix if possible
    # Format is typically _[file_tag]_npts_Ntimes_tfinal_factor
    tfinal_factor = 5  # Default value
    parts = suffix.strip('_').split('_')
    if len(parts) >= 3:
        try:
            tfinal_factor = int(parts[-1])
        except (ValueError, IndexError):
            # Use default if parsing fails
            pass
            
    # Calculate r_max for consistent plotting - do this once before creating the pool
    if x_range is not None:
        r_max = x_range[1]  # Use the calculated range from helper function
    else:
        # Original hardcoded behavior when not using median ranges
        r_max = 1.1 * np.max([np.max(r) for _, r, _ in density_snapshots if len(r) > 0])
        r_max = min(300, r_max)  # Cap at 300 kpc for reasonable display

    # Create frame_data tuples with the actual snapshot data, tfinal_factor, total_frames, ranges and project_root_path
    if not density_snapshots:
        print_status("Error: density_snapshots list is empty before pool creation.")
        return # Handle error appropriately
    
    # Prepare range information
    if config and hasattr(config, 'use_median_ranges') and config.use_median_ranges:
        # Use calculated robust ranges
        r_range = (0, r_max)  # Density profiles always start at r=0
        d_range = (0, density_max_value if density_max_value > 0 else 1.2e10)
    else:
        # Use original behavior
        r_range = (0, r_max)
        d_range = (0, density_max_value if density_max_value > 0 else 1.2e10)
    
    frame_data_list = [(density_snapshots[i], tfinal_factor, total_frames, r_range, d_range, PROJECT_ROOT) for i in range(total_frames)]

    # Set up imageio writer
    density_anim_output = f"results/Density_Profile_Animation{suffix}.gif"
    # Use seconds per frame for imageio v2 duration
    frame_duration_sec_v2 = duration / 1000.0 # Convert ms to seconds
    try:
        # Use mode='I' for multiple images, loop=0 for infinite loop
        writer = imageio.get_writer(
            density_anim_output,
            format='GIF-PIL',        # Explicitly use Pillow
            mode='I',
            # quantizer='nq',          # Temporarily removed
            palettesize=256,         # Ensure full palette
            duration=frame_duration_sec_v2,
            loop=0
        )
    except Exception as e:
        print_status(f"Error creating GIF writer: {e}")
        return # Cannot proceed without writer

    with mp.Pool(mp.cpu_count()) as pool:
        # Use a custom tqdm instance with two progress displays
        # First, create a standard tqdm for the counter on the first line
        # Determine description and format dynamically
        try:
            term_width = shutil.get_terminal_size().columns
        except OSError:
            term_width = 100 # Fallback width
        selected_desc, selected_bar_format = select_tqdm_format('gen_dens_frames', term_width)
        
        counter_tqdm = tqdm(
            total=total_frames,
            desc=selected_desc, # Use dynamic description
            unit="frame",
            position=0,
            leave=True,
            miniters=1,
            dynamic_ncols=True,
            ncols=None,
            bar_format=selected_bar_format
        )


        bar_tqdm = tqdm(
            total=total_frames,
            position=1,
            leave=True,
            dynamic_ncols=True,
            ncols=None,
            bar_format="{bar} {percentage:3.1f}%",
            ascii=False
        )

        # Process the frames and append directly to the writer
        frame_count = 0
        for frame_image in pool.imap(render_density_frame, frame_data_list):
            if frame_image is not None:
                try:
                    writer.append_data(frame_image) # Append directly to writer
                    frame_count += 1
                except Exception as e:
                    log_message(f"Error appending frame {frame_count+1}: {e}", level="error")
                    break
            # Update progress bars
            counter_tqdm.update(1)
            bar_tqdm.update(1)


        counter_tqdm.close()
        bar_tqdm.close()
        
        # Close the writer
        try:
            writer.close()
        except Exception as e:
            log_message(f"Error closing GIF writer: {e}", level="error")

        # No delay between frame generation and animation saving to improve fluidity

    # Log to file
    log_message(f"Generated and saved {frame_count}/{total_frames} density profile frames successfully to {density_anim_output}")

    if frame_count > 0:
        print_status(f"Animation saved: {get_file_prefix(density_anim_output)}")
        # Add separator line after animation completion
        sys.stdout.write(get_separator_line(char='-') + "\n")
    else:
        print_status("Failed to generate any density profile frames.")
    
    # Encourage garbage collection
    gc.collect()

def create_psi_animation(suffix, duration, config=None):
    """
    Create psi profile animation.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.
    duration : float
        Duration of each frame in milliseconds.
    config : Configuration, optional
        Configuration object containing plot settings. If provided,
        enables robust dynamic ranging for axes.
        
    Returns
    -------
    None
        Function does not return a value, but saves the animation to a file.
        
    Notes
    -----
    Uses parallel processing for rendering animation frames.
    Saves the animation incrementally using `imageio.get_writer` to reduce
    peak memory usage compared to collecting all frames first.
    Requires imageio v2 (`pip install imageio==2.*`).
    """
    global psi_snapshots, psi_max_value

    if not psi_snapshots:
        print_status("No psi data available for animation.")
        return
        
    # Calculate global maximum values for consistent scaling
    x_range, y_max = _calculate_profile_animation_ranges(
        psi_snapshots, config, data_index=2, x_default_max=250.0,
        animation_name="Psi"
    )
    
    if y_max is not None:
        psi_max_value = y_max
    else:
        calculate_global_max_values()

    total_frames = len(psi_snapshots)
    # Log detailed info to file only
    log_message(f"Generating psi profile frames for {total_frames} snapshots...")

    # Extract tfinal_factor from suffix if possible
    # Format is typically _[file_tag]_npts_Ntimes_tfinal_factor
    tfinal_factor = 5  # Default value
    parts = suffix.strip('_').split('_')
    if len(parts) >= 3:
        try:
            tfinal_factor = int(parts[-1])
        except (ValueError, IndexError):
            # Use default if parsing fails
            pass
            
    # Calculate r_max for consistent plotting - do this once before creating the pool
    if x_range is not None:
        r_max = x_range[1]  # Use the calculated range from helper function
    else:
        # Original hardcoded behavior when not using median ranges
        r_max = 1.1 * np.max([np.max(r) for _, r, _ in psi_snapshots if len(r) > 0])
        r_max = min(250, r_max)  # Cap at 250 kpc for reasonable display

    # Create frame_data tuples with the actual snapshot data, tfinal_factor, total_frames, ranges and project_root_path
    if not psi_snapshots:
        print_status("Error: psi_snapshots list is empty before pool creation.")
        return # Handle error appropriately
    
    # Prepare range information
    if config and hasattr(config, 'use_median_ranges') and config.use_median_ranges:
        # Use calculated robust ranges
        r_range = (0, r_max)  # Psi profiles always start at r=0
        p_range = (0, psi_max_value if psi_max_value > 0 else 0.072)
    else:
        # Use original behavior
        r_range = (0, r_max)
        p_range = (0, psi_max_value if psi_max_value > 0 else 0.072)
    
    frame_data_list = [(psi_snapshots[i], tfinal_factor, total_frames, r_range, p_range, PROJECT_ROOT) for i in range(total_frames)]

    # Set up imageio writer
    psi_anim_output = f"results/Psi_Profile_Animation{suffix}.gif"
    # Use seconds per frame for imageio v2 duration
    frame_duration_sec_v2 = duration / 1000.0 # Convert ms to seconds
    try:
        # Use mode='I' for multiple images, loop=0 for infinite loop
        writer = imageio.get_writer(
            psi_anim_output,
            format='GIF-PIL',        # Explicitly use Pillow
            mode='I',
            # quantizer='nq',          # Temporarily removed
            palettesize=256,         # Ensure full palette
            duration=frame_duration_sec_v2,
            loop=0
        )
    except Exception as e:
        print_status(f"Error creating GIF writer: {e}")
        return # Cannot proceed without writer

    with mp.Pool(mp.cpu_count()) as pool:
        # Use a custom tqdm instance with two progress displays
        # First, create a standard tqdm for the counter on the first line
        # Determine description and format dynamically
        try:
            term_width = shutil.get_terminal_size().columns
        except OSError:
            term_width = 100 # Fallback width
        selected_desc, selected_bar_format = select_tqdm_format('gen_psi_frames', term_width)
        
        counter_tqdm = tqdm(
            total=total_frames,
            desc=selected_desc, # Use dynamic description
            unit="frame",
            position=0,
            leave=True,
            miniters=1,
            dynamic_ncols=True,
            ncols=None,
            bar_format=selected_bar_format
        )


        bar_tqdm = tqdm(
            total=total_frames,
            position=1,
            leave=True,
            dynamic_ncols=True,
            ncols=None,
            bar_format="{bar} {percentage:3.1f}%",
            ascii=False
        )

        # Process the frames and append directly to the writer
        frame_count = 0
        for frame_image in pool.imap(render_psi_frame, frame_data_list):
            if frame_image is not None:
                try:
                    writer.append_data(frame_image) # Append directly to writer
                    frame_count += 1
                except Exception as e:
                    log_message(f"Error appending frame {frame_count+1}: {e}", level="error")
                    break
            # Update progress bars
            counter_tqdm.update(1)
            bar_tqdm.update(1)


        counter_tqdm.close()
        bar_tqdm.close()
        
        # Close the writer
        try:
            writer.close()
        except Exception as e:
            log_message(f"Error closing GIF writer: {e}", level="error")

        # No delay between frame generation and animation saving to improve fluidity

    # Log to file
    log_message(f"Generated and saved {frame_count}/{total_frames} psi profile frames successfully to {psi_anim_output}")

    if frame_count > 0:
        print_status(f"Animation saved: {get_file_prefix(psi_anim_output)}")
        # Add separator line after animation completion
        sys.stdout.write(get_separator_line(char='-') + "\n")
    else:
        print_status("Failed to generate any psi profile frames.")
    
    # Encourage garbage collection
    gc.collect()

def process_single_rank_file_for_histogram(suffix):
    """
    Process a single rank file (snapshot 0) for generating a histogram.
    This is more efficient than processing all snapshots when we only need one.

    Parameters
    ----------
    suffix : str
        Suffix for input/output files.

    Returns
    -------
    numpy.ndarray or None
        The processed data from snapshot 0, or None if not available.
    """
    # Find the initial snapshot file
    initial_file = f"data/Rank_Mass_Rad_VRad_sorted_t00000{suffix}.dat"

    # If the initial file doesn't exist, try to find any sorted rank file
    if not os.path.exists(initial_file):
        rank_files = glob.glob(f"data/Rank_Mass_Rad_VRad_sorted_t*{suffix}.dat")
        if rank_files:
            # Sort files to ensure we get the earliest snapshot
            rank_files.sort()
            initial_file = rank_files[0]
        else:
            print("No rank files found for histogram.")
            return None


    data = safe_load_and_filter_bin(initial_file, ncol_Rank_Mass_Rad_VRad_sorted,
                                  dtype=[np.int32, np.float32, np.float32, np.float32,
                                         np.float32, np.float32, np.float32, np.float32])
    if data is None or data.shape[0] == 0:
        print(f"No data to process in {initial_file}")
        return None

    return data

def generate_rank_histogram(rank_decimated_data, suffix, output_file=None, config=None):
    """
    Generate 2D histogram from rank data.

    Parameters
    ----------
    rank_decimated_data : list or numpy.ndarray
        Processed rank data, either as a list of (snap, data) tuples
        or as a single data array for a specific snapshot.
    suffix : str
        Suffix for input/output files.
    output_file : str, optional
        Custom output file path. If None, a default path will be used.
    config : dict, optional
        Configuration dictionary with plot settings
    """
    logger.info("Generating histograms from rank data...")

    # Check if input is a list of (snap, data) tuples or a single data array
    if isinstance(rank_decimated_data, list):
        # Processing list of (snap, data) tuples - locate snapshot 0
        initial_data = None
        for snap, decimated in rank_decimated_data:
            if snap == 0:
                initial_data = decimated
                break

        if initial_data is None:
            logger.warning("No initial snapshot found in the existing data")
            return None
    else:
        # Processing single data array (from process_single_rank_file_for_histogram)
        initial_data = rank_decimated_data

    # Extract columns from the decimated data
    ranks = initial_data[:, 0]
    masses = initial_data[:, 1]
    radii = initial_data[:, 2]
    vrad = initial_data[:, 3]
    psi = initial_data[:, 4]
    energy = initial_data[:, 5]
    angular_momentum = initial_data[:, 6]

    # Calculate total velocity (including angular momentum component)
    # Filter out zero radii before division
    nonzero_mask = radii > 0
    radii_nz = radii[nonzero_mask]
    vrad_nz = vrad[nonzero_mask]
    angular_momentum_nz = angular_momentum[nonzero_mask]

    if len(radii_nz) == 0:
        logger.warning("No particles with non-zero radius found in rank data for histogram.")
        return None

    # Calculate the total velocity in simulation internal units
    total_velocity_internal = np.sqrt(vrad_nz**2 + (angular_momentum_nz**2 / (radii_nz**2)))

    # Convert to km/s
    kmsec_to_kpcmyr = 1.0227e-3  # Conversion factor
    total_velocity = total_velocity_internal * (1.0 / kmsec_to_kpcmyr)

    # Set up histogram parameters using robust ranging
    # Calculate dynamic ranges using the robust range helper
    r_min, r_max = _calculate_robust_range(
        radii_nz, percentile=config.x_percentile,
        percentile_multiplier=config.x_percentile_multiplier,
        min_abs_extent=config.min_x_range_abs,
        default_if_empty=(0, 250.0), can_be_negative=False,
        axis_name="radius"
    )
    
    v_min, v_max = _calculate_robust_range(
        total_velocity, percentile=config.y_percentile,
        percentile_multiplier=config.y_percentile_multiplier,
        min_abs_extent=config.min_y_range_abs, 
        default_if_empty=(0, 320.0), can_be_negative=False,
        axis_name="velocity"
    )

    # Create the 2D histogram using non-zero radius data
    hist, xedges, yedges = np.histogram2d(
        radii_nz,
        total_velocity,
        bins=[400, 400],
        range=[[r_min, r_max], [v_min, v_max]]
    )

    # Transpose for correct orientation
    hist = hist.T


    plt.figure(figsize=(8, 6))
    plt.pcolormesh(xedges, yedges, hist, cmap='viridis')
    plt.colorbar(label='Counts')
    plt.xlabel(r'$r$ (kpc)', fontsize=12)
    plt.ylabel(r'$v$ (km/s)', fontsize=12)

    plt.title(r'Initial Phase Space Distribution', fontsize=14)
    plt.xlim(r_min, r_max)
    plt.ylim(v_min, v_max)


    if output_file is None:
        # Define default output filename
        output_file = f"results/part_data_histogram_initial{suffix}.png"
    plt.savefig(output_file, dpi=150)
    plt.close()

    # Log to file (progress tracking by the caller)
    logger.info(f"Histogram saved: {output_file}")

    return output_file

def process_variable_histograms(config):
    """
    Generates 1D variable distribution histograms comparing initial vs final distributions.
    Creates histograms for radius, velocity, radial velocity, and angular momentum.
    
    Note: This function creates 1D distributions, not 2D phase space plots,
    so it does not require robust ranging modifications.
    
    Parameters
    ----------
    config : Configuration
        The configuration object containing parsed arguments and settings.
        
    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    print_header("Generating 1D Variable Distribution Histograms")
    
    # Determine file paths directly within this function
    suffix = config.suffix
    initial_file = f"data/particles{suffix}.dat"
    final_unsorted_file = None
    unsorted_pattern_glob = f"data/Rank_Mass_Rad_VRad_unsorted_t*{suffix}.dat"
    log_message(f"Searching for unsorted files: {unsorted_pattern_glob}", level="debug")
    unsorted_files = glob.glob(unsorted_pattern_glob)
    log_message(f"Found {len(unsorted_files)} matching files.", level="debug")

    if unsorted_files:
        unsorted_regex_pattern = re.compile(r'Rank_Mass_Rad_VRad_unsorted_t(\d+)' + re.escape(suffix) + r'\.dat')
        try:
            unsorted_files.sort(key=lambda f: get_snapshot_number(f, pattern=unsorted_regex_pattern))
            last_file_candidate = unsorted_files[-1]
            if get_snapshot_number(last_file_candidate, pattern=unsorted_regex_pattern) != 999999999:
                final_unsorted_file = last_file_candidate
                log_message(f"Identified last unsorted snapshot: {os.path.basename(final_unsorted_file)}", level="info")
            else:
                log_message("Warning: Sorting unsorted files failed to identify latest.", level="warning")
        except Exception as e:
            log_message(f"Warning: Error sorting unsorted files: {e}. Using basic sort.", level="warning")
            unsorted_files.sort()
            if unsorted_files: 
                final_unsorted_file = unsorted_files[-1]

    # Prepare files_to_load_info using determined paths
    files_to_load_info = []
    if os.path.exists(initial_file):
        files_to_load_info.append((initial_file, ncol_particles_initial, np.float32))
    else:
        log_message(f"Initial particles file not found: {initial_file}", level="error")
        print_status(f"Error: Initial particles file not found: {initial_file}")
        return False  # Exit the function if initial file missing
        
    if final_unsorted_file and os.path.exists(final_unsorted_file):
        files_to_load_info.append((final_unsorted_file, ncol_Rank_Mass_Rad_VRad_unsorted, np.float32))
    else:
        err_msg = f"Last unsorted snapshot file not found or determined for suffix {suffix}."
        log_message(err_msg, level="error")
        print_status(f"Error: {err_msg}")
        return False  # Exit the function if final file missing

    # --- Data Loading Phase ---
    log_message("Starting data loading phase.", level="info")
    loading_section_key = "diagnostic_data_loading"
    start_combined_progress(loading_section_key, len(files_to_load_info))
    loaded_data = {}
    load_success = True
    for f_path, f_ncol, f_dtype in files_to_load_info:
        log_message(f"Loading data from: {f_path}", level="info")
        data = safe_load_and_filter_bin(f_path, f_ncol, f_dtype)
        loaded_data[f_path] = data
        if data is None:
            log_message(f"Failed to load or filter data from: {f_path}", level="warning")
            load_success = False
        update_combined_progress(loading_section_key, f_path)  # Pass file path for prefix

    initial_raw_data = loaded_data.get(initial_file)
    final_unsorted_raw_data = loaded_data.get(final_unsorted_file)

    if not load_success or initial_raw_data is None or final_unsorted_raw_data is None:
        log_message("Failed to load required data files.", level="error")
        print_status("Error: Failed to load required data files.")
        if loading_section_key in _combined_plot_trackers:
            _combined_plot_trackers.pop(loading_section_key, None)
            clear_line()
        return False
    log_message("Data loading complete.", level="info")

    log_message("Processing particle data...", level="info")
    init_r, init_vr, init_L, init_vtot, init_vperp = process_particle_data(initial_raw_data, ncol_particles_initial)
    final_r, final_vr, final_L, final_vtot, final_vperp = process_particle_data(final_unsorted_raw_data, ncol_Rank_Mass_Rad_VRad_unsorted)

    processed_data_valid = all(d is not None for d in [init_r, init_vr, init_L, init_vtot, init_vperp,
                                                      final_r, final_vr, final_L, final_vtot, final_vperp])
    if not processed_data_valid:
        log_message("Could not extract valid data after processing.", level="error")
        print_status("Error: Could not extract valid data after processing.")
        return False
    log_message("Particle data processing complete.", level="info")

    # --- Define plot specifications with refined labels and titles ---
    # Calculate robust ranges for radius and velocity if config enables it
    if config and hasattr(config, 'use_median_ranges') and config.use_median_ranges:
        # Calculate robust range for radius
        combined_r = np.concatenate([init_r, final_r])
        r_min, r_max = _calculate_robust_range(
            combined_r, 
            percentile=config.x_percentile,
            percentile_multiplier=config.x_percentile_multiplier,
            min_abs_extent=config.min_x_range_abs,
            default_if_empty=(0, 250.0), 
            can_be_negative=False,
            axis_name="1D radius histogram"
        )
        radius_range = [r_min, r_max]
        
        # Calculate robust range for velocity
        combined_v = np.concatenate([init_vtot, final_vtot])
        v_min, v_max = _calculate_robust_range(
            combined_v, 
            percentile=config.y_percentile,
            percentile_multiplier=config.y_percentile_multiplier,
            min_abs_extent=config.min_y_range_abs,
            default_if_empty=(0, 500.0), 
            can_be_negative=False,
            axis_name="1D velocity histogram"
        )
        velocity_range = [v_min, v_max]
    else:
        # Use original hardcoded ranges
        radius_range = [0, 250]
        velocity_range = [0, 500]
    
    plot_vars = [
        {'data1': init_r, 'data2': final_r,
         'xlabel': r'$r$ (kpc)',
         'title': r'Comparison of Radius Distribution $N(r)$',
         'filename': f'radius_hist_compare{suffix}.png', 'range': radius_range},

        {'data1': init_vtot, 'data2': final_vtot,
         'xlabel': r'$v$ (km/s)',
         'title': r'Comparison of Total Velocity Distribution $N(v)$',
         'filename': f'total_velocity_histogram_compare{suffix}.png', 'range': velocity_range},

        {'data1': init_vperp, 'data2': final_vperp,
         'xlabel': r'$v_{\perp}$ (km/s)',
         'title': r'Comparison of Tangential Velocity Magnitude Distribution $N(v_{\perp})$',
         'filename': f'tangential_velocity_histogram_compare{suffix}.png', 'range': None, 'plot_func': plot_tangential_velocity_histogram},

        {'data1': init_vr, 'data2': final_vr,
         'xlabel': r'$v_r$ (km/s)',
         'title': r'Comparison of Radial Velocity Distribution $N(v_r)$',
         'filename': f'radial_velocity_histogram_compare{suffix}.png', 'range': None},  # Auto range

        {'data1': init_L, 'data2': final_L,
         'xlabel': r'$\ell$ (simulation units)',  # Use \ell
         'title': r'Comparison of Angular Momentum Distribution $N(\ell)$',  # Use \ell
         'filename': f'angular_momentum_histogram_compare{suffix}.png', 'range': None}  # Auto range
    ]

    num_plots = len(plot_vars)
    log_message(f"Generating {num_plots} comparison histograms...", level="info")
    plots_generated = 0

    # --- Plot Saving Phase ---
    saving_section_key = "diagnostic_plot_saving"
    start_combined_progress(saving_section_key, num_plots)  # Uses default "Save:"

    for i, spec in enumerate(plot_vars, 1):
        output_path = os.path.join("results", spec['filename'])
        log_message(f"Generating plot: {output_path}", level="info")
        
        if 'plot_func' in spec:  # Check if a custom plot function is specified
            try:
                spec['plot_func'](spec['data1'], spec['data2'], output_path, plot_range=spec.get('range'))
                plots_generated += 1
                update_combined_progress(saving_section_key, output_path)
                log_message(f"Plot saved: {output_path}", level="debug")
            except Exception as e:
                clear_line()
                log_message(f"Error generating plot {output_path}: {e}", level="error")
                print(f"\nError generating plot for {spec['xlabel']}: {e}")
                update_combined_progress(saving_section_key, output_path)  # Pass path for prefix
        else:  # Original histogram plotting logic
            plt.figure(figsize=(10, 6))
            bins = 100
            hist_range = spec['range']
            if hist_range is None:
                combined = np.concatenate((spec['data1'], spec['data2']))
                finite = combined[np.isfinite(combined)]
                if finite.size > 0: 
                    hist_range = (np.min(finite), np.max(finite))
                else: 
                    hist_range = (-1, 1)

            try:
                plt.hist(spec['data1'], bins=bins, range=hist_range, alpha=0.6, color='blue', label='Initial', density=True)
                plt.hist(spec['data2'], bins=bins, range=hist_range, alpha=0.6, color='red', label='Final', density=True)
                plt.xlabel(spec['xlabel'], fontsize=12)
                plt.ylabel('Normalized Frequency', fontsize=12)
                plt.title(spec['title'], fontsize=14)
                plt.legend(fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)  # Match nsphere_plot grid style
                plt.tight_layout()
                plt.savefig(output_path, dpi=150)  # Consistent DPI
                plots_generated += 1
                update_combined_progress(saving_section_key, output_path)
                log_message(f"Plot saved: {output_path}", level="debug")
            except Exception as e:
                clear_line()
                log_message(f"Error generating plot {output_path}: {e}", level="error")
                print(f"\nError generating plot for {spec['xlabel']}: {e}")
                update_combined_progress(saving_section_key, output_path)  # Pass path for prefix
            finally:
                plt.close()

    # Corrected Footer message logic
    if plots_generated == num_plots:
        print_footer("1D Variable Comparison Histograms generated successfully.")
        log_message("1D Variable Comparison Histograms generated successfully.", level="info")
        return True
    else:
        final_message = f"1D Comparison Histograms partially generated ({plots_generated}/{num_plots})."
        print_status(final_message + " Check log for details.")
        log_message(final_message + " Some plots failed.", level="warning")
        return False

def generate_all_2d_histograms(config, rank_decimated_data=None):
    """
    Generates all 2D histogram plots including particle, nsphere, and rank histograms.

    Parameters
    ----------
    config : Configuration
        The configuration object.
    rank_decimated_data : list or None
        Pre-processed rank data that can be used as fallback for histogram generation.
    """
    print_header("Generating 2D Histograms")

    # Count total histogram plots (4 from particles and nsphere, 1 from rank)
    total_histograms = 5
    start_combined_progress("histogram_plots", total_histograms)

    # Process particles histograms with combined progress
    plot_particles_histograms(config.suffix, progress_callback=lambda output_file: update_combined_progress("histogram_plots", output_file), config=config)

    # Process nsphere histograms with combined progress
    plot_nsphere_histograms(config.suffix, progress_callback=lambda output_file: update_combined_progress("histogram_plots", output_file), config=config)

    # For rank histogram, only a single file is needed
    logger.info("Generating histograms from rank data...")
    single_rank_data = process_single_rank_file_for_histogram(config.suffix)

    # Generate rank histogram if we have data
    rank_output_file = f"results/part_data_histogram_initial{config.suffix}.png"
    if single_rank_data is not None:
        generate_rank_histogram(single_rank_data, config.suffix, rank_output_file, config)
        update_combined_progress("histogram_plots", rank_output_file)
    # No fallback available for histogram generation

    print_footer("2D histograms generated successfully.")

def generate_all_energy_plots(config, file_paths, unsorted_energy_data):
    """
    Generates all energy-related plots with status display and progress tracking.

    Displays the initial status, generates plots in parallel (unsorted and sorted 
    energy plots), and updates the console output with completion status.

    Parameters
    ----------
    config : Configuration
        The configuration object containing suffix and other settings.
    file_paths : dict or None
        Dictionary of file paths, will be created if None.
    unsorted_energy_data : list or None
        Pre-processed unsorted energy data from process_rank_files.
    
    Notes
    -----
    The console output is managed with a series of up/down cursor movements
    to create a clean, consistently updated display. Initial status appears
    immediately while plots are being generated, with the final update
    overwriting the status lines upon completion.
    """
    # Access the global paced_mode variable for the timer function
    global paced_mode

    # Setup file paths if they weren't provided
    if file_paths is None:
        file_paths = config.setup_file_paths()

    print_header("Generating Energy Plots", add_newline=False)

    # Import threading module
    import threading

    # Prepare static display lines
    energy_plot_start_time = time.time()
    unsorted_name = f"Energy_vs_timestep_unsorted{config.suffix}"
    sorted_name = f"Energy_vs_timestep_sorted{config.suffix}"

    # Log start of processing
    logger.info("Energy plots starting")

    # Set up progress bar appearance
    bar_length = 20
    half_filled = int(bar_length * 0.5)
    half_bar = '█' * half_filled + ' ' * (bar_length - half_filled)
    prefix = get_file_prefix(unsorted_name)

    # Timer thread will handle displaying the 50% unsorted status
    # print(f"Save: {half_bar} 50.0% | File: {prefix}")  # Commented out as timer thread shows this
    # print(get_separator_line(char='-'))

    # Create timer thread
    stop_timer = threading.Event()
    timer_thread = threading.Thread(target=update_timer_energy_plots, args=(stop_timer, energy_plot_start_time, paced_mode))
    timer_thread.daemon = True
    timer_thread.start()

    try:
        # Generate the unsorted energy plot
        if unsorted_energy_data:
            unsorted_output = generate_unsorted_energy_plot(unsorted_energy_data, config.suffix)
        else:
            unsorted_output = generate_unsorted_energy_plot([], config.suffix)

        # Stop the timer after first plot completes
        stop_timer.set()
        timer_thread.join(0.2)

        # --- Write separator line and move down ---
        # Timer status is on line X. Cursor is likely start of X+1.
        # Write separator on line X+1. Use truncate_and_pad for consistency.
        separator = get_separator_line(char='-')
        sys.stdout.write(f"{truncate_and_pad_string(separator)}\n") # Write separator + newline
        # Cursor is now at start of line X+2. Tqdm will start here.
        sys.stdout.flush()
        # --- End separator write + move down ---

        # Generate the sorted energy plot
        sorted_output = generate_sorted_energy_plot(config.suffix)
        
        # Generate total energy diagnostics plot if file exists
        total_energy_file = file_paths.get("total_energy_vs_time", "")
        if os.path.exists(total_energy_file):
            output_file = f"results/total_energy_diagnostics{config.suffix}.png"
            total_energy_output = plot_total_energy_diagnostics(total_energy_file, output_file)
            if total_energy_output:
                logger.info(f"Total energy diagnostics plot saved to {total_energy_output}")

        # Calculate final timing statistics
        elapsed = time.time() - energy_plot_start_time
        displayed_elapsed = max(elapsed, 0.01)
        nominal_elapsed = max(elapsed, 0.02)
        remaining = 0
        rate = 2.0 / nominal_elapsed
        rate = min(rate, 99.9)
        time_info = f" [{displayed_elapsed:.2f}<{remaining:.2f}, {rate:.1f}file/s]" # Final stats

    except Exception as e:
        # Stop timer thread on error
        if timer_thread.is_alive():
            stop_timer.set()
            timer_thread.join(0.2)
        
        # Log the error
        logger.error(f"Error during energy plot generation: {e}")
        logger.error(traceback.format_exc())
        print_status(f"Error during energy plot generation: {e}")
        
        # Mark plots as failed
        unsorted_output = None
        sorted_output = None

    # Show completion status
    full_bar = '█' * bar_length # Keep this
    final_prefix = get_file_prefix(sorted_name) # Use SORTED prefix

    # --- Start Final Update Logic ---
    # Move up 5 lines (1 init status + 1 init separator + 2 tqdm bars + 1 line below tqdm)
    move_up_final = "\033[5A"
    # Downward movement: We write status (line X), then separator (line X+1).
    # We need to land below where the second tqdm bar was (line X+4).
    # So, after writing separator on X+1, move down 3 lines.
    move_down_final = "\033[4B" # Relative Move

    sys.stdout.write(move_up_final) # Move up to initial status line (X)
    # Create the final status string
    content_string = f"Save: {full_bar} 100.0% | File: {final_prefix}{time_info}"
    # Clear line X and write final status
    sys.stdout.write(f"\r\033[2K{truncate_and_pad_string(content_string)}")

    # Move to next line (X+1), clear and write separator
    sys.stdout.write('\n') # Move cursor to start of line X+1
    sys.stdout.write(f"\r\033[2K{get_separator_line(char='-')}") # Clear/write separator

    # Move cursor down 4 lines relative to the current line (X+1) to land below tqdm area
    sys.stdout.write(move_down_final)
    sys.stdout.flush()
    # --- End Final Update Logic ---

    # Generate additional energy plots
    process_trajectory_energy_plots(file_paths, include_angular_momentum=False)

    # Display completion message
    logger.info("Energy data processing complete.")
    print_footer("Energy plots generated successfully.")

def display_configuration_summary(config):
    """
    Prints the execution banner and summarizes the configuration parameters.

    Parameters
    ----------
    config : Configuration
        The configuration object containing parsed arguments and settings.
    """
    # Display banner and parameter information
    print_header("NSphere Plot Generation Tool")

    # Display parameter values in a structured format
    print_status("\nParameter values requested:\n")
    print_status(f"  Number of Particles:          {config.npts}")
    print_status(f"  Number of Time Steps:         {config.Ntimes}")
    print_status(f"  Number of Dynamical Times:    {config.tfinal_factor}")
    print_status(f"  FPS:                          {config.fps} (frame duration: {config.duration:.1f}ms)")
    print_status(f"  Output Directory:             results")
    if config.file_tag:
        print_status(f"  Filename Tag:                 {config.file_tag}")
    else:
        print_status(f"  Filename Tag:                 [none]")
    if paced_mode:
        print_status(f"  Paced Mode:                   Enabled (section delay: 5.0s, progress bar delay: 2.0s)")
    else:
        print_status(f"  Paced Mode:                   Disabled (fast mode - no delays)")
    if enable_logging:
        print_status(f"  Logging:                      Full (log/nsphere_plot.log)")
    else:
        print_status(f"  Logging:                      Errors and warnings only (log/nsphere_plot.log)")
    print_status("")

def main():
    """
    Main function that orchestrates the visualization process.
    """
    global suffix, start_snap, end_snap, step_snap, duration, args
    global mass_snapshots, density_snapshots, psi_snapshots
    global showing_help

    # Create Configuration object (parses arguments and sets up parameters)
    config = Configuration()

    # If showing help, exit immediately
    if showing_help:
        return
        
    # Display configuration summary
    display_configuration_summary(config)

    # Set global variables from configuration
    args = config.args
    suffix = config.suffix
    start_snap = config.start_snap
    end_snap = config.end_snap
    step_snap = config.step_snap
    duration = config.duration

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Check if any "only" flags are active
    only_flags_active = config.only_specific_visualizations()


    file_paths = None
    rank_decimated_data = None

    # Only process phase space plots if specified or in normal mode
    if config.args.phase_space or not only_flags_active:
        # Skip if explicitly told not to generate phase space plots
        if not config.args.no_phase_space:
            print_header("Generating Phase Space Initial Histogram")
            generate_initial_phase_histogram(config.suffix, config=config)
            print_footer("Initial phase space histogram created successfully.")

            # If phase_space_only is true, also generate the animation right away
            if config.args.phase_space:
                print_header("Generating Phase Space Animation")
                generate_phase_space_animation(config, fps=config.fps)
                # Add separator and completion message for phase space animation
                sys.stdout.write(get_separator_line(char='-') + "\n")
                print_status("Phase space animation generated successfully.")
                # Log this message to file only, not to console
                log_message("Phase space animation created successfully.", "info")

    # Only process phase comparison if specified or in normal mode
    if config.args.phase_comparison or not only_flags_active:
        # Skip if explicitly told not to generate phase comparison
        if not config.args.no_phase_comparison:
            print_header("Generating Phase Space Comparison")
            generate_comparison_plot(config.suffix, config=config)
            print_footer("Phase space comparison created successfully.")
            
    # --- Generate 1D Variable Distributions ---
    if config.args.distributions or (not only_flags_active and not config.args.no_distributions):
        # This function handles its own header, data loading, processing, and plotting.
        process_variable_histograms(config)
        # Footer is printed by the function or progress bar completion

    # Process profile plots if requested or in normal mode
    if config.args.profile_plots or not only_flags_active:
        if not config.args.no_profile_plots:
            # Setup file paths
            file_paths = config.setup_file_paths()
            print_header("Generating Profile Plots")
            process_profile_plots(file_paths, config)
            print_footer("Profile plots generated successfully.")

    # Process trajectory plots if requested or in normal mode
    if config.args.trajectory_plots or not only_flags_active:
        if not config.args.no_trajectory_plots:
            # Setup file paths if not already set
            if file_paths is None:
                file_paths = config.setup_file_paths()
            print_header("Generating Trajectory and Diagnostic Plots")

            # Count the total plots to be generated from both functions
            trajectory_count = 3  # trajectories, single_trajectory, lowest_l_3panel
            energy_count = 3      # energy_vs_time, angular_momentum_vs_time, energy_compare

            # Initialize combined progress tracking for all trajectory-related plots
            total_plots = trajectory_count + energy_count
            start_combined_progress("trajectory_plots", total_plots)

            # Use a modified process_trajectory_plots function that updates the combined tracker
            process_trajectory_plots(file_paths)

            print_footer("Trajectory and diagnostic plots generated successfully.")

    # Generate 2D histograms if requested or in normal mode
    if getattr(config.args, '2d_histograms', False) or not only_flags_active:
        if not config.args.no_histograms:
            generate_all_2d_histograms(config, rank_decimated_data)

    # Generate convergence test plots if requested or in normal mode
    if config.args.convergence_tests or not only_flags_active:
        if not config.args.no_convergence_tests:
            print_header("Generating Convergence Test Plots")
            # The generate_convergence_test_plots function uses combined progress tracking
            generate_convergence_test_plots(config.suffix)
            print_footer("Convergence test plots generated successfully.")

    # Exit early if only specific, already-handled visualizations were requested
    # to avoid unnecessary rank file processing
    need_to_process_rank_files = config.need_to_process_rank_files()

    if only_flags_active and not need_to_process_rank_files:
        print_header("Visualization Complete")
        print_footer("All requested visualizations completed successfully.")
        return

     # Process rank files first if they'll be needed for animations or energy plots
    rank_decimated_data = None
    unsorted_energy_data = None # Initialize to ensure it's defined
    if need_to_process_rank_files:
        print_header("Processing Particle Snapshot Data")

        # Setup file paths if not already set
        if file_paths is None:
            file_paths = config.setup_file_paths()

        # Process rank files and prepare data for animations and energy plots
        # The function will print its own completion message
        _, unsorted_energy_data = process_rank_files(config.suffix, config.start_snap, config.end_snap, config.step_snap)
        rank_decimated_data = None

    # Process energy plots if requested or in normal mode
    if config.args.energy_plots or (not only_flags_active and not config.args.no_energy_plots):
        generate_all_energy_plots(config, file_paths, unsorted_energy_data)
       
    # Process animations if requested or in normal mode
    if config.args.animations or (not only_flags_active and not config.args.no_animations):
        print_header("Generating Animations")

        # Use parallel processing internally for animations; manage progress sequentially here
        # This provides parallel processing with clean progress output
        generate_all_1D_animations(config.suffix, config.duration, config)

        # Add completion message for profile animations (no separator needed here)
        print_status("Mass, density, and psi animations generated successfully.")

        # Generate phase space animation if not explicitly disabled and not already generated
        if not config.args.no_phase_space and not config.args.phase_space:  # Skip if already generated in phase-space-only mode
            print_header("Generating Phase Space Animation")
            generate_phase_space_animation(config, fps=config.fps) # Pass full config
            # Add completion message for phase space animation (separator is already added in the function)
            print_status("Phase space animation generated successfully.")
            # Log this message to file only, not to console
            log_message("Phase space animation created successfully.", "info")

        # Log this message to file only, not to console
        log_message("All animations generated successfully.", "info")


    # If doing only specific visualizations, exit now
    if only_flags_active:
        print_header("Visualization Complete")
        # Log completion message to file only, not to console
        log_message("All Visualizations Complete", "info")
        return
    else:

        print_header("All Visualizations Complete")
        # Log completion message to file only, not to console
        log_message("All visualizations generated successfully. Results saved to 'results' directory.", "info")

def run_main():
    """
    Main entry point that runs the application with the cursor hidden.
    Ensures cursor is always restored when the program exits.
    """
    try:
        # Hide cursor at program start
        hide_cursor()
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        # Re-raise any exceptions after showing cursor
        logger.error(f"Error: {str(e)}")
        logger.error(traceback.format_exc()) # Log full traceback
        # Print error to console as well for visibility
        print_status(f"\nAn error occurred: {str(e)}")
        raise e
    finally:
        # Always ensure cursor is visible before exiting
        show_cursor()

if __name__ == '__main__':
    run_main()