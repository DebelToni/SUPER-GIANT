#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Treat unset variables as an error when substituting.
set -u
# Pipestatus: exit status of the last command in a pipeline is used.
set -o pipefail

# --- Configuration ---
PYTHON_VERSION_FULL="3.13.3"
# Extract major.minor (e.g., 3.13)
PYTHON_VERSION_MAJOR_MINOR=$(echo "${PYTHON_VERSION_FULL}" | cut -d. -f1,2)

# Installation directory for this Python version
# Using /opt/python/ for custom Python installations is a common practice.
INSTALL_PREFIX="/opt/python/${PYTHON_VERSION_FULL}"

# Virtual environment configuration
VENV_NAME="my_app_env_${PYTHON_VERSION_MAJOR_MINOR}"
# Path where the virtual environment will be created.
# Defaulting to a 'venvs' directory in the user's home.
VENV_BASE_PATH="${HOME}/venvs"
VENV_PATH="${VENV_BASE_PATH}/${VENV_NAME}"

# Packages to install into the virtual environment
PACKAGES_TO_INSTALL=(
    "jax[cuda12]"  # Handles jax and jaxlib with CUDA 12 support
    "transformers"
    "datasets"
)

# Temporary directory for downloading and building
BUILD_DIR="/tmp/python_build_${PYTHON_VERSION_FULL}"

# Number of CPU cores to use for make (speeds up compilation)
# Uses all available cores by default.
NUM_CORES=$(nproc)

# --- Helper Functions ---
log() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

error_exit() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
    exit 1
}

# --- Main Script ---

log "Starting Python ${PYTHON_VERSION_FULL} installation, venv creation, and package installation script."

# 1. Install Build Dependencies
# -----------------------------
log "Updating package lists and installing build dependencies..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y \
        build-essential \
        zlib1g-dev \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libssl-dev \
        libreadline-dev \
        libffi-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libbz2-dev \
        liblzma-dev \
        tk-dev \
        uuid-dev \
        git || error_exit "Failed to install dependencies."
elif command -v dnf &> /dev/null; then
    # For Fedora/RHEL based systems (RunPod is usually Ubuntu, but good to have)
    sudo dnf groupinstall -y "Development Tools"
    sudo dnf install -y \
        gcc \
        openssl-devel \
        bzip2-devel \
        libffi-devel \
        zlib-devel \
        readline-devel \
        sqlite-devel \
        xz-devel \
        tk-devel \
        uuid-devel \
        wget \
        curl \
        llvm \
        git || error_exit "Failed to install dependencies."
else
    error_exit "Unsupported package manager. Please install dependencies manually."
fi
log "Build dependencies installed successfully."

# 2. Download Python Source Code
# ------------------------------
log "Creating build directory: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}" || error_exit "Failed to cd into ${BUILD_DIR}"

PYTHON_DOWNLOAD_URL="https://www.python.org/ftp/python/${PYTHON_VERSION_FULL}/Python-${PYTHON_VERSION_FULL}.tgz"
log "Downloading Python ${PYTHON_VERSION_FULL} from ${PYTHON_DOWNLOAD_URL}..."
wget -q "${PYTHON_DOWNLOAD_URL}" || error_exit "Failed to download Python source."
# Optional: Add GPG signature verification here for production use
log "Python source downloaded successfully."

# 3. Extract and Compile Python
# -----------------------------
log "Extracting Python-${PYTHON_VERSION_FULL}.tgz..."
tar -xf "Python-${PYTHON_VERSION_FULL}.tgz" || error_exit "Failed to extract Python source."
cd "Python-${PYTHON_VERSION_FULL}" || error_exit "Failed to cd into Python source directory."

log "Configuring Python build (prefix: ${INSTALL_PREFIX})..."
# --enable-optimizations: Enables expensive optimizations, build takes longer but Python runs faster.
# --with-ensurepip=install: Ensures pip is installed with Python.
# --enable-shared: Important for some packages that link against libpython.
# LDFLAGS with rpath helps find the shared library.
./configure \
    --prefix="${INSTALL_PREFIX}" \
    --enable-optimizations \
    --with-ensurepip=install \
    --with-lto \
    --enable-shared \
    CPPFLAGS="-I${INSTALL_PREFIX}/include" \
    LDFLAGS="-L${INSTALL_PREFIX}/lib -Wl,-rpath,${INSTALL_PREFIX}/lib" || error_exit "Python configure failed."

log "Compiling Python ${PYTHON_VERSION_FULL} (using ${NUM_CORES} cores)... This may take a while."
make -j"${NUM_CORES}" || error_exit "Python make failed."

log "Installing Python ${PYTHON_VERSION_FULL} using 'altinstall'..."
# 'altinstall' is used to prevent overwriting the system's default python3 binary.
sudo make altinstall || error_exit "Python make altinstall failed."
log "Python ${PYTHON_VERSION_FULL} installed to ${INSTALL_PREFIX}"

# Ensure the shared library cache is updated if needed, especially if not using rpath or if it's a system-wide install
if command -v ldconfig &> /dev/null; then
    log "Running ldconfig to update shared library cache..."
    sudo ldconfig "${INSTALL_PREFIX}/lib" || log "ldconfig command failed, this might be okay on some systems or if not needed."
else
    log "ldconfig not found. Manual update of shared library paths might be needed if issues arise."
fi


# 4. Verify the New Python Installation
# -------------------------------------
PYTHON_EXECUTABLE="${INSTALL_PREFIX}/bin/python${PYTHON_VERSION_MAJOR_MINOR}"
PIP_EXECUTABLE="${INSTALL_PREFIX}/bin/pip${PYTHON_VERSION_MAJOR_MINOR}" # This is the pip for the *installed* Python, not the venv yet.

if [ ! -f "${PYTHON_EXECUTABLE}" ]; then
    error_exit "Python executable ${PYTHON_EXECUTABLE} not found after installation."
fi
if [ ! -f "${PIP_EXECUTABLE}" ]; then
    error_exit "Pip executable ${PIP_EXECUTABLE} not found after installation."
fi

log "Verifying Python installation..."
INSTALLED_PYTHON_VERSION=$("${PYTHON_EXECUTABLE}" --version)
INSTALLED_PIP_VERSION=$("${PIP_EXECUTABLE}" --version)

log "Installed Python version: ${INSTALLED_PYTHON_VERSION}"
log "Installed Pip version for ${PYTHON_VERSION_FULL}: ${INSTALLED_PIP_VERSION}"

if [[ ! "${INSTALLED_PYTHON_VERSION}" == *"${PYTHON_VERSION_FULL}"* ]]; then
    error_exit "Installed Python version mismatch. Expected ~${PYTHON_VERSION_FULL}, Got ${INSTALLED_PYTHON_VERSION}"
fi

# 5. Create a Virtual Environment
# -------------------------------
log "Creating virtual environment at ${VENV_PATH}..."
mkdir -p "${VENV_BASE_PATH}" || error_exit "Failed to create base venv directory ${VENV_BASE_PATH}"

# Check if venv already exists
if [ -d "${VENV_PATH}" ]; then
    log "Virtual environment ${VENV_PATH} already exists. Skipping creation. Will attempt to install packages into it."
else
    "${PYTHON_EXECUTABLE}" -m venv "${VENV_PATH}" || error_exit "Failed to create virtual environment."
    log "Virtual environment created successfully at ${VENV_PATH}"
fi

# 6. Install Python Packages into Virtual Environment
# ---------------------------------------------------
VENV_PIP_EXECUTABLE="${VENV_PATH}/bin/pip"
if [ ! -f "${VENV_PIP_EXECUTABLE}" ]; then
    error_exit "Pip executable not found in virtual environment: ${VENV_PIP_EXECUTABLE}"
fi

log "Installing packages into virtual environment: ${VENV_PATH}"
log "Packages to install: ${PACKAGES_TO_INSTALL[*]}"

# Using -U to upgrade if already present (though in a new venv, it's a fresh install)
# Note: Ensure your RunPod instance has CUDA drivers compatible with JAX's CUDA 12 requirements.
# The script assumes CUDA toolkit and drivers are already on the system.
"${VENV_PIP_EXECUTABLE}" install -U "${PACKAGES_TO_INSTALL[@]}" || error_exit "Failed to install packages into virtual environment."

log "Packages installed successfully into the virtual environment."

# You can add a test for JAX GPU availability here if desired
# Example:
# log "Testing JAX GPU availability..."
# "${VENV_PATH}/bin/python" -c "import jax; print(f'JAX devices: {jax.devices()}'); print(f'JAX default backend: {jax.default_backend()}')"


# 7. Output Activation Instructions
# ---------------------------------
log "---------------------------------------------------------------------"
log "Python ${PYTHON_VERSION_FULL} installation, venv setup, and package installation complete!"
log "Python installed at: ${INSTALL_PREFIX}"
log "Python executable: ${PYTHON_EXECUTABLE}"
log ""
log "Virtual environment '${VENV_NAME}' is ready at: ${VENV_PATH}"
log "The following packages have been installed into the venv:"
for pkg in "${PACKAGES_TO_INSTALL[@]}"; do
  log "  - ${pkg}"
done
log ""
log "To activate the virtual environment, run:"
log "  source ${VENV_PATH}/bin/activate"
log "Once activated, 'python' and 'pip' will use the environment's versions."
log "---------------------------------------------------------------------"

# 8. Clean Up
# -----------
log "Cleaning up build directory: ${BUILD_DIR}..."
cd /tmp # Move out of the build directory before removing it
sudo rm -rf "${BUILD_DIR}"
log "Build directory cleaned up."

log "Script finished successfully."
exit 0

