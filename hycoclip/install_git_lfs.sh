#!/bin/bash

# Script to download and install Git LFS without sudo
# This installs Git LFS to ~/.local/bin

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Installing Git LFS without sudo...${NC}"

# Create local bin directory if it doesn't exist
LOCAL_BIN="$HOME/.local/bin"
mkdir -p "$LOCAL_BIN"

# Detect architecture
ARCH=$(uname -m)
case $ARCH in
    x86_64)
        ARCH="amd64"
        ;;
    aarch64)
        ARCH="arm64"
        ;;
    armv7l)
        ARCH="arm"
        ;;
    *)
        echo -e "${RED}Unsupported architecture: $ARCH${NC}"
        exit 1
        ;;
esac

# Detect OS
OS=$(uname -s | tr '[:upper:]' '[:lower:]')

echo -e "${YELLOW}Detected OS: $OS, Architecture: $ARCH${NC}"

# Get latest release info from GitHub API
echo -e "${YELLOW}Fetching latest Git LFS release...${NC}"
LATEST_RELEASE=$(curl -s https://api.github.com/repos/git-lfs/git-lfs/releases/latest | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')

if [ -z "$LATEST_RELEASE" ]; then
    echo -e "${RED}Failed to fetch latest release info${NC}"
    exit 1
fi

echo -e "${GREEN}Latest version: $LATEST_RELEASE${NC}"

# Construct download URL
DOWNLOAD_URL="https://github.com/git-lfs/git-lfs/releases/download/$LATEST_RELEASE/git-lfs-$OS-$ARCH-$LATEST_RELEASE.tar.gz"

# Download Git LFS
echo -e "${YELLOW}Downloading Git LFS from: $DOWNLOAD_URL${NC}"
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

if ! curl -L -o git-lfs.tar.gz "$DOWNLOAD_URL"; then
    echo -e "${RED}Failed to download Git LFS${NC}"
    exit 1
fi

# Extract and install
echo -e "${YELLOW}Extracting and installing...${NC}"
tar -xzf git-lfs.tar.gz

# Find the git-lfs binary (it might be in a subdirectory)
GIT_LFS_BINARY=$(find . -name "git-lfs" -type f | head -1)

if [ -z "$GIT_LFS_BINARY" ]; then
    echo -e "${RED}Could not find git-lfs binary in extracted archive${NC}"
    echo -e "${YELLOW}Archive contents:${NC}"
    ls -la
    exit 1
fi

echo -e "${GREEN}Found git-lfs binary at: $GIT_LFS_BINARY${NC}"
cp "$GIT_LFS_BINARY" "$LOCAL_BIN/"
chmod +x "$LOCAL_BIN/git-lfs"

# Clean up
cd - > /dev/null
rm -rf "$TEMP_DIR"

# Check if ~/.local/bin is in PATH
if [[ ":$PATH:" != *":$LOCAL_BIN:"* ]]; then
    echo -e "${YELLOW}Adding $LOCAL_BIN to PATH...${NC}"
    
    # Add to ~/.bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        echo -e "${GREEN}Added to ~/.bashrc${NC}"
    fi
    
    # Add to ~/.profile as fallback
    if [ -f "$HOME/.profile" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.profile"
        echo -e "${GREEN}Added to ~/.profile${NC}"
    fi
    
    # Export for current session
    export PATH="$LOCAL_BIN:$PATH"
    echo -e "${YELLOW}Please run 'source ~/.bashrc' or restart your terminal to update PATH${NC}"
fi

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
if "$LOCAL_BIN/git-lfs" version; then
    echo -e "${GREEN}Git LFS installed successfully!${NC}"
else
    echo -e "${RED}Installation verification failed${NC}"
    exit 1
fi

# Initialize Git LFS
echo -e "${YELLOW}Initializing Git LFS...${NC}"
if "$LOCAL_BIN/git-lfs" install; then
    echo -e "${GREEN}Git LFS initialized successfully!${NC}"
    echo -e "${GREEN}You can now use 'git lfs' commands${NC}"
else
    echo -e "${RED}Git LFS initialization failed${NC}"
    exit 1
fi

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}Git LFS is installed in: $LOCAL_BIN/git-lfs${NC}" 