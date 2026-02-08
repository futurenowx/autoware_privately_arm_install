#!/bin/bash
set -e  # stop on error

# Get the actual user's home directory (works for any user, even with sudo)
if [ -n "$SUDO_USER" ]; then
    USER_HOME=$(eval echo ~$SUDO_USER)
    ACTUAL_USER="$SUDO_USER"
else
    USER_HOME="$HOME"
    ACTUAL_USER="$USER"
fi

echo "Running installation for user: $ACTUAL_USER"
echo "Home directory: $USER_HOME"

# Go to user's home directory
cd "$USER_HOME"

# Check if folder autoware.privately-owned-vehicles exists, if yes rename it to autoware.privately-owned-vehicles_old
if [ -d "autoware.privately-owned-vehicles" ]; then
    mv autoware.privately-owned-vehicles autoware.privately-owned-vehicles_old
    echo "✅ Folder renamed to autoware.privately-owned-vehicles_old"
else
    echo "ℹ️ Folder autoware.privately-owned-vehicles not found, skipping rename"
fi

# Check if folder exists autoware_projects, if yes, rename it to autoware_projects_old
if [ -d "autoware_projects" ]; then
    mv autoware_projects autoware_projects_old
    echo "✅ Folder renamed to autoware_projects_old"
else
    echo "ℹ️ Folder autoware_projects not found, skipping rename"
fi

cd "$USER_HOME"

# Clone Autoware repository
git clone https://github.com/autowarefoundation/autoware.privately-owned-vehicles.git

# Fix ownership and permissions if running as sudo
if [ -n "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER autoware.privately-owned-vehicles
    echo "✅ Ownership set to $SUDO_USER"
fi

# Set proper permissions (owner: read/write/execute, group: read/execute, others: read/execute)
chmod -R 755 autoware.privately-owned-vehicles
echo "✅ Permissions set correctly"

# Rename Models folder
mv autoware.privately-owned-vehicles/Models/visualizations \
   autoware.privately-owned-vehicles/Models/visualizations_old

# Go to autoware_privately_arm_install folder
cd "$USER_HOME/autoware_privately_arm_install" || { echo "❌ Cannot enter autoware_privately_arm_install folder"; exit 1; }

# Debug: List contents of current directory
echo "📂 Contents of autoware_privately_arm_install:"
ls -la

# copy folder autoware_projects to user home
SOURCE="autoware_projects"
DEST="$USER_HOME"
if [ -d "$SOURCE" ]; then
    cp -r "$SOURCE" "$DEST"
    echo "✅ autoware_projects folder copied successfully"
    # Fix ownership if running as sudo
    if [ -n "$SUDO_USER" ]; then
        chown -R $SUDO_USER:$SUDO_USER "$DEST/autoware_projects"
        echo "✅ Ownership set to $SUDO_USER for autoware_projects"
    fi
else
    echo "❌ Error: autoware_projects folder does not exist in $USER_HOME/autoware_privately_arm_install"
    exit 1
fi

####################################
# CREATE OUTPUT FOLDER STRUCTURE
####################################

# Create videos output directory and subdirectories
mkdir -p "$USER_HOME/autoware_projects/videos/output/EgoLanes"
mkdir -p "$USER_HOME/autoware_projects/videos/output/Scene3D"
mkdir -p "$USER_HOME/autoware_projects/videos/output/SceneSeg"
mkdir -p "$USER_HOME/autoware_projects/videos/output/DomainSeg"
mkdir -p "$USER_HOME/autoware_projects/videos/output/AutoSpeed"
mkdir -p "$USER_HOME/autoware_projects/videos/output/AutoSteer"

# Create images output directory and subdirectories
mkdir -p "$USER_HOME/autoware_projects/images/output/EgoLanes"
mkdir -p "$USER_HOME/autoware_projects/images/output/Scene3D"
mkdir -p "$USER_HOME/autoware_projects/images/output/SceneSeg"
mkdir -p "$USER_HOME/autoware_projects/images/output/DomainSeg"
mkdir -p "$USER_HOME/autoware_projects/images/output/AutoSpeed"
mkdir -p "$USER_HOME/autoware_projects/images/output/AutoSteer"

echo "✅ Created videos and images output folder structure"

# Fix ownership if running as sudo
if [ -n "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER "$USER_HOME/autoware_projects/videos"
    chown -R $SUDO_USER:$SUDO_USER "$USER_HOME/autoware_projects/images"
    echo "✅ Ownership set to $SUDO_USER for videos and images folders"
fi

# Go back to autoware_privately_arm_install to copy visualizations
cd "$USER_HOME/autoware_privately_arm_install"

# copy folder visualizations to autoware.privately-owned-vehicles/Models
SOURCE="visualizations"
DEST="$USER_HOME/autoware.privately-owned-vehicles/Models"

echo "🔍 Looking for visualizations folder at: $USER_HOME/autoware_privately_arm_install/$SOURCE"

if [ -d "$SOURCE" ]; then
    cp -r "$SOURCE" "$DEST"
    echo "✅ visualizations folder copied successfully"
    # Fix ownership if running as sudo
    if [ -n "$SUDO_USER" ]; then
        chown -R $SUDO_USER:$SUDO_USER "$DEST/visualizations"
        echo "✅ Ownership set to $SUDO_USER for visualizations"
    fi
else
    echo "❌ Error: visualizations folder does not exist"
    echo "Current directory: $(pwd)"
    echo "Looking for: $SOURCE"
    echo "Full path would be: $(pwd)/$SOURCE"
    exit 1
fi

echo "✅ Step1 install completed successfully, now download the weights and run 03_autoware_weights_install.sh."
echo "Please check the install.txt!"
