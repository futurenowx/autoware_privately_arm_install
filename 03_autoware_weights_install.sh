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

echo "Running weights installation for user: $ACTUAL_USER"
echo "Home directory: $USER_HOME"

####################################
# DOWNLOAD & PREPARE WEIGHTS
####################################
# Go to Downloads
cd "$USER_HOME/Downloads" || { echo "❌ Cannot enter Downloads folder"; exit 1; }

# Extract weights
unzip autoware_weights.zip -d "$USER_HOME/autoware_weights_temp"

# Fix ownership if running as sudo
if [ -n "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER "$USER_HOME/autoware_weights_temp"
fi

####################################
# CREATE WEIGHTS FOLDER STRUCTURE
####################################
cd "$USER_HOME"

# Create weights directory and subdirectories
mkdir -p autoware_projects/weights/EgoLanes
mkdir -p autoware_projects/weights/Scene3D
mkdir -p autoware_projects/weights/SceneSeg
mkdir -p autoware_projects/weights/DomainSeg
mkdir -p autoware_projects/weights/AutoSpeed
mkdir -p autoware_projects/weights/AutoSteer

echo "✅ Created weights folder structure"

# Fix ownership if running as sudo
if [ -n "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER "$USER_HOME/autoware_projects/weights"
fi

####################################
# COPY WEIGHTS INTO PROJECT
####################################

# Copy weights
cp "$USER_HOME/autoware_weights_temp/weights/EgoLanes"/weights_egolanes.pth \
   autoware_projects/weights/EgoLanes/

cp "$USER_HOME/autoware_weights_temp/weights/Scene3D"/weights_scene3d.pth \
   autoware_projects/weights/Scene3D/

cp "$USER_HOME/autoware_weights_temp/weights/SceneSeg"/weights_sceneseg.pth \
   autoware_projects/weights/SceneSeg/

cp "$USER_HOME/autoware_weights_temp/weights/DomainSeg"/weights_domainseg.pth \
   autoware_projects/weights/DomainSeg/

cp "$USER_HOME/autoware_weights_temp/weights/AutoSpeed"/weights_autospeed.pth \
   autoware_projects/weights/AutoSpeed/

cp "$USER_HOME/autoware_weights_temp/weights/AutoSteer"/weights_autosteer.pth \
   autoware_projects/weights/AutoSteer/

# Duplicate EgoLanes weights into AutoSteer (as requested)
cp "$USER_HOME/autoware_weights_temp/weights/EgoLanes"/weights_egolanes.pth \
   autoware_projects/weights/AutoSteer/

# Fix ownership of weights if running as sudo
if [ -n "$SUDO_USER" ]; then
    chown -R $SUDO_USER:$SUDO_USER "$USER_HOME/autoware_projects/weights"
    echo "✅ Ownership set to $SUDO_USER for weights"
fi

echo "✅ Weights downloaded, extracted, and copied successfully"

# Delete autoware_weights_temp if it exists
if [ -d "$USER_HOME/autoware_weights_temp" ]; then
    rm -rf "$USER_HOME/autoware_weights_temp"
    echo "✅ Deleted ~/autoware_weights_temp"
fi

# Delete autoware_privately_x86_install if it exists
if [ -d "$USER_HOME/autoware_privately_x86_install" ]; then
    rm -rf "$USER_HOME/autoware_privately_x86_install"
    echo "✅ Deleted ~/autoware_privately_x86_install"
fi

# Delete the ZIP file if it exists
if [ -f "$USER_HOME/Downloads/autoware_weights.zip" ]; then
    rm -f "$USER_HOME/Downloads/autoware_weights.zip"
    echo "✅ Deleted ~/Downloads/autoware_weights.zip"
fi
    
echo "✅ Step2 weights install completed successfully, final installation done!!!!"
