#!/bin/bash

# Generate a random string of 7 alphabetic characters for the folder name
folder_name=$(tr -dc 'a-zA-Z' </dev/urandom | head -c 7)
mkdir "$folder_name"
mkdir "$folder_name/extensionne"
cd "$folder_name/extensionne" || exit
folder_path=$(pwd)

# Download and unzip the required files
wget -q https://github.com/BingChillz/Propeller/archive/refs/heads/main.zip -O propeller.zip
unzip -q propeller.zip

wget -q https://github.com/sr2echa/thottathukiduven-v2/archive/refs/heads/main.zip -O thottathukiduven-v2.zip
unzip -q thottathukiduven-v2.zip

# Function to clean up the user data directory after Chrome closes
cleanup() {
    echo "Cleaning up..."
    rm -rf "$folder_path"
    echo "User data directory deleted."
}

# Set up a trap to call the cleanup function on script exit
trap cleanup EXIT

# Launch Google Chrome with the required extensions in the background
nohup google-chrome --user-data-dir="$folder_path" \
    --load-extension="$(pwd)/Propeller-main","$(pwd)/thottathukiduven-v2-main" \
    --no-first-run &

# Disown the background process to detach it from the terminal
disown
