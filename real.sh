#!/bin/bash

# Generate a random string of 7 alphabetic characters for the folder name
folder_name=$(tr -dc 'a-zA-Z' </dev/urandom | head -c 7)
mkdir "$folder_name"
cd "$folder_name" || exit
folder_path=$(pwd)
mkdir "extensionne"
cd "extensionne" || exit

# Download and unzip the required files
wget -q https://github.com/BingChillz/Propeller/archive/refs/heads/main.zip -O propeller.zip
unzip -q propeller.zip
rm propeller.zip

wget -q https://github.com/sr2echa/thottathukiduven-v2/archive/refs/heads/main.zip -O thottathukiduven-v2.zip
unzip -q thottathukiduven-v2.zip
rm thottathukiduven-v2.zip

wget -q https://github.com/waltuhf/NeoTrident/archive/refs/heads/main.zip -O neoexam.zip
unzip -q neoexam.zip

wget -q https://github.com/jswanner/DontF-WithPaste/archive/refs/heads/master.zip -O paste.zip
unzip -q paste.zip

wget -q https://github.com/brian-girko/always-active/archive/refs/heads/master.zip -O window.zip
unzip -q window.zip

# Function to clean up the user data directory
cleanup() {
    echo "Cleaning up..."
    rm -rf "$folder_path"
    echo "User data directory deleted."
}

# Launch Google Chrome with the required extensions in the background
google-chrome --user-data-dir="$folder_path" \
    --load-extension="$(pwd)/Propeller-main","$(pwd)/NeoTrident-main","$(pwd)/DontF-WithPaste-master","$(pwd)/always-active-master/v3" \
    --no-first-run &

chrome_pid=$!

# Wait for Chrome to close
wait $chrome_pid

# Run cleanup after Chrome closes
cleanup
