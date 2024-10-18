#!/bin/bash

# Generate a random string of 7 alphabetic characters for the folder name
folder_name=$(tr -dc 'a-zA-Z' </dev/urandom | head -c 7)
mkdir "$folder_name"
mkdir "extensionne"
cd "$folder_name/extensionne" || exit
folder_path=$(pwd)

# Download and unzip the required files
wget -q https://github.com/BingChillz/Propeller/archive/refs/heads/main.zip -O propeller.zip
unzip -q propeller.zip

wget -q https://github.com/sr2echa/thottathukiduven-v2/archive/refs/heads/main.zip -O thottathukiduven-v2.zip
unzip -q thottathukiduven-v2.zip

#wget -q https://github.com/0xRad1ant/Enable-Copy-Paste-neocolab/archive/refs/heads/main.zip -O neocolab.zip
#unzip -q neocolab.zip

#wget -q https://github.com/jswanner/DontF-WithPaste/archive/refs/heads/master.zip -O paste.zip
#unzip -q paste.zip

#wget -q https://github.com/brian-girko/always-active/archive/refs/heads/master.zip -O window.zip
#unzip -q window.zip

# Launch Google Chrome with the required extensions
google-chrome --user-data-dir="$folder_path" \
    --load-extension="$(pwd)/Propeller-main","$(pwd)/thottathukiduven-v2-main"#,"$(pwd)/Enable-Copy-Paste-neocolab-main","$(pwd)/DontF-WithPaste-master","$(pwd)/always-active-master"
