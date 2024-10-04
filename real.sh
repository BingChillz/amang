#!/bin/bash

wget https://github.com/sr2echa/thottathukiduven-v2/archive/refs/heads/main.zip -O thottathukiduven-v2.zip
unzip thottathukiduven-v2.zip

wget https://github.com/0xRad1ant/Enable-Copy-Paste-neocolab/archive/refs/heads/main.zip/ -O main.zip
unzip main.zip

wget https://github.com/BingChillz/Propeller/archive/refs/heads/main.zip -O real.zip
unzip real.zip

wget https://github.com/jswanner/DontF-WithPaste/archive/refs/heads/master.zip -O paste.zip
unzip paste.zip

wget https://github.com/brian-girko/always-active/archive/refs/heads/master.zip -O window.zip
unzip window.zip

#"$(pwd)/thottathukiduven-v2-main",

google-chrome --load-extension="$(pwd)/Enable-Copy-Paste-neocolab-main/Package","$(pwd)/Propeller-main","$(pwd)/DontF-WithPaste-master","$(pwd)/always-active-master/v3
"
