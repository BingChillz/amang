#!/bin/bash

wget https://github.com/sr2echa/thottathukiduven-v2/archive/refs/heads/main.zip -O thottathukiduven-v2.zip
unzip thottathukiduven-v2.zip

wget https://github.com/0xRad1ant/Enable-Copy-Paste-neocolab/archive/refs/heads/main.zip/ -O main.zip
unzip main.zip

wget https://github.com/BingChillz/Propeller/archive/refs/heads/main.zip -O real.zip
unzip real.zip

google-chrome --load-extension="$(pwd)/thottathukiduven-v2-main","$(pwd)/Enable-Copy-Paste-neocolab-main/Package","$(pwd)/Propeller-main"
