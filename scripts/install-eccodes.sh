#!/bin/bash
# 
ECCODES_DIR=/usr/local
mkdir -p /tmp/eccodes
cd /tmp/eccodes
wget https://confluence.ecmwf.int/download/attachments/45757960/eccodes-2.32.0-Source.tar.gz?api=v2 -O - | tar -xz -C . --strip-component=1 
#
mkdir /tmp/eccodes/build
cd /tmp/eccodes/build
cmake .. -DCMAKE_INSTALL_PREFIX=$ECCODES_DIR

make -j 4
sudo make install
rm -rf /tmp/eccodes
