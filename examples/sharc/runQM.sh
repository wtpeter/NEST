#!/bin/sh
set -eu
cd QM
python -m nest.interfaces.sharc_nttda QM.in --config nttda.json >> QM.log 2>> QM.err
