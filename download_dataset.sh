#!/usr/bin/env bash
mkdir -p ./data/RNA_data
curl -SL "https://archive.ics.uci.edu/ml/machine-learning-databases/00401/TCGA-PANCAN-HiSeq-801x20531.tar.gz" | tar -xvz -C data/RNA_data --transform='s/.*\///'
