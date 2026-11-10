#!/bin/bash
#PBS -N rag_job
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o rag_output.log

cd /lustre/home/lshehu/rag_project

echo "RUNNING FROM:"
pwd
ls -l

source rag_env/bin/activate

export PYTHONPATH=/lustre/home/lshehu/rag_project

echo "START PYTHON"

python -u main.py

echo "DONE"
