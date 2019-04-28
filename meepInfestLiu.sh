#!/bin/bash
module add prog/miniconda3/4.5.11
echo "module add prog/miniconda3/4.5.11" >> ~/.bashrc
conda create -n mp -c conda-forge pymeep
conda create -n pmp -c conda-forge pymeep=*=mpi_mpich_*

echo "Hej Kalle"
echo "Activera mp genom att skriva"
echo "  source activate mp"
echo "Activera pmp genom att skriva"
echo "  source activate pmp"