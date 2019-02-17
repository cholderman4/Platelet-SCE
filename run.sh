#!/bin/csh

#$ qrsh
#$ -q  gpu-debug    # Specify queue
#$ -l gpu_card=1
#$ -pe smp 4        # specifies threads??? maybe
#$ -N  Test	        # Specify job name


module purge
module load gcc/6.2.0
module load cuda/9.1
echo -n "It is currently: ";date
echo -n "I am logged on as ";whoami
echo -n "This computer is called ";hostname
echo -n "I am currently in the directory ";pwd
