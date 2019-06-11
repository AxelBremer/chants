#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared_course
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axelbremer94@gmail.com

NAME=Abraham

source activate dl
cp -a $HOME/chants "$TMPDIR"
cd "$TMPDIR"/chants

python $HOME/chants/code/train.py --name $NAME --device cuda

wait
cp -au "$TMPDIR"/chants/output/$NAME $HOME/chants/output