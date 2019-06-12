#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared_course
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axelbremer94@gmail.com

source activate dl
cp -a $HOME/chants "$TMPDIR"
cd "$TMPDIR"/chants

python $TMPDIR/chants/code/train.py --name Claus --device cuda --seq_length 30 --lstm_num_hidden 256 --lstm_num_layers 2
python $HOME/chants/code/train.py --name Dolores --device cuda --seq_length 60 --lstm_num_hidden 256 --lstm_num_layers 2
python $HOME/chants/code/train.py --name Edward --device cuda --seq_length 30 --lstm_num_hidden 128 --lstm_num_layers 1

wait
cp -au "$TMPDIR"/chants/output $HOME/chants