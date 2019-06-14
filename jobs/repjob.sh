#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -p gpu_shared_course
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=axelbremer94@gmail.com

source activate dl
cp -a $HOME/chants "$TMPDIR"
cd "$TMPDIR"/chants

python $TMPDIR/chants/code/train.py --name Neumes --device cuda --seq_length 30 --representation neume --lstm_num_hidden 256 --lstm_num_layers 2
python $TMPDIR/chants/code/train.py --name Syllables --device cuda --seq_length 30 --representation syl --lstm_num_hidden 256 --lstm_num_layers 2
python $TMPDIR/chants/code/train.py --name Words --device cuda --seq_length 30 --representation word --lstm_num_hidden 256 --lstm_num_layers 2

wait
cp -au "$TMPDIR"/chants/output $HOME/chants