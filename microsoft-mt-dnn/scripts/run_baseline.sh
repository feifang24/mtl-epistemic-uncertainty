for VAR in 1 2 3
do
    python ../run.py --train --data-dir ../../glue_data --mc-dropout-samples 8
done