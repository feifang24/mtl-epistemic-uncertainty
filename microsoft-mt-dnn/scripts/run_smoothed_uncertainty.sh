for VAR in 1 2
do
    python ../run.py --train --data-dir ../../glue_data --uncertainty-based-sampling --mc-dropout-samples 8 --smooth-uncertainties 0.25
done
for VAR in 1 2 3
do
    python ../run.py --train --data-dir ../../glue_data --uncertainty-based-sampling --mc-dropout-samples 8 --smooth-uncertainties 0.125
done
for VAR in 1 2 3
do
    python ../run.py --train --data-dir ../../glue_data --uncertainty-based-sampling --mc-dropout-samples 8 --smooth-uncertainties 0.375
done
