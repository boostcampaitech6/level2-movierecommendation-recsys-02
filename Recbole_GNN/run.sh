#!/bin/bash




model_name=(BPR NeuMF SGL HMLET NCL SimGCL XSimGCL)

for md in "${model_name[@]}"
do
python run_recbole_gnn.py --model "${md}" --data "ML"
done

echo "model tests finished" &
