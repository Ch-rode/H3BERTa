#!/bin/bash

lr=(5e-5 1e-6 1e-5) #
config=(config3.json)
batch_size=(1024) #1024 
subpipeline=(SUB-PIPELINE3:Igall_Bsources SUB-PIPELINE4:Igall_UnsortedBcells) 

for b in "${batch_size[@]}"; do
    for r in "${lr[@]}"; do
        for c in "${config[@]}"; do
            for s in "${subpipeline[@]}"; do
            n_sub=$(echo "$s" | grep -oP '(?<=SUB-PIPELINE)\d')
        
            job_content="#!/bin/bash
#SBATCH -J ${n_sub}_${r}_${b}_${c}
#SBATCH --gres=gpu:a100:1

mkdir ./${s}/${c}_lr${r}_bs${b}/
echo '*** TRAINING : ${c}_lr${r}_bs${b} ***' >> ./${s}/${c}_lr${r}_bs${b}/${c}_lr${r}_bs${b}.log
echo '*** SUBPIPELINE ${s} ***' >> ./${s}/${c}_lr${r}_bs${b}/${c}_lr${r}_bs${b}.log

python ./src/run_mlm_no_trainer_perplexity_blosum.py \
    --model_type 'roberta' \
    --tokenizer_name ./src/ProteinTokenizer \
    --train_file /ibmm_data/rodelc/DALM/LM/HEAVY/CDRH3/LM/HEALTHY/P3-pipelines/data/${s}/5_TRAINING_SETS/train.csv \
    --validation_file /ibmm_data/rodelc/DALM/LM/HEAVY/CDRH3/LM/HEALTHY/P3-pipelines/data/${s}/5_TRAINING_SETS/val.csv \
    --num_train_epochs 150 \
    --lr_scheduler_type 'linear' \
    --seed 42 \
    --data_seed 42 \
    --project_name DALM_CDRH3_PIPELINE3 \
    --weight_decay 0.1 \
    --max_seq_length 100 \
    --report_to 'wandb' \
    --with_tracking \
    --checkpointing_steps 'epoch' \
    --per_device_train_batch_size ${b} \
    --per_device_eval_batch_size ${b} \
    --output_dir ./${s}/${c}_lr${r}_bs${b} \
    --learning_rate ${r} \
    --run_name '${s}_${c%.json}_lr${r}_bs${b}' \
    --config_name ./src/${c} >> ./${s}/${c}_lr${r}_bs${b}/${c}_lr${r}_bs${b}.log"

                # Write the job content to a .job file
                job_filename="${s}_${c}_lr${r}_bs${b}.job"
                echo "$job_content" > "$job_filename"
                sbatch "$job_filename"
            done
        done
    done
done



