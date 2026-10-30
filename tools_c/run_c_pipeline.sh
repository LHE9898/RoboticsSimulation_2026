#!/usr/bin/env bash
# C-axis (grasp_err) pipeline after annotation: generate -> add grasp_err -> train 4 runs -> recorded eval.
#   conda activate lerobot-arena && cd ~/IsaacLab_RS
#   bash tools_c/run_c_pipeline.sh [generate|convert|smoke|train|eval|all]
# Each stage can be run on its own; "all" runs generate..eval in order.
# Second round (fingertip-frame errors, imitation_pland_v2.md):
#   bash tools_c/run_c_pipeline.sh [convert_rel|smoke_rel|train_rel|eval_rel|rel]
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
MIMIC_TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0
ANNOTATED=./datasets/team10_lee_s3_annotated.hdf5
N_MIMIC=500  # same generation count as the earlier mimic_v2_500 run
MIMIC=./datasets/team10_lee_s3_mimic_$N_MIMIC.hdf5
GE=./datasets/mimic_${N_MIMIC}_ge.hdf5
REL=./datasets/mimic_${N_MIMIC}_rel.hdf5  # same demos as $GE + grasp_err_ee, place_err_ee
SEEDS="101 111"
EPOCH=2000
export PYTHONUNBUFFERED=1

latest_ckpt() {  # $1 = experiment name (with _seedN suffix)
    ls -td ./logs/robomimic/$TASK/$1/*/ | head -1 | sed "s|/$|/models/model_epoch_$EPOCH.pth|"
}

stage_generate() {
    ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
        --task $MIMIC_TASK --headless --num_envs 100 --generation_num_trials $N_MIMIC \
        --input_file $ANNOTATED --output_file $MIMIC
}

stage_convert() {
    # keep at most N_MIMIC successful generated demos; same offset/frame as the env ObsTerm
    ./isaaclab.sh -p tools_c/add_grasp_err_to_hdf5.py --input $MIMIC --output $GE --frame base --max_demos $N_MIMIC
}

stage_smoke() {  # ~50 epochs, then check shape_metadata contains grasp_err: [6]
    echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
        --algo bc_grasperr --dataset $GE --epochs 50 --name smoke_grasperr
    python -c "
import glob, torch
p = sorted(glob.glob('./logs/robomimic/$TASK/smoke_grasperr/*/models/*.pth'))[-1]
print(p, torch.load(p, map_location='cpu', weights_only=False)['shape_metadata']['all_shapes'])"
}

stage_train() {
    for SEED in $SEEDS; do
        echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
            --algo bc --dataset $GE --seed $SEED
        echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
            --algo bc_grasperr --dataset $GE --seed $SEED
    done
}

stage_eval() {
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        for PAIR in base:bc_rnn_low_dim_agibot_toy2box ge:bc_rnn_low_dim_agibot_toy2box_grasperr; do
            OUT=./outputs/rollouts/${PAIR%%:*}_s$SEED
            ./isaaclab.sh -p tools_c/play_record.py --task $TASK --checkpoint "$(latest_ckpt ${PAIR#*:}_seed$SEED)" \
                --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT | tee $OUT.log
        done
    done
    python tools_c/analyze_rollouts.py \
        --baseline outputs/rollouts/base_s101 outputs/rollouts/base_s111 \
        --exp outputs/rollouts/ge_s101 outputs/rollouts/ge_s111 | tee outputs/rollouts/judgement.md
}

stage_convert_rel() {
    ./isaaclab.sh -p tools_c/add_grasp_err_to_hdf5.py --input $MIMIC --output $REL --frame base --max_demos $N_MIMIC
}

stage_smoke_rel() {  # 2 epochs each, then check shape_metadata contains the new keys
    for ALGO in bc_ge_ee bc_ge_ee_place; do
        echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
            --algo $ALGO --dataset $REL --epochs 2 --name smoke_$ALGO
    done
    python -c "
import glob, torch
for a in ['bc_ge_ee', 'bc_ge_ee_place']:
    p = sorted(glob.glob(f'./logs/robomimic/$TASK/smoke_{a}/*/models/*.pth'))[-1]
    print(a, torch.load(p, map_location='cpu', weights_only=False)['shape_metadata']['all_shapes'])"
}

stage_train_rel() {
    for SEED in $SEEDS; do
        for ALGO in bc_ge_ee bc_ge_ee_place; do
            echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
                --algo $ALGO --dataset $REL --seed $SEED
        done
    done
}

stage_eval_rel() {
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        for PAIR in geee:bc_rnn_low_dim_agibot_toy2box_ge_ee gepl:bc_rnn_low_dim_agibot_toy2box_ge_ee_place; do
            OUT=./outputs/rollouts/${PAIR%%:*}_s$SEED
            # rollout-episode-K.mp4 in videos/rel/<run> matches rollout_K.npz in $OUT
            rm -rf $OUT ./videos/rel/${PAIR%%:*}_s$SEED
            ./isaaclab.sh -p tools_c/play_record.py --task $TASK --checkpoint "$(latest_ckpt ${PAIR#*:}_seed$SEED)" \
                --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT \
                --video --video_length 800 --video_folder ./videos/rel/${PAIR%%:*}_s$SEED | tee $OUT.log
        done
    done
    R=outputs/rollouts
    {
        echo "## frame: grasp_err_ee vs grasp_err (base)"
        python tools_c/analyze_rollouts.py --criteria frame --baseline $R/ge_s101 $R/ge_s111 --exp $R/geee_s101 $R/geee_s111
        echo; echo "## place: grasp_err_ee + place_err_ee vs grasp_err_ee"
        python tools_c/analyze_rollouts.py --criteria place --baseline $R/geee_s101 $R/geee_s111 --exp $R/gepl_s101 $R/gepl_s111
    } | tee $R/judgement_v2.md
}

case "${1:-all}" in
    generate) stage_generate ;;
    convert) stage_convert ;;
    smoke) stage_smoke ;;
    train) stage_train ;;
    eval) stage_eval ;;
    all) stage_generate; stage_convert; stage_smoke; stage_train; stage_eval ;;
    convert_rel) stage_convert_rel ;;
    smoke_rel) stage_smoke_rel ;;
    train_rel) stage_train_rel ;;
    eval_rel) stage_eval_rel ;;
    rel) stage_train_rel; stage_eval_rel ;;
    *) echo "unknown stage: $1"; exit 1 ;;
esac
