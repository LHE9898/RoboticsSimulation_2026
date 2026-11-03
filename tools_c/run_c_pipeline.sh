#!/usr/bin/env bash
# C-axis (grasp_err) pipeline after annotation: generate -> add grasp_err -> train 4 runs -> recorded eval.
#   conda activate lerobot-arena && cd ~/IsaacLab_RS
#   bash tools_c/run_c_pipeline.sh [generate|convert|smoke|train|eval|all]
# Each stage can be run on its own; "all" runs generate..eval in order.
# Second round (fingertip-frame errors, imitation_pland_v2.md):
#   bash tools_c/run_c_pipeline.sh [convert_rel|smoke_rel|train_rel|eval_rel|rel]
# Gripper event classifier (results/event_clf/plan_B.md):
#   bash tools_c/run_c_pipeline.sh [convert_evt|smoke_evt|train_evt|calib_evt|eval_evt|evt]
# evr with the geometric close rule (results/event_clf/plan_XY.md):
#   bash tools_c/run_c_pipeline.sh eval_rule
# evr + close rule with arm actions in EE axes (results/event_clf/plan_EE.md):
#   bash tools_c/run_c_pipeline.sh [convert_ee|replay_ee|train_ee|calib_ee|eval_ee|ee]
#   bash tools_c/run_c_pipeline.sh [convert_rel2|train_rel2|calib_rel2|eval_rel2|rel2]   (relative-only obs, plan_REL.md)
#   bash tools_c/run_c_pipeline.sh eval_z2   (eez2 = EE-action checkpoints + close rule 0.04,0.02, judged against evrz)
#   bash tools_c/run_c_pipeline.sh [train_base|eval_base]   (baseline: algo bc, 23D obs)
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
MIMIC_TASK=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0
ANNOTATED=./datasets/team10_lee_s3_annotated.hdf5
N_MIMIC=500  # same generation count as the earlier mimic_v2_500 run
MIMIC=./datasets/team10_lee_s3_mimic_$N_MIMIC.hdf5
GE=./datasets/mimic_${N_MIMIC}_ge.hdf5
REL=./datasets/mimic_${N_MIMIC}_rel.hdf5  # same demos as $GE + grasp_err_ee, place_err_ee
EVT=./datasets/mimic_${N_MIMIC}_evt.hdf5  # same as $REL + grip_state / grip_event_* labels (bc_event)
EVT_EE=./datasets/mimic_${N_MIMIC}_evt_ee.hdf5  # same as $EVT, arm actions in EE axes
TASK_EE=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-EEAct-v0
EVT_REL=./datasets/mimic_${N_MIMIC}_evt_ee_rel.hdf5  # same as $EVT_EE + obs/gravity_ee, event stats over the relative obs
TASK_REL=Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-EEActRel-v0
AGENTS=source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/agents/robomimic
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

stage_convert_evt() {
    python tools_c/add_grip_event_to_hdf5.py --input $REL --output $EVT
}

stage_smoke_evt() {  # CPU only and without the simulator (train.py launches Isaac Sim)
    for V in ready ready_inbox; do
        CUDA_VISIBLE_DEVICES= python tools_c/smoke_bc_event.py --config $AGENTS/bc_event_$V.json --data $EVT --out /tmp/bc_event_smoke_$V
    done
}

stage_train_evt() {
    for SEED in $SEEDS; do
        for ALGO in bc_event_ready bc_event_ready_inbox; do
            echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
                --algo $ALGO --dataset $EVT --seed $SEED
        done
    done
}

stage_calib_evt() {  # gripper thresholds picked on the training demos -> <run>/event_tau.json (CPU)
    for SEED in $SEEDS; do
        for EXP in bc_rnn_low_dim_agibot_toy2box_event_ready bc_rnn_low_dim_agibot_toy2box_event_ready_inbox; do
            CUDA_VISIBLE_DEVICES= python tools_c/calibrate_event_tau.py --ckpt "$(latest_ckpt ${EXP}_seed$SEED)"
        done
    done
}

stage_eval_evt() {
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        for PAIR in evr:bc_rnn_low_dim_agibot_toy2box_event_ready evi:bc_rnn_low_dim_agibot_toy2box_event_ready_inbox; do
            OUT=./outputs/rollouts/${PAIR%%:*}_s$SEED
            rm -rf $OUT ./videos/evt/${PAIR%%:*}_s$SEED
            ./isaaclab.sh -p tools_c/play_record.py --task $TASK --checkpoint "$(latest_ckpt ${PAIR#*:}_seed$SEED)" \
                --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT \
                --video --video_length 800 --video_folder ./videos/evt/${PAIR%%:*}_s$SEED | tee $OUT.log
        done
    done
    R=outputs/rollouts
    {
        echo "## primary: bc_event (ready) vs gepl"
        python tools_c/judge_event.py --baseline $R/gepl_s101 $R/gepl_s111 --exp $R/evr_s101 $R/evr_s111
        echo; echo "## secondary: ready_inbox vs ready (release)"
        python tools_c/judge_event.py --release --baseline $R/evr_s101 $R/evr_s111 --exp $R/evi_s101 $R/evi_s111
    } | tee $R/judgement_evt.md
}

stage_eval_rule() {  # evr + geometric close rule (results/event_clf/plan_XY.md); no retraining, evr only
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        OUT=./outputs/rollouts/evrz_s$SEED
        rm -rf $OUT ./videos/evt/evrz_s$SEED
        ./isaaclab.sh -p tools_c/play_record.py --task $TASK --checkpoint "$(latest_ckpt bc_rnn_low_dim_agibot_toy2box_event_ready_seed$SEED)" \
            --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT --close_rule 0.04,0.03 \
            --video --video_length 800 --video_folder ./videos/evt/evrz_s$SEED | tee $OUT.log
    done
    R=outputs/rollouts
    CUDA_VISIBLE_DEVICES= python tools_c/judge_rule.py --gepl $R/gepl_s101 $R/gepl_s111 --evr $R/evr_s101 $R/evr_s111 \
        --exp $R/evrz_s101 $R/evrz_s111 | tee $R/judgement_rule.md
}

stage_convert_ee() {
    CUDA_VISIBLE_DEVICES= python tools_c/convert_actions_ee.py --input $EVT --output $EVT_EE
}

stage_replay_ee() {  # gate: converted actions in the EE-action env must track the recorded states as well as the original
    mkdir -p ./outputs/logs_ee
    ./isaaclab.sh -p scripts/tools/replay_demos.py --task $TASK --dataset_file $EVT --select_episodes 0 1 2 3 4 \
        --validate_states --headless > ./outputs/logs_ee/replay_base.log 2>&1
    ./isaaclab.sh -p scripts/tools/replay_demos.py --task $TASK_EE --dataset_file $EVT_EE --select_episodes 0 1 2 3 4 \
        --validate_states --headless > ./outputs/logs_ee/replay_ee.log 2>&1
    python - <<'PY'
import re, sys
r = {}
for k in ("base", "ee"):
    t = open(f"outputs/logs_ee/replay_{k}.log").read()
    m, x = len(re.findall(r"- matched\.", t)), len(re.findall(r"- mismatched\.", t))
    r[k] = m / max(m + x, 1)
    print(f"replay {k}: matched steps {m} / {m + x} = {r[k]:.3f}")
ok = r["ee"] >= r["base"] - 0.02 and (r["base"] > 0 or r["ee"] > 0)
print("replay gate:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
PY
}

stage_train_ee() {
    for SEED in $SEEDS; do
        echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK_EE \
            --algo bc_event_ready --dataset $EVT_EE --seed $SEED --name bc_rnn_low_dim_agibot_toy2box_event_ready_ee
    done
}

latest_ckpt_ee() {
    ls -td ./logs/robomimic/$TASK_EE/$1/*/ | head -1 | sed "s|/$|/models/model_epoch_$EPOCH.pth|"
}

stage_calib_ee() {
    for SEED in $SEEDS; do
        CUDA_VISIBLE_DEVICES= python tools_c/calibrate_event_tau.py --ckpt "$(latest_ckpt_ee bc_rnn_low_dim_agibot_toy2box_event_ready_ee_seed$SEED)"
    done
}

stage_eval_ee() {  # same evaluation as eval_rule (close rule 0.04,0.03), EE-action env
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        OUT=./outputs/rollouts/eez_s$SEED
        rm -rf $OUT ./videos/ee/eez_s$SEED
        ./isaaclab.sh -p tools_c/play_record.py --task $TASK_EE --checkpoint "$(latest_ckpt_ee bc_rnn_low_dim_agibot_toy2box_event_ready_ee_seed$SEED)" \
            --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT --close_rule 0.04,0.03 \
            --video --video_length 800 --video_folder ./videos/ee/eez_s$SEED | tee $OUT.log
    done
    R=outputs/rollouts
    CUDA_VISIBLE_DEVICES= python tools_c/judge_ee.py --ref $R/evrz_s101 $R/evrz_s111 --exp $R/eez_s101 $R/eez_s111 | tee $R/judgement_ee.md
}

stage_convert_rel2() {  # relative-only observations (results/event_clf/plan_REL.md)
    python tools_c/add_gravity_ee.py --input $EVT_EE --output ./datasets/_tmp_ee_grav.hdf5
    python tools_c/add_grip_event_to_hdf5.py --input ./datasets/_tmp_ee_grav.hdf5 --output $EVT_REL \
        --keys gripper_pos grasp_err_ee place_err_ee gravity_ee
    rm ./datasets/_tmp_ee_grav.hdf5
    CUDA_VISIBLE_DEVICES= python tools_c/smoke_bc_event.py --config $AGENTS/bc_event_ready_rel.json --data $EVT_REL --out /tmp/bc_event_smoke_rel
}

latest_ckpt_rel() {
    ls -td ./logs/robomimic/$TASK_REL/$1/*/ | head -1 | sed "s|/$|/models/model_epoch_$EPOCH.pth|"
}

stage_train_rel2() {  # the two seeds train in parallel (about 4 GB GPU memory each)
    mkdir -p ./outputs/logs_rel
    for SEED in $SEEDS; do
        echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK_REL \
            --algo bc_event_ready_rel --dataset $EVT_REL --seed $SEED --name bc_rnn_low_dim_agibot_toy2box_event_ready_rel \
            > ./outputs/logs_rel/train_s$SEED.log 2>&1 &
    done
    wait
}

stage_calib_rel2() {
    for SEED in $SEEDS; do
        CUDA_VISIBLE_DEVICES= python tools_c/calibrate_event_tau.py --ckpt "$(latest_ckpt_rel bc_rnn_low_dim_agibot_toy2box_event_ready_rel_seed$SEED)"
    done
}

stage_eval_rel2() {  # same evaluation as eval_ee; the two seeds run in parallel (one process per seed keeps the seed-42 order)
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        OUT=./outputs/rollouts/eerel_s$SEED
        rm -rf $OUT ./videos/rel/eerel_s$SEED
        ./isaaclab.sh -p tools_c/play_record.py --task $TASK_REL --checkpoint "$(latest_ckpt_rel bc_rnn_low_dim_agibot_toy2box_event_ready_rel_seed$SEED)" \
            --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT --close_rule 0.04,0.03 \
            --video --video_length 800 --video_folder ./videos/rel/eerel_s$SEED > $OUT.log 2>&1 &
    done
    wait
    R=outputs/rollouts
    CUDA_VISIBLE_DEVICES= python tools_c/judge_rel.py --ref $R/eez_s101 $R/eez_s111 --exp $R/eerel_s101 $R/eerel_s111 | tee $R/judgement_rel.md
}

stage_eval_z2() {  # eez checkpoints, close rule depth tolerance 3 cm -> 2 cm (results/event_clf/plan_Z2.md); no retraining, seeds in parallel
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        OUT=./outputs/rollouts/eez2_s$SEED
        rm -rf $OUT ./videos/ee/eez2_s$SEED
        ./isaaclab.sh -p tools_c/play_record.py --task $TASK_EE --checkpoint "$(latest_ckpt_ee bc_rnn_low_dim_agibot_toy2box_event_ready_ee_seed$SEED)" \
            --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT --close_rule 0.04,0.02 \
            --video --video_length 800 --video_folder ./videos/ee/eez2_s$SEED > $OUT.log 2>&1 &
    done
    wait
    R=outputs/rollouts
    CUDA_VISIBLE_DEVICES= python tools_c/judge_z2.py --ref $R/evrz_s101 $R/evrz_s111 --ref_name evrz \
        --exp $R/eez2_s101 $R/eez2_s111 | tee $R/judgement_z2.md
}

stage_train_base() {  # baseline: algo bc (23D obs) on $REL; bc does not read the added obs keys, so the input equals $GE
    for SEED in $SEEDS; do
        echo n | ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task $TASK \
            --algo bc --dataset $REL --seed $SEED
    done
}

stage_eval_base() {  # same evaluation as the other models (seed 42, 100 rollouts, horizon 800)
    mkdir -p ./outputs/rollouts
    for SEED in $SEEDS; do
        OUT=./outputs/rollouts/base_s$SEED
        rm -rf $OUT
        ./isaaclab.sh -p tools_c/play_record.py --task $TASK --checkpoint "$(latest_ckpt bc_rnn_low_dim_agibot_toy2box_seed$SEED)" \
            --num_rollouts 100 --horizon 800 --seed 42 --headless --out_dir $OUT | tee $OUT.log
    done
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
    convert_evt) stage_convert_evt ;;
    smoke_evt) stage_smoke_evt ;;
    train_evt) stage_train_evt ;;
    calib_evt) stage_calib_evt ;;
    eval_evt) stage_eval_evt ;;
    evt) stage_train_evt; stage_calib_evt; stage_eval_evt ;;
    eval_rule) stage_eval_rule ;;
    convert_ee) stage_convert_ee ;;
    replay_ee) stage_replay_ee ;;
    train_ee) stage_train_ee ;;
    calib_ee) stage_calib_ee ;;
    eval_ee) stage_eval_ee ;;
    ee) stage_replay_ee; stage_train_ee; stage_calib_ee; stage_eval_ee ;;
    convert_rel2) stage_convert_rel2 ;;
    train_rel2) stage_train_rel2 ;;
    calib_rel2) stage_calib_rel2 ;;
    eval_rel2) stage_eval_rel2 ;;
    rel2) stage_train_rel2; stage_calib_rel2; stage_eval_rel2 ;;
    eval_z2) stage_eval_z2 ;;
    train_base) stage_train_base ;;
    eval_base) stage_eval_base ;;
    *) echo "unknown stage: $1"; exit 1 ;;
esac
