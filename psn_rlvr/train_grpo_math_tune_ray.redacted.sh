#!/usr/bin/env bash
set -euo pipefail

pip install math_verify tensordict

# mkdir -p "$HDFS_CHECKPOINT_PATH" "$HDFS_LOG_PATH"
export NCCL_DEBUG=DEBUG
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_IB_DISABLE=1

# Runtime env
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export WANDB_MODE=offline
export WANDB_ANONYMOUS=must

export PROJECT_NAME=verl_train_adaptive
export WANDB_OFFICIAL=1
export VLLM_ATTENTION_BACKEND=XFORMERS

# Redacted personal/local paths. Replace these with your own environment paths.
export HDFS_DATA_PATH=/path/to/simpleRL-reason/data
export HDFS_MODEL_PATH=/path/to/ckpts/llm
export HDFS_CHECKPOINT_PATH=/path/to/verl/checkpoints
export HDFS_LOG_PATH=/path/to/verl/logs
### NEW: 备份原始参数（因为后面 parsing 会 shift 掉 $@）
ORIG_ARGS=("$@")

today=$(date +%m%d)   # 如果你想更清晰可改成：date +%Y%m%d
export ARNOLD_WORKER_NUM=1

# Default values
TRAIN_BATCH_SIZE=256
VAL_BATCH_SIZE=16
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=3072
LEARNING_RATE=3e-7
PPO_MINI_BATCH_SIZE=36
PPO_MICRO_BATCH_SIZE=36
CLIP_RATIO=0.2
KL_LOSS_COEF=0.001
ENTROPY_COEFFIENT=0.001
KL_LOSS_TYPE="low_var_kl"
TEMPERATURE=1.0
LOG_PROB_MICRO_BATCH_SIZE=36
ROLLOUT_N=8
KL_COEF=0.001
TOTAL_EPOCHS=20
DATASET_NAME=simplelr_math_35
ROLLOUT_GPU_MEMORY_UTIL=0.6
MODEL_NAME=Qwen2.5-Math-7B
SAVE_FREQ=20
TEST_FREQ=5
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
MAX_NUM_BATCHED_TOKENS=16384

REMOVE_PREVIOUS_CKPT=False
PARAM_NOISE_BASE_SEED=null
NOISE_SIGMA=0.001
NOISE_TARGET_MODULES="['.mlp.']"
USE_PARAM_NOISE=True

TIS_IMP_RATIO_CAP=10.0

# Adaptive noise defaults
ADAPTIVE_NOISE=False
ADAPTIVE_NOISE_TARGET_KL=0.003
ADAPTIVE_NOISE_COEFF=1.01
ADAPTIVE_NOISE_MIN_SIGMA=0.000001
ADAPTIVE_NOISE_MAX_SIGMA=0.5
ADAPTIVE_NOISE_INITIAL_SIGMA=0.001

NOISE_SIGMA_EXPLICIT=False

generate_suffix() {
  local suffix=""
  local dataset_provided=false
  local model_provided=false
  local suffix_provided=false

  while [[ "$#" -gt 0 ]]; do
    case $1 in
      --train_batch_size) suffix+="_batch$2"; shift 2 ;;
      --val_batch_size) suffix+="_valbatch$2"; shift 2 ;;
      --max_prompt_length) suffix+="_max_prompt$2"; shift 2 ;;
      --max_response_length) suffix+="_max_response$2"; shift 2 ;;
      --learning_rate) suffix+="_lr$2"; shift 2 ;;
      --ppo_mini_batch_size) suffix+="_ppomini$2"; shift 2 ;;
      --ppo_micro_batch_size) shift 2 ;;
      --kl_loss_coef) suffix+="_klcoef$2"; shift 2 ;;
      --entropy_coeffient) suffix+="_entcoef$2"; shift 2 ;;
      --clip_ratio) suffix+="_clipratio$2"; shift 2 ;;
      --kl_loss_type) suffix+="_kltype$2"; shift 2 ;;
      --temperature) suffix+="_temp$2"; shift 2 ;;
      --log_prob_micro_batch_size) suffix+="_logprobbatch$2"; shift 2 ;;
      --rollout_n) suffix+="_rollout$2"; shift 2 ;;
      --kl_coef) suffix+="_klcontrol$2"; shift 2 ;;
      --total_epochs) suffix+="_epochs$2"; shift 2 ;;
      --rollout_gpu_memory_util) shift 2 ;;
      --dataset_name) suffix+="_$2"; dataset_provided=true; shift 2 ;;
      --model_name) suffix+="_$2"; model_provided=true; shift 2 ;;
      --noise_sigma) suffix+="_sigma$2"; shift 2 ;;
      --param_noise_base_seed) suffix+="_seed$2"; shift 2 ;;
      --tis_imp_ratio_cap) suffix+="_tiscap$2"; shift 2 ;;
      --adaptive_noise) suffix+="_adapt$2"; shift 2 ;;
      --noise_target_kl) suffix+="_tgkl$2"; shift 2 ;;
      --suffix) input_suffix="$2"; suffix_provided=true; shift 2 ;;
      *) shift ;;
    esac
  done

  if [ "$dataset_provided" = false ]; then
    suffix+="_$DATASET_NAME"
  fi

  if [ "$model_provided" = false ]; then
    suffix+="_$MODEL_NAME"
  fi

  if [ "$suffix_provided" = true ]; then
    suffix+="_$input_suffix"
  fi

  echo "$suffix"
}

echo "Arguments received: ${ORIG_ARGS[*]}"

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
  echo "Processing: $1"
  case "$1" in
    --train_batch_size) TRAIN_BATCH_SIZE="$2"; shift 2 ;;
    --val_batch_size) VAL_BATCH_SIZE="$2"; shift 2 ;;
    --max_prompt_length) MAX_PROMPT_LENGTH="$2"; shift 2 ;;
    --max_response_length) MAX_RESPONSE_LENGTH="$2"; shift 2 ;;
    --learning_rate) LEARNING_RATE="$2"; shift 2 ;;
    --ppo_mini_batch_size) PPO_MINI_BATCH_SIZE="$2"; shift 2 ;;
    --ppo_micro_batch_size) PPO_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --kl_loss_coef) KL_LOSS_COEF="$2"; shift 2 ;;
    --entropy_coeffient) ENTROPY_COEFFIENT="$2"; shift 2 ;;
    --clip_ratio) CLIP_RATIO="$2"; shift 2 ;;
    --kl_loss_type) KL_LOSS_TYPE="$2"; shift 2 ;;
    --temperature) TEMPERATURE="$2"; shift 2 ;;
    --log_prob_micro_batch_size) LOG_PROB_MICRO_BATCH_SIZE="$2"; shift 2 ;;
    --rollout_n) ROLLOUT_N="$2"; shift 2 ;;
    --rollout_gpu_memory_util) ROLLOUT_GPU_MEMORY_UTIL="$2"; shift 2 ;;
    --rollout_tp) ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="$2"; shift 2 ;;
    --max_num_batched_tokens) MAX_NUM_BATCHED_TOKENS="$2"; shift 2 ;;
    --kl_coef) KL_COEF="$2"; shift 2 ;;
    --total_epochs) TOTAL_EPOCHS="$2"; shift 2 ;;
    --dataset_name) DATASET_NAME="$2"; shift 2 ;;
    --model_name) MODEL_NAME="$2"; shift 2 ;;
    --save_freq) SAVE_FREQ="$2"; shift 2 ;;
    --test_freq) TEST_FREQ="$2"; shift 2 ;;
    --remove_previous_ckpt) REMOVE_PREVIOUS_CKPT="$2"; shift 2 ;;
    --noise_sigma) NOISE_SIGMA="$2"; NOISE_SIGMA_EXPLICIT=True; shift 2 ;;
    --param_noise_base_seed) PARAM_NOISE_BASE_SEED="$2"; shift 2 ;;
    --tis_imp_ratio_cap) TIS_IMP_RATIO_CAP="$2"; shift 2 ;;
    --adaptive_noise) ADAPTIVE_NOISE="$2"; shift 2 ;;
    --noise_target_kl) ADAPTIVE_NOISE_TARGET_KL="$2"; shift 2 ;;
    --adaptive_noise_coeff) ADAPTIVE_NOISE_COEFF="$2"; shift 2 ;;
    --adaptive_noise_initial_sigma) ADAPTIVE_NOISE_INITIAL_SIGMA="$2"; shift 2 ;;
    --adaptive_noise_min_sigma) ADAPTIVE_NOISE_MIN_SIGMA="$2"; shift 2 ;;
    --adaptive_noise_max_sigma) ADAPTIVE_NOISE_MAX_SIGMA="$2"; shift 2 ;;
    --suffix) SUFFIX="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 如果开了 adaptive 且用户没显式给 noise_sigma，则用 initial_sigma
if [ "$ADAPTIVE_NOISE" = "True" ] && [ "$NOISE_SIGMA_EXPLICIT" = "False" ]; then
  NOISE_SIGMA="$ADAPTIVE_NOISE_INITIAL_SIGMA"
fi

### NEW: 用一个 sanitize 函数把 RUN_NAME 里非法字符（/[]' 空格等）替换成 _
sanitize() {
  echo "$1" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

### NEW: 组装 RUN_NAME（包含 date + model + noise + adaptive）
MODEL_TAG=$(sanitize "$MODEL_NAME")
NOISE_MOD_TAG=$(sanitize "$NOISE_TARGET_MODULES")

RUN_NAME="d${today}"\
"_m${MODEL_TAG}"\
"_psn${USE_PARAM_NOISE}"\
"_sigma${NOISE_SIGMA}"\
"_mods${NOISE_MOD_TAG}"\
"_seed${PARAM_NOISE_BASE_SEED}"\
"_adapt${ADAPTIVE_NOISE}"\
"_tkl${ADAPTIVE_NOISE_TARGET_KL}"\
"_acoef${ADAPTIVE_NOISE_COEFF}"\
"_ainit${ADAPTIVE_NOISE_INITIAL_SIGMA}"\
"_amin${ADAPTIVE_NOISE_MIN_SIGMA}"\
"_amax${ADAPTIVE_NOISE_MAX_SIGMA}"\
"_tiscap${TIS_IMP_RATIO_CAP}"

### CHANGED: suffix 用 ORIG_ARGS（否则 $@ 已经空了）
SUFFIX=$(generate_suffix "${ORIG_ARGS[@]}")

# 如果你还想保留用户 --suffix 的自由文本，就让 generate_suffix 帮你拼进 SUFFIX（已支持）
RUN_NAME="${RUN_NAME}"

LOG_FILE_PATH="$HDFS_LOG_PATH/$RUN_NAME.log"

echo "Training with the following parameters:"
echo "RUN_NAME: $RUN_NAME"
echo "Train Batch Size: $TRAIN_BATCH_SIZE"
echo "Adaptive Noise: $ADAPTIVE_NOISE"
echo "Noise Target KL: $ADAPTIVE_NOISE_TARGET_KL"
echo "Noise Initial Sigma: $ADAPTIVE_NOISE_INITIAL_SIGMA"
echo "TIS Imp Ratio Cap: $TIS_IMP_RATIO_CAP"
echo "LOG FILE PATH: $LOG_FILE_PATH"

ROLLOUT_LOGPROBS=False
if [ "$ADAPTIVE_NOISE" = "True" ] || [ "$TIS_IMP_RATIO_CAP" != "0" ]; then
  ROLLOUT_LOGPROBS=True
fi

ROLLOUT_IS_LEVEL=null
ROLLOUT_IS_THRESHOLD=2.0
if [ "$TIS_IMP_RATIO_CAP" != "0" ]; then
  ROLLOUT_IS_LEVEL=token
  ROLLOUT_IS_THRESHOLD="$TIS_IMP_RATIO_CAP"
fi
# ... 后面 python3 -m verl.trainer.main_ppo 不变 ...
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HDFS_DATA_PATH/$DATASET_NAME/train.parquet \
  data.val_files=$HDFS_DATA_PATH/$DATASET_NAME/test.parquet \
  data.train_batch_size=$TRAIN_BATCH_SIZE \
  data.val_batch_size=$VAL_BATCH_SIZE \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  actor_rollout_ref.model.path=$HDFS_MODEL_PATH/$MODEL_NAME \
  actor_rollout_ref.actor.optim.lr=$LEARNING_RATE \
  actor_rollout_ref.actor.ppo_mini_batch_size=12 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=$KL_LOSS_COEF \
  actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEFFIENT \
  actor_rollout_ref.actor.clip_ratio=$CLIP_RATIO \
  actor_rollout_ref.actor.kl_loss_type=$KL_LOSS_TYPE \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  +actor_rollout_ref.actor.fsdp_config.grad_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.actor.adaptive_noise=$ADAPTIVE_NOISE \
  actor_rollout_ref.actor.adaptive_noise_target_kl=$ADAPTIVE_NOISE_TARGET_KL \
  actor_rollout_ref.actor.adaptive_noise_coeff=$ADAPTIVE_NOISE_COEFF \
  actor_rollout_ref.actor.adaptive_noise_min_sigma=$ADAPTIVE_NOISE_MIN_SIGMA \
  actor_rollout_ref.actor.adaptive_noise_max_sigma=$ADAPTIVE_NOISE_MAX_SIGMA \
  actor_rollout_ref.actor.adaptive_noise_initial_sigma=$ADAPTIVE_NOISE_INITIAL_SIGMA \
  actor_rollout_ref.rollout.calculate_log_probs=$ROLLOUT_LOGPROBS \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTIL \
  actor_rollout_ref.rollout.n=$ROLLOUT_N \
  actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
  actor_rollout_ref.rollout.use_param_noise=$USE_PARAM_NOISE \
  actor_rollout_ref.rollout.noise_sigma=$NOISE_SIGMA \
  actor_rollout_ref.rollout.noise_target_modules="$NOISE_TARGET_MODULES" \
  actor_rollout_ref.rollout.param_noise_base_seed=$PARAM_NOISE_BASE_SEED \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=12 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.rollout_correction.rollout_is=$ROLLOUT_IS_LEVEL \
  algorithm.rollout_correction.rollout_is_threshold=$ROLLOUT_IS_THRESHOLD \
  trainer.critic_warmup=0 \
  actor_rollout_ref.rollout.free_cache_engine=True \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$PROJECT_NAME \
  trainer.experiment_name=$RUN_NAME \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=$ARNOLD_WORKER_NUM \
  trainer.val_before_train=False \
  trainer.save_freq=50 \
  trainer.test_freq=50 \
  trainer.default_local_dir=$HDFS_CHECKPOINT_PATH/$RUN_NAME \
  trainer.total_epochs=$TOTAL_EPOCHS \
  2>&1 | tee -a "$LOG_FILE_PATH"
    # trainer.val_before_train=False \
