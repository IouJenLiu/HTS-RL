cd ..
LOG_PATH=/tmp/hts-rl_resutls

N_ENVS=16
N_STEPS=5000000

ENV=academy_corner
for i in 0;
do
  EXP_NAME=hts-rl_${ENV}_${N_ENVS}envs_${i}
  mkdir -p $LOG_PATH/$EXP_NAME
  CUDA_VISIBLE_DEVICES=0,1 python main.py --cuda-deterministic --eval-every-step 100000 --use-gae --num-processes ${N_ENVS} --use-linear-lr-decay  --num-actors 4  --env-name ${ENV} --num-agents 1 --num-left-agents 1 --num-right-agents 0 --base CNNBaseGfootball --num-env-steps ${N_STEPS} --seed ${i} --exp-name $EXP_NAME --clip-param 0.08 --gamma 0.993 --entropy-coef 0.003 --max-grad-norm 0.64 --lr 0.000343 --ppo-epoch 2 --num-mini-batch 8 --sync-every 128 --log-dir ${LOG_PATH} 2>&1 | tee $LOG_PATH/$EXP_NAME/stdout.txt
done