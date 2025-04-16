load_model=''
proj_dir='out/L12-D768-v7/'
data_file='/public/home/ssjxzkz/Projects/rhineai/data/target/prot/vocab_65536/datasets'

n_layer=12
n_embd=768

micro_bsz=28
epoch_save=1
epoch_steps=1000 #6171
ctx_len=4096

source /public/home/ssjxzkz/Projects/rwkv-cc/.venv/bin/activate
cd /public/home/ssjxzkz/Projects/rwkv-cc
export PYTHONPATH="/public/home/ssjxzkz/Projects/rwkv-cc:$PYTHONPATH"
export WANDB_MODE=offline

python /public/home/ssjxzkz/Projects/rwkv-cc/train.py \
--proj_dir $proj_dir --data_file $data_file --wandb "rhineai" \
--vocab_size 65536 \
--n_layer $n_layer --n_embd $n_embd \
--ctx_len $ctx_len --micro_bsz $micro_bsz --magic_prime 6921791\
--epoch_steps $epoch_steps --epoch_count 9999999 --epoch_begin 0 --epoch_save $epoch_save \
--lr_init 3e-4 --lr_final 1e-4 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-18 \
--accelerator gpu --devices 4 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
--my_testing "x070" \
--data_type binidx
# sft --sft_field query response --sft_split "train[:1000]"