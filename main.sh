export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16
export OPENBLAS_NUM_THREADS=16
export NUMEXPR_NUM_THREADS=16

gpu=0

for L in 4 3 2
do
  for pretrain_lr in 1e-4 5e-5 2e-5 1e-5
    do
  name="mvbnfm_dp0.5_attn1_L"${L}"_lr"${pretrain_lr}".pth"
  python main_pretrain.py --lr ${pretrain_lr} --dropout 0.5 --self_att_layers 1 --gnn_layers ${L} --gpu ${gpu} --save_name ${name} --batch_size 64
  for data in huashan_schaefer100 adni_schaefer100 abide_schaefer100 HCPGender_schaefer100 HCPAge_schaefer100 adhd_schaefer100
  do
      for lr in 1e-3 7e-4 5e-4 2e-4 1e-4 5e-5
      do
        path="./exp_results/fmri/graph_mae_pretrain/gcl/"${name}
        echo ${path}
        python main_finetune.py \
          --data_name ${data} \
          --batch_size 128 \
          --epochs 50 \
          --lr ${lr} \
          --warmup_ratio 0.1 \
          --min_lr_ratio 0.1 \
          --hidden_dim 256 --nhead 8 --self_att_layers 1 --gnn_layers ${L} \
          --ff_hidden_size 256 \
          --dropout 0.0 --gpu ${gpu} \
          --pretrained ${path} \
          --moe_experts 1 \
          --fold 10 \
          --csv_path ./exp_results/fmri/graph_mae_pretrain/gcl/cv_summary_schaefer100.csv
        done
    done
  done
done