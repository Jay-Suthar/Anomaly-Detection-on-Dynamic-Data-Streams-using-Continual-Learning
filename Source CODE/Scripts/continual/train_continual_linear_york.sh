for seed in {0..4}
do
  echo "Running for seed $seed"
  python main.py --strategy "SynapticIntelligence" --dataset "york" \
        --data_dir "./Data/york.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/linear_target-10.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --seed $seed --training_regime 'continual'
done


for seed in {0..4}
do
  echo "Running for seed $seed"
  python main.py --strategy "GEM" --dataset "york" \
        --data_dir "./Data/york.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/linear_target-10.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --replay_mem_size 500 --seed $seed --training_regime 'continual'
done


for seed in {0..4}
do
  echo "Running for seed $seed"
  python main.py --strategy "MAS" --dataset "york" \
        --data_dir "./Data/york.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/linear_target-10.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --ewc_lambda 10.0 --seed $seed --training_regime 'continual'
done
