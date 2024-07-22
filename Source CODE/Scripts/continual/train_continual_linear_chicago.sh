for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "SynapticIntelligence" --dataset "chicago" \
        --data_dir "./Data/chicago.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/linear_target-12.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --seed $seed --training_regime 'continual'
done


for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "GEM" --dataset "chicago" \
        --data_dir "./Data/chicago.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/linear_target-12.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --replay_mem_size 500 --seed $seed --training_regime 'continual'
done


for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "MAS" --dataset "chicago" \
        --data_dir "./Data/chicago.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/linear_target-12.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --ewc_lambda 10.0 --seed $seed --training_regime 'continual'
done

