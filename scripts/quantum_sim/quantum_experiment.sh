

for fun in rastrigin schwefel easom squared ackley griewank
  do
    for method in cmaes quads grover
    do
      tsp python models/run_methods.py --init_normal_mean 0.2 --name ${method}_${fun}_4 --method $method --func $fun --n_dim 4 --sampler_type quantum --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-quantum --entity preview-control
    done
done

for fun in styblinski_tang
  do
    for method in cmaes quads grover
    do
      tsp python models/run_methods.py --init_normal_mean 0.8 --name ${method}_${fun}_4 --method $method --func $fun --n_dim 4 --sampler_type quantum --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-quantum --entity preview-control
    done
done

