

for fun in rastrigin schwefel styblinski_tang easom squared ackley griewank
  do
    for method in cmaes quads grover
    do
      tsp python models/run_methods.py --name ${method}_${fun}_3 --method $method --func $fun --n_dim 3 --sampler_type quantum --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-quantum --entity preview-control
    done
done

