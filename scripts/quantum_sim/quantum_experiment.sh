

for dim in 1 2 3
do
  # for fun in rastrigin schwefel styblinski_tang squared ackley griewank rosenbrock xin_she_yang
  for fun in rastrigin schwefel styblinski_tang squared ackley griewank 
    do
      for method in cmaes quads grover adam
      do
        tsp python models/run_methods.py --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type quantum --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-quantum2 --entity preview-control --n_jobs 2
      done
  done
done

for fun in rastrigin schwefel styblinski_tang squared ackley griewank 
do
  for method in cmaes quads grover adam
  do
    tsp python models/run_methods.py --name ${method}_${fun}_4 --method $method --func $fun --n_dim 4 --sampler_type quantum --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-quantum2 --entity preview-control --n_digits 7 --n_jobs 2
  done
done
