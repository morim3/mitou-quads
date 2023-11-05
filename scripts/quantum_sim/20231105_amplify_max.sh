

for dim in 3
do
  # for fun in rastrigin schwefel styblinski_tang squared ackley griewank rosenbrock xin_she_yang
  for fun in rastrigin schwefel styblinski_tang squared ackley griewank rosenbrock alpine01 alpine02 mishra deflectedCorrugatedSpring wavy 
    do
      for method in cmaes quads grover
      do
        tsp python models/run_methods.py --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type quantum --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-quantum3 --entity preview-control --n_jobs 10
      done
  done
done
