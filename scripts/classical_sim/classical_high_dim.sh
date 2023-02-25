
for dim in 2 3 4 5 6 7
do
  # for fun in rastrigin schwefel
  for fun in rastrigin schwefel
  do
    for method in cmaes quads grover
    do
      tsp python models/run_methods.py --init_normal_mean 0.2 --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type classical --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-classical --entity preview-control
    done
  done
done

for dim in 8 9 10 11 12
do
for fun in rastrigin schwefel
do
  for method in cmaes quads
  do
    tsp python models/run_methods.py --init_normal_mean 0.2 --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type classical --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-classical --entity preview-control
  done
done
done
   
$fun=styblinski_tang
for dim in 2 3 4 5 6 7
do
for method in cmaes quads grover
    do
      tsp python models/run_methods.py --init_normal_mean 0.8 --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type classical --eval_limit_per_update 10000000000 --group $fun --project_name mitou-quads-classical --entity preview-control
    done
done

for dim in 8 9 10 11 12
do
  for method in cmaes quads
  do
    tsp python models/run_methods.py --init_normal_mean 0.8 --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type classical --eval_limit_per_update 10000000000 --group $fun  --project_name mitou-quads-classical --entity preview-control
  done
done
