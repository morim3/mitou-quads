
for dim in 2 3 4 
do
  # for fun in rastrigin schwefel
  for fun in styblinski_tang
  do
    for method in cmaes quads grover
    do
      tsp python models/run_methods.py --init_normal_mean 0.8 --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type classical --eval_limit_one_sample 100000000 --group $fun
    done
  done

done

for dim in 5 6 7
do
for fun in styblinski_tang
do
  for method in cmaes quads
  do
    tsp python models/run_methods.py --init_normal_mean 0.8 --name ${method}_${fun}_${dim} --method $method --func $fun --n_dim $dim --sampler_type classical --eval_limit_one_sample 100000000 --group $fun
  done
done
done
   
