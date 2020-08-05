counter=0
while [ $counter -le 5 ]
do
   
    taskset -c $counter python ./sim.py $counter > $counter.log &
    echo $counter
    counter=$(( $counter + 1 ))

done