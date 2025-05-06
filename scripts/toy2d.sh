EXAMPLE="toy2d"


IP="0.02,0.98"
#IP="load"

GEN="10000"
BATCH="256"
HIDDEN="64"
REPEAT="1"
NORM="Linf"

#echo "**************** Training Vanilla ****************"
#python run_example.py $EXAMPLE --verbose --train --norm $NORM --batch_size $BATCH --hidden $HIDDEN --method vanilla --gen_sample
#python run_example.py $EXAMPLE --verbose --train --norm $NORM --batch_size $BATCH --hidden $HIDDEN --interior_point $IP --method vanilla --gen_sample

#echo "**************** Testing Vanilla ****************"
#python run_example.py $EXAMPLE --verbose --test --norm $NORM --n_gen $GEN --interior_point $IP --repeat $REPEAT --method vanilla

#echo "**************** Testing Reflect ****************"
#python run_example.py $EXAMPLE --verbose --test --n_gen $GEN --repeat $REPEAT --method reflect
#
#echo "**************** Testing Project ****************"
#python run_example.py $EXAMPLE --verbose --test --n_gen $GEN --repeat $REPEAT --method project


#echo "**************** Training Gauge ****************"
#python run_example.py $EXAMPLE --verbose --norm $NORM --train --batch_size $BATCH --hidden $HIDDEN --interior_point $IP --method gauge_vanilla.yaml
#
#echo "**************** Testing Gauge Vanilla ****************"
#python run_example.py $EXAMPLE --verbose --norm $NORM --test --interior_point $IP --n_gen $GEN --repeat $REPEAT --method gauge_vanilla.yaml
#
#echo "**************** Testing Gauge Reflect ****************"
#python run_example.py $EXAMPLE --verbose --norm $NORM --test --interior_point $IP --n_gen $GEN --repeat $REPEAT --method gauge_reflect
#
#echo "**************** Testing Gauge Project ****************"
#python run_example.py $EXAMPLE --verbose --norm $NORM --test --interior_point $IP --n_gen $GEN --repeat $REPEAT --method gauge_project.yaml


echo "**************** Training Mirror ****************"
python run_example.py $EXAMPLE --verbose --norm $NORM --train --batch_size $BATCH --hidden $HIDDEN --interior_point $IP --method gauge_mirror

echo "**************** Testing Mirror ****************"
python run_example.py $EXAMPLE --verbose --norm $NORM --test --interior_point $IP --n_gen $GEN --repeat $REPEAT --method gauge_mirror
