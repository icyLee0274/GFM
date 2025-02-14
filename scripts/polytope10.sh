EXAMPLE="polytope10"

DATA="polytope_d10_1"
#DATA="polytope_d10_2"
#DATA="polytope_d10_3"
#DATA="polytope_d10_4"
#DATA="polytope_d10_5"

GEN="10000"
BATCH="256"
HIDDEN="128"
REPEAT="10"
NORM="Linf"

echo "**************** Training Vanilla ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --train --batch_size $BATCH --hidden $HIDDEN --method vanilla --gen_sample

echo "**************** Testing Vanilla ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --test --n_gen $GEN --repeat $REPEAT --method vanilla

echo "**************** Testing Reflect ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --test --n_gen $GEN --repeat $REPEAT --method reflect

echo "**************** Testing Project ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --test --n_gen $GEN --repeat $REPEAT --method project


echo "**************** Training Gauge ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --norm $NORM --train --batch_size $BATCH --hidden $HIDDEN --method gauge_vanilla

echo "**************** Testing Gauge Vanilla ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --norm $NORM --test --n_gen $GEN --repeat $REPEAT --method gauge_vanilla

echo "**************** Testing Gauge Reflect ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --norm $NORM --test --n_gen $GEN --repeat $REPEAT --method gauge_reflect

echo "**************** Testing Gauge Project ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --norm $NORM --test --n_gen $GEN --repeat $REPEAT --method gauge_project


echo "**************** Training Mirror ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --norm L2 --train --batch_size $BATCH --hidden $HIDDEN --method gauge_mirror

echo "**************** Testing Mirror ****************"
python run_example.py $EXAMPLE --verbose --data "data/$DATA.npz" --norm L2 --test --n_gen $GEN --repeat $REPEAT --method gauge_mirror
