#!/bin/bash -exu

SEED_START=$((100 + "$AWS_BATCH_JOB_ARRAY_INDEX" * "$RANGE"))
SEED_END=$((100 + "$AWS_BATCH_JOB_ARRAY_INDEX" * "$RANGE" + "$RANGE"))

mkdir out
crystal build --release -D local main.cr
for (( i = $SEED_START; i < $SEED_END; i++ )); do
	seed=$(printf "%04d" $i)
	echo -n "seed:${seed} " >> log.txt
	./main $seed >> log.txt  2> /dev/null
done

aws s3 cp log.txt s3://marathon-tester/$RESULT_PATH/$(printf "%03d" $SEED_START).txt
