#!/bin/bash
if [ -z $1 ]
then
        echo "Forgot to add the batch size? Try again"
        exit 1
fi

BATCH=$1

# nsys file
NSYS_RAW_LOG="$BATCH/nsys-runtime-$BATCH.qdrep"
NSYS_METRICS_LOG="$BATCH/nsys-metric-$BATCH.log"
rm -rf NSYS_RAW_LOG NSYS_METRICS_LOG

CMD="/opt/conda/bin/python3 main.py --epochs 1 --batch-size $BATCH"
# METRICS="dram__bytes_write.sum,dram__bytes_read.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
METRICS="dram__bytes_write.sum,dram__bytes_read.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,smsp__sass_thread_inst_executed_op_hfma_pred_on.sum,smsp__sass_thread_inst_executed_op_hmul_pred_on.sum"

#NSYS profiling
echo "********** NSYS PROFILING STARTS **********"
sudo /usr/local/cuda/bin/nsys profile -f true -o $NSYS_RAW_LOG $CMD
# rm -rf $BATCH/tmp_nsys_gputrace.csv
sudo /usr/local/cuda/bin/nsys stats --report gputrace $NSYS_RAW_LOG -o $BATCH/tmp_nsys
# compute runtime
echo "Runtime(ns)" >> $NSYS_METRICS_LOG
tail -n +2 $BATCH/tmp_nsys_gputrace.csv | sed -e "s/,/ /g" | awk '{print $2}' | paste -sd+ | bc >> $NSYS_METRICS_LOG
# echo "Total Throughput(Mb/s)" >> $NSYS_METRICS_LOG
# tail -n +2 $BATCH/tmp_nsys_gputrace.csv | sed -e "s/,/ /g" | awk '{print $2}' | paste -sd+ | bc >> $NSYS_METRICS_LOG
# echo "Total lines" >> $NSYS_METRICS_LOG
# tail -n +2 $BATCH/tmp_nsys_gputrace.csv | sed -e "s/,/ /g" | awk '{print NR}' | tail -1 >> $NSYS_METRICS_LOG
echo "********** NSYS PROFILING ENDED **********"