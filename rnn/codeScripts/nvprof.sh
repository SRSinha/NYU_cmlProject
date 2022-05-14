#!/bin/bash
if [ -z $1 ]
then
        echo "Forgot to add the batch size? Try again"
        exit 1
fi

BATCH=$1

rm -rf ../../../$BATCH
mkdir ../../../$BATCH
# CMD="/opt/conda/bin/python3 main.py --epochs 1 --batch-size $BATCH"
CMD="python3 rnnMain.py --epochs 1 --batch-size $BATCH"
# METRICS="dram__bytes_write.sum,dram__bytes_read.sum,smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,smsp__sass_thread_inst_executed_op_ffma_pred_on.sum"
NVPROF_METRICS="warp_execution_efficiency,dram_read_transactions,dram_write_transactions,dram_read_throughput,dram_write_throughput,flop_count_dp,flop_count_sp,flop_count_hp,dram_utilization,sysmem_utilization,tex_cache_hit_rate"

#nvprof
NVPROF_METRICS_LOG="../../../$BATCH/nvprof-gpusum-$BATCH.log"
NVPROF_OPMETRICS_LOG="../../../$BATCH/nvprof-gpuop-$BATCH.log"
rm -rf NVPROF_METRICS_LOG

#nvprof
echo '********** NVPROF PROFILING BEGINS **********'
sudo /usr/local/cuda/bin/nvprof --print-gpu-summary --log-file $NVPROF_METRICS_LOG $CMD

DTOH=`cat $NVPROF_METRICS_LOG | grep -e "CUDA memcpy DtoH"  | sed -e "s/,/ /g" | awk '{print($2, $3, $4, $5, $6, $7, $8)}'`
HTOD=`cat $NVPROF_METRICS_LOG | grep -e "CUDA memcpy HtoD"  | sed -e "s/,/ /g" | awk '{print($2, $3, $4, $5, $6, $7, $8)}'`
CMEM=`cat $NVPROF_METRICS_LOG | grep -e "CUDA memset"  | sed -e "s/,/ /g" | awk '{print($2, $3, $4, $5, $6, $7, $8)}'`
echo "DTOH" >> $NVPROF_OPMETRICS_LOG
echo $DTOH >> $NVPROF_OPMETRICS_LOG
echo "HTOD" >> $NVPROF_OPMETRICS_LOG
echo $HTOD >> $NVPROF_OPMETRICS_LOG
echo "Cuda memset" >> $NVPROF_OPMETRICS_LOG
echo $CMEM >> $NVPROF_OPMETRICS_LOG

echo $'     **** NVPROF CUSTOM METRICS PROFILING BEGINS ****'
sudo /usr/local/cuda/bin/nvprof --metrics $NVPROF_METRICS --log-file $NVPROF_METRICS_LOG $CMD

TK=`cat $NVPROF_METRICS_LOG | grep -e "warp_execution_efficiency" | sed -e "s/,/ /g" | awk '{print NR}' | tail -1`
echo "total kernels $TK"
T1=`cat $NVPROF_METRICS_LOG | grep -e "warp_execution_efficiency"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# echo "T1 $T1"
# T11=$((T1/TK))
T11=`echo $T1/$TK | bc`
# echo "T11 $T11"
T2=`cat $NVPROF_METRICS_LOG | grep -e "dram_read_transactions"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
T3=`cat $NVPROF_METRICS_LOG | grep -e "dram_write_transactions"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
T4=`cat $NVPROF_METRICS_LOG | grep -e "dram_read_throughput"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
T41=`echo $T4/$TK | bc`
T5=`cat $NVPROF_METRICS_LOG | grep -e "dram_write_throughput"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
T51=`echo $T5/$TK | bc`
T6=`cat $NVPROF_METRICS_LOG | grep -e "flop_count_dp" -e "flop_count_sp" -e "flop_count_hp"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# T6=`cat $NVPROF_METRICS_LOG | grep -e "flop_count_sp"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# T7=`cat $NVPROF_METRICS_LOG | grep -e "flop_count_hp"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
T7=`cat $NVPROF_METRICS_LOG | grep -e "dram_utilization"  | sed -e "s/,/ /g" | awk '{print($8)}'`
T8=`cat $NVPROF_METRICS_LOG | grep -e "sysmem_utilization"  | sed -e "s/,/ /g" | awk '{print($8)}'`
T9=`cat $NVPROF_METRICS_LOG | grep -e "tex_cache_hit_rate"  | sed -e "s/,/ /g" | awk '{print($8)+0}'| paste -sd+ | bc`
T91=`echo $T9/$TK | bc`

echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "warp_execution_efficiency" >> $NVPROF_OPMETRICS_LOG
# echo $T1 >> $NVPROF_OPMETRICS_LOG
echo $T11 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_read_transactions" >> $NVPROF_OPMETRICS_LOG
echo $T2 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_write_transactions" >> $NVPROF_OPMETRICS_LOG
echo $T3 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_read_throughput" >> $NVPROF_OPMETRICS_LOG
# echo $T4 >> $NVPROF_OPMETRICS_LOG
echo $T41 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_write_throughput" >> $NVPROF_OPMETRICS_LOG
# echo $T5 >> $NVPROF_OPMETRICS_LOG
echo $T51 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "FLOPS" >> $NVPROF_OPMETRICS_LOG
echo $T6 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_utilization" >> $NVPROF_OPMETRICS_LOG
echo $T7 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "sysmem_utilization" >> $NVPROF_OPMETRICS_LOG
echo $T8 >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "tex_cache_hit_rate" >> $NVPROF_OPMETRICS_LOG
# echo $T9 >> $NVPROF_OPMETRICS_LOG
echo $T91 >> $NVPROF_OPMETRICS_LOG

echo $'********** NVPROF PROFILING ENDS **********'


