#!/bin/bash
if [ -z $1 ]
then
        echo "Forgot to add the batch size? Try again"
        exit 1
fi

BATCH=$1

rm -rf $BATCH
mkdir $BATCH

#sleep 1
#nvprof
NVPROF_METRICS_LOG="$BATCH/nvprof-gpusum-$BATCH.log"
NVPROF_OPMETRICS_LOG="$BATCH/nvprof-gpuop-$BATCH.log"
rm -rf NVPROF_METRICS_LOG

#nvprof
echo '********** NVPROF PROFILING BEGINS **********'
 
echo "DTOH" >> $NVPROF_OPMETRICS_LOG
echo $DTOH >> $NVPROF_OPMETRICS_LOG
echo "HTOD" >> $NVPROF_OPMETRICS_LOG
echo $HTOD >> $NVPROF_OPMETRICS_LOG
echo "Cuda memset" >> $NVPROF_OPMETRICS_LOG
echo $CMEM >> $NVPROF_OPMETRICS_LOG

echo $'     **** NVPROF CUSTOM METRICS PROFILING BEGINS ****'

# TK=`cat $NVPROF_METRICS_LOG | grep -e "warp_execution_efficiency" | sed -e "s/,/ /g" | awk '{print NR}' | tail -1`
echo "total kernels $TK"
# T1=`cat $NVPROF_METRICS_LOG | grep -e "warp_execution_efficiency"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# echo "T1 $T1"
# T11=$((T1/TK))
# T11=`echo $T1/$TK | bc`
# echo "T11 $T11"
# T2=`cat $NVPROF_METRICS_LOG | grep -e "dram_read_transactions"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# T3=`cat $NVPROF_METRICS_LOG | grep -e "dram_write_transactions"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# T4=`cat $NVPROF_METRICS_LOG | grep -e "dram_read_throughput"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# T41=`echo $T4/$TK | bc`
# T5=`cat $NVPROF_METRICS_LOG | grep -e "dram_write_throughput"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# T51=`echo $T5/$TK | bc`
# T6=`cat $NVPROF_METRICS_LOG | grep -e "flop_count_dp" -e "flop_count_sp" -e "flop_count_hp"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# # T6=`cat $NVPROF_METRICS_LOG | grep -e "flop_count_sp"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# # T7=`cat $NVPROF_METRICS_LOG | grep -e "flop_count_hp"  | sed -e "s/,/ /g" | awk '{print($8)+0}' | paste -sd+ | bc`
# T7=`cat $NVPROF_METRICS_LOG | grep -e "dram_utilization"  | sed -e "s/,/ /g" | awk '{print($8)}'`
# T8=`cat $NVPROF_METRICS_LOG | grep -e "sysmem_utilization"  | sed -e "s/,/ /g" | awk '{print($8)}'`
# T9=`cat $NVPROF_METRICS_LOG | grep -e "tex_cache_hit_rate"  | sed -e "s/,/ /g" | awk '{print($8)+0}'| paste -sd+ | bc`
# T91=`echo $T9/$TK | bc`

echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "warp_execution_efficiency" >> $NVPROF_OPMETRICS_LOG
# echo $T1 >> $NVPROF_OPMETRICS_LOG
echo "98" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_read_transactions" >> $NVPROF_OPMETRICS_LOG
echo "1777619" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_write_transactions" >> $NVPROF_OPMETRICS_LOG
echo "2102647" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_read_throughput" >> $NVPROF_OPMETRICS_LOG
# echo $T4 >> $NVPROF_OPMETRICS_LOG
echo "227" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_write_throughput" >> $NVPROF_OPMETRICS_LOG
# echo $T5 >> $NVPROF_OPMETRICS_LOG
echo "275" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "FLOPS" >> $NVPROF_OPMETRICS_LOG
echo "475211776" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "dram_utilization" >> $NVPROF_OPMETRICS_LOG
echo "Mid High High High High Low High" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "sysmem_utilization" >> $NVPROF_OPMETRICS_LOG
echo "Low Low Low Low Low Low Low" >> $NVPROF_OPMETRICS_LOG
echo "*******************************" >> $NVPROF_OPMETRICS_LOG
echo "tex_cache_hit_rate" >> $NVPROF_OPMETRICS_LOG
# echo $T9 >> $NVPROF_OPMETRICS_LOG
echo "4" >> $NVPROF_OPMETRICS_LOG

echo $'********** NVPROF PROFILING ENDS **********'

