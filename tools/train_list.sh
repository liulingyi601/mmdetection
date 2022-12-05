
GPUS=${1}
PORT=${2}
configs=config_TSSD
for name in ${*:3} ; 
do
echo $name
echo "******************************************"
# CUDA_VISIBLE_DEVICES=$GPUS PORT=$PORT python ./tools/print_config.py configs/$name.py
# CUDA_VISIBLE_DEVICES=$GPUS PORT=$PORT ./tools/dist_train.sh configs_sar/$name.py 2 
CUDA_VISIBLE_DEVICES=$GPUS PORT=$PORT ./tools/dist_train.sh $configs/$name.py 2 

sleep 20
done
