
GPUS=${1}
work_dirs=work_dirs
for name in ${*:2} ; 
do
echo $name
echo "******************************************"
# CUDA_VISIBLE_DEVICES=$GPUS PORT=$PORT python ./tools/print_config.py configs/$name.py
CUDA_VISIBLE_DEVICES=$GPUS python tools/test.py $work_dirs/$name/$name.py $work_dirs/$name/latest.pth  --eval bbox --out work_dirs/$name/results.pkl
sleep 5
python tools/analysis_tools/eval_metric.py $work_dirs/$name/$name.py $work_dirs/$name/results.pkl --format-only --eval-options "jsonfile_prefix=$work_dirs/${name}/results"
sleep 5
# python tools/analysis_tools/analyze_results.py work_dirs/$name/$name.py work_dirs/$name/results.pkl work_dirs/$name/show_result --topk 100 --show-score-thr 0.3
# sleep 5
python tools/analysis_tools/coco_error_analysis.py $work_dirs/${name}/results.bbox.json $work_dirs/${name}/error_analysis --ann datasets/TSSD_aug/annotations/test.json
sleep 5
done

