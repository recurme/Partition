imbs=(0.03 0.05 0.07 0.1 0.12 0.15 0.18 0.2)
# 运行python脚本：python partition_with_epoch.py --imb number
for imb in ${imbs[@]}
do
    python partition_for_die1x4_middle_block2.py --imb $imb --sel 1
done
