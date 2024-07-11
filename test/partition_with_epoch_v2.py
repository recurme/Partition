# 批量化的处理net文件，并进行partition

import os
import torch
import mtkahypar
import logging
import argparse
mydir = os.path.dirname(os.path.realpath(__file__))


mtkahypar.initializeThreadPool(16) # use all available cores
# Setup partitioning context
context = mtkahypar.Context()
# context.loadPreset(mtkahypar.PresetType.DETERMINISTIC) # corresponds to Mt-KaHyPar-Dset
context.loadPreset(mtkahypar.PresetType.DEFAULT) # corresponds to Mt-KaHyPar-D
number_of_blocks = 4
# 创建解析器
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--imb', type=float, required=True, help='The imbalance for partitioning the hypergraph')
parser.add_argument('--sel', type=int, required=True, help='choose dataset')
args = parser.parse_args()
imbalance = args.imb
sel = args.sel

context.setPartitioningParameters(
  number_of_blocks,                       # number of blocks
  imbalance,                    # imbalance parameter
  mtkahypar.Objective.KM1) # objective function
mtkahypar.setSeed(42)      # seed
context.logging = True

ispd_file_folder = ['ispd2016/','ispd2017/']
fpga_file_folder = ['FPGA01','FPGA02','FPGA03','FPGA04','FPGA05','FPGA06','FPGA07','FPGA08','FPGA09','FPGA10','FPGA11','FPGA12']
clk_fpga_file_folder = ['CLK-FPGA01','CLK-FPGA02','CLK-FPGA03','CLK-FPGA04','CLK-FPGA05','CLK-FPGA06','CLK-FPGA07','CLK-FPGA08','CLK-FPGA09','CLK-FPGA10','CLK-FPGA11','CLK-FPGA12','CLK-FPGA13']
def partition_with_epoch(ispd, fpga_file_folder):
    for fpga in fpga_file_folder:
        full_path = '/home/song/partition/test/net_inst_map/'+ispd+fpga+'/'
        inst_pin_map_b2as = torch.load(full_path + 'inst_pin_map_b2as.pt')
        net_pin_map_b_starts = torch.load(full_path + 'net_pin_map_b_starts.pt')
        fixed_inst_locs_xyz = torch.load(full_path + 'inst_locs_xyz.pt')
        with open(full_path + 'fixed_range.txt', 'r') as f:
            fixed_range = [int(line.rstrip('\n')) for line in f.readlines()]
        with open(full_path + 'diearea.txt', 'r') as f:
            diearea = [float(line.rstrip('\n')) for line in f.readlines()[-2:]]
        inst_max_id = fixed_range[1]
        net_max_id = net_pin_map_b_starts.shape[0] - 1
        net = []
        for i, net_cur in enumerate(net_pin_map_b_starts):
            if i == 0:
                net_last = 0
                continue     
            data_list = inst_pin_map_b2as[net_last:net_cur].tolist()
            net.append(data_list)
            net_last = net_cur

        # fixed constraints
        # block 1x4
        block_ids = [
            3 if loc[1] > diearea[1] / 4 * 3 else
            2 if loc[1] > diearea[1] / 4 * 2 else
            1 if loc[1] > diearea[1] / 4 else
            0
            for loc in fixed_inst_locs_xyz
        ]
        # 统计block_ids数组中0-3的个数
        block_ids_count = [block_ids.count(i) for i in range(4)]
        print(f"{fpga}-block_ids_count: {block_ids_count}")

        context.max_block_weights = [
            int(inst_max_id*(block_ids_count[i]/len(block_ids)))+1 if block_ids_count[i]>0 else 0 for i in range(4) 
        ]
        # block 2x2
        # block_ids = [
        #     1 if loc[1] > diearea[1] / 2 and loc[0] < diearea[0] / 2 else
        #     3 if loc[1] > diearea[1] / 2 else
        #     0 if loc[0] < diearea[0] / 2 else
        #     2
        #     for loc in fixed_inst_locs_xyz
        # ]
        fixed_vertices = [-1] * fixed_range[0]
        fixed_vertices.extend(block_ids) 

        # create hypergraph
        hypergraph = mtkahypar.Hypergraph(
            inst_max_id,               # with seven nodes
            net_max_id,               # and four hyperedges
            net,
            [1 for i in range(inst_max_id)], # node weights
            [1 for j in range(net_max_id)])       # hyperedge weights
        hypergraph.addFixedVertices(
                fixed_vertices,number_of_blocks
        )

        # Partition hypergraph
        partitioned_hg = hypergraph.partition(context)
        block_files_path = '/home/song/partition/test/result_fixed_aware_block/' + ispd + 'imb{imbalance}/'.format(imbalance=imbalance) + fpga  + '/'
        os.makedirs(block_files_path, exist_ok=True)
        block_files = [open(block_files_path +f"block_{i}.txt", "w") for i in range(number_of_blocks)]
        block_fixed_files = [open(block_files_path +f"block_fixed_{i}.txt", "w") for i in range(number_of_blocks)]
        
        hypergraph.doForAllNodes(lambda node : (
        block_files[partitioned_hg.blockID(node)].write(str(node) + "\n") if node not in range(fixed_range[0],fixed_range[1]) else block_fixed_files[partitioned_hg.blockID(node)].write(str(node) + "\n"),
        ))
        for file in block_files:
            file.close()

        # Create a logger
        logger = logging.getLogger(block_files_path)
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(block_files_path + 'log.txt',"w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Log the metrics
        logger.info("Input Stats:")
        logger.info("fixed_range:" + str(fixed_range))
        logger.info("diearea:    " + str(diearea))
        logger.info("Hypergraph Stats:")
        logger.info("Number of Nodes      = " + str(hypergraph.numNodes()))
        logger.info("Number of Hyperedges = " + str(hypergraph.numEdges()))
        logger.info("Number of Pins       = " + str(hypergraph.numPins()))
        logger.info("Weight of Hypergraph = " + str(hypergraph.totalWeight()))
        logger.info("\n")
        logger.info("Partition Stats:")
        logger.info("Imbalance = " + str(partitioned_hg.imbalance()))
        logger.info("km1       = " + str(partitioned_hg.km1()))
        logger.info("cut       = " + str(partitioned_hg.cut()))
        logger.info("Block Weights:")
        logger.info("Weight of Block 0 = " + str(partitioned_hg.blockWeight(0)))
        logger.info("Weight of Block 1 = " + str(partitioned_hg.blockWeight(1)))
        logger.info("Weight of Block 2 = " + str(partitioned_hg.blockWeight(2)))
        logger.info("Weight of Block 3 = " + str(partitioned_hg.blockWeight(3)))

        # Remove the handler at the end of the loop
        logger.removeHandler(file_handler)
if sel == 1:
    partition_with_epoch(ispd_file_folder[1], clk_fpga_file_folder)
elif sel == 0:
    partition_with_epoch(ispd_file_folder[0], fpga_file_folder)
    