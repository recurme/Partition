import torch
import mtkahypar
import os
mydir = os.path.dirname(os.path.realpath(__file__))
import logging
# Initialize thread pool
# mtkahypar.initializeThreadPool(multiprocessing.cpu_count()) # use all available cores
mtkahypar.initializeThreadPool(16) # use all available cores

# Setup partitioning context
context = mtkahypar.Context()
context.loadPreset(mtkahypar.PresetType.DETERMINISTIC) # corresponds to Mt-KaHyPar-D
# In the following, we partition a hypergraph into two blocks
# with an allowed imbalance of 3% and optimize the connectivity metric
number_of_blocks = 4
context.setPartitioningParameters(
  number_of_blocks,                       # number of blocks
  0.15,                    # imbalance parameter
  mtkahypar.Objective.KM1) # objective function
mtkahypar.setSeed(42)      # seed

context.logging = True


inst_pin_map_b2as = torch.load('/home/song/partition/test/net_inst_map/inst_pin_map_b2as.pt')
net_pin_map_b_starts = torch.load('/home/song/partition/test/net_inst_map/net_pin_map_b_starts.pt')
# net_pin_map_bs = torch.load('/home/song/partition/test/net_inst_map/net_pin_map_bs.pt')
inst_locs_xyz = torch.load('/home/song/partition/test/net_inst_map/inst_locs_xyz.pt')
inst_max_id = torch.max(inst_pin_map_b2as).item() + 1
net_max_id = net_pin_map_b_starts.shape[0] - 1
# fixed range (105117, 105273)
fixed_range = (105117, 105273)
diearea = (168,480)
# context.max_block_weights = [
#     int(105273/3),
#     int(105273/3),
#     int(105273/5)*3,
#     int(105273/3)
# ]
net = []
with open('/home/song/partition/test/net_inst_map/net_inst.hgr', 'w') as f:
    for i, net_cur in enumerate(net_pin_map_b_starts):
        if i == 0:
            f.write(str(net_max_id)+' '+str(inst_max_id)+'\n')
            net_last = 0
            continue     
        # print(inst_pin_map_b2as[net_last:net_cur])
        # 将张量转换为列表
        if(inst_pin_map_b2as[net_last:net_cur].shape[0] > 50):
            data_list = inst_pin_map_b2as[net_last:net_cur].tolist()
        else:
            data_list = inst_pin_map_b2as[net_last:net_cur].tolist()
        net.append(data_list)
        # 输出到文件，每个元素后都有一个空格
        f.write(' '.join(map(str, data_list)))
        # 每次循环换行
        f.write('\n')
        net_last = net_cur
# print(cnt)
# set_fixedinst2block
block_ids = [
    1 if loc[1] > diearea[1] / 2 and loc[0] < diearea[0] / 2 else
    3 if loc[1] > diearea[1] / 2 else
    0 if loc[0] < diearea[0] / 2 else
    2
    for loc in inst_locs_xyz
]

fixed_vertices = [-1] * fixed_range[0]
fixed_vertices.extend(block_ids) 

# Creates a weighted hypergraph
hypergraph = mtkahypar.Hypergraph(
  inst_max_id,               # with seven nodes
  net_max_id,               # and four hyperedges
  net,
  [1 for i in range(inst_max_id)], # node weights
  [1 for j in range(net_max_id)])       # hyperedge weights
hypergraph.addFixedVertices(
    fixed_vertices,number_of_blocks
)
# Output statistics of hypergraph
print("Hypergraph Stats:")
print("Number of Nodes      = " + str(hypergraph.numNodes()))
print("Number of Hyperedges = " + str(hypergraph.numEdges()))
print("Number of Pins       = " + str(hypergraph.numPins()))
print("Weight of Hypergraph = " + str(hypergraph.totalWeight()))
print()


# Partition hypergraph
partitioned_hg = hypergraph.partition(context)

# Output metrics
print("Partition Stats:")
print("Imbalance = " + str(partitioned_hg.imbalance()))
print("km1       = " + str(partitioned_hg.km1()))
print("cut       = " + str(partitioned_hg.cut()))
print("Block Weights:")
print("Weight of Block 0 = " + str(partitioned_hg.blockWeight(0)))
print("Weight of Block 1 = " + str(partitioned_hg.blockWeight(1)))
print("Weight of Block 2 = " + str(partitioned_hg.blockWeight(2)))
print("Weight of Block 3 = " + str(partitioned_hg.blockWeight(3)))

block_files = [open(f"/home/song/partition/test/result/block_{i}.txt", "w") for i in range(number_of_blocks)]
block_fixed_files = [open(f"/home/song/partition/test/result/block_fixed_{i}.txt", "w") for i in range(number_of_blocks)]
hypergraph.doForAllNodes(lambda node : (
  block_files[partitioned_hg.blockID(node)].write(str(node) + "\n") if node not in range(fixed_range[0],fixed_range[1]) else block_fixed_files[partitioned_hg.blockID(node)].write(str(node) + "\n"),
))
for file in block_files:
    file.close()
block_files_path = '/home/song/partition/test/result/'
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
