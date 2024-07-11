import multiprocessing
import mtkahypar

# Initialize thread pool
mtkahypar.initializeThreadPool(multiprocessing.cpu_count()) # use all available cores

# Setup partitioning context
context = mtkahypar.Context()
context.loadPreset(mtkahypar.PresetType.DEFAULT) # corresponds to Mt-KaHyPar-D
# In the following, we partition a hypergraph into two blocks
# with an allowed imbalance of 3% and optimize the connectivity metric
context.setPartitioningParameters(
  2,                       # number of blocks
  0.03,                    # imbalance parameter
  mtkahypar.Objective.KM1) # objective function
mtkahypar.setSeed(42)      # seed
context.logging = True     # enables partitioning output

# Load hypergraph from file
hypergraph = mtkahypar.Hypergraph(
  "../tests/test_instances/ibm01.hgr", # hypergraph file
  mtkahypar.FileFormat.HMETIS) # hypergraph is stored in hMetis file format

# Partition hypergraph
partitioned_hg = hypergraph.partition(context)

# Output metrics
print("Partition Stats:")
print("Imbalance = " + str(partitioned_hg.imbalance()))
print("cut/kml/soed:cut = " + str(partitioned_hg.cut()))
print("Block Weights:")
print("Weight of Block 0 = " + str(partitioned_hg.blockWeight(0)))
print("Weight of Block 1 = " + str(partitioned_hg.blockWeight(1)))