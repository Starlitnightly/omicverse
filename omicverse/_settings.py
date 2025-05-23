

class omicverseConfig:

    def __init__(self,mode='cpu'):
        self.mode = mode

    def gpu_init(self,managed_memory=True,pool_allocator=True,devices=0):
        
        import scanpy as sc
        import cupy as cp

        import time
        import rapids_singlecell as rsc

        import warnings
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
        rmm.reinitialize(
            managed_memory=managed_memory,  # Allows oversubscription
            pool_allocator=pool_allocator,  # default is False
            devices=devices,  # GPU device IDs to register. By default registers only GPU 0.
        )
        cp.cuda.set_allocator(rmm_cupy_allocator)
        print('GPU mode activated')
        self.mode = 'gpu'
    
    def cpu_init(self):
        print('CPU mode activated')
        self.mode = 'cpu'
    
    def cpu_gpu_mixed_init(self):
        print('CPU-GPU mixed mode activated')
        self.mode = 'cpu-gpu-mixed'
        

settings = omicverseConfig()
        