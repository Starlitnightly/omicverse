# 
# Global configuration
# 

class MOFAConfig:
	"""
	MOFA config manager
	"""
	def __init__(self, use_float32: bool = False):
		self.use_float32 = use_float32

	@property
	def use_float32(self):
		return self.float32

	@use_float32.setter
	def use_float32(self, float32):
		if type(float32) != bool:
			raise TypeError(f"True or False has to be provided.")
		self.float32 = float32

config = MOFAConfig()