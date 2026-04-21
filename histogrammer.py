import os
# Must be set BEFORE import torch
#os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline


hgm__cuda_source = None
hgm__cpp_source = None
hgm__cuda_flags = None
hgm__module = None

#----------------------------------------
# Compile the histogram cuda kernel
#----------------------------------------
def Compile():
	global hgm__cuda_source
	global hgm__cpp_source
	global hgm__cuda_flags
	global hgm__module

	with open('histogrammer.cu', 'r') as fin:
		hgm__cuda_source = fin.read()

	print(hgm__cuda_source)

	hgm__cpp_source = '''
		torch::Tensor histogram_cuda(torch::Tensor x, torch::Tensor steps);
		torch::Tensor quantiles_cuda(torch::Tensor count, torch::Tensor steps);
		void quantiles_bounds_cuda(torch::Tensor guess_steps, torch::Tensor quantiles_lo_x, torch::Tensor quantiles_lo_y, torch::Tensor quantiles_hi_x, torch::Tensor quantiles_hi_y, torch::Tensor count, torch::Tensor steps);
	'''

	def get_cuda_arch_flags():
		"""Detects local GPU and returns the 'sm_XX' flag."""
		if not torch.cuda.is_available():
			return []
		
		major, minor = torch.cuda.get_device_capability()
		arch_version = f"{major}{minor}"
		
		# Optional: Safety cap for Blackwell (compute 12+) if using older compilers
		# Many compilers in early 2026 still prefer sm_90 for stability
		if major >= 12:
			print(f"Targeting Blackwell ({arch_version}) with sm_90 compatibility mode.")
			return [f"-arch=sm_90"]
			
		return [f"-arch=sm_{arch_version}"]

	# Get the flags dynamically
	hgm__cuda_flags = get_cuda_arch_flags()

	# Compiles on the fly!
	hgm__module = load_inline(
		name='inline_extension',
		cpp_sources=[hgm__cpp_source],
		cuda_sources=[hgm__cuda_source],
		functions=['histogram_cuda', 'quantiles_cuda', 'quantiles_bounds_cuda'],
		with_cuda=True,
		extra_cuda_cflags=hgm__cuda_flags
	)


class Histogrammer():
	
	@torch.no_grad()
	def __init__(self, device, nFilters, nBins=10000, megabatch_size=32768):
	
		self.epoch = 0
		self.device = device
		self.nFilters = nFilters
		self.nBins = nBins
		self.megabatch_size = megabatch_size
		self.curr_size = 0
	
		# Make sure the histogram module is compiled
		if hgm__module is None:
			Compile()
			
		# Allocate the bounds and megabatch
		self.megabatch = torch.zeros((nFilters,megabatch_size), dtype=torch.float32, device=device)

		# Allocate the global histogram
		self.histogram = torch.zeros((nFilters,nBins), dtype=torch.int64, device=device)
		
		# Allocate the steps and quantiles
		self.steps     = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device)
		self.quantiles_lo_x = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device)
		self.quantiles_lo_y = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device)
		self.quantiles_hi_x = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device)
		self.quantiles_hi_y = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device)
		self.quantiles      = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device)


		# Allocate the min and max bounds
		self.minval = None  #torch.zeros((nFilters,), dtype=torch.float32, device=device)
		self.maxval = None  #torch.zeros((nFilters,), dtype=torch.float32, device=device)

	@torch.no_grad()
	def begin_epoch(self):
		
		# Reset the minmax
		if self.epoch==0:
			self.minval = None
			self.maxval = None
		
		# Reset the histogram
		else:
			self.histogram.fill_(0)
			self.curr_size = 0
		
			self.steps.copy_(self.quantiles)
			
	#
	# x [batch nFilters]
	#
	@torch.no_grad()
	def add_batch(self, x):
		
		#print('    BEGIN add_batch')
		batch_size,nFilters = x.shape
		
		# in epoch 0, we just compute min/max values
		if self.epoch==0:
			curr_minval = torch.min(x, dim=0)[0]
			curr_maxval = torch.max(x, dim=0)[0]

			if self.minval is None:
				self.minval = curr_minval
				self.maxval = curr_maxval
			else:
				self.minval = torch.minimum(self.minval, curr_minval)
				self.maxval = torch.maximum(self.maxval, curr_maxval)
		
		# All other epochs we build the histogram
		else:

			# if we do not have enough space,
			#  then flush the megabatch
			if self.curr_size+batch_size > self.megabatch_size:
				self.flush()
			
			# append the transposed batch to the megabatch
			sidx = self.curr_size
			eidx = sidx + batch_size
			self.curr_size = eidx
			
			#print('megabatch', self.megabatch.shape)
			#print('sidx', sidx, 'eidx', eidx)
			#print('x', x.shape)
			self.megabatch[:,sidx:eidx] = x.transpose(0,1).contiguous()
			
		#print('    END add_batch')
	
	@torch.no_grad()	
	def flush(self):
		
		#print('      BEGIN flush')
		
		# If the megabatch is empty there is nothing to flush
		if self.curr_size==0:
			return
		
		# Else we flush by running the CUDA kernel
		for f in range(self.nFilters):
			
			#print('        flush f', f,  'nFilters', self.nFilters)
			
			# Input tensors x, steps
			#print('        Input tensors x, steps')
			#count = torch.zeros((self.nBins,), dtype=torch.int32, device=self.device)
			x     = self.megabatch[f,0:self.curr_size].contiguous()
			steps = self.steps[f].contiguous()
						
			# Run the CUDA kernel for the module
			#print('        Run the CUDA kernel for the module')
			torch.cuda.synchronize()
			count = hgm__module.histogram_cuda(x, steps)
			#print('        torch.cuda.synchronize()')
			torch.cuda.synchronize()
		
			#print('         count', count)
		
			# Update the histogram
			#print('        Update the histogram')
			self.histogram[f] = self.histogram[f] + count
			
		# We are flushed
		self.curr_size = 0
		
		#print('      END flush')
	
	@torch.no_grad()
	def end_epoch(self):
		
		self.flush()
		
		if self.epoch==0:

			# Make sure we have found the min and max values
			if self.minval is None:
				raise Exception('ERROR: end_epoch have not completed epoch 0 yet')
			
			# Initialize the quantiles to a uniform range between the values
			for f in range(self.nFilters):
				self.quantiles[f,:] = torch.linspace(self.minval[f], self.maxval[f], self.nBins+1)
			self.quantiles_lo_x.copy_( torch.reshape(self.minval, (self.nFilters,1)) )
			self.quantiles_lo_y.fill_( 0.0 )
			self.quantiles_hi_x.copy_( torch.reshape(self.maxval, (self.nFilters,1)) )
			self.quantiles_hi_y.fill_( 1.0 )
			
			# 
			self.epoch+=1
			
		else:
			
			#print('Begin run quantiles')
			
			# Compute the quantiles from the histogram
			torch.cuda.synchronize()
			hgm__module.quantiles_bounds_cuda(
				self.quantiles,
				self.quantiles_lo_x,
				self.quantiles_lo_y,
				self.quantiles_hi_x,
				self.quantiles_hi_y,
				self.histogram,
				self.steps)
			torch.cuda.synchronize()

			#print('End run quantiles')
			

print('---------------------------------------------')
print('---------------------------------------------')
print('Construct histogrammer')
print('---------------------------------------------')
print('---------------------------------------------')

def test():

	device='cuda'
	nFilters = 1
	nBins = 10
	batch_size = 100
	n_batch = 1000
	megabatch_size = 32768  #1024

	hg = Histogrammer(device, nFilters, nBins, megabatch_size)

	hg_epochs = 15

	print('-----')
	print(' Generate a dataset N(0,1)')
	print('-----')
	batches = []
	with torch.no_grad():
		for b in range(n_batch):
			batches.append(  torch.randn((batch_size,nFilters), dtype=torch.float32, device=device)  )
			#batches[b] = F.relu(batches[b])
			batches[b].fill_(0.0);


	print('---------')
	print(' Fit Histogrammer')
	print('---------')
	with torch.no_grad():
		for hg_epoch in range(hg_epochs):

			print('epoch', hg_epoch)
			#input('enter');
			hg.begin_epoch()

			for b in range(n_batch):

				hg.add_batch(batches[b])
			hg.end_epoch()

			# print the histogram
			for b in range(hg.nBins):
				print('bin (%.3f %.3f)   count %d ' % (hg.steps[0,b], hg.steps[0,b+1], hg.histogram[0,b]))

			# print the histogram
			for b in range(hg.nBins+1):
				print('lo (%.3f %.3f)   hi (%.3f %.3f)   guess  (%.3f %.3f)' % (
					hg.quantiles_lo_x[0,b], hg.quantiles_lo_y[0,b],
					hg.quantiles_hi_x[0,b], hg.quantiles_hi_y[0,b],
					hg.quantiles[0,b],   (float)( b / hg.nBins )) )
			input('quantiles   enter')

	print('Done!')
	
	
