import os
# Must be set BEFORE import torch
#os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0"
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import numpy as np
import matplotlib.pyplot as plt


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
		void histogram_cuda(torch::Tensor count, torch::Tensor x, torch::Tensor steps);
		void histogram_2d_cuda(torch::Tensor count, torch::Tensor a, torch::Tensor b, torch::Tensor steps_a, torch::Tensor steps_b);
		torch::Tensor quantiles_cuda(torch::Tensor count, torch::Tensor steps);
		void quantiles_bounds_cuda(torch::Tensor guess_steps, torch::Tensor quantiles_lo_x, torch::Tensor quantiles_lo_y, torch::Tensor quantiles_hi_x, torch::Tensor quantiles_hi_y, torch::Tensor count, torch::Tensor steps);
		void cdf_cuda(torch::Tensor cdf, torch::Tensor count);
		void pit_cuda(torch::Tensor pit, torch::Tensor X, torch::Tensor cdf, torch::Tensor steps);
		void copula_legendre_cuda(torch::Tensor copula, torch::Tensor obs, torch::Tensor pred);
		void plot_copula_legendre_cuda(torch::Tensor plot,torch::Tensor copula);
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
		functions=['histogram_cuda', 'quantiles_cuda', 'quantiles_bounds_cuda', 'cdf_cuda', 'pit_cuda', 'copula_legendre_cuda', 'plot_copula_legendre_cuda'],
		with_cuda=True,
		extra_cuda_cflags=hgm__cuda_flags
	)


class Histogrammer():
	
	@torch.no_grad()
	def __init__(self, device, nFilters, nBins=10000):
	
		self.epoch = 0
		self.device = device
		self.nFilters = nFilters
		self.nBins = nBins
		self.curr_size = 0
	
		# Make sure the histogram module is compiled
		if hgm__module is None:
			Compile()
			
		# Allocate the bounds and megabatch
		#self.megabatch = torch.zeros((nFilters,megabatch_size), dtype=torch.float32, device=device, requires_grad=False)

		# Allocate the global histogram
		self.histogram = torch.zeros((nFilters,nBins), dtype=torch.int64, device=device, requires_grad=False)
		
		# Allocate the steps and quantiles
		self.steps     = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device, requires_grad=False)
		self.quantiles_lo_x = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device, requires_grad=False)
		self.quantiles_lo_y = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device, requires_grad=False)
		self.quantiles_hi_x = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device, requires_grad=False)
		self.quantiles_hi_y = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device, requires_grad=False)
		self.quantiles      = torch.zeros((nFilters,nBins+1), dtype=torch.float32, device=device, requires_grad=False)

		# Allocate the min and max bounds
		self.minval = None  #torch.zeros((nFilters,), dtype=torch.float32, device=device)
		self.maxval = None  #torch.zeros((nFilters,), dtype=torch.float32, device=device)
		
		# No cdf just yet
		self.cdf = None

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

			x_contig = x.contiguous()
			count = torch.zeros((self.nFilters, self.nBins), dtype=torch.int32, requires_grad=False, device=self.device)

			torch.cuda.synchronize()
			hgm__module.histogram_cuda(count, x_contig, self.steps)
			torch.cuda.synchronize()
			
			self.histogram += count

	@torch.no_grad()
	def end_epoch(self):
		
		#self.flush()
		
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
			
	@torch.no_grad()
	def calc_cdf(self):

		print('Beg cdf', flush=True)

		self.cdf = torch.zeros((self.nFilters, (self.nBins+1)), dtype=torch.float32, requires_grad=False, device=self.device)

		torch.cuda.synchronize()
		hgm__module.cdf_cuda(self.cdf, self.histogram);
		torch.cuda.synchronize()

		print('End cdf', flush=True)


	@torch.no_grad()
	def pit(self, X, stretch=True):

		if (len(X.shape)!=2):
			print('ERROR, histogrammer.pit expect 2D shape [N F]')
			sys.exit(1)

		# Have we run the cdf first?
		if self.cdf == None:
			self.calc_cdf()

		# Calculate dimensions
		pit = torch.zeros(X.shape, dtype=torch.float32, requires_grad=False, device=self.device)

		torch.cuda.synchronize()
		hgm__module.pit_cuda(pit, X, self.cdf, self.steps);
		torch.cuda.synchronize()

		if stretch:
			pit = 2.0*pit - 1.0

		return pit

	@torch.no_grad()
	def copula_legendre(self, obs, pred, M=11, run_pit=True):
		
		if (len(obs.shape)!=2 or len(pred.shape)!=2):
			print('ERROR, histogrammer.copula_legendre expect 2D shape [N F]')
			sys.exit(1)
	
		if run_pit:
			obs  = self.pit(obs)
			pred = self.pit(pred)
	
		# Allocate copula
		F = obs.shape[1]
		copula = torch.zeros((M,M,F), dtype=torch.float32, requires_grad=False, device=self.device)
		
		# Run the legendre copula
		torch.cuda.synchronize()
		hgm__module.copula_legendre_cuda(copula, obs, pred)
		torch.cuda.synchronize()

		return copula

	@torch.no_grad()
	def plot_copula_legendre(self, rows,cols,copula):

		# Allocate plot
		F = copula.shape[2]
		plot = torch.zeros((rows,cols,F), dtype=torch.float32, requires_grad=False, device=self.device)
	
		# Plot the copula
		torch.cuda.synchronize()
		hgm__module.plot_copula_legendre_cuda(plot, copula)
		torch.cuda.synchronize()
		
		return plot

	def plt_plot_copula(self, copula_plot_np, f):
		fontsize = 12
		plot_rows = copula_plot_np.shape[0]
		plot_cols = copula_plot_np.shape[1]
		bin_a_mesh, bin_b_mesh = np.meshgrid(np.arange(plot_cols), np.arange(plot_rows))
		plt.pcolormesh(bin_b_mesh, bin_a_mesh, copula_plot_np[:,:,f], shading='auto', cmap='viridis', vmin = 0 , vmax = 1.0)
		nbins = plot_rows-1
		bin_pos   = [0.0*nbins, 0.25*nbins, 0.5*nbins, 0.75*nbins, 1.0*nbins]
		bin_label = [-1.0, -0.5, 0.0, 0.5, 1.0] 
		plt.xticks(bin_pos, bin_label, fontsize = fontsize)
		plt.yticks(bin_pos, bin_label, fontsize = fontsize)
		cbar = plt.colorbar(label='Count')
		cbar.set_label('Count', fontsize = fontsize)
		cbar.ax.tick_params(labelsize=fontsize)		




def test():

	print('---------------------------------------------')
	print(' Test histogrammer')
	print('---------------------------------------------')

	device='cuda'
	nFilters = 10
	nBins = 10
	batch_size = 1000
	n_batch = 100
	#megabatch_size = 32768  #1024

	hg = Histogrammer(device, nFilters, nBins)#, megabatch_size)

	hg_epochs = 2

	print('-----')
	print(' Generate a dataset N(0,1)')
	print('-----')
	batches = []
	with torch.no_grad():
		for b in range(n_batch):
			batches.append(  torch.randn((batch_size,nFilters), dtype=torch.float32, device=device)  )
			#batches[b] = F.relu(batches[b])
			#batches[b].fill_(0.0);


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

	print('---------')
	print(' Run Copula')
	print('---------')
	with torch.no_grad():

		print('Calculate copula')
		nMoments = 11
		copula = torch.zeros((nMoments, nMoments, nFilters), dtype=torch.float32, device=device, requires_grad=False)
		copula_count = 0

		for b in range(n_batch//2):
			print('  >batch', b, flush=True)
			obs  = batches[2*b]
			#pred = batches[2*b+1]
			pred = 0.7*batches[2*b] + 0.3*batches[2*b+1]
						
			copula_batch = hg.copula_legendre(obs, pred, nMoments)
			copula += copula_batch
			copula_count += batch_size
		
		print('  divide by count')
		copula /= copula_count    # rescale by num points
		
		print('Plot copula')
		plot_rows = 512
		plot_cols = 512
		copula_plot = hg.plot_copula_legendre(plot_rows,plot_cols,copula)
		
		copula_plot_np = copula_plot.detach().cpu().numpy()
		
		print('Show Plots')
		for f in range(nFilters):
			plt.figure(figsize=(10, 6))
			hg.plt_plot_copula(copula_plot_np, f)
			plt.show()
		

if __name__ == "__main__":
	test()

	










#--------------------------------------------
# DEPRECATED
#--------------------------------------------

	
	#@torch.no_grad()	
	#def flush(self):
		
		#print('      BEGIN flush')
		
		# If the megabatch is empty there is nothing to flush
		#if self.curr_size==0:
		#	return
		
		
		# Flush by running the CUDA kernel
		#x = self.megabatch[:,0:self.curr_size].contiguous()
		
	#	torch.cuda.synchronize()
	#	hgm__module.histogram_cuda(self.histogram, x, steps)
	#	torch.cuda.synchronize()
		
		

	#	# Else we flush by running the CUDA kernel
	#	for f in range(self.nFilters):
			
	#		#print('        flush f', f,  'nFilters', self.nFilters)
			
	#		# Input tensors x, steps
	#		#print('        Input tensors x, steps')
	#		#count = torch.zeros((self.nBins,), dtype=torch.int32, device=self.device)
	#		x     = self.megabatch[f,0:self.curr_size].contiguous()
	#		steps = self.steps[f].contiguous()
						
	#		# Run the CUDA kernel for the module
	#		#print('        Run the CUDA kernel for the module')
	#		torch.cuda.synchronize()
	#		count = hgm__module.histogram_cuda(x, steps)
	#		#print('        torch.cuda.synchronize()')
	#		torch.cuda.synchronize()
		
	#		#print('         count', count)
		
	#		# Update the histogram
	#		#print('        Update the histogram')
	#		self.histogram[f] = self.histogram[f] + count

		
	#	# We are flushed
	#	self.curr_size = 0
		
	#	#print('      END flush')
	

			# if we do not have enough space,
			#  then flush the megabatch
			#if self.curr_size+batch_size > self.megabatch_size:
			#	self.flush()
			
			# append the transposed batch to the megabatch
			#sidx = self.curr_size
			#eidx = sidx + batch_size
			#self.curr_size = eidx
			
			#print('megabatch', self.megabatch.shape)
			#print('sidx', sidx, 'eidx', eidx)
			#print('x', x.shape)
			#self.megabatch[:,sidx:eidx] = x.transpose(0,1).contiguous()
			
		#print('    END add_batch')

