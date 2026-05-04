#include <iostream>
#include <cstdio>
#include <torch/extension.h>

//----------------------
//  Histogram Kernel
//  in/out:
//   count    [F B]     (int32)
//  input:
//     x      [N F]     (float32)
//   steps    [F B+1]   (float32)
//-----------------------
__global__ void histogram_kernel(
	int* __restrict__    out_count,
	const float* __restrict__ x,
	const float* __restrict__ in_steps,
	int F, int N, int B) 
{
	//printf("BEGIN histogram_kernel\n");

	// Who am I ?
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	int f = blockIdx.y;


	//
	// Allocate the shared count and steps
	//
	extern __shared__ float s_data[];
	int   *count = (int*)s_data;          // [B]
	float *steps = (float*)(s_data + B);  // [B+1]

	//
	// Initialize the shared memory
	//
	int count_start = (B * threadIdx.x) / blockDim.x;
	int count_end   = (B * (threadIdx.x+1)) / blockDim.x; 
	int steps_start = ((B+1) * threadIdx.x) / blockDim.x;
	int steps_end   = ((B+1) * (threadIdx.x+1)) / blockDim.x; 
	//printf("rank  %d  count %d : %d  steps %d : %d  N %d B %d\n", 
	//	rank, count_start, count_end, steps_start, steps_end, N, B);
	for (int i=count_start; i<count_end; i++)
		count[i] = 0;
	for (int i=steps_start; i<steps_end; i++)
		steps[i] = in_steps[f*(B+1) + i];
	__syncthreads();

	//
	// If we are even in the problem
	//

	if (n<N)
	{
		
		//
		// Find the correct histogram bin (binary search)
		//
		float val = x[n*F + f];
		//printf("**  rank %d val %.3f\n", rank, val);
		if (val>=steps[0] && val<=steps[B])
		{			
			int bin_lo  = 0;
			int bin_hi  = B;
			while (bin_hi-bin_lo > 1)
			{
				int bin_mid = (bin_lo+bin_hi) / 2;
				float val_mid = steps[bin_mid];
				
				if (val<val_mid) {
					bin_hi = bin_mid;
				} else {
					bin_lo = bin_mid;
				}
			}
			
			//
			// Add to the bin
			//
			atomicAdd(&count[bin_lo], 1);
		}
	}

	//
	// Global barrier
	//
	__syncthreads();
		
	if (n<N)
	{
		//
		// Add to the overall output
		//
		for (int i=count_start; i<count_end; i++) {
			atomicAdd(&out_count[f*B + i], count[i]);
		}
	}
}


//----------------------
//  Histogram Cuda
//  in/out:
//   count    [F B]     (int32)
//  input:
//     x      [N F]     (float32)
//   steps    [F B+1]   (float32)
//-----------------------
void histogram_cuda(
	torch::Tensor count,
	torch::Tensor x,
	torch::Tensor steps)
{
	printf("BEGIN histogram_cuda\n");

	// Pointer to the data
	int* count_data      = count.data_ptr<int>();
	float*   x_data      = x.data_ptr<float>();
	float*   steps_data  = steps.data_ptr<float>();
	
	// What are the shapes of the input tensors
	int N = x.sizes()[0];
	int F = x.sizes()[1];
	int B = steps.sizes()[1] - 1;

	// Arrange the blocks
	dim3 nThr(256,1);
	dim3 nBlk((N+255)/256, F);
	
	// Allocate shared memory
	size_t shared_size = (2*B + 1) * sizeof(float);
	
	// Run the kernel
	histogram_kernel<<<nBlk, nThr, shared_size>>>(
	  count_data, x_data, steps_data, F, N, B);

	printf("END histogram_cuda\n");
}



//----------------------
//  Histogram2D
//  output:
//   count    [F A B]     (int32)
//  input:
//     a      [N F]     (float32)
//     b      [N F]     (float32)
//   steps_a  [F A+1]   (float32)
//   steps_b  [F B+1]   (float32)
//----------------------
__global__
void histogram_2d_kernel(
	int* __restrict__ count_data,
	const float* __restrict__ a_data,
	const float* __restrict__ b_data,
	const float* __restrict__ steps_a_data,
	const float* __restrict__ steps_b_data,
	int F, int N, int A, int B)
{
	//printf("BEGIN histogram_kernel\n");

	// Who am I ?
	int n = blockIdx.x*blockDim.x + threadIdx.x;
	int f = blockIdx.y;


	//
	// Allocate the shared count and steps
	//
	extern __shared__ float s_data[];
	int   *count = (int*)s_data;                    // [A*B]
	float *steps_a = (float*)(s_data + A*B);        // [A+1]
	float *steps_b = (float*)(s_data + A*B + A+1);  // [B+1]

	//
	// Initialize the shared memory
	//
	int count_start = (A*B * threadIdx.x) / blockDim.x;
	int count_end   = (A*B * (threadIdx.x+1)) / blockDim.x; 
	int steps_a_start = ((A+1) * threadIdx.x) / blockDim.x;
	int steps_a_end   = ((A+1) * (threadIdx.x+1)) / blockDim.x; 
	int steps_b_start = ((A+1) * threadIdx.x) / blockDim.x;
	int steps_b_end   = ((A+1) * (threadIdx.x+1)) / blockDim.x; 
	//printf("rank  %d  count %d : %d  steps %d : %d  N %d B %d\n", 
	//	rank, count_start, count_end, steps_start, steps_end, N, B);
	for (int i=count_start; i<count_end; i++)
		count[i] = 0;
	for (int i=steps_a_start; i<steps_a_end; i++)
		steps_a[i] = steps_a_data[f*(A+1) + i];
	for (int i=steps_b_start; i<steps_b_end; i++)
		steps_b[i] = steps_b_data[f*(B+1) + i];
	__syncthreads();

	//
	// If we are even in the problem
	//

	if (n<N)
	{

		// Read the values
		float val_a = a_data[n*F + f];
		float val_b = b_data[n*F + f];

		// If the values are on the map
		if (val_a>=steps_a[0] && val_a<=steps_a[A] & val_b>=steps_b[0] && val_b<=steps_b[B])
		{
			//
			// Find the correct histogram bin in A (binary search)
			//
			int bin_lo  = 0;
			int bin_hi  = A;
			while (bin_hi-bin_lo > 1)
			{
				int bin_mid = (bin_lo+bin_hi) / 2;
				float val_mid = steps_a[bin_mid];
				
				if (val_a<val_mid) {
					bin_hi = bin_mid;
				} else {
					bin_lo = bin_mid;
				}
			}
			int bin_a = bin_lo;

			//
			// Find the correct histogram bin in B (binary search)
			//
			bin_lo  = 0;
			bin_hi  = B;
			while (bin_hi-bin_lo > 1)
			{
				int bin_mid = (bin_lo+bin_hi) / 2;
				float val_mid = steps_b[bin_mid];
				
				if (val_b<val_mid) {
					bin_hi = bin_mid;
				} else {
					bin_lo = bin_mid;
				}
			}
			int bin_b = bin_lo;

			//
			// Add to the bin
			//
			atomicAdd(&count[bin_a*B + bin_b], 1);	
		}
	}

	//
	// Global barrier
	//
	__syncthreads();
		
	if (n<N)
	{
		//
		// Add to the overall output
		//
		for (int i=count_start; i<count_end; i++) {
			atomicAdd(&count_data[f*A*B + i], count[i]);
		}
	}
}


//----------------------
//  Histogram2D
//  output:
//   count    [F A B]     (int32)
//  input:
//     a      [N F]     (float32)
//     b      [N F]     (float32)
//   steps_a  [F A+1]   (float32)
//   steps_b  [F B+1]   (float32)
//-----------------------
void histogram_2d_cuda(
	torch::Tensor count,
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor steps_a,
	torch::Tensor steps_b)
{
	//printf("BEGIN histogram_cuda\n");

	// Pointer to the data
	int     *count_data = count.data_ptr<int>();
	float   *a_data   = a.data_ptr<float>();
	float   *b_data   = a.data_ptr<float>();
	float   *steps_a_data   = steps_a.data_ptr<float>();
	float   *steps_b_data   = steps_b.data_ptr<float>();
	
	// Get the sizes
	int F = count.sizes()[0];
	int N = a.sizes()[0];
	int A = count.sizes()[1];
	int B = count.sizes()[2];

	// Allocate shared memory
	size_t shared_size = (A*B + A+1 + B+1) * sizeof(float);
	
	dim3 nThr(256,1);
	dim3 nBlk((N+255)/256, F);
	
	// Run the kernel
	histogram_2d_kernel<<<nBlk,nThr,shared_size>>>(
		count_data,
		a_data, b_data,
		steps_a_data, steps_b_data,
		F, N, A, B);
	  
	//printf("END histogram_cuda\n");
}







//----------------------
//  Quantiles Kernel
//  output:
//   quantiles [F B+1]   (float32)
//  input:
//   count     [F B]     (int64)
//   steps     [F B+1]   (float32)
//     F       nFilters  (int)
//     B       nBins     (int)
//-----------------------

__global__ void quantiles_kernel(float *quantiles_data, long long *count_data, float *steps_data, int F, int B)
{
	// Who am I ?
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int n_thread = blockDim.x;
	int rank = bid * n_thread + tid;

	// Shard on nFilters
	if (rank<F)
	{
		// Pointer to the beginning of the shard
		long long *count     = count_data     + (rank *  B   );
		float     *steps     = steps_data     + (rank * (B+1));
		float     *quantiles = quantiles_data + (rank * (B+1));
		
		// Figure out the total count for the histogram
		long long total = 0;
		for (int i=0; i<B; i++)
			total += count[i];
		long long sum=0;         // current total
		
		// What's the coordinates of the first CDF step
		int s = 0;
		double x0 = (double)(steps[0]);
		double x1 = (double)(steps[1]);
		double y0 = 0.0;
		double y1 = (double)(count[0]) / (double)total;
		sum += count[0];
		
		// Hard-code the first quantile
		quantiles[0] = steps[0];
		
		// For every middle quantile
		for (int q=1; q<B; q++) {
				
			// What is the current y value ?
			double y = (double)q / (double)B;
		
			//printf("-----------------\n");
			//#printf("q  %d       y %.4f   s %d  x [ %.4f  %.4f ]   y [ %.4f  %.4f ]  \n", 
			//	q, y, s, x0, x1, y0, y1);

		
			// Keep stepping forward until we find the correct step
			while (y1<y && s<B) {
				s++;
				x0=x1;
				y0=y1;
				x1 = (double)(steps[s+1]);
				sum += count[s];
				y1 = (double)sum / (double)total;

				//printf("q  %d       y %.4f   s %d  x [ %.4f  %.4f ]   y [ %.4f  %.4f ]  \n", 
				//	q, y, s, x0, x1, y0, y1);
			}
			
			// Estimate the correct quantile
			double x;
			if (y<=y0) {        // Unexpected corner case
				x = x0;
			}
			else if (y<y1) {    // Linear interpolation
				double delta = (y-y0) / (y1-y0 + 0.00000001);
				x = x0 + delta*(x1-x0);
			}
			else if (y==y1) {   // Unexpected corner case
			}
			
			// Save this quantile
			quantiles[q] = x;
		}
		
		// Hard-code the last quantile
		quantiles[B] = steps[B];
	}
}


//----------------------
//  Quantiles Cuda
//  input:
//   count    [F B]     (int64)
//   steps    [F B+1]   (float32)
//  output:
//  quantiles [F B+1]   (float32)
//-----------------------
torch::Tensor quantiles_cuda(torch::Tensor count, torch::Tensor steps)
{
	std::cout << "BEGIN quantiles_cuda" << std::endl;
	//std::cout << "count " << std::endl;
	//std::cout << count.sizes() << std::endl;
	//std::cout << count.dtype() << std::endl;
	//std::cout << "steps " << std::endl;
	//std::cout << steps.sizes() << std::endl;
	//std::cout << steps.dtype() << std::endl;
	
	// Pointer to the data
	long long *count_data  = (long long*)count.data_ptr<int64_t>();
	float *steps_data  = steps.data_ptr<float>();
	
	//printf("10\n");
	
	// What are the shapes of the input tensors
	int F = count.sizes()[0];
	int B = count.sizes()[1];
	int Bplus1 = B+1;
	
	//printf("20\n");
	
	// Construct quantiles [B]
	auto options = torch::TensorOptions()
		.dtype(torch::kFloat32)  // Data type (e.g., kInt, kDouble, kHalf)
		.device(torch::kCUDA)    // Device (kCPU or kCUDA)
		.requires_grad(false);   // Autograd tracking
	torch::Tensor quantiles = torch::zeros({F, Bplus1}, options);

	//printf("30\n");

	float *quantiles_data  = quantiles.data_ptr<float>();

	//printf("40\n");

	// Run the kernel
	quantiles_kernel<<<(F+255)/256, 256>>>(quantiles_data, count_data, steps_data, F, B);

	std::cout << "END quantiles_cuda" << std::endl;

	
	return quantiles;
}





//----------------------
//  Quantiles Bounds Kernel
//  output:
//   guess_steps  [F B+1]   (float32)
//  in/out:
//   quantiles_lo_x [F B+1]   (float32)
//   quantiles_lo_y [F B+1]   (float32)
//   quantiles_hi_x [F B+1]   (float32)
//   quantiles_hi_y [F B+1]   (float32)
//  input:
//   count     [F B]     (int64)
//   steps     [F B+1]   (float32)
//     F       nFilters  (int)
//     B       nBins     (int)
//-----------------------

__global__ void quantiles_bounds_kernel(
	float *guess_steps_data,
	float *quantiles_lo_x_data,
	float *quantiles_lo_y_data,
	float *quantiles_hi_x_data,
	float *quantiles_hi_y_data,
	long long *count_data,
	float *steps_data,
	int F, int B)
{
	// Who am I ?
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int n_thread = blockDim.x;
	int rank = bid * n_thread + tid;

	// Shard on nFilters
	if (rank<F)
	{
		// Pointer to the beginning of the shard
		float     *guess_steps    = guess_steps_data    + (rank * (B+1));
		float     *quantiles_lo_x = quantiles_lo_x_data + (rank * (B+1));
		float     *quantiles_lo_y = quantiles_lo_y_data + (rank * (B+1));
		float     *quantiles_hi_x = quantiles_hi_x_data + (rank * (B+1));
		float     *quantiles_hi_y = quantiles_hi_y_data + (rank * (B+1));
		long long *count          = count_data     + (rank *  B   );
		float     *steps          = steps_data     + (rank * (B+1));
		
		// Figure out the total count for the histogram
		long long total = 0;
		for (int i=0; i<B; i++)
			total += count[i];
		long long sum=0;         // current total
		
		//printf("-------------------------\n");
		//printf(" Print counts\n");
		//for (int q=0; q<B; q++)
		//	printf("%d  %d\n", q, (int)(count[q]));
		//printf("total %d\n", (int)total);
	
		// What's the coordinates of the first CDF step
		int s = 0;
		float step_x0 = (float)(steps[0]);
		float step_x1 = (float)(steps[1]);
		float step_y0 = 0.0;
		float step_y1 = (float)((double)(count[0]) / (double)total);
		sum += count[0];
		
		//printf("first  x0 %.7f x1 %.7f  y0 %.7f y1 %.7f\n",
		//	step_x0,step_x1,step_y0,step_y1);
		
		// Hard-code the first quantile
		quantiles_lo_x[0] = steps[0];
		quantiles_lo_y[0] = 0.0;
		quantiles_hi_x[0] = steps[0];
		quantiles_hi_y[0] = 0.0;
		guess_steps[0]    = steps[0];
		
			
		//-----------
		// For every middle quantile
		//-----------
		for (int q=1; q<B; q++) {
			
			//printf("-------------------------\n");
			
			//-----
			// What is the current y value ?
			//-----
			float y = (float)( (double)q / (double)B );
		
			//-----
			// Keep stepping forward until we find the correct step
			//-----
			//printf("q  %d  y %.7f   x0 %.7f x1 %.7f  y0 %.7f y1 %.7f\n",
			//	q,y,step_x0,step_x1,step_y0,step_y1);
			while (step_y1<y && s<B) {
				s++;
				step_x0 = step_x1;
				step_y0 = step_y1;
				sum += count[s];
				step_x1 = (float)(   steps[s+1]  );
				step_y1 = (float)(  (double)sum / (double)total  );

				//printf("q  %d  y %.7f   x0 %.7f x1 %.7f  y0 %.7f y1 %.7f\n",
				//	q,y,step_x0,step_x1,step_y0,step_y1);
			}
			float x0 = step_x0;
			float x1 = step_x1;
			float y0 = step_y0;
			float y1 = step_y1;

			//-----
			//  Tighten the interval so that
			//     x0,y0  and   quantiles_lo   are equal
			//     x1,y1  and   quantiles_hi   are equal
			//-----

			//printf("--\n");
			//printf("BEFORE update\n");
			//printf(" x0 %.7f x1 %.7f   y0 %.7f  y1 %.7f\n",x0,x1,y0,y1);
			//printf("quantiles_lo %.7f %.7f   hi  %.7f %.7f\n",
			//	quantiles_lo_x[q], quantiles_lo_y[q],
			//	quantiles_hi_x[q], quantiles_hi_y[q] );

			// Try to raise the quantile lower bound
			if (quantiles_lo_x[q] < x0) {
				quantiles_lo_x[q] = x0;
				quantiles_lo_y[q] = y0;
			}
			else {    // otherwise tighten the interval
				x0 = quantiles_lo_x[q];
				y0 = quantiles_lo_y[q];
			}
			
			// Try to lower the quantile upper bound
			if (x1 < quantiles_hi_x[q]) {
				quantiles_hi_x[q] = x1;
				quantiles_hi_y[q] = y1;
			}
			else {    // otherwise tighten the interval
				x1 = quantiles_hi_x[q];
				y1 = quantiles_hi_y[q];
			}
			
			//printf("AFTER update\n");
			//printf(" x0 %.3f x1 %.3f   y0 %.3f  y1 %.3f\n",x0,x1,y0,y1);
			//printf("quantiles_lo %.3f %.3f   hi  %.3f %.3f\n",
			//	quantiles_lo_x[q], quantiles_lo_y[q],
			//	quantiles_hi_x[q], quantiles_hi_y[q] );
			
			//-----
			// Linear interpolation to guess the next step
			//-----
			float x;
			if (y1-y0 < 0.000001) {
				x = 0.5*(x0+x1);          // average interval
			}
			else {
				float delta = (y-y0) / (y1-y0);        // interpolate interval
				delta = 0.9*delta + 0.1*0.5;      // Conservative delta (to improve convergence
				x = x0 + (x1-x0)*delta;
			}
			guess_steps[q] = x;
		}

		// Hard-code the last quantile
		quantiles_lo_x[B] = steps[B];
		quantiles_lo_y[B] = 1.0;
		quantiles_hi_x[B] = steps[B];
		quantiles_hi_y[B] = 1.0;
		guess_steps[B]    = steps[B];
	}
}





//----------------------
//  Quantiles Upper/Lower Bounds Cuda
//  output:
//   guess_steps       [F B+1]   (float32)
//  in/out:
//   quantiles_lo_x    [F B+1]   (float32)
//   quantiles_lo_y    [F B+1]   (float32)
//   quantiles_hi_x    [F B+1]   (float32)
//   quantiles_hi_y    [F B+1]   (float32)
//   count       [F B]     (int64)
//   steps       [F B+1]   (float32)
//-----------------------
void quantiles_bounds_cuda(
	torch::Tensor guess_steps,
	torch::Tensor quantiles_lo_x,
	torch::Tensor quantiles_lo_y,
	torch::Tensor quantiles_hi_x,
	torch::Tensor quantiles_hi_y,
	torch::Tensor count, 
	torch::Tensor steps)
{
	std::cout << "BEGIN quantiles_bounds_cuda" << std::endl;

	// Pointer to the data
	float *guess_steps_data  = guess_steps.data_ptr<float>();
	float *quantiles_lo_x_data = quantiles_lo_x.data_ptr<float>();
	float *quantiles_lo_y_data = quantiles_lo_y.data_ptr<float>();
	float *quantiles_hi_x_data = quantiles_hi_x.data_ptr<float>();
	float *quantiles_hi_y_data = quantiles_hi_y.data_ptr<float>();
	long long *count_data    = (long long*)count.data_ptr<int64_t>();
	float *steps_data        = steps.data_ptr<float>();
		
	// What are the shapes of the input tensors
	int F = count.sizes()[0];
	int B = count.sizes()[1];
	
	// Run the kernel
	quantiles_bounds_kernel<<<(F+255)/256, 256>>>(
		guess_steps_data,
		quantiles_lo_x_data,
		quantiles_lo_y_data,
		quantiles_hi_x_data,
		quantiles_hi_y_data,
		count_data, steps_data,
		F, B);


	std::cout << "END quantiles_bounds_cuda" << std::endl;
}




//----------------------------
//  Figure out the CDF
// output:
//   cdf        [F B+1]   (float32)
// input:
//   count      [F B]     (int64)
//----------------------------
__global__
void cdf_kernel(
	float* __restrict__ cdf_data,
	const int64_t* __restrict__ count_data,
	int F, int B)
{
	// Who am I ?
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int n_thread = blockDim.x;
	int rank = bid * n_thread + tid;

	if (rank>=F)
		return;
	
	// Find the right data pointers
	float* cdf             = cdf_data    +  rank*(B+1);
	const int64_t* count = count_data  +  rank*B;
	
	// What is the grand total
	int64_t grand_total = 0;
	for (int b=0; b<B; b++)
		grand_total += count[b];
	
	// Fill in the cdf
	cdf[0] = 0.0;
	int64_t total = 0;
	for (int b=0; b<B; b++) {
		total += count[b];
		float y = (float)total / (float)grand_total;
		cdf[b+1] = y;
	}
}


//----------------------------
//  Figure out the Cumulative Distribution Function (CDF)
// output:
//   cdf        [F B+1]   (float32)
// input:
//   count      [F B]   (int64)
//----------------------------
void cdf_cuda(
	torch::Tensor cdf,
	torch::Tensor count)
{
	//printf("BEGIN cdf_cuda\n");
	
	// Pointer to the data
	float *cdf_data       = cdf.data_ptr<float>();
	int64_t *count_data = (int64_t*)count.data_ptr<int64_t>();
	
	// Get the shapes
	int cF = cdf.sizes()[0];
	int cB = cdf.sizes()[1]-1;
	int hF = count.sizes()[0];
	int hB = count.sizes()[1];
	if (cF!=hF) {
		printf("ERROR: quantiles_cdf_cuda nFeatures mismatch\n");
		exit(1);
	}
	if (cB!=hB) {
		printf("ERROR: quantiles_cdf_cuda n mBinsismatch\n");
		exit(1);
	}
	
	cdf_kernel<<<(cF+255)/256,256>>>(
		cdf_data,
		count_data,
		cF, cB);

	//printf("END   cdf_cuda\n");
}


//----------------------------
//    Probality Integral Transform   (PIT)
//  output:
//    pit    [N F]     (float32)
//  input:
//    X      [N F]     (float32)
//    cdf    [F B+1]   (float32)
//    steps  [F B+1]   (float32)
//----------------------------
__global__
void pit_kernel(
	float* __restrict__ pit,
	const float* __restrict__ X,
	const float* __restrict__ cdf_data,
	const float* __restrict__ steps_data,
	int N, int F, int B)
{
	// Who am I ?
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;
	int bdx  = blockDim.x;
	int bdy  = blockDim.y;
	
	// Where am I ?
	int n = bidx * bdx + tidx;
	int f = bidy * bdy + tidy;
	
	// Am I on the map?
	if (n>=N or f>=F)
		return;
	
	// Pointer into cdf and steps
	const float *cdf   = cdf_data   +  f*(B+1);
	const float *steps = steps_data +  f*(B+1);
	
	// What's the x value
	float x = X[n*F + f];
	float y;
	float x1 = steps[0];
	float x2 = steps[B];
	float y1 = cdf[0];
	float y2 = cdf[B];
	if (x<=x1)
		y = y1;
	else if (x>=x2)
		y = y2;
	else
	{
		// Binary Search
		int lo = 0;
		int hi = B;
		while (hi-lo>1) {
			int mid = (hi+lo)/2;
			float xmid = steps[mid];
			float ymid = cdf[mid];
			if (x < xmid) {
				hi = mid;
				x2 = xmid;
				y2 = ymid;
			}
			else {
				lo = mid;
				x1 = xmid;
				y1 = ymid;
			}
		}
		
		// Linear interpolation
		float delta = (x - x1) / (x2-x1 + 0.000001);
		y = y1 + (y2-y1)*delta;
	}

	// Write out the pit value
	pit[n*F + f] = y;
}


//----------------------------
//    Probality integral transform
//  output:
//    pit    [N F]     (float32)
//  input:
//    X      [N F]     (float32)
//    cdf    [F B+1]   (float32)
//    steps  [F B+1]   (float32)
//----------------------------
void pit_cuda(
	torch::Tensor pit,
	torch::Tensor X,
	torch::Tensor cdf,
	torch::Tensor steps)
{
	//printf("BEGIN pit_cuda\n");   fflush(stdout);
	
	// Pointer into the data
	float *pit_data   = pit.data_ptr<float>();
	float *X_data     = X.data_ptr<float>();
	float *cdf_data   = cdf.data_ptr<float>();
	float *steps_data = steps.data_ptr<float>();
	
	// Check the shapes
	int pN = pit.sizes()[0];
	int pF = pit.sizes()[1];
	int xN = X.sizes()[0];
	int xF = X.sizes()[1];
	int cF = cdf.sizes()[0];
	int cB = cdf.sizes()[1]-1;
	int sF = steps.sizes()[0];
	int sB = steps.sizes()[1]-1;
	
	// Check the shapes
	if (pN!=xN) {
		printf("ERROR quantiles_pit_cuda batch size mismatch\n");   fflush(stdout);
		exit(1);
	}
	if (pF!=xF || pF!=cF || pF!=sF) {
		printf("ERROR quantiles_pit_cuda nFeatures mismatch\n");   fflush(stdout);
		exit(1);
	}
	if (cB!=sB) {
		printf("ERROR quantiles_pit_cuda nBins mismatch\n");   fflush(stdout);
		exit(1);
	}

	// Launch the kernel
	dim3 nBlk, nThr;
	nThr.x = 256;
	nThr.y = 1;
	nThr.z = 1;
	nBlk.x = (pN+nThr.x-1) / nThr.x;
	nBlk.y = (pF+nThr.y-1) / nThr.y;
	nBlk.z = 1;
	//printf("pN %d pF %d xN %d xF %d cF %d cB %d sF %d sB %d\n", pN, pF, xN, xF, cF, cB, sF, sB);
	//printf("nBlk %d %d %d   nThr %d %d %d\n", nBlk.x, nBlk.y, nBlk.z, nThr.x, nThr.y, nThr.z);
	//fflush(stdout);
	pit_kernel<<<nBlk, nThr>>>(
		pit_data,X_data,
		cdf_data,steps_data,
		pN, pF, cB);

	//printf("END   pit_cuda\n");   fflush(stdout);
}

#define MAX_LEGENDRE_MOMENTS 11


//----------------------------
// Legendre polynomial copula
// output:
//    copula  [M M F]     (float32)     obs-vs-pred Legendre copula
// input:
//    obs:       [N F]    (float32)     PIT of obs values
//    pred:      [N F]    (float32)     PIT of pred values
//----------------------------
__global__
void copula_legendre_kernel(
	float* __restrict__ copula_data,
	const float* __restrict__ obs_data,
	const float* __restrict__ pred_data,
	int M, int N, int F)
{
	// Local storage
	__shared__ float local_copula[MAX_LEGENDRE_MOMENTS][MAX_LEGENDRE_MOMENTS];
	__shared__ float local_mom_obs[256][MAX_LEGENDRE_MOMENTS];
	__shared__ float local_mom_pred[256][MAX_LEGENDRE_MOMENTS];
	
	// Where am I ?
	int bidx = blockIdx.x;
	int tidx = threadIdx.x;
	int bdim = blockDim.x;
	int n0 = (tidx*N)/bdim;
	int n1 = ((tidx+1)*N)/bdim;
	int f  = bidx;

	// Pointer into the local moments
	float *mom_obs  = local_mom_obs[tidx];
	float *mom_pred = local_mom_pred[tidx];
	
	// Clear out local storage
	if (tidx<(MAX_LEGENDRE_MOMENTS*MAX_LEGENDRE_MOMENTS)) {          // Assumes bdim>MAX_LEGENDRE_MOMENTS^2 which should
		int mo = tidx/MAX_LEGENDRE_MOMENTS;                      //  always be true   bdim=256   MAX_LEGENDRE_MOMENTS^2=121
		int mp = tidx - (mo*MAX_LEGENDRE_MOMENTS);
		local_copula[mo][mp] = 0.0;
	}
	for (int i=0; i<MAX_LEGENDRE_MOMENTS; i++)
		mom_obs[i] = 0.0;
	for (int i=0; i<MAX_LEGENDRE_MOMENTS; i++)
		mom_pred[i] = 0.0;

	__syncthreads();
	
	
	// Loop over your data
	for (int n=n0; n<n1; n++)
	{	
		// What are values
		float oval = obs_data[n*F + f];     // Assumes already stretched to [-1 1]
		float pval = pred_data[n*F + f];
		
		// Calculate 1D moments
		mom_obs[0] = 1.0;
		mom_obs[1] = oval;
		mom_pred[0] = 1.0;
		mom_pred[1] = pval;

		// Bonnet's recurrence		
		for (int m=2; m<M; m++) {
			float over_m = 1.0 / m;
			mom_obs[m]  = over_m * ((2.0*(m-1)+1.0)*oval*mom_obs[m-1]  - (m-1.0)*mom_obs[m-2]);
			mom_pred[m] = over_m * ((2.0*(m-1)+1.0)*pval*mom_pred[m-1] - (m-1.0)*mom_pred[m-2]);
		}
		
		// Normalize by sqrt legendre integral
		//  see  cvpr-how-version/copula/scale_legendre.c
		//  for derivation.   these numbers are the 
		//  inverse sqrt of the output of this program.
		mom_obs[0] *= 0.7071067811865475;
		mom_obs[1] *= 1.2247448713915889;
		mom_obs[2] *= 1.5811388300841895;
		mom_obs[3] *= 1.8708286933869702;
		mom_obs[4] *= 2.1213203435596424;
		mom_obs[5] *= 2.3452078799117273;
		mom_obs[6] *= 2.5495097567963381;
		mom_obs[7] *= 2.7386127875255206;
		mom_obs[8] *= 2.9154759474239764;
		mom_obs[9] *= 3.0822070014802825;
		mom_obs[10] *= 3.0403539085279343;
		mom_pred[0] *= 0.7071067811865475;
		mom_pred[1] *= 1.2247448713915889;
		mom_pred[2] *= 1.5811388300841895;
		mom_pred[3] *= 1.8708286933869702;
		mom_pred[4] *= 2.1213203435596424;
		mom_pred[5] *= 2.3452078799117273;
		mom_pred[6] *= 2.5495097567963381;
		mom_pred[7] *= 2.7386127875255206;
		mom_pred[8] *= 2.9154759474239764;
		mom_pred[9] *= 3.0822070014802825;
		mom_pred[10] *= 3.0403539085279343;

		// Add to our copula for the batch
		for (int mo=0; mo<M; mo++) {
			for (int mp=0; mp<M; mp++) {
				atomicAdd(&(local_copula[mo][mp]), mom_obs[mo]*mom_pred[mp]);
			}
		}
	}

	__syncthreads();

	// Write to global   (only one block per feature, so no conflicts)
	if (tidx<(MAX_LEGENDRE_MOMENTS*MAX_LEGENDRE_MOMENTS)) {          // Assumes bdim>MAX_LEGENDRE_MOMENTS^2 which should
		int mo = tidx/MAX_LEGENDRE_MOMENTS;                      //  always be true   bdim=256   MAX_LEGENDRE_MOMENTS^2=121
		int mp = tidx - (mo*MAX_LEGENDRE_MOMENTS);
		copula_data[mo*M*F + mp*F + f]  +=   local_copula[mo][mp];
	}
}

//----------------------------
// Legendre polynomial copula
// output:
//    copula  [M M F]     (float32)     obs-vs-pred Legendre copula
// input:
//    obs:       [N F]    (float32)     PIT of obs values
//    pred:      [N F]    (float32)     PIT of pred values
//----------------------------
void copula_legendre_cuda(
	torch::Tensor copula,
	torch::Tensor obs,
	torch::Tensor pred)
{
	//printf("BEGIN copula_legendre_cuda\n");   fflush(stdout);

	// Pointer into the data
	float *copula_data = copula.data_ptr<float>();
	float *obs_data    = obs.data_ptr<float>();
	float *pred_data   = pred.data_ptr<float>();
	
	// Check shapes
	int cM1 = copula.sizes()[0];
	int cM2 = copula.sizes()[1];
	int cF  = copula.sizes()[2];
	int oN  = obs.sizes()[0];
	int oF  = obs.sizes()[1];
	int pN  = pred.sizes()[0];
	int pF  = pred.sizes()[1];
	if (cM1!=cM2) {
		printf("copula_legendre_cuda  num moments mismatch\n");   fflush(stdout);
		exit(1);
	}
	if (cM1>MAX_LEGENDRE_MOMENTS) {
		printf("ERROR: cM1 too many legendre moments\n");   fflush(stdout);
		exit(1);
	}
	if (cF!=oF || cF!=pF) {
		printf("copula_legendre_cuda  numFeatures mismatch\n");   fflush(stdout);
		exit(1);
	}
	if (oN!=pN) {
		printf("copula_legendre_cuda  num values mismatch\n");   fflush(stdout);
		exit(1);
	}
	
	// Run the kernel
	copula_legendre_kernel<<<cF, 256>>>(
		copula_data,
		obs_data,
		pred_data,
		cM1, oN, cF);
	
	
	//printf("END   copula_legendre_cuda\n");   fflush(stdout);
}


//----------------------------
// Plot the Legendre copula
// output:
//    plot   [Y X F]
// input:
//    copula [M M F]
//    Y X F  plot size
//    M      num moments
//----------------------------
__global__
void plot_copula_legendre_kernel(
	float* __restrict__ plot,
	const float* __restrict__ copula,
	int Y, int X, int F, int M)
{
	__shared__ float local_mom_obs[16][32][MAX_LEGENDRE_MOMENTS];
	__shared__ float local_mom_pred[16][32][MAX_LEGENDRE_MOMENTS];

	// Where am I ?
	int ix = blockDim.x*blockIdx.x + threadIdx.x;
	int iy = blockDim.y*blockIdx.y + threadIdx.y;
	int iz = blockDim.z*blockIdx.z + threadIdx.z;
	
	// Am I off the map ?
	if (ix>=X || iy>=Y || iz>=F)
		return;

	// Pointer into local storage
	float *mom_obs  = local_mom_obs[threadIdx.y][threadIdx.x];
	float *mom_pred = local_mom_pred[threadIdx.y][threadIdx.x];

	// Where am I (range [0.0 to 1.0])
	float fx = ((float)ix + 0.5) / X;
	float fy = ((float)iy + 0.5) / Y;
	
	// Rescale (range [-1.0 to 1.0])
	fx = 2.0*fx - 1.0;
	fy = 2.0*fy - 1.0;
	
	// Plug into legendre moments
	mom_obs[0] = 1.0;
	mom_obs[1] = fy;
	mom_pred[0] = 1.0;
	mom_pred[1] = fx;

	// Bonnet's recurrence		
	for (int m=2; m<M; m++) {
		float over_m = 1.0 / m;
		mom_obs[m]  = over_m * ((2.0*(m-1)+1.0)*fy*mom_obs[m-1]  - (m-1.0)*mom_obs[m-2]);
		mom_pred[m] = over_m * ((2.0*(m-1)+1.0)*fx*mom_pred[m-1] - (m-1.0)*mom_pred[m-2]);
	}
	
	// Normalize by sqrt legendre integral
	//  see  cvpr-how-version/copula/scale_legendre.c
	//  for derivation.   these numbers are the 
	//  inverse sqrt of the output of this program.
	mom_obs[0] *= 0.7071067811865475;
	mom_obs[1] *= 1.2247448713915889;
	mom_obs[2] *= 1.5811388300841895;
	mom_obs[3] *= 1.8708286933869702;
	mom_obs[4] *= 2.1213203435596424;
	mom_obs[5] *= 2.3452078799117273;
	mom_obs[6] *= 2.5495097567963381;
	mom_obs[7] *= 2.7386127875255206;
	mom_obs[8] *= 2.9154759474239764;
	mom_obs[9] *= 3.0822070014802825;
	mom_obs[10] *= 3.0403539085279343;
	mom_pred[0] *= 0.7071067811865475;
	mom_pred[1] *= 1.2247448713915889;
	mom_pred[2] *= 1.5811388300841895;
	mom_pred[3] *= 1.8708286933869702;
	mom_pred[4] *= 2.1213203435596424;
	mom_pred[5] *= 2.3452078799117273;
	mom_pred[6] *= 2.5495097567963381;
	mom_pred[7] *= 2.7386127875255206;
	mom_pred[8] *= 2.9154759474239764;
	mom_pred[9] *= 3.0822070014802825;
	mom_pred[10] *= 3.0403539085279343;
	
	// Calculate the plot value
	float sum = 0.0;
	for (int mo=0; mo<M; mo++) {
		for (int mp=0; mp<M; mp++) {
			sum += copula[mo*M*F + mp*F + iz] * mom_obs[mo]*mom_pred[mp];
		}
	}
	plot[iy*X*F + ix*F + iz] = sum;   // save the plot pixel
}


//----------------------------
// Plot the Legendre copula
// output:
//    plot   [Y X F]
// input:
//    copula [M M F]
//----------------------------
void plot_copula_legendre_cuda(
	torch::Tensor plot,
	torch::Tensor copula)
{
	printf("BEGIN plot_copula_legendre_cuda\n");   fflush(stdout);

	// Pointer into the data
	float *plot_data   = plot.data_ptr<float>();
	float *copula_data = copula.data_ptr<float>();
	
	// Check shapes
	int pY  = plot.sizes()[0];
	int pX  = plot.sizes()[1];
	int pF  = plot.sizes()[2];
	int cM1 = copula.sizes()[0];
	int cM2 = copula.sizes()[1];
	int cF  = copula.sizes()[2];
	if (cM1!=cM2) {
		printf("plot_copula_legendre_cuda  num moments mismatch\n");   fflush(stdout);
		exit(1);
	}
	if (cF!=pF) {
		printf("plot_copula_legendre_cuda  numFeatures mismatch\n");   fflush(stdout);
		exit(1);
	}
	
	// Run the kernel
	dim3 nBlk, nThr;
	nThr.x = 32;
	nThr.y = 16;
	nThr.z = 1;
	nBlk.x = (pX+nThr.x-1)/nThr.x;
	nBlk.y = (pY+nThr.y-1)/nThr.y;
	nBlk.z = (pF+nThr.z-1)/nThr.z;
	plot_copula_legendre_kernel<<<nBlk, nThr>>>(
		plot_data,
		copula_data,
		pY, pX, pF, cM1);
	
	
	printf("END   plot_copula_legendre_cuda\n");   fflush(stdout);
}









//-------------------------------------------------------------
//-------------------------------------------------------------
// CUDA version of fitting the q distribution to
//  the cdf.
//-------------------------------------------------------------
//-------------------------------------------------------------
#define DISTRIB_UNIFORM 0
#define DISTRIB_GAUSSIAN 1
#define DISTRIB_EXPONENTIAL 2
#define DISTRIB_WEIBULL 3
#define DISTRIB_GPD 4

#define PI 3.14159265358979323846264
#define SQRT2 1.41421356237

// Error function approximation (Abramowitz & Stegun, 1964)
__device__
float erf_approx(float x) {
    // Constants
    const float a1 = 0.254829592;
    const float a2 = -0.284496736;
    const float a3 = 1.421413741;
    const float a4 = -1.453152027;
    const float a5 = 1.061405429;
    const float p = 0.3275911;

    // Save the sign of x
    int sign = (x < 0) ? -1 : 1;
    x = fabsf(x);

    // Compute approximation
    float t = 1.0 / (1.0 + p * x);
    float y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

    return sign * y;
}

//----------------
//   in/out
//     guess          {param1, param2, param1_step, param2_step}
//   input
//     type
//-------------------------------
__device__
void BoundsDistribution(float *guess, int type)
{
	switch (type) {
		case DISTRIB_UNIFORM:
			if (guess[1] < 0.00001)
				guess[1] = fabsf(guess[1]) + 0.00001;  // halfwidth must be positive
			break;
		case DISTRIB_GAUSSIAN:
			if (guess[1] < 0.00001)
				guess[1] = fabsf(guess[1]) + 0.00001;  // stdev must be positive
			break;
		case DISTRIB_EXPONENTIAL:
			if (guess[0] < 0.00001)
				guess[0] = fabsf(guess[0]) + 0.00001;  // lamda must be positive
			break;
		case DISTRIB_WEIBULL:
			if (guess[0] < 0.00001)
				guess[0] = fabsf(guess[0]) + 0.00001;  // lamda must be positive
			if (guess[1] < 0.00001)
				guess[1] = fabsf(guess[1]) + 0.00001;  // k must be positive
			break;
		case DISTRIB_GPD:
			if (guess[0] < 0.00001)
				guess[0] = fabsf(guess[0]) + 0.00001;  // s must be positive
	}
}


//
// PlotDistribution using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
__device__
void PlotDistribution(int type, float param1, float param2, const float *X, float *Y, int startBin, int nBins) {
	int i;

	// premature optimization
	float unif_mu = param1;
	float unif_hw = param2;
	float unif_x0 = unif_mu - unif_hw;
	float unif_x1 = unif_mu + unif_hw;
	float unif_y  = 0.5 / unif_hw;
	float norm_mu = param1;
	float norm_sig = param2;
	float norm_coef = 1.0/sqrt(2.0*PI*norm_sig*norm_sig);
	float norm_scale = -1.0/(2.0*norm_sig*norm_sig);
	float exp_lamda = param1;
	float gam_alpha = param1;
	float gam_beta  = param2;
	float gam_coef  = pow(gam_beta, gam_alpha) / tgamma(gam_alpha);
	float wei_lam   = param1;
	float wei_k     = param2;
	float trunc_norm_F0 = 0.5 * (1.0 + erf(-norm_mu / (norm_sig*SQRT2)));
	//float trunc_norm_coef = 1.0 / (1.0 - trunc_norm_F0);
	float gpd_s    = param1;
	float gpd_xi   = param2;
	float gpd_s_over_xi = gpd_s / gpd_xi;
	float gpd_over_s = 1.0 / gpd_s;
	float gpd_over_xi = 1.0 / gpd_xi;

	// Plot the PDF
	for (i=startBin; i<nBins; i++)
	{
		// Pick x as the midpoint for midpoint rule
		float x = 0.5*(X[i]+X[i+1]);
		
		switch(type)
		{
		case DISTRIB_UNIFORM:
			if (x<unif_x0 || x>unif_x1)
				Y[i] = 0.0;
			else
				Y[i] = unif_y;
			break;

		case DISTRIB_GAUSSIAN:
				Y[i] = norm_coef * exp(norm_scale*(x-norm_mu)*(x-norm_mu));
			break;

		case DISTRIB_EXPONENTIAL:
			if (x<0.00001)
				Y[i] = 0.0;
			else
				Y[i] = exp_lamda * exp(-exp_lamda * x);
			break;

		case DISTRIB_WEIBULL:
			if (x<0.00001)
				Y[i] = 0.0;
			else
				Y[i] = (wei_k / wei_lam) * pow((x/wei_lam),(wei_k-1)) * exp(-pow((x/wei_lam),(wei_k)));
			break;

		case DISTRIB_GPD:
			// Negative x
			if (x<0.00001)
				Y[i] = 0.0;
			// Beyond maximum value
			else if (gpd_xi < -0.00001 && x>=-gpd_s_over_xi)
				Y[i] = 0.0;
			// Exponential case
			else if (gpd_xi >= -0.00001 && gpd_xi < 0.00001)
				Y[i] = gpd_over_s * exp(-x * gpd_over_s);
			// General case
			else
				Y[i] = gpd_over_s * pow( (1.0 + gpd_xi*x*gpd_over_s),  -(gpd_over_xi + 1.0));
			break;
		}
	}
}


//
// PlotDistribution CDF using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
__device__
void PlotCdfDistribution(int type, float param1, float param2, const float *X, float *Y, int startBin, int nBins) 
{
	int i;

	// premature optimization
	float unif_mu = param1;
	float unif_hw = param2;
	float unif_x0 = unif_mu - unif_hw;
	float unif_x1 = unif_mu + unif_hw;
	//float unif_y  = 0.5 / unif_hw;
	float norm_mu = param1;
	float norm_sig = param2;
	float norm_coef = 1.0/sqrt(2.0*PI*norm_sig*norm_sig);
	//float norm_scale = -1.0/(2.0*norm_sig*norm_sig);
	float exp_lamda = param1;
	//float gam_alpha = param1;
	//float gam_beta  = param2;
	//float gam_coef  = 1.0 / tgamma(gam_alpha);
	float wei_lam   = param1;
	float inv_wei_lam = 1.0 / wei_lam;
	float wei_k     = param2;
	float trunc_norm_F0 = 0.5 * ( 1 + erf_approx((0.0-norm_mu)/(norm_sig*1.41421356237)));
	float gpd_s    = param1;
	float gpd_xi   = param2;
	float gpd_s_over_xi = gpd_s / gpd_xi;
	float gpd_over_s = 1.0 / gpd_s;
	float gpd_over_xi = 1.0 / gpd_xi;

	// Plot the PDF
	for (i=startBin; i<nBins; i++)
	{
		// Pick x as the midpoint for midpoint rule
		float x = 0.5*(X[i]+X[i+1]);
		
		switch(type)
		{
		case DISTRIB_UNIFORM:
			if (x<unif_x0)
				Y[i] = 0.0;
			else if (x<unif_x1)
				Y[i] = (x-unif_x0) / (unif_x1-unif_x0);
			else
				Y[i] = 1.0;
			break;

		case DISTRIB_GAUSSIAN:// {
				//float A = x-norm_mu;
				//float B = norm_sig*1.41421356237;
				//float C = A/B;
				//float D = erf_approx(C);
				//float E = 0.5 * ( 1 + D );
				//Y[i] = E;
				//printf("i %d x %f norm_mu %f norm_sig %f x-norm_mu %f norm_sig*1.41421356237 %f (x-norm_mu)/(norm_sig*1.41421356237) %f erf %f Y %f\n", i, x, norm_mu, norm_sig, A,B,C,D,E);
				Y[i] = 0.5 * ( 1 + erf_approx((x-norm_mu)/(norm_sig*1.41421356237)));
			break;
			//}

		case DISTRIB_EXPONENTIAL:
			if (x<0.00001)
				Y[i] = 0.0;
			else
				Y[i] = 1.0 - exp(-exp_lamda * x);
			break;

		case DISTRIB_WEIBULL:
			if (x<0.00001)
				Y[i] = 0.0;
			else {
				//printf("i %d wei_lam %f  x %f  wei_k %f   -wei_lam*x %f   pow(-wei_lam*x,wei_k) %f exp  %f  Y %f\n",
				// i, wei_lam, x, wei_k, -wei_lam*x, -pow(wei_lam*x,wei_k), exp(-pow(wei_lam*x,wei_k)), 1.0 - exp(-pow(wei_lam * x, wei_k)) );
				Y[i] =  1.0 - exp(-pow(x * inv_wei_lam, wei_k));
			}
			break;

		case DISTRIB_GPD:
			// If out of bounds
			if (x<0.00001)
				Y[i] = 0.0;
			// Beyond maximum value
			else if (gpd_xi < -0.00001 && x>=-gpd_s_over_xi)
				Y[i] = 1.0;
			// Exponential case
			else if (gpd_xi >= -0.00001 && gpd_xi < 0.00001)
				Y[i] = 1.0 - exp(-x * gpd_over_s);
			// General case
			else
				Y[i] = 1.0 - pow( (1.0 + gpd_xi*x*gpd_over_s),  -gpd_over_xi);
			break;
		}
	}
}


//
// PlotLogDistribution using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
__device__
void PlotLogDistribution(int type, float param1, float param2, const float *X, float *Y, int startBin, int nBins) {
	int i;

	// premature optimization
	float unif_mu = param1;
	float unif_hw = param2;
	float unif_x0 = unif_mu - unif_hw;
	float unif_x1 = unif_mu + unif_hw;
	float unif_y  = 0.5 / unif_hw;
	float norm_mu = param1;
	float norm_sig = param2;
	float norm_coef = 1.0/sqrt(2.0*PI*norm_sig*norm_sig);
	float norm_scale = -1.0/(2.0*norm_sig*norm_sig);
	float exp_lamda = param1;
	float gam_alpha = param1;
	float gam_beta  = param2;
	float gam_coef  = pow(gam_beta, gam_alpha) / tgamma(gam_alpha);
	float wei_lam   = param1;
	float wei_k     = param2;
	float trunc_norm_F0 = 0.5 * (1.0 + erf(-norm_mu / (norm_sig*SQRT2)));
	//float trunc_norm_coef = 1.0 / (1.0 - trunc_norm_F0);
	float e = 2.718281828459045235360287471352;
	float gpd_s    = param1;
	float gpd_xi   = param2;
	float gpd_s_over_xi = gpd_s / gpd_xi;
	float gpd_over_s = 1.0 / gpd_s;
	float gpd_over_xi = 1.0 / gpd_xi;

	// Plot the PDF
	for (i=startBin; i<nBins; i++)
	{
		// Pick x as the midpoint for midpoint rule
		float x = 0.5*(X[i]+X[i+1]);
		
		switch(type)
		{
		case DISTRIB_UNIFORM:
			if (x<unif_x0 || x>unif_x1)
				Y[i] = log2(1e-10);     // to avoid log2(0)
			else
				Y[i] = log2(unif_y);
			break;

		case DISTRIB_GAUSSIAN:
				
				//Y[i] = norm_coef * exp(norm_scale*(x-norm_mu)*(x-norm_mu));
				Y[i] = log2(norm_coef) + log2(e)*(norm_scale*(x-norm_mu)*(x-norm_mu));
			break;

		case DISTRIB_EXPONENTIAL:
			if (x<0.00001)
				Y[i] = log2(1e-10);     // to avoid log2(0)
			else
				//Y[i] = exp_lamda * exp(-exp_lamda * x);
				Y[i] = log2(exp_lamda) + log2(e)*(-exp_lamda * x);
			break;

		case DISTRIB_WEIBULL:
			if (x<0.00001)
				Y[i] = log2(1e-10);     // to avoid log2(0)
			else
				//Y[i] = (wei_k / wei_lam) * pow((x/wei_lam),(wei_k-1)) * exp(-pow((x/wei_lam),(wei_k)));
				Y[i] = log2(wei_k) - log2(wei_lam) + (wei_k - 1) * log2( x / wei_lam) - pow(x / wei_lam, wei_k)*log2(e);
				
			break;

		case DISTRIB_GPD:
			// Negative x
			if (x<0.00001)
				// Y[i] = 0.0
				Y[i] = log2(1e-10);     // to avoid log2(0)
			// Beyond maximum value
			else if (gpd_xi < -0.00001 && x>=-gpd_s_over_xi)
				//Y[i] = 0.0
				log2(1e-10);     // to avoid log2(0)
			// Exponential case
			else if (gpd_xi >= -0.00001 && gpd_xi < 0.00001)
				//Y[i] = gpd_over_s * exp(-x * gpd_over_s)
				Y[i] = log2(gpd_over_s) + log2(e)*(-gpd_over_s * x);
			// General case
			else
				//Y[i] = gpd_over_s * pow( (1.0 + gpd_xi*x*gpd_over_s),  -(gpd_over_xi + 1.0))
				Y[i] = log2(gpd_over_s) + log2( (1.0 + gpd_xi*x*gpd_over_s) ) * (-(gpd_over_xi + 1.0));
			break;
		}
	}
}


//
// Cross entropy using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
__device__
float CrossEntropy(
	const float *X, const float *Y, const float *Yhat,
	int startBin, int endBin)
{
	int i;

	// Calculate cross entropy
	float entropy = 0.0;
	for (i=startBin; i<endBin; i++) {
		float x0 = X[i];
		float x1 = X[i+1];
		float step = x1-x0;    // What is the step size of the bin ?
		
		float y = Y[i];
		float yhat = Yhat[i];
		entropy -= step * y * log2(yhat);
	}
	return entropy;
}


//
// Log of Cross entropy using the midpoint rule
//
//    X     length nBins+1   (for midpoint rule)
//    Y     length nBins     (at midpoints)
//    Yhat  length nBins     (at midpoints)
//
__device__
float LogCrossEntropy(
	const float *X, const float *Y, const float *log2Yhat,
	int startBin, int endBin)
{
	int i;

	// Calculate cross entropy
	float entropy = 0.0;
	for (i=startBin; i<endBin; i++) {
		float x0 = X[i];
		float x1 = X[i+1];
		float step = x1-x0;    // What is the step size of the bin ?
		
		float y = Y[i];
		float log2_yhat = log2Yhat[i];
		entropy -= step * y * log2_yhat;
	}
	return entropy;
}

__device__
float WassersteinDistance(
	const float *X, const float *cdfY, const float *cdfYhat,
	int startBin, int endBin)
{
	int i;
	
	float dist = 0.0;
	for (i=startBin; i<endBin; i++) {
		float x0 = X[i];
		float x1 = X[i+1];
		float step = x1-x0;    // What is the step size of the bin ?
		
		float cdf_y = cdfY[i];
		float cdf_yhat = cdfYhat[i];
		float diff = cdf_y - cdf_yhat;
		dist += step * fabsf(diff);
	}
	return dist;
}



//
// Initial guess using the midpoint rule
//
//   output:
//    guess    {guess_param1, guess_param2, guess_param1_step, guess_param2_step}
//
//   input:
//    type  type of distribution
//    X     length N+1   (for midpoint rule)
//    Y     length N     (at midpoints)
//
__device__
void InitialGuessDistribution(
	float* __restrict__ guess,
	int type,
	const float* __restrict__ X,
	const float* __restrict__ Y,
	int N)
{
	int i;

	// Calculate the mean
	float total = 0.0;
	float count = 0.0;
	for (i=0; i<N; i++) {
		float x = 0.5*(X[i]+X[i+1]);   // midpoint rule
		float step = X[i+1]-X[i];
		float weight = step*Y[i];
		total += x*weight;
		count += weight;
		//printf("total_mean %f\n", total);
		//printf("count_mean %f\n", count);
		//printf("bin %d - x: %f, total_mean: %f, count_mean: %f, step : %f, weight: %f\n", 
                       // i, x, total, count, step, weight);      
	}
	float mean = total / count;

	// Calculate the variance
	total = 0.0;
	count = 0.0;
	for (i=0; i<N; i++) {
		float x = 0.5*(X[i]+X[i+1]);   // midpoint rule
		float step = X[i+1]-X[i];
		float weight = step*Y[i];
		
		//printf("bin %d - x: %f, mean: %f, step: %f, weight: %f\n", i, x, mean, step, weight);
		
		total += (x-mean)*(x-mean)*weight;
		count += weight;
		//printf("total_var %f\n", total);
		//printf("count_var %f\n", count);
		//printf("bin %d - total_var: %f, count_var: %f, weight: %f\n", 
                     //   i, total, count, weight);
	}
	float variance = total / count;

	// Calculate standard deviation
	float stdev = sqrt(variance);

	// Estimate parameters
	float param1;  //    mean       mean        lamda      alpha     lamda     s
	float param2;  // halfwidth     stdev        N/A       beta        k       xi
	float param1_step;
	float param2_step;
	switch(type) {
	case DISTRIB_UNIFORM:
		param1 = mean;    // mean
		param2 = stdev;   // halfwidth
		param1_step = fabsf(mean) + stdev;
		param2_step = stdev;
		break;
	case DISTRIB_GAUSSIAN:
		param1 = mean;    // mean
		param2 = stdev;   // stdev
		param1_step = fabsf(mean) + stdev;
		param2_step = stdev;
		break;
	case DISTRIB_EXPONENTIAL:
		param1 = 1.0 / mean;   // lamda
		param2 = 0.0;
		param1_step = param1;
		param2_step = 0.0;
		break;

	case DISTRIB_WEIBULL:              // exponential guess
		param1 = mean;
		param2 = 1.0;
		param1_step = mean;
		param2_step = 1.0;
		break;

	case DISTRIB_GPD:                  // exponential guess
		param1 = mean;
		param2 = 0.0;
		param1_step = mean;
		param2_step = 0.5;

	}

	//------
	// Pack up the initial guess
	//------
	guess[0] = param1;
	guess[1] = param2;
	guess[3] = param1_step;
	guess[4] = param2_step;
}






//-----------------
//  Avalanche mixup!   by   Appleby  SMhasher.
//
// Pseudorandom from [0, 2^32-1]
//   Extremely fast, but one known flaw,
//    random(0) = 0
//   Also, it is invertable, so not cryptographic
//   But excellent statistical properties!
//-----------------

__device__ uint32_t random(uint32_t h)
{
	h ^= h >> 16;
	h *= 0x85ebca6b;
	h ^= h >> 13;
	h *= 0xc2b2ae35;
	h ^= h >> 16;
	return h;
}

__device__ float uint_to_float(uint32_t u) {
    // 0x3f800000 is the bit representation of 1.0f
    // 0x007fffff masks the 23 bits for the mantissa
    unsigned int res = 0x3f800000 | (u >> 9);
    return __uint_as_float(res) - 1.0f;
}

// Pseudorandom float32 from [0, 1)
__device__ float random_f(uint32_t seed)
{
	return uint_to_float(random(seed));
}

// Pseudorandom float32 from [-1, 1)
__device__ float random_f_bal(uint32_t seed)
{
	return 2.0 * random_f(seed) - 1.0;
}


//-----------------
//  Fit a Distribution using the
//   method of simulated annealing
//
//  output:              [0]     [1]       [2]          [3]       [4]      [5]
//     distr          {param1, param2, cross_entropy, entropy, kl_diver, was_dist}
//     guess          {param1, param2, param1_step, param2_step}
//
//  input:
//     metric         QD_METRIC_KL,  or  QD_METRIC_WASSERSTEIN
//-----------------
#define _PARAM1 0
#define _PARAM2 1
#define _CROSS_ENTROPY 2
#define _ENTROPY 3
#define _KL_DIVER 4
#define _WAS_DIST 5

#define _PARAM1_STEP 2
#define _PARAM2_STEP 3

#define QD_METRIC_KL 0
#define QD_METRIC_WASSERSTEIN 1

__device__
void FitDistribution(          //       [0]     [1]       [2]          [3]       [4]
	float* distr,              //    {param1, param2, cross_entropy, entropy, kl_diver}
	float* guess,              //    {param1, param2, param1_step, param2_step}
	int type, int metric,
	const float *X,
	const float *Y,
	const float *cdf_Y,
	float *log_Yhat,
	float *cdf_Yhat,
	int N0, int N, int seed)
{
//qd_debug=1;
	int iter;
	distr[_PARAM1]        = -9999.0;
	distr[_PARAM2]        = -9999.0;
	distr[_CROSS_ENTROPY] = -9999.0;
	distr[_ENTROPY]       = -9999.0;
	distr[_KL_DIVER]      = -9999.0;
	distr[_WAS_DIST]      = -9999.0;

	// Initial guess . . .
	InitialGuessDistribution(guess, type, X, Y, N);
	BoundsDistribution(guess, type);

	// HACK
	if (type == DISTRIB_WEIBULL)
	{
		// Starting bin for initial guess fit
		int iN0 = N0;
		//if (iN0 < N/4)
		//	iN0 = N/4;
			
		// Second bin for initial guess fit
		int iN1 = iN0 + 9*(N-iN0) / 10;
		if (iN1 >= N)
			iN1 = N-1;
		float x0 = 0.5*(X[iN0]+X[iN0+1]);    // first x midpoint rule
		float x1 = 0.5*(X[iN1]+X[iN1+1]);    // second x midpoint rule
		float F0 = cdf_Y[iN0];
		float F1 = cdf_Y[iN1];
		//printf(" HACK x0 %f  x1 %f  F0 %f  F1 %f   N0 %d iN0 %d N %d iN1 %dn", x0, x1, F0, F1, N0, iN0, N, iN1);
		//input();
		
		// Strong initial guess using the two points
		float wei_k   = log( log(1-F0)/log(1-F1) ) / log( x0/x1 );
		float wei_lam = x0 / pow( -log(1-F0), 1.0/wei_k );

		// Store in the guess
		guess[_PARAM1] = wei_lam;
		guess[_PARAM2] = wei_k;
		guess[_PARAM1_STEP] = wei_lam;
		guess[_PARAM2_STEP] = wei_k;
	}


	// Calculate Entropy
	distr[_ENTROPY] = CrossEntropy(X, Y, Y, N0, N);

	// Fit using simulated annealing
	distr[_PARAM1] = guess[_PARAM1];
	distr[_PARAM2] = guess[_PARAM2];
	//PlotDistribution(type, param1, param2, X, log_Yhat, N0, N);
	PlotLogDistribution(type, distr[_PARAM1], distr[_PARAM2], X, log_Yhat, N0, N);
	PlotCdfDistribution(type, distr[_PARAM1], distr[_PARAM2], X, cdf_Yhat, N0, N);
	//float cross_entropy = CrossEntropy(X, Y, log_Yhat, N0, N);
	distr[_CROSS_ENTROPY] = LogCrossEntropy(X, Y, log_Yhat, N0, N);
	distr[_WAS_DIST]      = WassersteinDistance(X, cdf_Y, cdf_Yhat, N0, N);
	float loss = (metric==QD_METRIC_KL) ? distr[_CROSS_ENTROPY] : distr[_WAS_DIST];
	//printf(">   guess   %s %s %f %s %f cross_entr %f was_dist %f\n", distr_name[type], distr_p1_name[type], param1, distr_p2_name[type], param2, cross_entropy, was_dist);
	//if (qd_debug) {
	//	//printf("cdf_Y   ");
	//	for (i=0; i<10; i++) {
	//		int idx = N0 + i*(N-N0-1)/9;
	//		//printf("  %f", cdf_Y[idx]);
	//	}
	//	//printf("\n");
	//	//printf("cdf_Yhat");
	//	for (i=0; i<10; i++) {
	//		int idx = N0 + i*(N-N0-1)/9;
	//		//printf("  %f", cdf_Yhat[idx]);
	//	}
	//	//printf("\n");
	//	//printf("Y       ");
	//	for (i=0; i<10; i++) {
	//		int idx = N0 + i*(N-N0-1)/9;
	//		//printf("  %f", Y[idx]);
	//	}
	//	//printf("\n");
	//	PlotDistribution(type, param1, param2, X, temparray, N0, N);
	//	//printf("Yhat    ");
	//	for (i=0; i<10; i++) {
	//		int idx = N0 + i*(N-N0-1)/9;
	//		//printf("  %f", temparray[idx]);
	//	}
	//	//printf("\n");
	//}


	for (iter=0; iter<500; iter++) {
		int myseed = seed + 2*iter;
		guess[_PARAM1] = distr[_PARAM1] + random_f_bal(myseed)   * guess[_PARAM1_STEP];
		guess[_PARAM2] = distr[_PARAM2] + random_f_bal(myseed+1) * guess[_PARAM2_STEP];

		BoundsDistribution(guess, type);

		//PlotDistribution(type, new_param1, new_param2, X, log_Yhat, N);
		PlotLogDistribution(type, guess[_PARAM1], guess[_PARAM2], X, log_Yhat, N0, N);
		PlotCdfDistribution(type, guess[_PARAM1], guess[_PARAM2], X, cdf_Yhat, N0, N);
		float new_cross_entropy = LogCrossEntropy(X, Y, log_Yhat, N0, N);
		float new_was_dist      = WassersteinDistance(X, cdf_Y, cdf_Yhat, N0, N);
		float new_loss = (metric==QD_METRIC_KL) ? new_cross_entropy : new_was_dist;
		if (new_loss < loss && !isnan(new_loss) && !isinf(new_loss)) {
			distr[_PARAM1] = guess[_PARAM1];
			distr[_PARAM2] = guess[_PARAM2];
			distr[_CROSS_ENTROPY] = new_cross_entropy;
			distr[_WAS_DIST] = new_was_dist;
			distr[_KL_DIVER] = distr[_CROSS_ENTROPY] - distr[_ENTROPY];
			loss = new_loss;
			//if (qd_debug) {
			//	printf(">  new %s %s %f %s %f cross_entr %f was_dist %f\n", distr_name[type], distr_p1_name[type], param1,distr_p2_name[type], param2, cross_entropy, was_dist);
			//	printf("cdf_Y   ");
			//	for (i=0; i<10; i++) {
			//		int idx = N0 + i*(N-N0-1)/9;
			//		printf("  %f", cdf_Y[idx]);
			//	}
			//	printf("\n");
			//	printf("cdf_Yhat");
			//	for (i=0; i<10; i++) {
			//		int idx = N0 + i*(N-N0-1)/9;
			//		printf("  %f", cdf_Yhat[idx]);
			//	}
			//	printf("\n");
			//	printf("Y       ");
			//	for (i=0; i<10; i++) {
			//		int idx = N0 + i*(N-N0-1)/9;
			//		printf("  %f", Y[idx]);
			//	}
			//	printf("\n");
			//	PlotDistribution(type, new_param1, new_param2, X, temparray, N0, N);
			//	printf("Yhat    ");
			//	for (i=0; i<10; i++) {
			//		int idx = N0 + i*(N-N0-1)/9;
			//		printf("  %f", temparray[idx]);
			//	}
			//	printf("\n");
			//}
		}

		guess[_PARAM1_STEP] *= 0.97;   // falloff
		guess[_PARAM2_STEP] *= 0.97;   // falloff
	}

	//printf(">  solution  %s %s %f %s %f cross %f entro %f kl %f was %f\n",
	//	distr_name[type], distr_p1_name[type], param1,
	//	 distr_p2_name[type], param2,
	//	 cross_entropy, entropy, distr.kl_diver, distr.was_dist);
	//if(qd_debug)input();
	//if (qd_test) {
	//	printf("qd_test !!!!\n");
	//	for (i=N0; i<N; i++) {
	//		printf(" i %d cdf_Y %.16f cdf_Yhat %.16f Y %.16f Yhat %.16f\n", i, cdf_Y[i], cdf_Yhat[i], Y[i], temparray[i]);
	//	}
	//	qd_test=0;
	//}
	//return distr;
}



//-----------------
//  Fit a Distribution using the
//   method of simulated annealing
//
//  Define the following shape variables
//      N nBins    C nChannels
//
//  output:                        [0]     [1]         [2]         [3]       [4]      [5]
//     distr     [C 6]           {param1, param2, cross_entropy, entropy, kl_diver, was_dist}
//     guess     [C 4]           {param1, param2, param1_step, param2_step}
//     Y_mid     [C N]
//     cdf_Y_mid [C N]
//     log_Yhat  [C N]
//     cdf_Yhat  [C N]
//
//  input:
//     type           DISTRIB_UNIFORM  DISTRIB_GAUSSIAN  DISTRIB_EXPONENTIAL  DISTRIB_WEIBULL  DISTRIB_GPD
//     metric         QD_METRIC_KL,  or  QD_METRIC_WASSERSTEIN
//     Y0             starting percentile (float32)
//     X         [C N+1]
//     Y         [C N+1]
//
//-----------------
__global__
void fit_distributions_kernel(
	float* __restrict__ distr_data,         // output:  [C 5]  float32
	float* __restrict__ guess_data,         // output:  [C 4]  float32
	float* __restrict__ Y_mid_data,         // output:  [C N]  float32
	float* __restrict__ cdf_Y_mid_data,     // output:  [C N]  float32
	float* __restrict__ log_Yhat_data,      // output:  [C N]  float32
	float* __restrict__ cdf_Yhat_data,      // output:  [C N]  float32
	int type, int metric,
	float Y0,
	const float* __restrict__ X_data,             // input:   [C N+1]
	const float* __restrict__ cdf_Y_data,         // input:   [C N+1]
	int C, int N, int base_seed)
{
	// What is our rank  (shard on channel)
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	
	// If we're off the map, return
	if (c>C)
		return;
	
	// Pointer into the data  (shard on channel)
	float* distr       = distr_data     +  c * 6;
	float* guess       = guess_data     +  c * 4;
	float* Y_mid       = Y_mid_data     +  c * N;
	float* cdf_Y_mid   = cdf_Y_mid_data +  c * N;
	float* log_Yhat    = log_Yhat_data  +  c * N;
	float* cdf_Yhat    = cdf_Yhat_data  +  c * N;
	const float* X     = X_data         +  c * (N+1);
	const float* cdf_Y = cdf_Y_data     +  c * (N+1);
	int seed = base_seed + c*2*500;
	
	//----
	// Calculate the Y_mid and cdf_Y_mid (midpoint rule)
	//    output:  cdf_Y_mid   Y_mid   (filled tensors)
	//    output:  N0       starting bin given starting quantile Y0
	//----
	int N0 = -9999;
	float cdf1 = cdf_Y[0];
	float x1   = X[0];
	for (int n=0; n<N; n++) {
		float cdf0 = cdf1;
		float x0   = x1;
		cdf1 = cdf_Y[n];     // We have x0,x1  and cdf0,cdf1
		x1   = X[n];
		float area = cdf1-cdf0;
		float step = x1-x0;
		step = fmaxf(step,0.0000001);
		float y    = area / step;    // divide area  by base to get height
		cdf_Y_mid[n] = 0.5*(cdf1 + cdf0);
		Y_mid[n]     = y;
		
		if (N0==-9999 && cdf0>=Y0)
			N0 = n;
	}
	
	//----
	// Fit the distribution using simulated annealing
	//----
	FitDistribution(        //       [0]     [1]       [2]          [3]       [4]
		distr,              //    {param1, param2, cross_entropy, entropy, kl_diver}
		guess,              //    {param1, param2, param1_step, param2_step}
		type, metric,
		X,
		Y_mid,
		cdf_Y_mid,
		log_Yhat,
		cdf_Yhat,
		N0, N, seed);
}


//-----------------
//  Fit a Distribution using the
//   method of simulated annealing
//
//  Define the following shape variables
//      N nBins    C nChannels
//
//  output:                        [0]      [1]       [2]          [3]       [4]       [5]
//     distr     [C 6]           {param1, param2, cross_entropy, entropy, kl_diver, was_dist}
//     guess     [C 4]           {param1, param2, param1_step, param2_step}
//     Y_mid     [C N]               Y values of distribution using the midpoint rule
//     cdf_Y_mid [C N]           cdf_Y values of distribution using the midpoint rule
//     log_Yhat  [C N]
//     cdf_Yhat  [C N]
//
//  input:
//     type           DISTRIB_UNIFORM  DISTRIB_GAUSSIAN  DISTRIB_EXPONENTIAL  DISTRIB_WEIBULL  DISTRIB_GPD
//     metric         QD_METRIC_KL,  or  QD_METRIC_WASSERSTEIN
//     Y0             starting percentile (float32)
//     X         [C N+1]          X,Y values of distribution in histogrammer format
//     cdf_Y     [C N+1]
//
//  return:
//     seed (updated)
//
//-----------------
int fit_distributions_cuda(
	torch::Tensor distr,         // output:  [C 6]  float32
	torch::Tensor guess,         // output:  [C 4]  float32
	torch::Tensor Y_mid,         // output:  [C N]  float32
	torch::Tensor cdf_Y_mid,     // output:  [C N]  float32
	torch::Tensor log_Yhat,      // output:  [C N]  float32
	torch::Tensor cdf_Yhat,      // output:  [C N]  float32
	int type, int metric,
	float Y0,
	torch::Tensor X,             // input:   [C N+1]
	torch::Tensor cdf_Y,         // input:   [C N+1]
	int seed)
{
	printf("BEGIN fit_qdistr_cuda\n");  fflush(stdout);
	
	// Check Shapes . . .
	int dC   = distr.sizes()[0];
	int d6   = distr.sizes()[1];
	int gC   = guess.sizes()[0];
	int g4   = guess.sizes()[1];
	int ymC  = Y_mid.sizes()[0];
	int ymN  = Y_mid.sizes()[1];
	int cymC = cdf_Y_mid.sizes()[0];
	int cymN = cdf_Y_mid.sizes()[1];
	int lyhC = log_Yhat.sizes()[0];
	int lyhN = log_Yhat.sizes()[1];
	int xC   = X.sizes()[0];
	int xN   = X.sizes()[1]-1;
	int cyC  = cdf_Y.sizes()[0];
	int cyN  = cdf_Y.sizes()[1]-1;
	
	if (dC!=gC || dC!=ymC || dC!=cymC || dC!=lyhC || dC!=xC || dC!=cyC) {
		printf("ERROR: fit_distributions_cuda channels mismatch\n");
		exit(1);
	}
	if (d6!=6) {
		printf("ERROR: fit_distributions_cuda distribution must have 5 parts\n");
		exit(1);
	}
	if (g4!=4) {
		printf("ERROR: fit_distributions_cuda guess must have 4 parts\n");
		exit(1);
	}
	if (ymN!=cymN || ymN!=lyhN || ymN!=xN || ymN!=cyN) {
		printf("ERROR: fit_distributions_cuda num elements mismatch\n");
		exit(1);
	}
	
	// Pointer into data
	float* distr_data     = distr.data_ptr<float>();
	float* guess_data     = guess.data_ptr<float>();
	float* Y_mid_data     = Y_mid.data_ptr<float>();
	float* cdf_Y_mid_data = cdf_Y_mid.data_ptr<float>();
	float* log_Yhat_data  = log_Yhat.data_ptr<float>();
	float* cdf_Yhat_data  = cdf_Yhat.data_ptr<float>();
	float* X_data         = X.data_ptr<float>();
	float* cdf_Y_data     = cdf_Y.data_ptr<float>();
	
	// How many blocks
	dim3 thPerBlk(256,1,1);
	dim3 nBlk((xC+255)/256,1,1);
	
	// Launch the kernel
	fit_distributions_kernel<<<nBlk, thPerBlk>>>(
		distr_data,         // output:  [C 5]  float32
		guess_data,         // output:  [C 4]  float32
		Y_mid_data,         // output:  [C N]  float32
		cdf_Y_mid_data,     // output:  [C N]  float32
		log_Yhat_data,      // output:  [C N]  float32
		cdf_Yhat_data,      // output:  [C N]  float32
		type, metric,
		Y0,
		X_data,             // input:   [C N+1]
		cdf_Y_data,         // input:   [C N+1]
		xC, xN, seed);

	printf("END fit_qdistr_cuda\n");  fflush(stdout);
	
	return seed + xC * 2 * 500;     // seed  +  num consumed
}	