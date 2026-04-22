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











/*
**********************************************
* DEPRECATED BELOW
**********************************************


//----------------------
//  Histogram Kernel
//  input:
//     x        [N]
//   in_steps  [B+1]
//
//  output:
//   out_count  [B]
//-----------------------
__global__ void histogram_kernel(int* out_count, const float* x, const float *in_steps, int N, int B) 
{
	//printf("BEGIN histogram_kernel\n");

	// Who am I ?
	int bid = blockIdx.x;
	int tid = threadIdx.x;
	int n_thread = blockDim.x;
	int rank = bid * n_thread + tid;

	//
	// Allocate the shared count and steps
	//
	extern __shared__ float s_data[];
	int   *count = (int*)s_data;          // [B]
	float *steps = (float*)(s_data + B);  // [B+1]

	//
	// Initialize the shared memory
	//
	int count_start = (B * tid) / n_thread;
	int count_end   = (B * (tid+1)) / n_thread; 
	int steps_start = ((B+1) * tid) / n_thread;
	int steps_end   = ((B+1) * (tid+1)) / n_thread; 
	//printf("rank  %d  count %d : %d  steps %d : %d  N %d B %d\n", 
	//	rank, count_start, count_end, steps_start, steps_end, N, B);
	for (int i=count_start; i<count_end; i++)
		count[i] = 0;
	for (int i=steps_start; i<steps_end; i++)
		steps[i] = in_steps[i];
	__syncthreads();

	//
	// If we are even in the problem
	//

	if (rank<N)
	{
		
		//
		// Find the correct histogram bin (binary search)
		//
		float val = x[rank];
		//printf("**  rank %d val %.3f\n", rank, val);
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

	//
	// Global barrier
	//
	__syncthreads();
		
	if (rank<N)
	{
		//
		// Add to the overall output
		//
		for (int i=count_start; i<count_end; i++) {
			atomicAdd(&out_count[i], count[i]);
		}
	}


	//printf("END histogram_kernel\n");
}


//----------------------
//  Histogram Cuda
//  output:
//   count    [B]     (int32)
//  input:
//     x      [N]     (float32)
//   steps    [B+1]   (float32)
//-----------------------
torch::Tensor histogram_cuda(torch::Tensor x, torch::Tensor steps)
{
	//printf("BEGIN histogram_cuda\n");

	// Pointer to the data
	float *x_data      = x.data_ptr<float>();
	float *steps_data  = steps.data_ptr<float>();
	
	// What are the shapes of the input tensors
	int N = x.sizes()[0];
	int B = steps.sizes()[0] - 1;
	
	// Construct count [B]
	auto options = torch::TensorOptions()
		.dtype(torch::kInt32)  // Data type (e.g., kInt, kDouble, kHalf)
		.device(torch::kCUDA)    // Device (kCPU or kCUDA)
		.requires_grad(false);   // Autograd tracking
	torch::Tensor count = torch::zeros({B}, options);
	int   *count_data  = count.data_ptr<int>();
	
	// Allocate shared memory
	size_t shared_size = (2*B + 1) * sizeof(float);
	
	//printf("(N + (N_THREADS-1)) / N_THREADS  %d\n", (N + (N_THREADS-1)) / N_THREADS);
	//printf("N_THREADS                        %d\n", N_THREADS);
	//printf("shared_size                      %d\n", shared_size);
	//printf("N %d  B %d\n", N, B);
	
	// Run the kernel
	histogram_kernel<<<(N + (N_THREADS-1)) / N_THREADS, N_THREADS, shared_size>>>(
	  count_data, x_data, steps_data, N, B);
	  
	//printf("END histogram_cuda\n");

	return count;
}
*/
