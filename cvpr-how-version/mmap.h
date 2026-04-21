#ifndef __MMAP_H__
#define __MMAP_H__


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <pthread.h>
#include "mmap.h"

#define MIN(a,b) ((a)<(b) ? (a) : (b))
#define MAX(a,b) ((a)>(b) ? (a) : (b))

#define testf  if(enableTest)printf

#define i64 unsigned long long int


static int enableTest;


// How many CPUs ?
static int NumCPU() {
	return sysconf( _SC_NPROCESSORS_ONLN );
}



//-------------------------------------------------------------
// Regular File IO
//-------------------------------------------------------------
static FILE *fopen_orDie(const char *name, const char *flag)
{
	FILE *f = fopen(name,flag);
	if (f==NULL) {
		printf("ERROR, could not read %s for %s\n", name, flag);
		exit(1);
	}
	return f;
}
static void fread_orDie(void * ptr, size_t size, size_t count, FILE * stream, const char *name)
{
	size_t read = fread(ptr, size, count, stream);
	if (read != count) {
		printf("ERROR failed to read %zu elements from %s (return %zu)\n", count, name, read);
		exit(1);
	}
}
static void fwrite_orDie(void * ptr, size_t size, size_t count, FILE * stream, const char *name)
{
	size_t write = fwrite(ptr, size, count, stream);
	if (write != count) {
		printf("ERROR failed to write %zu elements to %s (return %zu)\n", count, name, write);
		exit(1);
	}
}
static void *malloc_orDie(size_t size,  const char *comment)
{
	void *p = malloc(size);
	if (p==NULL) {
		printf("ERROR could not malloc %lld bytes for %s\n", (i64)size, comment);
		fflush(stdout); exit(1);
	}
	return p;
}
static void *calloc_orDie(size_t num, size_t sizeOf, const char *comment)
{
	void *p = calloc(num,sizeOf);
	if (p==NULL) {
		printf("ERROR could not calloc %lld elems of size %lld for %s\n",
			(i64)num, (i64)sizeOf, comment);
		fflush(stdout); exit(1);
	}
	return p;
}


//-------------------------------------------------------------
// The god-damn simplest possible pthread implementation that makes any fucking sense
//-------------------------------------------------------------
static void (*spawn_threads_func)(int);
static void *SpawnThreadsWrapper(void *_rank) {
	size_t rank = (size_t)_rank;
	spawn_threads_func(rank);
}
static void SpawnThreads(int numThreads, void (*func)(int) ) {
	size_t i;
	spawn_threads_func = func;
	pthread_t *threads = malloc_orDie(numThreads*sizeof(pthread_t), "SpawnThreads 'pthread_t *threads'");
	for (i=0; i<numThreads; i++)
		pthread_create(&(threads[i]), 0, SpawnThreadsWrapper, (void*)i);
	for (i=0; i<numThreads; i++)
		pthread_join  (threads[i], 0);
}

// MMap File IO
static int open_orDie(const char *pathname, int flags, mode_t mode) {
	int fd = open(pathname,flags,mode);
	if (fd == -1) {
		perror("Error opening file for writing");
		exit(1);
	}
	return fd;
}
static int ftruncate_orDie (int fd, off_t length) {
	int rt=ftruncate(fd,length);
	if (rt==-1) {
		perror("Error, could not ftruncate");
		exit(1);
	}
	return rt;
}
static void *mmap_orDie(void *addr, size_t length, int prot, int flags, int fd, off_t offset) {
	void *map = mmap(addr,length,prot,flags,fd,offset);
	if (map == MAP_FAILED) {
		perror("Error mmapping the file");
		exit(1);
	}
	return map;
}
static int msync_orDie(void *addr, size_t length) {
	int rt = msync(addr, length, MS_SYNC);
	if (rt == -1) {
		perror("Error msync");
		exit(1);
	}
	return rt;
}
static int munmap_orDie(void *addr, size_t length) {
	int rt=munmap(addr,length);
	if (rt == -1) {
		perror("Error un-mmapping the file");
		exit(1);
	}
	return rt;
}


// The god-damn simplest possible mmio wrapper that makes any fucking sense
static int OpenR(const char *name) {
	return open_orDie(name, O_RDONLY, 0600);
}
static off_t FileSize(int fd) {
	struct stat s;
	fstat(fd, &s);
	return s.st_size;
}
static int OpenRW(const char *name, size_t size) {
	int fd = open_orDie(name, O_RDWR | O_CREAT | O_TRUNC, (mode_t)0600);
	ftruncate_orDie(fd, size);
	return fd;
}
static void  *MMapR(int fd, size_t size) {
	return mmap_orDie(0, size, PROT_READ, MAP_SHARED, fd, 0);
}
static void  *MMapRW(int fd, size_t size) {
	return mmap_orDie(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
}
static int MUnMap(void *addr, size_t size) {
	msync_orDie(addr, size);    // Important synchronize the region
	return munmap_orDie(addr, size);
}



//-------------------------------------------------------------
//read big endian
//-------------------------------------------------------------
static int ReadInt(FILE *fin) {
	unsigned int r;
	r  = fgetc(fin) << 24;
	r |= fgetc(fin) << 16;
	r |= fgetc(fin) << 8;
	r |= fgetc(fin);
	return (int)r;
}
//read big endian
static short int ReadShort(FILE *fin) {
	unsigned short int r;
	r  = fgetc(fin) << 8;
	r |= fgetc(fin);
	return (short int)r;
}
static void ReadStr(char *str, int len, FILE *fin) {
	int rt=fread(str, 1, len, fin);
	str[len] = '\0';
}

//Performance timing
static double SysTime() {    //for profiling
        struct timeval tv;
        double time;
        gettimeofday (&tv, NULL);
        time = (double)((double)tv.tv_usec / 1000000.0);
        time += (double)tv.tv_sec;
        return time;
}


//-------------------------------------------------------------
// High-level Binfile interface for large arrays
//-------------------------------------------------------------
#define FLOAT32 1
#define INT32 2

typedef struct Binfile {
	void *data;
	int fd;
	i64 shape[8];
	int nshape;
	i64 nbytes;
	int type;
	int is_ronly;
} Binfile;

static Binfile MapBinfileR(char *path)
{
	int i;

	Binfile x;

	//-----------------
	// Read the metadata
	//-----------------
	char path2[4100];
	(sprintf)(path2, "%s.txt", path);

	FILE *f = fopen(path2,"r");
	if (f==NULL) {
		printf("ERROR cannot open %s for r\n", path2);
		exit(1);
	}

	// read the dims
	char dims[9][16];
	int n=fscanf(f, "%s %s %s %s %s %s %s %s %s", dims[0], dims[1], dims[2], dims[3], dims[4], dims[5], dims[6], dims[7], dims[8]);
	printf("n: %d\n", n);
	for (i=0; i<n; i++)
		printf("dims[%d] %s\n", i, dims[i]);

	// write out the dims
	x.nshape = n-1;
	for (i=0; i<x.nshape; i++) {
		x.shape[i] = atoi(dims[i]);
		printf("shape %d: %lld\n", i, x.shape[i]);
	}

	// read the type
	if (strcmp(dims[n-1], "int32")==0)
		x.type = INT32;
	else if (strcmp(dims[n-1], "float32")==0)
		x.type = FLOAT32;
	else {
		printf("ERROR unknown type %s\n", dims[n-1]);
		exit(1);
	}
	printf("type: %d %s\n", x.type, dims[n-1]);

	fclose(f);

	//-----------------
	// Memory map the file
	//-----------------
	x.nbytes = 4ull;
	for (i=0; i<x.nshape; i++)
		x.nbytes *= x.shape[i];
	printf("nbytes %lld\n", x.nbytes);

	// Open the file for reading
	x.fd  = OpenR(path);

	// Memory map the file
	x.data = MMapR (x.fd, x.nbytes);

	// flag it as readonly
	x.is_ronly = 1;

	// Return the mapped file
	return x;
}

static Binfile MapBinfileRW(char *path, int *shape, int type)
{
	int i;

	// Set some basic attributes
	Binfile x;
	x.is_ronly = 0;
	for (i=0; shape[i]; i++)
		x.shape[i] = shape[i];
	x.nshape = i;
	x.type = type;

	// how big?
	x.nbytes = 4ull;
	for (i=0; i<x.nshape; i++)
		x.nbytes *= x.shape[i];
	printf("nbytes %lld\n", x.nbytes);

	//------------------
	// Memory map the file
	//------------------
	x.fd   = OpenRW(path, x.nbytes );
	x.data = MMapRW(x.fd, x.nbytes);

	//------------------
	// Write the meta data
	//------------------
	char path2[4096];
	(sprintf)(path2, "%s.txt", path);
	FILE *f = fopen(path2, "w");
	for (i=0; i<x.nshape; i++)
		fprintf(f, "%lld ", x.shape[i]);
	if (type==INT32)
		fprintf(f, "int32\n");
	else
		fprintf(f, "float32\n");
	fclose(f);

	return x;
}

static void UnmapBinfile(Binfile x)
{
	// Synchroize and Unmap
	MUnMap(x.data, x.nbytes);

	// Close the file
	close(x.fd);
}



#endif // __MMAP_H__

