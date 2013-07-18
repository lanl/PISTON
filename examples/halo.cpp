
using namespace std;

//-------

#include <piston/halo_naive.h>
#include <piston/halo_kd.h>
#include <piston/halo_vtk.h>
#include <piston/halo_merge.h>

//-------

#include <sys/time.h>
#include <stdio.h>
#include <math.h>

#define STRINGIZE(x) #x
#define STRINGIZE_VALUE_OF(x) STRINGIZE(x)

using namespace piston;

// given three vectors, compare the i th  element of the first two vectors& write 0 or 1 in the third vector
struct compare
{
	int *a, *b, *c;

	__host__ __device__
	compare(int* a, int* b, int* c) :
		a(a), b(b), c(c) {}

	__host__ __device__
	void operator()(int i)
	{
		if(a[i] != b[i]) c[i] = 1;
	}
};

// given two vectors, compare their elements
void compareResults(thrust::device_vector<int> a, thrust::device_vector<int> b, int numOfParticles, string txt)
{
	thrust::device_vector<int> c(numOfParticles);
	thrust::fill(c.begin(), c.end(), 0);

	thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
			compare(thrust::raw_pointer_cast(&*a.begin()), thrust::raw_pointer_cast(&*b.begin()), thrust::raw_pointer_cast(&*c.begin())));
	int count = thrust::reduce(c.begin(), c.begin() + numOfParticles);

	std::string output = (count==0) ? txt+" - Result is the same" : txt+" - Result is NOT the same";
  std::cout << output << std::endl;
	if(count != 0) std::cout << "count " << count << std::endl << std::endl;
}

// given one vector & a txt file with results, compare their elements
void compareResultsTxt(string filename, int numOfParticles, thrust::device_vector<int> d, string txt)
{
	int num = 0;
	std::string line;
	thrust::device_vector<int> items(numOfParticles);

	std::ifstream *myfile = new std::ifstream(filename.c_str(), std::ios::in);
	while(!myfile->eof())
	{
		getline(*myfile,line);

		if(line=="") continue;

		if(num<numOfParticles)
			items[num++] = atof(strtok((char*)line.c_str(), " "));

		for(int i=1; i<15; i++)
		{
			if(num<numOfParticles)
				items[num++] = atof(strtok(NULL, " "));
		}
	}

	int count = 0;
	for(int i=0; i<numOfParticles; i++)
	{
		if(d[i] != items[i]) count++;
	}

	std::string output = (count==0) ? txt+" - Result is the same" : txt+" - Result is NOT the same";
	std::cout << output << std::endl << std::endl;
	if(count != 0) std::cout << "count " << count << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
  //---------------------------- set parameters

  halo *halo;

  float max_linkLength = 0.2;	// maximum linking length
  float min_linkLength = 0.2; // maximum linking length
  float linkLength     = 0.2; // linking length
  int   particleSize   = 1;		// particle size
  int   np = 256; // number of particles in one dimension
  float rL = 64;  // used to determine the scale factor when readig .cosmo data
  int   n  = 1;   //if you want a fraction of the file to load, use this.. 1/n
	
  char filename[1024]; // set file name
  sprintf(filename, "%s/24474Results/24474/24474", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
  std::string format = "cosmo";

  std::cout << "min_linkLength " << min_linkLength << std::endl;
  std::cout << "max_linkLength " << max_linkLength << std::endl;
  std::cout << "linkLength " << linkLength << std::endl;
  std::cout << "particleSize " << particleSize << std::endl;
  std::cout << filename << std::endl;
  std::cout << std::endl;

  //---------------------------- run different versions

//  std::cout << "Naive result" << std::endl;
//
//  halo = new halo_naive(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> a = halo->getHalos();
//
//  std::cout << "VTK based result (thrust version)" << std::endl;
//
//  halo = new halo_vtk(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> b = halo->getHalos();
//
  std::cout << "Kdtree based result" << std::endl;

  halo = new halo_kd(filename, format, n, np, rL);
  (*halo)(linkLength, particleSize);
  thrust::device_vector<int> c = halo->getHalos();

  std::cout << "Merge tree based result" << std::endl;

  halo = new halo_merge(min_linkLength, max_linkLength, true, filename, format, n, np, rL);
  (*halo)(linkLength, particleSize);
  thrust::device_vector<int> d = halo->getHalos();

  //---------------------------- compare results

	std::cout << "Comparing results" << std::endl;

	compareResultsTxt((string)filename+"_Vtk.txt", halo->numOfParticles, d, "Vtk vs Mergetree");
//	compareResults(a, c, halo->numOfParticles, "Naive vs Kdtree");
//	compareResults(b, c, halo->numOfParticles, "Vtk vs Kdtree");
	compareResults(c, d, halo->numOfParticles, "Kdtree vs Mergetree");

  std::cout << "--------------------" << std::endl;

	return 0;
}


