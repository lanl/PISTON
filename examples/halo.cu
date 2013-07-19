
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

struct compare
{
	int *a, *b, *c;

	__host__ __device__
	compare(int* a, int* b, int* c) :
		a(a), b(b), c(c) {}

	__host__ __device__
	void operator()(int i)
	{
		if(a[i] != b[i])
		{
			c[i] = 1;
//			std::cout << i << " " << a[i] << " " << b[i] << ", ";
		}
	}
};

bool compareResults(thrust::device_vector<int> a, thrust::device_vector<int> b, int numOfParticles)
{
	thrust::device_vector<int> c(numOfParticles);
	thrust::fill(c.begin(), c.end(), 0);

	thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
			compare(thrust::raw_pointer_cast(&*a.begin()), thrust::raw_pointer_cast(&*b.begin()), thrust::raw_pointer_cast(&*c.begin())));
	int result = thrust::reduce(c.begin(), c.begin() + numOfParticles);

	if(result==0)	return true;

	std::cout << " count " << result << std::endl;

	return false;
}

void compareWithVtk(string filename, int numOfParticles, thrust::device_vector<int> d)
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
		if(d[i] != items[i])
		{
			//std::cout << i << " - " << d[i] << " " << items[i] << std::endl;
			count++;
		}
	}

	std::string output = (count==0) ? "Vtk vs Mergetree - Result is the same" : "Vtk vs Mergetree - Result is NOT the same";
	std::cout << output << std::endl << std::endl;
	if(count != 0) std::cout << count << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
  halo *halo;

  float linkLength, max_linkLength, min_linkLength, rL;
  int   particleSize, np, n;

  max_linkLength = 0.25;
  min_linkLength = 0.19;
  linkLength     = 0.19;
  particleSize   = 1;//100;
  np = 256;
  rL = 64;
  n  = 1; //if you want a fraction of the file to load, use this.. 1/n

  char filename[1024];
// sprintf(filename, "%s/sub-8435", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
// std::string format = "csv";
//  sprintf(filename, "%s/sub-24474", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
//  std::string format = "csv";
//  sprintf(filename, "%s/sub-80289", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
//  std::string format = "csv";
//  sprintf(filename, "%s/256", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
//  std::string format = "cosmo";

  sprintf(filename, "%s/5005-sameCube", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
  std::string format = "csv";

//  sprintf(filename, "/home/wathsala/Cosmo/35015-sameCube", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
//  std::string format = "csv";
//	sprintf(filename, "/home/wathsy/Cosmo/256", STRINGIZE_VALUE_OF(DATA_DIRECTORY));
//  std::string format = "cosmo";

  std::cout << "min_linkLength " << min_linkLength << std::endl;
  std::cout << "max_linkLength " << max_linkLength << std::endl;
  std::cout << "linkLength " << linkLength << std::endl;
  std::cout << "particleSize " << particleSize << std::endl;
  std::cout << filename << std::endl;
  std::cout << std::endl;

  //----------------------------

//  std::cout << "Naive result" << std::endl;
//
//  halo = new halo_naive(filename, format, n, np, rL);
//  (*halo)(linkLength, particleSize);
//  thrust::device_vector<int> a = halo->getHalos();
//
//  std::cout << "VTK based result" << std::endl;
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

//	compareWithVtk("/home/wathsy/Cosmo/PISTONSampleData/24474Results2/391584_Vtk.txt", halo->numOfParticles, d);

  //----------------------------

//  std::cout << "Comparing results" << std::endl;
//  std::string output1 = (compareResults(a, c, halo->numOfParticles)==true) ? "Naive vs Kdtree     - Result is the same" : "Naive vs Kdtree        - Result is NOT the same";
//  std::cout << output1 << std::endl;
//  std::string output2 = (compareResults(b, c, halo->numOfParticles)==true) ? "Vtk vs Kdtree     - Result is the same" : "Vtk vs Kdtree     - Result is NOT the same";
//  std::cout << output2 << std::endl;
  std::string output3 = (compareResults(c, d, halo->numOfParticles)==true) ? "Kdtree vs Mergetree - Result is the same" : "Kdtree vs Mergetree - Result is NOT the same";
  std::cout << output3 << std::endl;
  std::cout << "--------------------" << std::endl;

//	std::cout << "a "; thrust::copy(a.begin(), a.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
//  std::cout << "c "; thrust::copy(c.begin(), c.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
//  std::cout << "d "; thrust::copy(d.begin(), d.begin()+halo->numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;	

  return 0;
}


