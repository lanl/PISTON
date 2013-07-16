#ifndef HALO_H_
#define HALO_H_

#include <thrust/copy.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/tuple.h>
#include <thrust/count.h>
#include <thrust/replace.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/partition.h>
#include <thrust/merge.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>
#include <thrust/binary_search.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/set_operations.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>

#include <piston/kd.h>

#include <sstream>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/time.h>
#include <cmath>

namespace piston
{

class halo
{
public:

	struct DataItem
	{
		float xx;  float yy;  float zz;
	  float vx;  float vy;  float vz;
	  float ms;  int   pt;

	  __host__ __device__
	  DataItem(){}

	  __host__ __device__
	  DataItem(float xx, float vx, float yy, float vy, float zz, float vz, float ms, int pt) :
		   xx(xx), vx(vx), yy(yy), vy(vy), zz(zz), vz(vz), ms(ms), pt(pt) {}
	};

	struct Point
	{
	    float x, y, z;

	    __host__ __device__
	    Point(){}

	    __host__ __device__
	    Point(float x, float y, float z) : x(x), y(y), z(z) {}
	};

  // definitions
  typedef thrust::device_vector<float>::iterator FloatIterator;
  typedef thrust::device_vector<int>::iterator   IntIterator;

  typedef thrust::tuple<float, float> Float2;

  typedef thrust::tuple<float, float, float> Float3;
  typedef thrust::tuple<FloatIterator, FloatIterator, FloatIterator> Float3TupleIterator;
  typedef thrust::zip_iterator<Float3TupleIterator> Float3zipIterator;

  typedef Float3 ParticleTuple;
  typedef Float3TupleIterator ParticleTupleIterator;
  typedef Float3zipIterator ParticleTupleZipIterator;

  typedef thrust::counting_iterator<int> CountingIterator;
  typedef thrust::constant_iterator<int> ConstantIterator;

  //inputs
  float linkLength;     // linking length used to link two particles
  int   particleSize;   // number of particles in a halo
  int   numOfParticles; // total number of particles
  float rL;
  int   np;             // number of particles in a dimension
  bool  periodic;

  int   numOfHalos;     		// total number of halos
  int   numOfHaloParticles;

  thrust::device_vector<float>  inputX, inputY, inputZ;		 // positions for each particle
  thrust::device_vector<int>    haloIndex;	  			       // halo indices for each particle
  thrust::device_vector<int>    haloIndexUnique;           // unique halo indexes
  thrust::device_vector<float>  haloColorsR, haloColorsG, haloColorsB; // colors for each halo

	int numOfHaloParticles_f;
	thrust::device_vector<int>    haloIndex_f;	  			      
  thrust::device_vector<float>  inputX_f, inputY_f, inputZ_f;    

	Point lBoundS, uBoundS; // lower & upper bounds of the entire space

  // variables needed to create random numbers
  thrust::default_random_engine rng;
  thrust::uniform_real_distribution<float> u01;

  halo(std::string filename="", std::string format=".cosmo", int n=1, int np=1, float rL=-1, bool periodic=false)
  {
    u01 = thrust::uniform_real_distribution<float>(0.0f, 1.0f);

    rL       = rL;
    np       = np;
    periodic = periodic;

    if(!readHaloFile(filename, n, np, rL, format))
    {
			readData(); // read in a txt data file
//			generateUniformData();		// generate uniformly spaced points		
//			  generateNonUniformData(); // generate nearby points to real data

/*
			numOfParticles = 8;
			inputX = thrust::host_vector<float>(numOfParticles);
			inputY = thrust::host_vector<float>(numOfParticles);
			inputZ = thrust::host_vector<float>(numOfParticles);

			inputX[0] = 1.0;	inputY[0] = 1.0;	inputZ[0] = 0.0;
			inputX[1] = 0.0;	inputY[1] = 4.0;	inputZ[1] = 0.0;
			inputX[2] = 8.0;	inputY[2] = 3.0;	inputZ[2] = 0.0;
			inputX[3] = 2.0;	inputY[3] = 6.0;	inputZ[3] = 0.0;
			inputX[4] = 4.0;	inputY[4] = 2.0;	inputZ[4] = 0.0;
			inputX[5] = 5.0;	inputY[5] = 5.0;	inputZ[5] = 0.0;
			inputX[6] = 3.0;	inputY[6] = 4.0;	inputZ[6] = 0.0;
			inputX[7] = 3.5;	inputY[7] = 4.5;	inputZ[7] = 0.0;
*/

			std::cout << "Test data loaded \n";
    }

    haloIndex.resize(numOfParticles);
    thrust::copy(CountingIterator(0), CountingIterator(0)+numOfParticles, haloIndex.begin());

    std::cout << "numOfParticles : " << numOfParticles << " \n";
  }

  virtual void operator()(float linkLength , int  particleSize) {}

	void readData()
	{
		std::cout << "readData \n";

    // open input file
    std::ifstream *myfile = new std::ifstream("/home/wathsy/Cosmo/PISTONSampleData/24474Results2/97896.txt", std::ios::in);
    if (!myfile->is_open()) {   std::cout << "File cannot be opened \n"; return; }

    std::vector<Point> vec;
    float x, y, z;

    std::string line;

		double minX, minY, minZ;
		double maxX, maxY, maxZ;

    while(!myfile->eof())
    {
			getline(*myfile,line);
			if(line=="") continue; 

			x = atof(strtok((char*)line.c_str(), " "));
			y = atof(strtok(NULL, " "));
			z = atof(strtok(NULL, " "));
			
			if(vec.size()==0)
			{
				minX = x; minY = y; minZ = z;
				maxX = x; maxY = y; maxZ = z;
			}
			else
			{		
				minX = (x<minX) ? x : minX; 	maxX = (x>maxX) ? x : maxX; 
				minY = (y<minY) ? y : minY;		maxY = (y>maxY) ? y : maxY;
				minZ = (z<minZ) ? z : minZ;		maxZ = (z>maxZ) ? z : maxZ;
			}

			vec.push_back(Point(x,y,z));
    }

		// set bounds
		lBoundS = Point(minX, minY, minZ);
		uBoundS = Point(maxX, maxY, maxZ);

    numOfParticles = vec.size();

		inputX.resize(numOfParticles);
		inputY.resize(numOfParticles);
		inputZ.resize(numOfParticles);

		for (int i=0; i<numOfParticles; i++)
    {
			Point p = vec.at(i);

			inputX[i] = p.x;
			inputY[i] = p.y;
			inputZ[i] = p.z;
    }

    vec.clear();
	}

	void writeData()
	{
		std::string filename = convertInt(numOfParticles) + ".txt";
		std::ofstream out(filename.c_str(), std::ofstream::out);	
		for(int i=0; i<numOfParticles; i++)
			out << inputX[i] << " " << inputY[i] << " " << inputZ[i] << std::endl;
		out.close();
	}

	void generateUniformData()
	{
		lBoundS = Point(0.1, 0.1, 0.1);
		uBoundS = Point(11.2, 11.2, 11.2);

		int nX=16, nY=16, nZ=16;

		numOfParticles = nX*nY*nZ;
		inputX = thrust::host_vector<float>(numOfParticles);
		inputY = thrust::host_vector<float>(numOfParticles);
		inputZ = thrust::host_vector<float>(numOfParticles);

		double startX=lBoundS.x;
		double startY=lBoundS.y;
		double startZ=lBoundS.z; 
		
		double stepX=(uBoundS.x-lBoundS.x)/(nX-1);
		double stepY=(uBoundS.y-lBoundS.y)/(nY-1);
		double stepZ=(uBoundS.z-lBoundS.z)/(nZ-1);

		int i = 0;
		for(int x=0; x<nX; x++)
		{
			for(int y=0; y<nY; y++)
			{
				for(int z=0; z<nZ; z++)
				{
					inputX[i] = (double)(startX+stepX*x);	
					inputY[i] = (double)(startY+stepY*y);	
					inputZ[i] = (double)(startZ+stepZ*z);
					i++;
				}
			}
		}	

		std::cout << "UniformData loaded \n";
	}

	void generateNonUniformData()
	{
		// variables needed to create random numbers
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> u;

		//readHaloFile("/home/wathsala/PISTONSampleData/256", 1, 256, 64.0f, "cosmo");	
		//readHaloFile("/home/wathsala/PISTONSampleData/sub-24474", 1, 256, 64.0f, "csv");	

		readHaloFile("/home/wathsy/Cosmo/PISTONSampleData/sub-24474", 1, 256, 64.0f, "csv");
		//readHaloFile("/home/wathsy/Cosmo/PISTONSampleData/35015-sameCube", 1, 256, 64.0f, "csv");	

		int n = 1;

		inputX.resize(numOfParticles*n);
		inputY.resize(numOfParticles*n);
		inputZ.resize(numOfParticles*n);

		u = thrust::uniform_real_distribution<float>(-0.5f, 0.5f);
		for(int i=0; i<numOfParticles; i++)
		{
			for(int j=1; j<n; j++)
			{
				double x = inputX[i] + u(rng);
				double y = inputY[i] + u(rng);
				double z = inputZ[i] + u(rng);

				while(x<lBoundS.x || x>uBoundS.x) x = inputX[i] + u(rng);
				while(y<lBoundS.y || y>uBoundS.y) y = inputY[i] + u(rng);
				while(z<lBoundS.z || z>uBoundS.z) z = inputZ[i] + u(rng);		

				inputX[numOfParticles + i*(n-1) + (j-1)] = x;
				inputY[numOfParticles + i*(n-1) + (j-1)] = y;
				inputZ[numOfParticles + i*(n-1) + (j-1)] = z;				
			}
		}

		numOfParticles *= n;

		writeData();
	}	

	std::string convertInt(int number)
	{
		std::stringstream ss;//create a stringstream
		ss << number;//add number to the stream
		return ss.str();//return a string with the contents of the stream
	}

	// get lower & upper bounds of the entire space
	void getBounds()
	{
		typedef thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> result_type;
		result_type result1 = thrust::minmax_element(inputX.begin(), inputX.end());
		result_type result2 = thrust::minmax_element(inputY.begin(), inputY.end());
		result_type result3 = thrust::minmax_element(inputZ.begin(), inputZ.end());

		lBoundS = Point(*result1.first,  *result2.first,  *result3.first);
		uBoundS = Point(*result1.second, *result2.second, *result3.second);
	}

  // get start of vertices
  Float3zipIterator vertices_begin()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(inputX.begin(), inputY.begin(), inputZ.begin()));
  }

  // get end of vertices
  Float3zipIterator vertices_end()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(inputX.end(), inputY.end(), inputZ.end()));
  }

	// get start of vertices_f
  Float3zipIterator vertices_begin_f()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(inputX_f.begin(), inputY_f.begin(), inputZ_f.begin()));
  }

  // get end of vertices_f
  Float3zipIterator vertices_end_f()
  {
    return thrust::make_zip_iterator(thrust::make_tuple(inputX_f.end(), inputY_f.end(), inputZ_f.end()));
  }

  // get start of halos
  IntIterator halos_begin()
  {
    return haloIndex.begin();
  }

  // get end of halos
  IntIterator halos_end()
  {
    return haloIndex.end();
  }

	// get start of halos_f
  IntIterator halos_begin_f()
  {
    return haloIndex_f.begin();
  }

  // get end of halos_f
  IntIterator halos_end_f()
  {
    return haloIndex_f.end();
  }

  // get the color of halo i
  Float3 getColor(int i)
  {
    return thrust::make_tuple(haloColorsR[i], haloColorsG[i], haloColorsB[i]);
  }

  // get the index in haloIndexUnique for a halo i
  int getHaloInd(int i, bool useF=false)
  {
		int id = (useF) ? haloIndex_f[i] : haloIndex[i];
    IntIterator ite = thrust::find(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfHalos, id);		
    return (ite!=haloIndexUnique.begin()+numOfHalos) ? ite - haloIndexUnique.begin() : -1;
  }

  // read input file - currently can read a .cosmo file or a .csv file
  // .csv file usage - when you have a big data file and you want a piece of it, load it in VTK and slice it and save it as .csv file
  bool readHaloFile(std::string filename, float n, int np, float rL, std::string format)
  {
    // check filename
    if(filename == "") { std::cout << "no input file specified \n"; return false; }

    if(format == "csv")    return readCsvFile(filename, n, np, rL, format);

    if(format == "cosmo")  return readCosmoFile(filename, n, np, rL, format);

    return false;
  }

  // read a .cosmo file and load the data to inputX, inputY & inputZ
	bool readCosmoFile(std::string filename, float n, int np, float rL, std::string format)
	{
		std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
		if (!myfile->is_open()) {   std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

		// compute the number of particles
		myfile->seekg(0L, std::ios::end);
		numOfParticles = myfile->tellg() / 32;  // get particle size in file

		// resize inputX, inputY, inputZ temp vectors
		thrust::device_vector<float> inputXtmp(numOfParticles);
		thrust::device_vector<float> inputYtmp(numOfParticles);
		thrust::device_vector<float> inputZtmp(numOfParticles);

		// scale amount for particles
    float xscal;
    if(rL==-1) xscal = 1;
    else       xscal = rL / (1.0*np);

		// rewind file to beginning for particle reads
		myfile->seekg(0L, std::ios::beg);

		// declare temporary read buffers
		int nfloat = 7, nint = 1;
		float fBlock[nfloat];
		int iBlock[nint];

		for (int i=0; i<numOfParticles; i++)
		{
			// Set file pointer to the requested particle
			myfile->read((char *)fBlock, nfloat * sizeof(float));

			if (myfile->gcount() != (int)(nfloat * sizeof(float))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			myfile->read((char *)iBlock, nint * sizeof(int));
			if (myfile->gcount() != (int)(nint * sizeof(int))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			// These files are always little-endian
			//vtkByteSwap::Swap4LERange(fBlock, nfloat);
			//vtkByteSwap::Swap4LERange(iBlock, nint);

			// sanity check
/*
			if (fBlock[0] > rL || fBlock[2] > rL || fBlock[4] > rL) {
				std::cout << "rL is too small" << std::endl;
				exit (-1);
			}
*/
			inputXtmp[i] = fBlock[0] / xscal;
			inputYtmp[i] = fBlock[2] / xscal;
			inputZtmp[i] = fBlock[4] / xscal;
		}

		// get bounds of the space
		typedef thrust::pair<thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator> result_type;
		result_type result1 = thrust::minmax_element(inputXtmp.begin(), inputXtmp.end());
		result_type result2 = thrust::minmax_element(inputYtmp.begin(), inputYtmp.end());
		result_type result3 = thrust::minmax_element(inputZtmp.begin(), inputZtmp.end());

		lBoundS = Point(*result1.first,  *result2.first,  *result3.first);
		uBoundS = Point(*result1.second, *result2.second, *result3.second);

		// get the fraction wanted		
		numOfParticles = numOfParticles / n;    

		inputX.resize(numOfParticles);
		inputY.resize(numOfParticles);
		inputZ.resize(numOfParticles);
		
		thrust::copy(inputXtmp.begin(), inputXtmp.begin()+numOfParticles, inputX.begin());
		thrust::copy(inputYtmp.begin(), inputYtmp.begin()+numOfParticles, inputY.begin());
		thrust::copy(inputZtmp.begin(), inputZtmp.begin()+numOfParticles, inputZ.begin());

		return true;
	}


  // read a .csv file and load the data to inputX, inputY & inputZ
  bool readCsvFile(std::string filename, float n, int np, float rL, std::string format)
  {
    // open input file
    std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
    if (!myfile->is_open()) {   std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

    std::vector<Point> vec;
    float x, y, z;

    // scale amount for particles
    float xscal;
    if(rL==-1) xscal = 1;
    else       xscal = rL / (1.0*np);

    std::string line;
    getline(*myfile,line);

		double minX, minY, minZ;
		double maxX, maxY, maxZ;

    while(!myfile->eof())
    {
			getline(*myfile,line);

			if(line=="") continue;
			strtok((char*)line.c_str(), ",");

			int i = 0;
			while(++i<7) strtok (NULL, ",");

			x = atof(strtok(NULL, ",")) / xscal;
			y = atof(strtok(NULL, ",")) / xscal;
			z = atof(strtok(NULL, ",")) / xscal;
			
			if(vec.size()==0)
			{
				minX = x; minY = y; minZ = z;
				maxX = x; maxY = y; maxZ = z;
			}
			else
			{		
				minX = (x<minX) ? x : minX; 	maxX = (x>maxX) ? x : maxX; 
				minY = (y<minY) ? y : minY;		maxY = (y>maxY) ? y : maxY;
				minZ = (z<minZ) ? z : minZ;		maxZ = (z>maxZ) ? z : maxZ;
			}

			vec.push_back(Point(x,y,z));
    }

		// set bounds
		lBoundS = Point(minX, minY, minZ);
		uBoundS = Point(maxX, maxY, maxZ);

    numOfParticles = vec.size() / n;

		inputX.resize(numOfParticles);

		inputY.resize(numOfParticles);
		inputZ.resize(numOfParticles);

    for (int i=0; i<numOfParticles; i++)
    {
			Point p = vec.at(i);

			// sanity check
//			if (!(rL==-1 && xscal==1) && (p.x > rL || p. y > rL || p.z > rL)) {   std::cout << "rL is too small \n"; exit (-1); }

			inputX[i] = p.x;
			inputY[i] = p.y;
			inputZ[i] = p.z;
    }

    vec.clear();

    return true;
  }

	void writeHaloResults()
	{
		thrust::device_vector<int> a,b;
		
		a.resize(numOfParticles);		
		b.resize(numOfParticles);	
		thrust::copy(haloIndex.begin(), haloIndex.end(), a.begin());
		thrust::sequence(b.begin(), b.end());

		std::string filename = convertInt(numOfParticles) + "_MTree.txt";
		std::ofstream out(filename.c_str(), std::ofstream::out); 
		for(int i=0; i<numOfParticles; i++)
		{
			if(i%15==0) out << "\n";
			out << a[i] << " ";
		}
		out.close();
	}

  // return a vector with N random values in the range [min,max)
  thrust::host_vector<float> random_vector(const size_t N, float max, float min)
  {
    thrust::host_vector<float> tmp(N);
    for(size_t i = 0; i < N; i++) tmp[i] = (u01(rng)*(max-min) + min);
    return tmp;
  }

  // clear vectors & variables
  void clear()
  {
    haloIndex.clear();
    haloIndexUnique.clear();

    haloColorsR.clear();
    haloColorsG.clear();
    haloColorsB.clear();

    numOfHalos = 0;
    numOfParticles = inputX.size();

    haloIndex.resize(numOfParticles);
		thrust::copy(CountingIterator(0), CountingIterator(0)+numOfParticles, haloIndex.begin());
  }


  // set colors to halos
  void setColors()
  {
    // set color range
    float minrangeC = 0.1;
    float maxrangeC = 1;

    // for each halo, set unique colors
    u01 = thrust::uniform_real_distribution<float>(0.0f, 1.0f);
    haloColorsR = random_vector(numOfHalos, maxrangeC, minrangeC);
    haloColorsG = random_vector(numOfHalos, maxrangeC, minrangeC);
    haloColorsB = random_vector(numOfHalos, maxrangeC, minrangeC);
  }


  // get unique halo ids & numOfHalos
  void getUniqueHalos(int particleSize)
  {
    thrust::device_vector<int> haloUnique(numOfParticles);
    thrust::copy(haloIndex.begin(), haloIndex.end(), haloUnique.begin());
    thrust::sort(haloUnique.begin(), haloUnique.end()); //sort halo ids

    thrust::device_vector<int> haloSize(numOfParticles);
    thrust::fill(haloSize.begin(), haloSize.end(), 1);

    thrust::device_vector<int> a(numOfParticles);
    thrust::device_vector<int> b(numOfParticles);

    thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end; //get size of halo
    new_end = thrust::reduce_by_key(haloUnique.begin(), haloUnique.end(), haloSize.begin(), a.begin(), b.begin());

    int numUniqueHalos = thrust::get<0>(new_end) - a.begin();

    // get the number of invalid halos & their ids
    thrust::device_vector<int>::iterator new_end1 = thrust::remove_if(a.begin(), thrust::get<0>(new_end), b.begin(), validHalo(particleSize));
    int numOfInvalidHalos = new_end1 - a.begin();

    // for all particals in invalid halos, set halo id to -1
		for(int i=0; i<numOfInvalidHalos; i++)
			thrust::replace(haloIndex.begin(), haloIndex.end(), ((int)a[i]), -1);

		// get the number of valid halos & their ids
		new_end  = thrust::reduce_by_key(haloUnique.begin(), haloUnique.end(), haloSize.begin(), a.begin(), b.begin());
		new_end1 = thrust::remove_if(a.begin(), thrust::get<0>(new_end), b.begin(), invalidHalo(particleSize));

		numOfHalos = new_end1 - a.begin();

    haloIndexUnique = thrust::device_vector<int>(numOfHalos);
    thrust::copy(a.begin(), a.begin()+numOfHalos, haloIndexUnique.begin());

    haloUnique.clear(); haloSize.clear(); a.clear(); b.clear();
  }

  struct is_NotEqual
  {
    int haloId;

		__host__ __device__
    is_NotEqual(int haloId) : haloId(haloId) {}

    __host__ __device__
    bool operator()(const int x) { return haloId != x; }
  };

  //reset the halo id to -1, if a particle belongs to a invalid halo
  struct resetInvalidParticles
  {
    int *haloIndex;
    int *haloUnique;
    int  numUniqueHalos, particleSize;  

		__host__ __device__
    resetInvalidParticles(int *haloUnique, int numUniqueHalos, int particleSize, int* haloIndex) :
		    haloUnique(haloUnique), numUniqueHalos(numUniqueHalos), particleSize(particleSize), haloIndex(haloIndex) {}

    __host__ __device__
    void operator()(int i)
    {
			bool found = false;
			for(int j=0; j<numUniqueHalos; j++) if(haloUnique[j] == haloIndex[i]) { found = true; break; }

			if(!found) haloIndex[i] = -1;
    }
  };

  // check whether number of particles in this halo exceed particleSize
  struct validHalo
  {
    int particleSize;

		__host__ __device__
    validHalo(int particleSize) : particleSize(particleSize) {}

    __host__ __device__
    bool operator()(int i) { return i >= particleSize; }
  };

	struct invalidHalo
  {
    int particleSize;

		__host__ __device__
    invalidHalo(int particleSize) : particleSize(particleSize) {}

    __host__ __device__
    bool operator()(int i) { return i < particleSize; }
  };


  thrust::device_vector<int> getHalos()
	{
  	return haloIndex;
	}
};

}

#endif /* HALO_H_ */
