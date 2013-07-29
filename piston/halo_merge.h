#ifndef HALO_MERGE_H
#define HALO_MERGE_H

#include <piston/halo.h>

// When TEST is defined, output all results
//#define TEST

namespace piston
{

class halo_merge : public halo
{
public:
	float cubeLen;					// length of the cube
	double max_ll, min_ll;   // maximum & minimum linking lengths
	
	float totalTime; 				// total time taken for halo finding

	int mergetreeSize;      // total size of the global merge tree
  int numOfEdges;         // total number of edges in space

	int  side, size, ite; 	// variable needed to determine the neighborhood cubes

	unsigned int  numOfCubes;					 // total number of cubes in space
	int  cubesNonEmpty, cubesEmpty;		 // total number of nonempty & empty cubes in space
	int  cubesInX, cubesInY, cubesInZ; // number of cubes in each dimension

	int cubes, chunks; // amount of chunks & cube sizes used in computation steps, usually cubes is equal to cubesNonEmpty
	
  thrust::device_vector<int>   particleId; 						// for each particle, particle id
	thrust::device_vector<int>   particleSizeOfCubes; 	// number of particles in cubes
	thrust::device_vector<int>   particleStartOfCubes;	// stratInd of cubes  (within particleId)

  thrust::device_vector<int>   cubeId; // for each particle, cube id
  thrust::device_vector<int>   cubeMapping, cubeMappingInv; // mapping which seperates empty & nonempty cubes
  thrust::device_vector<int>   sizeOfChunks;  // size of each chunk of cubes
  thrust::device_vector<int>   startOfChunks; // start of each chunk of cubes

	thrust::device_vector<Edge>  edges;						 // edge of cubes
	thrust::device_vector<int>   edgeSizeOfCubes;  // size of edges in cubes
	thrust::device_vector<int>   edgeStartOfCubes; // start of edges in cubes
	
	thrust::device_vector<int>   tmpIntArray, tmpIntArray1;	// temperary arrays used 
	thrust::device_vector<int>   tmpNxt, tmpFree;  // stores details of free items in merge tree

	halo_merge(float min_linkLength, float max_linkLength, std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1) : halo(0,0) //: halo(filename, format, n, np, rL)
	{
		u01 = thrust::uniform_real_distribution<float>(0.0f, 1.0f);

		this->n    		 = n;

		// scale amount for particles
	  if(rL==-1) xscal = 1;
	  else       xscal = rL / (1.0*np);

    if(!readHaloFile(filename, format))
    {	
//			generateUniformData();		// generate uniformly spaced points		
			generateNonUniformData(); // generate nearby points to real data

			std::cout << "Test data loaded \n";
    }

    haloIndex.resize(numOfParticles);
    thrust::copy(CountingIterator(0), CountingIterator(0)+numOfParticles, haloIndex.begin());

    std::cout << "numOfParticles : " << numOfParticles << " \n";

		if(numOfParticles!=0)
		{
			struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3;

			//---- init stuff

		  // Unnormalize linkLengths so that it will work with box size distances
			min_ll  = min_linkLength*xscal; // get min_linkinglength
			max_ll  = max_linkLength*xscal; // get max_linkinglength
			cubeLen = min_ll / std::sqrt(3); // min_ll*min_ll = 3*cubeLen*cubeLen

			if(cubeLen <= 0) { std::cout << "--ERROR : plase specify a valid cubeLen... current cubeLen is " << cubeLen << std::endl; return; }

			initDetails();

			std::cout << "lBoundS " << lBoundS.x << " " << lBoundS.y << " " << lBoundS.z << std::endl;
			std::cout << "uBoundS " << uBoundS.x << " " << uBoundS.y << " " << uBoundS.z << std::endl;

			//---- divide space into cubes
			gettimeofday(&begin, 0);
			divideIntoCubes();
			gettimeofday(&mid1, 0);

			std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ") ... cubeLen " << cubeLen << std::endl;

			#ifdef TEST
				outputCubeDetails("init cube details"); // output cube details
			#endif

			//------- METHOD :
	    // parallel for each cube, create the local merge tree & get the set of edges.
			// globally combine the cubes, two cubes at a time by considering the edges

			gettimeofday(&mid2, 0);
			localStep();
			gettimeofday(&mid3, 0);

			std::cout << "-- localStep done" << std::endl;

			gettimeofday(&mid4, 0);
			globalStep();
			gettimeofday(&end, 0);

			std::cout << "-- globalStep done" << std::endl;

			checkValidMergeTree(); 
			getSizeOfMergeTree(); 
			clearSuperParents();

			particleId.clear();	  particleSizeOfCubes.clear();  particleStartOfCubes.clear();
			edges.clear();			  edgeSizeOfCubes.clear();		  edgeStartOfCubes.clear();
			cubeId.clear();			  cubeMapping.clear();				  cubeMappingInv.clear();
			tmpIntArray.clear();  tmpNxt.clear();							  tmpFree.clear();
			sizeOfChunks.clear(); startOfChunks.clear();			

			std::cout << std::endl;
			timersub(&mid1, &begin, &diff1);
			float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
			std::cout << "Time elapsed: " << seconds1 << " s for dividing space into cubes"<< std::endl << std::flush;
			timersub(&mid3, &mid2, &diff2);
			float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
			std::cout << "Time elapsed: " << seconds2 << " s for localStep - finding inter-cube edges"<< std::endl << std::flush;
			timersub(&end, &mid4, &diff3);
			float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
			std::cout << "Time elapsed: " << seconds3 << " s for globalStep - adjusting the merge trees"<< std::endl << std::flush;
			totalTime = seconds1 + seconds2 + seconds3;
			std::cout << "Total time elapsed: " << totalTime << " s for constructing the global merge tree" << std::endl << std::endl;
		}
	}

	void operator()(float linkLength, int  particleSize)
	{
		clear();

		// Unnormalize linkLength so that it will work with box size distances
		linkLength   = linkLength*xscal;
		particleSize = particleSize;

		// if no valid particles, return
		if(numOfParticles==0) return;

		struct timeval begin, end, diff;

		gettimeofday(&begin, 0);
		findHalos(linkLength, particleSize); // find halos
		gettimeofday(&end, 0);

		timersub(&end, &begin, &diff);
		float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
		totalTime +=seconds;

		std::cout << "Total time elapsed: " << seconds << " s for finding halos at linking length " << linkLength/xscal << " and has particle size >= " << particleSize << std::endl << std::endl;

		getHaloDetails();     // get the unique halo ids & set numOfHalos
		getHaloParticles();   // get the halo particles & set numOfHaloParticles
		setColors();          // set colors to halos
		writeHaloResults();	  // write halo results

		std::cout << "Number of Particles   : " << numOfParticles << std::endl;
		std::cout << "Number of Halos found : " << numOfHalos << std::endl;
		std::cout << "Merge tree size : " << mergetreeSize << std::endl;
    std::cout << "Min_ll  : " << min_ll/xscal  << std::endl;
    std::cout << "Max_ll  : " << max_ll/xscal << std::endl << std::endl;
		std::cout << "-----------------------------" << std::endl << std::endl;
	}

  // read input file - currently can read a .cosmo file or a .csv file
  // .csv file - when you have a big data file and you want a piece of it, load it in VTK and slice it and save it as .csv file, within this function it will rewrite the date to .cosmo format
  bool readHaloFile(std::string filename, std::string format)
  {
    // check filename
    if(filename == "") { std::cout << "no input file specified \n"; return false; }
		if(format == "hcosmo") return readHCosmoFile(filename, format);
		if(format == "cosmo")  return readCosmoFile(filename, format);
    if(format == "csv")    return readCsvFile(filename, format);   

    return false;
  }

	bool readCosmoFile(std::string filename, std::string format)
	{
		// open .cosmo file
		std::ifstream *myfile = new std::ifstream((filename+"."+format).c_str(), std::ios::in);
		if (!myfile->is_open()) { std::cout << "File: " << filename << "." << format << " cannot be opened \n"; return false; }

		int nfloat = 7, nint = 1;
		
		// compute the number of particles
		myfile->seekg(0L, std::ios::end);
		numOfParticles = myfile->tellg() / 32; // get particle size in file

		// get the fraction wanted
		numOfParticles = numOfParticles / n; 

		// resize nodes
		nodes.resize(numOfParticles);
		index.resize(numOfParticles);

		// rewind file to beginning for particle reads
		myfile->seekg(0L, std::ios::beg);

		// declare temporary read buffers
		float fBlock[nfloat];
		int   iBlock[nint];

		float minX, minY, minZ, maxX, maxY, maxZ;
		for (int i=0; i<numOfParticles; i++)
		{
			// Set file pointer to the requested particle
			myfile->read(reinterpret_cast<char*>(fBlock), nfloat * sizeof(float));

			if (myfile->gcount() != (int)(nfloat * sizeof(float))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			myfile->read(reinterpret_cast<char*>(iBlock), nint * sizeof(int));
			if (myfile->gcount() != (int)(nint * sizeof(int))) {
				std::cout << "Premature end-of-file" << std::endl;
				return false;
			}

			Node n = Node();
			n.nodeId = i;	n.haloId = i;	n.count  = 1;
			n.pos = Point(fBlock[0], fBlock[2], fBlock[4]);
			n.vel = Point(fBlock[1], fBlock[3], fBlock[5]);
			n.mass = fBlock[6];
			nodes[i] = n;

			index[i] = iBlock[0];

			if(i==0)
			{	
				minX = n.pos.x; maxX = n.pos.x;
				minY = n.pos.y; maxY = n.pos.y;
				minZ = n.pos.z; maxZ = n.pos.z;	
			}
			else
			{
				minX = std::min(minX, n.pos.x);	maxX = std::max(maxX, n.pos.x);
				minY = std::min(minY, n.pos.y);	maxY = std::max(maxY, n.pos.y);
				minZ = std::min(minZ, n.pos.z);	maxZ = std::max(maxZ, n.pos.z);
			}
		}

		// get bounds of the space
		lBoundS = Point(minX, minY, minZ);
		uBoundS = Point(maxX, maxY, maxZ);

		return true;
	}

	// find halo ids 
	void findHalos(float linkLength, int particleSize)
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setHaloId(thrust::raw_pointer_cast(&*nodes.begin()),
						 		  thrust::raw_pointer_cast(&*haloIndex.begin()),
						     	linkLength, particleSize));
	}

	// for a given node set its halo id, for particles in filtered halos set id to -1
	struct setHaloId : public thrust::unary_function<int, void>
	{
		Node  *nodes;
		int   *haloIndex;

		int    particleSize;
		float  linkLength;

		__host__ __device__
		setHaloId(Node *nodes, int *haloIndex, float linkLength, int particleSize) :
			nodes(nodes), haloIndex(haloIndex), linkLength(linkLength), particleSize(particleSize) {}

		__host__ __device__
		void operator()(int i)
		{			
      Node *n = &nodes[i];

      if(n->parentSuper!=NULL && n->parentSuper->value<=linkLength)
        n = n->parentSuper;

			while(n->parent!=NULL && n->parent->value<=linkLength)
			  n = n->parent;

			nodes[i].parentSuper = n;

			haloIndex[i] = (n->count >= particleSize) ? n->haloId : -1;
		}
	};

	// get the unique halo indexes & number of halos
	void getHaloDetails()
	{
		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
		
		// find unique halo ids & one particle id which belongs to that halo
		haloIndexUnique.resize(numOfParticles);		
		thrust::copy(haloIndex.begin(), haloIndex.end(), haloIndexUnique.begin());
		thrust::device_vector<int> tmp(numOfParticles);
		thrust::sequence(tmp.begin(), tmp.end());	

		thrust::stable_sort_by_key(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfParticles, tmp.begin(),  thrust::greater<int>());
	  new_end = thrust::unique_by_key(haloIndexUnique.begin(), haloIndexUnique.begin()+numOfParticles, tmp.begin());
	  
	  numOfHalos = thrust::get<0>(new_end) - haloIndexUnique.begin();
		if(haloIndexUnique[numOfHalos-1]==-1) numOfHalos--;

		thrust::reverse(tmp.begin(), tmp.begin()+numOfHalos);

		// get the halo stats
		haloCount.resize(numOfHalos);
		haloX.resize(numOfHalos);
		haloY.resize(numOfHalos);
		haloZ.resize(numOfHalos);
		haloVX.resize(numOfHalos);
		haloVY.resize(numOfHalos);
		haloVZ.resize(numOfHalos);

		thrust:: for_each(CountingIterator(0), CountingIterator(0)+numOfHalos,
				setHaloStats(thrust::raw_pointer_cast(&*nodes.begin()),
										 thrust::raw_pointer_cast(&*tmp.begin()),
										 thrust::raw_pointer_cast(&*haloCount.begin()),
										 thrust::raw_pointer_cast(&*haloX.begin()),
										 thrust::raw_pointer_cast(&*haloY.begin()),
										 thrust::raw_pointer_cast(&*haloZ.begin()),
										 thrust::raw_pointer_cast(&*haloVX.begin()),
										 thrust::raw_pointer_cast(&*haloVY.begin()),
										 thrust::raw_pointer_cast(&*haloVZ.begin()),
										 linkLength, particleSize));
/*
		std::cout << "haloCount		"; thrust::copy(haloCount.begin(), haloCount.begin()+numOfHalos, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
*/
	}

	// for each halo, get its stats
	struct setHaloStats : public thrust::unary_function<int, void>
	{
		Node  *nodes;
		
		int    particleSize;
		float  linkLength;

		int *particleId;
		int *haloCount;
		float *haloX, *haloY, *haloZ;
		float *haloVX, *haloVY, *haloVZ;

		__host__ __device__
		setHaloStats(Node *nodes, int *particleId, int *haloCount,
			float *haloX, float *haloY, float *haloZ,
			float *haloVX, float *haloVY, float *haloVZ,
			float linkLength, int particleSize) :
			nodes(nodes), particleId(particleId), haloCount(haloCount),
			haloX(haloX), haloY(haloY), haloZ(haloZ),
			haloVX(haloVX), haloVY(haloVY), haloVZ(haloVZ),
			linkLength(linkLength), particleSize(particleSize) {}

		__host__ __device__
		void operator()(int i)
		{			
      Node *n = (&nodes[particleId[i]])->parentSuper;

			haloCount[i] = n->count;
			haloX[i] = (float)(n->pos.x/n->count);	haloVX[i] = (float)(n->pos.x/n->count);
			haloY[i] = (float)(n->pos.y/n->count);	haloVY[i] = (float)(n->pos.y/n->count);
			haloZ[i] = (float)(n->pos.z/n->count);	haloVZ[i] = (float)(n->pos.z/n->count);		
		}
	};

	// get particles of valid halos & get number of halo particles 
	void getHaloParticles()
	{
	  thrust::device_vector<int>::iterator new_end;

	  tmpIntArray.resize(numOfParticles);
		thrust::sequence(tmpIntArray.begin(), tmpIntArray.begin()+numOfParticles);

	  new_end = thrust::remove_if(tmpIntArray.begin(), tmpIntArray.begin()+numOfParticles,
				invalidHalo(thrust::raw_pointer_cast(&*haloIndex.begin())));

	  numOfHaloParticles = new_end - tmpIntArray.begin();

		haloIndex_f.resize(numOfHaloParticles);
	  inputX_f.resize(numOfHaloParticles);
    inputY_f.resize(numOfHaloParticles);
    inputZ_f.resize(numOfHaloParticles);

		thrust::gather(tmpIntArray.begin(), tmpIntArray.begin()+numOfHaloParticles, haloIndex.begin(), haloIndex_f.begin());
		
		thrust:: for_each(CountingIterator(0), CountingIterator(0)+numOfHaloParticles,
			getHaloParticlePositions(thrust::raw_pointer_cast(&*nodes.begin()),
															 thrust::raw_pointer_cast(&*tmpIntArray.begin()),
															 thrust::raw_pointer_cast(&*inputX_f.begin()),
															 thrust::raw_pointer_cast(&*inputY_f.begin()),
															 thrust::raw_pointer_cast(&*inputZ_f.begin())));
	}

	// given a haloIndex of a particle, check whether this particle DOES NOT belong to a halo
	struct invalidHalo : public thrust::unary_function<int, bool>
	{
		int  *haloIndex;

		__host__ __device__
		invalidHalo(int *haloIndex) : haloIndex(haloIndex) {}

		__host__ __device__
		bool operator()(int i)
		{			
      return (haloIndex[i]==-1);
		}
	};

	// for each particle in a halo, get its positions
	struct getHaloParticlePositions : public thrust::unary_function<int, void>
	{
		Node  *nodes;
		
		int *particleId;

		float *inputX_f, *inputY_f, *inputZ_f;

		__host__ __device__
		getHaloParticlePositions(Node *nodes, int *particleId, 
			float *inputX_f, float *inputY_f, float *inputZ_f) :
			nodes(nodes), particleId(particleId),
			inputX_f(inputX_f), inputY_f(inputY_f), inputZ_f(inputZ_f) {}

		__host__ __device__
		void operator()(int i)
		{			
      Node *n = &nodes[particleId[i]];

			inputX_f[i] = n->pos.x;
			inputY_f[i] = n->pos.y;
			inputZ_f[i] = n->pos.z;		
		}
	};

	// check whether the merge tree is valid or not
	void checkValidMergeTree()
	{
		thrust::host_vector<Node>  nodes_h;
		nodes_h.resize(numOfParticles);

		thrust::copy(nodes.begin(), nodes.end(), nodes_h.begin());
	
		bool invalid = false;
		for(int i=0; i<numOfParticles; i++)
		{
			Node *n = &nodes_h[i];

			int count = 0;
			while(n && n->value <= min_ll)
			{ n = n->parent;	count++; }

			// if a node has more than one parent node with its value <= min_ll, the tree is invalid
			if(count > 2)
			{ invalid = true; 	std::cout << i << " " << count << std::endl; break; }
		}	
		std::cout << std::endl;	

		if(invalid) std::cout << "-- ERROR: invalid merge tree " << std::endl;
		else std::cout << "-- valid merge tree " << std::endl;
	}

	// get the size of the merge tree
	void getSizeOfMergeTree()
	{
		thrust::host_vector<Node>  nodes_h;
		nodes_h.resize(numOfParticles);

		thrust::copy(nodes.begin(), nodes.end(), nodes_h.begin());

		mergetreeSize = 0;
		for(int i=0; i<numOfParticles; i++)
		{
			Node *n = &nodes_h[i];

			while(n->parent!=NULL)
			{
				n = (n->parent);
				if(n->nodeId != -2) mergetreeSize++;
				n->nodeId = -2;								
			}
		}

		mergetreeSize += numOfParticles;
	}

	// clear super parents of all nodes
	void clearSuperParents()
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				clearParentSuper(thrust::raw_pointer_cast(&*nodes.begin())));
	}

	// for a given node, set its super parent to null
	struct clearParentSuper : public thrust::unary_function<int, void>
	{
		Node  *nodes;

		__host__ __device__
		clearParentSuper(Node *nodes) : nodes(nodes) {}

		__host__ __device__
		void operator()(int i)
		{			
      nodes[i].parentSuper = NULL;
		}
	};

	

	//------- init stuff
	void initDetails()
	{
		initParticleIds();	// set particle ids
		setNumberOfCubes();	// get total number of cubes
	}

  // set initial particle ids
	void initParticleIds()
	{
		particleId.resize(numOfParticles);
		thrust::sequence(particleId.begin(), particleId.end());
	}

	// get total number of cubes
	void setNumberOfCubes()
	{
		cubesInX = (std::ceil((uBoundS.x - lBoundS.x)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.x - lBoundS.x)/cubeLen);
		cubesInY = (std::ceil((uBoundS.y - lBoundS.y)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.y - lBoundS.y)/cubeLen);
		cubesInZ = (std::ceil((uBoundS.z - lBoundS.z)/cubeLen) == 0) ? 1 : std::ceil((uBoundS.z - lBoundS.z)/cubeLen);

		numOfCubes = cubesInX*cubesInY*cubesInZ; // set number of cubes
	}



	//------- divide space into cubes
	void divideIntoCubes()
	{
	  struct timeval begin, mid1, mid2, end, diff0, diff1, diff2;
	  gettimeofday(&begin, 0);
		setCubeIds();		      		// for each particle, set cube id
	  gettimeofday(&mid1, 0);
		sortParticlesByCubeID();  // sort Particles by cube Id
	  gettimeofday(&mid2, 0);
		getSizeAndStartOfCubes(); // for each cube, count its particles
	  gettimeofday(&end, 0);
    std::cout << std::endl;
	  std::cout << "'divideIntoCubes' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff0);
    float seconds0 = diff0.tv_sec + 1.0E-6*diff0.tv_usec;
    std::cout << "Time elapsed0: " << seconds0 << " s for setCubeIds"<< std::endl << std::flush;
    timersub(&mid2, &mid1, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << " s for sortParticlesByCubeID"<< std::endl << std::flush;
    timersub(&end, &mid2, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed2: " << seconds2 << " s for getSizeAndStartOfCubes"<< std::endl << std::flush;
	}

	// set cube ids of particles
	void setCubeIds()
	{
	  cubeId.resize(numOfParticles);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setCubeIdOfParticle(thrust::raw_pointer_cast(&*nodes.begin()),
														thrust::raw_pointer_cast(&*cubeId.begin()),
														cubeLen, lBoundS, cubesInX, cubesInY, cubesInZ));
	}

	// for a given particle, set its cube id
	struct setCubeIdOfParticle : public thrust::unary_function<int, void>
	{
		float  cubeLen;
		Point  lBoundS;
		int    cubesInX, cubesInY, cubesInZ;

		int   *cubeId;
		Node  *nodes;

		__host__ __device__
		setCubeIdOfParticle(Node  *nodes, int *cubeId, float cubeLen, Point lBoundS,
			int cubesInX, int cubesInY, int cubesInZ) :
			nodes(nodes), cubeId(cubeId), cubeLen(cubeLen), lBoundS(lBoundS),
			cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ){}

		__host__ __device__
		void operator()(int i)
		{
			Node n = nodes[i];
	
			// get x,y,z coordinates for the cube
			int z = (((n.pos.z-lBoundS.z) / cubeLen)>=cubesInZ) ? cubesInZ-1 : (n.pos.z-lBoundS.z) / cubeLen;
			int y = (((n.pos.y-lBoundS.y) / cubeLen)>=cubesInY) ? cubesInY-1 : (n.pos.y-lBoundS.y) / cubeLen;
			int x = (((n.pos.x-lBoundS.x) / cubeLen)>=cubesInX) ? cubesInX-1 : (n.pos.x-lBoundS.x) / cubeLen;
			
			cubeId[i] = (z*(cubesInX*cubesInY) + y*cubesInX + x); // get cube id
		}
	};

	// sort particles by cube id
	void sortParticlesByCubeID()
	{
    thrust::stable_sort_by_key(cubeId.begin(), cubeId.end(), particleId.begin());
	}
		
	// sort cube id by particles
	void sortCubeIDByParticles()
	{
    thrust::stable_sort_by_key(particleId.begin(), particleId.end(), cubeId.begin());
	}

	// for each cube, get the size & start of cube particles (in particleId array)
	void getSizeAndStartOfCubes()
	{
		int num = (numOfParticles<numOfCubes) ? numOfParticles : numOfCubes;

		cubeMapping.resize(num);
		particleSizeOfCubes.resize(num);

		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
		new_end = thrust::reduce_by_key(cubeId.begin(), cubeId.end(), ConstantIterator(1), cubeMapping.begin(), particleSizeOfCubes.begin());

		cubesNonEmpty = thrust::get<0>(new_end) - cubeMapping.begin();
		cubesEmpty    = numOfCubes - cubesNonEmpty;	

		cubes = cubesNonEmpty; // get the cubes which should be considered

		// get the mapping for nonempty cubes
		cubeMappingInv.resize(numOfCubes);
		thrust::fill(cubeMappingInv.begin(), cubeMappingInv.end(), -1);
		thrust::scatter(CountingIterator(0), CountingIterator(0)+cubesNonEmpty, cubeMapping.begin(), cubeMappingInv.begin());

		// get the size & start details for only non empty cubes
		particleStartOfCubes.resize(cubesNonEmpty);
		thrust::exclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubesNonEmpty, particleStartOfCubes.begin());
	}
	


	//------- output results

	// print cube details from device vectors
	void outputCubeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << std::endl << "-- Outputs------------" << std::endl << std::endl;

		std::cout << "-- Dim    (" << lBoundS.x << "," << lBoundS.y << "," << lBoundS.z << "), (";
		std::cout << uBoundS.x << "," << uBoundS.y << "," << uBoundS.z << ")" << std::endl;
		std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;

		std::cout << std::endl << "----------------------" << std::endl << std::endl;

		std::cout << "particleId 	"; thrust::copy(particleId.begin(), particleId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "cubeID		"; thrust::copy(cubeId.begin(), cubeId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "startOfCube	"; thrust::copy(particleStartOfCubes.begin(), particleStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "----------------------" << std::endl << std::endl;
	}

	// print edge details from device vectors
	void outputEdgeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;
		std::cout << "numOfEdges			 " << numOfEdges << std::endl << std::endl;
		std::cout << "edgeSizeOfCubes	 "; thrust::copy(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "edgeStartOfCubes "; thrust::copy(edgeStartOfCubes.begin(), edgeStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << std::endl;

		for(int i=0; i<numOfCubes; i++)
		{
			for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)	
			{
				std::cout << "---- " << ((Edge)edges[j]).srcId << "," << ((Edge)edges[j]).desId << "," << ((Edge)edges[j]).weight <<  ")" << std::endl;
			}
		}
		std::cout << std::endl << "----------------------" << std::endl << std::endl;
	}
	
	// print merge tree details from device vectors
	void outputMergeTreeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;
		std::cout << "MergeTreeNodes " << std::endl;
		for(int i=0; i<numOfParticles; i++)
		{
			Node n = ((Node)nodes[i]);
			int k  = (n.parent==NULL)?0:1;
			std::cout << "(" << n.pos.x << "," << n.pos.y << "," << n.pos.z << " : " << n.value << "," << n.nodeId << "," << n.haloId << "," << k << ")";

			while(n.parent!=NULL)
			{
				n = *(n.parent);
				k = (n.parent==NULL)?0:1;
				std::cout << "(" << n.value << "," << n.nodeId << "," << n.haloId << "," << k << ")";
			}
			
			n = ((Node)nodes[i]);
			if(n.parentSuper!=NULL)
			{
				n = *(n.parentSuper);
				std::cout << " - parentSuper (" << n.value << "," << n.nodeId << "," << n.haloId << ")";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl << "----------------------" << std::endl << std::endl;
	}



	//------- METHOD : parallel for each cube, get the set of edges & create the submerge tree. Globally combine them

	//---------------- METHOD - Local functions

	// locally, get intra-cube edges for each cube & create the local merge trees
	void localStep()
	{
		// resize vectors necessary for merge tree construction
		tmpNxt.resize(cubes);
		tmpFree.resize(numOfParticles);
		nodes.resize(numOfParticles);
		nodesTmp1.resize(numOfParticles);

    struct timeval begin, mid1, mid2, mid3, mid4, mid5, end, diff1, diff2, diff3, diff4, diff5, diff6;
    gettimeofday(&begin, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				initNodes(thrust::raw_pointer_cast(&*nodes.begin()),
									thrust::raw_pointer_cast(&*nodesTmp1.begin()),
									thrust::raw_pointer_cast(&*tmpFree.begin()),
									numOfParticles));
    gettimeofday(&mid1, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				initNxt(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
								thrust::raw_pointer_cast(&*tmpNxt.begin()),
								thrust::raw_pointer_cast(&*tmpFree.begin()),
								numOfCubes, numOfParticles));
    gettimeofday(&mid2, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				createSubMergeTree(thrust::raw_pointer_cast(&*nodes.begin()),
													 thrust::raw_pointer_cast(&*nodesTmp1.begin()),
													 thrust::raw_pointer_cast(&*particleId.begin()),
													 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
												 	 thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),								
													 thrust::raw_pointer_cast(&*tmpNxt.begin()),
													 thrust::raw_pointer_cast(&*tmpFree.begin()),
													 min_ll));
		gettimeofday(&mid3, 0);
		initArrays();  				    // init arrays needed for storing edges
		gettimeofday(&mid4, 0);
		getEdgesPerCube(); 				// for each cube, get the set of edges
		gettimeofday(&mid5, 0);
		sortCubeIDByParticles();	// sort cube ids by particle id
    gettimeofday(&end, 0);

		#ifdef TEST
			outputMergeTreeDetails("The local merge trees.."); // output merge tree details
			outputEdgeDetails("Edges to be considered in the global step.."); // output edge details
		#endif

    std::cout << std::endl;
	  std::cout << "'localStep' Time division: " << std::endl << std::flush;
    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed0: " << seconds1 << " s for initNodes " << std::endl << std::flush;
		timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed1: " << seconds2 << " s for initNxt " << std::endl << std::flush;
		timersub(&mid3, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed2: " << seconds3 << " s for createSubMergeTree " << std::endl << std::flush;   
		timersub(&mid4, &mid3, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed3: " << seconds4 << " s for initEdgeArrays " << std::endl << std::flush;
		timersub(&mid5, &mid4, &diff5);
    float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
    std::cout << "Time elapsed4: " << seconds5 << " s for getEdgesPerCube " << std::endl << std::flush;
		timersub(&end, &mid5, &diff6);
    float seconds6 = diff6.tv_sec + 1.0E-6*diff6.tv_usec;
    std::cout << "Time elapsed5: " << seconds6 << " s for sortCubeIDByParticles " << std::endl << std::flush;
	}

	// for each particle, init the nodes array with node id, halo id, count & set the tmpFree array with its initial nxt free id
  struct initNodes : public thrust::unary_function<int, void>
  {
		Node *nodes, *nodesTmp1;
		int  *tmpFree;

		int numOfParticles;

		__host__ __device__
		initNodes(Node *nodes, Node *nodesTmp1, int *tmpFree, int numOfParticles) : 
			nodes(nodes), nodesTmp1(nodesTmp1), tmpFree(tmpFree), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {
			nodesTmp1[i].nodeId = i+numOfParticles;
			tmpFree[i] = i+1;
    }
 	};

  // finalize the init of tmpFree & tmpNxt arrays
  struct initNxt : public thrust::unary_function<int, void>
  {
    int  *particleSizeOfCubes, *particleStartOfCubes;
		int  *tmpFree, *tmpNxt;

		int   numOfCubes, numOfParticles;

		__host__ __device__
		initNxt(int *particleSizeOfCubes, int *particleStartOfCubes, 
			int *tmpNxt, int *tmpFree,
			int numOfCubes, int numOfParticles) : 
			particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes),			
			tmpNxt(tmpNxt), tmpFree(tmpFree),
			numOfCubes(numOfCubes), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {
			int start = particleStartOfCubes[i];
			int size  = particleSizeOfCubes[i];

		  tmpNxt[i] = (size>0) ? start : -1;

			tmpFree[start+size-1] = -1;
    }
 	};

	// create the submerge tree for each cube
	struct createSubMergeTree : public thrust::unary_function<int, void>
  {
		Node *nodes, *nodesTmp1;

		int *particleSizeOfCubes, *particleStartOfCubes, *particleId;
		int *tmpNxt, *tmpFree;

	  float min_ll;

		__host__ __device__
		createSubMergeTree(Node *nodes, Node *nodesTmp1, int *particleId,
			int *particleSizeOfCubes, int *particleStartOfCubes, 
			int *tmpNxt, int *tmpFree, float min_ll) :
			nodes(nodes), nodesTmp1(nodesTmp1), particleId(particleId),
			particleSizeOfCubes(particleSizeOfCubes), particleStartOfCubes(particleStartOfCubes), 
			tmpNxt(tmpNxt), tmpFree(tmpFree), min_ll(min_ll) {}

    __host__ __device__
    void operator()(int i)
    {
			if(particleSizeOfCubes[i]<=1) return;
			
			// get the next free node & set it as the parent
			Node *n = &nodesTmp1[tmpNxt[i]];
			int tmpVal = tmpFree[tmpNxt[i]];
			tmpFree[tmpNxt[i]] = -2;
			tmpNxt[i] = tmpVal;

			float x=0, y=0, z=0;
			float vx=0, vy=0, vz=0;

			int minValue = -1;
			for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
			{
				Node *tmp = &nodes[particleId[j]];

				tmp->parent = n;
				
				if(!n->childS) {	n->childS = tmp;	n->childE = tmp; }
				else {	 n->childE->sibling = tmp;	n->childE = tmp; }

				minValue = (minValue==-1) ? tmp->haloId : (minValue<tmp->haloId ? minValue : tmp->haloId);

			  x += tmp->pos.x;	vx += tmp->vel.x;
				y += tmp->pos.y;	vy += tmp->vel.y;
				z += tmp->pos.z;	vz += tmp->vel.z;
			}

			n->value  = min_ll;
      n->haloId = minValue;
      n->count += particleSizeOfCubes[i];
			n->pos = Point(x,y,z);
			n->vel = Point(vx,vy,vz);
    }
 	};

  // for each cube, init arrays needed for storing edges
	void initArrays()
	{
		// for each vube, get the details of how many neighbors should be checked 
		side = (1 + std::ceil(max_ll/cubeLen)*2);
		size = side*side*side;
		ite = (size-1)/2;

		std::cout << std::endl << "side " << side << " cubeSize " << size << " ite " << ite << std::endl << std::endl;
		std::cout << cubesEmpty << " of " << numOfCubes << " cubes are empty. (" << (((double)cubesEmpty*100)/(double)numOfCubes) << "%) ... non empty cubes " << cubesNonEmpty << std::endl;

		edgeSizeOfCubes.resize(cubes);
		edgeStartOfCubes.resize(cubes);

		// for each cube, get neighbor details
		tmpIntArray.resize(cubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				getNeighborDetails(thrust::raw_pointer_cast(&*cubeMapping.begin()),
													 thrust::raw_pointer_cast(&*cubeMappingInv.begin()),
													 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
													 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
												   thrust::raw_pointer_cast(&*tmpIntArray.begin()),
												   ite, side, cubesInX, cubesInY, cubesInZ));
		setChunks(); // group cubes in to chunks
		tmpIntArray.clear();
	
		// for each cube, set the space required for storing edges		
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes, edgeStartOfCubes.begin());
		numOfEdges = edgeStartOfCubes[cubes-1] + edgeSizeOfCubes[cubes-1]; // size of edges array 

		std::cout << std::endl << "numOfEdges before " << numOfEdges << std::endl;

		// init edge arrays
		edges.resize(numOfEdges);
	}

	//for each cube, sum the number of particles in neighbor cubes & get the sum of non empty neighbor cubes
	struct getNeighborDetails : public thrust::unary_function<int, void>
  {
		int *cubeMapping, *cubeMappingInv;
		int *particleSizeOfCubes;
		int *tmpIntArray, *edgeSizeOfCubes;

		int  ite, side;
		int  cubesInX, cubesInY, cubesInZ;

		__host__ __device__
		getNeighborDetails(int *cubeMapping, int *cubeMappingInv, 
				int *particleSizeOfCubes, int *edgeSizeOfCubes, int *tmpIntArray, 
				int ite, int side, int cubesInX, int cubesInY, int cubesInZ) : 
				cubeMapping(cubeMapping), cubeMappingInv(cubeMappingInv), 
				particleSizeOfCubes(particleSizeOfCubes), edgeSizeOfCubes(edgeSizeOfCubes), tmpIntArray(tmpIntArray), 
				ite(ite), side(side), cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ) {}

    __host__ __device__
    void operator()(int i)
    {
			int i_mapped = cubeMapping[i];

			// get x,y,z coordinates for the cube
			int tmp = i_mapped % (cubesInX*cubesInY);
			int z = i_mapped / (cubesInX*cubesInY);
			int y = tmp / cubesInX;
			int x = tmp % cubesInX;

			int len = (side-1)/2;

			for(int num=0; num<ite; num++)
			{
				int tmp1 = num % (side*side);
				int z1 = num / (side*side);
				int y1 = tmp1 / side;
				int x1 = tmp1 % side;		

				// get x,y,z coordinates for the current cube 
				int currentX = x - len + x1;
				int currentY = y - len + y1;
				int currentZ = z - len + z1;

				int cube_mapped = -1, cube = -1;
				if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
				{
					cube_mapped = (currentZ*(cubesInY*cubesInX) + currentY*cubesInX + currentX);
					cube = cubeMappingInv[cube_mapped];
				}

				if(cube_mapped==-1 || cube==-1 || particleSizeOfCubes[i]==0 || particleSizeOfCubes[cube]==0) continue;

				edgeSizeOfCubes[i]++;  //sum the non empty neighbor cubes
				tmpIntArray[i]	+= particleSizeOfCubes[cube]; // sum the number of particles in neighbor cubes
			}

			tmpIntArray[i] *= particleSizeOfCubes[i]; // multiply by particles in this cube
    }
 	};

//------------ TODO : Do accurate chunking, Look at whether I am doing the right thing in chunking
	// group the set of cubes to chunks of same computation sizes (for load balance)
	void setChunks()
	{
		thrust::device_vector<int>::iterator maxSize = thrust::max_element(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes);

		startOfChunks.resize(cubes);
		sizeOfChunks.resize(cubes);

		thrust::fill(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
		thrust::inclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+cubes, sizeOfChunks.begin());
/*
		std::cout << "particleSizeOfCubes	 "; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "inclusive_scan	 "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+100, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
*/
		thrust::copy_if(CountingIterator(0), CountingIterator(0)+cubes, startOfChunks.begin(),
				isStartOfChunks(thrust::raw_pointer_cast(&*sizeOfChunks.begin()), *maxSize));	
		chunks = cubes - thrust::count(startOfChunks.begin(), startOfChunks.begin()+cubes, -1);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
				setSizeOfChunks(thrust::raw_pointer_cast(&*startOfChunks.begin()),
												thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
												chunks, cubes));

		std::cout << "maxSize " << *maxSize << " chunks " << chunks << std::endl;
/*
		std::cout << "startOfChunks	 "; thrust::copy(startOfChunks.begin(), startOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "sizeOfChunks	 "; thrust::copy(sizeOfChunks.begin(), sizeOfChunks.begin()+chunks, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;

		std::cout << "amountOfChunks	 ";
		for(int i=0; i<chunks; i++)
		{
			int count = 0;
			for(int j=startOfChunks[i]; j<startOfChunks[i]+sizeOfChunks[i]; j++)
				count += particleSizeOfCubes[j];
			std::cout << count << " ";
		}
		std::cout << std::endl << std::endl;
*/
	}

	// check whether this cube is the start of a chunk
  struct isStartOfChunks : public thrust::unary_function<int, void>
  {
		int max;	
		int *sizeOfChunks;

		__host__ __device__
		isStartOfChunks(int *sizeOfChunks, int max) : 
			sizeOfChunks(sizeOfChunks), max(max) {}

    __host__ __device__
    bool operator()(int i)
    {
			if(i==0) return true;
	
			int a = sizeOfChunks[i] / max;	int b = sizeOfChunks[i-1] / max;	
			int d = sizeOfChunks[i] % max;	int e = sizeOfChunks[i-1] % max;

			if((a!=b && d==0 && e==0) ||(a!=b && d!=0) || (a==b && e==0)) return true;

			return false;
    }
 	};

	// set the size of cube
  struct setSizeOfChunks : public thrust::unary_function<int, void>
  {
		int chunks, cubes;	
		int *startOfChunks, *sizeOfChunks;

		__host__ __device__
		setSizeOfChunks(int *startOfChunks, int *sizeOfChunks, int chunks, int cubes) : 
			startOfChunks(startOfChunks), sizeOfChunks(sizeOfChunks), chunks(chunks), cubes(cubes) {}

    __host__ __device__
    void operator()(int i)
    {
			if(i==chunks-1) sizeOfChunks[i] = cubes - startOfChunks[i];
			else 						sizeOfChunks[i] = startOfChunks[i+1] - startOfChunks[i];
    }
 	};

	// for each cube, get the set of edges by running them in chunks of cubes
	void getEdgesPerCube()
	{	
		thrust::fill(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes,0);		
		
		thrust::for_each(CountingIterator(0), CountingIterator(0)+chunks,
				getEdges(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
								 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*startOfChunks.begin()),
								 thrust::raw_pointer_cast(&*sizeOfChunks.begin()),
								 thrust::raw_pointer_cast(&*cubeMapping.begin()),
								 thrust::raw_pointer_cast(&*cubeMappingInv.begin()),
								 thrust::raw_pointer_cast(&*nodes.begin()),
								 thrust::raw_pointer_cast(&*particleId.begin()),
								 max_ll, min_ll, ite, cubesInX, cubesInY, cubesInZ, side,
								 thrust::raw_pointer_cast(&*edges.begin()),
								 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));		

		numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubes); // set the correct number of edges

		std::cout << "numOfEdges after " << numOfEdges << std::endl;
	}

//------------ TODO : try to do the edge calculation parallely for the number of iterations as well
	// for each cube, get the set of edges after comparing
	struct getEdges : public thrust::unary_function<int, void>
	{
		int    ite;
		float  max_ll, min_ll;
		Node  *nodes;

		int   *startOfChunks, *sizeOfChunks;
		int   *cubeMapping, *cubeMappingInv;
		int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;

		int    side;
		int    cubesInX, cubesInY, cubesInZ;
		
		Edge  *edges;
		int   *edgeStartOfCubes, *edgeSizeOfCubes;

		__host__ __device__
		getEdges(int *particleStartOfCubes, int *particleSizeOfCubes, 
				int *startOfChunks, int *sizeOfChunks,
				int *cubeMapping, int *cubeMappingInv, Node *nodes,
				int *particleId, float max_ll, float min_ll, int ite,
				int cubesInX, int cubesInY, int cubesInZ, int side, 
				Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes) :
 				particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
				startOfChunks(startOfChunks), sizeOfChunks(sizeOfChunks),
				cubeMapping(cubeMapping), cubeMappingInv(cubeMappingInv), nodes(nodes),
				particleId(particleId), max_ll(max_ll), min_ll(min_ll), ite(ite), 
				cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), side(side),
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

		__host__ __device__
		void operator()(int i)
		{	
			for(int l=startOfChunks[i]; l<startOfChunks[i]+sizeOfChunks[i]; l++)
			{		
				int i_mapped = cubeMapping[l];

				// get x,y,z coordinates for the cube
				int tmp = i_mapped % (cubesInX*cubesInY);
				int z = i_mapped / (cubesInX*cubesInY);
				int y = tmp / cubesInX;
				int x = tmp % cubesInX;

				int len = (side-1)/2;

				for(int num=0; num<ite; num++)
				{
					int tmp1 = num % (side*side);
					int z1 = num / (side*side);
					int y1 = tmp1 / side;
					int x1 = tmp1 % side;		

					// get x,y,z coordinates for the current cube 
					int currentX = x - len + x1;
					int currentY = y - len + y1;
					int currentZ = z - len + z1;

					int cube_mapped = -1, cube = -1;
					if((currentX>=0 && currentX<cubesInX) && (currentY>=0 && currentY<cubesInY) && (currentZ>=0 && currentZ<cubesInZ))
					{
						cube_mapped = (currentZ*(cubesInY*cubesInX) + currentY*cubesInX + currentX);
						cube = cubeMappingInv[cube_mapped];
					}

					if(cube_mapped==-1 || cube==-1 || particleSizeOfCubes[l]==0 || particleSizeOfCubes[cube]==0) continue;

					Edge  e;

					// for each particle in this cube
					float dist_min = max_ll+1;
					for(int j=particleStartOfCubes[l]; j<particleStartOfCubes[l]+particleSizeOfCubes[l]; j++)
					{
						int pId_j = particleId[j];

						// compare with particles in neighboring cube
						for(int k=particleStartOfCubes[cube]; k<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; k++)
						{
							int pId_k = particleId[k];

							Node node_j = nodes[pId_j];
							Node node_k = nodes[pId_k];

							double xd = (node_j.pos.x-node_k.pos.x);  if (xd < 0.0f) xd = -xd;
							double yd = (node_j.pos.y-node_k.pos.y);  if (yd < 0.0f) yd = -yd;
							double zd = (node_j.pos.z-node_k.pos.z);  if (zd < 0.0f) zd = -zd;
																	                                                                                                                   								if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
							{
								double dist = (double)std::sqrt(xd*xd + yd*yd + zd*zd);

								if(dist <= max_ll && dist < dist_min)
								{
									int srcV = (pId_j <= pId_k) ? pId_j : pId_k;
									int desV = (srcV == pId_k)  ? pId_j : pId_k;							

									dist_min = dist;		
									e = Edge(srcV, desV, dist);		

									if(dist_min <= min_ll) goto loop;
								}							
							}			
						}
					}

					// add edge
					loop:
					if(dist_min < max_ll + 1)
					{
						edges[edgeStartOfCubes[l] + edgeSizeOfCubes[l]] = e;
						edgeSizeOfCubes[l]++;
					}
				}
			}
		}
	};




	//---------------- METHOD - Global functions

  // combine two local merge trees, two cubes at a time
	void globalStep()
	{
		int cubesOri = cubes;
		int sizeP = 2;

		// set new number of cubes
		int cubesOld = cubes;
		cubes = (int)std::ceil(((double)cubes/2));

		std::cout << std::endl;

		if(numOfEdges==0) return;
	
		// iteratively combine the cubes two at a time
		while(cubes!=cubesOld && cubes>0)
		{
	    struct timeval begin, end, diff;
	    gettimeofday(&begin, 0);
			thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes, 
				combineFreeLists(thrust::raw_pointer_cast(&*tmpNxt.begin()),
												 thrust::raw_pointer_cast(&*tmpFree.begin()),
												 sizeP, cubesOri));
			thrust::for_each(CountingIterator(0), CountingIterator(0)+cubes,
				combineMergeTrees(thrust::raw_pointer_cast(&*cubeMapping.begin()),
													thrust::raw_pointer_cast(&*cubeId.begin()),
													thrust::raw_pointer_cast(&*nodesTmp1.begin()),
													thrust::raw_pointer_cast(&*nodes.begin()),
													thrust::raw_pointer_cast(&*tmpNxt.begin()),
													thrust::raw_pointer_cast(&*tmpFree.begin()),
													thrust::raw_pointer_cast(&*edges.begin()),
													thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
													thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
													min_ll, sizeP, cubesOri, numOfParticles));
			gettimeofday(&end, 0);

      timersub(&end, &begin, &diff);
      float seconds = diff.tv_sec + 1.0E-6*diff.tv_usec;
      std::cout << "Time elapsed: " << seconds << " s for nonEmptyCubes " << cubes;

			numOfEdges = thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+cubesOri);
			std::cout << " numOfEdges after " << numOfEdges << std::endl;

			// set new number of cubes & sizeP
			sizeP *= 2;
			cubesOld = cubes;
			cubes = (int)std::ceil(((double)cubes/2));
		}

		#ifdef TEST
			outputMergeTreeDetails("The sub merge trees.."); // output merge tree details
		#endif
	}

	// combine free nodes lists of each cube at this iteration
	struct combineFreeLists : public thrust::unary_function<int, void>
  {
		int   sizeP, numOfCubesOri;

		int  *tmpNxt, *tmpFree;

		__host__ __device__
    combineFreeLists(int *tmpNxt, int *tmpFree, int sizeP, int numOfCubesOri) :
				tmpNxt(tmpNxt), tmpFree(tmpFree), sizeP(sizeP), numOfCubesOri(numOfCubesOri) {}

    __host__ __device__
    void operator()(int i)
    {	
			int cubeStart = sizeP*i;
			int cubeEnd   = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

			int k;
			for(k=cubeStart; k<cubeEnd; k+=sizeP/2)
			{ if(tmpNxt[k]!=-1) { tmpNxt[cubeStart] = tmpNxt[k];  break; } }

			int nxt;
			while(k<cubeEnd)
			{
				nxt = tmpNxt[k];
				while(nxt!=-1 && tmpFree[nxt]!=-1)  nxt = tmpFree[nxt];

				k += sizeP/2;
				if(k<cubeEnd) tmpFree[nxt] = tmpNxt[k];
			}
		}
	};

	// combine two local merge trees
	struct combineMergeTrees : public thrust::unary_function<int, void>
  {
    float  min_ll;
		int    sizeP, numOfCubesOri, numOfParticles;

    int   *cubeId, *cubeMapping;
		int   *tmpNxt, *tmpFree;

		Edge *edges;
		int  *edgeStartOfCubes, *edgeSizeOfCubes;

    Node  *nodes, *nodesTmp1;

    __host__ __device__
    combineMergeTrees(int *cubeMapping, int *cubeId, 
			Node *nodesTmp1, Node *nodes, int *tmpNxt, int *tmpFree, 
				Edge *edges, int *edgeStartOfCubes, int *edgeSizeOfCubes,
				float min_ll, int sizeP, int numOfCubesOri, int numOfParticles) :
        cubeMapping(cubeMapping), cubeId(cubeId), 
				nodesTmp1(nodesTmp1), nodes(nodes),	tmpNxt(tmpNxt), tmpFree(tmpFree), 
				edges(edges), edgeStartOfCubes(edgeStartOfCubes), edgeSizeOfCubes(edgeSizeOfCubes),
				min_ll(min_ll), sizeP(sizeP), numOfCubesOri(numOfCubesOri), numOfParticles(numOfParticles) {}

    __host__ __device__
    void operator()(int i)
    {	
			int cubeStart  = sizeP*i;
			int cubeEnd    = (sizeP*(i+1)<=numOfCubesOri) ? sizeP*(i+1) : numOfCubesOri;

			int cubeStartM = cubeMapping[cubeStart];
			int cubeEndM   = cubeMapping[cubeEnd-1];

			// get the edges
			for(int k=cubeStart; k<cubeEnd; k++)
			{	
				int size = 0;
				for(int j=edgeStartOfCubes[k]; j<edgeStartOfCubes[k]+edgeSizeOfCubes[k]; j++)
				{
					Edge e = ((Edge)edges[j]);

					if(!(cubeId[e.srcId]>=cubeStartM && cubeId[e.srcId]<=cubeEndM) || 
						 !(cubeId[e.desId]>=cubeStartM && cubeId[e.desId]<=cubeEndM))
					{
						edges[edgeStartOfCubes[k] + size++] = e;
						continue;
					}

					// use this edge (e), to combine the merge trees
					//-----------------------------------------------------

					Node *src = &nodes[e.srcId];
					Node *des = &nodes[e.desId];

					float weight = (e.weight < min_ll) ? min_ll : e.weight; 

					// find the src & des nodes just below the required weight
					while(src->parent!=NULL && src->parent->value <= weight)	src = src->parent;
					while(des->parent!=NULL && des->parent->value <= weight)	des = des->parent;

					// if src & des already have the same halo id, do NOT do anything
					if(src->haloId==des->haloId) continue;

					int srcCount = src->count;
					int desCount = des->count;

					float srcX = src->pos.x; float desX = des->pos.x;	float srcVX = src->vel.x; float desVX = des->vel.x;
					float srcY = src->pos.y; float desY = des->pos.y;	float srcVY = src->vel.y; float desVY = des->vel.y;
					float srcZ = src->pos.z; float desZ = des->pos.z;	float srcVZ = src->vel.z; float desVZ = des->vel.z;

					// get the original parents of src & des nodes
					Node *srcTmp = (src->parent!=NULL) ? src->parent : NULL;
					Node *desTmp = (des->parent!=NULL) ? des->parent : NULL;

					// remove the src & des from the child list of their parents
					if(srcTmp)
					{
						Node *child = srcTmp->childS;
						if(child && child->value==src->value && child->haloId==src->haloId) 
							srcTmp->childS = child->sibling;
						else
						{
							while(child && child->sibling)
							{
								if(child->sibling->value==src->value && child->sibling->haloId==src->haloId)
								{ child->sibling = child->sibling->sibling;	break; }
								child = child->sibling;
							}

							if(child && !child->sibling) srcTmp->childE = child;
						}

						if(!srcTmp->childS) srcTmp->childE = NULL;
					}
					src->parent =NULL;	src->sibling=NULL;

					if(desTmp)
					{
						Node *child = desTmp->childS;
						if(child && child->value==des->value && child->haloId==des->haloId)
							desTmp->childS = child->sibling;
						else
						{
							while(child && child->sibling)
							{
								if(child->sibling->value==des->value && child->sibling->haloId==des->haloId)
								{ child->sibling = child->sibling->sibling;	break; }
								child = child->sibling;
							}

							if(child && !child->sibling) desTmp->childE = child;
						}

						if(!desTmp->childS) desTmp->childE = NULL;
					}
					des->parent =NULL;	des->sibling=NULL;



					// set n node
					Node *n;
					bool freeDes=false;
					if(src->value==weight && des->value==weight) // merge src & des, free des node, set n to src, then connect their children & fix the loop
					{ 
						n = src;					
						Node *child = des->childS;
						while(child!=NULL) { child->parent = n;	child = child->sibling;	}

						if(n->childE)  n->childE->sibling = des->childS;
						else  n->childS = des->childS;
						n->childE = des->childE;	
						freeDes = true;
					}
					else if(src->value==weight) // set des node's parent to be src, set n to src, then fix the loop
					{ 
						n = src;	n->childE->sibling = des;		n->childE = des;		des->parent = n; 	
					}
					else if(des->value==weight) // set src node's parent to be des, set n to des, then fix the loop
					{ 
						n = des;	n->childE->sibling = src;		n->childE = src;		src->parent = n;
					}
					else if(src->value!=weight && des->value!=weight) // create a new node, set this as parent of both src & des, then fix the loop
					{ 
						if(tmpNxt[cubeStart]!=-1)
						{
							n = &nodesTmp1[tmpNxt[cubeStart]];
							int tmpVal = tmpFree[tmpNxt[cubeStart]];
							tmpFree[tmpNxt[cubeStart]] = -2;
							tmpNxt[cubeStart] = tmpVal;

							n->childS = src;	n->childE = des;						
							src->parent = n; 	des->parent = n;
							src->sibling = des;
						}
						#if THRUST_DEVICE_BACKEND != THRUST_DEVICE_BACKEND_CUDA
						else
							std::cout << "***no Free item .... this shouldnt happen*** " << cubeStart << " " << e.weight << " " << min_ll << " " << std::endl;
						#endif
					}
					n->value  = weight;
					n->count  = src->count + des->count;
					n->pos = Point(src->pos.x+des->pos.x, src->pos.y+des->pos.y, src->pos.z+des->pos.z);
					n->vel = Point(src->vel.x+des->vel.x, src->vel.y+des->vel.y, src->vel.z+des->vel.z);
					n->haloId = (src->haloId < des->haloId) ? src->haloId : des->haloId;

					if(freeDes && des->nodeId>=numOfParticles)
					{
						// free des node
						int tmpVal = tmpNxt[cubeStart];
						tmpNxt[cubeStart] = des->nodeId-numOfParticles;
						tmpFree[tmpNxt[cubeStart]] = tmpVal;

						des->nodeId = des->nodeId;
						des->haloId = -1;
						des->value  = 0.0f;
						des->count  = 0;
						des->pos = Point(0,0,0);
						des->vel = Point(0,0,0);
						des->parent = NULL;
						des->parentSuper = NULL;
						des->childS = NULL;
						des->childE = NULL;
						des->sibling = NULL;
					}



					bool done = false;
					while(srcTmp!=NULL && desTmp!=NULL)
					{
						if(srcTmp->value < desTmp->value)
						{
							n->parent = srcTmp;
							if(srcTmp->childE)  srcTmp->childE->sibling = n;
							else  srcTmp->childS = n;
							srcTmp->childE = n;
							srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
							srcCount = srcTmp->count;
							srcTmp->count += desCount;
							srcX = srcTmp->pos.x;		srcVX = srcTmp->vel.x;
							srcY = srcTmp->pos.y;		srcVY = srcTmp->vel.y;
							srcZ = srcTmp->pos.z;		srcVZ = srcTmp->vel.z;
							srcTmp->pos = Point(srcTmp->pos.x+desX, srcTmp->pos.y+desY, srcTmp->pos.z+desZ);
							srcTmp->vel = Point(srcTmp->vel.x+desVX, srcTmp->vel.y+desVY, srcTmp->vel.z+desVZ);

							n = srcTmp;
							srcTmp = srcTmp->parent;

							if(srcTmp)
							{
								Node *child = srcTmp->childS;
								if(child && child->value==n->value && child->haloId==n->haloId)
									srcTmp->childS = child->sibling;
								else
								{
									while(child && child->sibling)
									{
										if(child->sibling->value==n->value && child->sibling->haloId==n->haloId)
										{ child->sibling = child->sibling->sibling;	break; }
										child = child->sibling;
									}

									if(child && !child->sibling) srcTmp->childE = child;
								}

								if(!srcTmp->childS) srcTmp->childE = NULL;
							}
							n->parent  = NULL;	n->sibling = NULL;
						}
						else if(srcTmp->value > desTmp->value)
						{
							n->parent = desTmp;
							if(desTmp->childE)  desTmp->childE->sibling = n;
							else  desTmp->childS = n;
							desTmp->childE = n;
							desTmp->haloId = (desTmp->haloId < n->haloId) ? desTmp->haloId : n->haloId;
							desCount = desTmp->count;
							desTmp->count += srcCount;
							desX = desTmp->pos.x;		desVX = desTmp->vel.x;
							desY = desTmp->pos.y;		desVY = desTmp->vel.y;
							desZ = desTmp->pos.z;		desVZ = desTmp->vel.z;
							desTmp->pos = Point(desTmp->pos.x+srcX, desTmp->pos.y+srcY, desTmp->pos.z+srcZ);
							desTmp->vel = Point(desTmp->vel.x+srcVX, desTmp->vel.y+srcVY, desTmp->vel.z+srcVZ);

							n = desTmp;
							desTmp = desTmp->parent;

							if(desTmp)
							{
								Node *child = desTmp->childS;
								if(child && child->value==n->value && child->haloId==n->haloId)
									desTmp->childS = child->sibling;
								else
								{
									while(child && child->sibling)
									{
										if(child->sibling->value==n->value && child->sibling->haloId==n->haloId)
										{ child->sibling = child->sibling->sibling;	break; }
										child = child->sibling;
									}

									if(child && !child->sibling) desTmp->childE = child;
								}

								if(!desTmp->childS) desTmp->childE = NULL;
							}
							n->parent  = NULL;	n->sibling = NULL;
						}
						else if(srcTmp->value == desTmp->value)
						{
							if(srcTmp->haloId != desTmp->haloId) // combine srcTmp & desTmp			
							{	
								Node *child = desTmp->childS;
								while(child!=NULL) { child->parent = srcTmp;  child = child->sibling; }
								if(srcTmp->childE)  srcTmp->childE->sibling = desTmp->childS;
								else  srcTmp->childS = desTmp->childS;
								srcTmp->childE = desTmp->childE;
								srcTmp->haloId = (srcTmp->haloId < desTmp->haloId) ? srcTmp->haloId : desTmp->haloId;
								srcTmp->count += desTmp->count;
								srcTmp->pos = Point(srcTmp->pos.x+desTmp->pos.x, srcTmp->pos.y+desTmp->pos.y, srcTmp->pos.z+desTmp->pos.z);
								srcTmp->vel = Point(srcTmp->vel.x+desTmp->vel.x, srcTmp->vel.y+desTmp->vel.y, srcTmp->vel.z+desTmp->vel.z);
							}

							if(!srcTmp->childS && !srcTmp->childE)
							{
								if(srcTmp->parent)
								{
									Node *child = srcTmp->parent->childS;
									if(child && child->value==srcTmp->value && child->haloId==srcTmp->haloId)
										srcTmp->parent->childS = child->sibling;
									else
									{
										while(child && child->sibling)
										{
											if(child->sibling->value==srcTmp->value && child->sibling->haloId==srcTmp->haloId)
											{ child->sibling = child->sibling->sibling;	break; }
											child = child->sibling;
										}

										if(child && !child->sibling)  srcTmp->parent->childE = child;
									}

									if(!srcTmp->parent->childS) srcTmp->parent->childE=NULL;
								}

								Node *tmp = srcTmp->parent;
								if(srcTmp->nodeId>=numOfParticles)
								{
									int tmpVal = tmpNxt[cubeStart];
									tmpNxt[cubeStart] = srcTmp->nodeId-numOfParticles;
									tmpFree[tmpNxt[cubeStart]] = tmpVal;

									srcTmp->nodeId = srcTmp->nodeId;
									srcTmp->haloId = -1;
									srcTmp->value  = 0.0f;
									srcTmp->count  = 0;
									srcTmp->pos = Point(0,0,0);
									srcTmp->vel = Point(0,0,0);
									srcTmp->parent = NULL;
									srcTmp->parentSuper = NULL;
									srcTmp->childS = NULL;
									srcTmp->childE = NULL;
									srcTmp->sibling = NULL;
								}
								srcTmp = tmp;
							}

							if(srcTmp)
							{
								n->parent = srcTmp;
								if(srcTmp->childE)  srcTmp->childE->sibling = n;
								else  srcTmp->childS = n;
								srcTmp->childE = n;
								srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
							}
							else	n->parent =NULL;

							done = true;
							break;
						}
					}



					if(!done && srcTmp!=NULL)
					{
						n->parent = srcTmp;
						if(srcTmp->childE)  srcTmp->childE->sibling = n;
						else  srcTmp->childS = n;
						srcTmp->childE = n;
						srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
						srcCount = srcTmp->count;
						srcTmp->count += desCount;
						srcX = srcTmp->pos.x;		srcVX = srcTmp->vel.x;
						srcY = srcTmp->pos.y;		srcVY = srcTmp->vel.y;
						srcZ = srcTmp->pos.z;		srcVZ = srcTmp->vel.z;
						srcTmp->pos = Point(srcTmp->pos.x+desX, srcTmp->pos.y+desY, srcTmp->pos.z+desZ);
						srcTmp->vel = Point(srcTmp->vel.x+desVX, srcTmp->vel.y+desVY, srcTmp->vel.z+desVZ);

						n = srcTmp;
						srcTmp = srcTmp->parent;
					}
					while(!done && srcTmp!=NULL)
					{
						srcTmp->haloId = (srcTmp->haloId < n->haloId) ? srcTmp->haloId : n->haloId;
						srcCount = srcTmp->count;
						srcTmp->count += desCount;
						srcX = srcTmp->pos.x;		srcVX = srcTmp->vel.x;
						srcY = srcTmp->pos.y;		srcVY = srcTmp->vel.y;
						srcZ = srcTmp->pos.z;		srcVZ = srcTmp->vel.z;
						srcTmp->pos = Point(srcTmp->pos.x+desX, srcTmp->pos.y+desY, srcTmp->pos.z+desZ);
						srcTmp->vel = Point(srcTmp->vel.x+desVX, srcTmp->vel.y+desVY, srcTmp->vel.z+desVZ);

						n = srcTmp;
						srcTmp = srcTmp->parent;
					}

					if(!done && desTmp!=NULL)
					{		
						n->parent = desTmp;
						if(desTmp->childE)  desTmp->childE->sibling = n;
						else  desTmp->childS = n;
						desTmp->childE = n;
						desTmp->haloId = (desTmp->haloId < n->haloId) ? desTmp->haloId : n->haloId;
						desCount = desTmp->count;
						desTmp->count += srcCount;
						desX = desTmp->pos.x;		desVX = desTmp->vel.x;	
						desY = desTmp->pos.y;		desVY = desTmp->vel.y;
						desZ = desTmp->pos.z;		desVZ = desTmp->vel.z;
						desTmp->pos = Point(desTmp->pos.x+srcX, desTmp->pos.y+srcY, desTmp->pos.z+srcZ);
						desTmp->pos = Point(desTmp->vel.x+srcVX, desTmp->vel.y+srcVY, desTmp->vel.z+srcVZ);

						n = desTmp;
						desTmp = desTmp->parent;
					}
					while(!done && desTmp!=NULL)
					{
						desTmp->haloId = (desTmp->haloId < n->haloId) ? desTmp->haloId : n->haloId;
						desCount = desTmp->count;
						desTmp->count += srcCount;
						desX = desTmp->pos.x;		desVX = desTmp->vel.x;	
						desY = desTmp->pos.y;		desVY = desTmp->vel.y;
						desZ = desTmp->pos.z;		desVZ = desTmp->vel.z;
						desTmp->pos = Point(desTmp->pos.x+srcX, desTmp->pos.y+srcY, desTmp->pos.z+srcZ);
						desTmp->pos = Point(desTmp->vel.x+srcVX, desTmp->vel.y+srcVY, desTmp->vel.z+srcVZ);

						n = desTmp;
						desTmp = desTmp->parent;
					}

					//-----------------------------------------------------

				}

				edgeSizeOfCubes[k] = size;
			}
		}		
	};	
};

}

#endif

