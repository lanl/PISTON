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
	struct Node
	{
		int   nodeId;
		float value;
		int   haloId;
		Node *parent;
		Node *parentSuper;

		__host__ __device__
		Node() { nodeId=-1; value=0.0f; haloId=-1; parent=NULL; parentSuper=NULL; }

		__host__ __device__
		Node(int nodeId, float value, int haloId, Node *parent, Node *parentSuper) :
			nodeId(nodeId), value(value), haloId(haloId), parent(parent), parentSuper(parentSuper) {}
	};

	struct Edge
	{
		int srcId, desId;
		float weight;

		__host__ __device__
		Edge() { srcId=-1; desId=-1; weight=-1; }
		
		__host__ __device__
		Edge(int srcId, int desId, float weight) : srcId(srcId), desId(desId), weight(weight) {}
	};

  typedef thrust::tuple<int, int> Int2;

	thrust::device_vector<int> particleId; // for each particle, particle id
	thrust::device_vector<int> cubeId;	   // for each particle, cube id

	float max_ll;   				// maximum linking length
	Point lBoundS, uBoundS; // lower & upper bounds of the entire space

	int numOfCubes;					   	  // total number of cubes in space
	int cubesInX, cubesInY, cubesInZ;     // number of cubes in each dimension
	thrust::device_vector<int>   particleSizeOfCubes; 	// number of particles in cubes
	thrust::device_vector<int>   particleStartOfCubes;	// stratInd of cubes  (within particleId)

	thrust::device_vector<int>   neighborsOfCubes;		// neighbors of cubes (for each cube, store only 13 of them)
	thrust::device_vector<int>   sizeOfNeighborCubes;	// neighbors of cubes (for each cube, store only 13 of them)
	
	thrust::device_vector<Node>  nodes, nodesTmp1, nodesTmp2;

	int numOfEdges, numOfEdgesN;
	thrust::device_vector<Edge>  edges;
	thrust::device_vector<int>   edgeSizeOfCubes, edgeSizeOfCubesN;
	thrust::device_vector<int>   edgeStartOfCubes, edgeStartOfCubesN;

	thrust::device_vector<int>   tmpArray;

	halo_merge(float max_linkLength, std::string filename="", std::string format=".cosmo", 
						 int n = 1, int np=1, float rL=-1, bool periodic=false) : halo(filename, format, n, np, rL, periodic)
	{
		if(numOfParticles!=0)
		{
			struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3;

			//---- init stuff

			max_ll = max_linkLength;  // get max_linkinglength

			initParticleIds();	      // set particle ids
			initCubeIds(); 	          // set cube ids, initially set each particle to a unique cube id
			getBounds();		      		// get bounds of the entire space

			//---- divide space into cubes (TODO: need to change this like in kdtree)

			gettimeofday(&begin, 0);
			setNumberOfCubes();	      // get total number of cubes
			setCubeIds();		      		// for each particle, set cube id
			sortParticlesByCubeID();  // sort Particles by cube Id

			getSizeOfCubes();  		  // for each cube, count particles
			getStartOfCubes();	      // for each cube, get the start of cube particles (in particleId array)
      getNeighborsOfCubes();    // for each cube, get its neighbors
			getSizeofNeighborCubes(); // for each cube, count the particle size in its neighbors
			gettimeofday(&mid1, 0);

			std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;

			#ifdef TEST
				outputCubeDetails("init cube details"); // output cube details
			#endif

			//------- METHOD :
	    // parallel for each cube, get the set of edges & create the submerge tree.
			// globally combine them, two cubes at a time (by sorting their edge sets)

			gettimeofday(&mid2, 0);
			localStepMethod();
			gettimeofday(&mid3, 0);

			neighborsOfCubes.clear();
			particleSizeOfCubes.clear();
			particleStartOfCubes.clear();
			std::cout << "-- localStep done" << std::endl;

			gettimeofday(&mid4, 0);
			globalStepMethod();
			gettimeofday(&end, 0);

			edges.clear();
			sizeOfNeighborCubes.clear();
			edgeSizeOfCubes.clear();
			edgeStartOfCubes.clear();
			std::cout << "-- globalStep done" << std::endl;


			timersub(&mid1, &begin, &diff1);
			float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
			std::cout << "Time elapsed: " << seconds1 << " s for dividing space into cubes"<< std::endl << std::flush;
			timersub(&mid3, &mid2, &diff2);
			float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
			std::cout << "Time elapsed: " << seconds2 << " s for localStep - finding edges in each cube"<< std::endl << std::flush;
			timersub(&end, &mid4, &diff3);
			float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
			std::cout << "Time elapsed: " << seconds3 << " s for globalStep - combine edges & creating the merge tree"<< std::endl << std::flush;
		}
	}

	void operator()(float linkLength, int  particleSize)
	{
		clear();

		linkLength    = linkLength;
		particleSize  = particleSize;

		// no valid particles, return
		if(numOfParticles==0) return;

		struct timeval begin, mid, end, diff1, diff2;
		gettimeofday(&begin, 0);
		findHalos(linkLength);        //find halos
		gettimeofday(&mid, 0);
		getUniqueHalos(particleSize); // get the unique valid halo ids
		gettimeofday(&end, 0);

		timersub(&mid, &begin, &diff1);
		float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
		std::cout << "Time elapsed: " << seconds1 << " s for merging"<< std::endl << std::flush;
		timersub(&end, &mid, &diff2);
		float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
		std::cout << "Time elapsed: " << seconds2 << " s for finding valid halos"<< std::endl << std::flush;

		setColors(); // set colors to halos
		std::cout << "Number of Particles : " << numOfParticles <<  " Number of Halos found : " << numOfHalos << std::endl << std::endl;
	}

	void findHalos(float linkLength)
	{
	  thrust::transform(nodes.begin(), nodes.end(), haloIndex.begin(), setHaloId(linkLength));
	}

	// for a given node, set its halo id
  struct setHaloId : public thrust::unary_function<Node, int>
  {
    float linkLength;

    __host__ __device__
    setHaloId(float linkLength) : linkLength(linkLength) {}

    __host__ __device__
    int operator()(Node n)
    {
      Node nChild;

      if(n.parentSuper!=NULL && (n.parentSuper)->value<=linkLength)
        n = *(n.parentSuper);

      while(n.parent!=NULL)
      {
        nChild = n;
        n= *(n.parent);

        if(n.value >= linkLength)
        {
          n = nChild;
          break;
        }
      }

      return n.haloId;
    }
  };



	//------- init stuff

  // set initial particle ids
	void initParticleIds()
	{
		particleId.resize(numOfParticles);
		thrust::copy(CountingIterator(0), CountingIterator(0)+numOfParticles, particleId.begin());
	}

	// set initial cube ids
	void initCubeIds()
	{
		cubeId.resize(numOfParticles);
		thrust::copy(CountingIterator(0), CountingIterator(0)+numOfParticles, cubeId.begin());
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



	//------- divide space into cubes

	// get total number of cubes
	void setNumberOfCubes()
	{
		cubesInX = (std::ceil((uBoundS.x - lBoundS.x)/max_ll) == 0) ? 1 : std::ceil((uBoundS.x - lBoundS.x)/max_ll);
		cubesInY = (std::ceil((uBoundS.y - lBoundS.y)/max_ll) == 0) ? 1 : std::ceil((uBoundS.y - lBoundS.y)/max_ll);
		cubesInZ = (std::ceil((uBoundS.z - lBoundS.z)/max_ll) == 0) ? 1 : std::ceil((uBoundS.z - lBoundS.z)/max_ll);

		numOfCubes = cubesInX*cubesInY*cubesInZ;
	}

	// set cube ids of particles
	void setCubeIds()
	{
	  thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(inputX.begin(), inputY.begin(), inputZ.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(inputX.end(),   inputY.end(),   inputZ.end())),
                      cubeId.begin(), setCubeIdOfParticle(max_ll, lBoundS, cubesInX, cubesInY));
	}

	// for a given particle, set its cube id
	struct setCubeIdOfParticle : public thrust::unary_function<Float3, int>
	{
		Point  lBoundS;
		float  max_ll;
		int    cubesInX, cubesInY;

		__host__ __device__
		setCubeIdOfParticle(float max_ll, Point lBoundS, int cubesInX, int cubesInY) :
			max_ll(max_ll), lBoundS(lBoundS), cubesInX(cubesInX), cubesInY(cubesInY) {}

		__host__ __device__
		int operator()(const Float3 a)
		{
			// get x,y,z coordinates for the cube
			int z = (thrust::get<2>(a)-lBoundS.z) / max_ll;
			int y = (thrust::get<1>(a)-lBoundS.y) / max_ll;
			int x = (thrust::get<0>(a)-lBoundS.x) / max_ll;

			// get cube id
			return z*(cubesInX*cubesInY) + y*cubesInX + x;
		}
	};

	// sort particles by cube id
	void sortParticlesByCubeID()
	{
		thrust::stable_sort_by_key(cubeId.begin(), cubeId.end(), particleId.begin());
	}
		
	// for each cube, count its particles
	void getSizeOfCubes()
	{
		thrust::device_vector<int> tmpA; tmpA.resize(numOfParticles);
		thrust::device_vector<int> tmpB; tmpB.resize(numOfParticles);
		thrust::device_vector<int> tmpC; tmpC.resize(numOfCubes);
		thrust::device_vector<int> tmpD; tmpD.resize(numOfCubes);

		thrust::copy(cubeId.begin(), cubeId.end(), tmpA.begin());
		thrust::stable_sort(tmpA.begin(), tmpA.end());
		thrust::fill(tmpB.begin(), tmpB.end(), 1);

		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
		new_end = thrust::reduce_by_key(tmpA.begin(), tmpA.end(), tmpB.begin(), tmpC.begin(), tmpD.begin());
		particleSizeOfCubes.resize(numOfCubes);
		thrust::for_each(thrust::make_zip_iterator(make_tuple(tmpC.begin(), tmpD.begin())),
                     thrust::make_zip_iterator(make_tuple(thrust::get<0>(new_end), thrust::get<1>(new_end))),
                     setNumberOfparticlesForCube(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin())));
	}

	// for a given cube, set the number of particles
	struct setNumberOfparticlesForCube : public thrust::unary_function<Int2, void>
	  {
	    int *particlesOfCube;

	    __host__ __device__
	    setNumberOfparticlesForCube(int *particlesOfCube) :
	      particlesOfCube(particlesOfCube) {}

	    __host__ __device__
	    void operator()(const Int2 a)
	    {
	      particlesOfCube[thrust::get<0>(a)] = thrust::get<1>(a);
	    }
	  };

	// for each cube, get the start of cube particles (in particleId array)
	void getStartOfCubes()
	{
		particleStartOfCubes.resize(numOfCubes);
		thrust::exclusive_scan(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, particleStartOfCubes.begin());
	}
	
	// for each cube, get its neighbors
	void getNeighborsOfCubes()
	{
		neighborsOfCubes.resize(13*numOfCubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				setNeighboringCubes(cubesInX, cubesInY, cubesInZ,
														thrust::raw_pointer_cast(&*neighborsOfCubes.begin())));
	}

	// for a given cube where center is (x,y,z), get its neighboring cubes
	struct setNeighboringCubes : public thrust::unary_function<int, void>
	{
		int  cubesInX, cubesInY, cubesInZ;
		int *neighborsOfCubes;

		__host__ __device__
		setNeighboringCubes(int cubesInX, int cubesInY, int cubesInZ, int *neighborsOfCubes) :
				cubesInX(cubesInX), cubesInY(cubesInY), cubesInZ(cubesInZ), neighborsOfCubes(neighborsOfCubes) {}

		__host__ __device__
		void operator()(int i)
		{
			int startInd = i*13;

			int tmp = i % (cubesInX*cubesInY);

			// get x,y,z coordinates for the cube
			int z = i / (cubesInX*cubesInY);
			int y = tmp / cubesInX;
			int x = tmp % cubesInX;

			neighborsOfCubes[startInd+0]  = (z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + y*cubesInX + x) : -1;  																				// x,y,z+1
			neighborsOfCubes[startInd+1]  = (y+1<cubesInY) ? (z*(cubesInX*cubesInY) + (y+1)*cubesInX + x) : -1;	 																				// x,y+1,z
			neighborsOfCubes[startInd+2]  = (y+1<cubesInY && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + (y+1)*cubesInX + x) : -1;	 										// x,y+1,z+1
			neighborsOfCubes[startInd+3]  = (y+1<cubesInY && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + (y+1)*cubesInX + x) : -1;			   										// x,y+1,z-1
			neighborsOfCubes[startInd+4]  = (x+1<cubesInX) ? (z*(cubesInX*cubesInY) + y*cubesInX + (x+1)) : -1; 																				// x+1,y,z
			neighborsOfCubes[startInd+5]  = (x+1<cubesInX && y+1<cubesInY) ? (z*(cubesInX*cubesInY) + (y+1)*cubesInX + (x+1)) : -1; 										// x+1,y+1,z
			neighborsOfCubes[startInd+6]  = (x+1<cubesInX && y-1>=0) ? (z*(cubesInX*cubesInY) + (y-1)*cubesInX + (x+1)) : -1;		      									// x+1,y-1,z
			neighborsOfCubes[startInd+7]  = (x+1<cubesInX && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + y*cubesInX + (x+1)) : -1; 										// x+1,y,z+1
			neighborsOfCubes[startInd+8]  = (x+1<cubesInX && y+1<cubesInY && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + (y+1)*cubesInX + (x+1)) : -1; // x+1,y+1,z+1
			neighborsOfCubes[startInd+9]  = (x+1<cubesInX && y-1>=0 && z+1<cubesInZ) ? ((z+1)*(cubesInX*cubesInY) + (y-1)*cubesInX + (x+1)) : -1; 			// x+1,y-1,z+1
			neighborsOfCubes[startInd+10] = (x+1<cubesInX && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + y*cubesInX + (x+1)) : -1;		     										// x+1,y,z-1
			neighborsOfCubes[startInd+11] = (x+1<cubesInX && y+1<cubesInY && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + (y+1)*cubesInX + (x+1)) : -1;		    // x+1,y+1,z-1
			neighborsOfCubes[startInd+12] = (x+1<cubesInX && y-1>=0 && z-1>=0) ? ((z-1)*(cubesInX*cubesInY) + (y-1)*cubesInX + (x+1)) : -1;		    		  // x+1,y-1,z-1
		}
	};

	// for each cube, count- the size of particle size in its neighbors 
	void getSizeofNeighborCubes()
	{
	  sizeOfNeighborCubes.resize(numOfCubes);
    thrust::transform(CountingIterator(0), CountingIterator(0)+numOfCubes, sizeOfNeighborCubes.begin(),
        countNeighborParticles(thrust::raw_pointer_cast(&*neighborsOfCubes.begin()),
                               thrust::raw_pointer_cast(&*particleSizeOfCubes.begin())));
	}

	// for a given cube, count particle size in its neighbors
	struct countNeighborParticles : public thrust::unary_function<int, int>
	{
		int *neighborsOfCubes, *particleSizeOfCubes;

		__host__ __device__
		countNeighborParticles(int *neighborsOfCubes, int *particleSizeOfCubes) :
				neighborsOfCubes(neighborsOfCubes), particleSizeOfCubes(particleSizeOfCubes) {}

		__host__ __device__
		int operator()(int i)
		{
			int sum = 0;
			for(int j=(i*13); j<(i*13)+13; j++)
				sum += particleSizeOfCubes[neighborsOfCubes[j]];

			return sum;
		}
	};



	//------- output results

	// print cube details from device vectors
	void outputCubeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << std::endl << "-- Outputs------------" << std::endl << std::endl;

		std::cout << "-- Dim     (" << lBoundS.x << "," << lBoundS.y << "," << lBoundS.z << "), (";
		std::cout << uBoundS.x << "," << uBoundS.y << "," << uBoundS.z << ")" << std::endl;
		std::cout << "-- max_ll " << max_ll << std::endl;
		std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;

		std::cout << std::endl << "----------------------" << std::endl << std::endl;

		std::cout << "particleId 	"; thrust::copy(particleId.begin(), particleId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "cubeID		"; thrust::copy(cubeId.begin(), cubeId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputX		"; thrust::copy(inputX.begin(), inputX.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputY		"; thrust::copy(inputY.begin(), inputY.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputZ		"; thrust::copy(inputZ.begin(), inputZ.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "startOfCube	"; thrust::copy(particleStartOfCubes.begin(), particleStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//		std::cout << "neighborsOfCubes	"; thrust::copy(neighborsOfCubes.begin(), neighborsOfCubes.begin()+13*numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;

		std::cout << "----------------------" << std::endl << std::endl;
	}

	// print edge details from device vectors
	void outputEdgeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << "numOfEdges		" << numOfEdges << std::endl << std::endl;
		std::cout << "edgeSizeOfCubes		"; thrust::copy(edgeSizeOfCubes.begin(), edgeSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << "edgeStartOfCubes	"; thrust::copy(edgeStartOfCubes.begin(), edgeStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
		std::cout << std::endl;

		for(int i=0; i<numOfEdges; i++)
			std::cout << "---- " << ((Edge)edges[i]).srcId << "," << ((Edge)edges[i]).desId << "," << ((Edge)edges[i]).weight <<  ")" << std::endl;
		std::cout << std::endl;

		std::cout << "----------------------" << std::endl << std::endl;
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
			std::cout << "(" << inputX[i] << "," << inputY[i] << "," << inputZ[i] << " : " << n.value << "," << n.nodeId << "," << n.haloId << "," << k << ")";

			while(n.parent!=NULL)
			{
				n = *(n.parent);
				k = (n.parent==NULL)?0:1;
				std::cout << "(" << n.value << "," << n.nodeId << "," << n.haloId << "," << k << ")";
			}
			std::cout << std::endl;
		}

		std::cout << std::endl;
		std::cout << "----------------------" << std::endl << std::endl;
	}



	//------- METHOD : parallel for each cube, get the set of edges & create the submerge tree. Globally combine them

	//---------------- METHOD - Local functions

	// locally, get the set of edges for each cube & create the submerge tree
	void localStepMethod()
	{
		initEdgeArrays();   // init arrays needed for storing edges		

		getEdgesPerCube();  // for each cube, get the set of edges, within the cube
		getNEdgesPerCube();	// get all sets of edges in neighboring cubes, for each cube

		#ifdef TEST
			outputEdgeDetails("After removing unecessary edges found in each cube..");	  // output edge details
			outputMergeTreeDetails("The sub merge trees.."); // output merge tree details
		#endif	
	}

	// for each cube, init arrays needed for storing edges
	void initEdgeArrays()
	{
		// for each cube, set the space required for storing edges
		edgeSizeOfCubes.resize(numOfCubes);
		edgeSizeOfCubesN.resize(numOfCubes);
		
		thrust::transform(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes,
		    particleSizeOfCubes.begin(), edgeSizeOfCubes.begin(), thrust::multiplies<int>());
    thrust::transform(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes,
        sizeOfNeighborCubes.begin(), edgeSizeOfCubesN.begin(), thrust::multiplies<int>());

		edgeStartOfCubesN.resize(numOfCubes);
		edgeStartOfCubes.resize(numOfCubes);
		thrust::exclusive_scan(edgeSizeOfCubes.begin(),  edgeSizeOfCubes.end(),  edgeStartOfCubes.begin());
		thrust::exclusive_scan(edgeSizeOfCubesN.begin(), edgeSizeOfCubesN.end(), edgeStartOfCubesN.begin());

    // init edge numbers
		numOfEdges  = thrust::reduce(edgeSizeOfCubes.begin(),  edgeSizeOfCubes.end());
		numOfEdgesN = thrust::reduce(edgeSizeOfCubesN.begin(), edgeSizeOfCubesN.end());

    // init edge arrays
		edges.resize(numOfEdges);
	}

	// for each cube, get the set of edges
	void getEdgesPerCube()
	{	
	  struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3, diff4, diff5;
    gettimeofday(&begin, 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				getEdges(thrust::raw_pointer_cast(&*neighborsOfCubes.begin()), 
								 thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
								 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*inputX.begin()),
								 thrust::raw_pointer_cast(&*inputY.begin()),
								 thrust::raw_pointer_cast(&*inputZ.begin()),
								 thrust::raw_pointer_cast(&*particleId.begin()),
								 max_ll, 0, numOfEdges,
								 thrust::raw_pointer_cast(&*edges.begin()),
								 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
								 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));	
    gettimeofday(&mid1, 0);
		int numOfEdgesOld = numOfEdges;
		removeEmptyEdges(numOfEdges, 0, edgeStartOfCubes, edgeSizeOfCubes); // remove empty items in edge sets, set new numOfEdges, sizeOfEdges & startOfEdges
    gettimeofday(&mid2, 0);
//		sortEdgesPerCube(); 			// for each cube, sort the set of edges by weight
//    gettimeofday(&mid3, 0);
//		sortCubeIDByParticleID(); // sort cube ids by particle ids
//    gettimeofday(&mid4, 0);
//		getSubMergeTreePerCube(); // for each cube, get the sub merge tree
//    gettimeofday(&end, 0);
//
//		sortParticlesByCubeID();

		edges.resize(numOfEdges+numOfEdgesN);
		thrust::device_vector<Edge> tmp(numOfEdgesOld-numOfEdges);
		thrust::copy(tmp.begin(), tmp.end(), edges.begin()+numOfEdges);


    timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << std::endl << std::flush;
    timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed2: " << seconds2 << std::endl << std::flush;
//    timersub(&mid3, &mid2, &diff3);
//    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
//    std::cout << "Time elapsed3: " << seconds3 << std::endl << std::flush;
//    timersub(&mid4, &mid3, &diff4);
//    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
//    std::cout << "Time elapsed4: " << seconds4 << std::endl << std::flush;
//    timersub(&end, &mid4, &diff5);
//    float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
//    std::cout << "Time elapsed5: " << seconds5 << std::endl << std::flush;
	}

	// for each cube, get the set of edges
	void getNEdgesPerCube()
	{	
	  struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3, diff4, diff5;
	  gettimeofday(&begin, 0);
		for(int i=1; i<=13; i++) 
		{
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
					getEdges(thrust::raw_pointer_cast(&*neighborsOfCubes.begin()), 
									 thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
									 thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
									 thrust::raw_pointer_cast(&*inputX.begin()),
									 thrust::raw_pointer_cast(&*inputY.begin()),
									 thrust::raw_pointer_cast(&*inputZ.begin()),
									 thrust::raw_pointer_cast(&*particleId.begin()),
									 max_ll, i, numOfEdges,
									 thrust::raw_pointer_cast(&*edges.begin()),
									 thrust::raw_pointer_cast(&*edgeSizeOfCubesN.begin()),
									 thrust::raw_pointer_cast(&*edgeStartOfCubesN.begin())));
		}
		gettimeofday(&mid1, 0);
		removeEmptyEdges(numOfEdgesN, numOfEdges, edgeStartOfCubesN, edgeSizeOfCubesN); // remove empty items in edge sets, set new numOfEdges, sizeOfEdges & startOfEdges
		gettimeofday(&mid2, 0);
		sortEdgesPerCubeSpecial(); // for each cube, sort the set of edges by weight
		gettimeofday(&mid3, 0);
		sortCubeIDByParticleID();  // sort cube ids by particle ids
		gettimeofday(&mid4, 0);
		getSubMergeTreePerCube();  // for each cube, get the sub merge tree
		gettimeofday(&end, 0);

		timersub(&mid1, &begin, &diff1);
    float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
    std::cout << "Time elapsed1: " << seconds1 << std::endl << std::flush;
    timersub(&mid2, &mid1, &diff2);
    float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
    std::cout << "Time elapsed2: " << seconds2 << std::endl << std::flush;
    timersub(&mid3, &mid2, &diff3);
    float seconds3 = diff3.tv_sec + 1.0E-6*diff3.tv_usec;
    std::cout << "Time elapsed3: " << seconds3 << std::endl << std::flush;
    timersub(&mid4, &mid3, &diff4);
    float seconds4 = diff4.tv_sec + 1.0E-6*diff4.tv_usec;
    std::cout << "Time elapsed4: " << seconds4 << std::endl << std::flush;
    timersub(&end, &mid4, &diff5);
    float seconds5 = diff5.tv_sec + 1.0E-6*diff5.tv_usec;
    std::cout << "Time elapsed5: " << seconds5 << std::endl << std::flush;
	}

	// for each cube, get the set of edges after comparing
	struct getEdges : public thrust::unary_function<int, void>
	{
		int    num, numOfEdges;
		float  max_ll;
		float *inputX, *inputY, *inputZ;

		int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;
		int   *neighborsOfCubes;
		
		Edge *edges;
		int  *edgeStartOfCubes, *edgeSizeOfCubes;

		__host__ __device__
		getEdges(int *neighborsOfCubes, int *particleStartOfCubes, int *particleSizeOfCubes, 
				float *inputX, float *inputY, float *inputZ,
				int *particleId, float max_ll, int num, int numOfEdges,
				Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes) :
				neighborsOfCubes(neighborsOfCubes), 
				particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
				inputX(inputX), inputY(inputY), inputZ(inputZ),
				particleId(particleId), max_ll(max_ll), num(num), numOfEdges(numOfEdges),
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

		 __host__ __device__
     void operator()(int i)
		 {
	     int offset = 0;
			 int cube = i;
			 if(num==0 || num==1)
				 edgeSizeOfCubes[i] = 0;
			 
			 if(num>0)
			 {
				 offset = numOfEdges;
				 cube = neighborsOfCubes[13*i+(num-1)];
			 }

			 if(cube==-1 || particleSizeOfCubes[i]==0 || particleSizeOfCubes[cube]==0) return;

			 // for each particle in cube
			 for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
			 {
				 float currentX = inputX[particleId[j]];
				 float currentY = inputY[particleId[j]];
				 float currentZ = inputZ[particleId[j]];

				 // compare with particles in this cube
				 int start = (num==0) ? j+1 : particleStartOfCubes[cube];
				 for(int k=start; k<particleStartOfCubes[cube]+particleSizeOfCubes[cube]; k++)
				 {
					float otherX = inputX[particleId[k]];
					float otherY = inputY[particleId[k]];
					float otherZ = inputZ[particleId[k]];

					float xd, yd, zd;
					xd = (currentX-otherX);  if (xd < 0.0f) xd = -xd;
					yd = (currentY-otherY);  if (yd < 0.0f) yd = -yd;
					zd = (currentZ-otherZ);  if (zd < 0.0f) zd = -zd;
			                                                                                                                                                         						
					if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
					{
						float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);
						if(dist <= max_ll)
						{
							int src = (particleId[j] <= particleId[k]) ? particleId[j] : particleId[k];
							int des = (src == particleId[k]) ? particleId[j] : particleId[k];
							                                                                                                                                                               
							// add edge
							edges[offset + edgeStartOfCubes[i] + edgeSizeOfCubes[i]] = Edge(src, des, dist);

							edgeSizeOfCubes[i]++;
						}
					}
				}				
			}
		}
	};

	// remove empty items in edge sets
	void removeEmptyEdges(int &numEdges, int offset, thrust::device_vector<int>& start, thrust::device_vector<int>& size)
	{
		thrust::device_vector<Edge>::iterator new_end;
		new_end = thrust::remove_if(edges.begin()+offset, edges.begin()+offset+numEdges, isEmpty());

		// get new number of edges
		numEdges = new_end - (edges.begin() + offset);	

		// for each cube, get new start of edges
		thrust::exclusive_scan(size.begin(), size.end(), start.begin());
	}

	// given a edge item, check whether its empty
  struct isEmpty : public thrust::unary_function<int, bool>
  {
    __host__ __device__
    bool operator()(Edge e)
    {
      return (e.srcId==-1 && e.desId==-1 && e.weight==-1);
    }
  };

	// for each cube, sort the set of edges by weight
	void sortEdgesPerCubeSpecial()
	{
		thrust::device_vector<int>   values;
		thrust::device_vector<float> keys;
		values.resize(numOfEdges+numOfEdgesN);
		keys.resize(numOfEdges+numOfEdgesN);

    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
        setValuesAndKeys(thrust::raw_pointer_cast(&*edges.begin()),
                 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
                 thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
                 max_ll, 0,
                 thrust::raw_pointer_cast(&*values.begin()),
                 thrust::raw_pointer_cast(&*keys.begin())));
    thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
        setValuesAndKeys(thrust::raw_pointer_cast(&*edges.begin()),
                 thrust::raw_pointer_cast(&*edgeSizeOfCubesN.begin()),
                 thrust::raw_pointer_cast(&*edgeStartOfCubesN.begin()),
                 max_ll, numOfEdges,
                 thrust::raw_pointer_cast(&*values.begin()),
                 thrust::raw_pointer_cast(&*keys.begin())));

	  thrust::stable_sort_by_key(keys.begin(), keys.end(), values.begin());

		thrust::device_vector<Edge> edgesTmp(numOfEdges+numOfEdgesN);
		thrust::gather(values.begin(), values.end(), edges.begin(), edgesTmp.begin());
		edges = edgesTmp;

		thrust::transform(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end(), edgeSizeOfCubesN.begin(), edgeSizeOfCubes.begin(), thrust::plus<int>());
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end(), edgeStartOfCubes.begin());
		numOfEdges += numOfEdgesN;
	}

	// for each cube, sort the set of edges by weight
	void sortEdgesPerCube()
	{
		thrust::device_vector<int>   values;
		thrust::device_vector<float> keys;
		values.resize(numOfEdges);
		keys.resize(numOfEdges);

		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				setValuesAndKeys(thrust::raw_pointer_cast(&*edges.begin()),
						  	 thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
							   thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
							   max_ll, 0,
                 thrust::raw_pointer_cast(&*values.begin()),
								 thrust::raw_pointer_cast(&*keys.begin())));

		thrust::stable_sort_by_key(keys.begin(), keys.end(), values.begin());

		thrust::device_vector<Edge> edgesTmp(numOfEdges);
		thrust::gather(values.begin(), values.end(), edges.begin(), edgesTmp.begin());
		edges = edgesTmp;
	}

	//*** set values & keys arrays needed for edge sorting
	struct setValuesAndKeys : public thrust::unary_function<int, void>
	{
		float max_ll;
		int   offset;

		float *keys;
		int   *values;

		Edge *edges;
		int  *edgeSizeOfCubes, *edgeStartOfCubes;

		__host__ __device__
		setValuesAndKeys(Edge *edges, int *edgeSizeOfCubes,int *edgeStartOfCubes,
				float max_ll, int offset, int *values, float *keys) :
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
				max_ll(max_ll), offset(offset), values(values), keys(keys) {}

		__host__ __device__
		void operator()(int i)
		{
			for(int j=offset+edgeStartOfCubes[i]; j<offset+edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
			{
				keys[j]   = (float) (edges[j].weight + 2*i*max_ll);
				values[j] = j;
			}
		}
	};

	// sort cube id by particle id
	void sortCubeIDByParticleID()
	{
		thrust::stable_sort_by_key(particleId.begin(), particleId.end(), cubeId.begin());
	}

	// for each cube, compute the sub merge trees
	void getSubMergeTreePerCube(bool consider=false)
	{
		// clear stuff
		nodes.clear();
		nodesTmp1.clear();
		nodesTmp2.clear();		

		//get start for nodeTmp2
		thrust::device_vector<int> nodesTmp2Start;
		nodesTmp2Start.resize(numOfCubes);
		thrust::exclusive_scan(sizeOfNeighborCubes.begin(), sizeOfNeighborCubes.begin()+numOfCubes, nodesTmp2Start.begin());

		nodes.resize(numOfParticles);
		nodesTmp1.resize(numOfEdges);
		nodesTmp2.resize(thrust::reduce(sizeOfNeighborCubes.begin(), sizeOfNeighborCubes.begin()+numOfCubes));

    thrust::transform(CountingIterator(0), CountingIterator(0)+numOfParticles, nodes.begin(),
        initNodes());

		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				getSubMergeTree(thrust::raw_pointer_cast(&*tmpArray.begin()),
						thrust::raw_pointer_cast(&*edges.begin()),
		    		thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
						thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
						thrust::raw_pointer_cast(&*cubeId.begin()),
						thrust::raw_pointer_cast(&*sizeOfNeighborCubes.begin()),
						thrust::raw_pointer_cast(&*nodesTmp2Start.begin()),
						thrust::raw_pointer_cast(&*nodesTmp1.begin()), 
						thrust::raw_pointer_cast(&*nodesTmp2.begin()),
						thrust::raw_pointer_cast(&*nodes.begin()),
						consider));

		removeEmptyEdges(numOfEdges, 0, edgeStartOfCubes, edgeSizeOfCubes);   // remove empty items in edge sets
	}

  // init the nodes array with node id & halo id
  struct initNodes : public thrust::unary_function<int, Node>
  {
    __host__ __device__
    Node operator()(int i)
    {
      return Node(i, 0.0f, i, NULL, NULL);
    }
  };

	// for a given cube, compute its sub merge tree
	struct getSubMergeTree : public thrust::unary_function<int, void>
	{
		bool  considerTmp;
		int  *tmp;

		Edge *edges;
		int  *edgeStartOfCubes, *edgeSizeOfCubes;

		int  *cubeId;

		Node *nodes, *nodesTmp1, *nodesTmp2;
		int  *nodesTmp2Start, *nodesTmp2Size;

		__host__ __device__
		getSubMergeTree(int *tmp, Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes,
				int *cubeId, int *nodesTmp2Size, int *nodesTmp2Start,
				Node *nodesTmp1, Node *nodesTmp2, Node *nodes, bool considerTmp=true) :
				tmp(tmp), edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
				cubeId(cubeId), nodesTmp2Size(nodesTmp2Size), nodesTmp2Start(nodesTmp2Start),
				nodesTmp1(nodesTmp1), nodesTmp2(nodesTmp2), nodes(nodes), considerTmp(considerTmp) {}

		__host__ __device__
		void operator()(int i)
		{
			if(considerTmp && tmp[i]==1)
      {
			  return;
      }

			int size = 0;
			int nxt  = 0;
			for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
			{
				Edge e = ((Edge)edges[j]);

				// get merge tree nodes to compare
				Node *tmp1, *tmp2, *tmp3, *tmp4;

				bool setTmp1 = false;
				bool setTmp2 = false;

				if(cubeId[e.srcId]==i)
				{
					tmp1    = &nodes[e.srcId];
					setTmp1 = true;
				}

				if(cubeId[e.desId]==i)
				{
					tmp2    = &nodes[e.desId];
					setTmp2 = true;
				}

				if(!setTmp1 || !setTmp2)
				{
					for(int k = nodesTmp2Start[i]; k<nodesTmp2Start[i]+nodesTmp2Size[i]; k++)
					{
						if(!setTmp1 && (nodesTmp2[k].nodeId==e.srcId || nodesTmp2[k].nodeId==-1))
						{
							setTmp1 = true;
							tmp1 = &nodesTmp2[k];
							tmp1->nodeId = e.srcId;
							tmp1->haloId = e.srcId;
							continue;
						}

						if(!setTmp2 && (nodesTmp2[k].nodeId==e.desId || nodesTmp2[k].nodeId==-1))
						{
							setTmp2 = true;
							tmp2 = &nodesTmp2[k];
							tmp2->nodeId = e.desId;
							tmp2->haloId = e.desId;
							continue;
						}

						if(setTmp1 && setTmp2)
							break;
					}
				}

				tmp3 = (tmp1->parentSuper==NULL) ? tmp1 : tmp1->parentSuper;
				tmp4 = (tmp2->parentSuper==NULL) ? tmp2 : tmp2->parentSuper;

				while(tmp3->parentSuper==NULL && tmp3->parent!=NULL)
					tmp3 = tmp3->parent;

				while(tmp4->parentSuper==NULL && tmp4->parent!=NULL)
					tmp4 = tmp4->parent;

				tmp1->parentSuper = tmp3;
				tmp2->parentSuper = tmp4;

				// if haloIds are different, connect them
				if(tmp3->haloId != tmp4->haloId)
				{
					size++;
					int minValue = (tmp3->haloId < tmp4->haloId) ? tmp3->haloId : tmp4->haloId;

					if(tmp3->value == e.weight)
					{
						tmp3->haloId	    = minValue;
						tmp4->parent 	    = tmp3;
						tmp2->parentSuper = tmp3;
					}
					else if(tmp4->value == e.weight)
					{
						tmp4->haloId      = minValue;
						tmp3->parent 	    = tmp4;
						tmp1->parentSuper = tmp4;
					}
					else
					{
						Node *n   = &nodesTmp1[edgeStartOfCubes[i]+nxt];
						n->value  = e.weight;
						n->haloId = minValue;

						tmp3->parent      = n;
						tmp4->parent 	    = n;
						tmp1->parentSuper = n;
						tmp2->parentSuper = n;

						nxt++;
					}
				}
				else
				{
          edges[j] = Edge();
				}
			}

			edgeSizeOfCubes[i] = size;
		}
	};



	//---------------- METHOD - Global functions

	// globally, combine the sub merge trees by combining two cubes at a time
	void globalStepMethod()
	{
	  thrust::device_vector<int> tmpCombined;
    thrust::device_vector<int> A, B;

		// set new number of cubes
		int numOfCubesOld = numOfCubes;
		numOfCubes = (int)std::ceil(((double)numOfCubes/2));

		if(numOfEdges==0) return;

		while(numOfCubes!=numOfCubesOld && numOfCubes>0)
		{
		  tmpCombined.resize(2*numOfCubes);
      thrust::merge(CountingIterator(0), CountingIterator(0)+numOfCubes, CountingIterator(0), CountingIterator(0)+numOfCubes, tmpCombined.begin());

			// set new cube Ids
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles, 
					setNewCubeID(thrust::raw_pointer_cast(&*cubeId.begin())));

			#ifdef TEST
				outputCubeDetails("The new cube details.."); // output cube details
			#endif

			tmpArray.resize(numOfCubes);
      thrust::transform(CountingIterator(0), CountingIterator(0)+numOfCubes, tmpArray.begin(),
           setCount(thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()), numOfCubesOld));

			// set tmp arrays
			A.resize(numOfCubes);
			B.resize(numOfCubes);

			//set new sizeOfEdges
			thrust::reduce_by_key(tmpCombined.begin(), tmpCombined.begin()+numOfCubesOld, edgeSizeOfCubes.begin(), A.begin(), B.begin());
			edgeSizeOfCubes = B;

			// set new startOfEdges
			thrust::unique_by_key_copy(tmpCombined.begin(), tmpCombined.begin()+numOfCubesOld, edgeStartOfCubes.begin(), A.begin(), B.begin());
      edgeStartOfCubes = B;

			// set new sizeOfNeighborCubes
			thrust::reduce_by_key(tmpCombined.begin(), tmpCombined.begin()+numOfCubesOld, sizeOfNeighborCubes.begin(), A.begin(), B.begin());
			sizeOfNeighborCubes = B;

			#ifdef TEST
				outputEdgeDetails("The edge details for new cubes. ."); // output edge details
			#endif

			// for each new cube, sort new sets of edges
			sortEdgesPerCube();

			#ifdef TEST
				outputEdgeDetails("After sorting edges for new cubes. ."); // output edge details
			#endif

			// for each new cube, get the sub merge tree
			getSubMergeTreePerCube(true);

			#ifdef TEST
				outputMergeTreeDetails("The new sub merge trees.."); // output merge tree details
			#endif

			// set new number of cubes
			numOfCubesOld = numOfCubes;
			numOfCubes = (int)std::ceil(((double)numOfCubes/2));
		}
	}

	// for each cube, set new cube Ids
	struct setNewCubeID
	{
		int *cubeId;

		__host__ __device__
		setNewCubeID(int *cubeId) : cubeId(cubeId) {}

		__host__ __device__
		void operator()(int i)
		{
			cubeId[i] = (cubeId[i]%2 == 0) ? cubeId[i]/2 : (cubeId[i]-1)/2;
		}
	};

  struct setCount : public thrust::unary_function<int, int>
  {
    int *size;
    int num;

    __host__ __device__
    setCount(int *size, int num) :
      size(size), num(num) {}

    __host__ __device__
    int operator()(int i)
    {
      if(2*i   < num && size[2*i] == 0)   return 1;
      if(2*i+1 < num && size[2*i+1] == 0) return 1;

      return 0;
    }
  };
};

}

#endif
