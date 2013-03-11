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

	thrust::device_vector<int> particleId; // for each particle, particle id
	thrust::device_vector<int> cubeId;	   // for each particle, cube id
	thrust::device_vector<int> haloId;	   // for each particle, halo id

	float max_ll;   		// maximum linking length
	Point lBoundS, uBoundS; // lower & upper bounds of the entire space

	int numOfCubes;					   	  // total number of cubes in space
	int cubesInX, cubesInY, cubesInZ;     // number of cubes in each dimension

	thrust::device_vector<Point> lBoundOfCubes; 		// lower bounds of cubes
	thrust::device_vector<Point> uBoundOfCubes; 		// upper bounds of cubes
	thrust::device_vector<int> 	 particleSizeOfCubes; 	// number of particles in cubes
	thrust::device_vector<int> 	 particleStartOfCubes;	// stratInd of cubes  (within particleId)

	thrust::device_vector<int>   neighborsOfCubes;		// neighbors of cubes (for each cube, store all 26 sequentially)
	thrust::device_vector<int>   sizeOfNeighborCubes;	// neighbors of cubes (for each cube, store all 26 sequentially)
	
	thrust::device_vector<Node>  nodes, nodesTmp1, nodesTmp2;

	int numOfEdges;
	thrust::device_vector<Edge>  edges;
	thrust::device_vector<int>   edgeSizeOfCubes;
	thrust::device_vector<int>   edgeStartOfCubes;

	halo_merge(float max_linkLength, std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1, bool periodic=false) : halo(filename, format, n, np, rL, periodic)
	{
		if(numOfParticles!=0)
		{
			struct timeval begin, mid1, mid2, mid3, mid4, end, diff1, diff2, diff3;

			//---- init stuff

			max_ll = max_linkLength;  // get max_linkinglength

			initParticleIds();	      // set particle ids
			initHaloIds(); 	  	      // set halo ids
			initCubeIds(); 	          // set cube ids
			getBounds();		      // get bounds of the entire space

			//---- divide space into cubes (TODO: need to change this like in kdtree)

			gettimeofday(&begin, 0);
			setNumberOfCubes();	      // get total number of cubes
			divideIntoCubes();	      // divide space into cubes
			setCubeIds();		      // for each particle, set cube id
			sortParticlesByCubeID();  // sort Particles by cube Id
			getSizeOfCubes();  		  // for each cube, count particles
			getStartOfCubes();	      // for each cube, get the start of cube particles (in particleId array)
		        getNeighborsOfCubes();    // for each cube, get its neighbors
			getSizeofNeighborCubes(); // for each cube, count the particle size in its neighbors
			gettimeofday(&mid1, 0);

			lBoundOfCubes.clear();
			uBoundOfCubes.clear();

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

			gettimeofday(&mid4, 0);
			globalStepMethod();
			gettimeofday(&end, 0);

			edges.clear();
			sizeOfNeighborCubes.clear();
			edgeSizeOfCubes.clear();
			edgeStartOfCubes.clear();


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
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setHaloId(thrust::raw_pointer_cast(&*nodes.begin()),
						  thrust::raw_pointer_cast(&*haloIndex.begin()),
						  linkLength));
	}

	// for a given node, set its halo id
	struct setHaloId : public thrust::unary_function<int, void>
	{
		Node *nodes;
		int *haloIndex;
		float linkLength;

		__host__ __device__
		setHaloId(Node *nodes, int *haloIndex, float linkLength) :
			nodes(nodes), haloIndex(haloIndex), linkLength(linkLength) {}

		__host__ __device__
		void operator()(int i)
		{
			Node n = ((Node)nodes[i]);
			if(n.nodeId == -1 && n.haloId == -1)
			{
				n.nodeId = i;
				n.haloId = i;
			}

			Node nChild;
			while(n.parent!=NULL)
			{
				nChild = n;
				n = *(n.parent);

				if(n.value >= linkLength)
				{
					n = nChild;
					break;
				}
			}

			haloIndex[i] = n.haloId;
		}
	};



	//------- init stuff

        // set initial particle ids
	void initParticleIds()
	{
		particleId.resize(numOfParticles);
		thrust::sequence(particleId.begin(), particleId.end());
	}

	// set initial halo ids
	void initHaloIds()
	{
		haloId.resize(numOfParticles);
		thrust::sequence(haloId.begin(), haloId.end());
	}

	// set initial cube ids
	void initCubeIds()
	{
		cubeId.resize(numOfParticles);
		thrust::fill(cubeId.begin(), cubeId.end(), -1);
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

	// get bound details for each cube
	void divideIntoCubes()
	{
		lBoundOfCubes.resize(numOfCubes);
		uBoundOfCubes.resize(numOfCubes);

		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				setCubeBounds(thrust::raw_pointer_cast(&*lBoundOfCubes.begin()),
						      thrust::raw_pointer_cast(&*uBoundOfCubes.begin()),
						      max_ll, cubesInX, cubesInY, lBoundS, uBoundS));
	}

	// for each cube, set its bounds
	struct setCubeBounds : public thrust::unary_function<int, void>
	{
		float max_ll;
		int cubesInX, cubesInY;

		Point *lBoundOfCubes, *uBoundOfCubes;
		Point lBoundS, uBoundS;

		__host__ __device__
		setCubeBounds(Point *lBoundOfCubes, Point *uBoundOfCubes, float max_ll,
			int cubesInX, int cubesInY, Point lBoundS, Point uBoundS) :
			lBoundOfCubes(lBoundOfCubes), uBoundOfCubes(uBoundOfCubes), max_ll(max_ll),
			cubesInX(cubesInX), cubesInY(cubesInY), lBoundS(lBoundS), uBoundS(uBoundS) {}

		__host__ __device__
		void operator()(int i)
		{
			int tmp = i % (cubesInX*cubesInY);

			// get x,y,z coordinates for the cube
			int z = i / (cubesInX*cubesInY);
			int y = tmp / cubesInX;
			int x = tmp % cubesInX;

			lBoundOfCubes[i].x = (lBoundS.x + max_ll*x <= uBoundS.x) ? lBoundS.x + max_ll*x : uBoundS.x;
			lBoundOfCubes[i].y = (lBoundS.y + max_ll*y <= uBoundS.y) ? lBoundS.y + max_ll*y : uBoundS.y;
			lBoundOfCubes[i].z = (lBoundS.z + max_ll*z <= uBoundS.x) ? lBoundS.z + max_ll*z : uBoundS.z;

			uBoundOfCubes[i].x = (lBoundOfCubes[i].x+max_ll <= uBoundS.x) ? lBoundOfCubes[i].x+max_ll : uBoundS.x;
			uBoundOfCubes[i].y = (lBoundOfCubes[i].y+max_ll <= uBoundS.y) ? lBoundOfCubes[i].y+max_ll : uBoundS.y;
			uBoundOfCubes[i].z = (lBoundOfCubes[i].z+max_ll <= uBoundS.z) ? lBoundOfCubes[i].z+max_ll : uBoundS.z;
		}
	};

	// set cube ids of particles
	void setCubeIds()
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				setCubeIdOfParticle(thrust::raw_pointer_cast(&*inputX.begin()),
									thrust::raw_pointer_cast(&*inputY.begin()),
									thrust::raw_pointer_cast(&*inputZ.begin()),
									thrust::raw_pointer_cast(&*cubeId.begin()),
									max_ll, lBoundS, cubesInX, cubesInY));
	}

	// for a given particle, set its cube id
	struct setCubeIdOfParticle : public thrust::unary_function<int, void>
	{
		float  max_ll;

		Point  lBoundS;
		int   *cubeId;
		int    cubesInX, cubesInY;

		float *inputX, *inputY, *inputZ;

		__host__ __device__
		setCubeIdOfParticle(float *inputX, float *inputY, float *inputZ,
			int *cubeId, float max_ll, Point lBoundS,
			int cubesInX, int cubesInY) :
			inputX(inputX), inputY(inputY), inputZ(inputZ),
			cubeId(cubeId), max_ll(max_ll), lBoundS(lBoundS),
			cubesInX(cubesInX), cubesInY(cubesInY){}

		__host__ __device__
		void operator()(int i)
		{
			// get x,y,z coordinates for the cube
			int z = (inputZ[i]-lBoundS.z) / max_ll;
			int y = (inputY[i]-lBoundS.y) / max_ll;
			int x = (inputX[i]-lBoundS.x) / max_ll;

			// get cube id
			int id = z*(cubesInX*cubesInY) + y*cubesInX + x;
			cubeId[i] = id;
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
		thrust::device_vector<int> tmpC; tmpC.resize(numOfParticles);
		thrust::device_vector<int> tmpD; tmpD.resize(numOfParticles);

		thrust::copy(cubeId.begin(), cubeId.end(), tmpA.begin());
		thrust::stable_sort(tmpA.begin(), tmpA.end());
		thrust::fill(tmpB.begin(), tmpB.end(), 1);

		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;
		new_end = thrust::reduce_by_key(tmpA.begin(), tmpA.end(), tmpB.begin(), tmpC.begin(), tmpD.begin());

		particleSizeOfCubes.resize(numOfCubes);
		thrust::fill(particleSizeOfCubes.begin(), particleSizeOfCubes.end(), 0);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+(thrust::get<0>(new_end)-tmpC.begin()),
				setNumberOfparticlesForCube(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
										    thrust::raw_pointer_cast(&*tmpC.begin()),
										    thrust::raw_pointer_cast(&*tmpD.begin())));
	}

	// for a given cube, set the number of particles
	struct setNumberOfparticlesForCube : public thrust::unary_function<int, void>
	{
		int *tmp1, *tmp2, *particlesOfCube;

		__host__ __device__
		setNumberOfparticlesForCube(int *particlesOfCube, int *tmp1, int *tmp2) :
			particlesOfCube(particlesOfCube), tmp1(tmp1), tmp2(tmp2) {}

		__host__ __device__
		void operator()(int i)
		{
			particlesOfCube[tmp1[i]] = tmp2[i];
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
		neighborsOfCubes.resize(26*numOfCubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				setNeighboringCubes(numOfCubes,
									cubesInX, cubesInY,
									lBoundS,  uBoundS,
									thrust::raw_pointer_cast(&*neighborsOfCubes.begin()),
									thrust::raw_pointer_cast(&*lBoundOfCubes.begin()),
									thrust::raw_pointer_cast(&*uBoundOfCubes.begin())));
	}

	//*** for a given cube where center is (x,y,z), get its neighboring cubes
	struct setNeighboringCubes : public thrust::unary_function<int, void>
	{
		int  numOfCubes;
		int  cubesInX, cubesInY;
		int *neighborsOfCubes;

		Point  lBoundS, uBoundS;
		Point *lBoundOfCubes, *uBoundOfCubes;

		__host__ __device__
		setNeighboringCubes(int numOfCubes, int cubesInX, int cubesInY,
				Point  lBoundS, Point uBoundS, int *neighborsOfCubes,
				Point *lBoundOfCubes, Point *uBoundOfCubes) :
				numOfCubes(numOfCubes), cubesInX(cubesInX), cubesInY(cubesInY),
				lBoundS(lBoundS), uBoundS(uBoundS), neighborsOfCubes(neighborsOfCubes),
				lBoundOfCubes(lBoundOfCubes), uBoundOfCubes(uBoundOfCubes) {}

		__host__ __device__
		void operator()(int cubeId)
		{
			int startInd = cubeId*26;

			Point p1 = lBoundOfCubes[cubeId];
			Point p2 = uBoundOfCubes[cubeId];

			neighborsOfCubes[startInd+0] = ((cubeId-1)>=0 && p1.x!=lBoundS.x) ? cubeId-1 : -1;		   // x-1,y,z
			neighborsOfCubes[startInd+1] = ((cubeId+1)<numOfCubes && p2.x!=uBoundS.x) ? cubeId+1 : -1; // x+1,y,z

			neighborsOfCubes[startInd+2] = ((cubeId-cubesInX)>=0 && p1.y!=lBoundS.y) ? cubeId-cubesInX : -1;		 // x,y-1,z
			neighborsOfCubes[startInd+3] = ((cubeId+cubesInX)<numOfCubes && p2.y!=uBoundS.y) ? cubeId+cubesInX : -1; // x,y+1,z

			neighborsOfCubes[startInd+4] = ((cubeId-(cubesInX*cubesInY))>=0 && p1.z!=lBoundS.z) ? cubeId-(cubesInX*cubesInY) : -1;		  // x,y,z-1
			neighborsOfCubes[startInd+5] = ((cubeId+(cubesInX*cubesInY))<numOfCubes && p2.z!=uBoundS.z)? cubeId+(cubesInX*cubesInY) : -1; // x,y,z+1

			neighborsOfCubes[startInd+6] = (neighborsOfCubes[startInd+0]!=-1) ? ((cubeId-1-cubesInX)>=0 ? cubeId-1-cubesInX : -1) : -1;		    // x-1,y-1,z
			neighborsOfCubes[startInd+7] = (neighborsOfCubes[startInd+0]!=-1) ? ((cubeId-1+cubesInX)<numOfCubes ? cubeId-1+cubesInX : -1) : -1; // x-1,y+1,z
			neighborsOfCubes[startInd+8] = (neighborsOfCubes[startInd+1]!=-1) ? ((cubeId+1-cubesInX)>=0 ? cubeId+1-cubesInX : -1) : -1;		    // x+1,y-1,z
			neighborsOfCubes[startInd+9] = (neighborsOfCubes[startInd+1]!=-1) ? ((cubeId+1+cubesInX)<numOfCubes ? cubeId+1+cubesInX : -1) : -1; // x+1,y+1,z

			neighborsOfCubes[startInd+10] = (neighborsOfCubes[startInd+0]!=-1) ? ((cubeId-1-(cubesInX*cubesInY))>=0 ? cubeId-1-(cubesInX*cubesInY) : -1) : -1;		   // x-1,y,z-1
			neighborsOfCubes[startInd+11] = (neighborsOfCubes[startInd+0]!=-1) ? ((cubeId-1+(cubesInX*cubesInY))<numOfCubes ? cubeId-1+(cubesInX*cubesInY) : -1) : -1; // x-1,y,z+1
			neighborsOfCubes[startInd+12] = (neighborsOfCubes[startInd+1]!=-1) ? ((cubeId+1-(cubesInX*cubesInY))>=0 ? cubeId+1-(cubesInX*cubesInY) : -1) : -1;		   // x+1,y,z-1
			neighborsOfCubes[startInd+13] = (neighborsOfCubes[startInd+1]!=-1) ? ((cubeId+1+(cubesInX*cubesInY))<numOfCubes ? cubeId+1+(cubesInX*cubesInY) : -1) : -1; // x+1,y,z+1

			neighborsOfCubes[startInd+14] = (neighborsOfCubes[startInd+2]!=-1) ? ((cubeId-cubesInX-(cubesInX*cubesInY))>=0 ? cubeId-cubesInX-(cubesInX*cubesInY) : -1) : -1;			 // x,y-1,z-1
			neighborsOfCubes[startInd+15] = (neighborsOfCubes[startInd+2]!=-1) ? ((cubeId-cubesInX+(cubesInX*cubesInY))<numOfCubes ? cubeId-cubesInX+(cubesInX*cubesInY) : -1) : -1;	 // x,y-1,z+1
			neighborsOfCubes[startInd+16] = (neighborsOfCubes[startInd+3]!=-1) ? ((cubeId+cubesInX-(cubesInX*cubesInY))>=0 ? cubeId+cubesInX-(cubesInX*cubesInY) : -1) : -1;			 // x,y+1,z-1
			neighborsOfCubes[startInd+17] = (neighborsOfCubes[startInd+3]!=-1) ? ((cubeId+cubesInX+(cubesInX*cubesInY))<numOfCubes ? cubeId+cubesInX+(cubesInX*cubesInY) : -1) : -1;	 // x,y+1,z+1

			neighborsOfCubes[startInd+18] = (neighborsOfCubes[startInd+6]!=-1) ? ((cubeId-1-cubesInX-(cubesInX*cubesInY))>=0 ? cubeId-1-cubesInX-(cubesInX*cubesInY) : -1) : -1;	     // x-1,y-1,z-1
			neighborsOfCubes[startInd+19] = (neighborsOfCubes[startInd+6]!=-1) ? ((cubeId-1-cubesInX+(cubesInX*cubesInY))<numOfCubes ? cubeId-1-cubesInX+(cubesInX*cubesInY) : -1) : -1; // x-1,y-1,z+1
			neighborsOfCubes[startInd+20] = (neighborsOfCubes[startInd+7]!=-1) ? ((cubeId-1+cubesInX-(cubesInX*cubesInY))>=0 ? cubeId-1+cubesInX-(cubesInX*cubesInY) : -1) : -1;		 // x-1,y+1,z-1
			neighborsOfCubes[startInd+21] = (neighborsOfCubes[startInd+7]!=-1) ? ((cubeId-1+cubesInX+(cubesInX*cubesInY))<numOfCubes ? cubeId-1+cubesInX+(cubesInX*cubesInY) : -1) : -1; // x-1,y+1,z+1
			neighborsOfCubes[startInd+22] = (neighborsOfCubes[startInd+8]!=-1) ? ((cubeId+1-cubesInX-(cubesInX*cubesInY))>=0 ? cubeId+1-cubesInX-(cubesInX*cubesInY) : -1) : -1;		 // x+1,y-1,z-1
			neighborsOfCubes[startInd+23] = (neighborsOfCubes[startInd+8]!=-1) ? ((cubeId+1-cubesInX+(cubesInX*cubesInY))<numOfCubes ? cubeId+1-cubesInX+(cubesInX*cubesInY) : -1) : -1; // x+1,y-1,z+1
			neighborsOfCubes[startInd+24] = (neighborsOfCubes[startInd+9]!=-1) ? ((cubeId+1+cubesInX-(cubesInX*cubesInY))>=0 ? cubeId+1+cubesInX-(cubesInX*cubesInY) : -1) : -1;		 // x+1,y+1,z-1
			neighborsOfCubes[startInd+25] = (neighborsOfCubes[startInd+9]!=-1) ? ((cubeId+1+cubesInX+(cubesInX*cubesInY))<numOfCubes ? cubeId+1+cubesInX+(cubesInX*cubesInY) : -1) : -1; // x+1,y+1,z+1
		}
	};

	// for each cube, count- the size of particle size in its neighbors 
	void getSizeofNeighborCubes()
	{
		sizeOfNeighborCubes.resize(numOfCubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
			countNeighborParticles(thrust::raw_pointer_cast(&*sizeOfNeighborCubes.begin()),
								   thrust::raw_pointer_cast(&*neighborsOfCubes.begin()),
								   thrust::raw_pointer_cast(&*particleSizeOfCubes.begin())));
	}

	//*** for a given cube, count particle size in its neighbors
	struct countNeighborParticles
	{
		int *sizeOfNeighborCubes, *neighborsOfCubes, *particleSizeOfCubes;

		__host__ __device__
		countNeighborParticles(int *sizeOfNeighborCubes, int *neighborsOfCubes, int *particleSizeOfCubes) :
				sizeOfNeighborCubes(sizeOfNeighborCubes), neighborsOfCubes(neighborsOfCubes), particleSizeOfCubes(particleSizeOfCubes) {}

		__host__ __device__
		void operator()(int cubeId)
		{
			int startInd = cubeId*26;

			int sum = 0;
			for(int i=0; i<26; i++)
				sum += particleSizeOfCubes[neighborsOfCubes[startInd+i]];

			sizeOfNeighborCubes[cubeId] = sum;
		}
	};



	//------- output results

	// print cube details from device vectors
	void outputCubeDetails(std::string title)
	{
		std::cout << title << std::endl << std::endl;

		std::cout << std::endl << "-- Outputs------------" << std::endl << std::endl;

		std::cout << "-- Dim    (" << lBoundS.x << "," << lBoundS.y << "," << lBoundS.z << "), (";
		std::cout << uBoundS.x << "," << uBoundS.y << "," << uBoundS.z << ")" << std::endl;
		std::cout << "-- max_ll " << max_ll << std::endl;

		std::cout << std::endl << "----------------------" << std::endl << std::endl;

		std::cout << "-- Cubes  " << numOfCubes << " : (" << cubesInX << "*" << cubesInY << "*" << cubesInZ << ")" << std::endl;
		for(int i=0; i<numOfCubes; i++)
		{
			Point p1 = lBoundOfCubes[i];
			Point p2 = uBoundOfCubes[i];
			std::cout << "---- " << i << " (" << p1.x << "," << p1.y << "," << p1.z << ") , (";
			std::cout << p2.x << "," << p2.y << "," << p2.z << ")"<< std::endl;
		}

		std::cout << std::endl << "----------------------" << std::endl << std::endl;

		std::cout << "particleId 	"; thrust::copy(particleId.begin(), particleId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "cubeID		"; thrust::copy(cubeId.begin(), cubeId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "haloID		"; thrust::copy(haloId.begin(), haloId.begin()+numOfParticles, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputX		"; thrust::copy(inputX.begin(), inputX.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputY		"; thrust::copy(inputY.begin(), inputY.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "inputZ		"; thrust::copy(inputZ.begin(), inputZ.begin()+numOfParticles, std::ostream_iterator<float>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "sizeOfCube	"; thrust::copy(particleSizeOfCubes.begin(), particleSizeOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;
		std::cout << "startOfCube	"; thrust::copy(particleStartOfCubes.begin(), particleStartOfCubes.begin()+numOfCubes, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl << std::endl;
//		std::cout << "neighborsOfCubes	"; thrust::copy(neighborsOfCubes.begin(), neighborsOfCubes.begin()+26*numOfCubes, std::ostream_iterator<int>(std::cout, " "));   std::cout << std::endl << std::endl;

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
		{
			Edge e = edges[i];
			std::cout << "---- " << e.srcId << "," << e.desId << "," << e.weight <<  ")" << std::endl;
		}

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
		getEdgeSetsOfCubes();	  // get all sets of edges

		#ifdef TEST
			outputEdgeDetails("Edges found for each cube..");	  // output edge details
		#endif

		getSubMergeTreePerCube(); // for each cube, get the sub merge tree

		#ifdef TEST
			outputEdgeDetails("After removing unecessary edges found in each cube..");	  // output edge details
			outputMergeTreeDetails("The sub merge trees.."); // output merge tree details
		#endif
	}

	// for each cube, get the set of edges
	void getEdgeSetsOfCubes()
	{
		initEdgeArrays();  // init arrays needed for storing edges

		getEdgesPerCube(); // for each cube, get the set of edges
	}

	// for each cube, init arrays needed for storing edges
	void initEdgeArrays()
	{
		// for each cube, set the space required for storing edges
		edgeSizeOfCubes.resize(numOfCubes);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				getSpaceRequired(thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
						 	     thrust::raw_pointer_cast(&*sizeOfNeighborCubes.begin()),
						 	     thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin())));

		edgeStartOfCubes.resize(numOfCubes);
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end(), edgeStartOfCubes.begin());

		// init edge arrays
		edges.resize(thrust::reduce(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end()));
	}

	//*** for each cube, get the space required for storing edges
	struct getSpaceRequired : public thrust::unary_function<int, void>
	{
		int *particleSizeOfCubes, *sizeOfNeighborCubes;
		int *size;

		__host__ __device__
		getSpaceRequired(int *particleSizeOfCubes, int *sizeOfNeighborCubes, int *size) :
			particleSizeOfCubes(particleSizeOfCubes), sizeOfNeighborCubes(sizeOfNeighborCubes), size(size) {}

		__host__ __device__
		void operator()(int i)
		{
			size[i] = (particleSizeOfCubes[i] + sizeOfNeighborCubes[i])*particleSizeOfCubes[i];
		}
	};

	// for each cube, get the set of edges
	void getEdgesPerCube()
	{
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				getSetsOfEdges(thrust::raw_pointer_cast(&*particleStartOfCubes.begin()),
							   thrust::raw_pointer_cast(&*particleSizeOfCubes.begin()),
							   thrust::raw_pointer_cast(&*neighborsOfCubes.begin()),
							   thrust::raw_pointer_cast(&*inputX.begin()),
							   thrust::raw_pointer_cast(&*inputY.begin()),
							   thrust::raw_pointer_cast(&*inputZ.begin()),
							   thrust::raw_pointer_cast(&*particleId.begin()),
							   max_ll,
							   thrust::raw_pointer_cast(&*edges.begin()),
							   thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
							   thrust::raw_pointer_cast(&*edgeStartOfCubes.begin())));

		removeEmptyEdges(); // remove empty items in edge sets, set the new numOfEdges, sizeOfEdges & startOfEdges
	}

	//*** for each cube, get the set of edges after comparing
	struct getSetsOfEdges : public thrust::unary_function<int, void>
	{
		float  max_ll;
		float *inputX, *inputY, *inputZ;

		int   *particleId, *particleStartOfCubes, *particleSizeOfCubes;
		int   *neighborsOfCubes;

		Edge *edges;
		int  *edgeStartOfCubes, *edgeSizeOfCubes;

		__host__ __device__
		getSetsOfEdges(int *particleStartOfCubes, int *particleSizeOfCubes, int *neighborsOfCubes,
				float *inputX, float *inputY, float *inputZ,
				int *particleId, float max_ll,
				Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes) :
				particleStartOfCubes(particleStartOfCubes), particleSizeOfCubes(particleSizeOfCubes),
				neighborsOfCubes(neighborsOfCubes),
				inputX(inputX), inputY(inputY), inputZ(inputZ),
				particleId(particleId), max_ll(max_ll),
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes) {}

		__host__ __device__
		void operator()(int i)
		{
			edgeSizeOfCubes[i] = 0;

			// for each particle in cube
			for(int j=particleStartOfCubes[i]; j<particleStartOfCubes[i]+particleSizeOfCubes[i]; j++)
			{
				float currentX = inputX[particleId[j]];
				float currentY = inputY[particleId[j]];
				float currentZ = inputZ[particleId[j]];

				// compair with particles in this cube
				for(int k=j+1; k<particleStartOfCubes[i]+particleSizeOfCubes[i]; k++)
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

							//add edge
							edges[edgeStartOfCubes[i] + edgeSizeOfCubes[i]] = Edge(src, des, dist);
							edgeSizeOfCubes[i]++;
						}
					}
				}
				
				//compair with particles in neighboring cubes
				int startInd = i*26;
				for(int k=0; k<26; k++)
				{
					int nCube = neighborsOfCubes[startInd+k];

					if(nCube==-1 || particleSizeOfCubes[nCube]==0) continue;

					for(int l=particleStartOfCubes[nCube]; l<particleStartOfCubes[nCube]+particleSizeOfCubes[nCube]; l++)
					{
						float otherX = inputX[particleId[l]];
						float otherY = inputY[particleId[l]];
						float otherZ = inputZ[particleId[l]];

						float xd, yd, zd;
						xd = (currentX-otherX);  if (xd < 0.0f) xd = -xd;
						yd = (currentY-otherY);  if (yd < 0.0f) yd = -yd;
						zd = (currentZ-otherZ);  if (zd < 0.0f) zd = -zd;

						if(xd<=max_ll && yd<=max_ll && zd<=max_ll)
						{
							float dist = (float)std::sqrt(xd*xd + yd*yd + zd*zd);
							if(dist <= max_ll)
							{
								int src = (particleId[j] <= particleId[l]) ? particleId[j] : particleId[l];
								int des = (src == particleId[l]) ? particleId[j] : particleId[l];

								// add edge
								edges[edgeStartOfCubes[i] + edgeSizeOfCubes[i]] = Edge(src, des, dist);
								edgeSizeOfCubes[i]++;
							}
						}
					}
				}				
			}
		}
	};

	// remove empty items in edge sets
	void removeEmptyEdges()
	{
		thrust::device_vector<Edge>::iterator new_end;
		new_end = thrust::remove_if(edges.begin(), edges.end(), isEmpty());

		// get new number of edges
		numOfEdges = new_end - edges.begin();

		// for each cube, get new start of edges
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end(), edgeStartOfCubes.begin());
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

	// for each cube, compute the sub merge trees
	void getSubMergeTreePerCube()
	{
		// clear stuff
		nodes.clear();
		nodesTmp1.clear();
		nodesTmp2.clear();

		sortEdgesPerCube();       // for each cube, sort the set of edges by weight

		sortCubeIDByParticleID(); // sort cube ids by particle ids

		//get start and size for nodeTmp2
		thrust::device_vector<int> nodesTmp2Size;
		thrust::device_vector<int> nodesTmp2Start;
		nodesTmp2Size.resize(numOfCubes);
		nodesTmp2Start.resize(numOfCubes);
		thrust::copy(sizeOfNeighborCubes.begin(), sizeOfNeighborCubes.begin()+numOfCubes, nodesTmp2Size.begin());
		thrust::exclusive_scan(nodesTmp2Size.begin(), nodesTmp2Size.end(), nodesTmp2Start.begin());

		nodes.resize(numOfParticles);
		nodesTmp1.resize(numOfEdges);
		nodesTmp2.resize(thrust::reduce(nodesTmp2Size.begin(), nodesTmp2Size.end()));

		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				initNodes(thrust::raw_pointer_cast(&*nodes.begin())));

		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				getSubMergeTree(thrust::raw_pointer_cast(&*edges.begin()),
							    thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
							    thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
							    thrust::raw_pointer_cast(&*cubeId.begin()),
							    thrust::raw_pointer_cast(&*nodesTmp2Size.begin()),
							    thrust::raw_pointer_cast(&*nodesTmp2Start.begin()),
							    thrust::raw_pointer_cast(&*nodesTmp1.begin()), 
							    thrust::raw_pointer_cast(&*nodesTmp2.begin()),
							    thrust::raw_pointer_cast(&*nodes.begin())));

		removeEmptyEdges();   // remove empty items in edge sets
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
							     max_ll,
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

		int   *values;
		float *keys;

		Edge *edges;
		int  *edgeSizeOfCubes, *edgeStartOfCubes;

		__host__ __device__
		setValuesAndKeys(Edge *edges, int *edgeSizeOfCubes,int *edgeStartOfCubes,
				float max_ll, int *values, float *keys) :
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
				max_ll(max_ll), values(values), keys(keys) {}

		__host__ __device__
		void operator()(int i)
		{
			for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
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


	// init the nodes array with node id & halo id
        struct initNodes
        {
		Node *nodes;

		__host__ __device__
		initNodes(Node * nodes) : nodes(nodes) {}

                __host__ __device__
                void operator()(int i)
                {
			nodes[i].nodeId = i;
	        	nodes[i].haloId = i;
	        }
 	};

	//*** for a given cube, compute its sub merge tree
	struct getSubMergeTree : public thrust::unary_function<int, void>
	{
		Edge *edges;
		int  *edgeStartOfCubes, *edgeSizeOfCubes;

		int  *cubeId;

		Node *nodes, *nodesTmp1, *nodesTmp2;
		int  *nodesTmp2Start, *nodesTmp2Size;

		__host__ __device__
		getSubMergeTree(Edge *edges, int *edgeSizeOfCubes, int *edgeStartOfCubes,
				int *cubeId, int *nodesTmp2Size, int *nodesTmp2Start,
				Node *nodesTmp1, Node *nodesTmp2, Node *nodes) :
				edges(edges), edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes),
				cubeId(cubeId), nodesTmp2Size(nodesTmp2Size), nodesTmp2Start(nodesTmp2Start),
				nodesTmp1(nodesTmp1), nodesTmp2(nodesTmp2), nodes(nodes) {}

		__host__ __device__
		void operator()(int i)
		{
			int size = 0;

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
						tmp3->haloId	  = minValue;
						tmp4->parent 	  = tmp3;
						tmp2->parentSuper = tmp3;
					}
					else if(tmp4->value == e.weight)
					{
						tmp4->haloId      = minValue;
						tmp3->parent 	  = tmp4;
						tmp1->parentSuper = tmp4;
					}
					else
					{
						Node *n   = &nodesTmp1[j];
						n->value  = e.weight;
						n->haloId = minValue;

						tmp3->parent      = n;
						tmp4->parent 	  = n;
						tmp1->parentSuper = n;
						tmp2->parentSuper = n;
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
		thrust::device_vector<int> tmp1, tmp2, tmpCombined;
		tmp1.resize(numOfCubes);
		tmp2.resize(numOfCubes);
                thrust::sequence(tmp1.begin(), tmp1.end());
                thrust::sequence(tmp2.begin(), tmp2.end());

		thrust::device_vector<int> A, B;

		// set new number of cubes
		int numOfCubesOld = numOfCubes;
		numOfCubes = (int)std::ceil(((double)numOfCubes/2));

		if(edges.size()==0) return;

		while(numOfCubes!=numOfCubesOld && numOfCubes>0)
		{
			tmpCombined.resize(2*numOfCubes);
			thrust::merge(tmp1.begin(), tmp1.begin()+numOfCubes, tmp2.begin(), tmp2.begin()+numOfCubes, tmpCombined.begin());		

			// set new cube Ids
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles, 
					setNewCubeID(thrust::raw_pointer_cast(&*cubeId.begin())));

			#ifdef TEST
				outputCubeDetails("The new cube details.."); // output cube details
			#endif

			// set tmp arrays
			A.resize(numOfCubesOld);
			B.resize(numOfCubesOld);

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
		
			// for each new cube, remove duplicates, set the new numOfEdges, sizeOfEdges & startOfEdges
			removeDuplicatesPerCube();

			#ifdef TEST
				outputEdgeDetails("After removing duplicate edges for new cubes. ."); // output edge details
			#endif

			// for each new cube, get the sub merge tree
			getSubMergeTreePerCube();

			//nodes.resize(numOfParticles);

			#ifdef TEST
				outputMergeTreeDetails("The new sub merge trees.."); // output merge tree details
			#endif

			// set new number of cubes
			numOfCubesOld = numOfCubes;
			numOfCubes = (int)std::ceil(((double)numOfCubes/2));
		}
	}

	//*** for each cube, set new cube Ids
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

	// remove duplicate edges in each cube
	void removeDuplicatesPerCube()
	{
		thrust::device_vector<int> stencil(numOfEdges);
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfCubes,
				isDuplicateInCube(thrust::raw_pointer_cast(&*edgeSizeOfCubes.begin()),
								  thrust::raw_pointer_cast(&*edgeStartOfCubes.begin()),
								  thrust::raw_pointer_cast(&*edges.begin()),
								  thrust::raw_pointer_cast(&*stencil.begin())));
		
		thrust::device_vector<Edge>::iterator new_end;
		new_end = thrust::remove_if(edges.begin(), edges.begin()+numOfEdges, stencil.begin(), thrust::identity<int>());

		// get new number of edges
		numOfEdges = new_end - edges.begin();

		// for each cube, get new start of edges
		thrust::exclusive_scan(edgeSizeOfCubes.begin(), edgeSizeOfCubes.end(), edgeStartOfCubes.begin());		
	}

	//*** find if a given edge is a duplicate
	struct isDuplicateInCube
	{
		int *edgeSizeOfCubes, *edgeStartOfCubes;
		int *stencil;

		Edge *edges;

		__host__ __device__
		isDuplicateInCube(int *edgeSizeOfCubes, int *edgeStartOfCubes, Edge *edges, int *stencil) :
			edgeSizeOfCubes(edgeSizeOfCubes), edgeStartOfCubes(edgeStartOfCubes), edges(edges), stencil(stencil) {}

		__host__ __device__
		void operator()(int i)
		{
			int size = 0;
			for(int j=edgeStartOfCubes[i]; j<edgeStartOfCubes[i]+edgeSizeOfCubes[i]; j++)
			{
				stencil[j] = 0;

				Edge e = edges[j];

				int k = j+1;
				while(k<edgeStartOfCubes[i]+edgeSizeOfCubes[i]) //check with next ones
				{
					Edge eNxt = edges[k++];

					if(e.weight == eNxt.weight)
					{
						if(e.srcId == eNxt.srcId && e.desId == eNxt.desId)
						{
							stencil[j] = 1;
							size++;
							break;
						}
					}
					else
						break;
				}
			}

			edgeSizeOfCubes[i] -= size;
		}		
	};

};

}

#endif
