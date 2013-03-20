#ifndef HALO_KD_H
#define HALO_KD_H

#include <piston/halo.h>


namespace piston
{

class halo_kd : public halo
{
public:

    struct KDtreeNode
    {
        int i;    //ind in the kdtree
        int ind;  //ind in data
        int parent, leftC, rightC;
        int startInd;
        int size;
        float splitValue;
        bool isRight;

        __host__ __device__
        KDtreeNode() : i(-1), ind(-1), parent(-1), leftC(-1), rightC(-1), startInd(-1), size(-1), splitValue(-1), isRight(false) {}
    };


    struct Level
    {
        int startInd;
        int size;
        int splitAxis;

        __host__ __device__
        Level() {}
    };


    thrust::device_vector<float> uBoundsX, uBoundsY, uBoundsZ;
    thrust::device_vector<float> lBoundsX, lBoundsY, lBoundsZ;

    thrust::device_vector<Level> levelInfo;
    int levels;
    KDTree* ktree;

    thrust::device_vector<KDtreeNode> kd_tree;

    //union find data structure
    thrust::device_vector<int> nextp;

    thrust::device_vector<int> segStartInd; 	
    thrust::device_vector<int> segSize; 		
    thrust::device_vector<int> subSegStartInd;      
    thrust::device_vector<int> subSegSize; 		
    thrust::device_vector<int> childBB; 		
    thrust::device_vector<int> toCompare; 		
    thrust::device_vector<float> splitValue; 	
    thrust::device_vector<int> C; 
    thrust::device_vector<int> D;

    thrust::device_vector<int> haloIndexOriginal;


    halo_kd(std::string filename="", std::string format=".cosmo", int n = 1, int np=1, float rL=-1, bool periodic=false) : halo(filename, format, n, np, rL, periodic)
    {
        if(numOfParticles!=0)
        {			
			levels    = (int) std::ceil(log2((double)(numOfParticles)))+1;
			haloIndex.resize(numOfParticles);
			thrust::sequence(haloIndex.begin(), haloIndex.end());

			haloIndexOriginal.resize(numOfParticles);
			thrust::sequence(haloIndexOriginal.begin(), haloIndexOriginal.end());

			struct timeval begin, mid1, mid2, end, diff1, diff2;
			gettimeofday(&begin, 0);
			getBalancedKdTree();
			gettimeofday(&mid1, 0);

			// set the size for computing bounds
			uBoundsX.resize(numOfParticles*2);
			uBoundsY.resize(numOfParticles*2);
			uBoundsZ.resize(numOfParticles*2);
			lBoundsX.resize(numOfParticles*2);
			lBoundsY.resize(numOfParticles*2);
			lBoundsZ.resize(numOfParticles*2);

			gettimeofday(&mid2, 0);
			computeBounds();
			gettimeofday(&end, 0);

			timersub(&mid1, &begin, &diff1);
			float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
			std::cout << "Time elapsed: " << seconds1 << " s for KD tree construction"<< std::endl << std::flush;
			timersub(&end, &mid2, &diff2);
			float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
			std::cout << "Time elapsed: " << seconds2 << " s for computing bounds"<< std::endl << std::flush;
        }
    }


    void operator()(float linkLength , int  particleSize)
    {
        clear();

        linkLength    = linkLength;
        particleSize  = particleSize;

        // no valid particles, return
        if(numOfParticles==0) return;

        // set vectors for halo finding
        nextp.resize(numOfParticles);
        thrust::fill(nextp.begin(), nextp.end(), -1);

        struct timeval begin, mid, end, diff1, diff2;
        gettimeofday(&begin, 0);
        findHalos(linkLength, numOfParticles);
        gettimeofday(&mid, 0);
        getUniqueHalos(particleSize); // get the unique valid halo ids
        gettimeofday(&end, 0);

        thrust::device_vector<int> tmp; tmp.resize(haloIndex.size());
		thrust::scatter(haloIndex.begin(), haloIndex.end(), haloIndexOriginal.begin(), tmp.begin());
		thrust::copy(tmp.begin(), tmp.end(), haloIndex.begin());

		// set correct halo ids
		thrust::fill(tmp.begin(), tmp.end(), -1);
		for(int i=0; i<numOfParticles; i++)
		{
			if(tmp[i]==-1)
			{
				thrust::for_each(CountingIterator(i), CountingIterator(i)+(numOfParticles-i),
						setCorrectHaloId(thrust::raw_pointer_cast(&*haloIndex.begin()), thrust::raw_pointer_cast(&*tmp.begin()), haloIndex[i], i));
			}
		}
		thrust::copy(tmp.begin(), tmp.end(), haloIndex.begin());

        timersub(&mid, &begin, &diff1);
        float seconds1 = diff1.tv_sec + 1.0E-6*diff1.tv_usec;
        std::cout << "Time elapsed: " << seconds1 << " s for merging"<< std::endl << std::flush;
        timersub(&end, &mid, &diff2);
        float seconds2 = diff2.tv_sec + 1.0E-6*diff2.tv_usec;
        std::cout << "Time elapsed: " << seconds2 << " s for finding valid halos"<< std::endl << std::flush;

        setColors(); // set colors to halos
        std::cout << "Number of Particles : " << numOfParticles <<  " Number of Halos found : " << numOfHalos << std::endl << std::endl;
    }

    struct setCorrectHaloId
    {
    	int* haloIndex;
		int* tmp;
		int n, i;

		__host__ __device__
		setCorrectHaloId(int* haloIndex, int* tmp, int n, int i) :
			  haloIndex(haloIndex), tmp(tmp), n(n), i(i) {}

		__host__ __device__
		void operator()(int j)
		{
			if(haloIndex[j] == n)
				tmp[j] = i;
		}
    };

    // build the b-Kdtree & split the data
    void getBalancedKdTree()
    {
        // set the size for balanced kd tree
        kd_tree.resize(numOfParticles*2);

        //set the size for the level details array
        levelInfo.resize(levels); 

        thrust::device_vector<KDtreeNode> lChildren;
        thrust::device_vector<KDtreeNode> rChildren;

        int curIndex = 0, count = 1;

        KDtreeNode current = kd_tree[count-1];
        current.i    	   =  0;
        current.ind 	   = -1;
        current.parent	   = -1;
        current.startInd   =  0;
        current.size       =  numOfParticles;
        kd_tree[count-1]   = current;
        for(int l=0; l<levels; l++)
        {			
			if(curIndex>=count) return;

			int size = count-curIndex;

			Level currentL     = levelInfo[l];
			currentL.startInd  = curIndex;
			currentL.size      = size;
			currentL.splitAxis = l%3;
			levelInfo[l]       = currentL;

			// get left & right children
			lChildren.resize(size); 
			rChildren.resize(size); 
			thrust::transform(CountingIterator(curIndex), CountingIterator(curIndex)+size,
					thrust::make_zip_iterator(thrust::make_tuple(lChildren.begin(), rChildren.begin())),
					getChildren(thrust::raw_pointer_cast(&*kd_tree.begin()), l));

			// remove invalid children in lChildren & rChildren
			thrust::device_vector<KDtreeNode>::iterator new_end1 = thrust::remove_if(lChildren.begin(), lChildren.end(), notValidKDtreeNode());
			thrust::device_vector<KDtreeNode>::iterator new_end2 = thrust::remove_if(rChildren.begin(), rChildren.end(), notValidKDtreeNode());

			// get number of valid left & right children
			int numValidlChildren = new_end1-lChildren.begin();
			int numValidrChildren = new_end2-rChildren.begin();

			// set parent child details, & insert to tree
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numValidlChildren,
				   nodeInsert(thrust::raw_pointer_cast(&*kd_tree.begin()), count, thrust::raw_pointer_cast(&*lChildren.begin())) );
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numValidrChildren,
				   nodeInsert(thrust::raw_pointer_cast(&*kd_tree.begin()), count+numValidlChildren, thrust::raw_pointer_cast(&*rChildren.begin())) );

			count   += (numValidlChildren + numValidrChildren);
			curIndex = curIndex+size;
        }
		
	getRanksForAllSegments(thrust::raw_pointer_cast(&*kd_tree.begin()));

        /*  std::cout << "Levels: " << std::endl << std::flush;
        for (unsigned int i=0; i<levelInfo.size(); i++)
        {
          Level li = levelInfo[i];
            std::cout << li.startInd << " " << li.size << " " << li.splitAxis << std::endl << std::flush;
        }

          std::cout << "Tree: " << std::endl << std::flush;
        for (unsigned int i=0; i<kd_tree.size(); i++)
        {
          KDtreeNode kdn = kd_tree[i];
            std::cout << kdn.startInd << " " << kdn.size << " " << kdn.splitValue << std::endl << std::flush;
        }*/
    }


    // get left & right children of each node if they exist
    struct getChildren : public thrust::unary_function<int, thrust::tuple<KDtreeNode, KDtreeNode> >
    {
        KDtreeNode* tree;
        int level;

        __host__ __device__
        getChildren(KDtreeNode* tree, int level) : tree(tree), level(level) {}

				__host__ __device__
				thrust::tuple<KDtreeNode, KDtreeNode>  operator()(int i)
				{
					KDtreeNode* current = &tree[i];

					int lSplit = (current->size!=1) ? (float)(current->size/2) : 0;
					int rSplit = (current->size!=1) ? (current->size - lSplit) : 0;

					KDtreeNode left = KDtreeNode();
					if(lSplit>0)
					{
						left.parent      = current->i;
						left.startInd    = current->startInd;
						left.size        = lSplit;
						left.isRight     = false;
						left.ind         = (left.size==1) ? left.startInd : -1;
					}

					KDtreeNode right = KDtreeNode();
					if(rSplit>0)
					{
						right.parent      = current->i;
						right.startInd    = current->startInd+lSplit;
						right.size        = rSplit;
						right.isRight     = true;
						right.ind         = (right.size==1) ? right.startInd : -1;
					}

					return thrust::make_tuple(left,right);
        }
    };


    // check whether this node is a valid item in the kdtree
    struct notValidKDtreeNode : public thrust::unary_function<int, bool>
    {
        __host__ __device__
        bool operator()(KDtreeNode n) { return (n.parent==-1); }
    };


    // set parent child details, & insert the node to kdtree
    struct nodeInsert : public thrust::unary_function<int, void>
    {
        KDtreeNode* children;
		KDtreeNode* tree;
		int count;

		__host__ __device__
		nodeInsert(KDtreeNode* tree, int count, KDtreeNode* children) :
			   tree(tree), count(count), children(children) { };

		__host__ __device__
		void operator()(int i)
		{
			KDtreeNode child = children[i];

			child.i = count+i;
			if(!child.isRight) tree[child.parent].leftC  = child.i;
			else tree[child.parent].rightC = child.i;

			tree[child.i] = child;
		}
    };

    
    // do the rankSplit for all levels
    void getRanksForAllSegments(KDtreeNode* tree)
    {
        ktree = new KDTree();
        ktree->initializeTree(inputX, inputY, inputZ);

        thrust::device_vector<int>::iterator rankFirst, pointId;
        for(int l=0; l<levels; l++)
        {
          if (l%3 == 0)      rankFirst = ktree->m_xrank.begin();
          else if (l%3 == 1) rankFirst = ktree->m_yrank.begin();
          else               rankFirst = ktree->m_zrank.begin();

          pointId = ktree->m_pointId.begin();

          Level currentL = levelInfo[l];
          thrust::for_each(CountingIterator(currentL.startInd), CountingIterator(currentL.startInd)+currentL.size,
              setSplitValue(thrust::raw_pointer_cast(&*rankFirst),
                  thrust::raw_pointer_cast(&*inputX.begin()), thrust::raw_pointer_cast(&*inputY.begin()), thrust::raw_pointer_cast(&*inputZ.begin()),
                  thrust::raw_pointer_cast(&*pointId), tree, l));
          if (l+1 < levels) ktree->buildTreeLevel(l+1);
        }

        thrust::device_vector<float> inputReorder;  inputReorder.resize(inputX.size());
        thrust::copy(thrust::make_permutation_iterator(inputX.begin(), pointId), thrust::make_permutation_iterator(inputX.end(), pointId+inputX.size()), inputReorder.begin());
        thrust::copy(inputReorder.begin(), inputReorder.end(), inputX.begin());
        thrust::copy(thrust::make_permutation_iterator(inputY.begin(), pointId), thrust::make_permutation_iterator(inputY.end(), pointId+inputX.size()), inputReorder.begin());
        thrust::copy(inputReorder.begin(), inputReorder.end(), inputY.begin());
        thrust::copy(thrust::make_permutation_iterator(inputZ.begin(), pointId), thrust::make_permutation_iterator(inputZ.end(), pointId+inputX.size()), inputReorder.begin());
        thrust::copy(inputReorder.begin(), inputReorder.end(), inputZ.begin());

        thrust::device_vector<int> haloIdReorder;  haloIdReorder.resize(haloIndexOriginal.size());
        thrust::copy(thrust::make_permutation_iterator(haloIndexOriginal.begin(), pointId), thrust::make_permutation_iterator(haloIndexOriginal.end(), pointId+haloIndexOriginal.size()), haloIdReorder.begin());
		thrust::copy(haloIdReorder.begin(), haloIdReorder.end(), haloIndexOriginal.begin());
		thrust::copy(haloIdReorder.begin(), haloIdReorder.end(), haloIndex.begin());

        delete ktree;
    }


    // set the split value for the segment
    struct setSplitValue : public thrust::unary_function<int, void>
    {
        int *rankFirst, *pointInd;
		float *inputX, *inputY, *inputZ;
		KDtreeNode* tree;
		int l;

		__host__ __device__
		setSplitValue(int* rankFirst, float *inputX, float *inputY, float *inputZ, int* pointInd, KDtreeNode* tree, int l) :
				  rankFirst(rankFirst), inputX(inputX), inputY(inputY), inputZ(inputZ), pointInd(pointInd), tree(tree), l(l) {}

		__host__ __device__
		void operator()(int i)
		{
			KDtreeNode *current = &tree[i];

			if(current->size==1) return;

			int lSplit = (current->size!=1) ? (float)(current->size/2) : 0;

			int leftInd, rightInd;
			for(int j=0; j<current->size; j++)
			{
				if(rankFirst[current->startInd+j] == lSplit-1) leftInd = pointInd[current->startInd+j];
				if(rankFirst[current->startInd+j] == lSplit) rightInd = pointInd[current->startInd+j];
			}
			if(l%3 == 0) current->splitValue = (float) (inputX[leftInd] + inputX[rightInd])/2;
			else if(l%3 == 1) current->splitValue = (float) (inputY[leftInd] + inputY[rightInd])/2;
			else current->splitValue = (float) (inputZ[leftInd] + inputZ[rightInd])/2;
		}
    };


    //----------- methods for computing bounds
    void computeBounds()
    {
		for (int l=levels-1; l>=0; l--)
		{
			Level currentL = levelInfo[l];
			int startInd = currentL.startInd;
			int size     = currentL.size;

			thrust::for_each(CountingIterator(startInd), CountingIterator(startInd)+size, getBound(thrust::raw_pointer_cast(&*kd_tree.begin()),
				   thrust::raw_pointer_cast(&*inputX.begin()), thrust::raw_pointer_cast(&*inputY.begin()), thrust::raw_pointer_cast(&*inputZ.begin()),
				   thrust::raw_pointer_cast(&*uBoundsX.begin()), thrust::raw_pointer_cast(&*uBoundsY.begin()), thrust::raw_pointer_cast(&*uBoundsZ.begin()),
				   thrust::raw_pointer_cast(&*lBoundsX.begin()), thrust::raw_pointer_cast(&*lBoundsY.begin()), thrust::raw_pointer_cast(&*lBoundsZ.begin()) ));
		}
    }


    struct getBound : public thrust::unary_function<int, void>
    {
		KDtreeNode* kd_tree;
		float  *uBoundsX, *uBoundsY, *uBoundsZ;
		float  *lBoundsX, *lBoundsY, *lBoundsZ;

		float *inputX, *inputY, *inputZ;

		__host__ __device__
		getBound(KDtreeNode* kd_tree,
			 float *inputX, float *inputY, float *inputZ,
			 float *uBoundsX, float *uBoundsY, float *uBoundsZ,
			 float *lBoundsX, float *lBoundsY, float *lBoundsZ) :
			 kd_tree(kd_tree), inputX(inputX), inputY(inputY), inputZ(inputZ),
				 uBoundsX(uBoundsX), uBoundsY(uBoundsY), uBoundsZ(uBoundsZ),
			 lBoundsX(lBoundsX), lBoundsY(lBoundsY), lBoundsZ(lBoundsZ) {}

		__host__ __device__
		void operator()(int i)
		{
			KDtreeNode n = kd_tree[i];

			int lChildInd = n.leftC;
			int rChildInd = n.rightC;

			if(lChildInd == -1 && rChildInd == -1)
			{
				uBoundsX[i] = inputX[n.ind];
				lBoundsX[i] = inputX[n.ind];

				uBoundsY[i] = inputY[n.ind];
				lBoundsY[i] = inputY[n.ind];

				uBoundsZ[i] = inputZ[n.ind];
				lBoundsZ[i] = inputZ[n.ind];
				return;
			}
			else if(lChildInd != -1 && rChildInd == -1)
			{
				KDtreeNode lChild = kd_tree[lChildInd];

				uBoundsX[i] = uBoundsX[lChild.i];
				lBoundsX[i] = lBoundsX[lChild.i];

				uBoundsY[i] = uBoundsY[lChild.i];
				lBoundsY[i] = lBoundsY[lChild.i];

				uBoundsZ[i] = uBoundsZ[lChild.i];
				lBoundsZ[i] = lBoundsZ[lChild.i];
			}
			else if(rChildInd != -1 && lChildInd == -1)
			{
				KDtreeNode rChild = kd_tree[rChildInd];

				uBoundsX[i] = uBoundsX[rChild.i];
				lBoundsX[i] = lBoundsX[rChild.i];

				uBoundsY[i] = uBoundsY[rChild.i];
				lBoundsY[i] = lBoundsY[rChild.i];

				uBoundsZ[i] = uBoundsZ[rChild.i];
				lBoundsZ[i] = lBoundsZ[rChild.i];
			}
			else
			{
				KDtreeNode lChild = kd_tree[lChildInd];
				KDtreeNode rChild = kd_tree[rChildInd];

				uBoundsX[i] = uBoundsX[lChild.i] > uBoundsX[rChild.i] ? uBoundsX[lChild.i] : uBoundsX[rChild.i];
				lBoundsX[i] = lBoundsX[lChild.i] < lBoundsX[rChild.i] ? lBoundsX[lChild.i] : lBoundsX[rChild.i];

				uBoundsY[i] = uBoundsY[lChild.i] > uBoundsY[rChild.i] ? uBoundsY[lChild.i] : uBoundsY[rChild.i];
				lBoundsY[i] = lBoundsY[lChild.i] < lBoundsY[rChild.i] ? lBoundsY[lChild.i] : lBoundsY[rChild.i];

				uBoundsZ[i] = uBoundsZ[lChild.i] > uBoundsZ[rChild.i] ? uBoundsZ[lChild.i] : uBoundsZ[rChild.i];
				lBoundsZ[i] = lBoundsZ[lChild.i] < lBoundsZ[rChild.i] ? lBoundsZ[lChild.i] : lBoundsZ[rChild.i];
			}
		}
    };


    //----------- methods for merging particles into halos
    void findHalos(float linkLength, int numOfParticles)
    {
		segStartInd.resize(numOfParticles);
		segSize.resize(numOfParticles);
		subSegStartInd.resize(numOfParticles);
		subSegSize.resize(numOfParticles);
		childBB.resize(numOfParticles);
		toCompare.resize(numOfParticles);
		splitValue.resize(numOfParticles);

		C.resize(numOfParticles);
		D.resize(numOfParticles);
		thrust::device_vector<KDtreeNode>::iterator tree;
		Level currentL;
		thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<int>::iterator> new_end;

		for (int l=levels-2; l>=0; l--)
		{
			tree = kd_tree.begin();
			currentL = levelInfo[l];
			int start = currentL.startInd;
			int size  = currentL.size;

			thrust::fill(segStartInd.begin(),    segStartInd.end(),    0);
			thrust::fill(segSize.begin(),        segSize.end(), 	   0);
			thrust::fill(subSegStartInd.begin(), subSegStartInd.end(), 0);
			thrust::fill(subSegSize.begin(),     subSegSize.end(),     0);
			thrust::fill(splitValue.begin(),     splitValue.end(),     0);
			thrust::fill(toCompare.begin(),      toCompare.end(),      0);

			// set details of segments
			thrust::for_each(CountingIterator(start), CountingIterator(start)+size,
				   setSegDetails(thrust::raw_pointer_cast(&*segStartInd.begin()),
						 thrust::raw_pointer_cast(&*segSize.begin()),
						 thrust::raw_pointer_cast(&*kd_tree.begin())));
			thrust::inclusive_scan(segStartInd.begin(), segStartInd.end(), segStartInd.begin(), thrust::maximum<int>());

			// set details of sub segments
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
				   setSubSegDetails(thrust::raw_pointer_cast(&*subSegStartInd.begin()),
							thrust::raw_pointer_cast(&*subSegSize.begin()),
							thrust::raw_pointer_cast(&*segStartInd.begin()),
							thrust::raw_pointer_cast(&*segSize.begin())));

			//---------

			//comment this to not to do the BB comparisons & uncomment the previous three lines
			{
			// get the bounds arrays for this level
			float *uBounds, *lBounds;
			if (l%3 == 0)      { uBounds = thrust::raw_pointer_cast(&*uBoundsX.begin());  lBounds = thrust::raw_pointer_cast(&*lBoundsX.begin()); }
			else if (l%3 == 1) { uBounds = thrust::raw_pointer_cast(&*uBoundsY.begin());  lBounds = thrust::raw_pointer_cast(&*lBoundsY.begin()); }
			else if (l%3 == 2) { uBounds = thrust::raw_pointer_cast(&*uBoundsZ.begin());  lBounds = thrust::raw_pointer_cast(&*lBoundsZ.begin()); }

			// for each node in this level, check whether its children's BB intersect
			thrust::fill(childBB.begin(), childBB.end(), 0);
			thrust::for_each(CountingIterator(start), CountingIterator(start)+size,
					 childBBIntersect(thrust::raw_pointer_cast(&*childBB.begin()), thrust::raw_pointer_cast(&*splitValue.begin()),
							  thrust::raw_pointer_cast(&*kd_tree.begin()), lBounds, uBounds, linkLength));
			thrust::inclusive_scan_by_key(segStartInd.begin(), segStartInd.end(), childBB.begin(), childBB.begin());
			}

			//---------

			//comment this do all of the n*n comparisons instead of m*q
			{
			// get the nodes within the boundary
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
					 setNodesWithinBoundary(thrust::raw_pointer_cast(&*childBB.begin()), thrust::raw_pointer_cast(&*splitValue.begin()),
								thrust::raw_pointer_cast(&*inputX.begin()), thrust::raw_pointer_cast(&*inputY.begin()), thrust::raw_pointer_cast(&*inputZ.begin()),
								l, linkLength, thrust::raw_pointer_cast(&*segStartInd.begin())));
			}

			//---------

			// update sub segment sizes

			new_end = thrust::reduce_by_key(subSegStartInd.begin(), subSegStartInd.end(), childBB.begin(), C.begin(), D.begin(), thrust::equal_to<int>());
			thrust::for_each(CountingIterator(0), CountingIterator(0)+(thrust::get<0>(new_end)-C.begin()),
				   updateSize(thrust::raw_pointer_cast(&*subSegSize.begin()), thrust::raw_pointer_cast(&*C.begin()), thrust::raw_pointer_cast(&*D.begin())));
			thrust::inclusive_scan_by_key(subSegStartInd.begin(), subSegStartInd.end(), subSegSize.begin(), subSegSize.begin());

			// get the number of iterations
			thrust::for_each(CountingIterator(0), CountingIterator(0)+(thrust::get<0>(new_end)-C.begin()), setValue(thrust::raw_pointer_cast(&*C.begin()), thrust::raw_pointer_cast(&*segStartInd.begin())));
			new_end = thrust::unique_by_key(C.begin(), C.begin()+(thrust::get<0>(new_end)-C.begin()), D.begin(), thrust::equal_to<int>());
			int ite = thrust::reduce(D.begin(), D.begin()+(thrust::get<0>(new_end)-C.begin()), 0, thrust::maximum<int>());

			// do m*n comparisons & merge nodes
			for(int i=0; i<ite; i++)
			{
			thrust::for_each(CountingIterator(0), CountingIterator(0)+(thrust::get<0>(new_end)-C.begin()),
					 getCurrent(thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*C.begin()),
						thrust::raw_pointer_cast(&*segStartInd.begin()), thrust::raw_pointer_cast(&*segSize.begin()),
							thrust::raw_pointer_cast(&*childBB.begin()), i));
			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles,
					 merge(thrust::raw_pointer_cast(&*nextp.begin()), thrust::raw_pointer_cast(&*D.begin()), thrust::raw_pointer_cast(&*childBB.begin()),
					   thrust::raw_pointer_cast(&*inputX.begin()), thrust::raw_pointer_cast(&*inputY.begin()), thrust::raw_pointer_cast(&*inputZ.begin()),
					   thrust::raw_pointer_cast(&*subSegStartInd.begin()), linkLength));
			}

			thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles, updateNextp(thrust::raw_pointer_cast(&*nextp.begin())));
		}
	  
		//set halo ids
		thrust::for_each(CountingIterator(0), CountingIterator(0)+numOfParticles, setHaloId(thrust::raw_pointer_cast(&*haloIndex.begin()), thrust::raw_pointer_cast(&*nextp.begin())));
		segStartInd.clear(); segSize.clear(); subSegStartInd.clear(); subSegSize.clear();
		childBB.clear(); toCompare.clear(); splitValue.clear();
		C.clear(); D.clear();
    }


    // sets the segment start index & size
    struct setSegDetails : public thrust::unary_function<int, void>
    {
		KDtreeNode* tree;
		int *segStartInd, *segSize;

		__host__ __device__
		setSegDetails(int *segStartInd, int *segSize, KDtreeNode* tree) :
				  segStartInd(segStartInd), segSize(segSize), tree(tree) {}

		__host__ __device__
		void operator()(int i)
		{
			KDtreeNode current = tree[i];

			segStartInd[current.startInd] = current.startInd;
			segSize[current.startInd]     = current.size;
		}
    };


    // sets the sub segment start index
    struct setSubSegDetails : public thrust::unary_function<int, void>
    {
        int *subSegStartInd, *subSegSize, *segStartInd, *segSize;

        __host__ __device__
        setSubSegDetails(int *subSegStartInd, int *subSegSize, int *segStartInd, int *segSize) :
	   	         subSegStartInd(subSegStartInd), subSegSize(subSegSize), segStartInd(segStartInd), segSize(segSize) {}

        __host__ __device__
        void operator()(int i)
        {
			int lSplit = (segSize[segStartInd[i]]!=1) ? (float)(segSize[segStartInd[i]]/2) : 0;

			if(i<segStartInd[i]+lSplit) subSegStartInd[i] = segStartInd[i];
			else subSegStartInd[i] = segStartInd[i]+lSplit;
        }
    };


    // for a kdtree node, check whether the BB of the two child nodes intersect, if they do set the split value
    struct childBBIntersect : public thrust::unary_function<int, void>
    {
		KDtreeNode* tree;
		float *lBounds, *uBounds;
		int   *childBB;
		float *splitValue;
		float linkLength;

		__host__ __device__
		childBBIntersect(int *childBB, float *splitValue, KDtreeNode* tree, float *lBounds, float *uBounds, float linkLength) :
				 childBB(childBB), splitValue(splitValue), tree(tree), lBounds(lBounds), uBounds(uBounds), linkLength(linkLength) {}

		__host__ __device__
		void operator()(int i)
		{
			KDtreeNode current = tree[i];

				//bounding boxes do not intersect
			if(current.leftC==-1 || current.rightC==-1)  return;

			KDtreeNode a = tree[current.leftC];
			KDtreeNode b = tree[current.rightC];

			float lR, uR;
			lR = lBounds[b.i];
			uR = uBounds[b.i];

			float lL = lBounds[a.i];
			float uL = uBounds[a.i];

			float dL = (uL - lL);  if (dL < 0.0f) dL = -dL;
			float dR = (uR - lR);  if (dR < 0.0f) dR = -dR;

			float c = uL > uR ? uL : uR;
			float d = lL < lR ? lL : lR;
			float dc = c - d;

			float dist = dc - dL - dR;

				//bounding boxes do not intersect
			if(dist>linkLength) return;

			childBB[current.startInd] = 1;

			splitValue[current.startInd] = current.splitValue;
		}
    };


    // get the split value of a given kdtree node
    struct getSplitValue : public thrust::unary_function<int, void>
    {
		KDtreeNode* tree;
		float *splitValue;

		__host__ __device__
		getSplitValue(float *splitValue, KDtreeNode* tree) : splitValue(splitValue), tree(tree) {}

		__host__ __device__
		void operator()(int i)
		{
			KDtreeNode current = tree[i];
			splitValue[current.startInd] = current.splitValue;
		}
    };


    // for a given node check whether it is within linking length in that axis
    struct setNodesWithinBoundary : public thrust::unary_function<int, void>
    {
		float *inputX, *inputY, *inputZ;
		int *childBB, *segStartInd;
		float *splitValue;
		int l;
		float linkLength;

		__host__ __device__
		setNodesWithinBoundary(int *childBB, float *splitValue, float *inputX, float *inputY, float *inputZ, int l, float linkLength, int *segStartInd) :
					   childBB(childBB), splitValue(splitValue), inputX(inputX), inputY(inputY), inputZ(inputZ), l(l), linkLength(linkLength), segStartInd(segStartInd) {}

		__host__ __device__
		void operator()(int i)
		{
			if(childBB[i]!=0)
			{
				float sValue = splitValue[segStartInd[i]];

				float dist;
				if (l%3==0) dist = (inputX[i] - sValue);
				else if (l%3==1) dist = (inputY[i] - sValue);
				else if (l%3==2) dist = (inputZ[i] - sValue);

				if (dist < 0.0f) dist = -dist;

				if(dist>linkLength) childBB[i] = 0;
			}
		}
    };


    // updata the size of the subsegments
    struct updateSize : public thrust::unary_function<int, void>
    {
		int *subSegSize, *C, *D;

		__host__ __device__
		updateSize(int *subSegSize, int *C, int *D) :
			   subSegSize(subSegSize), C(C), D(D) {}

		__host__ __device__
		void operator()(int i) { subSegSize[C[i]] = D[i]; }
    };


    // set the sub segment start Index
    struct setValue : public thrust::unary_function<int, void>
    {
		int *C, *segStartInd;

		__host__ __device__
		setValue(int *C, int *segStartInd) :
			 C(C), segStartInd(segStartInd) {}

		__host__ __device__
		void operator()(int i) { C[i] = segStartInd[C[i]]; }
    };


    // get the node with which the next to compair for the m*n comparisons which should be done
    struct getCurrent : public thrust::unary_function<int, void>
    {
		int *C, *D, *segStartInd, *segSize, *childBB;
		int ite;

		__host__ __device__
		getCurrent(int *D, int *C, int *segStartInd, int *segSize, int*childBB, int ite) :
			   D(D), C(C), segStartInd(segStartInd), segSize(segSize), childBB(childBB), ite(ite) {}

		__host__ __device__
		void operator()(int i)
		{
			int start = 0;
			start = C[i];
			int size  = 0;  int temp = 0;
			temp = segStartInd[C[i]];
			size = segSize[temp];

			int lSplit = (size!=1) ? (float)(size/2) : 0;

			int ind = 0;
			if (ite==0) ind = start;
			else ind = D[start+lSplit]+1;

			D[start+lSplit] = -1;
			for(int j=ind; j<start+lSplit; j++)
			{
				if (childBB[j]==1)
				{
					D[start+lSplit] = j;
					break;
				}
			}
		}
    };


    // merge two particles
    struct merge : public thrust::unary_function<int, void>
    {
		float *inputX, *inputY, *inputZ;
		int* nextp;
		int *D, *childBB, *subSegStartInd;
		float linkLength;

		__host__ __device__
		merge(int* nextp, int *D, int *childBB, float *inputX, float *inputY, float *inputZ, int *subSegStartInd, float linkLength) :
			  nextp(nextp), D(D), childBB(childBB), inputX(inputX), inputY(inputY), inputZ(inputZ), subSegStartInd(subSegStartInd), linkLength(linkLength) {}

		__host__ __device__
		void operator()(int i)
		{
			if ((childBB[i]==1) && (D[subSegStartInd[i]]!=-1))
			{
				int j = 0;
				j = D[subSegStartInd[i]];

				float xd, yd, zd;
				xd = (inputX[i] - inputX[j]);  if (xd < 0.0f) xd = -xd;
				yd = (inputY[i] - inputY[j]);  if (yd < 0.0f) yd = -yd;
				zd = (inputZ[i] - inputZ[j]);  if (zd < 0.0f) zd = -zd;

				if (xd<=linkLength && yd<=linkLength && zd<=linkLength)
				{
					float dist = (float)(xd*xd + yd*yd + zd*zd);
					if (dist <= linkLength*linkLength)
					{
						// find for a & b
						while (nextp[i]>=0) i = nextp[i];
						while (nextp[j]>=0) j = nextp[j];

						if(i==j) return;

						int m1 = i > j ? i : j;
						int m2 = i > j ? j : i;
						nextp[m1] = m2;
					}
				}
			}
		}
    };


    // update the union find data structure after one iteration
    struct updateNextp : public thrust::unary_function<int, void>
    {
		int* nextp;

		__host__ __device__
		updateNextp(int* nextp) : nextp(nextp) {}

		__host__ __device__
		void operator()(int i)
		{
			if(nextp[i]>=0)
			{
				int a = nextp[i];
				while(nextp[a]>=0) a = nextp[a];
				nextp[i] = a;
			}
		}
    };


    // set the halo id for each particle
    struct setHaloId : public thrust::unary_function<int, void>
    {
		int* haloIndex;
		int* nextp;

		__host__ __device__
		setHaloId(int* haloIndex, int* nextp) :
			  haloIndex(haloIndex), nextp(nextp) {}

		__host__ __device__
		void operator()(int i)
		{
			int a = i;
			while(nextp[a]>=0) a = nextp[a];

			haloIndex[i] = a;
		}
    };
};

}

#endif
