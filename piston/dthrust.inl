
namespace dthrust
{

template <typename T>
void get_mpi_type(const std::type_info& t, MPI_Datatype& dt, int& df)
{
    df = 1;
    if (t == typeid(int)) { dt = MPI_INT; }   
    else if (t == typeid(float)) { dt = MPI_FLOAT; }
    else if (t == typeid(double)) { dt = MPI_DOUBLE; }
    else if (t == typeid(float4)) { dt = MPI_FLOAT; df = 4; }
    else if (t == typeid(float3)) { dt = MPI_FLOAT; df = 3; }
    else { dt = MPI_CHAR; } 
}


template <typename T>
void host_to_device(int hsize, thrust::host_vector<T>& h, thrust::device_vector<T>& d)
{
    thrust::host_vector<T> hl;  hl.resize(hsize);
    MPI_Datatype dataType;  int dataTypeFactor;  dthrust::get_mpi_type<T>(typeid(T), dataType, dataTypeFactor);
    MPI_CHECK(MPI_Scatter(thrust::raw_pointer_cast(&*h.begin()), hsize, dataType, thrust::raw_pointer_cast(&*hl.begin()), hsize, dataType, 0, MPI_COMM_WORLD));
    d = hl;    
}


template <typename T>
void device_to_host(int hsize, thrust::device_vector<T>& d, thrust::host_vector<T>& h)
{
    int commSize; MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize)); 
    int commRank; MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

    int* recvCounts = 0;
    int* displs = 0;
    int gsize = 0;
    if (commRank == 0) recvCounts = new int[commSize];
    MPI_Datatype dataType;  int dataTypeFactor;  dthrust::get_mpi_type<T>(typeid(T), dataType, dataTypeFactor);
    
    hsize *= dataTypeFactor;
    MPI_CHECK(MPI_Gather(&hsize, 1, MPI_INT, recvCounts, 1, MPI_INT, 0, MPI_COMM_WORLD));

    if (commRank == 0)
    {
      gsize = 0; for (unsigned int i=0; i<commSize; i++) gsize += recvCounts[i]/dataTypeFactor;

      displs = new int[commSize];
      displs[0] = 0;
      for (unsigned int i=1; i<commSize; i++) displs[i] = displs[i-1] + recvCounts[i-1];
      h.resize(gsize);
    }

    thrust::host_vector<T> hl;    
    hl = d;
        
    MPI_CHECK(MPI_Gatherv(thrust::raw_pointer_cast(&*hl.begin()), hsize, dataType, thrust::raw_pointer_cast(&*h.begin()), recvCounts, displs, dataType, 0, MPI_COMM_WORLD));

    delete recvCounts;  delete displs;
}


template <typename InputIterator, typename OutputIterator, typename T, typename BinaryOperation>
OutputIterator scan(InputIterator first, InputIterator last, OutputIterator result, T init, bool inclusiveScan, BinaryOperation binop)
{
    int commSize; MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize)); 
    int commRank; MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));
    int N = last - first;

    T psum = 0;  if (!inclusiveScan) psum = *(last-1);

    if (inclusiveScan) thrust::inclusive_scan(first, last, result, binop);
    else if (commRank == 0) thrust::exclusive_scan(first, last, result, init, binop);
    else thrust::exclusive_scan(first, last, result, 0, binop);

    psum += *(result+N-1);  std::vector<T> psums;   
    MPI_Datatype dataType;  int dataTypeFactor;  dthrust::get_mpi_type<T>(typeid(T), dataType, dataTypeFactor);
    if (commRank == 0) { psums.resize(commSize+1); psums[0] = 0; }
    MPI_CHECK(MPI_Gather(&psum, 1, dataType, &psums[1], 1, dataType, 0, MPI_COMM_WORLD)); 
    if (commRank == 0) std::partial_sum(psums.begin(), psums.end(), psums.begin(), binop); 
    MPI_CHECK(MPI_Scatter(&psums[0], 1, dataType, &psum, 1, dataType, 0, MPI_COMM_WORLD));

    thrust::transform(result, result+N, thrust::make_constant_iterator(psum), result, binop);

    return (result+N);
}


template<typename InputIterator, typename OutputIterator, typename UnaryFunction, typename AssociativeOperator>
OutputIterator transform_inclusive_scan(InputIterator first, InputIterator last, OutputIterator result, UnaryFunction unary_op, AssociativeOperator binary_op)
{
    dthrust::transform(first, last, result, unary_op);
    dthrust::inclusive_scan(result, result+(last-first), result, binary_op); 
}


template <typename T>
void output_global_vector(thrust::device_vector<T>& testing, int gsize, int lsize)
{
    int commRank; MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));

    thrust::host_vector<T> test, test_local; 
    if (commRank == 0) test.resize(gsize);
        
    test_local = testing;
    MPI_CHECK(MPI_Gather(thrust::raw_pointer_cast(&*test_local.begin()), lsize, MPI_INT, thrust::raw_pointer_cast(&*test.begin()), lsize, MPI_INT, 0, MPI_COMM_WORLD));

    if (commRank == 0) { thrust::copy(test.begin(), test.begin()+gsize, std::ostream_iterator<int>(std::cout, " ")); std::cout << std::endl; }
    //MPI_Barrier(MPI_COMM_WORLD);
}


template<typename InputIterator, typename OutputVector>
typename OutputVector::iterator upper_bound_counting(InputIterator first, InputIterator last, int cntMax, OutputVector& result) 
{
    int commSize; MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &commSize));
    int commRank; MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &commRank));    

    int n = *(last-1) - *first;
    int p = 0;
    int q = 0;
    int l = last - first;
    int offset = commRank*l;

    MPI_Status stat;  int previousValue;  int lastValue = *(last-1);
    if (commRank < (commSize-1)) MPI_Send(&lastValue, 1, MPI_INT, commRank+1, 0, MPI_COMM_WORLD);
    if (commRank > 0) 
    {
      MPI_Recv(&previousValue, 1, MPI_INT, commRank-1, 0, MPI_COMM_WORLD, &stat);
      if (previousValue < *(first)) p += 1;
    }

    if ((commRank == commSize-1) && (cntMax >= lastValue)) q += (cntMax - lastValue + 1); 

    result.resize(n+p+q);
    if ((commRank > 0) && (previousValue < *(first))) result[0] = 0;
    if ((commRank == commSize-1) && (cntMax >= lastValue)) for (unsigned int i=n+p; i<=n+p+(cntMax-lastValue+1); i++) result[i] = l;
    
    thrust::upper_bound(first, last, thrust::counting_iterator<int>(*first), thrust::counting_iterator<int>(*first)+n, result.begin()+p); 
    thrust::transform(result.begin(), result.begin()+n+p+q, thrust::constant_iterator<int>(offset), result.begin(), thrust::plus<int>());
    return (result.begin()+n+p+q);
}

}
