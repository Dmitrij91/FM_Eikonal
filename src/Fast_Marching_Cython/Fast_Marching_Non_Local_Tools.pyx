import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from openmp cimport omp_get_max_threads, omp_get_thread_num
from libc.math cimport signbit,exp, pi,pow,sqrt
from libc.stdlib cimport malloc, free
from Fast_Marching_Graph_Utilities import normalize_adj






# ===============================================================================
# Compute Non_Local_Means Weights
# Returns a sparse Matrix of Image shape with weights as distance between patches
# ===============================================================================


# ===============================================================================
# Compute Non_Local_Means Weights
# Returns a sparse Matrix of Image shape with weights as distance between patches
# ===============================================================================

cdef double signum(double x) nogil:
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    else:
        return 0.0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Non_Local_Means_cython(cnp.ndarray[double,ndim = 2] Im,cnp.ndarray[double,ndim = 1] Sigma_Weights,cnp.ndarray[int,ndim = 1] indptr,cnp.ndarray[int,ndim = 1] indices\
                           ,(int,int) shape_img,double sigma,int window_size):
    cdef int m = Im.shape[0]
    cdef int n = Im.shape[1]
    print(n,m,'nm')
    cdef int l,s,index,k,Iter_Patsch,Length_NPatch,h,y,f,w
    cdef double sign
    size = len(Im[:,:].reshape(-1)[indices])
    cdef cnp.ndarray[double,ndim = 1] data = Im[:,:].reshape(-1)[indices]
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Output = np.zeros(window_size*window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Neigh_Patch = np.zeros(window_size*window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] vec = np.zeros(window_size*window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Patch_1 = np.zeros(window_size*window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Im_weights_data = np.empty(size)
    cdef double* Patch_mod = <double*> malloc(sizeof(double) *window_size*window_size)


    for k in range(m*n):

        # Length of Patch

        Iter_Patsch = indptr[k+1]-indptr[k]

        # Extract the Patch

        for index in range(Iter_Patsch):
            Patch_1[index]   = data[indptr[k]+index]
            Patch_mod[index] = Patch_1[index]
        for l in range(Iter_Patsch):
            Output[l] = 0

    # Run Trough Patch Pixels

        for s in prange(Iter_Patsch,nogil = True):

            # Neighbour Patches

            Length_NPatch = indptr[indices[indptr[k]+s]+1]-indptr[indices[indptr[k]+s]]

           # Compute Neighbour Patch

            if Iter_Patsch != Length_NPatch:
                for index in range(Length_NPatch):
                    Neigh_Patch[index] = data[indptr[indices[indptr[k]+s]]+index]
                sign = signum(Length_NPatch-Iter_Patsch)
                if sign == 1:
                    for h in range(Iter_Patsch,Length_NPatch):
                        Patch_mod[h] = 0
                    for w in range(Length_NPatch):
                        vec[w] = (Patch_mod[w]-Neigh_Patch[w])
                    for y in range(Length_NPatch):
                        Output[s] += pow(vec[y],2)/(2*pi*pow(sigma,2))*exp(-1/pow(sigma,2))
                elif sign == -1:
                    for h in range(Length_NPatch,Iter_Patsch):
                        Neigh_Patch[h] = 0
                    for w in range(Iter_Patsch):
                        vec[w] = Patch_mod[w]-Neigh_Patch[w]
                    for y in range(Iter_Patsch):
                        Output[s] += pow(vec[y],2)/(2*pi*pow(sigma,2))*exp(-1/pow(sigma,2))

                # Compute the output

                Im_weights_data[indptr[k]+s] = exp(-Output[s]/pow(Sigma_Weights[indices[indptr[k]]]\
                                                                    *Sigma_Weights[indices[indptr[k]+s]],2))
            else:

                # Compute Patch equal size

                for index in range(Length_NPatch):
                    Neigh_Patch[index] = data[indptr[indices[indptr[k]+s]]+index]
                for w in range(Iter_Patsch):
                    vec[w] = Patch_mod[w]-Neigh_Patch[w]
                for y in range(Iter_Patsch):
                    Output[s] += pow(vec[y],2)/(2*pi*pow(sigma,2))*exp(-1/pow(sigma,2))
                Im_weights_data[indptr[k]+s] = exp(-Output[s]/pow(Sigma_Weights[indices[indptr[k]]]\
                                                                    *Sigma_Weights[indices[indptr[k]+s]],2))
    Im_weights_data = normalize_adj(Im_weights_data,indptr)

    # Free Memory

    free(Patch_mod)


    return Im_weights_data, indices, indptr


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Non_Local_Means_cython_3D(cnp.ndarray[double,ndim = 3] Im,cnp.ndarray[double,ndim = 1] Sigma_Weights,cnp.ndarray[int,ndim = 1] indptr,cnp.ndarray[int,ndim = 1] indices\
                           ,(int,int,int) shape_img,double sigma,int window_size):
    cdef int m     = Im.shape[0]
    cdef int n     = Im.shape[1]
    cdef int z_dim = Im.shape[2]
    print(n*m*z_dim,'___Voxels')
    cdef int l,s,index,k,Iter_Patsch,Length_NPatch,h,y,f,w
    cdef double sign
    size = len(Im[:,:,:].reshape(-1)[indices])
    cdef cnp.ndarray[double,ndim = 1] data = Im[:,:,:].reshape(-1)[indices]
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Output = np.zeros(window_size*window_size**window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Neigh_Patch = np.zeros(window_size*window_size**window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] vec = np.zeros(window_size*window_size**window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Patch_1 = np.zeros(window_size*window_size**window_size)
    cdef cnp.ndarray[double, ndim=1, negative_indices=False] Im_weights_data = np.empty(size)
    cdef double* Patch_mod = <double*> malloc(sizeof(double) *window_size*window_size*window_size)

    
    for k in range(m*n*z_dim):
        
        # Length of Patch
        
        Iter_Patsch = indptr[k+1]-indptr[k]
        
        # Extract the Patch
        
        for index in range(Iter_Patsch):
            Patch_1[index]   = data[indptr[k]+index] 
            Patch_mod[index] = Patch_1[index]
        for l in range(Iter_Patsch):
            Output[l] = 0
            
    # Run Trough Patch Pixels
    
        for s in prange(Iter_Patsch,nogil = True):
            
            # Neighbour Patches
            
            Length_NPatch = indptr[indices[indptr[k]+s]+1]-indptr[indices[indptr[k]+s]]
            
           # Compute Neighbour Patch
        
            if Iter_Patsch != Length_NPatch:
                for index in range(Length_NPatch):
                    Neigh_Patch[index] = data[indptr[indices[indptr[k]+s]]+index] 
                sign = signum(Length_NPatch-Iter_Patsch)
                if sign == 1:
                    for h in range(Iter_Patsch,Length_NPatch):
                        Patch_mod[h] = 0
                    for w in range(Length_NPatch):
                        vec[w] = (Patch_mod[w]-Neigh_Patch[w])
                    for y in range(Length_NPatch):
                        Output[s] += pow(vec[y],2)/(2*pi*pow(sigma,2))*exp(-1/pow(sigma,2))
                elif sign == -1:
                    for h in range(Length_NPatch,Iter_Patsch):
                        Neigh_Patch[h] = 0
                    for w in range(Iter_Patsch):
                        vec[w] = Patch_mod[w]-Neigh_Patch[w]
                    for y in range(Iter_Patsch):
                        Output[s] += pow(vec[y],2)/(2*pi*pow(sigma,2))*exp(-1/pow(sigma,2))
                        
                # Compute the output
                
                Im_weights_data[indptr[k]+s] = exp(-Output[s]/pow(Sigma_Weights[indices[indptr[k]]]\
                                                                    *Sigma_Weights[indices[indptr[k]+s]],2))
            else:
                
                # Compute Patch equal size
                
                for index in range(Length_NPatch):
                    Neigh_Patch[index] = data[indptr[indices[indptr[k]+s]]+index] 
                for w in range(Iter_Patsch):
                    vec[w] = Patch_mod[w]-Neigh_Patch[w]
                for y in range(Iter_Patsch):
                    Output[s] += pow(vec[y],2)/(2*pi*pow(sigma,2))*exp(-1/pow(sigma,2))
                Im_weights_data[indptr[k]+s] = exp(-Output[s]/pow(Sigma_Weights[indices[indptr[k]]]\
                                                                    *Sigma_Weights[indices[indptr[k]+s]],2))
    Im_weights_data = normalize_adj(Im_weights_data,indptr)
    
    # Free Memory
    
    free(Patch_mod)

    
    return Im_weights_data, indices, indptr



ctypedef fused TYPE:
    double

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

# One way with Pointer for nogil, Second way use memoryview

cdef inline double Nonlocal_Gradient_Cython(TYPE* patch_data,\
                                            TYPE* patch_weights\
                       ,double beta,int Length_Patch) nogil:

    cdef int k
    cdef double Output = 0.0




    for k in range(Length_Patch):
            Output = Output + patch_weights[k]*pow(patch_data[k]-patch_data[0],2)

    #print(Output)

    Output = sqrt(pow(beta,2) + Output)

    return Output



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

def Gamma_Weights_Cython(cnp.ndarray[double,ndim = 1,negative_indices = False] data_weights,\
                         cnp.ndarray[int,ndim = 1,negative_indices = False] indptr, \
                         cnp.ndarray[int,ndim = 1,negative_indices = False] indices, \
                         cnp.ndarray[double,ndim = 1] f,double p,double beta,int window_size):

    ' Initialization '

    ' Output Variables '


    cdef cnp.ndarray[double, ndim = 1,negative_indices = False] data_gamma = np.zeros_like(data_weights)

    cdef int N                    = len(indptr)

    cdef double grad_value


    ' Variables for FOR-Loops '

    cdef int k,l,j,i,ind,s

    ' Initilize Threads '

    cdef int tid, num_threads = omp_get_max_threads()

    ' Allocate Memory to auxilary Variables '

    cdef double* patch_pixel           = <double*> malloc(sizeof(double)* window_size*window_size*num_threads)
    cdef double* patch_weights         = <double*> malloc(sizeof(double) * window_size * window_size*num_threads)
    cdef double* patch_data            = <double*> malloc(sizeof(double) * window_size * window_size*num_threads)
    cdef double* patch_weights_pixel   = <double*> malloc(sizeof(double) * window_size * window_size*num_threads)
    cdef int* patch_indices            = <int*> malloc(sizeof(int) * window_size * window_size*num_threads)

    cdef int Length_Patch
    cdef int Length_Patch_Value
    cdef int Length_patch_weights
    cdef double* Z_pixel
    cdef double* Z_weights
    cdef double* Z_weights_2
    cdef double* Z_data
    cdef int* Z_indices

    print('Compute Gamma Weights')

    for k in prange(N-1,nogil = True, schedule = 'static'):

        ' Call for thread iterarator index '

        tid = omp_get_thread_num()

        Length_Patch_Value = indptr[k+1]-indptr[k]

        ' Assumbling Helper Arrays '

        for l in prange(Length_Patch_Value,schedule = 'static'):

            tid = omp_get_thread_num()

            Z_pixel              = &patch_pixel[window_size*window_size*tid]
            Z_pixel[l]           = f[indices[indptr[k]+l]]
            Z_weights            = &patch_weights_pixel[window_size*window_size*tid]
            Z_weights[l]         = data_weights[indptr[k]+l]
            Z_indices            = &patch_indices[window_size*window_size*tid]
            Z_indices[l]         = indices[indptr[k]+l]

        grad_value  = Nonlocal_Gradient_Cython(Z_pixel,Z_weights,beta,Length_Patch_Value)

        for j in range(Length_Patch_Value):

            if indptr[Z_indices[j]+1] < indptr[N-1]:

                ' Assamble Patch Data Help Array '

                Length_Patch = indptr[Z_indices[j]+1]-indptr[Z_indices[j]]

                for i in range(Length_Patch):
                    Z_data                  = &patch_data[0]
                    Z_data[i]               = f[indices[indptr[Z_indices[j]]]+i]
                    Z_weights_2             = &patch_weights[0]
                    Z_weights_2[i]          = data_weights[indptr[Z_indices[j]]+i]

                Length_patch_weights    = indptr[Z_indices[j]+1]-indptr[Z_indices[j]]

                for ind in range(Length_patch_weights):

                    data_gamma[indptr[Z_indices[j]]+ind] = data_weights[indptr[k]+j]*\
                            (pow(Nonlocal_Gradient_Cython(Z_data,Z_weights_2,beta,Length_Patch),p-2)\
                            +pow(grad_value,p-2))

    print('Iteration finished')

    ' Free Alocated Memory '

    free(patch_pixel)
    free(patch_weights)
    free(patch_data)
    free(patch_weights_pixel)
    free(patch_indices)

    return data_gamma

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Gauss_Jacobi_Iteration_Cython(cnp.ndarray[double, ndim = 1]f_0,cnp.ndarray[double, ndim = 1,\
                                  negative_indices = False] weights,\
                                  cnp.ndarray[int, ndim = 1,negative_indices = False] indices,\
                                  cnp.ndarray[int, ndim = 1,negative_indices = False] \
                                  indptr, cnp.ndarray[double, ndim = 1] f_iter, double sigma_t, double p,\
                                  int Iter,int window_size,double beta):

    ' Intialization '

    cdef cnp.ndarray[double,ndim = 1] data_gamma = Gamma_Weights_Cython(weights,indptr,indices,f_iter,p\
                                                                        ,beta,window_size)



    cdef int N = len(f_iter)
    cdef double lambda_reg

    if Iter == 0:

        lambda_reg = 1/pow(sigma_t,2)

    else:

        lambda_reg = Update_Lambda_Cython(f_iter,f_0,sigma_t,data_gamma,indices,indptr)



    #if Iter == 0:
    #    lambda_reg = 1/sigma_t**2
    #else:
    #    lambda_reg = Update_Lambda(f_iter,f_0,sigma_t,data_gamma)
    cdef int k,l

    for k in range(N-1):

        f_iter[k] = (lambda_reg*f_0[k]+np.sum(data_gamma[indptr[k]:indptr[k+1]]*\
            f_iter[indices[indptr[k]:indptr[k+1]]],axis=0))\
            /(lambda_reg+np.sum(data_gamma[indptr[k]:indptr[k+1]],axis=0))



    return f_iter

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Update_Lambda_Cython(cnp.ndarray[double, ndim = 1] f_iter,cnp.ndarray[double, ndim = 1] f_0,double sigma_t,\
                  cnp.ndarray[double, ndim = 1] data_gamma,cnp.ndarray[int, ndim = 1,negative_indices = False]\
                  indices,cnp.ndarray[int, ndim = 1,negative_indices = False] indptr):

    ' Initialization '

    cdef int N        = len(f_iter)
    cdef double Lambda_t = 0.0
    cdef int k,l
    cdef double Patch_sum = 0

    for k in prange(N-1,nogil = True):

        Patch_sum = 0

        for l in range(indptr[k+1]-indptr[k]):

            Patch_sum += data_gamma[indptr[k]+l]*(-f_iter[indices[indptr[k]+l]]+f_iter[indices[indptr[k]]])

        Lambda_t += (1/(pow(sigma_t,2)*N))*(f_iter[k]-f_0[k])*Patch_sum


    return Lambda_t