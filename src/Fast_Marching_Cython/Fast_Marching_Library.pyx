' Import Modules '

import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange
from libc.math cimport fabs
from libc.stdlib cimport malloc, free
from libc.math cimport signbit,exp, pi,pow, fmax,fmin,sqrt,abs

' Fast Marching Implementation Code '
' Publication: 2014 Xavier Desquesnes Title: Eikonal equation adaption on '
' weighted graphs: fast geometric diffusion process for local and non-local image and data processing '

cdef double signum(double x) nogil:
    if x > 0.0:
        return 1.0
    elif x < 0.0:
        return -1.0
    else:
        return 0.0

'Sort List Subroutine '
'Returns: Array Sorted Indices,Sorted Values'

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef Sort(cnp.ndarray[double,negative_indices = True] List,cnp.ndarray[int,negative_indices = False] Indices ):
    cdef int Length              = List.shape[0]
    cdef int S_Ind               = 0
    cdef cnp.ndarray[double,ndim = 1,negative_indices = False] Sorted_List = np.zeros(Length)
    cdef cnp.ndarray[int,ndim = 1,negative_indices = False] Sorted_Indices = np.zeros(Length,dtype = np.int32)

    'Help Variables'

    cdef int l,k,s
    cdef double S_var = 0

    'Copy Array'

    for s in prange(Length,nogil = True):
        Sorted_List[s]    = List[s]
        Sorted_Indices[s] = s

    for k in range(Length):
        S_var = Sorted_List[k]
        S_Ind = Sorted_Indices[k]

        for l in range(k+1,Length):
            if S_var > Sorted_List[l]:
                S_var = Sorted_List[l]
                S_Ind = Sorted_Indices[l]

                'Outmemory the smallest value'

                Sorted_List[l] = Sorted_List[k]
                Sorted_List[k] = S_var
                Sorted_Indices[l] = Sorted_Indices[k]
                Sorted_Indices[k] = S_Ind
    for s in prange(Length,nogil = True):
        Indices[s] = Indices[Sorted_Indices[s]]

    return Sorted_List,Indices

'Find Index Subroutine '

@cython.boundscheck(False)
@cython.wraparound(False)

cdef Find_Index(cnp.ndarray[int,ndim = 1] Array, int Val):

    'Auxiliary Variables'

    cdef int Length = Array.shape[0]
    cdef int Index, k

    for k in range(Length):
        if Array[k] ==  Val:
            Index = k
            break
        else:
            continue
    return Index

' Check if Index in Array '

@cython.boundscheck(False)
@cython.wraparound(False)

cdef Check_Index(cnp.ndarray[int,ndim = 1] Array, int Index):

    'Auxiliary Variables'

    cdef int Length = Array.shape[0]
    cdef int Response = 0
    cdef int k

    for k in range(Length):
        if Array[k] == Index:
            Response = 1
            break
        else:
            continue
    return Response

' Append Index to last position in an array Array '

@cython.boundscheck(False)
@cython.wraparound(False)

cdef Append_Ind(cnp.ndarray[int,ndim = 1] Array, int Index):
    Array = np.append(Array,Index).astype(np.int32)
    return Array



' Delete Index on first position in an array Array '

@cython.boundscheck(False)
@cython.wraparound(False)

cdef Delete_Ind(cnp.ndarray[int,ndim = 1] Array, int Index):

    Array = np.delete(Array,Index).astype(np.int32)
    return Array

' Append Value to last position in an Value Array '

@cython.boundscheck(False)
@cython.wraparound(False)

cdef Append_Val(cnp.ndarray[double,ndim = 1] Array_Val, double Val):
    cdef cnp.ndarray[double,ndim  = 1] Up_Array = np.append(Array_Val,Val)
    return Up_Array






'Subroutine for value of one vertex while other are fixed'

' Returns:   Updated Value of vertex with minal time within the Narrowband '



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


def Local_Solution(cnp.ndarray[double,ndim = 1] f_data,cnp.ndarray[int,ndim = 1] f_indptr,\
                   cnp.ndarray[int,ndim = 1] f_index\
                   ,cnp.ndarray[double,ndim = 1] a_i, double p, \
                   cnp.ndarray[double,ndim = 1] h_i,int vertex):

    ' Compute the gradient on the grid (Characterization of P(x,y)) '

    ' a_i __ Values within the Narrow_Band cut Neighbouhood '

    cdef double Grad_min = Compute_Grad_norm_minus(a_i,p,h_i)
    cdef int End_Point = a_i.shape[0]
    cdef double Local_Solution = 100 # Set large to go to the loop
    cdef int l,s,j,index,k,Iter_Patsch,Length_NPatch,h,y,f,w
    cdef int m = 0

    if p == 1.0:
        while Local_Solution >= a_i[m+1] and m <= End_Point-1:
            Local_Solution = Solve_Subproblem(a_i,Grad_min,p,h_i,vertex)
            m += 1
    if p == 2.0:
        while Local_Solution >= a_i[m+1] and m <= End_Point-1:
            Local_Solution = Solve_Subproblem(a_i,Grad_min,p,h_i,vertex)
            m += 1
    return Local_Solution

' Subroutine for minimal time arrival value '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef Solve_Subproblem(cnp.ndarray[double,ndim = 1] a_i,double Norm, double p,cnp.ndarray[double,ndim = 1] h_i,int local_vertex):
    cdef int End_Point = a_i.shape[0]
    cdef double Sol = 0
    cdef double Term,Value_Inner_Loop
    cdef h_i_normalized = 0
    cdef int l,s,i,j,index,k,Iter_Patsch,Length_NPatch,h,y,f,w

    if p == 1.0:
        for l in range(End_Point):
            Sol = Sol+ a_i[l]/h_i[l]+Norm
        for s in range(End_Point):
            h_i_normalized = h_i_normalized + 1.0/h_i[s]
    elif p == 2.0:
        for l in range(End_Point):
            for i in range(End_Point):
                Value_Inner_Loop = 0
                for j in range(i+1,End_Point-1):
                    Value_Inner_Loop += pow((a_i[i]-a_i[j]),2.0)/(pow(1/sqrt(h_i[i]),2.0)*pow(h_i[j],2.0))
                Term = Term + pow(Norm,2.0)/pow(1/sqrt(h_i[i]),2.0)-Value_Inner_Loop
            Sol = Sol+ a_i[l]/pow(1/h_i[l],2.0)+sqrt(Term)
        for s in range(End_Point):
            h_i_normalized = h_i_normalized + 1.0/pow(h_i[s],2.0)
    return Sol/h_i_normalized


' Norm Subroutine with Ansitropic weights '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)



cdef Compute_Grad_norm_minus(cnp.ndarray[double,ndim = 1] Neigh_Data,double p,cnp.ndarray[double,ndim = 1] h_i):
    cdef int End_Point = len(Neigh_Data)
    cdef double Norm_Final = 0
    cdef int l,s,k,Iter_Patsch,Length_NPatch,h,y,f,w
    cdef cnp.ndarray[double,ndim = 1] Norm =np.zeros(End_Point)
    'Iterate over Pixels Neigh_Data[0] == center pixel'
    for l in prange(End_Point,nogil = True):
        Norm[l] =  (-1.0*pow(sqrt(1/h_i[l]),p)*fmin((Neigh_Data[l]-Neigh_Data[0]),0)+fmax((Neigh_Data[l]-Neigh_Data[0]),0))
    for s in range(End_Point):
        Norm_Final += Norm[s]
    Norm_Final = pow(Norm_Final,1.0/p)
    return Norm_Final

' Subroutine for updating Narrow_Band '

#from libcpp.list cimport list as cpplist # Appending a List Routine


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

def Update_Narrow_Band(cnp.ndarray[double,ndim = 1] data,cnp.ndarray[int,ndim = 1] indptr, cnp.ndarray[int,ndim = 1] indices,\
                       cnp.ndarray[int,ndim = 1] Far_Away,cnp.ndarray[double,ndim = 1] Narrow_Band_Val,
                       cnp.ndarray[int,ndim = 1] Narrow_Band,cnp.ndarray[double,ndim = 1] Image,
                       cnp.ndarray[int,ndim = 1] Indices,cnp.ndarray[double,ndim = 1] Image_Data_Graph,\
                       cnp.ndarray[double,ndim = 1] h_i):

    'Create Variables auscilary Variables'

    cdef int j,del_index,
    cdef double Hepl        = 0
    cdef int Length_In      = Indices.shape[0]
    cdef int Len            = Narrow_Band.shape[0]
    cdef cnp.ndarray[int,ndim =1] Narrow_Band_Shadow = np.zeros(Len,dtype = np.int32)

    for j in prange(Len,nogil = True):
        Narrow_Band_Shadow[j] = Narrow_Band[j]



    ' New Index, Value List within the Narrow Band '


    Stop = 0
    for j in range(Length_In):

            if Check_Index(Narrow_Band_Shadow,Indices[j]) == 1:
                Help = Local_Solution(Image_Data_Graph,indptr,\
                                      indices,a_i = Image_Data_Graph[indptr[j]:indptr[j+1]],
                                            p = 1.0, h_i = data[indptr[Indices[j]]:indptr[Indices[j]+1]],vertex = j)

                if Help < Image[Indices[j]]:
                    Image[Indices[j]]= Help

            elif Check_Index(Far_Away,Indices[j]) == 1:
                del_index       = Find_Index(Far_Away,Indices[j])
                Far_Away        = Delete_Ind(Far_Away,del_index)
                Narrow_Band     = Append_Ind(Narrow_Band,Indices[j])
                Narrow_Band_Val = Append_Val(Narrow_Band_Val,Image[Indices[j]])


            ' Update Position in Narrow Band by Sorting'
            'Sort Neighbours'

            #Narrow_Band_Val,Narrow_Band     =  Sort(Narrow_Band_Val,Narrow_Band)

    return Narrow_Band,Narrow_Band_Val,Far_Away,Image



