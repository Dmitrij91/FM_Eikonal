import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport signbit,exp, pi, fmax,sqrt,abs

import time

cdef extern from "math.h"nogil:
    cdef double sqrt(double)
    cdef double pow_double "pow"(double,double)
    cdef double f_min_double "fmin"(double,double)
    cdef double f_max_double "fmax"(double,double)
    
cdef extern from "math.h":
    int pow(int,int) nogil
    int fmin(int,int) nogil

cdef extern from "pyport.h":
    double Py_HUGE_VAL

cdef double inf = Py_HUGE_VAL
    
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

cdef Find_Index(int* Array, int Val,Length):
    
    'Auxiliary Variables'
    
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

cdef void Check_Index(int [:] Array, int Index,int* response) nogil:
    
    'Auxiliary Variables'
    
    cdef int Length = Array.shape[0]
    cdef int k
    
    for k in range(Length):
          
        if Array[k] == Index:
            
            response[0] = 1
            
            break
            
# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(False)

# cdef void Check_Index_pt(int* Array, int Index, int* response,int Length,int* del_index) nogil:
    
#     'Auxiliary Variables'

#     cdef int k
    
#     for k in range(Length):
          
#         if Array[k] == Index:
            
#             response[0] = 1
            
#             del_index[0] = k
            
#             break
            
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

cdef void Check_Index_pt(bint* Narr_Band_ind,int Index, int* response) nogil:
    
    'Auxiliary Variables'

          
    if Narr_Band_ind[Index] == True:
            
        response[0] = 1
            

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
       
                
                 
' Subroutine for value of one vertex while other are fixed '
' Returns:   Updated Value of vertex with minimal arrival time within the Narrowband '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)


cdef double Local_Solution(double* a_i, double p,double* h_i,int vertex, int End_Point):
    
    ' Compute the gradient on the grid (Characterization of P(x,y)) '
    
    ' a_i __ Values within the Narrow_Band cut Neighbouhood '
    
    # Allocate Array for gradient values 
    
    cdef int k
    
    cdef double Grad_min 
    
    Grad_min = Compute_Grad_norm_minus(a_i,p,h_i,End_Point)
    
    cdef double Local_Solution = 100 # Set large to go to the loop
    cdef int l,s,j,index,Iter_Patsch,Length_NPatch,h,y,f,w
    cdef int m = 0
    
    if p == 1.0:
        
        while Local_Solution >= a_i[m+1] and m <= End_Point-1:
            
                
            Local_Solution = Solve_Subproblem(a_i,Grad_min,p,h_i,vertex,End_Point)   
        
            m += 1
    
    elif p == 2.0:
        
        while Local_Solution >= a_i[m+1] and m <= End_Point-1:
            
            Local_Solution = Solve_Subproblem(a_i,Grad_min,p,h_i,vertex,End_Point)
            
            m += 1
    return Local_Solution

' Subroutine for minimal time arrival value '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)


cdef double Solve_Subproblem(double* a_i,double Norm, double p,double* h_i,int local_vertex, int End_Point)nogil:
    
    cdef double Sol = 0
    
    cdef double Term,Value_Inner_Loop 
    
    cdef double h_i_normalized = 0
    
    cdef int l,s,i,j,index,k,Iter_Patsch,Length_NPatch,h,y,f,w
        
    if p == 1.0:
    
        for l in range(End_Point):
            
            Sol += a_i[l]*sqrt(h_i[l])+Norm
            
            h_i_normalized += sqrt(h_i[l])
                
    elif p == 2.0:
        
        for l in range(End_Point):
            
            for i in range(End_Point):
                
                Value_Inner_Loop = 0
                
                for j in range(i+1,End_Point-1):
                    
                    Value_Inner_Loop += pow_double((a_i[i]-a_i[j]),2.0)/(pow_double(1/sqrt(h_i[i]),2.0)\
                                                                         *pow_double(h_i[j],2.0))
                    
                Term = Term + pow_double(Norm,2.0)/pow_double(1/sqrt(h_i[i]),2.0)-Value_Inner_Loop
                
            Sol = Sol+ a_i[l]/pow_double(1/h_i[l],2.0)+sqrt(Term)
            
        for s in range(End_Point):
            
            h_i_normalized += 1.0/pow_double(h_i[s],2.0)
            
    return Sol/h_i_normalized 

########################################
# Todo Improve Center_Pixel
"""

Norm Subroutine with Ansitropic weights 
 Compute gradient Term (F^+||grad^+u(x,t)||_p^p+F^-||grad^-u(x,t)||_p^p) 
 F = 1/P = sqrt(1/h_i) --> Speed of front wave propagation here h_i is the weight\
                                expressed by distance between nodes

"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline double Compute_Grad_norm_minus(double* Neigh_Data,double p,double* h_i,\
                                           int End_Point)nogil:

    
    cdef double Norm_Final = 0
    
    cdef int Index_center = (End_Point -1)/2  
    
    cdef double* Norm = <double*>malloc(sizeof(double)*End_Point) 
    
    cdef int l,s,Iter_Patsch,Length_NPatch,h,y,f,w
    
    'Iterate over Pixels Neigh_Data[0] == center pixel'
    
    for l in range(End_Point):
        
        Norm_Final +=  (pow_double(sqrt(h_i[l]),p))*f_max_double((Neigh_Data[Index_center]-Neigh_Data[l]),0)
    
        
    Norm_Final = pow_double(Norm_Final,1.0/p)
    
    return Norm_Final

' Subroutine for updating Narrow_Band '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

cdef inline (int*,double*) Update_Narrow_Band( int* indptr, int* indices, \
                    double* Narrow_Band_Val, int* Narrow_Band, double* Image,int* Indices,\
                    double[:] Image_Data_Graph, double* h_i,int Length_In,int* Length_Narrow,\
                    int Length_Far_Away,int Min_Index, int* levels,int* count,double* Image_Data_Graph_pt,\
                    double* h_i_pt,bint* Indices_Far_Away_bool, bint* Indices_Narrow_bool):
    
    'Create Variables auscilary Variables'
    
    cdef int j,i,k,l
    
    cdef int del_index = 0
    
    cdef double Help   = 0

    cdef int* Narrow_Band_Up 
    
    cdef double* Narrow_Band_Val_Up
    
    ' New Index, Value List within the Narrow Band '
        
    cdef int Stop = 0
    
    cdef int Response, Response_1
    
    cdef int number = Length_Narrow[0]
    
    cdef double Neigh_value
    
    cdef int count_pt  = count[0]
    
    cdef int levels_pt = levels[0]
    
    cdef int Length_neigh_j
    
    cdef int* Narrow_Band_pt = &Narrow_Band[0]
    
    cdef double* Narrow_Band_Val_pt = &Narrow_Band_Val[0]
    
    cdef int del_index_1 = 0
    
    cdef int index_up 
    
    for j in range(Length_In):
        
        Length_neigh_j = indptr[indices[indptr[Min_Index]+j]+1]-indptr[indices[indptr[Min_Index]+j]]
        
        Response    = 0
        
        Response_1  = 0
        
        del_index   = 0
        
        del_index_1 = 0
        
        
        ' Check if the index is in Narrow_Band then Update data '

        Check_Index_pt(Indices_Narrow_bool,Indices[j],&Response)
        
        Check_Index_pt(Indices_Far_Away_bool,Indices[j],&Response_1)
            
        for k in range(Length_neigh_j):
                    
            Image_Data_Graph_pt[k] = Image_Data_Graph[indptr[indices[indptr[Min_Index]+j]]+k]
            h_i_pt[k] = h_i[indptr[indices[indptr[Min_Index]+j]]+k]


        if Response == 1:
            
            
            Help = Local_Solution(&Image_Data_Graph_pt[0],1.0,h_i_pt, j,Length_neigh_j)
            
                

            if Help < Image[indices[indptr[Min_Index]+j]]:

                Image[indices[indptr[Min_Index]+j]] = Help
            
            
        elif Response_1 == 1:

            'Set Index in Far_Away to -1 to ignore it '
            
            Indices_Far_Away_bool[Indices[j]] = False


            ' Find Index in Far_Away with value Indices[j] '
                
            Neigh_value         = Image[indices[indptr[Min_Index]+j]]
            
            number              = pow(2,levels_pt)

            ' If indices positions exceed add new level '

            if count_pt >= number:

                ' Update available Reference size '


                number        = pow(2,levels_pt+1)

                ' Update Binary Tree '

                Narrow_Band_Val_Up,Narrow_Band_Up = add_level(Narrow_Band_pt, Narrow_Band_Val_pt,\
                                                            &levels_pt,add = 1)

                levels_pt += 1
            
            else: 

                Narrow_Band_Val_Up = Narrow_Band_Val_pt

                Narrow_Band_Up     = Narrow_Band_pt


                ' Free Narrow_Band and update '

                    
            index_up = (pow(2,levels_pt) - 1) + count_pt 

            Narrow_Band_Up[count_pt]        = indices[indptr[Min_Index]+j]

            Narrow_Band_Val_Up[index_up] = Neigh_value

            count_pt += 1

            ' Allocate Memory Again '    


            nodes_values_pt     = &Narrow_Band_Val_Up[0]

            ' Release Memory for Updated Narrow_Band '

            ' Update via for old number_size '


            update_one(&index_up,nodes_values_pt,levels_pt)
            
            Narrow_Band_pt = &Narrow_Band_Up[0]
            
            Narrow_Band_Val_pt = &Narrow_Band_Val_Up[0]
        
    count[0]  = count_pt
    levels[0] = levels_pt
    Length_Narrow[0] = number
    
    return Narrow_Band_pt,Narrow_Band_Val_pt

""""

Returns indices_1\indices indices_1 of type arange(Length_graph)

"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

cdef void set_diff_cython(int [:] indices,bint* Far_Away_pt,int Length_Indices):

    cdef int k
    
    for k in range(Length_Indices):
        
        Far_Away_pt[indices[k]] = False
        
' If index is in Far_Away return the position of the index within the Far_Away array ' 

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(False)

# cdef void set_diff_cython(int [:] indices,int* Far_Away_pt,int Length_Graph,int L):

#     cdef int k,l  
#     cdef int count = 0
#     cdef bint Response
    
#     for k in range(Length_Graph):
        
#         Response = 0
        
#         for l in range(L):
          
#             if indices[l] == k:
                
#                 Response = 1
                
                
#                 break
        
#         if Response == 0:
            
#             Far_Away_pt[count] = k
            
#             count = count + 1

# @cython.boundscheck(False)
# @cython.wraparound(False)
# @cython.cdivision(False)

# cdef void Intersect_Cython(int* Far_Away_ind,int Check_index,int* Response,int Length_Far_Away\
#                            ,int* Index)nogil:

#     cdef int k
    
#     for k in range(Length_Far_Away):
        
#         if Check_index == Far_Away_ind[k]:
            
#             Response[0] = 1
            
#             Index[0] = k
            
#             break 


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

cdef void Intersect_Cython(bint* Far_Away_ind,int Check_index,int* Response)nogil:
        
    if Far_Away_ind[Check_index] == True:
            
        Response[0] = 1


' Function to remove index from the binary tree '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline (int*,double*) Rem_Index_Binary_tree(int* narr_band_ref, double* narr_band_val,\
                                int rem_index,int* levels,int* count,int* size_narrow,int min_levels)nogil:
        
    
    cdef int k
        
    ' Get index  '
        
    # Start position
        
    cdef int Start_index = (1 << levels[0]) - 1 
        
    # End position
        
    cdef int End_index = Start_index + count[0] - 1

    # Define relative positions last level
                
    cdef int Ind_relative = rem_index - Start_index
        
    # Relative End Pos
        
    cdef int Rel_end = count[0] - 1
        
    ' Define Pointer to allocated arrays '
        
    cdef int* narr_band_ref_pt
        
    cdef double* narr_band_val_pt
    
    cdef int levels_pt 
        
    ' Copy last index and value into the index to be removed '
        
    narr_band_val[rem_index]    = narr_band_val[End_index]
    narr_band_ref[Ind_relative] = narr_band_ref[Rel_end]
        
    ' Set last index to infinity '
        
    narr_band_val[End_index] = inf
            
    ' Update Binary Tree from last position '
        
    count[0] -= 1
    
    if (levels[0] > min_levels) and (count[0] < (1 << (levels[0]-2))):
        
        narr_band_ref_pt = &narr_band_ref[0] 
            
        narr_band_val_pt = &narr_band_val[0]
        
        levels_pt        = levels[0]
        
        narr_band_val_pt,narr_band_ref_pt = add_level(narr_band_ref_pt, narr_band_val_pt, &levels_pt,add = -1) 
        
        levels[0]      = levels[0] - 1
        
        size_narrow[0] = pow(2,levels[0])
        
    else:
          
        
        levels_pt        = levels[0]
        
        narr_band_ref_pt = &narr_band_ref[0] 
            
        narr_band_val_pt = &narr_band_val[0]
        
        update_one(&rem_index,narr_band_val_pt,levels_pt)
        
        update_one(&End_index,narr_band_val_pt,levels_pt)
         
    
    return narr_band_ref_pt, narr_band_val_pt

' Function to extract the minimal value and index from the binary tree '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline (double,int*,double*) Get_Index_Narrow_Cython(int* levels, double* current_values,\
                    int* current_reference,int min_levels,int* popped_ref,int* count,int* size_narrow) nogil:
    
    ' Start at Level 1 '
    
    cdef int level
    cdef int i = 1
    cdef int local_index
    cdef double value
    
    cdef int* current_reference_pt
    
    cdef double* current_values_pt
    
    ' Iterate over whole binary tree '
    
    for level in range(1, levels[0]):
        
        ' Check the child nodes and move to the indices of the smallest '
        
        if current_values[i] <= current_values[i+1]:
            
            i = i * 2 + 1  
        
            ' Check next parent node '
        
        else:
            
            i = (i+1) * 2 + 1  
        
    ' Pick the index with the lowest value from the next level '
        
    if current_values[i] <= current_values[i+1]:
            
        i = i
        
    else:
            
        i += 1
    
    ' Get Local Reference index from next level '
        
    local_index = i - ((1 << levels[0]) - 1) 
                                                
    ' Copy value '
        
    value = current_values[i]
        
    ' Return current reference '
     
    popped_ref[0]        = current_reference[local_index]
    
    ' Remove index '
        
    current_reference_pt = &current_reference[0]
        
    current_values_pt    = &current_values[0]
    
    cdef int count_pt 
    
    count_pt = count[0]
    
    cdef int levels_pt      = levels[0]
    
    cdef int size_narrow_pt = size_narrow[0]
    
    current_reference_pt,current_values_pt = Rem_Index_Binary_tree(current_reference_pt,current_values_pt\
                                                        ,i,&levels_pt,&count_pt,&size_narrow_pt,min_levels)
    
    
    count[0]       = count_pt
    levels[0]      = levels_pt
    size_narrow[0] = size_narrow_pt
    
    return value,current_reference_pt,current_values_pt            

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void update_one(int* index_up,double* old_values,int current_levels) nogil:
    
    
    if index_up[0] % 2 == 0:
        
        index_up[0] -= 1
    
    ' Update Binary Tree from bottom '
    
    cdef int index
    cdef int level
    
    for level in range(current_levels, 1, -1):
        
        index = (index_up[0]-1) // 2

        if old_values[index_up[0]] < old_values[index_up[0]+1]:
                
            old_values[index] = old_values[index_up[0]]
        
        else:
            
            old_values[index] = old_values[index_up[0]+1]
        
        ' Update index_up to move to the next top level '
        
        if index % 2:
            
            ' if uneven '
            
            index_up[0] = index
            
        else:
            
            'Make uneven'
            
            index_up[0] = index - 1

' Function for ordering the binary tree '
        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
        
cdef void update_Binary_Tree(double* new_values, int new_levels) nogil:
    
    ' Initialization '
    
    
    cdef int node_index, level_start, level, next_node, i

    ' Update tree via reverse order '

    for level in range(new_levels, 1, -1):
        
        level_start = pow(2,level) - 1
        
        ' Number of nodes on level_start '
        
        next_node   = level_start+1
        
        ' Iterate over nodes '
        
        for i in range(level_start, level_start+next_node, 2):
            
            ' Parent Node located one level above '
            
            node_index = (i-1) // 2
            
            ' Check wich of legt or right child nodes to be moved to the top level '
            
            if new_values[i] < new_values[i+1]:
                
                ' Move Left child ' 
                
                new_values[node_index] = new_values[i]
            
            else:
                
                ' Move right Child '
                
                new_values[node_index] = new_values[i+1]
            
"""

Function to add one level to a binary tree:

Input: 1) Number of new Levels
       2) reference, values arrays to be modified
       3) Number of old Levels
       4) new empty reference and values arrays initialized before function call
       
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline (double*,int*) add_level(int* old_reference, double* old_values, int* old_levels,int add)nogil:
     
    ' Initialization '
        
    cdef int i,k, level_start_old, level_start_new , n
    
    cdef int new_levels
    
    if add == 1:
    
        ' Extend by one level '
        
        new_levels = old_levels[0] + 1
        
    elif add == -1:
    
        ' Descrease by one level ' 
    
        new_levels = old_levels[0] - 1
      
    cdef int nodes_new        = pow(2,new_levels)
    
    
    Narrow_Band_Val_Up    = <double*>malloc(nodes_new * 2 * sizeof(double))

    Narrow_Band_Up        = <int*>malloc(nodes_new * sizeof(int))
    
    ' Assign elements to infinity '
    
    for i in prange(nodes_new * 2,nogil = True):
        
        Narrow_Band_Val_Up[i] = inf
    
    for i in prange(nodes_new,nogil = True):
        
        Narrow_Band_Up[i] = -1

    ' Copy old data '
    
    level_start_new = pow(2,new_levels) - 1
    
    level_start_old = pow(2,old_levels[0]) - 1 
    
    n = fmin(level_start_new+1, level_start_old+1)
    
    ' Write old ref and vals from last old level to last new level '
    
    for i in prange(n,nogil = True):
        
        Narrow_Band_Val_Up[level_start_new+i] = old_values[level_start_old+i]
    
    for i in prange(n,nogil = True):
    
        Narrow_Band_Up[i] = old_reference[i]
    
    free(old_values)
    free(old_reference)
    
    ' Update the binary tree from bottom to top level '

    cdef double* new_value_pt 
    
    new_value_pt = &Narrow_Band_Val_Up[0]
    
    update_Binary_Tree(new_value_pt,new_levels)
    
    return &Narrow_Band_Val_Up[0],&Narrow_Band_Up[0]
            

' Variables Settings '

' Image             ______  Input Image 2D 3D reshape to 1D array' 
' Ad                ______  Graph as 1D array in a matrix sparse format containing data driven weights ' 
' Seeds_indices     ______  Start for Tracking Points setted to zero  '
' Distance          ______  Time Distance Output as 1 D array (Trace back via Ad Matrix to 2D,3D data)'
' Active List       ______  Optimal values of time '
' Narrow_Band       ______  Points on Active List complnement such that there exists one Neighbour within Active_List'
' Narrow_Band_Val   ______  1D array for keeping the current values for indices'
' Far_Away          ______  Ramaining Grid-Points'
' Image_Data_Graph  ______  Adopted weights values of the input image to the grid (a_i input) '


' Termination Criterium :      Empty Narrow_band '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Eikonal_Eq_Solve_Cython(double [:] Image,double [:] data,int [:] indices,int [:] indptr,int[:] Seeds_indices, int [:] shape_ad):
    
    assert len(data.shape) == len(indices.shape), ' data and indices not of equal shape '
    
    cdef int Length_Graph = Image.shape[0]
    cdef int k,l,i
    
    for k in range(Seeds_indices.shape[0]):
        
        Image[Seeds_indices[k]] = 0
    
    cdef double [:] Image_Data_Graph = Image.base[indices]   
    
    
    ' Initialization '
    
    cdef cnp.ndarray[ndim = 1, dtype = int,negative_indices = False] Active_List         = np.zeros(\
                                                                 Length_Graph,dtype = np.int32)
    cdef cnp.ndarray[ndim = 1, dtype = double,negative_indices = False] Active_List_Val  = np.zeros(\
                                                                             Length_Graph,)
    
    
    ' Create List for Intersect Indices '
    
    cdef bint* Indices_Narrow_bool = <bint*>malloc(sizeof(bint)*Length_Graph)
    
    cdef bint* Indices_Far_Away_bool = <bint*>malloc(sizeof(bint)*Length_Graph)
    
    for k in prange(Length_Graph,nogil = True):
        
        Indices_Narrow_bool[k] = False 
        
        Indices_Far_Away_bool[k] = True
    
    for k in range(Seeds_indices.shape[0]):
        
        Active_List[k]          = Seeds_indices[k]
        
        Active_List_Val[k]      = Image[Seeds_indices[k]] 
        
        ' Remove Index from boolean Far_Away array '
        
        Indices_Far_Away_bool[Seeds_indices[k]] = False
        
    cdef int Far_Away_Length = Length_Graph-Seeds_indices.shape[0] 
    
    #cdef int* Far_Away   = <int*>malloc(sizeof(int)*Far_Away_Length)
    
    #for k in prange(Far_Away_Length,nogil = True):
        
    #    Far_Away[k] = -1
    
    #cdef int* Far_Aray_pt
    
    cdef int* Response
    
    cdef int allocate_count = 0
    
    #########################################
    #########################################
    ########## ' Initilize Far_Away ' #######
    #########################################
    #########################################
    
    ' Compute setdifference '
    
    set_diff_cython(Active_List,&Indices_Far_Away_bool[0],Seeds_indices.shape[0])
    
    ############################################################################
    
    ' Initilize empty Narrow_Band with initial capacity '
    
    cdef int initial_capacity = 128
    
    cdef int levels = 0
    
    ' Number of Stored Values '
    
    
    cdef int count = 0
    
    ' Initialize Number Of Levels '
        
    while pow(2,levels) < initial_capacity:
            
            levels += 1
        
    cdef int min_levels = levels 

    # allocate arrays
        
    cdef int nodes_number = pow(2,levels)
        
    ' 2* times the level number reserved for values on the bottom level '
        
    cdef double* nodes_values = <double*> malloc(2*nodes_number*sizeof(double))
    
    for k in range(nodes_number*2):
        
        nodes_values[k] = inf
        
    cdef int* references_array = <int*>malloc(nodes_number*sizeof(int))
    
    for k in range(nodes_number):
        
        references_array[k] = -1
    
    cdef double* nodes_values_pt
    
    cdef int* references_array_pt
    
    ' Create Narrow Band '
   
    cdef int Neigh_ind
    
    cdef double Neigh_value
    
    cdef int Check_Index 
    
    cdef double* new_values
    
    cdef double* new_values_pt
    
    cdef int number
    
    cdef int* new_reference
    
    cdef int* new_reference_pt
    
    cdef int index_up
    
    cdef int Int_index = 0
    
    for k in range(Seeds_indices.shape[0]):
        
        
        Neigh_ind = indptr[Active_List[k]+1] - indptr[Active_List[k]]
        
        ' Get the Neighbours for vertex k '
        
        for l in range(Neigh_ind):
            
            Check_Index = 0

            ' Check if Index is in Far_Away '

            Intersect_Cython(&Indices_Far_Away_bool[0],indices[indptr[Active_List[k]]+l],&Check_Index)


            if Check_Index == 1: # Add to Narrow_Band --> Remove from Far_Away
                
                Indices_Far_Away_bool[indices[indptr[Active_List[k]]+l]] = False
                
                Indices_Narrow_bool[indices[indptr[Active_List[k]]+l]] = True
                
                Neigh_value         = Image[indices[indptr[Active_List[k]]+l]]

                nodes_values_pt     = &nodes_values[0]

                references_array_pt = &references_array[0]

                number              = pow(2,levels)

                ' If indices positions exceed --> add new level '

                if count >= number:


                    ' Update available Reference size '

                    number        = pow(2,levels+1)

                    ' Allocate Memory for updated arrays '


                    ' Update Binary Tree '

                    new_values, new_reference = add_level(references_array_pt, nodes_values_pt,\
                                                             &levels,add = 1)

                    ' Increase Level Count '

                    levels += 1

                else: 

                    new_values    = <double*>malloc(number * 2 * sizeof(double))

                    new_reference = <int*>malloc(number * sizeof(int))

                    ' Copy Old reference Array '

                    for i in range(number):

                        new_reference[i] = references_array[i]

                    for i in range(2*number):

                        new_values[i] = nodes_values[i]


                    new_values_pt    = &new_values[0]

                    new_reference_pt = &new_reference[0] 

                    free(nodes_values)
                    free(references_array)

                index_up = (pow(2,levels) - 1) + count 

                new_reference[count] = indices[indptr[Active_List[k]]+l]
                
                new_values[index_up] = Neigh_value

                count += 1

                ' Update Binary Tree '

                nodes_values     = <double*>malloc(number * 2 * sizeof(double))

                references_array = <int*>malloc(number * sizeof(int))

                ' Update via for old number_size '

                for i in range(number*2):

                    nodes_values[i] = new_values[i]

                for i in range(number):

                    references_array[i] = new_reference[i]

                nodes_values_pt     = &nodes_values[0]

                references_array_pt = &references_array[0]

                free(new_reference)
                free(new_values)

                update_one(&index_up,nodes_values_pt,levels)
    
    ' Process Time Distance '
    ' Get the minimal value in Narrow Band '

    cdef int Step = Seeds_indices.shape[0]
    
    cdef int Min_Index_Narrow = 0
    
    cdef int  Rem_Ind
    
    cdef double Min_Value_Narrow = 0
    
    cdef int* Indices_neig 
    
    cdef int Dim_data = data.shape[0]
    
    cdef double* data_pt
    
    data_pt = &data[0]
    
    cdef double* Image_Data_Graph_pt = <double*>malloc(sizeof(double)*shape_ad[0]**shape_ad[1]**shape_ad[2]+1)

    cdef double* h_i_pt = <double*>malloc(sizeof(double)*shape_ad[0]**shape_ad[1]**shape_ad[2]+1)
    
    cdef double Print_Time
    
    Print_Time = time.time()

    while Step < Length_Graph:

        'Append the first value [0] of Narrow Band to \
         Active_list, delete [0] from Narrow_Band --> Initilize array for with new values'
        
        nodes_values_pt     = &nodes_values[0]
                
        references_array_pt = &references_array[0]

        Min_Value_Narrow,references_array,nodes_values = Get_Index_Narrow_Cython(&levels,nodes_values_pt\
                                             ,references_array_pt,min_levels,\
                                               &Min_Index_Narrow,&count,&number)

        Active_List[Step]   = Min_Index_Narrow
        
        Indices_Narrow_bool[Min_Index_Narrow] = False
        
        Active_List_Val[Step] = Min_Value_Narrow
        
        Step = Step + 1
        
        Neigh_ind = indptr[Min_Index_Narrow+1] - indptr[Min_Index_Narrow]
        
        
        ' Allocate Memory for neighbour indices to be updated '
        
        Indices_neig         = <int*>malloc(sizeof(int)*Neigh_ind)

        
        for k in range(Neigh_ind):                    
                
            Indices_neig[k]     = indices[indptr[Min_Index_Narrow]+k]        

        
        'Update the values for vertices within the Neighbourhod'
        'Here the Dijkstra`s methhod utilizes nearest neighbor'
        
        nodes_values_pt     = &nodes_values[0]
                
        references_array_pt = &references_array[0]
        
        references_array,nodes_values = Update_Narrow_Band(&indptr[0],&indices[0],\
                    nodes_values_pt,references_array_pt,&Image[0],Indices_neig,Image_Data_Graph,\
                    data_pt,Neigh_ind,&number,Far_Away_Length,Min_Index_Narrow,&levels,&count,\
                    &Image_Data_Graph_pt[0],&h_i_pt[0],&Indices_Far_Away_bool[0],&Indices_Narrow_bool[0])
        
        

        free(Indices_neig)
    
    ' #print Test Far_Away '
       
    #' Append_Remaining Indices '
    
    free(nodes_values)
    free(references_array)
    free(Image_Data_Graph_pt)
    free(h_i_pt)
    free(Indices_Far_Away_bool)
    free(Indices_Narrow_bool)
    
    
    print('Computation finished in ___'+str(time.time()-Print_Time)+'___seconds')
    
    return Active_List,Active_List_Val