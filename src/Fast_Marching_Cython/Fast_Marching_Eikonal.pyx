import numpy as np
cimport numpy as cnp
import cython
from cython.parallel cimport prange
from libc.stdlib cimport malloc, free
from libc.math cimport sqrt,abs

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

cdef (double*,double*) Sort(double* List,double* weights,int Length):
    
    'Help Variables'
    
    cdef int l,k,s
    cdef double S_var = 0
    cdef double W_var = 0
    
    cdef double* Sorted = <double*>malloc(Length*sizeof(double))
    cdef double* Weights_Sorted = <double*> malloc(sizeof(double)*Length)
    'Copy Array'
    
    for s in prange(Length,nogil = True):
        Sorted[s]    = List[s]
        Weights_Sorted[s] = weights[s]
        
    for k in range(Length):
        S_var = Sorted[k]
        W_var = Weights_Sorted[k]

        for l in range(k+1,Length):
            if S_var > Sorted[l]:
                S_var = Sorted[l]
                W_var = Weights_Sorted[l]
                Sorted[l] = Sorted[k]
                Weights_Sorted[l] = weights[k]
                Sorted[k] = S_var
                Weights_Sorted[k] = W_var

    return Sorted,Weights_Sorted

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
       
                
                 
' Bubble Sort Subroutine for value of one vertex while other are fixed '
' Returns:   Updated Value of vertex with minimal arrival time within the Narrowband '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double Local_Solution(double* a_i,double Vel,double p,double* w_i,int vertex, int End_Point):
    
    ' Compute the gradient on the grid (Characterization of P(x,y)) '
    
    ' a_i __ Values within the Narrow_Band cut Neighbouhood '
    
    # Allocate Array for gradient values 
    
    cdef int k
    
    cdef double Grad_min = 1/(Vel+1e-8)
    cdef double* a_i_sorted
    cdef double* w_i_sorted
    cdef double Local_Solution = 1e10 # Set large to go to the loop
    cdef int l,s,j,index,Iter_Patsch,Length_NPatch,h,y,f,w
    cdef int m = 0

    a_i_sorted,w_i_sorted = Sort(a_i,w_i,End_Point)
    
    if p == 1.0:
        while Local_Solution >= a_i_sorted[m+1] and m <= End_Point-1:
            Local_Solution = Solve_Subproblem(a_i_sorted,w_i_sorted,Grad_min,p,vertex,m+1)  
            m += 1
    
    elif p == 2.0:
        while Local_Solution >=a_i_sorted[m+1] and m <= End_Point-1:
            Local_Solution = Solve_Subproblem(a_i_sorted,w_i_sorted,Grad_min,p,vertex,m+1)
            m += 1
    elif p == 3.0:# infinity
        Local_Solution = a_i_sorted[0]+w_i_sorted[0]*(Grad_min)
    
    #Local_Solution = f_min_double(Local_Solution,a_i[0])
    free(a_i_sorted)
    free(w_i_sorted) 
    
    return Local_Solution

' Subroutine for minimal time arrival value '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef double Solve_Subproblem(double* a_i,double* w_i_sorted,double Norm,double p,int local_vertex, int End_Point):
    
    cdef double Sol = 0
    cdef double Term = 0
    cdef double Value_Inner_Loop 
    cdef double h_i_normalized = 0
    cdef int l,s,i,j,index,k,Iter_Patsch,Length_NPatch,h,y,f,w

    if p == 1.0: 
        for l in range(End_Point):    
            Sol += a_i[l]*sqrt(w_i_sorted[l])
            h_i_normalized += sqrt(w_i_sorted[l])
        Sol += Norm
    
    elif p == 2.0:
        for i in range(End_Point):
            Value_Inner_Loop = 0

            for j in range(i+1,End_Point):
                Value_Inner_Loop += pow_double((a_i[i]-a_i[j]),2.0)*w_i_sorted[i]*w_i_sorted[j]

            Term += pow_double(Norm,2.0)*w_i_sorted[i]-Value_Inner_Loop

            Sol += a_i[i]*w_i_sorted[i]
        Sol += sqrt(Term)
        for s in range(End_Point):
            h_i_normalized += w_i_sorted[s]
            
    return Sol/h_i_normalized 


' Subroutine for updating Narrow_Band '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline (int*,double*) Update_Narrow_Band( int* indptr, int* indices, \
                    double* Narrow_Band_Val, int* Narrow_Band,int* Indices,\
                    double* Image_Data_Graph, double* h_i,int Length_In,int* Length_Narrow,\
                    int Min_Index, int* levels,int* count,bint* Indices_Far_Away_bool,\
                    bint* Indices_Narrow_bool,double* Vel,double p):
    
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
    cdef double* Image_Data_Graph_pt
    cdef double* h_i_pt
    
    for j in range(Length_In):
        Length_neigh_j = indptr[indices[indptr[Min_Index]+j]+1]-indptr[indices[indptr[Min_Index]+j]]
        Image_Data_Graph_pt = <double*> malloc(sizeof(double)*Length_neigh_j)
        h_i_pt = <double*> malloc(sizeof(double)*Length_neigh_j)
        Response    = 0
        Response_1  = 0
        del_index   = 0
        del_index_1 = 0
        
        
        ' Check if the index is in Narrow_Band then update solution '

        Check_Index_pt(Indices_Narrow_bool,Indices[j],&Response)
        Check_Index_pt(Indices_Far_Away_bool,Indices[j],&Response_1)
            
        for k in range(Length_neigh_j):
            Image_Data_Graph_pt[k] = Image_Data_Graph[indices[indptr[indices[indptr[Min_Index]+j]]+k]]
            h_i_pt[k] = h_i[indptr[indices[indptr[Min_Index]+j]]+k]

        Help = Local_Solution(&Image_Data_Graph_pt[0],Vel[indices[indptr[Min_Index]+j]],p,h_i_pt, j,Length_neigh_j)
        
        if Help < Image_Data_Graph[indices[indptr[Min_Index]+j]]:
            Image_Data_Graph[indices[indptr[Min_Index]+j]] = Help
            if Response_1 == 1:
                'Set Index in Far_Away to -1 to ignore it '
                
                Indices_Far_Away_bool[Indices[j]] = False

                ' Find Index in Far_Away with value Indices[j] '
                    
                Neigh_value  = Image_Data_Graph[indices[indptr[Min_Index]+j]]

                ' If indices positions exceed add new level '

                if count_pt+1 > number-pow(2,levels_pt-1):

                    ' Update Narrowband size '
                    
                    number += pow(2,levels_pt)

                    ' Add Level to Binary Heap '

                    Narrow_Band_Val_Up,Narrow_Band_Up = add_level(Narrow_Band_pt, Narrow_Band_Val_pt,Length_Narrow,\
                                                                &levels_pt,add = 1)
                    

                    levels_pt += 1
                    Min_Heap_Insert(Narrow_Band_Up, Narrow_Band_Val_Up, indices[indptr[Min_Index]+j],Neigh_value,\
                                                                                                        count_pt)
                    count_pt += 1 
                    Narrow_Band_pt = &Narrow_Band_Up[0]
                    Narrow_Band_Val_pt = &Narrow_Band_Val_Up[0]
                else: 
                    Narrow_Band_Val_Up = Narrow_Band_Val_pt
                    Narrow_Band_Up     = Narrow_Band_pt
                    Min_Heap_Insert(Narrow_Band_Up, Narrow_Band_Val_Up, indices[indptr[Min_Index]+j],Neigh_value, count_pt)
                    count_pt += 1
                    Narrow_Band_pt = &Narrow_Band_Up[0]
                    Narrow_Band_Val_pt = &Narrow_Band_Val_Up[0]
            elif Response == 1:
                Narrow_Band_Val_Up = Narrow_Band_Val_pt
                Narrow_Band_Up     = Narrow_Band_pt
                Min_Heap_Update_Node_Pos(Narrow_Band_Up, Narrow_Band_Val_Up,indices[indptr[Min_Index]+j],\
                       Neigh_value,count_pt,number)
                Narrow_Band_pt = &Narrow_Band_Up[0]
                Narrow_Band_Val_pt = &Narrow_Band_Val_Up[0]
            
          
        free(Image_Data_Graph_pt)
        free(h_i_pt)
        
    count[0]  = count_pt
    levels[0] = levels_pt
    Length_Narrow[0] = number
    
    return Narrow_Band_pt,Narrow_Band_Val_pt

"""

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef void Min_Heap_Update_Node_Pos(int* narr_band_ref, double* narr_band_val,int index,double Value,\
                                   int count,int Heap_Size):
    cdef int Index_node = -1
    cdef int parent,left,right
    cdef bint Min_Heap = False
    cdef double Aux_val
    cdef int m = 0
    cdef int Aux_ind
    for k in range(count):
        if narr_band_ref[k] == index:
            narr_band_val[k] = Value
            Index_node = k
            break
    if Index_node == -1:
        Min_Heap = True
        print("Not in Narrow Band Something wrong")

    ' Update Min Heap '
    while Min_Heap == False and m < Heap_Size:
        parent = int((Index_node -1)/2)
        left  = Index_node*2+1
        right = Index_node*2+2
        if narr_band_val[parent] > narr_band_val[Index_node]:
            ' Move Up '
            Aux_ind = narr_band_ref[parent]
            Aux_val = narr_band_val[parent]
            narr_band_val[parent] = narr_band_val[Index_node]  
            narr_band_ref[parent] = narr_band_ref[Index_node]
            narr_band_val[Index_node] = Aux_val
            narr_band_ref[Index_node] = Aux_ind
            Index_node = parent
        elif (left < Heap_Size and right < Heap_Size ):
            ' Move Up '
            if narr_band_val[Index_node] > narr_band_val[left] or narr_band_val[Index_node] > narr_band_val[right]:
                ' Determine which child to be moved to the top '
                parent = Index_node
                if narr_band_val[parent] > narr_band_val[left]:
                    parent = left
                if narr_band_val[parent] > narr_band_val[right]:
                    parent = right
                ' Swap with smallest child '
                Aux_ind = narr_band_ref[Index_node]
                Aux_val = narr_band_val[Index_node]
                narr_band_val[Index_node] = narr_band_val[parent]  
                narr_band_ref[Index_node] = narr_band_ref[parent]
                narr_band_val[parent] = Aux_val
                narr_band_ref[parent] = Aux_ind
                Index_node = parent
            else:
                Min_Heap = True
        else:
            Min_Heap = True
        m += 1
                
        
    
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(False)

cdef void Intersect_Cython(bint* Far_Away_ind,int Check_index,int* Response)nogil:
        
    if Far_Away_ind[Check_index] == True:
            
        Response[0] = 1


' Function to remove index from the binary Min-Heap '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline (int*,double*) Rem_Min_Index_Binary_Heap(int* narr_band_ref, double* narr_band_val,int* levels,\
                                                     int* count,int* size_narrow):
            
    cdef int End_Ind = count[0] - 1
    cdef double Aux_val
    cdef int Aux_ind
    cdef int left,right,parent,parent_aux
    cdef int i
        
    ' Define Pointer to allocated arrays '
        
    cdef int* narr_band_ref_pt    
    cdef double* narr_band_val_pt
    cdef int levels_pt 
        
    ' Swap Min-Node and Last node of the Binary Heap '
    
    narr_band_val[0] =  narr_band_val[End_Ind] 
    narr_band_ref[0] = narr_band_ref[End_Ind] 
    narr_band_val[End_Ind] = 1e10
    narr_band_ref[End_Ind] = -1
            
    ' Set index to the last element '
        
    count[0] -= 1
    parent = 0
    parent_aux = parent
    left = parent*2+1
    right =  parent*2+2
    while  narr_band_val[parent] > narr_band_val[left] or narr_band_val[parent] > narr_band_val[right]:
        if narr_band_val[parent_aux] > narr_band_val[left]:
            parent_aux = left
        if narr_band_val[parent_aux] > narr_band_val[right]:
            parent_aux = right
        if parent_aux != parent:
            Aux_ind = narr_band_ref[parent]
            Aux_val = narr_band_val[parent]
            narr_band_val[parent] = narr_band_val[parent_aux]  
            narr_band_ref[parent] = narr_band_ref[parent_aux]
            narr_band_val[parent_aux] = Aux_val
            narr_band_ref[parent_aux] = Aux_ind
            parent = parent_aux
            left = parent*2+1
            right =  parent*2+2
        if left > size_narrow[0] or right > size_narrow[0]:
            break 
    
    ' If empty Level remove level from Min-Heap '
    if  (count[0] < size_narrow[0]-pow(2,levels[0]-1)):
        narr_band_ref_pt = &narr_band_ref[0]             
        narr_band_val_pt = &narr_band_val[0]
        levels_pt = levels[0]
        narr_band_val_pt,narr_band_ref_pt = add_level(narr_band_ref_pt, narr_band_val_pt,size_narrow, &levels_pt,add = -1) 
        levels[0] = levels[0] - 1
    else:        
        levels_pt        = levels[0]
        narr_band_ref_pt = &narr_band_ref[0]             
        narr_band_val_pt = &narr_band_val[0]

         
    
    return narr_band_ref_pt, narr_band_val_pt

' Function to extract the minimal value and index from the binary tree '

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline (double,int*,double*) Get_Index_Narrow_Cython(int* levels, double* current_values,\
                    int* current_reference,int* popped_ref,int* count,int* size_narrow):
    
    
    cdef double value
    cdef int* current_reference_pt
    cdef double* current_values_pt
        
    ' Pick the index with the lowest value located at the top of the binary heap'
    
                                                
    ' Return current reference index and value '  
    value = current_values[0]
    popped_ref[0] = current_reference[0]
    
    
    ' Remove index '
        
    current_reference_pt = &current_reference[0]   
    current_values_pt    = &current_values[0]
    cdef int count_pt 
    count_pt = count[0]
    cdef int levels_pt      = levels[0]
    cdef int size_narrow_pt = size_narrow[0]
    current_reference_pt,current_values_pt = Rem_Min_Index_Binary_Heap(current_reference_pt,current_values_pt\
                                                        ,&levels_pt,&count_pt,&size_narrow_pt)
    
    count[0]       = count_pt
    levels[0]      = levels_pt
    size_narrow[0] = size_narrow_pt
    
    return value,current_reference_pt,current_values_pt            


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void Min_Heap_Insert(int* array_indices,double* array_values,int index,double value,int Ind_Heap):
    
    cdef int i = Ind_Heap
    cdef double Aux_val
    cdef int Aux_ind
    cdef int parent_ind
    'Insert at the end '
    array_values[Ind_Heap] = value
    array_indices[Ind_Heap] = index
    
    'Order the Min-Heap'
    if Ind_Heap == 0:
        array_indices[Ind_Heap] = index
        array_values[Ind_Heap] = value
    else:
        parent_ind = int((Ind_Heap-1)/2)

        while array_values[parent_ind] > array_values[i]:
            Aux_val = array_values[parent_ind]
            Aux_ind = array_indices[parent_ind]
            array_values[parent_ind] = array_values[i]
            array_indices[parent_ind] = array_indices[i]
            array_values[i] = Aux_val
            array_indices[i] = Aux_ind
            i = parent_ind
            parent_ind = int((i-1)/2)


        
    
    
"""
Comment: Dmitrij Sitenko
Function to add or to remove one level of Binary Min Heap:

Input: 1) Number of new Levels
       2) reference, values arrays to be modified
       3) Number of old Levels
       4) new empty reference and values arrays initialized before function call    
"""

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef (double*,int*) add_level(int* old_reference, double* old_values,int* number_nodes, int* old_levels,int add):
     
    ' Initialization '
        
    cdef int i,k, level_start_old, level_start_new , n
    cdef int number_nodes_new = number_nodes[0]
    
    if add == 1:      
        number_nodes_new += pow(2,old_levels[0])
    elif add == -1:
        number_nodes_new -= pow(2,old_levels[0]-1)
    
    Narrow_Band_Val_Up    = <double*>malloc(number_nodes_new*sizeof(double))
    Narrow_Band_Up        = <int*>malloc(number_nodes_new*sizeof(int))
    
    ' Assign elements to infinity '
    
    for i in prange(number_nodes_new,nogil = True): 
        Narrow_Band_Val_Up[i] = 1e10
    
    for i in prange(number_nodes_new,nogil = True):
        Narrow_Band_Up[i] = -1
    
    ' Write old ref and vals from last old level to last new level '
    
    n = fmin(number_nodes_new, number_nodes[0])
    
    for i in prange(n,nogil = True):
        Narrow_Band_Val_Up[i] = old_values[i]
        Narrow_Band_Up[i] = old_reference[i]
    
    free(old_values)
    free(old_reference)
    
    ' Update the binary tree from bottom to top level ' 
    
    number_nodes[0] = number_nodes_new
    
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

cpdef Eikonal_Eq_Solve_Cython(double [:] Image,double[:] Velocity,double [:] data,int [:] indices,int [:] indptr,int[:,:] Seeds_indices,double p):
    
    assert len(data.shape) == len(indices.shape), ' data and indices not of equal shape '
    
    cdef int Length_Graph = Image.shape[0]
    cdef int k,l,i
    cdef double [:] Image_Data_Graph = Image.copy() # Array of reshaped arrival times set to zero at seeds indices 
    for k in range(Seeds_indices.shape[0]):
        Image_Data_Graph[Seeds_indices[k,0]] = 0
    
    
    ' Initialization '
    
    cdef cnp.ndarray[int,ndim = 1] Active_List = np.zeros(Length_Graph,dtype = np.int32)
    cdef cnp.ndarray[double,ndim = 1] Active_List_Val  = np.zeros(Length_Graph,)
    
    
    ' Create List for Intersect Indices '
    
    cdef bint* Indices_Narrow_bool = <bint*>malloc(sizeof(bint)*Length_Graph)
    cdef bint* Indices_Far_Away_bool = <bint*>malloc(sizeof(bint)*Length_Graph)
    
    for k in prange(Length_Graph,nogil = True):
        Indices_Narrow_bool[k] = False 
        Indices_Far_Away_bool[k] = True
    
    for k in range(Seeds_indices.shape[0]):
        Active_List[k]          = Seeds_indices[k,0]
        Active_List_Val[k]      = Image_Data_Graph[Seeds_indices[k,0]] 
        Indices_Far_Away_bool[Seeds_indices[k,0]] = False #Remove Index from boolean Far_Away array
        
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
    
    ' Initilize empty Narrow_Band through a Min-Heap-Structure with initial capacity 256 '
    
    cdef int initial_capacity = 0
    cdef int levels = 0
    cdef int nodes_number = 0
    while initial_capacity < Seeds_indices.shape[0]:
        initial_capacity += pow(2,levels)
        levels += 1
    
    ' Number of Stored Values '
    
    cdef int count = 0
    
    ' Initialize Number Of Levels '
    cdef int lev = 0
    while nodes_number <= initial_capacity:
        nodes_number += pow(2,lev)
        lev += 1
    levels = lev

    ' Allocate memory for two arrays for binary min heap ' 
        
    cdef double* nodes_values = <double*> malloc(nodes_number*sizeof(double)) # Current Values of min time arrivels
    cdef int* references_array = <int*>malloc(nodes_number*sizeof(int))       # Corresponding Indices on Image grid 
    
    for k in range(nodes_number):  
        nodes_values[k] = 1e10
        references_array[k] = -1
    
    cdef double* nodes_values_pt
    cdef int* references_array_pt
    
    ' Create Narrow Band '
   
    cdef int Neigh_ind
    cdef double Neigh_value
    cdef int Check_Index 
    cdef double* new_values
    cdef double* new_values_pt
    cdef int* new_reference
    cdef int* new_reference_pt
    cdef int index_up
    cdef int Int_index = 0
    cdef int Neigh_ind_l
    cdef int q
    cdef double* Image_Data_Graph_pt
    cdef double* h_i_pt
    
    for k in range(Seeds_indices.shape[0]):
        Neigh_ind = indptr[Active_List[k]+1] - indptr[Active_List[k]] #'Neighbour-indices of each Seed '
        
        ' Get the Neighbours for vertex k '
        
        for l in range(Neigh_ind):
            Neigh_ind_l = indptr[indices[indptr[Active_List[k]]+l]+1]-indptr[indices[indptr[Active_List[k]]+l]]
            Image_Data_Graph_pt = <double*>malloc(sizeof(double)*Neigh_ind_l)
            h_i_pt  = <double*>malloc(sizeof(double)*Neigh_ind_l)
            Check_Index = 0
            
            ' Check if Index is in Far_Away '

            Intersect_Cython(&Indices_Far_Away_bool[0],indices[indptr[Active_List[k]]+l],&Check_Index)
            
            for q in range(Neigh_ind_l): 
                Image_Data_Graph_pt[q] = Image_Data_Graph[indices[indptr[indices[indptr[Active_List[k]]+l]]+q]]
                h_i_pt[q] = data[indptr[indices[indptr[Active_List[k]]+l]]+q]
            if Check_Index == 1: # Add to Narrow_Band --> Remove from Far_Away
                Indices_Far_Away_bool[indices[indptr[Active_List[k]]+l]] = False
                
                Indices_Narrow_bool[indices[indptr[Active_List[k]]+l]] = True
                
                Neigh_value         = Image_Data_Graph[indices[indptr[Active_List[k]]+l]]
                nodes_values_pt     = &nodes_values[0]
                references_array_pt = &references_array[0]
                
                ' If indices positions exceed --> add new level '

                if count >= nodes_number:
                    ' Update Size Binary Heap '
                    new_values, new_reference = add_level(references_array_pt, nodes_values_pt,&nodes_number,\
                                                             &levels,add = 1)
                    levels += 1

                else: 
                    new_values    = <double*>malloc(nodes_number*sizeof(double))
                    new_reference = <int*>malloc(nodes_number*sizeof(int))

                    ' Copy Old reference Array '

                    for i in range(nodes_number):
                        new_reference[i] = references_array[i]

                    for i in range(nodes_number):
                        new_values[i] = nodes_values[i]

                    free(nodes_values)
                    free(references_array)
                new_values_pt    = &new_values[0]
                new_reference_pt = &new_reference[0] 
                Help = Local_Solution(&Image_Data_Graph_pt[0],Velocity[indices[indptr[Active_List[k]]+l]],p,\
                                      &h_i_pt[0], l,Neigh_ind_l)
                Image_Data_Graph[indices[indptr[Active_List[k]]+l]] = Help
                Min_Heap_Insert(new_reference_pt,new_values_pt,indices[indptr[Active_List[k]]+l],Help,count)
                count += 1

                ' Update Binary Tree '

                nodes_values     = <double*>malloc(nodes_number* sizeof(double))
                references_array = <int*>malloc(nodes_number * sizeof(int))

                ' Update via for old number_size '

                for i in range(nodes_number):

                    nodes_values[i] = new_values[i]

                for i in range(nodes_number):

                    references_array[i] = new_reference[i]

                nodes_values_pt     = &nodes_values[0]

                references_array_pt = &references_array[0]
                free(new_reference)
                free(new_values)
            free(Image_Data_Graph_pt)
            free(h_i_pt)
                
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
    cdef double Print_Time
    Print_Time = time.time()

    while Step < Length_Graph-1:

        'Append the first value [0] of Narrow Band to \
         Active_list, delete [0] from Narrow_Band --> Initilize array for new values'
        
        nodes_values_pt     = &nodes_values[0]  
        references_array_pt = &references_array[0]
        ' Return and remove min value and index from Min Heap '
        
        
        Min_Value_Narrow,references_array,nodes_values = Get_Index_Narrow_Cython(&levels,nodes_values_pt\
                                             ,references_array_pt, &Min_Index_Narrow,&count,&nodes_number)

        Active_List[Step]   = Min_Index_Narrow
        
        Indices_Narrow_bool[Min_Index_Narrow] = False
        
        Active_List_Val[Step] = Min_Value_Narrow
        
        Step = Step + 1
        
        ' Allocate Memory for neighbour indices to be updated '
        
        Neigh_ind = indptr[Min_Index_Narrow+1] - indptr[Min_Index_Narrow] #Neighbours of removed node added to binary heap
        Indices_neig         = <int*>malloc(sizeof(int)*Neigh_ind) 
        
        for k in range(Neigh_ind):                    
            Indices_neig[k]     = indices[indptr[Min_Index_Narrow]+k]        

        'Update the values for vertices within the Neighbourhod'
        'Here the Dijkstra`s methhod utilizes nearest neighbor'
        
        nodes_values_pt     = &nodes_values[0]
        references_array_pt = &references_array[0]
                
        references_array,nodes_values = Update_Narrow_Band(&indptr[0],&indices[0],nodes_values_pt,\
                    references_array_pt,Indices_neig,&Image_Data_Graph[0],data_pt,Neigh_ind,&nodes_number,\
                    Min_Index_Narrow,&levels,&count,&Indices_Far_Away_bool[0],&Indices_Narrow_bool[0],&Velocity[0],p)
        free(Indices_neig)
       
    #' Append_Remaining Indices '
    
    free(nodes_values)
    free(references_array)
    free(Indices_Far_Away_bool)
    free(Indices_Narrow_bool)
    
    
    print('Computation finished in ___'+str(time.time()-Print_Time)+'___seconds')
    
    return Active_List, Active_List_Val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cdef inline (int*,double*) Update_Narrow_Band_Labeling( int* Labeling, double* S_Array, int* indptr, int* indices, \
                    double* Narrow_Band_Val, int* Narrow_Band,int* Indices,\
                    double* Image_Data_Graph, double* h_i,int Length_In,int* Length_Narrow,\
                    int Min_Index, int* levels,int* count,bint* Indices_Far_Away_bool,\
                    bint* Indices_Narrow_bool,double [:] Vel,double p):
    
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
    cdef double* Image_Data_Graph_pt
    cdef double* h_i_pt

    cdef int Label_Aux 
    for j in range(Length_In):
        Label_Aux = Labeling [Min_Index]
        Length_neigh_j = indptr[indices[indptr[Min_Index]+j]+1]-indptr[indices[indptr[Min_Index]+j]]
        Image_Data_Graph_pt = <double*> malloc(sizeof(double)*Length_neigh_j)
        h_i_pt = <double*> malloc(sizeof(double)*Length_neigh_j)
        Response    = 0
        Response_1  = 0
        del_index   = 0
        del_index_1 = 0
        
        
        ' Check if the index is in Narrow_Band then update solution '

        Check_Index_pt(Indices_Narrow_bool,Indices[j],&Response)
        Check_Index_pt(Indices_Far_Away_bool,Indices[j],&Response_1)
            
        for k in range(Length_neigh_j):
            Image_Data_Graph_pt[k] = Image_Data_Graph[indices[indptr[indices[indptr[Min_Index]+j]]+k]]
            h_i_pt[k] = h_i[indptr[indices[indptr[Min_Index]+j]]+k]

        Help = Local_Solution(&Image_Data_Graph_pt[0],Vel[indices[indptr[Min_Index]+j]],p,h_i_pt, j,Length_neigh_j)
        
        if Help < Image_Data_Graph[indices[indptr[Min_Index]+j]]:
            Image_Data_Graph[indices[indptr[Min_Index]+j]] = Help
            
            if Image_Data_Graph[Min_Index]/h_i[indptr[Min_Index]+j] < S_Array[indices[indptr[Min_Index]+j]]:
                S_Array[indices[indptr[Min_Index]+j]] = Image_Data_Graph[Min_Index]/h_i[indptr[Min_Index]+j]
                Labeling[indices[indptr[Min_Index]+j]] = Labeling[Min_Index]
            if Response_1 == 1:
                'Set Index in Far_Away to -1 to ignore it '
                
                Indices_Far_Away_bool[Indices[j]] = False

                ' Find Index in Far_Away with value Indices[j] '
                    
                Neigh_value  = Image_Data_Graph[indices[indptr[Min_Index]+j]]

                ' If indices positions exceed add new level '

                if count_pt+1 > number-pow(2,levels_pt-1):

                    ' Update Narrowband size '
                    
                    number += pow(2,levels_pt)

                    ' Add Level to Binary Heap '

                    Narrow_Band_Val_Up,Narrow_Band_Up = add_level(Narrow_Band_pt, Narrow_Band_Val_pt,Length_Narrow,\
                                                                &levels_pt,add = 1)
                    

                    levels_pt += 1
                    Min_Heap_Insert(Narrow_Band_Up, Narrow_Band_Val_Up, indices[indptr[Min_Index]+j],Neigh_value,\
                                                                                                        count_pt)
                    count_pt += 1 
                    Narrow_Band_pt = &Narrow_Band_Up[0]
                    Narrow_Band_Val_pt = &Narrow_Band_Val_Up[0]
                else: 
                    Narrow_Band_Val_Up = Narrow_Band_Val_pt
                    Narrow_Band_Up     = Narrow_Band_pt
                    Min_Heap_Insert(Narrow_Band_Up, Narrow_Band_Val_Up, indices[indptr[Min_Index]+j],Neigh_value, count_pt)
                    count_pt += 1
                    Narrow_Band_pt = &Narrow_Band_Up[0]
                    Narrow_Band_Val_pt = &Narrow_Band_Val_Up[0]
            elif Response == 1:
                Narrow_Band_Val_Up = Narrow_Band_Val_pt
                Narrow_Band_Up     = Narrow_Band_pt
                Min_Heap_Update_Node_Pos(Narrow_Band_Up, Narrow_Band_Val_Up,indices[indptr[Min_Index]+j],\
                       Neigh_value,count_pt,number)
                Narrow_Band_pt = &Narrow_Band_Up[0]
                Narrow_Band_Val_pt = &Narrow_Band_Val_Up[0]

            
          
        free(Image_Data_Graph_pt)
        free(h_i_pt)
        
    count[0]  = count_pt
    levels[0] = levels_pt
    Length_Narrow[0] = number
    
    return Narrow_Band_pt,Narrow_Band_Val_pt


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)

cpdef Eikonal_Eq_Labeling_Cython(int[:] Labeling, double [:] Image,double[:,:] Velocity,double [:] data,int [:] indices,int [:] indptr,int[:,:] Seeds_indices,double p):
    
    assert len(data.shape) == len(indices.shape), ' data and indices not of equal shape '
    
    cdef int Length_Graph = Image.shape[0]
    cdef int k,l,i
    
    cdef double [:] Image_Data_Graph = Image.copy() # Array of reshaped arrival times set to zero at seeds indices 
    for k in range(Seeds_indices.shape[0]):
        Image_Data_Graph[Seeds_indices[k,0]] = 0
    
    cdef cnp.ndarray[ndim = 1, dtype = int,negative_indices = False] Active_List = np.zeros(Length_Graph,dtype = np.int32)
    cdef cnp.ndarray[ndim = 1, dtype = int,negative_indices = False] Labeling_out = np.zeros(Length_Graph,dtype = np.int32)
    cdef cnp.ndarray[ndim = 1, dtype = double,negative_indices = False] Active_List_Val  = np.zeros(Length_Graph,)
    
    ' Create List for Intersect Indices '
    
    cdef bint* Indices_Narrow_bool = <bint*>malloc(sizeof(bint)*Length_Graph)
    cdef bint* Indices_Far_Away_bool = <bint*>malloc(sizeof(bint)*Length_Graph)
    
    cdef double* S_Array = <double*>malloc(sizeof(double)*Length_Graph)
    
    for k in prange(Length_Graph,nogil = True):
        Indices_Narrow_bool[k] = False 
        Indices_Far_Away_bool[k] = True
        S_Array[k] = 10e10
    
    for k in range(Seeds_indices.shape[0]):
        Labeling_out[Seeds_indices[k,0]]         = Labeling[Seeds_indices[k,0]]
        Active_List[k]          = Seeds_indices[k,0]
        Active_List_Val[k]      = Image_Data_Graph[Seeds_indices[k,0]] 
        Indices_Far_Away_bool[Seeds_indices[k,0]] = False #Remove Index from boolean Far_Away array
        
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
    
    ' Initilize empty Narrow_Band through a Min-Heap-Structure with initial capacity 256 '
    
    cdef int initial_capacity = 0
    cdef int levels = 0
    cdef int nodes_number = 0
    while initial_capacity < Seeds_indices.shape[0]:
        initial_capacity += pow(2,levels)
        levels += 1
    
    ' Number of Stored Values '
    
    cdef int count = 0
    
    ' Initialize Number Of Levels '
    cdef int lev = 0
    while nodes_number <= initial_capacity:
        nodes_number += pow(2,lev)
        lev += 1
    levels = lev

    ' Allocate memory for two arrays for binary min heap ' 
        
    cdef double* nodes_values = <double*> malloc(nodes_number*sizeof(double)) # Current Values of min time arrivels
    cdef int* references_array = <int*>malloc(nodes_number*sizeof(int))       # Corresponding Indices on Image grid 
    
    for k in range(nodes_number):  
        nodes_values[k] = 1e10
        references_array[k] = -1
    
    cdef double* nodes_values_pt
    cdef int* references_array_pt
    
    ' Create Narrow Band '
   
    cdef int Neigh_ind
    cdef double Neigh_value
    cdef int Check_Index 
    cdef double* new_values
    cdef double* new_values_pt
    cdef int* new_reference
    cdef int* new_reference_pt
    cdef int index_up
    cdef int Int_index = 0
    cdef int Neigh_ind_l
    cdef int q
    cdef double* Image_Data_Graph_pt
    cdef double* h_i_pt
    cdef int Label_Front
    
    for k in range(Seeds_indices.shape[0]):
        Label_Front = Labeling_out[Active_List[k]]
        Neigh_ind = indptr[Active_List[k]+1] - indptr[Active_List[k]] #'Neighbour-indices of each Seed '
        
        ' Get the Neighbours for vertex k '
        
        for l in range(Neigh_ind):
            Neigh_ind_l = indptr[indices[indptr[Active_List[k]]+l]+1]-indptr[indices[indptr[Active_List[k]]+l]]
            Image_Data_Graph_pt = <double*>malloc(sizeof(double)*Neigh_ind_l)
            h_i_pt  = <double*>malloc(sizeof(double)*Neigh_ind_l)
            Check_Index = 0
            
            ' Check if Index is in Far_Away '

            Intersect_Cython(&Indices_Far_Away_bool[0],indices[indptr[Active_List[k]]+l],&Check_Index)
            
            for q in range(Neigh_ind_l): 
                Image_Data_Graph_pt[q] = Image_Data_Graph[indices[indptr[indices[indptr[Active_List[k]]+l]]+q]]
                h_i_pt[q] = data[indptr[indices[indptr[Active_List[k]]+l]]+q]
            if Check_Index == 1: # Add to Narrow_Band --> Remove from Far_Away
                Indices_Far_Away_bool[indices[indptr[Active_List[k]]+l]] = False
                
                Indices_Narrow_bool[indices[indptr[Active_List[k]]+l]] = True
                
                Neigh_value         = Image_Data_Graph[indices[indptr[Active_List[k]]+l]]
                nodes_values_pt     = &nodes_values[0]
                references_array_pt = &references_array[0]
                
                ' If indices positions exceed --> add new level '

                if count >= nodes_number:
                    ' Update Size Binary Heap '
                    new_values, new_reference = add_level(references_array_pt, nodes_values_pt,&nodes_number,\
                                                             &levels,add = 1)
                    levels += 1

                else: 
                    new_values    = <double*>malloc(nodes_number*sizeof(double))
                    new_reference = <int*>malloc(nodes_number*sizeof(int))

                    ' Copy Old reference Array '

                    for i in range(nodes_number):
                        new_reference[i] = references_array[i]

                    for i in range(nodes_number):
                        new_values[i] = nodes_values[i]

                    free(nodes_values)
                    free(references_array)
                new_values_pt    = &new_values[0]
                new_reference_pt = &new_reference[0] 
                Help = Local_Solution(&Image_Data_Graph_pt[0],Velocity[Label_Front][indices[indptr[Active_List[k]]+l]],p,\
                                      &h_i_pt[0], l,Neigh_ind_l)
                if Help < Image_Data_Graph[indices[indptr[Active_List[k]]+l]]:
                    Image_Data_Graph[indices[indptr[Active_List[k]]+l]] = Help
                    if Image_Data_Graph[Active_List[k]]/data[indptr[Active_List[k]]+l]<S_Array[indices[indptr[Active_List[k]]+l]]:
                        S_Array[indices[indptr[Active_List[k]]+l]] = Image_Data_Graph[Active_List[k]]/data[indptr[Active_List[k]]+l]
                        Labeling_out[indices[indptr[Active_List[k]]+l]] = Labeling_out[Active_List[k]]
                Min_Heap_Insert(new_reference_pt,new_values_pt,indices[indptr[Active_List[k]]+l],Help,count)
                count += 1

                ' Update Binary Tree '

                nodes_values     = <double*>malloc(nodes_number* sizeof(double))
                references_array = <int*>malloc(nodes_number * sizeof(int))

                ' Update via for old number_size '

                for i in range(nodes_number):

                    nodes_values[i] = new_values[i]

                for i in range(nodes_number):

                    references_array[i] = new_reference[i]

                nodes_values_pt     = &nodes_values[0]

                references_array_pt = &references_array[0]
                free(new_reference)
                free(new_values)
            free(Image_Data_Graph_pt)
            free(h_i_pt)
                
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
    cdef double Print_Time
    Print_Time = time.time()

    while Step < Length_Graph-1:
        
        nodes_values_pt     = &nodes_values[0]  
        references_array_pt = &references_array[0]
        
        Min_Value_Narrow,references_array,nodes_values = Get_Index_Narrow_Cython(&levels,nodes_values_pt\
                                             ,references_array_pt, &Min_Index_Narrow,&count,&nodes_number)

        Active_List[Step]   = Min_Index_Narrow
        Label_Front = Labeling_out[Min_Index_Narrow]
        Indices_Narrow_bool[Min_Index_Narrow] = False
        
        Active_List_Val[Step] = Min_Value_Narrow
        
        Step = Step + 1
        
        ' Allocate Memory for neighbour indices to be updated '
        
        Neigh_ind = indptr[Min_Index_Narrow+1] - indptr[Min_Index_Narrow] #Neighbours of removed node added to binary heap
        Indices_neig         = <int*>malloc(sizeof(int)*Neigh_ind) 
        
        for k in range(Neigh_ind):                    
            Indices_neig[k]     = indices[indptr[Min_Index_Narrow]+k]        

        'Update the values for vertices within the Neighbourhod'
        'Here the Dijkstra`s methhod utilizes nearest neighbor'
        
        nodes_values_pt     = &nodes_values[0]
        references_array_pt = &references_array[0]

        references_array,nodes_values = Update_Narrow_Band_Labeling(&Labeling_out[0],&S_Array[0],&indptr[0],&indices[0],nodes_values_pt,\
                    references_array_pt,Indices_neig,&Image_Data_Graph[0],data_pt,Neigh_ind,&nodes_number,\
                    Min_Index_Narrow,&levels,&count,&Indices_Far_Away_bool[0],&Indices_Narrow_bool[0],Velocity[Label_Front],p)
        free(Indices_neig)
       
    #' Append_Remaining Indices '
    
    free(nodes_values)
    free(references_array)
    free(Indices_Far_Away_bool)
    free(Indices_Narrow_bool)
    free(S_Array)
    
    
    print('Computation finished in ___'+str(time.time()-Print_Time)+'___seconds')
    
    return Labeling_out,Active_List, Active_List_Val