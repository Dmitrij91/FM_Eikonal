import numpy as np
from Fast_Marching_Cython import Fast_Marching_Library as FML


def Create_Neighbours_data(k, Ad):
    return Ad.data[Ad.indptr[k]:Ad.indptr[k + 1]]


Create_Neigh_data_vec = np.vectorize(Create_Neighbours_data, otypes=[list])


# Implement in Cuda ! (TODO)

def Create_Neighbours_indices(k, Ad):
    return Ad.indices[Ad.indptr[k]:Ad.indptr[k + 1]]


Create_Neigh_indices_vec = np.vectorize(Create_Neighbours_indices, otypes=[list])

' Variables Settings '

' Image             ______  Input Image 2D 3D reshape to 1D array'
' Ad                ______  Graph as 1D array in a matrix sparse format containing data driven weights '
' Seeds_indices     ______  Start for Tracking Points setted to zero  '
' Distance          ______  Time Distance Output as 1 D array (Trace back via Ad Matrix to 2D,3D data)'
' Active List       ______  Optimal values of time '
' Narrow_Band       ______  Points on Active List complement such that there exists one Neighbour within Active_List'
' Narrow_Band_Val   ______  1D array for keeping the current values for indices'
' Far_Away          ______  Ramaining Grid-Points'
' Image_Data_Graph  ______  Adopted weights values of the input image to the grid (a_i input) '

' Termination Criterium :      Empty Narrow_band '


def Eikonal_Eq_Solve(Image, Ad, Seeds_indices):

    'Preprocessing Data (Weights)'

    Ad.data[Ad.data <= 0] = 1e-5

    'Increase the values for Far_Away'

    Image[Image <= 0] = 1e-5

    Length_Graph = Image.shape[0]

    ' Initialization '

    Distance = np.zeros(Length_Graph, )
    for k in Seeds_indices:
        Image[k] = 0
    Image_Data_Graph = Image[Ad.indices]
    Active_List = np.array(Seeds_indices)
    Active_List_Val = Image[Active_List]
    Far_Away = np.setdiff1d(np.array(range(Length_Graph)), Active_List)
    Narrow_Band = np.array([])

    ' Create Narrow Band '

    Ad_index = Create_Neigh_indices_vec(range(Length_Graph), Ad)

    for k in Active_List:
        Help_Index = []
        Help_Index.append(np.array(np.intersect1d(Ad_index[k], Far_Away)))
        Narrow_Band = np.append(Narrow_Band, (np.setdiff1d(np.array(Help_Index).flatten(), Narrow_Band)))

    'Initialization of Narrow_Band_Values'

    Narrow_Band_Val = Image[Narrow_Band.astype(int)]

    'Sort indices of the Narrow Band (increasing order)'

    Narrow_Band = Narrow_Band[np.argsort(Narrow_Band_Val)]

    'Sort the Preprocessed Values'

    Narrow_Band_Val = Narrow_Band_Val[np.argsort(Narrow_Band_Val)]

    ' Process Time Distance '
    ' Get the minimal value in Narrow Band '

    Active_List     = np.append(Active_List,Narrow_Band.astype(np.int32)[0])

    E = Active_List.astype(int)[-1]
        
    Active_List_Val = np.append(Active_List_Val,Image[Narrow_Band.astype(int)[0]])
        
    Narrow_Band     = Narrow_Band[1:]

    Narrow_Band_Val = Narrow_Band_Val[1:]
    
    Indices = np.setdiff1d(Ad.indices[Ad.indptr[E]:Ad.indptr[E + 1]],E)
        
    'Update the values for vertices within the Neighbourhod'
    'Here the Dijkstra`s methhod utilizes nearest neighbor'
        
    Narrow_Band,Narrow_Band_Val,Far_Away,Image = Update_Narrow_Band(Ad.data,Ad.indptr,Ad.indices,\
                       Far_Away.astype(np.int32),Narrow_Band_Val,\
                       Narrow_Band.astype(np.int32),Image,\
                        Indices.astype(np.int32),Image_Data_Graph, h_i = Ad.data[Ad.indptr[E]:Ad.indptr[E+1]])

    Narrow_Band     = Narrow_Band[np.argsort(Narrow_Band_Val)]
    
    Narrow_Band_Val = Narrow_Band_Val[np.argsort(Narrow_Band_Val)]
    

    while len(Far_Away) > 0:

        # Index

        Active_List = np.append(Active_List, Narrow_Band.astype(np.int32)[0])

        E = Active_List.astype(int)[-1]

        # Value

        Active_List_Val = np.append(Active_List_Val, Image[Narrow_Band.astype(int)[0]])
        Far_Away = np.setdiff1d(np.array(range(Length_Graph)),Active_List)
        Narrow_Band = Narrow_Band[1:]
        Narrow_Band_Val = Narrow_Band_Val[1:]
        Indices = Ad.indices[Ad.indptr[E]:Ad.indptr[E + 1]]

        'Update the values for vertices within the Neighbourhod'
        'Here the Dijkstra`s methhod utilizes nearest neighbor'

        Narrow_Band, Narrow_Band_Val, Far_Away, Image = FML.Update_Narrow_Band(Ad.data, Ad.indptr, Ad.indices, \
                                                                           Far_Away.astype(np.int32), Narrow_Band_Val, \
                                                                           Narrow_Band.astype(np.int32), Image, \
                                                                           Indices.astype(np.int32), Image_Data_Graph,
                                                                           h_i=Ad.data[Ad.indptr[E]:Ad.indptr[E + 1]])

        Narrow_Band = Narrow_Band[np.argsort(Narrow_Band_Val)]
        Narrow_Band_Val = Narrow_Band_Val[np.argsort(Narrow_Band_Val)]


    return Active_List, Active_List_Val
