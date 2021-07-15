import numpy as np
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import librosa as lb
import os.path
from pathlib import Path
import pickle
import multiprocessing
import time
import gc
from numba import jit, njit, cuda, float64, uint32
import torch
import timeit
import linmdtw
import pycuda

DTWDefaultSteps = np.array([[1, 1, 2],
                            [1, 2, 1]], dtype = np.uint32)

DTWDefaultWeights = np.array([2, 3, 3], dtype = np.float64)


subseqDTWDefaultSteps = np.array([[1, 1, 2],
                                  [1, 2, 1]], dtype = np.float64)

subseqDTWDefaultWeights = np.array([1, 1, 2], dtype = np.float64)


MAX_FLOAT = float('inf')

@njit(parallel = False)
def DTW_Cost_To_DandB(C, Steps = DTWDefaultSteps, weights = DTWDefaultWeights, subsequence=False):
    '''
    Find the accumulated cost matrix and backtrace matrix from a cost matrix using Dynamic time warping
    
    Arguments:
    C -- The Cost matrix
    Steps -- The available steps, where the first row is the row steps, the second row is the column steps
    weights -- The weights of the steps
    subsequence -- True if subsequence DTW should be performed rather than standard DTW
             
    Returns:
    D -- The accumulated cost matrix
    B -- The backtrace matrix
    '''
    '''
    Section for verifying input
    '''
    # Ensure parameters have correct dtypes and dimensions
    try:
        C = C.astype(np.float64)
        assert C.ndim == 2
    except:
        print("FAILURE: The type of the cost matrix is wrong - please pass in a 2-d numpy array")
        return

    try:
        Steps = Steps.astype(np.uint32)
        assert Steps.ndim == 2
    except:
        print("FAILURE: The type of the steps matrix is wrong - please pass in a 2-d numpy array")
        return

    try:
        weights = weights.astype(np.float64)
        assert weights.ndim == 1
    except:
        print("FAILURE: The type of the steps matrix is wrong - please pass in a 2-d numpy array")
        return
    
    # Ensure steps and weights sizes are appropriate
    stepsShape = Steps.shape
    numDifSteps = len(weights)
    
    if stepsShape[0] != 2:
        print("FAILURE: The size of the steps matrix is wrong - please pass in a 2-d numpy array with 2 rows")
        return
    if stepsShape[1] != numDifSteps:
        print("FAILURE: The number of steps and weights do not line up - please make sure that there are the same number of steps and weights")
        return
    
    '''
    Algorithm
    '''
    # Separate steps
    rowSteps = Steps[0,:]
    colSteps = Steps[1,:]

    # Define Relevant Variables
    numRows = C.shape[0]
    numCols = C.shape[1]
    
    numDifSteps = len(weights)
    maxRowStep = max(rowSteps)
    maxColStep = max(colSteps)
    
    # Set up accumulated cost matrix D and backtrace matrix B
    D = np.ones((numRows + maxRowStep, numCols + maxColStep), dtype = np.float64) * MAX_FLOAT
    B = np.zeros((numRows, numCols), dtype = np.uint32)
    
    # Fill up D and B
    if subsequence:  # Initialize entire bottom row of D for subsequence
        D[maxRowStep, maxColStep:] = C[0,:]
    else:
        D[maxRowStep, maxColStep] = C[0,0]  # Initialize bottom corner if for standard DTW
    for row in range(maxRowStep, numRows + maxRowStep, 1):
        for col in range(maxColStep, numCols + maxColStep, 1):
            bestCost = D[row, col]
            bestCostIndex = 0
            # Go through each step, find the best one
            for stepIndex in range(numDifSteps):
                costForStep = D[row - rowSteps[stepIndex], col - colSteps[stepIndex]] + weights[stepIndex] * C[row - maxRowStep, col - maxColStep]
                if costForStep < bestCost:
                    bestCost = costForStep
                    bestCostIndex = stepIndex
            # Save best cost and step
            D[row,col] = bestCost
            B[row - maxRowStep, col - maxColStep] = bestCostIndex
    
    # Return accumulated cost matrix D and backtrace matrix B
    return D[maxRowStep:, maxColStep:], B



@njit
def DTW_Backtrace(D, B, Steps=DTWDefaultSteps, subsequence=False, startCol=-1):
    '''
    Backtrace through an accumulated cost matrix and backtrace matrix to find a path
    
    Arguments:
    D -- The accumulated cost matrix
    B -- The backtrace matrix
    Steps -- The available steps
    subsequence -- True if subsequence DTW should be performed rather than standard DTW
    startCol -- The column to begin backtracing from, or -1 if not specified
    
    Returns:
    fwdPath -- A 2d numpy array storing the optimal path. The first row is the path through the rows.
            The second row is the path through the columns
    '''
    '''
    Section for verifying input
    '''
    # Ensure parameters have correct dtypes and dimensions
    try:
        D = D.astype(np.float64)
        assert D.ndim == 2
    except:
        print("FAILURE: The type of the accumulated cost matrix is wrong - please pass in a 2-d numpy array")
        return

    try:
        B = B.astype(np.uint32)
        assert B.ndim == 2
    except:
        print("FAILURE: The type of the backtrace matrix is wrong - please pass in a 2-d numpy array")
        return
    
    try:
        Steps = Steps.astype(np.uint32)
        assert Steps.ndim == 2
    except:
        print("FAILURE: The type of the steps matrix is wrong - please pass in a 2-d numpy array")
        return

    # Ensure that D and B are the same shape
    if D.shape != B.shape:
        print("FAILURE: The accumulated cost matrix and backtrace matrix are not the same size - please make sure their sizes match")
        return
    
    
    '''
    Backtrace through D and B
    '''
    # Separate steps
    rowSteps = Steps[0,:]
    colSteps = Steps[1,:]
    
    # Initialize variables
    numRows = D.shape[0]
    numCols = D.shape[1]
    
    curRow = numRows - 1  # Always start at last row
    curCol = numCols - 1  # Standard DTW: Start at top-right corner
    if startCol > 0:
        curCol = startCol
    elif subsequence:  # Subsequence: Choose lowest cost of top row
        curCol = np.argmin(D[numRows-1,:])
    
    endCol = curCol
    endCost = D[curRow, curCol]
    stepsInPath = 1
    stepIndex = 0
    done = (subsequence and curRow == 0) or (curRow == 0 and curCol == 0)
    path = np.zeros((2, numRows + numCols), dtype=np.uint32) # make as large as could need, then chop at the end
    path[0, 0] = curRow
    path[1, 0] = curCol
    # Backtrace
    while not done:
        if D[curRow, curCol] == MAX_FLOAT:  # No path exists to current location
            return
        
        # you're done if you've made it to the bottom left (non sub-sequence)
        # or just the bottom (sub-sequence)
        # find the step size
        curStepIndex = B[curRow, curCol]
        curRowStep = rowSteps[curStepIndex]
        curColStep = colSteps[curStepIndex]
        # backtrack by 1 step
        curRow = curRow - curRowStep
        curCol = curCol - curColStep
        # add your new location onto the path
        path[0, stepsInPath] = curRow
        path[1, stepsInPath] = curCol
        stepsInPath = stepsInPath + 1
        # check to see if you're done
        done = (subsequence and curRow == 0) or (curRow == 0 and curCol == 0)
    # reverse the path (a matrix with two rows) and return it
    fwdPath = np.fliplr(path[:, 0:stepsInPath])
    return fwdPath



@jit(forceobj = True)
def DTW(queryFeatureFile, refFeatureFile, Steps = DTWDefaultSteps, weights = DTWDefaultWeights, subsequence = False, \
        outfile = None, profile = False):
    '''
    Run DTW on query and reference feature matrices
    
    Arguments:
    queryFeatureFile -- The file containing the query feature matrix
    refFeatureFile -- The file containing the reference feature matrix
    Steps -- The steps matrix. The first row is the steps in along the rows,
             the second row is the steps along the columns
    weights -- The weights for the steps
    subsequence -- True if subsequence DTW should be performed rather than standard DTW
    
    Returns:
    path -- A 2d numpy array storing the optimal path. The first row is the path through the rows.
            The second row is the path through the columns
    '''
    # Extract Feature matrices
#     F1 = np.load(queryFeatureFile, allow_pickle = True)
#     F2 = np.load(refFeatureFile, allow_pickle = True)

    F1 = queryFeatureFile
    F2 = refFeatureFile
    
    # empty file if no valid path possible
    if not subsequence and max(F1.shape[1], F2.shape[1]) / min(F1.shape[1], F2.shape[1]) >= 2:
        if outfile:
            pickle.dump(None, open(outfile, 'wb'))
        return None
    
    # start time logging
    times = []
    times.append(time.time())
    
    # Compute Cost Matrix
    C = 1 - F1.T @ F2 # cos distance metric
    
    times.append(time.time())
    
    # Get D and B
    D, B = DTW_Cost_To_DandB(C, Steps = Steps, weights = weights, subsequence=subsequence)
    
    times.append(time.time())
    
    # Backtrace and return
    wp = DTW_Backtrace(D, B, Steps=Steps, subsequence=subsequence)
    
    times.append(time.time())
    
    # logging result if outfile exists
    if outfile:
        pickle.dump(wp, open(outfile, 'wb'))
    
    # return extra timing array if profiling timing
    if profile:
        return wp, np.diff(times)
    else:
        return wp

    


@njit
def getSegmentEndingLocs(wp):
    '''
    Takes in a segment level path through and returns the columns where each segment ends
    
    Arguments:
    wp -- A segment level path (as a 2d numpy array with 2 rows)
    
    Returns:
    endLocs -- A list where the ith entry is the column where the ith segment ends
    '''
    prevLoc = wp[:,0] # [r,c]
    endLocs = []
    for i in range(wp.shape[1]):
        curLoc = wp[:,i]
        if curLoc[0] != prevLoc[0]: # if row changes
            endLocs.append(curLoc[1])
        prevLoc = curLoc
        
    return endLocs



@cuda.jit
def GPU_DTW_shared(D, B, C, segLength, steps, weights):
    '''
    '''
    # Extract segments into shared memory
    i = cuda.grid(1)
    
    # init shared matrices
    #C_frag = cuda.shared.array((segLength,C.shape[1]), float64)
    #D_frag = cuda.shared.array((segLength,C.shape[1]), float64)
    C_shared = cuda.shared.array(shape=0, dtype=float64)
    
    segWidth = C.shape[1]
    
    #C_shared = np.zeros(segWidth*segLength*2 + 9, dtype = np.float64)
    
    
    COffset = 9
    DOffset = segWidth*segLength + COffset
    # fill matrices
    for r in range(segLength):
        for c in range(segWidth):
            C_shared[COffset+r*segWidth+c] = C[int(min(i*segLength+r, C.shape[0])), c]
            C_shared[DOffset+r*segWidth+c] = D[int(min(i*segLength+r, C.shape[0])), c]
    
    for j in range(steps.shape[0]):
        for k in range(steps.shape[1]):
            C_shared[3*j + k] = steps[j,k]
            
    for j in range(len(weights)):
        C_shared[6 + j] = weights[j]

    
    # Run DTW
    
    numSteps = 3
    
    # Fill up D and B
    for c in range(segWidth): # subsequence init
        C_shared[DOffset+c] = C_shared[COffset+c] 
    
    for row in range(segLength):
        for col in range(segWidth):
            bestCost = C_shared[DOffset+row*segWidth+col]
            bestCostIndex = 0
            # Go through each step, find the best one
            for stepIndex in range(numSteps):
                if row - C_shared[stepIndex] < 0 or col - C_shared[numSteps+stepIndex] < 0:
                    continue
                
                costForStep = C_shared[int(DOffset + (row - C_shared[stepIndex]) * segWidth + col \
                            - C_shared[numSteps + stepIndex])] + C_shared[int(2 * numSteps + stepIndex)] * \
                            C_shared[int(COffset + row*segWidth + col)]
                
                if costForStep < bestCost:
                    bestCost = costForStep
                    bestCostIndex = stepIndex
            # Save best cost and step
            C_shared[DOffset + row * segWidth + col] = bestCost
            B[row + i*segLength, col] = bestCostIndex
    
    # Copy D and B segments into global memory
    for r in range(segLength):
        if (i*segLength + r == C.shape[0]):
            break
        for c in range(segWidth):
            D[i*segLength+r, c] = C_shared[DOffset + r * segWidth + c]
       
    
    
@cuda.jit
def GPU_DTW_global(D, B, C, segLength, steps, weights):
    '''
    '''
    # Extract segments into shared memory
    i = cuda.grid(1)
    #i=0
    
    offset = i*segLength
    segWidth = C.shape[1]
    
    # Run DTW
    
    numSteps = 3
    
    # Fill up D and B
    for c in range(segWidth): # subsequence init
        D[offset,c] = C[offset,c]
    
    for row in range(segLength):
        if (offset + row == C.shape[0]):
            break
        for col in range(segWidth):
            bestCost = D[offset+row,col]
            bestCostIndex = 0
            # Go through each step, find the best one
            for stepIndex in range(numSteps):
                rowStep = steps[0, stepIndex]
                colStep = steps[1, stepIndex]
                if row - rowStep < 0 or col - colStep < 0:
                    continue
                
                costForStep = D[int(offset + (row - rowStep)), int(col - colStep)] + \
                              weights[stepIndex] * C[offset + row, col]
                
                if costForStep < bestCost:
                    bestCost = costForStep
                    bestCostIndex = stepIndex
            # Save best cost and step
            D[offset + row, col] = bestCost
            B[offset + row, col] = bestCostIndex


@jit(forceobj=True)
def GPU_WSDTW(queryFeatureFile, refFeatureFile, K, Steps = subseqDTWDefaultSteps, weights = subseqDTWDefaultWeights, \
              outfile = None, profile = False, gpu_backtrace=False):
    '''
    '''
    
    # Extract Feature matrices
    F1 = np.load(queryFeatureFile, allow_pickle = True)
    F2 = np.load(refFeatureFile, allow_pickle = True)
    
#     F1 = F1[:, :100]
#     F2 = F2[:, :100]
    
    
    
    if max(F1.shape[1], F2.shape[1]) / min(F1.shape[1], F2.shape[1]) >= 2: # no valid path possible
        print("FAILURE - Feature sizes are too different to align with DTW")
        return
    
    # start time logging
    times = []
    times.append(time.time())
    
    # Compute Cost
    # NOTE: This could also be done segment by segment.
    # I am doing it all up front to match the implementation in the old code.
    C = 1 - F1.T @ F2
    
    # Subsequence DTW each segment without backtracing
    segLength = int(np.ceil(F1.shape[1] / K))
    segWidth = C.shape[1]
    
    Cseg = np.zeros((K + 1, C.shape[1]), dtype = np.float64)
    D = np.ones(C.shape, dtype = np.float64) * MAX_FLOAT
    B = np.zeros(C.shape, dtype = np.int32)
    
    totalSize = C.nbytes + D.nbytes + B.nbytes
    totalSegmentSize = int(np.ceil((totalSize-B.nbytes) / K))  # If necessary, maybe just put C and D in shared memory
    times.append(time.time())
    if totalSize > torch.cuda.get_device_properties(0).total_memory:  # C > 11GB
        # Pass in fragment by fragment
        print('Fragment too large for global memory') 
        # Pass in as many fragments as possible at once
        # Some Function
        # Implement Later if Desired
    elif totalSegmentSize > 96000:  # Can't use shared memory
        print('Fragment too large for shared memory')
        GPU_DTW_global[K,1](D, B, C, segLength, Steps, weights)
        #GPU_DTW_global(D, B, C, segLength, Steps, weights)
    else:
        GPU_DTW_shared[K,1,0,totalSegmentSize+Steps.nbytes+weights.nbytes](D, B, C, segLength, Steps, weights)
        #GPU_DTW_shared(D, B, C, segLength, Steps, weights)
    times.append(time.time())
    for i in range(K):
        Cseg[i+1,:] = D[min((i+1)*segLength-1,D.shape[0]-1),:]
    
    # run segment-level DTW (Not subsequence)
    segmentSteps = np.array([[0, 1],
                             [1, C.shape[0]//(2 * K)]], #NOTE: This could cause differences with old implementation
                            dtype=np.uint32)
    segmentWeights = np.array([0, 1])
    
    Dseg, Bseg = DTW_Cost_To_DandB(Cseg, Steps = segmentSteps, weights = segmentWeights)
    times.append(time.time())
    wpseg = DTW_Backtrace(Dseg, Bseg, Steps=segmentSteps)
    times.append(time.time())
    # Backtrace through segments with segment level path as guide
    # On GPU?
    
    # Frame level backtrace segment by segment
    segmentEndIdxs = getSegmentEndingLocs(wpseg)
    
    # Final Backtrace ----------------------------------------------------------------------------
    if gpu_backtrace:
        times.append(time.time())
        path = np.zeros((2,K*np.sum(C.shape)), dtype=np.int32) - 1
        #final_path = []
        GPU_Backtrace_global[K,1](path, D, B, Steps, segLength, K, np.array(segmentEndIdxs))
        
        row_path = list(path[0,:])
        col_path = list(path[1,:])
        final_row_path = [i for i in row_path if i>=0]
        final_col_path = [j for j in col_path if j>=0]
        
        #p_indices = list(range(path.shape[1]))[::-1]
        #for p_row in path[0,:]:
        #    if p_row >= 0:
        #        final_path.append(p)
        #        print(final_path)
        #    else:
        #        continue
        #print(final_path)        
        wp_merged = np.fliplr(np.vstack((final_row_path, final_col_path)))
        
    else:
        path = []
        
        times.append(time.time())
        for i, endidx in enumerate(segmentEndIdxs):
            D_i = D[i*segLength:min((i+1)*segLength, D.shape[0]),:]
            B_i = B[i*segLength:min((i+1)*segLength, B.shape[0]),:]
            pathSeg = DTW_Backtrace(D_i, B_i, Steps = Steps, subsequence = True, startCol = endidx)
            # Add offset to row indices so they match with overall path
            pathSeg[0,:] = pathSeg[0,:] + (i * segLength)

            # Append fragment to full path
            path.append(pathSeg.copy())
            
        wp_merged = np.hstack(path)
        
    # --------------------------------------------------------------------------------------------
    times.append(time.time())
    
    if outfile:
        pickle.dump(wp_merged, open(outfile, 'wb'))

    if profile:
        return wp_merged, np.diff(times)
    else:
        return wp_merged
    

@cuda.jit
def GPU_Backtrace_global(path, D, B, steps, segLength, K, segmentEndIdxs):
    i = cuda.grid(1)
    
    # Initialize variables
    numRows = segLength
    numCols = D.shape[1]
    
    offset = i*segLength
    path_offset = (K-i-1)*numRows
    

    
    # init. path 1st row entries to -1 for post ordering
    for idx in range(numRows):
        path[0, path_offset+idx] = -1
        path[1, path_offset+idx] = -1
    
    curRow = min(numRows - 1, D.shape[0]-offset-1)  # Always start at last row
    curCol = segmentEndIdxs[i]
    
    endCol = curCol

    endCost = D[offset+curRow, curCol]
    stepsInPath = 1
    stepIndex = 0
    done = (curRow == 0) or (curRow == 0 and curCol == 0)
    #path = np.zeros((2, numRows + numCols), dtype=np.uint32) # make as large as could need, then chop at the end
    path[0, path_offset] = offset+curRow
    path[1, path_offset] = curCol
    
    # Backtrace
    while not done:
        if D[offset+curRow, curCol] == MAX_FLOAT:  # No path exists to current location
            return
        
        # you're done if you've made it to the bottom left (non sub-sequence)
        # or just the bottom (sub-sequence)
        # find the step size
        curStepIndex = B[offset+curRow, curCol]
        curRowStep = steps[0,curStepIndex]
        curColStep = steps[1,curStepIndex]
        # backtrack by 1 step
        curRow = int(curRow - curRowStep)
        curCol = int(curCol - curColStep)
        # add your new location onto the path
        path[0, path_offset+stepsInPath] = offset+curRow
        path[1, path_offset+stepsInPath] = curCol
        stepsInPath = stepsInPath + 1
        # check to see if you're done
        done = (curRow == 0) or (curRow == 0 and curCol == 0)


@jit(forceobj = True, parallel = False)
def WSDTW(queryFeatureFile, refFeatureFile, segments, Steps = subseqDTWDefaultSteps, weights = subseqDTWDefaultWeights, \
          outfile = None, profile = False):
    '''
    Runs a weakly-ordered segmental DTW between query and reference features matrices
    
    Arguments:
    queryFeatureFile -- The file containing the query feature matrix
    refFeatureFile -- The file containing the reference feature matrix
    segments -- The number of segments to divide F1 into
    Steps -- The allowed steps
    weights -- The weights for the steps
    
    Returns:
    path -- The optimal weakly-ordered, segmented alignment path between F1 and F2
    '''
    # Extract Feature matrices
#     F1 = np.load(queryFeatureFile, allow_pickle = True)#[:,:100]
#     F2 = np.load(refFeatureFile, allow_pickle = True)#[:,:100]

    F1 = queryFeatureFile
    F2 = refFeatureFile

    if max(F1.shape[1], F2.shape[1]) / min(F1.shape[1], F2.shape[1]) >= 2: # no valid path possible
        print("FAILURE - Feature sizes are too different to align with DTW")
        return
    
    # start time logging
    times = []
    times.append(time.time())
    
    # Compute Cost
    # NOTE: This could also be done segment by segment.
    # I am doing it all up front to match the implementation in the old code.
    C = 1 - F1.T @ F2
    
    times.append(time.time())
    
    # Subsequence DTW each segment without backtracing
    segLength = int(np.ceil(F1.shape[1] / segments))
    Cseg = np.zeros((segments + 1, C.shape[1]), dtype = np.float64)
    Dparts = []
    Bparts = []
    for i in range(segments):
        segStart = i * segLength
        segEnd = min(segStart + segLength, C.shape[0])
        # Ensuring that the segment is contiguous here ensures best performance in later computations
        currentSeg = np.ascontiguousarray(C[segStart:segEnd,:])
        D_i, B_i = DTW_Cost_To_DandB(currentSeg, Steps = Steps, weights = weights, subsequence = True)
        
        # Store D_i and B_i for segment and construct Cseg
        Dparts.append(D_i)
        Bparts.append(B_i)
        
        # NOTE: May need to separate this into a different loop for runtime profiling
        Cseg[i + 1,:] = D_i[-1,:]
    
    #return Dparts, Bparts
    times.append(time.time())
    
    # run segment-level DTW (Not subsequence)
    segmentSteps = np.array([[0, 1],
                             [1, C.shape[0]//(2 * segments)]], #NOTE: This could cause differences with old implementation
                            dtype=np.uint32)
    segmentWeights = np.array([0, 1])
    
    Dseg, Bseg = DTW_Cost_To_DandB(Cseg, Steps = segmentSteps, weights = segmentWeights)
    
    times.append(time.time())
    
    wpseg = DTW_Backtrace(Dseg, Bseg, Steps=segmentSteps)
    
    times.append(time.time())
    
    # Backtrace through segments with segment level path as guide
    path = []
    # Frame level backtrace segment by segment
    segmentEndIdxs = getSegmentEndingLocs(wpseg)
    
    times.append(time.time())
    
    for i, endidx in enumerate(segmentEndIdxs):
        pathSeg = DTW_Backtrace(Dparts[i], Bparts[i], Steps = Steps, subsequence = True, startCol = endidx)
        # Add offset to row indices so they match with overall path
        pathSeg[0,:] = pathSeg[0,:] + (i * segLength)

        # Append fragment to full path
        path.append(pathSeg.copy())
    
    wp_merged = np.hstack(path)
    
    times.append(time.time())
    
    if outfile:
        pickle.dump(wp_merged, open(outfile, 'wb'))

    if profile:
        return wp_merged, np.diff(times)
    else:
        return wp_merged
    

@jit(forceobj=True)
def GPU_WSDTW_fragmented(queryFeatureFile, refFeatureFile, K, M, \
                         Steps = subseqDTWDefaultSteps, weights = subseqDTWDefaultWeights, \
                         outfile = None, profile = False, gpu_backtrace=False, gpu_cost=False, version = 1):
    '''
    '''
    
    # Extract Feature matrices
#     F1 = np.load(queryFeatureFile, allow_pickle = True)
#     F2 = np.load(refFeatureFile, allow_pickle = True)
    
    F1 = queryFeatureFile
    F2 = refFeatureFile
    if max(F1.shape[1], F2.shape[1]) / min(F1.shape[1], F2.shape[1]) >= 2: # no valid path possible
        print("FAILURE - Feature sizes are too different to align with DTW")
        return
    
    segLength = int(np.ceil(F1.shape[1]/K))
    fragLength = int(np.ceil(F2.shape[1]/M) + 2*segLength)
    
    # start time logging
    times = []
    cost_times = []
    
    times.append(time.time())
    cost_times.append(time.time())
    # load steps/weights onto GPU
    steps_global = cuda.to_device(Steps)
    weights_global = cuda.to_device(weights)
    
    # allocate mem. for padding matrices
    overlap_global = cuda.device_array((F1.shape[1], 2*segLength*(M-1)))
    
    # Allocate mem for cum. cost and backtrace matrices
    cost_dims = (F1.shape[1], F2.shape[1])
    D_global = cuda.device_array(cost_dims)
    B_global = cuda.device_array(cost_dims, dtype=np.uint32)

    if gpu_cost:
        # load features onto GPU
        F1_global = cuda.to_device(F1)
        F2_global = cuda.to_device(F2)

        # allocate mem. for cost
        C_global = cuda.device_array(cost_dims)


        cost_times.append(time.time())

        GPU_Cost_And_DTW[K, M](C_global, D_global, B_global, \
                               F1_global, F2_global, \
                               steps_global, weights_global, \
                               segLength, fragLength, overlap_global, MAX_FLOAT)

        cost_times.append(time.time())

        D = D_global.copy_to_host()
        if version == 1:
            B = B_global.copy_to_host()

        times.append(time.time())
        cost_times.append(time.time())
        
    else:
        times.append(time.time())
        cost_times.append(time.time())

        # Compute Cost
        # NOTE: This could also be done segment by segment.
        # I am doing it all up front to match the implementation in the old code.
        C = 1 - F1.T @ F2

        times.append(time.time())
        cost_times.append(time.time())

        # load cost/steps/weights onto GPU
        C_global = cuda.to_device(C)
        steps_global = cuda.to_device(Steps)
        weights_global = cuda.to_device(weights)

        # allocate mem. for padding matrices
        overlap_global = cuda.device_array((F1.shape[1], 2*segLength*(M-1)))

        # allocate mem. for cost, cum. cost, and backtrace matrices
        D_global = cuda.device_array(C.shape)
        B_global = cuda.device_array(C.shape, dtype=np.uint32)

        cost_times.append(time.time())

        GPU_DTW_fragmented[K,M](C_global, D_global, B_global, \
                                steps_global, weights_global, \
                                segLength, fragLength, overlap_global, MAX_FLOAT)

        cost_times.append(time.time())

        D = D_global.copy_to_host()
        if version == 1:
            B = B_global.copy_to_host()

        times.append(time.time())
        cost_times.append(time.time())

    # Cseg compostion (could be GPU implemented)
    Cseg = np.zeros((K + 1, D.shape[1]), dtype = np.float64)
    for i in range(K):
        Cseg[i+1,:] = D[min((i+1)*segLength-1,D.shape[0]-1),:]

    # segment lvl. steps/weights
    segmentSteps = np.array([[0, 1],
                             [1, D.shape[0]//(2 * K)]], #NOTE: This could cause differences with old implementation
                            dtype=np.uint32)
    segmentWeights = np.array([0, 1])
    
#     if gpu_dtw_seg: # need to rework implementation to account for [0,1] skips to parallelize
#         # transfer segment-lvl. C to GPU
#         Cseg_global = cuda.to_device(Cseg)
        
#         # allocate mem. for seg. lvl. D & B
#         Dseg_global = cuda.device_array(Cseg.shape)
#         Bseg_global = cuda.device_array(Cseg.shape)
        
#         # transfer seg. lvl. steps/weights
#         steps_seg_global = cuda.to_device(segmentSteps)
#         weights_seg_global = cuda.to_device(segmentWeights)
        
#         # allocate mem. for seg. lvl. overlap
#         overlap_seg_global = cuda.device_array(K+1, 2*segLength*(M-1))
        
#         GPU_DTW_fragmented[1,M](Cseg_global, Dseg_global, Bseg_global, \
#                                 steps_seg_global, weights_seg_global, \
#                                 K+1, fragLength, overlap_seg_global, MAX_FLOAT)
        
    Dseg, Bseg = DTW_Cost_To_DandB(Cseg, Steps = segmentSteps, weights = segmentWeights)
    
    times.append(time.time())
    
#     if gpu_backtrace_seg:
        
#         # allocate mem. for path
#         path_seg_global = cuda.device_array((2,))
        
#         # allocate mem. for seg. lvl. D & B
#         Dseg_global = cuda.device_array(Cseg.shape)
#         Bseg_global = cuda.device_array(Cseg.shape)
        
#         # transfer seg. lvl. steps
#         steps_seg_global = cuda.to_device(segmentSteps)
        
#         GPU_Backtrace_global[K,1](path_seg_global, Dseg_global, Bseg_global, \
#                                   steps_seg_global, K+1, 1, segmentEndIdxs)
    
    wpseg = DTW_Backtrace(Dseg, Bseg, Steps=segmentSteps)
    # Backtrace through segments with segment level path as guide
    # On GPU?
    
    times.append(time.time())
    
    # Frame level backtrace segment by segment
    segmentEndIdxs = getSegmentEndingLocs(wpseg)
    
    times.append(time.time())
    
    # Final Backtrace ----------------------------------------------------------------------------
    if gpu_backtrace:
        times.append(time.time())
        
        # allocate mem. for path on GPU
        path_global = cuda.device_array((2, K*segLength), dtype=np.int32)
        #path_global = cuda.to_device(np.zeros((2,D.shape[0]), dtype=np.int32) - 1)
        
        # copy ending indices to GPU
        seg_end_idxs_global = cuda.to_device(segmentEndIdxs)
        
        GPU_Backtrace_global[K,1](path_global, D_global, B_global, \
                                  steps_global, segLength, K, seg_end_idxs_global)
#         GPU_Backtrace_global(path_global, D_global, B_global, \
#                                   steps_global, segLength, K, seg_end_idxs_global)
        
        
        path = path_global.copy_to_host()
        
#         row_path = list(path[0,:])
#         col_path = list(path[1,:])
#         final_row_path = [i for i in row_path if i>=0]
#         final_col_path = [j for j in col_path if j>=0]
#         wp_merged = np.fliplr(np.vstack((final_row_path, final_col_path)))
        
        wp_merged = np.fliplr(np.array([p for p in list(path.T) if p[0] >=0]).T)   
        
        times.append(time.time())
        
    else:
        times.append(time.time())
        
        path = []
        
        for i, endidx in enumerate(segmentEndIdxs):
            D_i = D[i*segLength:min((i+1)*segLength, D.shape[0]),:]
            B_i = B[i*segLength:min((i+1)*segLength, B.shape[0]),:]
            pathSeg = DTW_Backtrace(D_i, B_i, Steps = Steps, subsequence = True, startCol = endidx)
            # Add offset to row indices so they match with overall path
            pathSeg[0,:] = pathSeg[0,:] + (i * segLength)

            # Append fragment to full path
            path.append(pathSeg.copy())
            
        wp_merged = np.hstack(path)
        
        times.append(time.time())
        
    # --------------------------------------------------------------------------------------------
    
    if outfile:
        pickle.dump(wp_merged, open(outfile, 'wb'))

    if profile:
        return wp_merged, np.diff(times), np.diff(cost_times)
    else:
        return wp_merged

    

@cuda.jit
def GPU_Cost_And_DTW(C, D, B, \
                     F1, F2, \
                     steps, weights, \
                     segLength, fragLength, overlap, MAX_FLOAT):
    """
    """
    
    # index blocks/threads
    #i, j = cuda.grid(2)
    i = cuda.blockIdx.x
    j = cuda.threadIdx.x
    
    # offsets for diff. threads
    row_offset = i*segLength
    new_region_start = j*(fragLength - 2*segLength)  # Where the new information for this fragment starts
    overlap_offset = (j-1)*2*segLength
    
    # set actual segment/fragment lengths
    seg_length_adjusted = min(segLength, F1.shape[1]-row_offset)
    col_offset = max(0, new_region_start - 2*segLength)  # Where the fragment starts
    frag_end = min(new_region_start + fragLength - 2*segLength, F2.shape[1])
    frag_length_adjusted = frag_end - col_offset
    
    for row in range(seg_length_adjusted):
        for col in range(frag_length_adjusted):   
            # filling overlap and D w/ MAX_FLOAT
            if col_offset + col < new_region_start and j>0:
                overlap[row_offset + row, overlap_offset + col] = MAX_FLOAT
            else:
                D[row_offset+row, col_offset+col] = MAX_FLOAT
                # cost matrix composition
                s = 0
                for idx in range(F1.shape[0]):
                    s += F1[idx, row_offset + row] * F2[idx, col_offset + col]
                C[row_offset+row, col_offset+col] = 1 - s
                
    cuda.syncthreads()

    # initialize first row of D matrix
    for col in range(frag_length_adjusted): # subsequence init
        if col_offset + col < new_region_start and j>0:
            overlap[row_offset, overlap_offset + col] = C[row_offset, col_offset + col]
        else:
            D[row_offset, col_offset + col] = C[row_offset, col_offset + col]
    
    # Run DTW
    numSteps = weights.shape[0]
    
    for row in range(seg_length_adjusted):
        for col in range(frag_length_adjusted):
            if col_offset + col < new_region_start and j>0:
                bestCost = overlap[row_offset + row, overlap_offset + col]
            
                # Go through each step, find the best one
                for stepIndex in range(numSteps):
                    rowStep = steps[0, stepIndex]
                    colStep = steps[1, stepIndex]
                    if row - rowStep < 0 or col - colStep < 0:
                        continue

                    costForStep = overlap[int(row_offset + row - rowStep), int(overlap_offset + col - colStep)] + \
                                  weights[stepIndex] * C[row_offset + row, col_offset + col]

                    if costForStep < bestCost:
                        bestCost = costForStep
                # Save best cost and step
                overlap[row_offset + row, overlap_offset + col] = bestCost
            
            else:
                bestCost = D[row_offset + row, col_offset + col]
                bestCostIndex = 0

                # Go through each step, find the best one
                for stepIndex in range(numSteps):
                    rowStep = steps[0, stepIndex]
                    colStep = steps[1, stepIndex]
                    if row - rowStep < 0 or col - colStep < 0:
                        continue
                    
                    if col_offset + col - colStep < new_region_start:
                        costForStep = overlap[int(row_offset + row - rowStep), int(overlap_offset + col - colStep)] + \
                                      weights[stepIndex] * C[row_offset + row, col_offset + col]
                    else:
                        costForStep = D[int(row_offset + row - rowStep), int(col_offset + col - colStep)] + \
                                      weights[stepIndex] * C[row_offset + row, col_offset + col]

                    if costForStep < bestCost:
                        bestCost = costForStep
                        bestCostIndex = stepIndex
                # Save best cost and step
                D[row_offset + row, col_offset + col] = bestCost
                B[row_offset + row, col_offset + col] = bestCostIndex



                
@cuda.jit
def GPU_DTW_fragmented(C, D, B, \
                       steps, weights, \
                       segLength, fragLength, overlap, MAX_FLOAT):
    
    # index blocks/threads
    #i, j = cuda.grid(2)
    j = cuda.blockIdx.x
    i = cuda.threadIdx.x
    
    # offsets for diff. threads
    row_offset = i*segLength
    new_region_start = j*(fragLength - 2*segLength)  # Where the new information for this fragment starts
    overlap_offset = (j-1)*2*segLength
    
    # set actual segment/fragment lengths
    seg_length_adjusted = min(segLength, C.shape[0]-row_offset)
    col_offset = max(0, new_region_start - 2*segLength)  # Where the fragment starts
    frag_end = min(new_region_start + fragLength - 2*segLength, C.shape[1])
    frag_length_adjusted = frag_end - col_offset
    
    for row in range(seg_length_adjusted):
        for col in range(frag_length_adjusted):   
            # filling overlap and D w/ MAX_FLOAT
            if col_offset + col < new_region_start and j>0:
                overlap[row_offset + row, overlap_offset + col] = MAX_FLOAT
            else:
                D[row_offset+row, col_offset+col] = MAX_FLOAT
                
    # initialize first row of D matrix
    for col in range(frag_length_adjusted): # subsequence init
        if col_offset + col < new_region_start and j>0:
            overlap[row_offset, overlap_offset + col] = C[row_offset, col_offset + col]
        else:
            D[row_offset, col_offset + col] = C[row_offset, col_offset + col]
    
    # Run DTW
    numSteps = weights.shape[0]
    
    for row in range(seg_length_adjusted):
        for col in range(frag_length_adjusted):
            if col_offset + col < new_region_start and j>0:
                bestCost = overlap[row_offset + row, overlap_offset + col]
            
                # Go through each step, find the best one
                for stepIndex in range(numSteps):
                    rowStep = steps[0, stepIndex]
                    colStep = steps[1, stepIndex]
                    if row - rowStep < 0 or col - colStep < 0:
                        continue

                    costForStep = overlap[int(row_offset + row - rowStep), int(overlap_offset + col - colStep)] + \
                                  weights[stepIndex] * C[row_offset + row, col_offset + col]

                    if costForStep < bestCost:
                        bestCost = costForStep
                # Save best cost and step
                overlap[row_offset + row, overlap_offset + col] = bestCost
            
            else:
                bestCost = D[row_offset + row, col_offset + col]
                bestCostIndex = 0

                # Go through each step, find the best one
                for stepIndex in range(numSteps):
                    rowStep = steps[0, stepIndex]
                    colStep = steps[1, stepIndex]
                    if row - rowStep < 0 or col - colStep < 0:
                        continue
                    
                    if col_offset + col - colStep < new_region_start:
                        costForStep = overlap[int(row_offset + row - rowStep), int(overlap_offset + col - colStep)] + \
                                      weights[stepIndex] * C[row_offset + row, col_offset + col]
                    else:
                        costForStep = D[int(row_offset + row - rowStep), int(col_offset + col - colStep)] + \
                                      weights[stepIndex] * C[row_offset + row, col_offset + col]

                    if costForStep < bestCost:
                        bestCost = costForStep
                        bestCostIndex = stepIndex
                # Save best cost and step
                D[row_offset + row, col_offset + col] = bestCost
                B[row_offset + row, col_offset + col] = bestCostIndex
            

            


@cuda.jit
def GPU_DTW_costAsYouGo(D, B, F1, F2, steps, weights, segLength, fragLength, overlap, MAX_FLOAT):
    # Compute cost as you go
    
    # index blocks/threads
    i = cuda.blockIdx.x
    j = cuda.threadIdx.x
    
    # offsets for diff. threads
    row_offset = i*segLength
    new_region_start = j*(fragLength - 2*segLength)  # Where the new information for this fragment starts
    overlap_offset = (j-1)*2*segLength
    
    curCost = numba.float64(0)
    s = numba.float64(0)
    
    # set actual segment/fragment lengths
    seg_length_adjusted = min(segLength, F1.shape[1]-row_offset)
    col_offset = max(0, new_region_start - 2*segLength)  # Where the fragment starts
    frag_end = min(new_region_start + fragLength - 2*segLength, F2.shape[1])
    frag_length_adjusted = frag_end - col_offset
    
    for row in range(seg_length_adjusted):
        for col in range(frag_length_adjusted):   
            # filling overlap and D w/ MAX_FLOAT
            if col_offset + col < new_region_start and j>0:
                overlap[row_offset + row, overlap_offset + col] = MAX_FLOAT
            else:
                D[row_offset+row, col_offset+col] = MAX_FLOAT
                # cost matrix composition
#                 s = 0
#                 for idx in range(F1.shape[0]):
#                     s += F1[idx, row_offset + row] * F2[idx, col_offset + col]
#                 C[row_offset+row, col_offset+col] = 1 - s
                
    #cuda.syncthreads()

    # initialize first row of D matrix
    for col in range(frag_length_adjusted): # subsequence init
        s = 0
        for idx in range(F1.shape[0]):
            s += F1[idx, row_offset] * F2[idx, col_offset + col]
        curCost = numba.float64(1) - s
        if col_offset + col < new_region_start and j>0:
            overlap[row_offset, overlap_offset + col] = curCost
        else:
            D[row_offset, col_offset + col] = curCost
    
    # Run DTW
    numSteps = weights.shape[0]
    
    for row in range(seg_length_adjusted):
        for col in range(frag_length_adjusted):
            # calculate current elem in cost matrix
            s = 0
            for idx in range(F1.shape[0]):
                s += F1[idx, row_offset + row] * F2[idx, col_offset + col]
            curCost = numba.float64(1) - s
            
            if col_offset + col < new_region_start and j>0:
                bestCost = overlap[row_offset + row, overlap_offset + col]
            
                # Go through each step, find the best one
                for stepIndex in range(numSteps):
                    rowStep = steps[0, stepIndex]
                    colStep = steps[1, stepIndex]
                    if row - rowStep < 0 or col - colStep < 0:
                        continue

                    costForStep = overlap[int(row_offset + row - rowStep), int(overlap_offset + col - colStep)] + \
                                  weights[stepIndex] * curCost

                    if costForStep < bestCost:
                        bestCost = costForStep
                # Save best cost and step
                overlap[row_offset + row, overlap_offset + col] = bestCost
            
            else:
                bestCost = D[row_offset + row, col_offset + col]
                bestCostIndex = 0

                # Go through each step, find the best one
                for stepIndex in range(numSteps):
                    rowStep = steps[0, stepIndex]
                    colStep = steps[1, stepIndex]
                    if row - rowStep < 0 or col - colStep < 0:
                        continue
                    
                    if col_offset + col - colStep < new_region_start:
                        costForStep = overlap[int(row_offset + row - rowStep), int(overlap_offset + col - colStep)] + \
                                      weights[stepIndex] * curCost
                    else:
                        costForStep = D[int(row_offset + row - rowStep), int(col_offset + col - colStep)] + \
                                      weights[stepIndex] * curCost

                    if costForStep < bestCost:
                        bestCost = costForStep
                        bestCostIndex = stepIndex
                # Save best cost and step
                D[row_offset + row, col_offset + col] = bestCost
                B[row_offset + row, col_offset + col] = bestCostIndex

                

    
    
@cuda.jit
def GPU_Parallelizing_Test(D, B, C, segLength, steps, weights):
    '''
    '''
    # Extract segments into shared memory
    i = cuda.grid(1)
    #i=0
    
    offset = i*segLength
    segWidth = C.shape[1]
    
    # Run DTW
    
    numSteps = 3
    
    # Fill up D and B
    for c in range(segWidth): # subsequence init
        D[offset,c] = C[0,c]
    
    for row in range(segLength):
        if (offset + row == C.shape[0]):
            break
        for col in range(segWidth):
            bestCost = D[offset+row,col]
            bestCostIndex = 0
            # Go through each step, find the best one
            for stepIndex in range(numSteps):
                rowStep = steps[0, stepIndex]
                colStep = steps[1, stepIndex]
                if row - rowStep < 0 or col - colStep < 0:
                    continue
                
                costForStep = D[int(offset + (row - rowStep)), int(col - colStep)] + \
                              weights[stepIndex] * C[row, col]
                
                if costForStep < bestCost:
                    bestCost = costForStep
                    bestCostIndex = stepIndex
            # Save best cost and step
            D[offset + row, col] = bestCost
            B[offset + row, col] = bestCostIndex