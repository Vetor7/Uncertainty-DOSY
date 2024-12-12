from matplotlib.patches import Rectangle
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

def find_bounding_box(matrix, expand_margin=1):
    bounding_boxes = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i][j] != 0:
                # Find boundaries of the circle
                top, bottom, left, right = i, i, j, j
                while top > 0 and matrix[top-1][j] != 0:
                    top -= 1
                while bottom < rows - 1 and matrix[bottom+1][j] != 0:
                    bottom += 1
                while left > 0 and matrix[i][left-1] != 0:
                    left -= 1
                while right < cols - 1 and matrix[i][right+1] != 0:
                    right += 1
                # Calculate bounding box
                bounding_boxes.append(((left-expand_margin, top), (right+expand_margin, bottom)))
    return bounding_boxes

def merge_bounding_boxes(bounding_boxes, expand_margin_x=5, expand_margin_y=5):
    merged_boxes = []
    for box in bounding_boxes:
        merged = False
        for index, m_box in enumerate(merged_boxes):
            (x1, y1), (x2, y2) = box
            (mx1, my1), (mx2, my2) = m_box
            if x1 <= mx2 + expand_margin_x and x2 + expand_margin_x >= mx1 and y1 <= my2 + expand_margin_y and y2 + expand_margin_y >= my1:
                merged_boxes[index] = ((min(x1, mx1), min(y1, my1)), (max(x2, mx2), max(y2, my2)))
                merged = True
                break
        if not merged:
            merged_boxes.append(box)
    
    # Remove small rectangles inside larger rectangles
    final_boxes = []
    for box1 in merged_boxes:
        keep = True
        for box2 in merged_boxes:
            if box1 != box2:
                (x1, y1), (x2, y2) = box1
                (mx1, my1), (mx2, my2) = box2
                if x1 >= mx1 and y1 >= my1 and x2 <= mx2 and y2 <= my2:
                    keep = False
                    break
        if keep:
            final_boxes.append(box1)
    
    return final_boxes


def normalize_matrix_with_max_values(matrix1, matrix2, bounding_boxes, HNMR):
    normalized_matrix2 = np.copy(matrix2)
    
    max_values = []
    for bb in bounding_boxes:
        (x1, y1), (x2, y2) = bb
        if x1 == x2 or y1 == y2:
            continue
        box_values1 = matrix1[y1:y2+1, x1:x2+1]
        max_value = np.max(box_values1)
        if max_value != 0:
            max_values.append(max_value)
    
    # Lmax_value = np.max(HNMR[x1:x2,0]) if HNMR.size > 0 else 1
    
    for bb in bounding_boxes:
        (x1, y1), (x2, y2) = bb
        if x1 == x2 or y1 == y2: 
            continue
        box_values1 = matrix1[y1:y2+1, x1:x2+1]
        max_value = np.max(box_values1)
        box_values2 = matrix2[y1:y2+1, x1:x2+1]

        if max_value != 0:
            normalized_values = box_values2 / max_value
        else:
            # normalized_values = box_values2 / Lmax_value
            # normalized_values = np.where(normalized_values > 0.5, 1, 0)
            normalized_values = 0
        normalized_matrix2[y1:y2+1, x1:x2+1] = normalized_values
    
    return normalized_matrix2

def calculate_difference_mean(matrix1, bounding_boxes):
    diff_mean_list = []
    for bb in bounding_boxes:
        (x1, y1), (x2, y2) = bb
        if x1 == x2 or y1 == y2:
            continue
        box_values1 = matrix1[y1:y2+1, x1:x2+1]
        box_max1 = np.max(box_values1)

        diff_mean_list.append(box_max1)

    return diff_mean_list

def find_box(matrix, expand_margin=1, expand_margin_x=5, expand_margin_y=0):
    matrix = matrix.copy()
    matrix[matrix > 0] = 1
    dilated_var = binary_dilation(matrix, iterations=2)
    eroded_var = binary_erosion(dilated_var, iterations=1)
    matrix_eroded = eroded_var

    bounding_boxes = find_bounding_box(matrix_eroded, expand_margin)
    merged_boxes = merge_bounding_boxes(bounding_boxes, expand_margin_x=expand_margin_x, expand_margin_y=expand_margin_y)

    return merged_boxes

def show_box(axs, boxs):
    for bb in boxs:
        (x1, y1), (x2, y2) = bb
        width = x2 - x1 + 1
        height = y2 - y1 + 1
        rect = Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        axs.add_patch(rect)

def process_samples(start_idx, end_idx, Var, Test_out, Diff, labels, module):
    peak_max_list = np.array([]) 
    uncertainty_max_list = np.array([])
    diff_max_list = np.array([]) 
    
    for i in range(start_idx, end_idx):
        var = Var[i]
        test_out = Test_out[i]
        diff = Diff[i]
        label = labels[i]
        
        test_out[test_out < np.tile(((np.max(test_out, axis=1)) * 0.7)[:, np.newaxis], [1, module.hparams.label_size])] = 0
        test_out[test_out < np.max(test_out) * 0.1] = 0
        var[var < np.tile(((np.max(test_out, axis=1)) * 0.7)[:, np.newaxis], [1, module.hparams.label_size])] = 0
        var[var < np.max(var) * 0.2] = 0
        label[label < np.max(label) * 0.1] = 0

        out_boxes = find_box(var.T)

        stats = calculate_statistics(test_out.T, var.T, diff.T, bounding_boxes=out_boxes)
        peak_max_list = np.concatenate((peak_max_list, stats[0]))
        uncertainty_max_list = np.concatenate((uncertainty_max_list, stats[1]))
        diff_max_list = np.concatenate((diff_max_list, stats[2]))
    
    return peak_max_list, uncertainty_max_list, diff_max_list

def calculate_statistics(*matrices, bounding_boxes):
    return [box_max(matrix, bounding_boxes) for matrix in matrices]

def box_max(matrix, bounding_boxes):
    box_max_list = []
    for bb in bounding_boxes:
        (x1, y1), (x2, y2) = bb
        if x1 == x2 or y1 == y2:
            continue
        box_max = np.max(matrix[y1:y2+1, x1:x2+1])
        box_max_list.append(box_max)
    return box_max_list