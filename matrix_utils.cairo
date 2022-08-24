from starkware.cairo.common.alloc import alloc
from starkware.cairo.common.math import (
        unsigned_div_rem,
        sign
)

from matrix_console_utils import (
    show_matrix,
    show_array,
    show_fix_array
)

from matrix_data_types import Matrix_2d

from small_math import (
    Fix64x61,
    show_Fix,
    Small_Math_mul,
    Small_Math_div,
    Small_Math_add,
    Small_Math_sub
)

######################################################################
####################### COMPLEX OPERATIONS ###########################
######################################################################

func get_triangular_normal_matrix{range_check_ptr}(matrix: Matrix_2d) -> (res: Matrix_2d):
    alloc_locals
    let (local temp_res: Matrix_2d) = get_canonical_form (matrix, matrix.col_size)
    show_matrix(temp_res, 4)
    let (local res: Matrix_2d) = get_diagonal_form(temp_res, matrix.col_size)
    return(res =res)
end

func get_diagonal_form{range_check_ptr}(matrix: Matrix_2d, col_size: felt) -> (res: Matrix_2d):
    alloc_locals
    if col_size == 1:
        return(res = matrix)
    end
    let (local zero_values_above_matrix: Matrix_2d) = get_zeros_above(matrix, col_size - 1, col_size - 1)
    let (local res: Matrix_2d) = get_diagonal_form(zero_values_above_matrix, col_size - 1)
    return(res = res)
end

func get_zeros_above{range_check_ptr}(matrix: Matrix_2d, base_index: felt, col_size: felt) -> (res: Matrix_2d):
    alloc_locals
    if col_size == 0:
        return(res = matrix)
    end
    let (local res_temp: Matrix_2d) = get_zeros_above(matrix, base_index, col_size - 1)
    let (local normalizing_scalar: Fix64x61) = get_value(matrix, col_size-1, base_index)
    let (local base_row: Fix64x61*) = get_values_for_row(matrix, base_index)
    let (local adjusted_base_row: Fix64x61*) = mul_fix_array_by_scalar(base_row, matrix.row_size, normalizing_scalar)
    let (local current_row: Fix64x61*) = get_values_for_row(matrix, col_size - 1)
    let (local new_row: Fix64x61*) = sub_fix_array(current_row, matrix.row_size, adjusted_base_row, matrix.row_size)
    let (local new_zero_value_matrix: Matrix_2d) = replace_row(
        res_temp, new_row, matrix.row_size, col_size - 1)
    return(res = new_zero_value_matrix)
end

func get_canonical_form{range_check_ptr}(matrix: Matrix_2d, col_size: felt) -> (res: Matrix_2d):
    alloc_locals
    if col_size == 0:
        return(res = matrix)
    end
    let (local res: Matrix_2d) = get_canonical_form(matrix, col_size - 1)
    let (local non_zero_base_value_matrix: Matrix_2d) = get_non_zero_base_value(res, col_size - 1)
    let (local zero_values_under_matrix: Matrix_2d) = get_zeros_under(non_zero_base_value_matrix, col_size - 1)
    return(res = zero_values_under_matrix)
end

func get_non_zero_base_value(matrix: Matrix_2d, base_value: felt) -> (res: Matrix_2d):
    alloc_locals
    let (local index_first_non_zero) = get_first_non_zero_in_col(matrix, base_value)
    let (local res: Matrix_2d) = swap_rows(matrix, base_value, index_first_non_zero)
    return(res = res)
end

func get_zeros_under{range_check_ptr}(matrix: Matrix_2d, row_index: felt) -> (res: Matrix_2d):
    alloc_locals
    # get normalized row for row_index
    let (local normalizing_scalar: Fix64x61) = get_value(matrix, row_index, row_index)
    let (local current_row: Fix64x61*) = get_values_for_row(matrix, row_index)
    let (local normalized_row: Fix64x61*) = div_fix_array_by_scalar(current_row, matrix.row_size, normalizing_scalar)
    let (local norm_row_matrix: Matrix_2d) = replace_row(
        matrix, normalized_row, matrix.row_size, row_index)
    let (local res: Matrix_2d) = get_zeros_under_inner(norm_row_matrix, normalized_row, row_index, matrix.col_size - 1)
    return(res = res)
end

func get_zeros_under_inner{range_check_ptr}(
        matrix: Matrix_2d, normalized_row: Fix64x61*, base_index: felt, row_index: felt
        ) -> (res: Matrix_2d):
    alloc_locals
    if row_index == base_index:
        return(res = matrix)
    end
    let (local intermediate_res: Matrix_2d) = get_zeros_under_inner(matrix, normalized_row, base_index, row_index - 1)
    let (local current_row: Fix64x61*) = get_values_for_row(matrix, row_index)
    let (local scalar_under: Fix64x61) = get_value(matrix, row_index, base_index)
    let (local row_to_substract: Fix64x61*) = mul_fix_array_by_scalar(
        normalized_row, matrix.row_size, scalar_under)
    let (local new_row: Fix64x61*) = sub_fix_array(
        current_row, matrix.row_size, row_to_substract, matrix.row_size)
    let (local res: Matrix_2d) = replace_row(
        intermediate_res, new_row, matrix.row_size, row_index)
    return(res = res)
end

######################################################################
########################## MATH OPERATIONS ###########################
######################################################################

func divide_row_by_value_in_index{range_check_ptr}(matrix: Matrix_2d, row_target: felt, val_index: felt) -> (res: Matrix_2d):
    alloc_locals
    let (local row_selection: felt*) = get_indexes_for_row(matrix, row_target)
    local values: Fix64x61* = matrix.values
    local scalar: Fix64x61 = [values + val_index]
    let (local res_values: Fix64x61*) = divide_selection_by_scalar(matrix, matrix.size, row_selection, matrix.row_size, scalar)
    local res: Matrix_2d = Matrix_2d(matrix.size, matrix.row_size, matrix.col_size, res_values)
    return(res = res)
end

######################################################################
###################### CONSTRUCTIVE OPERATIONS #######################
######################################################################

func swap_rows(matrix: Matrix_2d, row_1_index: felt, row_2_index: felt) -> (res: Matrix_2d):
    alloc_locals
    let (local permutation: felt*) = get_indexes_two_elem_permutation(matrix.col_size, row_1_index, row_2_index)
    let (local new_indexes: felt*) = get_indexes_for_swapping_rows(matrix, permutation)
    let (res_values: Fix64x61*) = pick_fix_array_values(matrix.values, matrix.size, new_indexes, matrix.size)
    local res: Matrix_2d = Matrix_2d(matrix.size, matrix.row_size, matrix.col_size, res_values)
    return(res = res)
end

func add_row_to_matrix (matrix: Matrix_2d, row: Fix64x61*, row_size: felt) -> (res: Matrix_2d):
    alloc_locals
    if row_size != matrix.row_size:
        return(res = matrix)
    end
    concat_fix_array(matrix.values, matrix.size, row, row_size)
    local res: Matrix_2d = Matrix_2d(matrix.size + row_size, matrix.row_size,  matrix.col_size + 1, matrix.values)
    return(res = res)
end 

# as usual the redundant value row_size is required for security measures
func replace_row(matrix: Matrix_2d, row: Fix64x61*, row_size: felt, row_index: felt
        ) -> (res: Matrix_2d):
    alloc_locals
    # This should look more like a with_attr error handling: replace return val by void matrix
    if row_size != matrix.row_size:
        return(res = matrix)
    end
    let (local res_val: Fix64x61*) = replace_row_inner(matrix, matrix.col_size, row, row_index)
    return(res = Matrix_2d(matrix.size, matrix.row_size, matrix.col_size, res_val))
end

func replace_row_inner(matrix: Matrix_2d, size: felt, row: Fix64x61*, row_index: felt
        ) -> (res: Fix64x61*):
    alloc_locals
    if size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (res: Fix64x61*) = replace_row_inner(matrix, size - 1, row, row_index)
    let (local next_row: Fix64x61*) = choose_row(matrix, size - 1, row, row_index)
    concat_fix_array(res, (size - 1) * matrix.row_size, next_row, matrix.row_size)
    return(res = res)
end

func choose_row(matrix: Matrix_2d, current_index: felt, row: Fix64x61*, row_index) -> (res: Fix64x61*):
    alloc_locals
    if current_index == row_index:
        return(res = row)
    end
    let (local res: Fix64x61*) = get_values_for_row(matrix, current_index)
    return(res = res)
end

# func column functions

######################################################################
########################### GETTING VALUES ###########################
######################################################################

# returns a value in a matrix, given row and col
func get_value(matrix: Matrix_2d, row_index: felt, col_index: felt) -> (res: Fix64x61):
    let matrix_values: Fix64x61* = matrix.values
    let res: Fix64x61 = [matrix_values + row_index * matrix.row_size + col_index]
    return(res = res)
end

# Gets the values of a matrix corresponding to a given set of indexes
func get_selected_values (matrix: Matrix_2d, indexes: felt*, indexes_size: felt) -> (selected_values: Fix64x61*):
    alloc_locals
    if indexes_size == 0:
        let (selected_values: Fix64x61*) = alloc()
        return(selected_values = selected_values)
    end
    let (selected_values: Fix64x61*) = get_selected_values (matrix, indexes, indexes_size - 1)
    let matrix_values: Fix64x61* = matrix.values
    tempvar matrix_values_index = [indexes + indexes_size - 1]
    assert [selected_values + indexes_size - 1] = [matrix_values + matrix_values_index]
    return (selected_values = selected_values)
end

# ROW SELECTOR WRAPPER 
# inner function uses index starting from 0
func get_values_for_row (
        matrix: Matrix_2d,
        row_index: felt,
        ) -> (
        values: Fix64x61*
        ):
    let (values: Fix64x61*) = get_values_for_row_inner (matrix, row_index, matrix.row_size)
    return (values = values)
end

func get_values_for_row_inner (
        matrix: Matrix_2d,
        row_index: felt,
        row_size: felt
        ) -> (
        values: Fix64x61*
        ):
    alloc_locals
    # Using recursive pattern: callfirst with Fix array
    if row_size == 0:
        let (values: Fix64x61*) = alloc()
        return(values = values)
    end
    let (values: Fix64x61*) = get_values_for_row_inner (matrix, row_index, row_size - 1)
    # index selector: collects all values starting from row_index * matrix.row_size (first element of row row_index)
    tempvar matrix_index = row_index * matrix.row_size + row_size - 1
    assert [values + row_size - 1] = [matrix.values + matrix_index]
    return(values = values)
end

# COL SELECTOR WRAPPER 
# inner function uses index starting from 0
func get_values_for_col (
        matrix: Matrix_2d,
        col_index: felt,
        ) -> (
        values: Fix64x61*
        ):
    let (values: Fix64x61*) = get_values_for_col_inner (matrix, col_index, matrix.col_size)
    return (values = values)
end

func get_values_for_col_inner (
        matrix: Matrix_2d,
        col_index: felt,
        col_size: felt
        ) -> (
        values: Fix64x61*
        ):
    alloc_locals
    # Using recursive pattern: callfirst with Fix array
    if col_size == 0:
        let (values: Fix64x61*) = alloc()
        return(values = values)
    end
    let (values: Fix64x61*) = get_values_for_col_inner (matrix, col_index, col_size - 1)
    # index selector: collects all values starting from col_index (first element of col col_index) and adding row_size
    tempvar matrix_index = col_index + matrix.row_size * (col_size - 1)
    tempvar element = [matrix.values + matrix_index]
    assert [values + col_size - 1] = [matrix.values + matrix_index]
    return(values = values)
end

# gets index for the first non zero element in a given column
func get_first_non_zero_in_col (matrix: Matrix_2d, col_target: felt) -> (res: felt):
    let (res) = get_first_non_zero_in_col_inner(matrix, col_target, matrix.col_size)
    return(res = res)
end

func get_first_non_zero_in_col_inner(matrix: Matrix_2d, col_target: felt, col_size: felt) -> (res: felt):
    alloc_locals
    if col_size == col_target:
        return(res = -1)
    end
    let (res) = get_first_non_zero_in_col_inner(matrix, col_target, col_size - 1)
    local val_index = (col_size - 1) * matrix.row_size + col_target
    local values: Fix64x61* = matrix.values
    local current_value = [values + val_index].val
    if current_value != 0 and res == -1:
        return(res = col_size - 1)
    end
    return(res = res)
end

# Receives an index and a fix_array and returns 1 if the corresponding value is not zero, otherwise 0
func check_val_non_zero (fix_array: Fix64x61*, index: felt) -> (res: felt):
    if [fix_array + index * Fix64x61.SIZE] == 0:
        return(res = 0)
    end
    return(res = 1)
end

######################################################################
################## CHECKING CONSISTENCY ##############################
######################################################################
func check_row_index {range_check_ptr}(matrix: Matrix_2d, index: felt) -> (res: felt):
    let (index_sign) = sign(index - 1)
    if index_sign == -1:
        return (res = 0)
    end
    # col_size = number of rows
    let (int_div, _) = unsigned_div_rem (index - 1, matrix.col_size)
    if int_div != 0:
        return (res = 0)
    end
    return (res = 1)
end

func check_col_index {range_check_ptr}(matrix: Matrix_2d, index: felt) -> (res: felt):
    let (index_sign) = sign(index - 1)
    if index_sign == -1:
        return (res = 0)
    end
    # row_size = number of cols
    let (int_div, _) = unsigned_div_rem (index - 1, matrix.row_size)
    if int_div != 0:
        return (res = 0)
    end
    return (res = 1)
end

######################################################################
################## ROW AND COL SELECTORS #############################
######################################################################

##### WARNING: THESE ARE INTERNAL FUNCTIONS AND DON'T FOLLOW THE CONVENTION ABOUT INDExES STARTING WITH 1 #####

func get_indexes_for_swapping_cols (
        matrix: Matrix_2d, permutation: felt*
        ) -> (new_indexes: felt*):
    let (new_indexes: felt*) = get_indexes_for_swapping_cols_inner(
        matrix, permutation, matrix.size, matrix.col_size, matrix.row_size)
    return (new_indexes = new_indexes)
end

func get_indexes_for_swapping_cols_inner (
        matrix: Matrix_2d, permutation: felt*, matrix_size: felt, row_index: felt, col_index: felt
        ) -> (new_indexes: felt*):
    alloc_locals
    if matrix_size == 0:
        let (new_indexes: felt*) = alloc()
        return (new_indexes = new_indexes)
    end
    let (row_var, col_var) = standard_fill_get_indexes_var (matrix, row_index, col_index) 

    let (new_indexes: felt*) = get_indexes_for_swapping_cols_inner (
        matrix, permutation, matrix_size - 1, row_index + row_var, col_index + col_var)  
    assert [new_indexes + matrix_size - 1] = (row_index - 1) * matrix.row_size + [permutation + col_index - 1]

    return(new_indexes = new_indexes)
end

func get_indexes_for_swapping_rows (
        matrix: Matrix_2d, permutation: felt*
        ) -> (new_indexes: felt*):
    let (new_indexes: felt*) = get_indexes_for_swapping_rows_inner(
        matrix, permutation, matrix.size, matrix.col_size, matrix.row_size)
    return (new_indexes = new_indexes)
end

func get_indexes_for_swapping_rows_inner (
        matrix: Matrix_2d, permutation: felt*, matrix_size: felt, row_index: felt, col_index: felt
        ) -> (new_indexes: felt*):
    alloc_locals
    if matrix_size == 0:
        let (new_indexes: felt*) = alloc()
        return (new_indexes = new_indexes)
    end
    let (row_var, col_var) = standard_fill_get_indexes_var (matrix, row_index, col_index) 

    let (new_indexes: felt*) = get_indexes_for_swapping_rows_inner (
        matrix, permutation, matrix_size - 1, row_index + row_var, col_index + col_var)  
    assert [new_indexes + matrix_size - 1] = (col_index - 1) + [permutation + row_index - 1] * matrix.row_size
    return(new_indexes = new_indexes)
end

# This little func is used for adjusting row and col variations while using the normal_fill recursive pattern
# Take into consideration that it uses the indexes starting with 1, not 0, the same as the size/index in the pattern,
# so you may have to substract one (i.e. row_index - 1, col_index) to get the actual index of the felt*
func standard_fill_get_indexes_var (matrix: Matrix_2d, row_index: felt, col_index: felt) -> (row_var: felt, col_var: felt):
    if col_index == 1:
        tempvar col_var = matrix.row_size - 1
        return(row_var = -1, col_var = col_var)
    end
    return(row_var = 0, col_var = -1)
end

# returns a permutation of two elements in an array, the rest of elements preserve their 'natural' value
func get_indexes_two_elem_permutation(size: felt, elem_1: felt, elem_2: felt) -> (res: felt*):
    alloc_locals
    if size == 0:
        let (res: felt*) = alloc()
        return(res = res)
    end
    let (local res: felt*) = get_indexes_two_elem_permutation(size - 1, elem_1, elem_2)
    let (local new_elem) = assign_two_elem_permutation(size - 1, elem_1, elem_2)
    assert[res + size - 1] = new_elem
    return(res = res)
end

func assign_two_elem_permutation(index: felt, elem_1: felt, elem_2: felt) -> (res: felt):
    if index == elem_1:
        return(res = elem_2)
    end
    if index == elem_2:
        return(res = elem_1)
    end
    return(res = index)
end

### WARNING ###
# ROW INDEXES SELECTOR WRAPPER ### WARNING ### returns indexes starting from 0, these are internal functions
func get_indexes_for_row (
        matrix: Matrix_2d,
        row_index: felt,
        ) -> (
        indexes: felt*
        ):
    let (indexes: felt*) = get_indexes_for_row_inner (matrix, row_index, matrix.row_size)
    return (indexes = indexes)
end

func get_indexes_for_row_inner (
        matrix: Matrix_2d,
        row_index: felt,
        row_size: felt
        ) -> (
        indexes: felt*
        ):
    alloc_locals
    # Using recursive pattern: callfirst with Fix array
    if row_size == 0:
        let (indexes: felt*) = alloc()
        return(indexes = indexes)
    end
    let (indexes: felt*) = get_indexes_for_row_inner (matrix, row_index, row_size - 1)
    # index selector: collects all indexes starting from row_index * matrix.row_size (first element of row row_index)
    tempvar matrix_index = row_index * matrix.row_size + row_size - 1
    assert [indexes + row_size - 1] = matrix_index
    return(indexes = indexes)
end

### WARNING ###
# COL INDExES SELECTOR WRAPPER ### WARNING ### returns indexes starting from 0, these are internal functions
func get_indexes_for_col (
        matrix: Matrix_2d,
        col_index: felt,
        ) -> (
        indexes: felt*
        ):
    let (indexes: felt*) = get_indexes_for_col_inner (matrix, col_index, matrix.col_size)
    return (indexes = indexes)
end

func get_indexes_for_col_inner (
        matrix: Matrix_2d,
        col_index: felt,
        col_size: felt
        ) -> (
        indexes: felt*
        ):
    alloc_locals
    # Using recursive pattern: callfirst with Fix array
    if col_size == 0:
        let (indexes: felt*) = alloc()
        return(indexes = indexes)
    end
    let (indexes: felt*) = get_indexes_for_col_inner (matrix, col_index, col_size - 1)
    # index selector: collects all indexes starting from col_index (first element of col col_index) and adding row_size
    tempvar matrix_index = col_index + matrix.row_size * (col_size - 1)
    assert [indexes + col_size - 1] = matrix_index
    return(indexes = indexes)
end

######################################################################
##################### MATRIX OPERATIONS ##############################
######################################################################

func multiply_matrix_by_scalar {range_check_ptr}(
        matrix: Matrix_2d, scalar: Fix64x61
        ) -> (
        result_matrix: Matrix_2d
        ):
    let (result_matrix_values: Fix64x61*) = multiply_matrix_by_scalar_inner (matrix, matrix.size, scalar)
    let result_matrix = Matrix_2d(matrix.size, matrix.row_size, matrix.col_size, result_matrix_values)
    return (result_matrix = result_matrix)
end

func multiply_matrix_by_scalar_inner {range_check_ptr}(matrix: Matrix_2d, matrix_size: felt, scalar: Fix64x61) -> (result_matrix_values: Fix64x61*):
    alloc_locals
    if matrix_size == 0:
        let result_matrix_values: Fix64x61* = alloc()
        return(result_matrix_values = result_matrix_values)
    end
    let (result_matrix_values: Fix64x61*) = multiply_matrix_by_scalar_inner (matrix, matrix_size - 1, scalar)
    let matrix_value: Fix64x61 = [matrix.values + matrix_size - 1]
    let (result_value: Fix64x61) = Small_Math_mul (matrix_value, scalar)
    assert [result_matrix_values + matrix_size - 1] = result_value
    return(result_matrix_values = result_matrix_values)
end

func add_scalar_to_matrix {range_check_ptr}(
        matrix: Matrix_2d, scalar: Fix64x61
        ) -> (
        result_matrix: Matrix_2d
        ):
    let (result_matrix_values: Fix64x61*) = add_scalar_to_matrix_inner (matrix, matrix.size, scalar)
    let result_matrix = Matrix_2d(matrix.size, matrix.row_size, matrix.col_size, result_matrix_values)
    return (result_matrix = result_matrix)
end

func add_scalar_to_matrix_inner {range_check_ptr}(matrix: Matrix_2d, matrix_size: felt, scalar: Fix64x61) -> (result_matrix_values: Fix64x61*):
    alloc_locals
    if matrix_size == 0:
        let result_matrix_values: Fix64x61* = alloc()
        return(result_matrix_values = result_matrix_values)
    end
    let (result_matrix_values: Fix64x61*) = add_scalar_to_matrix_inner (matrix, matrix_size - 1, scalar)
    let matrix_value: Fix64x61 = [matrix.values + matrix_size - 1]
    let (result_value: Fix64x61) = Small_Math_add (matrix_value, scalar)
    assert [result_matrix_values + matrix_size - 1] = result_value
    return(result_matrix_values = result_matrix_values)
end

######################################################################
##################### SELECTION OPERATIONS ###########################
######################################################################

# divides selection by scalar
func substract_selection_by_scalar {range_check_ptr}(
        matrix: Matrix_2d,
        matrix_size: felt,
        selection: felt*,
        selection_size: felt,
        scalar: Fix64x61
        ) -> (
        result_matrix_values: Fix64x61*
        ):
    alloc_locals
    if matrix_size == 0:
        let (result_matrix_values: Fix64x61*) = alloc()
        return(result_matrix_values = result_matrix_values)
    end
    let (result_matrix_values: Fix64x61*) = substract_selection_by_scalar(matrix, matrix_size - 1, selection, selection_size, scalar)
    let (value_is_in_array: felt) = is_in_array(selection, selection_size, matrix_size - 1)
    let current_fix_element: Fix64x61 =  [matrix.values + matrix_size - 1]
    let (resulting_value: Fix64x61) = conditional_div(current_fix_element, scalar, value_is_in_array)
    assert [result_matrix_values + matrix_size - 1] = resulting_value
    return(result_matrix_values = result_matrix_values)
end

# This func returns the division in case condition = 1, and returns primary_term otherwise
func conditional_sub {range_check_ptr}(primary_term: Fix64x61, secondary_term: Fix64x61, condition: felt) -> (result: Fix64x61):
    if condition == 1:
        let (result: Fix64x61) = Small_Math_sub(primary_term, secondary_term)
        return(result = result)
    end
    return(result = primary_term)
end

# multiplies selection by scalar
func multiply_selection_by_scalar {range_check_ptr}(
        matrix: Matrix_2d,
        matrix_size: felt,
        selection: felt*,
        selection_size: felt,
        scalar: Fix64x61
        ) -> (
        result_matrix_values: Fix64x61*
        ):
    alloc_locals
    if matrix_size == 0:
        let (result_matrix_values: Fix64x61*) = alloc()
        return(result_matrix_values = result_matrix_values)
    end
    let (result_matrix_values: Fix64x61*) = multiply_selection_by_scalar(matrix, matrix_size - 1, selection, selection_size, scalar)
    let (value_is_in_array: felt) = is_in_array(selection, selection_size, matrix_size - 1)
    let current_fix_element: Fix64x61 =  [matrix.values + matrix_size - 1]
    let (resulting_value: Fix64x61) = conditional_mul(current_fix_element, scalar, value_is_in_array)
    assert [result_matrix_values + matrix_size - 1] = resulting_value
    return(result_matrix_values = result_matrix_values)
end

# This func returns the multiplication of both terms in case condition = 1, and returns primary_term otherwise
func conditional_mul {range_check_ptr}(primary_term: Fix64x61, secondary_term: Fix64x61, condition: felt) -> (result: Fix64x61):
    if condition == 1:
        let (result: Fix64x61) = Small_Math_mul(primary_term, secondary_term)
        return(result = result)
    end
    return(result = primary_term)
end

# divides selection by scalar
func divide_selection_by_scalar {range_check_ptr}(
        matrix: Matrix_2d,
        matrix_size: felt,
        selection: felt*,
        selection_size: felt,
        scalar: Fix64x61
        ) -> (
        result_matrix_values: Fix64x61*
        ):
    alloc_locals
    if matrix_size == 0:
        let (result_matrix_values: Fix64x61*) = alloc()
        return(result_matrix_values = result_matrix_values)
    end
    let (result_matrix_values: Fix64x61*) = divide_selection_by_scalar(matrix, matrix_size - 1, selection, selection_size, scalar)
    let (value_is_in_array: felt) = is_in_array(selection, selection_size, matrix_size - 1)
    let current_fix_element: Fix64x61 =  [matrix.values + matrix_size - 1]
    let (resulting_value: Fix64x61) = conditional_div(current_fix_element, scalar, value_is_in_array)
    assert [result_matrix_values + matrix_size - 1] = resulting_value
    return(result_matrix_values = result_matrix_values)
end

# This func returns the division in case condition = 1, and returns primary_term otherwise
func conditional_div {range_check_ptr}(primary_term: Fix64x61, secondary_term: Fix64x61, condition: felt) -> (result: Fix64x61):
    if condition == 1:
        let (result: Fix64x61) = Small_Math_div(primary_term, secondary_term)
        return(result = result)
    end
    return(result = primary_term)
end

# Returns 1 if it's in the array, 0 otherwise
func is_in_array (array: felt*, array_size: felt, value: felt) -> (result: felt):
    if array_size == 0:
        return(result = 0)
    end
    let (result: felt) = is_in_array(array, array_size - 1, value) 
    tempvar current_array_value = [array + array_size - 1]
    if current_array_value == value:
        return(result = 1)
    end
    return (result = result)
end

######################################################################
############################## FIX UTILS #############################
######################################################################

func concat_fix_array(
        fix_array_1: Fix64x61*, fix_array_1_size: felt, 
        fix_array_2: Fix64x61*, fix_array_2_size: felt
        ):
    alloc_locals
    if fix_array_2_size == 0:
        let (res: Fix64x61*) = alloc()
        return()
    end
    concat_fix_array(fix_array_1, fix_array_1_size, fix_array_2, fix_array_2_size - 1)
    assert[fix_array_1 + fix_array_1_size + fix_array_2_size - 1] = [fix_array_2 + fix_array_2_size - 1]
    return()
end

# returns an array using values in a given array, selected according to new_indexes
# new_indexes must contain values < fix_array_size
func pick_fix_array_values(
        fix_array: Fix64x61*, fix_array_size: felt, 
        new_indexes: felt*, new_indexes_size: felt
        ) -> (
        res: Fix64x61*
        ):
    alloc_locals
    if new_indexes_size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (res: Fix64x61*) = pick_fix_array_values(fix_array, fix_array_size, new_indexes, new_indexes_size - 1)
    local fix_array_index = [new_indexes + new_indexes_size - 1]
    assert [res + new_indexes_size - 1] = [fix_array + fix_array_index]
    return(res = res)
end

func div_fix_array_by_scalar{range_check_ptr}(fix_array: Fix64x61*, fix_array_size: felt, scalar: Fix64x61) -> (res: Fix64x61*):
    alloc_locals
    if fix_array_size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (local res: Fix64x61*) = div_fix_array_by_scalar(fix_array, fix_array_size - 1, scalar)
    local old_val: Fix64x61 = [fix_array + fix_array_size - 1]
    let (local new_val: Fix64x61) = Small_Math_div(old_val, scalar)
    assert[res + fix_array_size - 1] = new_val
    return(res = res)
end

func mul_fix_array_by_scalar{range_check_ptr}(fix_array: Fix64x61*, fix_array_size: felt, scalar: Fix64x61) -> (res: Fix64x61*):
    alloc_locals
    if fix_array_size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (local res: Fix64x61*) = mul_fix_array_by_scalar(fix_array, fix_array_size - 1, scalar)
    local old_val: Fix64x61 = [fix_array + fix_array_size - 1]
    let (local new_val: Fix64x61) = Small_Math_mul(old_val, scalar)
    assert[res + fix_array_size - 1] = new_val
    return(res = res)
end

func add_fix_array{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_1_size: felt, 
        fix_array_2: Fix64x61*, fix_array_2_size: felt
        ) -> (
        res: Fix64x61*
        ):
    alloc_locals
    if fix_array_1_size != fix_array_2_size:
        return(res = fix_array_1)
    end
    let (local res: Fix64x61*) = add_fix_array_inner(fix_array_1, fix_array_2, fix_array_1_size)
    return(res = res)
end

func add_fix_array_inner{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_2: Fix64x61*, size: felt) -> (res: Fix64x61*):
    alloc_locals
    if size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (local res: Fix64x61*) = add_fix_array_inner(fix_array_1, fix_array_2, size - 1)
    local term_1: Fix64x61 = [fix_array_1 + size - 1]
    local term_2: Fix64x61 = [fix_array_2 + size - 1]
    let (local res_val: Fix64x61) = Small_Math_add(term_1, term_2)
    assert [res + size - 1] = res_val
    return(res = res)
end

func sub_fix_array{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_1_size: felt, 
        fix_array_2: Fix64x61*, fix_array_2_size: felt
        ) -> (
        res: Fix64x61*
        ):
    alloc_locals
    if fix_array_1_size != fix_array_2_size:
        return(res = fix_array_1)
    end
    let (local res: Fix64x61*) = sub_fix_array_inner(fix_array_1, fix_array_2, fix_array_1_size)
    return(res = res)
end

func sub_fix_array_inner{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_2: Fix64x61*, size: felt) -> (res: Fix64x61*):
    alloc_locals
    if size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (local res: Fix64x61*) = sub_fix_array_inner(fix_array_1, fix_array_2, size - 1)
    local term_1: Fix64x61 = [fix_array_1 + size - 1]
    local term_2: Fix64x61 = [fix_array_2 + size - 1]
    let (local res_val: Fix64x61) = Small_Math_sub(term_1, term_2)
    assert [res + size - 1] = res_val
    return(res = res)
end

func mul_fix_array{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_1_size: felt, 
        fix_array_2: Fix64x61*, fix_array_2_size: felt
        ) -> (
        res: Fix64x61*
        ):
    alloc_locals
    if fix_array_1_size != fix_array_2_size:
        return(res = fix_array_1)
    end
    let (local res: Fix64x61*) = mul_fix_array_inner(fix_array_1, fix_array_2, fix_array_1_size)
    return(res = res)
end

func mul_fix_array_inner{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_2: Fix64x61*, size: felt) -> (res: Fix64x61*):
    alloc_locals
    if size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (local res: Fix64x61*) = mul_fix_array_inner(fix_array_1, fix_array_2, size - 1)
    local term_1: Fix64x61 = [fix_array_1 + size - 1]
    local term_2: Fix64x61 = [fix_array_2 + size - 1]
    let (local res_val: Fix64x61) = Small_Math_mul(term_1, term_2)
    assert [res + size - 1] = res_val
    return(res = res)
end

func div_fix_array{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_1_size: felt, 
        fix_array_2: Fix64x61*, fix_array_2_size: felt
        ) -> (
        res: Fix64x61*
        ):
    alloc_locals
    if fix_array_1_size != fix_array_2_size:
        return(res = fix_array_1)
    end
    let (local res: Fix64x61*) = div_fix_array_inner(fix_array_1, fix_array_2, fix_array_1_size)
    return(res = res)
end

func div_fix_array_inner{range_check_ptr}(fix_array_1: Fix64x61*, fix_array_2: Fix64x61*, size: felt) -> (res: Fix64x61*):
    alloc_locals
    if size == 0:
        let (res: Fix64x61*) = alloc()
        return(res = res)
    end
    let (local res: Fix64x61*) = div_fix_array_inner(fix_array_1, fix_array_2, size - 1)
    local term_1: Fix64x61 = [fix_array_1 + size - 1]
    local term_2: Fix64x61 = [fix_array_2 + size - 1]
    let (local res_val: Fix64x61) = Small_Math_div(term_1, term_2)
    assert [res + size - 1] = res_val
    return(res = res)
end

