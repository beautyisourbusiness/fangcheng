%builtins range_check

from starkware.cairo.common.alloc import alloc

from small_math import (
        Fix64x61,
        Double,
        Double_to_Fix
)

from matrix_utils import (
        get_indexes_for_swapping_rows,
        get_indexes_for_swapping_cols,
        multiply_selection_by_scalar,
        multiply_matrix_by_scalar,
        add_scalar_to_matrix,
        get_selected_values,
        check_row_index,
        check_col_index,
        get_indexes_for_row,
        get_indexes_for_col,
        get_values_for_row,
        get_values_for_col,
        get_first_non_zero_in_col,
        divide_row_by_value_in_index,
        concat_fix_array,
        add_row_to_matrix,
        get_indexes_two_elem_permutation,
        swap_rows,
        pick_fix_array_values,
        div_fix_array_by_scalar,
        mul_fix_array_by_scalar,
        substract_selection_by_scalar,
        add_fix_array,
        sub_fix_array,
        replace_row,
        get_triangular_normal_matrix
)
from matrix_data_types import Matrix_2d
from matrix_console_utils import (
        show_matrix,
        show_fix_array,
        show_array,
        get_matrix_from_json_int,
        get_matrix_from_json_fix
)

func main{range_check_ptr}():
    alloc_locals
    #also you can use below get_matrix_from _json_fix(2)--->i.e 12345 in json will input --->123.45 as value
    let (local matrix_a: Matrix_2d) = get_matrix_from_json_int()
    show_matrix(matrix_a, 4)

    let (local triangular_matrix_a: Matrix_2d) = get_triangular_normal_matrix(matrix_a)
    #show solutions with nine decimals -could use more, only for showing-
    show_matrix(triangular_matrix_a, 9)

    return()
end 