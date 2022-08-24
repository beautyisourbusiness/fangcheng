from starkware.cairo.common.alloc import alloc
from matrix_data_types import Matrix_2d

from small_math import ( 
        Small_Math_div,
        Small_Math_mul,
        show_Fix,
        Fix64x61
)

func show_fix_array (array: Fix64x61*, size: felt):
    alloc_locals
    let (local values: felt*) = strip_type(array, size)
    %{
        values = ids.values
        size = ids.size
        text = '--------------------\n['
        for col in range(size):
            a = memory[values + col]
            if a > 1809251394333065606848661391547535052811553607665798349986546028067936010240:
                a = a - 3618502788666131213697322783095070105623107215331596699973092056135872020481
            text = text + '\t' + str(round((a/2**61)*10000)/10000)
        print(text+'\t]')
    %}
    tempvar useless = 0
    return()
end

func show_array (array: felt*, size: felt):
    if size == 0:
        return()
    end
    show_array (array, size - 1)
    tempvar value = [array + size - 1]
    %{
        print(ids.size, ':', ids.value)
    %}
    return()
end

func show_matrix (matrix : Matrix_2d, precision: felt):
    alloc_locals
    let (local values: felt*) = strip_type(matrix.values, matrix.size)
    local matrix_size = matrix.size
    local matrix_row_size = matrix.row_size
    %{
        values = ids.values
        matrix_size = ids.matrix_size
        matrix_row_size = ids.matrix_row_size
        matrix_col_size = int(matrix_size / matrix_row_size)
        element_count = 0
        print('--------------------\n')
        for row in range(matrix_col_size):
            text = '['
            for col in range(matrix_row_size):
                a = memory[values + row * matrix_row_size + col]
                if a > 1809251394333065606848661391547535052811553607665798349986546028067936010240:
                    a = a - 3618502788666131213697322783095070105623107215331596699973092056135872020481
                text = text + '\t' + str(round((a/2**61)*10**ids.precision)/10**ids.precision)
            print(text+'\t]')
    %}

    local closer
    return()
end

func strip_type (matrix_values: Fix64x61*, size: felt) -> (res: felt*):
    alloc_locals
    if size == 0:
        let (local res: felt*) = alloc()        
        return (res = res)
    end
    let (res: felt*) = strip_type(matrix_values, size - 1)
    assert[res + (size - 1)] = [matrix_values + (size - 1) * Fix64x61.SIZE].val
    return(res = res)
end

func wrap_type(matrix_values: felt*, size: felt) -> (res: Fix64x61*):
    alloc_locals
    if size == 0:
        let (local res: Fix64x61*) = alloc()        
        return (res = res)
    end
    let (res: Fix64x61*) = wrap_type(matrix_values, size - 1)
    local felt_elem = [matrix_values + (size - 1)]
    local fix_elem : Fix64x61 = Fix64x61(felt_elem)
    assert[res + (size - 1)] = fix_elem
    return(res = res)
end

func get_matrix_from_json_int() -> (res: Matrix_2d):
    alloc_locals
    local size: felt
    local row_size: felt
    local col_size: felt
    local val_list: felt*
    %{
        matrix_size = program_input['matrix_size']
        matrix_row_size = program_input['matrix_row_size']
        matrix_col_size = program_input['matrix_col_size']
        values = program_input['matrix_values']

        ids.size = matrix_size
        ids.row_size = matrix_row_size
        ids.col_size = matrix_col_size
        ids.val_list = val_list = segments.add()

        prime = 2**251 + 17 * 2**192 + 1

        for i, val in enumerate(values):
            new_val = val * 2**61
            if new_val < 0:
                new_val = prime + new_val
            memory[val_list + i] = new_val
    %}
    let (local values: Fix64x61*) = wrap_type(val_list, size)
    local res: Matrix_2d = Matrix_2d(size, row_size, col_size, values)
    return(res = res)
end

func get_matrix_from_json_fix{range_check_ptr}(decimals: felt) -> (res: Matrix_2d):
    alloc_locals
    local size: felt
    local row_size: felt
    local col_size: felt
    local val_list: felt*
    %{
        matrix_size = program_input['matrix_size']
        matrix_row_size = program_input['matrix_row_size']
        matrix_col_size = program_input['matrix_col_size']
        values = program_input['matrix_values']

        ids.size = matrix_size
        ids.row_size = matrix_row_size
        ids.col_size = matrix_col_size
        ids.val_list = val_list = segments.add()

        prime = 2**251 + 17 * 2**192 + 1

        for i, val in enumerate(values):
            new_val = int((val * 2**61) / (10**ids.decimals))
            if new_val < 0:
                new_val = prime + new_val
            memory[val_list + i] = new_val
    %}
    let (local values: Fix64x61*) = wrap_type(val_list, size)
    local res: Matrix_2d = Matrix_2d(size, row_size, col_size, values)
    return(res = res)
end
