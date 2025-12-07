from Z.Integer import Integer

def TRANS_INT_Z(I:int):
    arr_numbers = [int(char) for char in str(I) if char.isdigit()]
    if I<0:
        return Integer(1,len(arr_numbers)-1, arr_numbers)
    return Integer(0,len(arr_numbers)-1, arr_numbers)