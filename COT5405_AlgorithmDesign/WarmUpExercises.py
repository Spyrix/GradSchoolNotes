# Given an array A[1,n], find the smallest and largest number in A, minimizing
# comparisions

def my_min_max_finder(a):
    if(len(a) == 0):
        print("Error, array is not populated")
        return
    min = a[0]
    max = a[0]
    comparisons = 0
    for i in range(1, len(a)):
        if(a[i] > max):
            max = a[i]
            comparisons += 1
        elif(a[i] < min):
            min = a[i]
            comparisons += 1
    print(f'max: {max}, min: {min}, comparisons: {comparisons}')

# Find the max and second min_max_a1


# def max_max2_finder(a):


#min_max_a1 = [7, 3, 5, 9, 2, 4]
# my_min_max_finder(min_max_a1)


#max_max2_a = [5, 1, 3, 7, 8, 9, 4, 2]
# max_max2_finder(max_max2_a)

coin_problem(a):
