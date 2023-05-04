import random
import matplotlib.pyplot as plt
from collections import Counter
import sys
import copy

class Bat:
    def __init__(self, freq, loudness, emission_rate, matrix):
        self.freq = freq
        self.loudness = loudness
        self.emission_rate = emission_rate
        self.matrix = matrix


def print_matrix(matrix):
    for row in matrix:
        print(row)

def small_walk(matrix):
    n = len(matrix)
    m = len(matrix[0])
    
    # Select two random points
    i1, j1 = random.randint(0, n-1), random.randint(0, m-1)
    i2, j2 = random.randint(0, n-1), random.randint(0, m-1)
    # print("swap")
    # print(i1,j1)
    # print(i2,j2)
    e = matrix[i1][j1]
    r = matrix[i2][j2]
    # print(e,r)
    if matrix[i2][j2] in matrix[i1]:
        x_idx = matrix[i1].index(matrix[i2][j2])
        matrix[i1][x_idx] = matrix[i1][j1]
        matrix[i1][j1] = r
    if matrix[i1][j1] in matrix[i2]:
        x_idx = matrix[i2].index(matrix[i1][j1])
        matrix[i2][x_idx] = matrix[i2][j2]
        matrix[i2][j2] = e
    
    return matrix


def longest_gap_machine(matrix):
    begin = [[0 for j in range(k)] for i in range(k)]
    end = [[0 for j in range(k)] for i in range(k)]
    MET = [0 for j in range(k)]
    PET = [0 for j in range(k)]
    start,end,answer = Calculate_fitness(job_info,matrix,k,MET,PET,begin,end)
    # print(answer)
    max_gap = -1
    max_gap_machine = -1
    
    for i in range(len(start)):
        last_end_time = start[i][0]
        for j in range(1, len(start[i])):
            gap = start[i][j] - last_end_time
            if gap > max_gap:
                max_gap = gap
                max_gap_machine = i
            last_end_time = end[i][j]
    matrix[max_gap_machine][:] = matrix[max_gap_machine][len(matrix)-1:] + matrix[max_gap_machine][:len(matrix)-1]
    # print(max_gap_machine,max_gap)
    return matrix

def Shift_Up(matrix):
    num_cols = len(matrix[0])
    j = random.randint(0, num_cols-1)
    # print(j,"up\n")
    rotated_col = [matrix[i][j] for i in range(len(matrix))]
    # print(rotated_col)
    x = rotated_col[0]
    rotated_col = rotated_col[1:] 
    rotated_col.append(x)
    # print(rotated_col)

    for i in range(len(matrix)):
        if rotated_col[i] in matrix[i]:
            x_idx = matrix[i].index(rotated_col[i])
            matrix[i][x_idx] = matrix[i][j]
            matrix[i][j] = rotated_col[i]
    return matrix

def Shift_Down(matrix):
    num_cols = len(matrix[0])
    j = random.randint(0, num_cols-1)
    # print(j,"down\n")
    rotated_col = [matrix[i][j] for i in range(len(matrix))]
    # print(rotated_col)
    rotated_col[:] = rotated_col[len(matrix)-1:] + rotated_col[:len(matrix)-1]
    # print(rotated_col)

    for i in range(len(matrix)):
        if rotated_col[i] in matrix[i]:
            x_idx = matrix[i].index(rotated_col[i])
            matrix[i][x_idx] = matrix[i][j]
            matrix[i][j] = rotated_col[i]
    return matrix

def max_freq_dict(d):
    max_freq = 0
    max_elem = None
    for elem, freq in d.items():
        if freq > max_freq:
            max_freq = freq
            max_elem = elem
    return max_elem

def plot_gantt(begin, end,schedule):
    fig, ax = plt.subplots()

    # Set y-axis limits and ticks
    ax.set_ylim(0, len(begin))
    ax.set_yticks(range(len(begin)))
    ax.set_yticklabels([f'machine {i+1}' for i in range(len(begin))])

    # Set x-axis limits
    min_time = min([min(row) for row in begin])
    max_time = max([max(row) for row in end])
    ax.set_xlim(min_time, max_time)

    # Plot bars for each job
    for i in range(len(begin)):
        for j in range(len(begin[i])):
            start = begin[i][j]
            end_time = end[i][j]
            duration = end_time - start
            job = schedule[i][j]
            colors = ['blue', 'green', 'red', 'cyan','orange', 'purple', 'brown', 'pink', 'gray','blue']
            ax.broken_barh([(start, duration)], (i, 0.8), facecolors=(f'tab:{colors[job]}'))

    # Set chart title and axis labels
    ax.set_title('Gantt Chart')
    ax.set_xlabel('Time')
    plt.show()

def ColReuse(matrix):
    max_vals = []
    for j in range(len(matrix[0])):
        col = [matrix[i][j] for i in range(len(matrix))]
        count = Counter(col)
        max_freq = max(count.values())
        max_vals.append(max_freq)
    # print(max_vals)
    return max(max_vals)


def Calculate_fitness(job_info,schedule,k,MET,PET,begin,end):
    # ith elelemt being processed
    for i in range(k):
        #j is machine
        for j in range(k):
            job = schedule[j][i]
            duration = job_info[j][job]
            start = max(MET[j],PET[job])
            finish = start + duration
            begin[j][i] = start
            end[j][i] = finish
            MET[j]=PET[job] = finish
    makespan = 0
    for i in range(k):
        if(makespan<end[i][k-1]):
            makespan = end[i][k-1]
    return begin,end,makespan

def substitution(matrix):
    # Find the column with the most repetitive element
    max_reps = 0
    max_col = 0
    col_max_ele = []
    for col in range(len(matrix[0])):
        count = {}
        for row in range(len(matrix)):
            if matrix[row][col] in count:
                count[matrix[row][col]] += 1
            else:
                count[matrix[row][col]] = 1
        max_count = max(count.values())
        max_freq_element = max_freq_dict(count)
        col_max_ele.append(max_freq_element)
        if (max_count > max_reps) or (max_count == max_reps and random.random()>0.5):
            max_reps = max_count
            max_col = col

    # Replace the rows containing the repetitive element in the max column
    rep_elem = 0
    for row in range(len(matrix)):
        if matrix[row][max_col] == col_max_ele[max_col]:
            # print("row:",row," col:",max_col)
            rep_elem = matrix[row][max_col]
            matrix[row] = list(range(len(matrix[row])))
            matrix[row] = random.sample(matrix[row], len(matrix[row]))  # get a random permutation of the numbers
    return matrix

def Full_Reverse(matrix):
    """
    Takes a matrix as input and returns the matrix with the order of the
    elements in each row reversed.
    """
    # Get the number of rows and columns in the matrix
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    
    # Reverse the order of the elements in each row
    for i in range(num_rows):
        matrix[i] = matrix[i][::-1]
    
    # Return the modified matrix
    return matrix


def Generate_random_schedule(k):
    n = k*k  # size of the permutation
    numbers = list(range(0, n))  # create a list of numbers from 0 to n-1
    perm = random.sample(numbers, n)  # get a random permutation of the numbers
    # print(perm)  # print the permutation
    schedule = [[] for j in range(k)]
    for i in perm:
        schedule[int(i/k)].append(i%k) 
    # print_matrix(schedule)
    return schedule

import random

def Join(dest, source, num_rows=1):
    """
    Replaces a random selection of num_rows rows in dest with the
    corresponding rows from source.
    """
    dest_copy = dest.copy()  # Create a copy of dest to avoid modifying the original
    dest_rows = list(range(len(dest_copy)))
    random.shuffle(dest_rows)
    for i in range(num_rows):
        dest_copy[dest_rows[i]] = source[i]
    return dest_copy

def Best_bat(my_bats,Best_Answer,index,Best_solution):
    i = 0
    for bat in my_bats:
        begin = [[0 for j in range(k)] for i in range(k)]
        end = [[0 for j in range(k)] for i in range(k)]
        MET = [0 for j in range(k)]
        PET = [0 for j in range(k)]
        begin,end,answer = Calculate_fitness(job_info,bat.matrix,k,MET,PET,begin,end)
        if answer < Best_Answer:
            Best_solution = bat
            Best_Answer = answer
            index = i
        i = i + 1
    return Best_Answer,Best_solution,index


def Bat_Algorithm(fmin,fmax,alpha,initial_loudness,no_of_bats,MAX_ITERATIONS,curval):
    my_bats = []
    for i in range(no_of_bats):
        my_bats.append(Bat(freq=0, loudness=initial_loudness, emission_rate=1, matrix=Generate_random_schedule(k)))
    Best_solution = []
    Best_Answer = 999999999
    index = -1
    Best_Answer,Best_solution,index = Best_bat(my_bats,Best_Answer,index,Best_solution)

    for i in range(MAX_ITERATIONS):
        j = 0
        curval.append(Best_Answer)
        for bat in my_bats:
            bat.freq = random.randint(fmin, fmax)
            sel = random.randint(0,4)
            new_matrix = []
            if(sel==0):
                temp = copy.deepcopy(bat.matrix)
                new_matrix = Full_Reverse(temp)
            elif(sel==1):
                temp = copy.deepcopy(bat.matrix)
                u = random.randint(0,no_of_bats-1)
                new_matrix = Join(temp,my_bats[u].matrix)
            elif(sel==2):
                temp = copy.deepcopy(bat.matrix)
                new_matrix = Shift_Up(temp)
            elif(sel==3):
                temp = copy.deepcopy(bat.matrix)
                new_matrix = Shift_Down(temp)
            elif(sel==4):
                temp = copy.deepcopy(bat.matrix)
                new_matrix = substitution(temp)
            probability = random.random()
            if(probability>bat.emission_rate):
                new_matrix = small_walk(new_matrix)
                new_matrix = longest_gap_machine(new_matrix)
            probability = random.random()
            if(probability<bat.loudness):
                begin = [[0 for j in range(k)] for i in range(k)]
                end = [[0 for j in range(k)] for i in range(k)]
                MET = [0 for j in range(k)]
                PET = [0 for j in range(k)]
                begin,end,answer1 = Calculate_fitness(job_info,bat.matrix,k,MET,PET,begin,end)
                begin = [[0 for j in range(k)] for i in range(k)]
                end = [[0 for j in range(k)] for i in range(k)]
                MET = [0 for j in range(k)]
                PET = [0 for j in range(k)]
                begin,end,answer2 = Calculate_fitness(job_info,new_matrix,k,MET,PET,begin,end)
                if(answer2<answer1):
                    bat.matrix = copy.deepcopy(new_matrix)
                    if(answer2<Best_Answer):
                        Best_solution = copy.deepcopy(bat.matrix)
                        index = j
                        Best_Answer = answer2
            bat.emission_rate = 1 - (1/(MAX_ITERATIONS + 1 - j))
            # print("do something")
            # print(j)
            j = j + 1
        print(i)
    return Best_solution,Best_Answer
file = open('input_7_1.txt', 'r')
lines = file.readlines()
file.close()

job_info1 = []

for line in lines:
    row = line.strip().split(' ')
    for i in range(len(row)):
        row[i] = int(row[i])
    job_info1.append(row)


print(job_info1)
job_info = [list(row) for row in zip(*job_info1)]
print(job_info)
k = len(job_info)
# begin = [[0 for j in range(k)] for i in range(k)]
# end = [[0 for j in range(k)] for i in range(k)]
# MET = [0 for j in range(k)]
# PET = [0 for j in range(k)]
# schedule = Generate_random_schedule(k)
# begin,end,answer = Calculate_fitness(job_info,schedule,k,MET,PET,begin,end)
# print(answer)

# print(begin,end,answer)
# print(ColReuse(schedule))
# plot_gantt(begin, end,schedule)
# # schedule = substitution(schedule)
# print("after sustitution")
# # print_matrix(schedule)
# print("dfghj")
# # schedule = Full_Reverse(schedule)
# schedule = small_walk(schedule)
# print_matrix(schedule)
# print(ColReuse(schedule))
# begin = [[0 for j in range(k)] for i in range(k)]
# end = [[0 for j in range(k)] for i in range(k)]
# MET = [0 for j in range(k)]
# PET = [0 for j in range(k)]
# begin,end,answer = Calculate_fitness(job_info,schedule,k,MET,PET,begin,end)
# print(begin,end,answer)

# Example usage
curval = []
schedule,answer = Bat_Algorithm(0,150,0.52,0.95,100,2500,curval)
begin = [[0 for j in range(k)] for i in range(k)]
end = [[0 for j in range(k)] for i in range(k)]
MET = [0 for j in range(k)]
PET = [0 for j in range(k)]
# schedule = Generate_random_schedule(k)
begin,end,answer = Calculate_fitness(job_info,schedule,k,MET,PET,begin,end)
print("here")
print_matrix(job_info)
print(job_info1)
print_matrix(schedule)
print(begin,end,answer)
# print_matrix(schedule)
plot_gantt(begin, end,schedule)
y = list(range(len(curval)))
plt.plot(y, curval)
plt.show()

