import random
import os

my_board = [
    [True,True,True,True],
    [False,True,False,False],
    [True,True,True,False],
    [True,False,True,False],
]

#check if unlocked
def check_lock(board):
    for row in board:
        for col in row:
            if col == False:
                return False    
    return True

def shift_row_left(board,r_num):
    board[r_num].append(board[r_num][0])
    board[r_num].pop(0)
    return board

def shift_row_right(board,r_num):
    board[r_num].insert(0,board[r_num][-1])
    board[r_num].pop(-1)
    return board
def print_board(board):
    for row in board:
        print(row)

def shift_col_up(board,col):
    temp = board[0][col]
    board[0][col] = board[1][col]
    board[1][col] = board[2][col]
    board[2][col] = board[3][col]
    board[3][col] = temp
    return board

def shift_col_down(board,col):
    temp = board[3][col]
    board[3][col] = board[2][col]
    board[2][col] = board[1][col]
    board[1][col] = board[0][col]
    board[0][col] = temp
    return board

def invert(board):
    inverted = []
    new_row=[]
    for row in board:
        for col in row:
            if col == True:
                col = False
            else:
                col = True
            new_row.append(col)
        inverted.append(new_row)
        new_row = []
    return inverted


def solver(board):
    num = random.randint(0,3)
    func_list = [shift_row_left(board,num),shift_row_right(board,num),shift_col_up(board,num),shift_col_down(board,num)]
    for i in range(100):
        os.system('cls')
        if check_lock(board) == False:            
            
            
            print_board(random.choice(func_list))
        else: 
            print_board(board)
        
    


#print(check_lock(my_board))
#shift_row_left(my_board,1)
#shift_row_right(my_board,2)
#print(my_board)
#shift_col_up(my_board,2)
#shift_col_down(my_board,0)
print_board(my_board)
#my_board = invert(my_board)
#print_board(my_board)

solver(my_board)
