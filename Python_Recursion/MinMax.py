


#-----------------------------MINMAX--------------------------------------
'''
MINIMAX
Games as Trees
Have you ever played a game against someone and felt like they were always two steps ahead? No matter what clever move you tried, they had somehow envisioned it and had the perfect counterattack. This concept of thinking ahead is the central idea behind the minimax algorithm.

The minimax algorithm is a decision-making algorithm that is used for finding the best move in a two player game. It’s a recursive algorithm — it calls itself. In order for us to determine if making move A is a good idea, we need to think about what our opponent would do if we made that move.

We’d guess what our opponent would do by running the minimax algorithm from our opponent’s point of view. In the hypothetical world where we made move A, what would they do? Surely they want to win as badly as we do, so they’d evaluate the strength of their move by thinking about what we would do if they made move B.

As this process repeats, we can start to make a tree of these hypothetical game states. We’ll eventually reach a point where the game is over — we’ll reach a leaf of the tree. Either we won, our opponent won, or it was a tie. At this point, the recursion can stop. Because the game is over, we no longer need to think about how our opponent would react if we reached this point of the game.

Instructions
On this page, you’ll see the game tree of a Tic-Tac-Toe game that is almost complete. At the root of the node, it is "X"‘s turn.

Some of the leaves of the tree still have squares that can be filled in. Why are those boards leaves?

MINIMAX
Tic-Tac-Toe
For the rest of this exercise, we’re going to be writing the minimax algorithm to be used on a game of Tic-Tac-Toe. We’ve imported a Tic-Tac-Toe game engine in the file tic_tac_toe.py. Before starting to write the minimax function, let’s play around with some of the Tic-Tac-Toe functions we’ve defined for you in tic_tac_toe.py.

To begin, a board is represented as a list of lists. In script.py we’ve created a board named my_board where the X player has already made the first move. They’ve chosen the top right corner. To nicely print this board, use the print_board() function using my_board as a parameter.

Next, we want to be able to take a turn. The select_space() function lets us do this. Select space takes three parameters:

The board that you want to take the turn on.
The space that you want to fill in. This should be a number between 1 and 9.
The symbol that you want to put in that space. This should be a string — either an "X" or an "O".
We can also get a list of the available spaces using available_moves() and passing the board as a parameter.

Finally, we can check to see if someone has won the game. The has_won() function takes the board and a symbol (either "X" or "O"). It returns True if that symbol has won the game, and False otherwise.

Let’s test these functions! Write your code in script.py, but feel free to take a look at tic_tac_toe.py if you want to look at how the game engine works.'''

#------------script.py---------------
from tic_tac_toe import *

my_board = [
	["1", "2", "X"],
	["4", "5", "6"],
	["7", "8", "9"]
]

print_board(my_board)

select_space(my_board, 4, "O")
select_space(my_board, 5, "X")
select_space(my_board, 6, "O")
select_space(my_board, 7, "X")

print_board(my_board)

print(has_won(my_board, "X"))
print(has_won(my_board, "O"))

#---------tic_tac_toe.py---------------

def print_board(board):
    print("|-------------|")
    print("| Tic Tac Toe |")
    print("|-------------|")
    print("|             |")
    print("|    " + board[0][0] + " " + board[0][1] + " " + board[0][2] + "    |")
    print("|    " + board[1][0] + " " + board[1][1] + " " + board[1][2] + "    |")
    print("|    " + board[2][0] + " " + board[2][1] + " " + board[2][2] + "    |")
    print("|             |")
    print("|-------------|")
    print()


def select_space(board, move, turn):
    if move not in range(1,10):
        return False
    row = int((move-1)/3)
    col = (move-1)%3
    if board[row][col] != "X" and board[row][col] != "O":
        board[row][col] = turn
        return True
    else:
        return False

def available_moves(board):
    moves = []
    for row in board:
        for col in row:
            if col != "X" and col != "O":
                moves.append(int(col))
    return moves

def has_won(board, player):
    for row in board:
        if row.count(player) == 3:
            return True
    for i in range(3):
        if board[0][i] == player and board[1][i] == player and board[2][i] == player:
            return True
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False
    
'''

MINIMAX
Detecting Tic-Tac-Toe Leaves
An essential step in the minimax function is evaluating the strength of a leaf. If the game gets to a certain leaf, we want to know if that was a better outcome for player "X" or for player "O".

Here’s one potential evaluation function: a leaf where player "X" wins evaluates to a 1, a leaf where player "O" wins evaluates to a -1, and a leaf that is a tie evaluates to 0.

Let’s write this evaluation function for our game of Tic-Tac-Toe.

First, we need to detect whether a board is a leaf — we need know if the game is over. A game of Tic-Tac-Toe is over if either player has won, or if there are no more open spaces. We can write a function that uses has_won() and available_moves() to check to see if the game is over.

If the game is over, we now want to evaluate the state of the board. If "X" won, the board should have a value of 1. If "O" won, the board should have a value of -1. If neither player won, it was a tie, and the board should have a value of 0.

Instructions
1.
At the bottom of script.py, create a function called game_is_over() that takes a board as a parameter. The function should return True if the game is over and False otherwise.


Return True if has_won(board, "X") or has_won(board, "O") or if the length of available_moves(board) is 0.

2.
We’ve given you four different boards to test your function. Call game_is_over() on the boards start_board, x_won, o_won, and tie. Print the result of each.


One of these function calls should look like this:
'''
print(game_is_over(start_board))

'''
Your results from these four print statements should be False, True, True, True.

3.
Let’s write another function called evaluate_board() that takes board as a parameter. This function will only ever be called if we’ve detected the game is over. The function should return a 1 if "X" won, a -1 if "O" won, and a 0 otherwise.


Inside this function return 1 if has_won(board, "X") is True. Make a similar function call to see if "O" has won. If neither player has won, it’s a tie so return 0.

4.
Test your function on the four different boards! For each board, write an if statement checking if the game is over. If it is, evaluate the board and print the result. You just wrote the base case of the minimax algorithm!


Your code for one of the boards should look like this:
'''
if game_is_over(tie):
  print(evaluate_board(tie))

#--------script.py---------------
from tic_tac_toe import *

start_board = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

x_won = [
	["X", "O", "3"],
	["4", "X", "O"],
	["7", "8", "X"]
]

o_won = [
	["O", "X", "3"],
	["O", "X", "X"],
	["O", "8", "9"]
]

tie = [
	["X", "X", "O"],
	["O", "O", "X"],
	["X", "O", "X"]
]

test1 = [
	["1", "2", "O"],
	["O", "5", "X"],
	["X", "O", "X"]
]

test2 = [
	["X", "2", "O"],
	["O", "5", "X"],
	["X", "8", "X"]
]

test3 = [
	["1", "X", "O"],
	["3", "O", "X"],
	["7", "O", "X"]
]

test4 = [
	["X", "X", "3"],
	["O", "5", "6"],
	["X", "8", "9"]
]


def game_is_over(board):
  if not available_moves(board):
    return True
  if has_won(board, "O") or has_won(board, "X"):
    return True
  return False

print(game_is_over(start_board))
print(game_is_over(x_won))
print(game_is_over(o_won))
print(game_is_over(tie))

def evaluate_board(board):
  if game_is_over(board):
    if has_won(board, "O"):
      return -1
    elif has_won(board, "X"):
      return 1
  else:
    return 0
  
print(evaluate_board(start_board))
print(evaluate_board(x_won))
print(evaluate_board(o_won))
print(evaluate_board(tie))

#---------------tic_tac_toe.py---------------

def print_board(board):
    print("|-------------|")
    print("| Tic Tac Toe |")
    print("|-------------|")
    print("|             |")
    print("|    " + board[0][0] + " " + board[0][1] + " " + board[0][2] + "    |")
    print("|    " + board[1][0] + " " + board[1][1] + " " + board[1][2] + "    |")
    print("|    " + board[2][0] + " " + board[2][1] + " " + board[2][2] + "    |")
    print("|             |")
    print("|-------------|")
    print()


def select_space(board, move, turn):
    if move not in range(1,10):
        return False
    row = int((move-1)/3)
    col = (move-1)%3
    if board[row][col] != "X" and board[row][col] != "O":
        board[row][col] = turn
        return True
    else:
        return False

def available_moves(board):
    moves = []
    for row in board:
        for col in row:
            if col != "X" and col != "O":
                moves.append(int(col))
    return moves

def has_won(board, player):
    for row in board:
        if row.count(player) == 3:
            return True
    for i in range(3):
        if board[0][i] == player and board[1][i] == player and board[2][i] == player:
            return True
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False
    
'''MINIMAX
Copying Boards
One of the central ideas behind the minimax algorithm is the idea of exploring future hypothetical board states. Essentially, we’re saying if we were to make this move, what would happen. As a result, as we’re implementing this algorithm in our code, we don’t want to actually make our move on the board. We want to make a copy of the board and make the move on that one.

Let’s look at how copying works in Python. Let’s say we have a board that looks like this
'''
my_board = [
    ["X", "2", "3"],
    ["O", "O", "6"],
    ["X", "8", "9"]
]'''
If we want to create a copy of our board our first instinct might be to do something like this
'''
new_board = my_board'''
This won’t work the way we want it to! Python objects are saved in memory, and variables point to a location in memory. In this case, new_board, and my_board are two variables that point to the same object in memory. If you change a value in one, it will change in the other because they’re both pointing to the same object.

One way to solve this problem is to use the deepcopy() function from Python’s copy library.
'''
new_board = deepcopy(my_board)'''
new_board is now a copy of my_board in a different place in memory. When we change a value in new_board, the values in my_board will stay the same!'''



from tic_tac_toe import *
from copy import deepcopy

my_board = [
	["1", "2", "X"],
	["4", "5", "6"],
	["7", "8", "9"]
]

#copy the board in a wrong way
new_board = my_board

print_board(my_board)

print_board(new_board)

select_space(new_board, 5,"O")

print_board(my_board)

print_board(new_board)

#copy the board in a right way
new_board = deepcopy(my_board)

select_space(new_board, 7,"X")

print_board(my_board)

print_board(new_board)

   
'''
MINIMAX
The Minimax Function
We’re now ready to dive in and write our minimax() function. The result of this function will be the “value” of the best possible move. In other words, if the function returns a 1, that means a move exists that guarantees that "X" will win. If the function returns a -1, that means that there’s nothing that "X" can do to prevent "O" from winning. If the function returns a 0, then the best "X" can do is force a tie (assuming "O" doesn’t make a mistake).

Our minimax() function has two parameters. The first is the game state that we’re interested in finding the best move. When the minimax() function first gets called, this parameter is the current state of the game. We’re asking “what is the best move for the current player right now?”

The second parameter is a boolean named is_maximizing representing whose turn it is. If is_maximizing is True, then we know we’re working with the maximizing player. This means when we’re picking the “best” move from the list of moves, we’ll pick the move with the highest value. If is_maximizing is False, then we’re the minimizing player and want to pick the minimum value.

Let’s start writing our minimax() function.

Instructions
1.
We’ve started the minimax() function for you and included the base case we wrote a few exercises ago.

We now need to define what should happen if the node isn’t a leaf.

We’ll want to set up some variables that are different depending on whether is_maximizing is True or False.

Below the base case, write an if statement to check if is_maximizing is True.

Inside the if statement, create a variable named best_value. Since we’re working with the maximizing player right now, this is the variable that will keep track of the highest possible value from all of the potential moves.

Right now, we haven’t looked at any moves, so we should start best_value at something lower than the lowest possible value — -float("Inf").

Write an else statement. Inside this else statement we’ll be setting up variables for the minimizing player. In this case, best_value should start at float("Inf").

Return best_value after the else statement.


Your if statement should look like this:

if is_maximizing:
  # Set the initial value of best_value for the maximizing player.
else:
  # Set the initial value of best_value for the minimizing player.
2.
We now want to loop through all of the possible moves of input_board before the return statement. We’re looking to find the best possible move.

Remember, you can get all of the possible moves by calling available_moves() using input_board as a parameter.

After the else statement, but before you return best_value, loop through all of the possible moves and print each move.

Let’s call our function to see these print statements. Outside of your function definition, call minimax() using the parameters x_winning (the board we’re using) and True (we’re calling it as the maximizing player).


Your for loop might look something like this:
'''
for move in available_moves(input_board):'''
You’d then want to print(move) inside the for loop.

3.
Delete the print statements for move. Rather than just printing the move in this for loop, let’s create a copy of the game board and make the move!

Inside the for loop, create a deepcopy of input_board named new_board.

Apply the move to new_board by calling the select_space() function. select_space() takes three parameters.

The board you want to use (new_board)
The space you’re selecting (the move from the for loop)
The symbol you’re filling the space in with. This is different depending on whether we’re the maximizing or minimizing player. In your if and else statements, create a variable named symbol. symbol should be "X" if we’re the maximizing player and "O" if we’re the minimizing player. Use symbol as the third parameter of select_space().
To help us check if you accurately made the move, return new_board outside the for loop (instead of returning best_move). We’ll fix that return statement soon!


Inside the loop, make a copy of the board:
'''
new_board = deepcopy(input_board)'''
Finally, make the move on the new board. Make sure to fill in the third parameter with the correct symbol.
'''
select_space(new_board, move, ___)'''
Outside the for loop, return new_board.'''

from tic_tac_toe import *
from copy import deepcopy

x_winning = [
	["X", "2", "O"],
	["4", "O", "6"],
	["7", "8", "X"]
]

def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
  if has_won(board, "X"):
    return 1
  elif has_won(board, "O"):
    return -1
  else:
    return 0

print(available_moves(x_winning))
  
def minimax(input_board, is_maximizing):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board):
    return evaluate_board(input_board)
  
  if is_maximizing:
    best_value = -float('inf')
    symbol = "X"
  else:
    best_value = float('inf')
    symbol = "O"
  
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
      
  return new_board

minimax(x_winning, True)

'''
MINIMAX
Recursion In Minimax
Nice work! We’re halfway through writing our minimax() function — it’s time to make the recursive call.

We have our variable called best_value . We’ve made a hypothetical board where we’ve made one of our potential moves. We now want to know whether the value of that board is better than our current best_value.

In order to find the value of the hypothetical board, we’ll call minimax(). But this time our parameters are different! The first parameter isn’t the starting board. Instead, it’s new_board, the hypothetical board that we just made.

The second parameter is dependent on whether we’re the maximizing or minimizing player. If is_maximizing is True, then the new parameter should be false False. If is_maximizing is False, then we should give the recursive call True.

It’s like we’re taking the new board, passing it to the other player, and asking “what would the value of this board be if we gave it to you?”

To give the recursive call the opposite of is_maximizing, we can give it not is_maximizing.

That call to minimax() will return the value of the hypothetical board. We can then compare the value to our best_value. If the value of the hypothetical board was better than best_value, then we should make that value the new best_value.

Instructions
1.
Let’s make that recursive call!

Inside the for loop after calling select_space(), create a variable named hypothetical_value and set it equal to minimax() using the parameters new_board and not is_maximizing.

To help us check if you did this correctly, return hypothetical_value instead of best_value. We’ll change that return statement soon!


Fill in the correct parameters.`
'''
hypothetical_value = minimax(____, ____)'''
2.
Now that we have hypothetical_value we want to see if it is better than best_value.

Inside the for loop, write another set of if/else statements checking to see if is_maximizing is True or False

If is_maximizing is True, then best_value should become the value of hypothetical_value if hypothetical_value is greater than best_value.

If is_maximizing is False, then best_value should become the value of hypothetical_value if hypothetical_value is less than best_value.

Switch your return statements back to returning best_value.


Your code for the maximizing case should look like this
'''
if is_maximizing == True:
  if ____ > ____:
    best_value = hypothetical_value'''
3.
Wow! Great work, our minimax function is done. We’ve set up a couple of boards for you. Call minimax() three different times on the boards x_winning, and o_winning. In each of those boards, it’s "X"‘s turn, so set is_maximizing to True.

Print the results of each. What does it mean if the result is a -1, 0 or 1?

You can also try running minimax() on new_game. This might take a few seconds — the algorithm needs to traverse the entire game tree to reach the leaves!


One of these function calls should be the following:

print(minimax(x_winning, True))'''

from tic_tac_toe import *
from copy import deepcopy

def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
  if has_won(board, "X"):
    return 1
  elif has_won(board, "O"):
    return -1
  else:
    return 0

new_game = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

x_winning = [
	["X", "2", "O"],
	["4", "O", "6"],
	["7", "8", "X"]
]

o_winning = [
	["X", "X", "O"],
	["4", "X", "6"],
	["7", "O", "O"]
]

def minimax(input_board, is_maximizing):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board):
    return evaluate_board(input_board)
  if is_maximizing == True:
    best_value = -float("Inf")
    symbol = "X"
  else:
    best_value = float("Inf")
    symbol = "O"
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
    hypothetical_value = minimax(new_board, not is_maximizing)
    if is_maximizing:
      if hypothetical_value > best_value:
        best_value = hypothetical_value
    else:
      if hypothetical_value < best_value:
        best_value = hypothetical_value
    
  return best_value


print(minimax(new_game, True))
print(minimax(x_winning, True))
print(minimax(o_winning, True))


'''
MINIMAX
Which Move?
Right now our minimax() function is returning the value of the best possible move. So if our final answer is a 1, we know that "X" should be able to win the game. But that doesn’t really help us — we know that "X" should win, but we aren’t keeping track of what move will cause that!

To do this, we need to make two changes to our algorithm. We first need to set up a variable to keep track of the best move (let’s call it best_move). Whenever the algorithm finds a new best_value, best_move variable should be updated to be whatever move resulted in that value.

Second, we want the algorithm to return best_move at the very end. But in order for the recursion to work, the algorithm is dependent on returning best_value. To fix this, we’ll now return a list of two numbers — [best_value, best_move].

Let’s edit our minimax function to keep track of which move leads to the best possible value!

Instructions
1.
Instead of returning just the value, we’re going to return a list that looks like [best_value, best_move].

We haven’t kept track of the best move yet, so for now, let’s change both of our return statements to be [best_value, ""]. This includes the base case! The base case should return [evaluate_board(input_board), ""]

We also need to make sure when we’re setting hypothetical_value, we’re setting it equal to the value — not the entire list. The recursive call should now look like this.
'''
minimax(new_board, not is_maximizing)[0]'''

After changing the return statements to be a list, add [0] to the end of your lines of code where you define hypothetical_value. It should look like this:
'''
hypothetical_value = solution(new_board, False)[0]'''
2.
Let’s now keep track of which move was best.

Right after the base case, create a variable named best_move. Set it equal to the empty string ("")

For both the maximizing case and the minimizing case, if we’ve found a new best_value, we should also update best_move. Inside those two if statements, set best_move equal to your variable from your for loop (ours is named move). We’re now remembering which move goes with the best possible value.

Change your last return statement. Instead of returning [best_value, ""], return [best_value, best_move].


One of your if statements should look like this:
'''
if is_maximizing == True and hypothetical_value > best_value:
  best_value = hypothetical_value
  best_move = move'''
3.
Call your function on x_winning, and o_winning as the maximizing player. Print the results. What does the return value tell you now?

You can also try it on new_game. This might take a few seconds.


One of the function calls should look like this:

print(minimax(x_winning, True))
This will return a list of two items. The second item in the list is the optimal move!'''


from tic_tac_toe import *
from copy import deepcopy

def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
  if has_won(board, "X"):
    return 1
  elif has_won(board, "O"):
    return -1
  else:
    return 0

new_game = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

x_winning = [
	["X", "2", "O"],
	["4", "O", "6"],
	["7", "8", "X"]
]

o_winning = [
	["X", "X", "O"],
	["4", "X", "6"],
	["7", "O", "O"]
]

def minimax(input_board, is_maximizing):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board):
    return [evaluate_board(input_board),'']
  best_move = ""
  if is_maximizing == True:
    best_value = -float("Inf")
    symbol = "X"
  else:
    best_value = float("Inf")
    symbol = "O"
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
    hypothetical_value = minimax(new_board, not is_maximizing)[0]
    if is_maximizing == True and hypothetical_value > best_value:
      best_value = hypothetical_value
      best_move = move
    if is_maximizing == False and hypothetical_value < best_value:
      best_value = hypothetical_value
      best_move = move
  return [best_value, best_move]

print(minimax(x_winning,True))
print(minimax(o_winning,True))
print(minimax(new_game,True))

'''
MINIMAX
Play a Game
Amazing! Our minimax() function is now returning a list of [value, move]. move gives you the number you should pick to play an optimal game of Tic-Tac-Toe for any given game state.

This line of code instructs the AI to make a move as the "X" player:

select_space(my_board, minimax(my_board, True)[1], "X")
Take some time to really understand all of the parameters. Why do we pass True to minimax()? Why do we use [1] at the end of minimax()?'''
'''
Take some time to play a game against the computer. If you’re playing with "X"s, make your move as "X", and then call minimax() on the board using is_maximizing = False. The second item in that list will tell you the AI’s move. You can then enter the move for the AI as "O", make your next move as "X", and call the minimax() function again to get the AI’s next move.

You can also try having two AIs play each other. If you uncomment the code at the bottom of the file, two AI will play each other until the game is over. What do you think the result will be? The file will run for about 15 seconds before showing you the outcome of the game.'''

from tic_tac_toe import *

my_board = [
	["1", "2", "3"],
	["4", "5", "6"],
	["7", "8", "9"]
]

while not game_is_over(my_board):
  select_space(my_board, minimax(my_board, True)[1], "X")
  print_board(my_board)
  if not game_is_over(my_board):
    select_space(my_board, minimax(my_board, False)[1], "O")
    print_board(my_board)  
    
#------------tic_tac_toe.py---------------

def print_board(board):
    print("|-------------|")
    print("| Tic Tac Toe |")
    print("|-------------|")
    print("|             |")
    print("|    " + board[0][0] + " " + board[0][1] + " " + board[0][2] + "    |")
    print("|    " + board[1][0] + " " + board[1][1] + " " + board[1][2] + "    |")
    print("|    " + board[2][0] + " " + board[2][1] + " " + board[2][2] + "    |")
    print("|             |")
    print("|-------------|")
    print()


def select_space(board, move, turn):
    if move not in range(1,10):
        return False
    row = int((move-1)/3)
    col = (move-1)%3
    if board[row][col] != "X" and board[row][col] != "O":
        board[row][col] = turn
        return True
    else:
        return False

def available_moves(board):
    moves = []
    for row in board:
        for col in row:
            if col != "X" and col != "O":
                moves.append(int(col))
    return moves

def has_won(board, player):
    for row in board:
        if row.count(player) == 3:
            return True
    for i in range(3):
        if board[0][i] == player and board[1][i] == player and board[2][i] == player:
            return True
    if board[0][0] == player and board[1][1] == player and board[2][2] == player:
        return True
    if board[0][2] == player and board[1][1] == player and board[2][0] == player:
        return True
    return False
  
  
from tic_tac_toe import *
from copy import deepcopy

def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
  if has_won(board, "X"):
    return 1
  elif has_won(board, "O"):
    return -1
  else:
    return 0

def minimax(input_board, is_maximizing):
  # Base case - the game is over, so we return the value of the board
  if game_is_over(input_board):
    return [evaluate_board(input_board), ""]
  # The maximizing player
  if is_maximizing:
    # The best value starts at the lowest possible value
    best_value = -float("Inf")
    best_move = ""
    # Loop through all the available moves
    for move in available_moves(input_board):
      # Make a copy of the board and apply the move to it
      new_board = deepcopy(input_board)
      select_space(new_board, move, "X")
      # Recursively find your opponent's best move
      hypothetical_value = minimax(new_board, False)[0]
      # Update best value if you found a better hypothetical value
      if hypothetical_value > best_value:
        best_value = hypothetical_value
        best_move = move
    return [best_value, best_move]
  # The minimizing player
  else:
    # The best value starts at the highest possible value
    best_value = float("Inf")
    best_move = ""
    # Testing all potential moves
    for move in available_moves(input_board):
      # Copying the board and making the move
      new_board = deepcopy(input_board)
      select_space(new_board, move, "O")
      # Passing the new board back to the maximizing player
      hypothetical_value = minimax(new_board, True)[0]
      # Keeping track of the best value seen so far
      if hypothetical_value < best_value:
        best_value = hypothetical_value
        best_move = move
    return [best_value, best_move]
    
'''
MINIMAX
Review
Nice work! You implemented the minimax algorithm to create an unbeatable Tic Tac Toe AI! Here are some major takeaways from this lesson.

A game can be represented as a tree. The current state of the game is the root of the tree, and each potential move is a child of that node. The leaves of the tree are game states where the game has ended (either in a win or a tie).
The minimax algorithm returns the best possible move for a given game state. It assumes that your opponent will also be using the minimax algorithm to determine their best move.
Game states can be evaluated and given a specific score. This is relatively easy when the game is over — the score is usually a 1, -1 depending on who won. If the game is a tie, the score is usually a 0.
In our next lesson on the minimax algorithm, we’ll look at games that are more complex than Tic Tac Toe. How does the algorithm change if it’s too computationally intensive to reach the leaves of the game tree? What strategies can we use to traverse the tree in a smarter way? We’ll tackle these questions in our next lesson!'''
'''
Take a look at our Connect Four AI for a sneak preview of our next minimax lesson. In the terminal type python3 minimax.py to play against the AI.

You can make your move by typing the number of the column that you want to put your piece in.

In the code, you can change the “intelligence” of the AI by changing the parameter of play_game(). The parameter should be a number greater than 0. If you make it greater than 6 or 7, it will take the computer a long time to find their best move.

Make sure to click the Run button to save your code before running your file in the terminal!

You can also set up an AI vs AI game by commenting out play_game() and calling two_ai_game(). This function takes two parameters — the “intelligence” of each AI players. Try starting a game with a bad X player and a smart O player by calling two_ai_game(2, 6) and see what happens.

Feel free to test out more games with different AIs.'''


#-------------------minimax.py---------------

from connect_four import *

play_game(4)


#-------------------connect_four.py---------------

   from copy import deepcopy
import random
random.seed(108)

def print_board(board):
    print()
    print(' ', end='')
    for x in range(1, len(board) + 1):
        print(' %s  ' % x, end='')
    print()

    print('+---+' + ('---+' * (len(board) - 1)))

    for y in range(len(board[0])):

        print('|', end='')
        for x in range(len(board)):
            print(' %s |' % board[x][y], end='')
        print()

        print('+---+' + ('---+' * (len(board) - 1)))

def select_space(board, column, player):
    if not move_is_valid(board, column):
        return False
    if player != "X" and player != "O":
        return False
    for y in range(len(board[0])-1, -1, -1):
        if board[column-1][y] == ' ':
            board[column-1][y] = player
            return True
    return False

def board_is_full(board):
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == ' ':
                return False
    return True

def move_is_valid(board, move):
    if move < 1 or move > (len(board)):
        return False

    if board[move-1][0] != ' ':
        return False

    return True

def available_moves(board):
    moves = []
    for i in range(1, len(board)+1):
        if move_is_valid(board, i):
            moves.append(i)
    return moves

def has_won(board, symbol):
    # check horizontal spaces
    for y in range(len(board[0])):
        for x in range(len(board) - 3):
            if board[x][y] == symbol and board[x+1][y] == symbol and board[x+2][y] == symbol and board[x+3][y] == symbol:
                return True

    # check vertical spaces
    for x in range(len(board)):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x][y+1] == symbol and board[x][y+2] == symbol and board[x][y+3] == symbol:
                return True

    # check / diagonal spaces
    for x in range(len(board) - 3):
        for y in range(3, len(board[0])):
            if board[x][y] == symbol and board[x+1][y-1] == symbol and board[x+2][y-2] == symbol and board[x+3][y-3] == symbol:
                return True

    # check \ diagonal spaces
    for x in range(len(board) - 3):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x+1][y+1] == symbol and board[x+2][y+2] == symbol and board[x+3][y+3] == symbol:
                return True

    return False


def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
    if has_won(board, "X"):
      return float("Inf")
    elif has_won(board, "O"):
      return -float("Inf")
    else:
      x_streaks = count_streaks(board, "X")
      o_streaks = count_streaks(board, "O")
      return x_streaks - o_streaks

def count_streaks(board, symbol):
    count = 0
    for col in range(len(board)):
        for row in range(len(board[0])):
            if board[col][row] != symbol:
                continue
            # right
            if col < len(board) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #left
            if col > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-right
            if col < len(board) - 3 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-right
            if col < len(board) - 3 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-left
            if col > 2 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down
            num_in_streak = 0
            if row < len(board[0]) - 3:
                for i in range(4):
                    if row + i < len(board[0]):
                        if board[col][row + i] == symbol:
                            num_in_streak += 1
                        else:
                            break
            for i in range(4):
                if row - i > 0:
                    if board[col][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col][row - i] == " ":
                        break
                    else:
                        num_in_streak == 0
            if row < 3:
                if num_in_streak + row < 4:
                    num_in_streak = 0
            count += num_in_streak
    return count

def minimax(input_board, is_maximizing, depth, alpha, beta):
  if game_is_over(input_board) or depth == 0:
        return [evaluate_board(input_board), ""]
  if is_maximizing:
    best_value = -float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "X")
      hypothetical_value = minimax(new_board, False, depth - 1, alpha, beta)[0]
      if hypothetical_value > best_value:
        best_value = hypothetical_value
        best_move = move
      alpha = max(alpha, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]
  else:
    best_value = float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "O")
      hypothetical_value = minimax(new_board, True, depth - 1, alpha, beta)[0]
      if hypothetical_value < best_value:
        best_value = hypothetical_value
        best_move = move
      beta = min(beta, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]


def play_game(ai):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    board = []
    for x in range(BOARDWIDTH):
      board.append([' '] * BOARDHEIGHT)
    while not game_is_over(board):
        print_board(board)
        moves = available_moves(board)
        print("Available moves: " , moves)
        choice = 100
        good_move = False
        while not good_move:
            choice = input("Select a move:\n")
            try:
                move = int(choice)
            except ValueError:
                continue
            if move in moves:
                good_move = True
        select_space(board, int(choice), "X")
        if not game_is_over(board):
          result = minimax(board, False, ai, -float("Inf"), float("Inf"))
          print("Computer chose: ", result[1])
          select_space(board, result[1], "O")
    print_board(board)
    if has_won(board, "X"):
        print("X won!")
    elif has_won(board, "O"):
        print("O won!")
    else:
        print("It's a tie!")

def two_ai_game(ai1, ai2):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    my_board = []
    for x in range(BOARDWIDTH):
      my_board.append([' '] * BOARDHEIGHT)
    while not game_is_over(my_board):
      result = minimax(my_board, True, ai1, -float("Inf"), float("Inf"))
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        result = minimax(my_board, False, ai2, -float("Inf"), float("Inf"))
        print( "O Turn\nO selected ", result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")
 

'''  
ADVANCED MINIMAX
Connect Four
In our first lesson on the minimax algorithm, we wrote a program that could play the perfect game of Tic-Tac-Toe. Our AI looked at all possible future moves and chose the one that would be most beneficial. This was a viable strategy because Tic Tac Toe is a small enough game that it wouldn’t take too long to reach the leaves of the game tree.

However, there are games, like Chess, that have much larger trees. There are 20 different options for the first move in Chess, compared to 9 in Tic-Tac-Toe. On top of that, the number of possible moves often increases as a chess game progresses. Traversing to the leaves of a Chess tree simply takes too much computational power.

In this lesson, we’ll investigate two techniques to solve this problem. The first technique puts a hard limit on how far down the tree you allow the algorithm to go. The second technique uses a clever trick called pruning to avoid exploring parts of the tree that we know will be useless.

Before we dive in, let’s look at the tree of a more complicated game — Connect Four!

If you’ve never played Connect Four before, the goal is to get a streak of four of your pieces in any direction — horizontally, vertically, or diagonally. You can place a piece by picking a column. The piece will fall to the lowest available row in that column.

Instructions
1.
We’ve imported a Connect Four game engine along with a board that’s in the middle of a game.

To start, let’s call the print_board() function using half_done as a parameter.


Your function call should look like this:

print_board(half_done)
These players aren’t playing very well!

2.
Call the tree_size() function using half_done and "X" as parameters. Print the result. This will show you the number of game states in the tree, assuming half_done is the root of the tree and it is "X"‘s turn.


Finish this function call :

print(tree_size(half_done, _____))
3.
Let’s make a move and see how the size of the tree changes. Let’s place an "X" in column 6. Before calling the tree_size() function, call the select_space() function with the following three parameters:

half_done — The board that you’re making the move on.
6 — The column you’re selecting.
"X" — The type of piece you’re putting in column 6.
Since "X" has taken their turn, it is now "O"‘s turn. Change the second parameter in the tree_size() function to be "O".


Complete this function to make "X" take their turn:

select_space(half_done, ____, "X")
Then make sure to change the second parameter of tree_size().'''


#--------script.py---------------

from connect_four import *

print_board(half_done)

select_space(half_done,6,"X")

print(tree_size(half_done, 'O'))


#-------------Connect_four.py---------------
from copy import deepcopy
import random
random.seed(108)

def print_board(board):
    print()
    print(' ', end='')
    for x in range(1, len(board) + 1):
        print(' %s  ' % x, end='')
    print()

    print('+---+' + ('---+' * (len(board) - 1)))

    for y in range(len(board[0])):
        print('|   |' + ('   |' * (len(board) - 1)))

        print('|', end='')
        for x in range(len(board)):
            print(' %s |' % board[x][y], end='')
        print()

        print('|   |' + ('   |' * (len(board) - 1)))

        print('+---+' + ('---+' * (len(board) - 1)))

def select_space(board, column, player):
    if not move_is_valid(board, column):
        return False
    if player != "X" and player != "O":
        return False
    for y in range(len(board[0])-1, -1, -1):
        if board[column-1][y] == ' ':
            board[column-1][y] = player
            return True
    return False

def board_is_full(board):
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == ' ':
                return False
    return True

def move_is_valid(board, move):
    if move < 1 or move > (len(board)):
        return False

    if board[move-1][0] != ' ':
        return False

    return True

def available_moves(board):
    moves = []
    for i in range(1, len(board)+1):
        if move_is_valid(board, i):
            moves.append(i)
    return moves

def has_won(board, symbol):
    # check horizontal spaces
    for y in range(len(board[0])):
        for x in range(len(board) - 3):
            if board[x][y] == symbol and board[x+1][y] == symbol and board[x+2][y] == symbol and board[x+3][y] == symbol:
                return True

    # check vertical spaces
    for x in range(len(board)):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x][y+1] == symbol and board[x][y+2] == symbol and board[x][y+3] == symbol:
                return True

    # check / diagonal spaces
    for x in range(len(board) - 3):
        for y in range(3, len(board[0])):
            if board[x][y] == symbol and board[x+1][y-1] == symbol and board[x+2][y-2] == symbol and board[x+3][y-3] == symbol:
                return True

    # check \ diagonal spaces
    for x in range(len(board) - 3):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x+1][y+1] == symbol and board[x+2][y+2] == symbol and board[x+3][y+3] == symbol:
                return True

    return False


def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
    if has_won(board, "X"):
      return float("Inf")
    elif has_won(board, "O"):
      return -float("Inf")
    else:
      x_streaks = count_streaks(board, "X")
      o_streaks = count_streaks(board, "O")
      return x_streaks - o_streaks

def count_streaks(board, symbol):
    count = 0
    for col in range(len(board)):
        for row in range(len(board[0])):
            if board[col][row] != symbol:
                continue
            # right
            if col < len(board) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #left
            if col > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-right
            if col < len(board) - 3 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-right
            if col < len(board) - 3 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-left
            if col > 2 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down
            num_in_streak = 0
            if row < len(board[0]) - 3:
                for i in range(4):
                    if row + i < len(board[0]):
                        if board[col][row + i] == symbol:
                            num_in_streak += 1
                        else:
                            break
            for i in range(4):
                if row - i > 0:
                    if board[col][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col][row - i] == " ":
                        break
                    else:
                        num_in_streak == 0
            if row < 3:
                if num_in_streak + row < 4:
                    num_in_streak = 0
            count += num_in_streak
    return count

def minimax(input_board, is_maximizing, depth, alpha, beta):
  if game_is_over(input_board) or depth == 0:
        return [evaluate_board(input_board), ""]
  if is_maximizing:
    best_value = -float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "X")
      hypothetical_value = minimax(new_board, False, depth - 1, alpha, beta)[0]
      if hypothetical_value > best_value:
        best_value = hypothetical_value
        best_move = move
      alpha = max(alpha, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]
  else:
    best_value = float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "O")
      hypothetical_value = minimax(new_board, True, depth - 1, alpha, beta)[0]
      if hypothetical_value < best_value:
        best_value = hypothetical_value
        best_move = move
      beta = min(beta, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]


def play_game(ai):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    board = []
    for x in range(BOARDWIDTH):
      board.append([' '] * BOARDHEIGHT)
    while not game_is_over(board):
        print_board(board)
        moves = available_moves(board)
        print("Available moves: " , moves)
        choice = 100
        good_move = False
        while not good_move:
            choice = input("Select a move:\n")
            try:
                move = int(choice)
            except ValueError:
                continue
            if move in moves:
                good_move = True
        select_space(board, int(choice), "X")
        if not game_is_over(board):
          result = minimax(board, False, ai, -float("Inf"), float("Inf"))
          print("Computer chose: ", result[1])
          select_space(board, result[1], "O")

def two_ai_game(ai1, ai2):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    my_board = []
    for x in range(BOARDWIDTH):
      my_board.append([' '] * BOARDHEIGHT)
    while not game_is_over(my_board):
      result = minimax(my_board, True, ai1, -float("Inf"), float("Inf"))
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        result = minimax(my_board, False, ai2, -float("Inf"), float("Inf"))
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")

#two_ai_game(3, 4)
#play_game(5)


def tree_size(board, turn):
    if game_is_over(board):
        return 1
    count = 1
    for move in available_moves(board):
        new_board = deepcopy(board)
        if turn == "X":
            select_space(new_board, move, "X")
            count += tree_size(new_board, "O")
        else:
            select_space(new_board, move, "O")
            count += tree_size(new_board, "X")
    return count


def make_board():
    new_game = []
    for x in range(7):
        new_game.append([' '] * 6)
    return new_game


half_done = []
for x in range(7):
  half_done.append([' '] * 6)

for i in range(6):
    if i % 2 == 0:
        select_space(half_done, 1, "X")
    else:
        select_space(half_done, 1, "O")

for i in range(6):
    if i % 2 == 0:
        select_space(half_done, 7, "X")
    else:
        select_space(half_done, 7, "O")

for i in range(6):
    if i % 2 == 0:
        select_space(half_done, 3, "X")
    else:
        select_space(half_done, 3, "O")

for i in range(6):
    if i % 2 == 0:
        select_space(half_done, 2, "X")
    else:
        select_space(half_done, 2, "O")

for i in range(5):
    if i % 2 == 0:
        select_space(half_done, 6, "X")
    else:
        select_space(half_done, 6, "O")
  

'''
ADVANCED MINIMAX
Depth and Base Case
The first strategy we’ll use to handle these enormous trees is stopping the recursion early. There’s no need to go all the way to the leaves! We’ll just look a few moves ahead.

Being able to stop before reaching the leaves is critically important for the efficiency of this algorithm. It could take literal days to reach the leaves of a game of chess. Stopping after only a few levels limits the algorithm’s understanding of the game, but it makes the runtime realistic.

In order to implement this, we’ll add another parameter to our function called depth. Every time we make a recursive call, we’ll decrease depth by 1 like so:

def minimax(input_board, minimizing_player, depth):
  # Base Case
  if game_is over(input_bopard):
    return ...
  else:
    # …
    # Recursive Call
    hypothetical_value = minimax(new_board, True, depth - 1)[0]

We’ll also have to edit our base case condition. We now want to stop if the game is over (we’ve hit a leaf), or if depth is 0.

Instructions
1.
We’ve given you a minimax() function that recurses to the leaves. Edit it so it has a third parameter named depth.

Change the recursive call to decrease depth by 1 each time.

Change your base case — the recursion should stop when the game is over or when depth is 0.


The recursive call should have a third parameter — depth - 1.

2.
Outside the function, call minimax() on new_board as the maximizing player with a depth of 3 and print the results. Remember, minimax() returns a list of two numbers. The first is the value of the best possible move, and the second is the move itself.


The minimax() function now has three parameters. Print the results like this:

print(minimax(new_board, True, 3))'''

#-----------------scripy.py-------------------
from connect_four import *
import random
random.seed(108)

new_board = make_board()

# Add a third parameter named depth
def minimax(input_board, is_maximizing, depth):
  # Change this if statement to also check to see if depth = 0
  
  if game_is_over(input_board) or depth ==0:
    return [evaluate_board(input_board), ""]
  best_move = ""
  if is_maximizing == True:
    best_value = -float("Inf")
    symbol = "X"
  else:
    best_value = float("Inf")
    symbol = "O"
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
    #Add a third parameter to this recursive call
    hypothetical_value = minimax(new_board, not is_maximizing, depth -1)[0]
    if is_maximizing == True and hypothetical_value > best_value:
      best_value = hypothetical_value
      best_move = move
    if is_maximizing == False and hypothetical_value < best_value:
      best_value = hypothetical_value
      best_move = move
  return [best_value, best_move]

print(minimax(new_board, True, 3))

#----------------------connect_four.py---------------
from copy import deepcopy

def print_board(board):
    print()
    print(' ', end='')
    for x in range(1, len(board) + 1):
        print(' %s  ' % x, end='')
    print()

    print('+---+' + ('---+' * (len(board) - 1)))

    for y in range(len(board[0])):
        print('|   |' + ('   |' * (len(board) - 1)))

        print('|', end='')
        for x in range(len(board)):
            print(' %s |' % board[x][y], end='')
        print()

        print('|   |' + ('   |' * (len(board) - 1)))

        print('+---+' + ('---+' * (len(board) - 1)))

def select_space(board, column, player):
    if not move_is_valid(board, column):
        return False
    if player != "X" and player != "O":
        return False
    for y in range(len(board[0])-1, -1, -1):
        if board[column-1][y] == ' ':
            board[column-1][y] = player
            return True
    return False

def board_is_full(board):
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == ' ':
                return False
    return True

def move_is_valid(board, move):
    if move < 1 or move > (len(board)):
        return False

    if board[move-1][0] != ' ':
        return False

    return True

def available_moves(board):
    moves = []
    for i in range(1, len(board)+1):
        if move_is_valid(board, i):
            moves.append(i)
    return moves

def has_won(board, symbol):
    # check horizontal spaces
    for y in range(len(board[0])):
        for x in range(len(board) - 3):
            if board[x][y] == symbol and board[x+1][y] == symbol and board[x+2][y] == symbol and board[x+3][y] == symbol:
                return True

    # check vertical spaces
    for x in range(len(board)):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x][y+1] == symbol and board[x][y+2] == symbol and board[x][y+3] == symbol:
                return True

    # check / diagonal spaces
    for x in range(len(board) - 3):
        for y in range(3, len(board[0])):
            if board[x][y] == symbol and board[x+1][y-1] == symbol and board[x+2][y-2] == symbol and board[x+3][y-3] == symbol:
                return True

    # check \ diagonal spaces
    for x in range(len(board) - 3):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x+1][y+1] == symbol and board[x+2][y+2] == symbol and board[x+3][y+3] == symbol:
                return True

    return False


def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def evaluate_board(board):
    if has_won(board, "X"):
      return float("Inf")
    elif has_won(board, "O"):
      return -float("Inf")
    else:
      return 0

def count_streaks(board, symbol):
    count = 0
    for col in range(len(board)):
        for row in range(len(board[0])):
            if board[col][row] != symbol:
                continue
            # right
            if col < len(board) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #left
            if col > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-right
            if col < len(board) - 3 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-right
            if col < len(board) - 3 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-left
            if col > 2 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down
            num_in_streak = 0
            if row < len(board[0]) - 3:
                for i in range(4):
                    if row + i < len(board[0]):
                        if board[col][row + i] == symbol:
                            num_in_streak += 1
                        else:
                            break
            for i in range(4):
                if row - i > 0:
                    if board[col][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col][row - i] == " ":
                        break
                    else:
                        num_in_streak == 0
            if row < 3:
                if num_in_streak + row < 4:
                    num_in_streak = 0
            count += num_in_streak
    return count

def play_game(ai):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    board = []
    for x in range(BOARDWIDTH):
      board.append([' '] * BOARDHEIGHT)
    while not game_is_over(board):
        print_board(board)
        moves = available_moves(board)
        print("Available moves: " , moves)
        choice = 100
        good_move = False
        while not good_move:
            choice = input("Select a move:\n")
            try:
                move = int(choice)
            except ValueError:
                continue
            if move in moves:
                good_move = True
        select_space(board, int(choice), "X")
        if not game_is_over(board):
          result = minimax(board, False, ai, -float("Inf"), float("Inf"))
          print("Computer chose: ", result[1])
          select_space(board, result[1], "O")

def two_ai_game(ai1, ai2):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    my_board = []
    for x in range(BOARDWIDTH):
      my_board.append([' '] * BOARDHEIGHT)
    while not game_is_over(my_board):
      result = minimax(my_board, True, ai1, -float("Inf"), float("Inf"))
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        result = minimax(my_board, False, ai2, -float("Inf"), float("Inf"))
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")
        
def make_board():
    new_game = []
    for x in range(7):
        new_game.append([' '] * 6)
    return new_game      
    

'''
ADVANCED MINIMAX
Evaluation Function
By adding the depth parameter to our function, we’ve prevented it from spending days trying to reach the end of the tree. But we still have a problem: our evaluation function doesn’t know what to do if we’re not at a leaf. Right now, we’re returning positive infinity if Player 1 has won, negative infinity if Player 2 has won, and 0 otherwise. Unfortunately, since we’re not making it to the leaves of the board, neither player has won and we’re always returning 0. We need to rewrite our evaluation function.

Writing an evaluation function takes knowledge about the game you’re playing. It is the part of the code where a programmer can infuse their creativity into their AI. If you’re playing Chess, your evaluation function could count the difference between the number of pieces each player owns. For example, if white had 15 pieces, and black had 10 pieces, the evaluation function would return 5. This evaluation function would make an AI that prioritizes capturing pieces above all else.

You could go even further — some pieces might be more valuable than others. Your evaluation function could keep track of the value of each piece to see who is ahead. This evaluation function would create an AI that might really try to protect their queen. You could even start finding more abstract ways to value a game state. Maybe the position of each piece on a Chess board tells you something about who is winning.

It’s up to you to define what you value in a game. These evaluation functions could be incredibly complex. If the maximizing player is winning (by your definition of what it means to be winning), then the evaluation function should return something greater than 0. If the minimizing player is winning, then the evaluation function should return something less than 0.

Instructions
1.
Let’s rewrite our evaluation function for Connect Four. We’ll be editing the part of the evaluation function under the else statement. We need to define how to evaluate a board when nobody has won.

Let’s write a slightly silly evaluation function that prioritizes having the top piece of a column. If the board looks like the image below, we want our evaluation function to return 2 since the maximizing player ("X") has two more “top pieces” than "O".

A connect four board with four Xs on the top of columns and two Os on the top

For now, inside the else statement, delete the current return statement. Create two variables named num_top_x and num_top_o and start them both at 0. Return num_top_x - num_top_o.


2.
Before this new return statement, loop through every column in board. Inside that loop, loop through every square in column. You’re now looking at every square in every column going from top to bottom.

If square equals "X" add one to num_top_x and then break the inner for loop to go to the next column.


Your for loops should look like this:

for column in board:
  for square in column:
    # Checking square here
You can now check to see if square is an "X" or an "O". If it is an "X", add 1 to the proper variable and break.

if square == _____:
  num_top_x += 1
  break
3.
If square equals "O" add one to num_top_o and then break the inner for loop to go to the next column.


4.
We’ve imported three boards for you to test this function. We should first get an understanding of what these three boards look like.

Note that these boards aren’t game states you’d find in real games of Connect Four — "X" has been taking some extra turns. Nevertheless, we can still evaluate them!

Call print_board once per board — board_one, board_two, and board_three. What should the evaluation function return for those three boards?


5.
Call evaluate_board once on each board and print the results. Did we trick you with board_three?'''

#------------------script.py---------------
from connect_four import *
import random
random.seed(108)

def evaluate_board(board):
    if has_won(board, "X"):
      return float("Inf")
    elif has_won(board, "O"):
      return -float("Inf")
    else:
      num_top_x = 0
      num_top_o = 0
      for column in board:
        for square in column:
          if square == "X":
            num_top_x += 1
            break
          if square == "O":
            num_top_o += 1
            break
             
    return num_top_x - num_top_o

#print_board(board_one)
#print(board_one)

print_board(board_one)
print_board(board_two)
print_board(board_three)

print(evaluate_board(board_one))
print(evaluate_board(board_two))
print(evaluate_board(board_three))
#----------------------Connect_four.py---------------
from copy import deepcopy

def print_board(board):
    print()
    print(' ', end='')
    for x in range(1, len(board) + 1):
        print(' %s  ' % x, end='')
    print()

    print('+---+' + ('---+' * (len(board) - 1)))

    for y in range(len(board[0])):
        print('|   |' + ('   |' * (len(board) - 1)))

        print('|', end='')
        for x in range(len(board)):
            print(' %s |' % board[x][y], end='')
        print()

        print('|   |' + ('   |' * (len(board) - 1)))

        print('+---+' + ('---+' * (len(board) - 1)))

def select_space(board, column, player):
    if not move_is_valid(board, column):
        return False
    if player != "X" and player != "O":
        return False
    for y in range(len(board[0])-1, -1, -1):
        if board[column-1][y] == ' ':
            board[column-1][y] = player
            return True
    return False

def board_is_full(board):
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == ' ':
                return False
    return True

def move_is_valid(board, move):
    if move < 1 or move > (len(board)):
        return False

    if board[move-1][0] != ' ':
        return False

    return True

def available_moves(board):
    moves = []
    for i in range(1, len(board)+1):
        if move_is_valid(board, i):
            moves.append(i)
    return moves

def has_won(board, symbol):
    # check horizontal spaces
    for y in range(len(board[0])):
        for x in range(len(board) - 3):
            if board[x][y] == symbol and board[x+1][y] == symbol and board[x+2][y] == symbol and board[x+3][y] == symbol:
                return True

    # check vertical spaces
    for x in range(len(board)):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x][y+1] == symbol and board[x][y+2] == symbol and board[x][y+3] == symbol:
                return True

    # check / diagonal spaces
    for x in range(len(board) - 3):
        for y in range(3, len(board[0])):
            if board[x][y] == symbol and board[x+1][y-1] == symbol and board[x+2][y-2] == symbol and board[x+3][y-3] == symbol:
                return True

    # check \ diagonal spaces
    for x in range(len(board) - 3):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x+1][y+1] == symbol and board[x+2][y+2] == symbol and board[x+3][y+3] == symbol:
                return True

    return False


def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def count_streaks(board, symbol):
    count = 0
    for col in range(len(board)):
        for row in range(len(board[0])):
            if board[col][row] != symbol:
                continue
            # right
            if col < len(board) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #left
            if col > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-right
            if col < len(board) - 3 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-right
            if col < len(board) - 3 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-left
            if col > 2 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down
            num_in_streak = 0
            if row < len(board[0]) - 3:
                for i in range(4):
                    if row + i < len(board[0]):
                        if board[col][row + i] == symbol:
                            num_in_streak += 1
                        else:
                            break
            for i in range(4):
                if row - i > 0:
                    if board[col][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col][row - i] == " ":
                        break
                    else:
                        num_in_streak == 0
            if row < 3:
                if num_in_streak + row < 4:
                    num_in_streak = 0
            count += num_in_streak
    return count
  
def minimax(input_board, is_maximizing, depth, alpha, beta):
  if game_is_over(input_board) or depth == 0:
        return [evaluate_board(input_board), ""]
  if is_maximizing:
    best_value = -float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "X")
      hypothetical_value = minimax(new_board, False, depth - 1, alpha, beta)[0]
      if hypothetical_value > best_value:
        best_value = hypothetical_value
        best_move = move
      alpha = max(alpha, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]
  else:
    best_value = float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "O")
      hypothetical_value = minimax(new_board, True, depth - 1, alpha, beta)[0]
      if hypothetical_value < best_value:
        best_value = hypothetical_value
        best_move = move
      beta = min(beta, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]

def play_game(ai):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    board = []
    for x in range(BOARDWIDTH):
      board.append([' '] * BOARDHEIGHT)
    while not game_is_over(board):
        print_board(board)
        moves = available_moves(board)
        print("Available moves: " , moves)
        choice = 100
        good_move = False
        while not good_move:
            choice = input("Select a move:\n")
            try:
                move = int(choice)
            except ValueError:
                continue
            if move in moves:
                good_move = True
        select_space(board, int(choice), "X")
        if not game_is_over(board):
          result = minimax(board, False, ai, -float("Inf"), float("Inf"))
          print("Computer chose: ", result[1])
          select_space(board, result[1], "O")

def two_ai_game(ai1, ai2):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    my_board = []
    for x in range(BOARDWIDTH):
      my_board.append([' '] * BOARDHEIGHT)
    while not game_is_over(my_board):
      result = minimax(my_board, True, ai1, -float("Inf"), float("Inf"))
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        result = minimax(my_board, False, ai2, -float("Inf"), float("Inf"))
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")

def make_board():
    new_game = []
    for x in range(7):
        new_game.append([' '] * 6)
    return new_game        
        
board_one = make_board()
select_space(board_one, 3, "X")
select_space(board_one, 2, "X")
select_space(board_one, 3, "X")
select_space(board_one, 3, "O")
select_space(board_one, 1, "O")

board_two = make_board()
select_space(board_two, 4, "O")
select_space(board_two, 4, "X")
select_space(board_two, 4, "X")
select_space(board_two, 4, "O")
select_space(board_two, 4, "X")
select_space(board_two, 4, "O")
select_space(board_two, 1, "X")
select_space(board_two, 2, "X")
select_space(board_two, 3, "X")
select_space(board_two, 5, "X")
select_space(board_two, 6, "X")
select_space(board_two, 7, "X")

board_three = make_board()
select_space(board_three, 4, "X")
select_space(board_three, 5, "X")
select_space(board_three, 6, "X")
select_space(board_three, 7, "X")
select_space(board_three, 4, "O")
select_space(board_three, 5, "O")
select_space(board_three, 6, "O")
select_space(board_three, 7, "X")
select_space(board_three, 1, "O")
select_space(board_three, 2, "O")
select_space(board_three, 3, "O")




'''
Alpha-Beta Pruning

While examining the children of a maximizer, if v of maximizer > beta, prune the rest of the children.
While examining the children of a minimizer, if v of minimizer < alpha, prune the rest of the children.

from https://www.youtube.com/watch?v=xBXHtz4Gbdo
'''

'''
ADVANCED MINIMAX
Implement Alpha-Beta Pruning
Alpha-beta pruning is accomplished by keeping track of two variables for each node — alpha and beta. alpha keeps track of the minimum score the maximizing player can possibly get. It starts at negative infinity and gets updated as that minimum score increases.

On the other hand, beta represents the maximum score the minimizing player can possibly get. It starts at positive infinity and will decrease as that maximum possible score decreases.

For any node, if alpha is greater than or equal to beta, that means that we can stop looking through that node’s children.

To implement this in our code, we’ll have to include two new parameters in our function — alpha and beta. When we first call minimax() we’ll set alpha to negative infinity and beta to positive infinity.

We also want to make sure we pass alpha and beta into our recursive calls. We’re passing these two values down the tree.

Next, we want to check to see if we should reset alpha and beta. In the maximizing case, we want to reset alpha if the newly found best_value is greater than alpha. In the minimizing case, we want to reset beta if best_value is less than beta.

Finally, after resetting alpha and beta, we want to check to see if we can prune. If alpha is greater than or equal to beta, we can break and stop looking through the other potential moves.

Instructions
1.
Add two new parameters named alpha and beta to your minimax() function as the final two parameters. Inside your minimax() function, when you the recursive call, add alpha and beta as the final two parameters.


You’ll need to change two lines of code:

The definition of the function
The recursive call
2.
After resetting the value of best_value if is_maximizing is True, we want to check to see if we should reset alpha. Set alpha equal to the maximum of alpha and best_value. You can do this quickly by using the max() function. For example, the following line of code would set a equal to the maximum of b and c.

a = max(b, c)
Change both returns statements to include alpha as the last item in the list. For example, the base case return statement should be [evaluate_board(input_board), "", alpha].

Note that this third value in the return statement is not necessary for the algorithm — we need the value of alpha so we can check to see if you did this step correctly!


Use alpha = max(alpha, best_value) after the if statement comparing hypothetical_value and best_value.

3.
If we reset the value of best_value and is_maximizing is False, we want to set beta to be the minimum of beta and best_value. You can use the min() function this time.

In both return statements, add beta as the last item in the list. This is once again unnecessary for the algorithm, but we need it to check your code!


You can use the min() function:

beta = min(best, best_value)
4.
At the very end of the for loop, check to see if alpha is greater than or equal to beta. If that is true, break which will cause your program to stop looking through the remaining possible moves of the current game state.


Finish this block of code. It should be at the very end of the for loop

if ____ >= ____:
  break
5.
We’re going to call minimax() on board, but before we do let’s see what board looks like. Call print_board using board as a parameter.

6.
Call minimax() on board as the maximizing player and print the result. Set depth equal to 6. alpha should be -float("Inf") and beta should be float("Inf").


Finish this block of code:

print(minimax(_____, True, 6, -float("Inf"), _____))'''

#-----------------script.py---------------
from connect_four import *
import random
random.seed(108)

def minimax(input_board, is_maximizing, depth, alpha, beta):
  # Base case - the game is over, so we return the value of the board
  
  if game_is_over(input_board) or depth == 0:
    return [evaluate_board(input_board), "", alpha,beta]
  best_move = ""
  if is_maximizing == True:
    best_value = -float("Inf")
    symbol = "X"
  else:
    best_value = float("Inf")
    symbol = "O"
  for move in available_moves(input_board):
    new_board = deepcopy(input_board)
    select_space(new_board, move, symbol)
    hypothetical_value = minimax(new_board, not is_maximizing, depth - 1, alpha, beta)[0]
    if is_maximizing == True and hypothetical_value > best_value:
      best_value = hypothetical_value
      alpha = max( best_value, alpha)
      best_move = move
    if is_maximizing == False and hypothetical_value < best_value:
      best_value = hypothetical_value
      beta = min (best_value, beta)
      best_move = move
    if alpha > beta:
      break
    
  return [best_value, best_move, alpha, beta]
  
print_board(board)

print(minimax(board, True, 6, -float('Inf'), float("Inf")))


#----------------connect_four.py---------------
from copy import deepcopy

def print_board(board):
    print()
    print(' ', end='')
    for x in range(1, len(board) + 1):
        print(' %s  ' % x, end='')
    print()

    print('+---+' + ('---+' * (len(board) - 1)))

    for y in range(len(board[0])):
        print('|   |' + ('   |' * (len(board) - 1)))

        print('|', end='')
        for x in range(len(board)):
            print(' %s |' % board[x][y], end='')
        print()

        print('|   |' + ('   |' * (len(board) - 1)))

        print('+---+' + ('---+' * (len(board) - 1)))

def select_space(board, column, player):
    if not move_is_valid(board, column):
        return False
    if player != "X" and player != "O":
        return False
    for y in range(len(board[0])-1, -1, -1):
        if board[column-1][y] == ' ':
            board[column-1][y] = player
            return True
    return False

def board_is_full(board):
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == ' ':
                return False
    return True

def move_is_valid(board, move):
    if move < 1 or move > (len(board)):
        return False

    if board[move-1][0] != ' ':
        return False

    return True

def available_moves(board):
    moves = []
    for i in range(1, len(board)+1):
        if move_is_valid(board, i):
            moves.append(i)
    return moves

def has_won(board, symbol):
    # check horizontal spaces
    for y in range(len(board[0])):
        for x in range(len(board) - 3):
            if board[x][y] == symbol and board[x+1][y] == symbol and board[x+2][y] == symbol and board[x+3][y] == symbol:
                return True

    # check vertical spaces
    for x in range(len(board)):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x][y+1] == symbol and board[x][y+2] == symbol and board[x][y+3] == symbol:
                return True

    # check / diagonal spaces
    for x in range(len(board) - 3):
        for y in range(3, len(board[0])):
            if board[x][y] == symbol and board[x+1][y-1] == symbol and board[x+2][y-2] == symbol and board[x+3][y-3] == symbol:
                return True

    # check \ diagonal spaces
    for x in range(len(board) - 3):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x+1][y+1] == symbol and board[x+2][y+2] == symbol and board[x+3][y+3] == symbol:
                return True

    return False


def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def count_streaks(board, symbol):
    count = 0
    for col in range(len(board)):
        for row in range(len(board[0])):
            if board[col][row] != symbol:
                continue
            # right
            if col < len(board) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #left
            if col > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-right
            if col < len(board) - 3 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-right
            if col < len(board) - 3 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-left
            if col > 2 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down
            num_in_streak = 0
            if row < len(board[0]) - 3:
                for i in range(4):
                    if row + i < len(board[0]):
                        if board[col][row + i] == symbol:
                            num_in_streak += 1
                        else:
                            break
            for i in range(4):
                if row - i > 0:
                    if board[col][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col][row - i] == " ":
                        break
                    else:
                        num_in_streak == 0
            if row < 3:
                if num_in_streak + row < 4:
                    num_in_streak = 0
            count += num_in_streak
    return count

def evaluate_board(board):
    if has_won(board, "X"):
      return float("Inf")
    elif has_won(board, "O"):
      return -float("Inf")
    else:
      num_top_x = 0
      num_top_o = 0

      for col in board:
        for square in col:
          if square == "X":
            num_top_x += 1
            break
          elif square == "O":
            num_top_o += 1
            break

      return num_top_x - num_top_o


def play_game(ai):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    board = []
    for x in range(BOARDWIDTH):
      board.append([' '] * BOARDHEIGHT)
    while not game_is_over(board):
        print_board(board)
        moves = available_moves(board)
        print("Available moves: " , moves)
        choice = 100
        good_move = False
        while not good_move:
            choice = input("Select a move:\n")
            try:
                move = int(choice)
            except ValueError:
                continue
            if move in moves:
                good_move = True
        select_space(board, int(choice), "X")
        if not game_is_over(board):
          result = minimax(board, False, ai, -float("Inf"), float("Inf"))
          print("Computer chose: ", result[1])
          select_space(board, result[1], "O")

def two_ai_game(ai1, ai2):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    my_board = []
    for x in range(BOARDWIDTH):
      my_board.append([' '] * BOARDHEIGHT)
    while not game_is_over(my_board):
      result = minimax(my_board, True, ai1, -float("Inf"), float("Inf"))
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        result = minimax(my_board, False, ai2, -float("Inf"), float("Inf"))
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")

def make_board():
    new_game = []
    for x in range(7):
        new_game.append([' '] * 6)
    return new_game

board = make_board()
select_space(board, 3, "X")
select_space(board, 2, "X")
select_space(board, 3, "X")
select_space(board, 3, "O")
select_space(board, 1, "O")
select_space(board, 4, "O")

'''
ADVANCED MINIMAX
Review
Great work! We’ve now edited our minimax() function to work with games that are more complicated than Tic Tac Toe. The core of the algorithm is identical, but we’ve added two major improvements:

We wrote an evaluation function specific to our understanding of the game (in this case, Connect Four). This evaluation function allows us to stop the recursion before reaching the leaves of the game tree.
We implemented alpha-beta pruning. By cleverly detecting useless sections of the game tree, we’re able to ignore those sections and therefore look farther down the tree.
Now’s our chance to put it all together. We’ve written most of the function two_ai_game() which sets up a game of Connect Four played by two AIs. For each player, you need to call fill in the third parameter of their minimax() call.

Remember, right now our evaluation function is using a pretty bad strategy. An AI using the evaluation function we wrote will prioritize making sure its pieces are the top pieces of each column.

Do you think you could write an evaluation function that uses a better strategy? In the project for this course, you can try to write an evaluation function that can beat our AI!

Instructions
1.
Fill in the third parameter of both minimax() function calls. This parameter is the depth of the recursive call. The higher the number, the “smarter” the AI will be.

What happens if they have equal intelligence? What happens if one is significantly smarter than the other?

We suggest keeping these parameters under 7. Anything higher and the program will take a while to complete!'''

#scripty.py---------------
from connect_four import *

def two_ai_game():
    my_board = make_board()
    while not game_is_over(my_board):
      # Fill in the third parameter for the first player's "intelligence"
      result = minimax(my_board, True, 9, -float("Inf"), float("Inf"))
      print( "X Turn\nX selected ", result[1])
      print(result[1])
      select_space(my_board, result[1], "X")
      print_board(my_board)

      if not game_is_over(my_board):
        #Fill in the third parameter for the second player's "intelligence"
        result = minimax(my_board, False, 2, -float("Inf"), float("Inf"))
        print( "O Turn\nO selected ", result[1])
        print(result[1])
        select_space(my_board, result[1], "O")
        print_board(my_board)
    if has_won(my_board, "X"):
        print("X won!")
    elif has_won(my_board, "O"):
        print("O won!")
    else:
        print("It's a tie!")

two_ai_game()

#connect_four.py---------------

from copy import deepcopy
import random
random.seed(108)

def print_board(board):
    print()
    print(' ', end='')
    for x in range(1, len(board) + 1):
        print(' %s  ' % x, end='')
    print()

    print('+---+' + ('---+' * (len(board) - 1)))

    for y in range(len(board[0])):
        print('|   |' + ('   |' * (len(board) - 1)))

        print('|', end='')
        for x in range(len(board)):
            print(' %s |' % board[x][y], end='')
        print()

        print('|   |' + ('   |' * (len(board) - 1)))

        print('+---+' + ('---+' * (len(board) - 1)))

def select_space(board, column, player):
    if not move_is_valid(board, column):
        return False
    if player != "X" and player != "O":
        return False
    for y in range(len(board[0])-1, -1, -1):
        if board[column-1][y] == ' ':
            board[column-1][y] = player
            return True
    return False

def board_is_full(board):
    for x in range(len(board)):
        for y in range(len(board[0])):
            if board[x][y] == ' ':
                return False
    return True

def move_is_valid(board, move):
    if move < 1 or move > (len(board)):
        return False

    if board[move-1][0] != ' ':
        return False

    return True

def available_moves(board):
    moves = []
    for i in range(1, len(board)+1):
        if move_is_valid(board, i):
            moves.append(i)
    return moves

def has_won(board, symbol):
    # check horizontal spaces
    for y in range(len(board[0])):
        for x in range(len(board) - 3):
            if board[x][y] == symbol and board[x+1][y] == symbol and board[x+2][y] == symbol and board[x+3][y] == symbol:
                return True

    # check vertical spaces
    for x in range(len(board)):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x][y+1] == symbol and board[x][y+2] == symbol and board[x][y+3] == symbol:
                return True

    # check / diagonal spaces
    for x in range(len(board) - 3):
        for y in range(3, len(board[0])):
            if board[x][y] == symbol and board[x+1][y-1] == symbol and board[x+2][y-2] == symbol and board[x+3][y-3] == symbol:
                return True

    # check \ diagonal spaces
    for x in range(len(board) - 3):
        for y in range(len(board[0]) - 3):
            if board[x][y] == symbol and board[x+1][y+1] == symbol and board[x+2][y+2] == symbol and board[x+3][y+3] == symbol:
                return True

    return False


def game_is_over(board):
  return has_won(board, "X") or has_won(board, "O") or len(available_moves(board)) == 0

def count_streaks(board, symbol):
    count = 0
    for col in range(len(board)):
        for row in range(len(board[0])):
            if board[col][row] != symbol:
                continue
            # right
            if col < len(board) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #left
            if col > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-right
            if col < len(board) - 3 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-right
            if col < len(board) - 3 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col + i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col + i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #up-left
            if col > 2 and row > 2:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row - i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down-left
            if col > 2 and row < len(board[0]) - 3:
                num_in_streak = 0
                for i in range(4):
                    if board[col - i][row + i] == symbol:
                        num_in_streak += 1
                    elif board[col - i][row + i] != " ":
                        num_in_streak = 0
                        break
                count += num_in_streak
            #down
            num_in_streak = 0
            if row < len(board[0]) - 3:
                for i in range(4):
                    if row + i < len(board[0]):
                        if board[col][row + i] == symbol:
                            num_in_streak += 1
                        else:
                            break
            for i in range(4):
                if row - i > 0:
                    if board[col][row - i] == symbol:
                        num_in_streak += 1
                    elif board[col][row - i] == " ":
                        break
                    else:
                        num_in_streak == 0
            if row < 3:
                if num_in_streak + row < 4:
                    num_in_streak = 0
            count += num_in_streak
    return count

def evaluate_board(board):
    if has_won(board, "X"):
      return float("Inf")
    elif has_won(board, "O"):
      return -float("Inf")
    else:
      num_top_x = 0
      num_top_o = 0

      for col in board:
        for square in col:
          if square == "X":
            num_top_x += 1
            break
          elif square == "O":
            num_top_o += 1
            break

      return num_top_x - num_top_o

def minimax(input_board, is_maximizing, depth, alpha, beta):
  if game_is_over(input_board) or depth == 0:
        return [evaluate_board(input_board), ""]
  if is_maximizing:
    best_value = -float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "X")
      hypothetical_value = minimax(new_board, False, depth - 1, alpha, beta)[0]
      if hypothetical_value > best_value:
        best_value = hypothetical_value
        best_move = move
      alpha = max(alpha, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]
  else:
    best_value = float("Inf")
    moves = available_moves(input_board)
    random.shuffle(moves)
    best_move = moves[0]
    for move in moves:
      new_board = deepcopy(input_board)
      select_space(new_board, move, "O")
      hypothetical_value = minimax(new_board, True, depth - 1, alpha, beta)[0]
      if hypothetical_value < best_value:
        best_value = hypothetical_value
        best_move = move
      beta = min(beta, best_value)
      if alpha >= beta:
        break
    return [best_value, best_move]

def play_game(ai):
    BOARDWIDTH = 7
    BOARDHEIGHT = 6
    board = []
    for x in range(BOARDWIDTH):
      board.append([' '] * BOARDHEIGHT)
    while not game_is_over(board):
        print_board(board)
        moves = available_moves(board)
        print("Available moves: " , moves)
        choice = 100
        good_move = False
        while not good_move:
            choice = input("Select a move:\n")
            try:
                move = int(choice)
            except ValueError:
                continue
            if move in moves:
                good_move = True
        select_space(board, int(choice), "X")
        if not game_is_over(board):
          result = minimax(board, False, ai, -float("Inf"), float("Inf"))
          print("Computer chose: ", result[1])
          select_space(board, result[1], "O")



def make_board():
    new_game = []
    for x in range(7):
        new_game.append([' '] * 6)
    return new_game
