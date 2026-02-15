def attack(board, row, col, n):

    for i in range(row):
        if board[i][col] == 1:
            return True

    i, j = row-1, col-1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return True
        i -= 1
        j -= 1

    i, j = row-1, col+1
    while i >= 0 and j < n:
        if board[i][j] == 1:
            return True
        i -= 1
        j += 1

    return False
    
def N_queens(board, row, n):
    if row == n:
        return True

    for col in range(n):
        if not attack(board, row, col, n):
            board[row][col] = 1

            if N_queens(board, row + 1, n):
                return True

            # Backtrack
            board[row][col] = 0

    return False

n = int(input("Enter the number of queens "))

board = [[0 for _ in range(n)] for _ in range(n)]

if N_queens(board, 0, n):
    for row in board:
        print(row)
else:
    print("Solution does not exist")