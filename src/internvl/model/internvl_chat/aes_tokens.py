def score2aestoken(n):
    if not (0 <= n <= 100):
        raise ValueError("Score must be between 0 and 100 inclusive.")
    
    if 0 <= n <= 25:
        first = 'a'
        offset = n
    elif 26 <= n <= 50:
        first = 'c'
        offset = n - 26
    elif 51 <= n <= 75:
        first = 'd'
        offset = n - 51
    else:
        first = 'e'
        offset = n - 76

    second = chr(ord('a') + offset)
    return first + second

AESTHETICS_TOKEN_LIST = []

for i in range(101):
    AESTHETICS_TOKEN_LIST.append(score2aestoken(i))
