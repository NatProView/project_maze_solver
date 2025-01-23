## Generowanie labiryntów (Eller's Algorithm)

Algorytm Ellera pozwala generować labirynty w sposób iteracyjny, utrzymując spójność i brak cykli w strukturze. Kluczowe cechy algorytmu:
- Tworzenie kolejnych rzędów labiryntu na podstawie zestawów, które łączą komórki w spójne regiony.
- Dodawanie ścian pionowych i poziomych, aby zachować strukturę labiryntu.

via [Maze Generation: Eller's Algorithm](https://weblog.jamisbuck.org/2010/12/29/maze-generation-eller-s-algorithm)

**Fragment kodu** (z `maze_generating.py`):

```python
def ellers_algorithm(width, height):
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("Maze dimensions must be odd for Eller's Algorithm to work properly.")

    maze = [[1 for _ in range(width)] for _ in range(height)]
    sets = list(range(width))

    for row in range(0, height - 1, 2):
        for col in range(0, width - 2, 2):
            if sets[col] != sets[col + 2] and random.choice([True, False]):
                maze[row][col + 1] = 0
                old_set = sets[col + 2]
                for i in range(width):
                    if sets[i] == old_set:
                        sets[i] = sets[col]

        next_sets = [-1] * width
        for col in range(0, width, 2):
            if next_sets[col] == -1 or random.choice([True, False]):
                next_sets[col] = sets[col]
                maze[row + 1][col] = 0 

        sets = next_sets

    maze[0][0] = 0
    maze[height - 1][width - 1] = 0
    return torch.tensor(maze, dtype=torch.float32)
```