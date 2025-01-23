## Klasyczne Algorytmy (BFS i A*)

### BFS (Breadth-First Search / Przeszukiwanie wszerz)

Breadth-First Search (BFS) to algorytm przeszukiwania grafu, który eksploruje węzły warstwami, zaczynając od wierzchołka początkowego. BFS gwarantuje znalezienie najkrótszej ścieżki (pod względem liczby kroków) w grafie nieskierowanym, gdy wszystkie krawędzie mają jednakowy koszt.

**Implementacja:**

```python
def bfs(maze):
    start = (0, 0)  # Punkt początkowy
    goal = (len(maze) - 1, len(maze[0]) - 1)  # Punkt końcowy
    queue = deque([start])  # Kolejka wierzchołków do odwiedzenia
    came_from = {start: None}  # Słownik do śledzenia ścieżki

    while queue:
        current = queue.popleft()  # Pobranie wierzchołka z początku kolejki

        if current == goal:
            path = []  # Rekonstrukcja ścieżki
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Odwrócenie ścieżki na poprawną kolejność

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Ruchy w czterech kierunkach
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0]) and maze[neighbor[0]][neighbor[1]] == 0 and neighbor not in came_from:
                queue.append(neighbor)  # Dodanie sąsiada do kolejki
                came_from[neighbor] = current  # Zapisanie poprzednika
    return []
```
**Zalety algorytmu:**
- Gwarantuje znalezienie najkrótszej ścieżki (pod względem liczby kroków) w grafie o jednakowych wagach.
- Prosta implementacja i brak potrzeby użycia heurystyki.

**Ograniczenia:**
- Wysokie zużycie pamięci w przypadku dużych grafów (przechowywanie wszystkich węzłów w kolejce).
- Może być powolny w złożonych labiryntach o dużej liczbie węzłów.

### A* (A-Star)

A* to algorytm wyszukiwania ścieżki, który łączy zalety algorytmu Dijkstry (minimalizacja kosztu dojścia) i BFS (użycie heurystyki w celu przyspieszenia przeszukiwania). Algorytm A* wyznacza ścieżkę od punktu początkowego do celu, korzystając z następującej funkcji kosztu:

`f(n) = g(n) + h(n)`

- `g(n)`: koszt dojścia do węzła \(n\) od punktu początkowego.
- `h(n)`: heurystyka – przewidywany koszt dojścia od \(n\) do celu (np. odległość Manhattan).
- `f(n)`: łączny koszt oszacowany przez algorytm.

**Implementacja:**

```python
def a_star(maze):
    start = (0, 0)  # Punkt początkowy
    goal = (len(maze) - 1, len(maze[0]) - 1)  # Punkt końcowy
    open_set = []  # Lista węzłów do odwiedzenia
    heapq.heappush(open_set, (0, start))  # Dodanie punktu startowego do kolejki priorytetowej
    came_from = {}  # Przechowywanie poprzedników dla śledzenia ścieżki

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Heurystyka: odległość Manhattan

    g_score = {start: 0}  # Koszt dojścia do węzła startowego wynosi 0
    f_score = {start: heuristic(start, goal)}  # Szacowany koszt przejścia od startu do celu

    while open_set:
        _, current = heapq.heappop(open_set)  # Węzeł o najniższym f-score

        if current == goal:
            path = []  # Rekonstrukcja ścieżki
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Odwrócenie ścieżki na poprawną kolejność

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Ruchy w górę, dół, lewo, prawo
            neighbor = (current[0] + dx, current[1] + dy)
            if 0 <= neighbor[0] < len(maze) and 0 <= neighbor[1] < len(maze[0]) and maze[neighbor[0]][neighbor[1]] == 0:
                tentative_g_score = g_score[current] + 1  # Aktualizacja g(n)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current  # Zapisanie poprzednika
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)  # Obliczenie f(n)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))  # Dodanie sąsiada do kolejki
    return []
```

**Zalety algorytmu:**
- Gwarantuje znalezienie najkrótszej ścieżki (przy dopuszczalnej heurystyce).
- Efektywność dzięki ograniczeniu przeszukiwania niepotrzebnych gałęzi (heurystyka).

**Ograniczenia:**
- Wymaga dobrej heurystyki (niedopuszczalna heurystyka może prowadzić do błędów).
- W przypadku dużych grafów może być kosztowny pamięciowo i obliczeniowo.

---