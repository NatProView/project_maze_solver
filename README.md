# Maze Solver
Maze Solver jest aplikacją do generowania i rozwiązywania labiryntów. 

---

## Spis Treści
1. [Wprowadzenie](#wprowadzenie)
2. [Wymagania](#wymagania)
3. [Instalacja](#instalacja)
4. [Instrukcja Użycia](#instrukcja-użycia)
5. [Szczegóły Techniczne](#szczegóły-techniczne)
6. [Architektura Systemu](#architektura-systemu)
7. [Algorytmy i Rozwiązania](#algorytmy-i-rozwiązania)
8. [Testowanie](#testowanie)
9. [Przyszły Rozwój](#przyszły-rozwój)
10. [Autorzy i Podziękowania](#autorzy-i-podziękowania)
11. [Licencja](#licencja)
---

# Wprowadzenie

Projekt **Maze Solver** to aplikacja wykorzystująca algorytmy sztucznej inteligencji i klasyczne techniki rozwiązywania problemów do znajdowania optymalnych ścieżek w labiryntach o różnej wielkości i złożoności. 

Główne cele projektu to:
- Eksploracja zastosowań algorytmów AI (np. DQN, Double DQN) i klasycznych metod (np. BFS, A*) w rozwiązywaniu labiryntów.
- Porównanie wydajności różnych podejść w kontekście czasu rozwiązania, długości ścieżki i stabilności wyników.
- Udostępnienie interfejsu webowego, który umożliwia wizualizację i analizę wyników.


## Kluczowe funkcjonalności

- **Rozwiązywanie labiryntów**:
  - Algorytmy klasyczne: BFS, A*.
  - Algorytmy oparte na AI: DQN, Double DQN, algorytmy genetyczne i PSO.
- **Generowanie i wizualizacja labiryntów**:
  - Różne rozmiary i poziomy złożoności.
- **Interfejs webowy**:
  - Zarządzanie, analiza i porównanie wyników.
- **Analityka i wizualizacja danych**:
  - Wydajność algorytmów w zależności od rozmiaru i złożoności labiryntów.


## Użyte technologie

1. **Python**:
   - Główna logika aplikacji.
   - Algorytmy rozwiązywania labiryntów.
2. **PyTorch**:
   - Funkcjonalność machine learningowa (definicje modeli, trening, rozwiązywanie labiryntów)
2. **Flask**:
   - Interfejs webowy do interakcji z użytkownikiem.
3. **Chart.js**:
   - Wizualizacja wyników i analityki.
4. **Pandoc**:
   - Automatyczna konwersja dokumentacji do formatu PDF (via Github Actions)

## Instalacja

### 1. Sklonuj repozytorium
```bash
git clone https://github.com/NatProView/project_maze_solver maze_solver
cd maze_solver
```

### 2. Utwórz wirtualne środowisko
Na systemie Linux/macOS:
```bash
python3 -m venv env
source env/bin/activate
```

Na systemie Windows:
```powershell
python -m venv env
.\env\Scripts\activate
```
### Zainstaluj zależności
```bash
pip install -r requirements.txt
```

# Instrukcja użycia
## Uruchomienie aplikacji 
Uruchom aplikację
```
python3 app.py
```
Aplikacja zwróci w terminalu informację o adresie i porcie, na którym działa. Wejdź w ten adres.

Przykładowo
```bash
python3 app.py 
 * Serving Flask app 'app'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
 * Running on http://127.0.0.1:5000 # nasz adres
(...)
```

## Generowanie labiryntu
W sekcji `Generate Maze` podaj wymiary labiryntu oraz jego nazwę. 

> [!NOTE]  
> Uwaga: generator może nie wygenerować labiryntu powyżej pewnej złożoności (ok. 120x120) ze względu na rekursywną specyfikę aktualnej implementacji generatora.
![Generated Maze Example](documentation/generate_maze.png)
## Szczegóły Techniczne
Struktura projektu wygląda następująco

```bash
├── .github/             # Workflows GitHuba
│   ├── workflows/
├── logs/                # Wykresy uczenia modeli AI
│   ├── double_dqn/
│   ├── dqn/
│   ├── genetic/
│   └── pso/
├── logs_detailed/       # Logi z informacją o tempie uczenia się modeli 
│   ├── double_dqn/
│   ├── dqn/
│   ├── genetic/
│   └── pso/
├── mazes/               # Wygenerowane przez użytkownika labirynty 
├── metrics/             # Metryki związane z trenowaniem modeli AI
│   ├── double_dqn/
│   ├── dqn/
│   ├── genetic/
│   └── pso/
├── static/              # pliki css, js
├── templates/           # pliki html
├── trained_models/      # wytrenowane już modele AI
│   ├── double_dqn/
│   ├── dqn/
│   ├── genetic/
│   └── pso/
├── app.py              # Punkt startowy aplikacji (Flask, endpointy)
├── models.py           # Definicje modeli AI
├── maze_generating.py  # Generowanie labiryntów
├── maze_solving.py     # Rozwiązywanie labiryntów algorytmami nie-ai
```