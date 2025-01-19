import random
import torch
import tkinter as tk
from tkinter import filedialog
import requests

# pip install requests 
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# pip install flask
# pip install requests

def ellers_algorithm(width, height):
    """
    Generate a maze using Eller's Algorithm.
    """
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

    for col in range(0, width - 2, 2):
        if sets[col] != sets[col + 2]:
            maze[height - 2][col + 1] = 0 
            old_set = sets[col + 2]
            for i in range(width):
                if sets[i] == old_set:
                    sets[i] = sets[col]

    maze[0][0] = 0
    maze[height - 1][width - 1] = 0

    return torch.tensor(maze, dtype=torch.float32)

def generate_maze(width, height):
    maze = [[1 for _ in range(width)] for _ in range(height)]

    def carve_path(x, y):
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < height and 0 <= ny < width and maze[nx][ny] == 1:
                maze[nx][ny] = 0
                maze[x + dx // 2][y + dy // 2] = 0
                carve_path(nx, ny)

    maze[0][0] = 0
    carve_path(0, 0)

    maze[height - 1][width - 1] = 0

    return torch.tensor(maze, dtype=torch.float32)

def save_maze(maze, filename="maze.pt"):
    torch.save(maze, filename)
    print(f"Maze saved to file: {filename}")
    
def load_maze(filename="maze.pt"):
    maze = torch.load(filename)
    print(f"Maze loaded from file: {filename}")
    return maze

def create_gui(width=15, height=15):
    maze = [[0 for _ in range(width)] for _ in range(height)]

    def toggle_cell(row, col):
        nonlocal maze
        maze[row][col] = 1 if maze[row][col] == 0 else 0
        color = "black" if maze[row][col] == 1 else "white"
        canvas.itemconfig(cells[row][col], fill=color)

    def start_drawing(event):
        canvas.bind("<Motion>", drawing)

    def stop_drawing(event):
        canvas.unbind("<Motion>")

    def drawing(event):
        col, row = event.x // 20, event.y // 20
        if 0 <= row < height and 0 <= col < width:
            maze[row][col] = 1 if event.state & 0x0100 else 0
            color = "black" if maze[row][col] == 1 else "white"
            canvas.itemconfig(cells[row][col], fill=color)

    def save_to_file():
        nonlocal maze
        filename = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("Torch Files", "*.pt")])
        if filename:
            maze_tensor = torch.tensor(maze, dtype=torch.float32)
            save_maze(maze_tensor, filename)

    def upload_to_server():
        nonlocal maze
        name = maze_name_entry.get()
        if not name:
            print("Please enter a name for the maze before uploading.")
            return
        response = requests.post("http://127.0.0.1:5000/design-maze", json={
            "name": name,
            "grid": maze
        })
        print(response.json().get("message", "Upload failed!"))

    def print_to_terminal():
        nonlocal maze
        print("Maze:")
        for row in maze:
            print(row)

    def reset_grid():
        nonlocal maze
        maze = [[0 for _ in range(width)] for _ in range(height)]
        for row in range(height):
            for col in range(width):
                canvas.itemconfig(cells[row][col], fill="white")

    root = tk.Tk()
    root.title("Maze Designer")

    maze_name_label = tk.Label(root, text="Maze Name:")
    maze_name_label.pack()
    maze_name_entry = tk.Entry(root)
    maze_name_entry.pack()

    canvas = tk.Canvas(root, width=width * 20, height=height * 20, bg="white")
    canvas.pack()

    cells = []
    for row in range(height):
        cell_row = []
        for col in range(width):
            x1, y1 = col * 20, row * 20
            x2, y2 = x1 + 20, y1 + 20
            rect = canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="gray")
            canvas.tag_bind(rect, "<Button-1>", lambda e, r=row, c=col: toggle_cell(r, c))
            cell_row.append(rect)
        cells.append(cell_row)

    canvas.bind("<Button-1>", start_drawing)
    canvas.bind("<ButtonRelease-1>", stop_drawing)

    save_button = tk.Button(root, text="Save Maze", command=save_to_file)
    save_button.pack()

    upload_button = tk.Button(root, text="Upload Maze to Server", command=upload_to_server)
    upload_button.pack()

    print_button = tk.Button(root, text="Print Maze", command=print_to_terminal)
    print_button.pack()

    reset_button = tk.Button(root, text="Reset Grid", command=reset_grid)
    reset_button.pack()

    root.mainloop()

if __name__ == "__main__":
    create_gui(width=15, height=15)
