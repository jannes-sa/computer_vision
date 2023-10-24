import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageDraw


class PolygonDrawer:
    def __init__(self, master):
        self.master = master
        self.points = []
        self.canvas = tk.Canvas(master)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)
        self.canvas.bind("<Button-1>", self.on_click)

        self.controls_frame = tk.Frame(master)
        self.controls_frame.pack()

        self.load_button = tk.Button(self.controls_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(self.controls_frame, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.LEFT)

        self.image = None
        self.original_image = None
        self.draw = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if not file_path:
            return
        self.original_image = Image.open(file_path)
        self.reset_image()

    def reset_image(self):
        self.image = ImageTk.PhotoImage(self.original_image.copy())
        self.draw = ImageDraw.Draw(self.original_image)
        self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        self.points = []

    def on_click(self, event):
        x, y = event.x, event.y
        if len(self.points) > 2 and self.is_near_starting_point((x, y)):
            self.close_polygon(event)
        else:
            self.points.append((x, y))
            self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='red', width=0)
            if len(self.points) > 1:
                self.canvas.create_line(self.points[-2], self.points[-1], fill='red', width=2)

    def is_near_starting_point(self, point):
        start_x, start_y = self.points[0]
        distance = ((start_x - point[0]) ** 2 + (start_y - point[1]) ** 2) ** 0.5
        return distance < 10

    def close_polygon(self, event):
        if len(self.points) > 2:
            # Draw the polygon outline on the canvas (without the fill)
            self.canvas.create_polygon(self.points, outline='red')

            # Create an empty image (same size as the original) with a black (0) alpha channel
            mask_image = Image.new('RGBA', self.original_image.size, (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_image)

            # Fill the polygon on this new image with green color and an alpha value of 68
            mask_draw.polygon(self.points, fill=(0, 255, 0, 68))

            # Composite the original image with this transparent green filled mask
            composite_image = Image.alpha_composite(self.original_image.convert('RGBA'), mask_image)
            self.original_image = composite_image.convert('RGB')

            # Update the displayed image on the canvas
            self.image = ImageTk.PhotoImage(self.original_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

            # Print the points array
            print(self.points)

            self.points = []

    def clear_canvas(self):
        self.canvas.delete("all")
        self.reset_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = PolygonDrawer(root)
    root.mainloop()
