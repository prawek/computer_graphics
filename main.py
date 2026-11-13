import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from algorithms.demos import DEMOS, DEMO_BY_KEY, render_demo


class DemoApp:
    def __init__(self, root, initial="line", image_path=None):
        self.root = root
        self.source = None
        self.source_name = "Встроенный пример"
        self.current = None
        self.result = None
        self.inputs = {}
        self.seed = None
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.generation = 0
        self.closed = False
        self.resize_job = None
        self.photo_images = [None, None]
        self.image_layouts = [None, None]
        root.title("Компьютерная графика · Алгоритмы")
        root.geometry("1180x800")
        root.minsize(900, 630)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda event: self.close())
        root.bind("<Control-o>", lambda event: self.open_image())
        root.bind("<Return>", lambda event: self.calculate())
        self._build()
        if image_path:
            self._load_image(image_path)
        self.tree.selection_set(initial)
        self.tree.see(initial)
        self.select_demo()

    def _build(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)
        sidebar = ttk.Frame(self.root, padding=(12, 12, 8, 12))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.rowconfigure(1, weight=1)
        ttk.Label(sidebar, text="АЛГОРИТМЫ", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.tree = ttk.Treeview(sidebar, show="tree", selectmode="browse", height=24)
        self.tree.column("#0", width=250, stretch=False)
        self.tree.grid(row=1, column=0, sticky="ns")
        scrollbar = ttk.Scrollbar(sidebar, command=self.tree.yview)
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.configure(yscrollcommand=scrollbar.set)
        groups = {}
        for demo in DEMOS:
            if demo.group not in groups:
                groups[demo.group] = self.tree.insert("", "end", text=demo.group, open=True)
            self.tree.insert(groups[demo.group], "end", iid=demo.key, text=demo.title)
        self.tree.bind("<<TreeviewSelect>>", lambda event: self.select_demo())
        ttk.Label(sidebar, text="Enter — пересчитать\nCtrl+O — открыть изображение\nEsc — закрыть", foreground="#596579").grid(row=2, column=0, sticky="w", pady=(12, 0))

        content = ttk.Frame(self.root, padding=(8, 12, 14, 12))
        content.grid(row=0, column=1, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.rowconfigure(4, weight=1)
        self.title = ttk.Label(content, font=("Segoe UI", 18, "bold"))
        self.title.grid(row=0, column=0, sticky="w")
        self.description = ttk.Label(content, wraplength=780, justify="left")
        self.description.grid(row=1, column=0, sticky="ew", pady=(6, 12))
        toolbar = ttk.Frame(content)
        toolbar.grid(row=2, column=0, sticky="ew")
        self.open_button = ttk.Button(toolbar, text="Открыть изображение…", command=self.open_image)
        self.open_button.pack(side="left")
        ttk.Button(toolbar, text="Встроенный пример", command=self.reset_example).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Описание реализации", command=self.show_documentation).pack(side="left")
        self.parameter_frame = ttk.LabelFrame(content, text="Параметры", padding=10)
        self.parameter_frame.grid(row=3, column=0, sticky="ew", pady=12)

        self.image_frame = ttk.Frame(content)
        self.image_frame.grid(row=4, column=0, sticky="nsew")
        self.image_frame.columnconfigure((0, 1), weight=1, uniform="images")
        self.image_frame.rowconfigure(1, weight=1)
        ttk.Label(self.image_frame, text="ИСХОДНИК / УСЛОВИЕ", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
        ttk.Label(self.image_frame, text="РЕЗУЛЬТАТ", font=("Segoe UI", 9, "bold")).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 6))
        self.canvases = []
        for i in (0, 1):
            canvas = tk.Canvas(self.image_frame, background="#141b2a", highlightthickness=0, width=340, height=300)
            canvas.grid(row=1, column=i, sticky="nsew", padx=(0, 5) if i == 0 else (5, 0))
            canvas.bind("<Configure>", self.schedule_resize)
            self.canvases.append(canvas)
        self.canvases[0].bind("<Button-1>", self.pick_seed)
        self.status = ttk.Label(content, wraplength=780, justify="left", foreground="#385575")
        self.status.grid(row=5, column=0, sticky="ew", pady=(10, 4))
        self.details = ttk.Label(content, wraplength=780, justify="left")
        self.details.grid(row=6, column=0, sticky="ew")
        content.bind("<Configure>", lambda event: self._wrap(event.width))

    def _wrap(self, width):
        for label in (self.description, self.status, self.details):
            label.configure(wraplength=max(300, width - 20))

    def select_demo(self):
        selection = self.tree.selection()
        if not selection or selection[0] not in DEMO_BY_KEY:
            return
        key = selection[0]
        if key == self.current:
            return
        self.current, self.seed = key, None
        demo = DEMO_BY_KEY[key]
        self.title.configure(text=demo.title)
        self.description.configure(text=demo.description)
        self.open_button.configure(state="normal" if demo.accepts_image else "disabled")
        for child in self.parameter_frame.winfo_children():
            child.destroy()
        self.inputs = {}
        for i, parameter in enumerate(demo.parameters):
            frame = ttk.Frame(self.parameter_frame)
            frame.grid(row=0, column=i, sticky="w", padx=(0, 14))
            ttk.Label(frame, text=parameter.label).pack(anchor="w")
            value = tk.StringVar(value=str(parameter.default))
            self.inputs[parameter.key] = value
            if isinstance(parameter.default, str):
                ttk.Entry(frame, textvariable=value, width=20).pack(anchor="w", pady=(3, 0))
            else:
                ttk.Spinbox(frame, textvariable=value, from_=parameter.minimum, to=parameter.maximum,
                            increment=parameter.step, width=12, command=self.calculate).pack(anchor="w", pady=(3, 0))
        if not demo.parameters:
            ttk.Label(self.parameter_frame, text="Готовая сцена для этого алгоритма").grid(row=0, column=0, sticky="w")
        ttk.Button(self.parameter_frame, text="Пересчитать", command=self.calculate).grid(row=1, column=0, columnspan=4, sticky="w", pady=(8, 0))
        self.calculate()

    def _parameters(self):
        parameters = {}
        for parameter in DEMO_BY_KEY[self.current].parameters:
            text = self.inputs[parameter.key].get().strip()
            if isinstance(parameter.default, str):
                value = text
            else:
                value = float(text.replace(",", "."))
                if not parameter.minimum <= value <= parameter.maximum:
                    raise ValueError(f"{parameter.label}: диапазон {parameter.minimum}…{parameter.maximum}")
                if isinstance(parameter.default, int):
                    if not value.is_integer():
                        raise ValueError(f"{parameter.label}: требуется целое число")
                    value = int(value)
            parameters[parameter.key] = value
        if self.seed is not None:
            parameters["seed"] = self.seed
        return parameters

    def calculate(self):
        if self.current is None or self.closed:
            return
        self.generation += 1
        token = self.generation
        if self.future:
            self.future.cancel()
        try:
            parameters = self._parameters()
        except ValueError as error:
            self.status.configure(text=f"Ошибка параметров: {error}", foreground="#b33939")
            return
        self.status.configure(text="Вычисление…", foreground="#385575")
        source = self.source if DEMO_BY_KEY[self.current].accepts_image else None
        self.future = self.executor.submit(render_demo, self.current, parameters, source)
        future = self.future
        self.root.after(40, lambda: self._poll(token, future))

    def _poll(self, token, future):
        if self.closed or token != self.generation:
            return
        if not future.done():
            self.root.after(40, lambda: self._poll(token, future))
            return
        try:
            self.result = future.result()
        except Exception as error:
            self.result = None
            self.repaint()
            self.status.configure(text=f"Ошибка: {error}", foreground="#b33939")
            self.details.configure(text="Исправьте параметры или выберите другое изображение.")
            return
        self.repaint()
        source_name = self.source_name if self.source is not None and DEMO_BY_KEY[self.current].accepts_image else "Встроенный пример"
        size = self.result.processed.size
        self.status.configure(text=f"{source_name} · результат {size[0]}×{size[1]} · расчёт {self.result.elapsed_ms:.1f} мс", foreground="#385575")
        self.details.configure(text=self.result.info)

    def schedule_resize(self, event=None):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(80, self.repaint)

    def repaint(self):
        self.resize_job = None
        images = (self.result.original, self.result.processed) if self.result else (None, None)
        for i, (canvas, source) in enumerate(zip(self.canvases, images)):
            canvas.delete("all")
            self.image_layouts[i] = None
            if source is None:
                continue
            width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
            scale = min(width / source.width, height / source.height)
            size = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
            resample = Image.Resampling.NEAREST if self.current in ("line", "circle") else Image.Resampling.LANCZOS
            photo = ImageTk.PhotoImage(source.resize(size, resample))
            self.photo_images[i] = photo
            x, y = (width - size[0]) // 2, (height - size[1]) // 2
            canvas.create_image(x, y, image=photo, anchor="nw")
            self.image_layouts[i] = (x, y, size[0] / source.width, size[1] / source.height)

    def pick_seed(self, event):
        if self.current not in ("flood", "span") or not self.result or not self.image_layouts[0]:
            return
        x, y, sx, sy = self.image_layouts[0]
        px, py = int((event.x - x) / sx), int((event.y - y) / sy)
        if event.x >= x and event.y >= y and 0 <= px < self.result.original.width and 0 <= py < self.result.original.height:
            self.seed = px, py
            self.calculate()

    def _load_image(self, path):
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            original_size = image.size
            image.thumbnail((1280, 960), Image.Resampling.LANCZOS)
            self.source = image.copy()
        self.source_name = Path(path).name
        if original_size != self.source.size:
            self.source_name += f" (уменьшено до {self.source.width}×{self.source.height})"

    def open_image(self):
        if self.current is None or not DEMO_BY_KEY[self.current].accepts_image:
            return
        path = filedialog.askopenfilename(title="Изображение для обработки", filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff *.webp"), ("Все файлы", "*.*")])
        if path:
            try:
                self._load_image(path)
            except (OSError, ValueError) as error:
                messagebox.showerror("Не удалось открыть изображение", str(error), parent=self.root)
                return
            self.calculate()

    def reset_example(self):
        self.source, self.seed = None, None
        for parameter in DEMO_BY_KEY[self.current].parameters:
            self.inputs[parameter.key].set(str(parameter.default))
        self.calculate()

    def show_documentation(self):
        path = Path(__file__).parent / "algorithms" / DEMO_BY_KEY[self.current].folder / "README.md"
        window = tk.Toplevel(self.root)
        window.title(f"Реализация · {DEMO_BY_KEY[self.current].title}")
        window.geometry("780x560")
        text = tk.Text(window, wrap="word", padx=18, pady=14, font=("Segoe UI", 11))
        scrollbar = ttk.Scrollbar(window, command=text.yview)
        scrollbar.pack(side="right", fill="y")
        text.configure(yscrollcommand=scrollbar.set)
        text.pack(fill="both", expand=True)
        text.insert("1.0", path.read_text(encoding="utf-8"))
        text.configure(state="disabled")

    def close(self):
        self.closed = True
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main(initial=None):
    parser = argparse.ArgumentParser(description="Оконная демонстрация алгоритмов компьютерной графики")
    parser.add_argument("--demo", choices=DEMO_BY_KEY, default=initial or "line", help="Открыть выбранный алгоритм")
    parser.add_argument("--image", help="Изображение для фильтров, морфологии или чтения EAN-13")
    parser.add_argument("--list", action="store_true", help="Показать доступные демонстрации")
    arguments = parser.parse_args()
    if arguments.list:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        for demo in DEMOS:
            print(f"{demo.key:16} {demo.group}: {demo.title}")
        return
    root = tk.Tk()
    try:
        DemoApp(root, arguments.demo, arguments.image)
    except (OSError, ValueError) as error:
        messagebox.showerror("Ошибка запуска", str(error), parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
