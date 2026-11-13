import time
import tkinter as tk
from types import SimpleNamespace

from PIL import Image

from algorithms.demos import DEMOS
from main import DemoApp


def run():
    root = tk.Tk()
    errors = []
    root.report_callback_exception = lambda *error: errors.append(error)
    app = DemoApp(root)

    def wait_for_result(expect_error=False):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            root.update()
            text = app.status.cget("text")
            if text != "Вычисление…":
                if expect_error:
                    assert text.startswith("Ошибка"), text
                else:
                    assert not text.startswith("Ошибка"), text
                    assert app.result is not None
                    assert all(app.photo_images)
                assert not errors, errors
                return
            time.sleep(0.01)
        raise AssertionError(f"Расчёт {app.current} не завершился")

    def select(key):
        app.tree.selection_set(key)
        app.select_demo()
        wait_for_result()

    try:
        wait_for_result()
        root.withdraw()
        for demo in DEMOS:
            select(demo.key)
        select("gaussian")
        app.inputs["kernel"].set("4")
        app.calculate()
        wait_for_result(expect_error=True)
        app.reset_example()
        wait_for_result()
        app.source = Image.new("RGB", (73, 61), (90, 120, 150))
        app.source_name = "GUI smoke source"
        app.calculate()
        wait_for_result()
        assert app.result.processed.size == (73, 61)
        app.reset_example()
        wait_for_result()
        select("flood")
        app.repaint()
        x, y, sx, sy = app.image_layouts[0]
        app.pick_seed(SimpleNamespace(x=x + 200 * sx, y=y + 200 * sy))
        wait_for_result()
        assert app.seed == (200, 200), app.seed
        app.show_documentation()
        root.update()
        documents = [child for child in root.winfo_children() if isinstance(child, tk.Toplevel)]
        assert len(documents) == 1
        documents[0].destroy()

        app.tree.selection_set("phong")
        app.select_demo()
        app.tree.selection_set("binary")
        app.select_demo()
        wait_for_result()
        assert app.current == "binary"
        assert app.result.processed.mode == "L"
        assert not errors, errors
        print(f"GUI smoke OK: {len(DEMOS)} scenes, parameters, custom source, seed, docs, rapid switching")
    finally:
        app.close()


if __name__ == "__main__":
    run()
