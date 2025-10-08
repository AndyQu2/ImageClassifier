import main_frame
from main_frame import MainFrame

class MainWindow(MainFrame):
    def __init__(self, parent=None):
        main_frame.MainFrame.__init__(self, parent)

    def on_loadfile_button_click(self, event):
        return None

    def on_classify_button_click(self, event):
        return None