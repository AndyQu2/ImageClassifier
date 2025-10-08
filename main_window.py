import wx
import main_frame
from main_frame import MainFrame

class MainWindow(MainFrame):
    def __init__(self, parent=None):
        main_frame.MainFrame.__init__(self, parent)

    def on_loadfile_button_click(self, event):
        with wx.FileDialog(self, "Choose an image",
                           wildcard="Images (*.png *.jpg *.jpeg) | *.png; *.jpg; *.jpeg",
                           style=wx.FD_OPEN | wx.FD_CHANGE_DIR) as dlg:
            if dlg.ShowModal() == wx.ID_OK:
                file_path = dlg.GetPath()
                self.filepath_box.SetValue(file_path)

                image = wx.Image(file_path)

                width, height = self.images_widget.GetSize()
                image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
                bitmap = wx.Bitmap(image)
                self.images_widget.SetBitmap(bitmap)

                self.classify_button.Enable(True)
                self.SetStatusText(f"Loaded image: {file_path}")
        return None

    def on_classify_button_click(self, event):
        return None