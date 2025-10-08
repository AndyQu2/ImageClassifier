import wx
from main_window import MainWindow

app = wx.App(False)
frame = MainWindow(None)
frame.Show(True)
app.MainLoop()