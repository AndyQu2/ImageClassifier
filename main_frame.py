import wx
import wx.xrc

class MainFrame(wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title="Image Classifier",
                          pos=wx.DefaultPosition, size=wx.Size(700, 500),
                          style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.confident_box = None
        self.result_box = None
        self.classify_button = None
        self.loadfile_button = None
        self.filepath_box = None
        self.images_widget = None
        self.SetSizeHints(wx.Size(600, 400), wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_WINDOW))

        main_sizer = wx.BoxSizer(wx.VERTICAL)

        title_sizer = self.create_title_sizer()
        main_sizer.Add(title_sizer, 0, wx.EXPAND | wx.ALL, 5)

        image_sizer = self.create_image_sizer()
        main_sizer.Add(image_sizer, 1, wx.EXPAND | wx.ALL, 5)

        file_sizer = self.create_file_control_sizer()
        main_sizer.Add(file_sizer, 0, wx.EXPAND | wx.ALL, 5)

        result_sizer = self.create_result_sizer()
        main_sizer.Add(result_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.CreateStatusBar()
        self.SetStatusText("Ready to load and classify images...")

        self.SetSizer(main_sizer)
        self.Layout()
        self.Centre(wx.BOTH)

        self.loadfile_button.Bind(wx.EVT_BUTTON, self.on_loadfile_button_click)
        self.classify_button.Bind(wx.EVT_BUTTON, self.on_classify_button_click)

        self.classify_button.Disable()

    def create_title_sizer(self):
        title_sizer = wx.BoxSizer(wx.HORIZONTAL)
        title_text = wx.StaticText(self, wx.ID_ANY, "Image Classification Tool")
        title_font = wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD)
        title_text.SetFont(title_font)
        title_text.SetForegroundColour(wx.Colour(0, 0, 128))
        title_sizer.Add(title_text, 0, wx.ALL | wx.ALIGN_CENTER, 5)
        return title_sizer

    def create_image_sizer(self):
        image_sizer = wx.BoxSizer(wx.VERTICAL)

        image_label = wx.StaticText(self, wx.ID_ANY, "Preview:")
        image_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        image_sizer.Add(image_label, 0, wx.LEFT | wx.TOP, 5)

        image_panel = wx.Panel(self, wx.ID_ANY, style=wx.SIMPLE_BORDER)
        image_panel.SetBackgroundColour(wx.Colour(240, 240, 240))
        image_panel_sizer = wx.BoxSizer(wx.VERTICAL)

        self.images_widget = wx.StaticBitmap(image_panel, wx.ID_ANY, wx.NullBitmap,
                                             wx.DefaultPosition, wx.Size(300, 200))
        image_panel_sizer.Add(self.images_widget, 1, wx.ALL | wx.ALIGN_CENTER, 10)
        image_panel.SetSizer(image_panel_sizer)

        image_sizer.Add(image_panel, 1, wx.EXPAND | wx.ALL, 5)
        return image_sizer

    def create_file_control_sizer(self):
        file_sizer = wx.BoxSizer(wx.HORIZONTAL)

        file_label = wx.StaticText(self, wx.ID_ANY, "File path:")
        file_sizer.Add(file_label, 0, wx.ALL | wx.ALIGN_CENTER_VERTICAL, 5)

        self.filepath_box = wx.TextCtrl(self, wx.ID_ANY, wx.EmptyString,
                                        wx.DefaultPosition, wx.DefaultSize, wx.TE_READONLY)
        file_sizer.Add(self.filepath_box, 1, wx.ALL | wx.EXPAND, 5)

        self.loadfile_button = wx.Button(self, wx.ID_ANY, "Load Image",
                                         wx.DefaultPosition, wx.DefaultSize, 0)
        self.loadfile_button.SetBackgroundColour(wx.Colour(220, 230, 255))
        file_sizer.Add(self.loadfile_button, 0, wx.ALL, 5)

        self.classify_button = wx.Button(self, wx.ID_ANY, "Classify Image",
                                         wx.DefaultPosition, wx.DefaultSize, 0)
        self.classify_button.SetBackgroundColour(wx.Colour(220, 255, 220))
        file_sizer.Add(self.classify_button, 0, wx.ALL, 5)

        return file_sizer

    def create_result_sizer(self):
        result_sizer = wx.BoxSizer(wx.VERTICAL)

        result_title = wx.StaticText(self, wx.ID_ANY, "Classification Results:")
        result_title.SetFont(wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
        result_sizer.Add(result_title, 0, wx.LEFT | wx.TOP, 5)

        result_panel = wx.Panel(self, wx.ID_ANY, style=wx.SIMPLE_BORDER)
        result_panel.SetBackgroundColour(wx.Colour(250, 250, 250))
        result_panel_sizer = wx.GridSizer(2, 2, 5, 15)

        result_label = wx.StaticText(result_panel, wx.ID_ANY, "Result:")
        result_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        result_panel_sizer.Add(result_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.result_box = wx.StaticText(result_panel, wx.ID_ANY, "None")
        result_panel_sizer.Add(self.result_box, 0, wx.ALIGN_CENTER_VERTICAL)

        confident_label = wx.StaticText(result_panel, wx.ID_ANY, "Confidence:")
        confident_label.SetFont(wx.Font(10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
        result_panel_sizer.Add(confident_label, 0, wx.ALIGN_CENTER_VERTICAL)

        self.confident_box = wx.StaticText(result_panel, wx.ID_ANY, "0.00%")
        result_panel_sizer.Add(self.confident_box, 0, wx.ALIGN_CENTER_VERTICAL)

        result_panel.SetSizer(result_panel_sizer)
        result_sizer.Add(result_panel, 0, wx.EXPAND | wx.ALL, 5)

        return result_sizer

    def on_loadfile_button_click(self, event):
        event.Skip()

    def on_classify_button_click(self, event):
        event.Skip()