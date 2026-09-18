import ROOT
import numpy as np


def set_gradient_style():
    """
    Apply custom gradient style for 2D plots.
    Safe to call once before drawing histograms.
    """

    # Avoid scientific notation clutter
    ROOT.TGaxis.SetMaxDigits(3)

    NRGBs = 5
    NCont = 255

    stops = np.array([0.00, 0.34, 0.61, 0.84, 1.00], dtype='float64')
    red   = np.array([0.00, 0.00, 0.87, 1.00, 0.51], dtype='float64')
    green = np.array([0.00, 0.81, 1.00, 0.20, 0.00], dtype='float64')
    blue  = np.array([0.51, 1.00, 0.12, 0.00, 0.00], dtype='float64')

    ROOT.TColor.CreateGradientColorTable(
        NRGBs,
        stops,
        red,
        green,
        blue,
        NCont
    )

    ROOT.gStyle.SetNumberContours(NCont)