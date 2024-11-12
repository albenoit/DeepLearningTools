from matplotlib.colors import Colormap
from matplotlib.widgets import Slider, Button, RangeSlider, TextBox
from matplotlib.ticker import MaxNLocator
from matplotlib import colormaps
from numexpr import evaluate as evaluate_number
import matplotlib.pyplot as plt
import numpy as np

#region GUI
class AbstractGUI:
    def __init__(self):
        self.fig = plt.figure()

    def update(self):
        raise NotImplementedError()
    
    def update_draw(self):
        self.update()
        self.fig.canvas.draw()


class Heatmap(AbstractGUI):
    def __init__(self, matrix:np.ndarray, linelabels:list[str]=None, vmin:float=None, vmax:float=None):
        """
        :param matrix: The 2D square matrix containing the values to plot.
        :param linelabels: The labels to use for the rows and columns. By default, uses row and column indexes.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        super().__init__()
        self.fig.set_size_inches(5, 5.8)

        self.matrix = matrix
        self.vmin = self.xmin = vmin or np.nanmin(self.matrix)
        self.vmax = self.xmax = vmax or np.nanmax(self.matrix)
        self.cmap = colormaps['magma_r']

        self.ax:plt.Axes = self.fig.add_axes([0.1, 0.3, 0.64, 0.5])
        self.ax_title = ""
        self.ax_xlabel = ""
        self.ax_ylabel = ""
        self.linelabels = linelabels

        self.colorbar:plt.Colorbar = None

        # Range slider
        self.ax_range_slide = self.fig.add_axes([0.25, 0.01, 0.5, 0.02])
        self.range_slider = RangeSlider(self.ax_range_slide, "Range ", valmin=self.vmin, valmax=self.vmax, valinit=(self.xmin, self.xmax))
        self.range_slider.on_changed(self.range_slider_clb)

        # Cmap textbox
        self.ax_cmap_tb = self.fig.add_axes([0.4, 0.05, 0.2, 0.05])
        self.cmap_tb = TextBox(self.ax_cmap_tb, "Colormap ", initial=self.cmap.name)
        self.cmap_tb.on_submit(self.cmap_tb_clb)

        # Normalize button
        self.ax_norm_btn = self.fig.add_axes([0.7, 0.12, 0.2, 0.05])
        self.norm_btn = Button(self.ax_norm_btn, "Normalize")
        self.norm_btn.on_clicked(self.norm_btn_clb)

        # Local bounds button
        self.ax_local_btn = self.fig.add_axes([0.7, 0.06, 0.2, 0.05])
        self.local_btn = Button(self.ax_local_btn, "Local bounds")
        self.local_btn.on_clicked(self.local_btn_clb)

    def update(self):
        if self.colorbar:
            self.colorbar.remove()
            self.colorbar = None

        self.ax.cla()
        self.ax.set_title(self.ax_title)
        self.ax.set_xlabel(self.ax_xlabel)
        self.ax.set_ylabel(self.ax_ylabel)

        ticks = np.arange(0, self.matrix.shape[0])+0.5
        self.ax.set_xticks(ticks)
        self.ax.set_yticks(ticks)

        if self.linelabels is not None: 
            self.ax.set_xticklabels(self.linelabels)
            self.ax.set_yticklabels(self.linelabels)
        else:
            linelabels = list(map(int,ticks-0.5))
            self.ax.set_xticklabels(linelabels)
            self.ax.set_yticklabels(linelabels)

        heatmap = self.ax.pcolor(self.matrix, cmap=self.cmap, vmin=self.xmin, vmax=self.xmax)

        colorbar_ax = self.fig.add_axes([0.78, 0.3, 0.029, 0.5])
        self.colorbar = self.fig.colorbar(heatmap, cax=colorbar_ax)
        self.colorbar.set_ticks(np.linspace(self.xmin, self.xmax, 10, True))

    def range_update(self, vmin:float, vmax:float):
        self.range_slider.set_min(vmin)
        self.range_slider.set_max(vmax)

    def range_slider_clb(self, val:tuple[float, float]):
        self.xmin, self.xmax = val
        self.update_draw()

    def norm_btn_clb(self, e):
        self.range_update(self.vmin, self.vmax)

    def local_btn_clb(self, e):
        self.range_update(np.nanmin(self.matrix), np.nanmax(self.matrix))

    def cmap_tb_clb(self, val):
        if val in colormaps:
            self.cmap = colormaps[val]
            self.update_draw()


class RoundHeatmap(Heatmap):
    def __init__(self, matrices:np.ndarray, linelabels:list[str], vmin:float=None, vmax:float=None):
        """
        :param matrices: A list of 2D square matrices containing the values to plot.
        :param linelabels: The labels to use for the rows and columns. By default, uses row and column indexes.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        self.matrices = matrices
        self.round = 1
        vmin = vmin or np.nanmin(self.matrices)
        vmax = vmax or np.nanmax(self.matrices)
        super().__init__(self.get_current_matrix(), linelabels, vmin=vmin, vmax=vmax)

        self.ax_round_slide = self.fig.add_axes([0.25, 0.9, 0.5, 0.02])
        self.round_slider = Slider(self.ax_round_slide, "Round ", valmin=1, valmax=len(matrices), valinit=1, valfmt=" %d", valstep=1)
        self.round_slider.on_changed(self.round_slider_clb)

    def get_current_matrix(self):
        return self.matrices[self.round-1]
    
    def update(self):
        self.matrix = self.get_current_matrix()
        super().update()
    
    def round_slider_clb(self, val):
        self.round = int(val)
        self.update_draw()


class AggregationHeatmap(Heatmap):
    def __init__(self, matrices:list[np.ndarray], linelabels:list[str]=None, vmin:float=None, vmax:float=None):
        """
        :param matrices: A list of 2D square matrices to aggregate and plot.
        :param linelabels: The labels to use for the rows and columns. By default, uses row and column indexes.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        matrix = np.mean(matrices, axis=0)
        super().__init__(matrix, linelabels, vmin, vmax)


class RoundAggregationHeatmap(RoundHeatmap):
    def __init__(self, matrices_groups:list[np.ndarray], linelabels:list[str], vmin:float=None, vmax:float=None):
        """
        :param matrices_groups: A list of groups of 2D square matrices to aggregate and plot.
        :param linelabels: The labels to use for the rows and columns. By default, uses row and column indexes.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        matrices = np.mean(matrices_groups, axis=1) # (0:rounds, 1:expes, 2:width, 3:height)
        super().__init__(matrices, linelabels, vmin, vmax)


class AbstractHistogram(AbstractGUI):
    def __init__(self, vmin:float=None, vmax:float=None):
        """
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        super().__init__()
        self.fig.set_size_inches(5, 5.5)

        self.vmin = self.xmin = vmin or self.get_current_xmin()
        self.vmax = self.xmax = vmax or self.get_current_xmax()
        self.bins = 10
        self.midline = 0

        self.ax:plt.Axes = self.fig.add_axes([0.1, 0.3, 0.8, 0.5])
        self.ax_title = ""
        self.ax_xlabel = ""
        self.ax_ylabel = ""

        # Range slider
        self.ax_range_slide = self.fig.add_axes([0.25, 0.01, 0.5, 0.02])
        self.range_slider = RangeSlider(self.ax_range_slide, "Range ", valmin=self.vmin, valmax=self.vmax, valinit=(self.xmin, self.xmax))
        self.range_slider.on_changed(self.range_slider_clb)
        
        # Bins textbox
        self.ax_bins_tb = self.fig.add_axes([0.4, 0.05, 0.2, 0.05])
        self.bins_tb = TextBox(self.ax_bins_tb, "Bins ", initial=str(self.bins))
        self.bins_tb.on_submit(self.bins_tb_clb)

        # Parts textbox
        self.ax_midline_tb = self.fig.add_axes([0.4, 0.105, 0.2, 0.05])
        self.midline_tb = TextBox(self.ax_midline_tb, "Middle line ", initial=str(self.midline))
        self.midline_tb.on_submit(self.midline_tb_clb)

        # Normalize button
        self.ax_norm_btn = self.fig.add_axes([0.7, 0.12, 0.2, 0.05])
        self.norm_btn = Button(self.ax_norm_btn, "Normalize")
        self.norm_btn.on_clicked(self.norm_btn_clb)

        # Local bounds button
        self.ax_local_btn = self.fig.add_axes([0.7, 0.06, 0.2, 0.05])
        self.local_btn = Button(self.ax_local_btn, "Local bounds")
        self.local_btn.on_clicked(self.local_btn_clb)

    def get_current_xmin(self):
        raise NotImplementedError()
    
    def get_current_xmax(self):
        raise NotImplementedError()
    
    def update(self):
        self.ax.cla()
        self.ax.set_title(self.ax_title)
        self.ax.set_xlabel(self.ax_xlabel)
        self.ax.set_ylabel(self.ax_ylabel)
        self.ax.yaxis.grid(color='gray', linestyle='dashed')
        self.ax.set_axisbelow(True)

        X,Y = self.draw_bars()
        self.draw_middle_line(X, Y)

        self.ax.set_xticks(X)
        fontsize = 'xx-small' if self.bins > 40 else ('x-small' if self.bins > 20 else 'small')
        for label in self.ax.get_xticklabels():
            label.set_fontsize(fontsize)
            label.set_rotation(90)

        self.ax.set_ylim(bottom=0)
        self.ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        self.bins_tb.set_val(str(self.bins))
        self.midline_tb.set_val(str(self.midline))

    def draw_bars(self):
        raise NotImplementedError()
    
    def draw_middle_line(self, X:np.ndarray, Y:np.ndarray):
        # Can be simplified with access to the distance matrix
        if not (0 < self.midline < 1): return

        total = 0
        threshold = Y.sum()*self.midline
        if not threshold: return

        for i,y in enumerate(Y):
            new_total = total + y
            if new_total >= threshold:
                ilerp = (threshold-total)/y
                x = X[i] + (X[i+1]-X[i]) * ilerp
                self.ax.axvline(x, color="red", ls="--")
                return
            total = new_total

    def range_update(self, vmin:float, vmax:float):
        self.range_slider.set_min(vmin)
        self.range_slider.set_max(vmax)

    def range_slider_clb(self, val:tuple[float, float]):
        self.xmin, self.xmax = val
        self.update_draw()

    def bins_tb_clb(self, val:str):
        try: self.bins = min(max(int(val), 2), 100)
        except: return
        self.update_draw()

    def midline_tb_clb(self, val:str):
        try: self.midline = min(max(evaluate_number(val), 0), 1)
        except: return
        self.update_draw()

    def norm_btn_clb(self, e):
        self.range_update(self.vmin, self.vmax)

    def local_btn_clb(self, e):
        self.range_update(self.get_current_xmin(), self.get_current_xmax())


class Histogram(AbstractHistogram):
    def __init__(self, matrix:np.ndarray, vmin:float=None, vmax:float=None):
        """
        :param matrix: The 2D matrix containing the values to use in the histogram.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        self.matrix = matrix
        super().__init__(vmin, vmax)

    def get_current_xmin(self):
        return np.nanmin(self.matrix)
    
    def get_current_xmax(self):
        return np.nanmax(self.matrix)
    
    def draw_bars(self):
        Y,X = np.histogram(self.matrix, bins=self.bins, range=(self.xmin, self.xmax))
        width = X[1] - X[0]
        self.ax.bar(X[1:]-width/2, Y, width=width)
        self.ax.plot(X[1:]-width/2, Y, marker="o", color="red", ls="")
        return X,Y


class RoundHistogram(Histogram):
    def __init__(self, matrices:list[np.ndarray], vmin:float=None, vmax:float=None):
        """
        :param matrices: A list of 2D matrices containing the values to use in the histogram.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        self.matrices = matrices
        self.round = 1
        vmin = vmin or np.nanmin(self.matrices)
        vmax = vmax or np.nanmax(self.matrices)
        super().__init__(self.get_current_matrix(), vmin=vmin, vmax=vmax)

        self.ax_round_slide = self.fig.add_axes([0.25, 0.9, 0.5, 0.02])
        self.round_slider = Slider(self.ax_round_slide, "Round ", valmin=1, valmax=self.get_rounds(), valinit=1, valfmt=" %d", valstep=1)
        self.round_slider.on_changed(self.round_slider_clb)

    def get_rounds(self) -> int:
        return len(self.matrices)
    
    def get_current_matrix(self):
        return self.matrices[self.round-1]
    
    def round_slider_clb(self, val):
        self.round = val
        self.update_draw()
    
    def update(self):
        self.matrix = self.get_current_matrix()
        super().update()


class AggregationHistogram(AbstractHistogram):
    def __init__(self, matrices:list[np.ndarray], vmin:float=None, vmax:float=None):
        """
        :param matrices: A list of 2D matrices to aggregate and use for the histogram.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        self.matrices = matrices
        super().__init__(vmin, vmax)

    def get_current_xmin(self):
        return np.nanmin(self.matrices)
    
    def get_current_xmax(self):
        return np.nanmax(self.matrices)

    def draw_bars(self):
        histos = []
        X = None
        for matrix in self.matrices:
            Y,x = np.histogram(matrix, bins=self.bins, range=(self.xmin, self.xmax))
            histos.append(Y)
            if X is None: X = x

        Y = np.mean(histos, axis=0)
        E = np.std(histos, axis=0)

        width = X[1] - X[0]
        self.ax.bar(X[1:]-width/2, Y, width=width)
        self.ax.errorbar(X[1:]-width/2, Y, yerr=E, ls="", color="red", marker="o", ecolor="black", capsize=3)

        return X,Y
    

class RoundAggregationHistogram(AggregationHistogram):
    def __init__(self, matrices_groups:list[list[np.ndarray]], vmin:float=None, vmax:float=None):
        """
        :param matrices_groups: A list of groups of 2D matrices to aggregate and use for the histogram.
        :param vmin: The minimum value of the displayed range.
        :param vmax: The maximum value of the displayed range.
        """
        self.matrices_groups = matrices_groups
        self.round = 1
        vmin = vmin or np.nanmin(matrices_groups)
        vmax = vmax or np.nanmax(matrices_groups)
        super().__init__(self.get_current_matrices(), vmin, vmax)

        self.ax_round_slide = self.fig.add_axes([0.25, 0.9, 0.5, 0.02])
        self.round_slider = Slider(self.ax_round_slide, "Round ", valmin=1, valmax=self.get_rounds(), valinit=1, valfmt=" %d", valstep=1)
        self.round_slider.on_changed(self.round_slider_clb)

    def get_rounds(self) -> int:
        return len(self.matrices_groups)
    
    def get_current_matrices(self):
        return self.matrices_groups[self.round-1]
    
    def round_slider_clb(self, val):
        self.round = val
        self.update_draw()
    
    def update(self):
        self.matrices = self.get_current_matrices()
        super().update()

#endregion

#region Other utils
def find_colorbar_axes(fig:plt.Figure) -> plt.Axes | None:
    for ax in fig.get_axes():
        if ax.get_label() == "<colorbar>":
            return ax
    return None

def get_cmap_from_arg(cmap):
    if isinstance(cmap, Colormap): return cmap
    return colormaps[cmap or 'viridis']
#endregion

if __name__ == "__main__":
    matrices_group = [[np.random.normal(0, 1, (20, 20)) * (0.5 + 0.5 * np.random.random((1))[0]) for _ in range(5)] for __ in range(10)]
    graph = RoundAggregationHistogram(matrices_group)
    graph.fig.suptitle("Interclient similarity histogram")
    graph.ax_title = "metric:test"
    graph.ax_xlabel = graph.ax_ylabel = "Models"
    graph.update()
    plt.show()