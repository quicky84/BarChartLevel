"""
Visualization as discussed in
Ferreira, N., Fisher, D., & Konig, A. C. (2014, April).
Sample-oriented task-driven visualizations: allowing users to make better, more confident decisions.
In Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (pp. 571-580). ACM
"""
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st

class BarEx(object):
    """
    BarChart with draggable level
    """

    epsilon = 20  # max pixel distance to count as a vertex hit

    def __init__(self, axes, mybars, canvas):
        # bars = {'names': ( strings ), 'ci95': [(2x float)]}
        self.axes = axes
        self.canvas = canvas

        self.axes.set_frame_on(False)
        self.axes.yaxis.tick_right()

        self.bars_data = mybars
        self.axes.set_xticks(np.arange(len(self.bars_data['names'])))
        self.axes.set_xticklabels(self.bars_data['names'])

        self.base = Line2D([-1.5, len(self.bars_data['names'])+2], [self.bars_data['ci95'][0][0]]*2,
                           marker='o', markersize=10, markerfacecolor='r', color='y')

        self.base_dragged = False
        self.axes.add_artist(self.base)

        self.annotate(0)
        self.boxes(0)

        canvas.mpl_connect('button_press_event', self.button_press)
        canvas.mpl_connect('button_release_event', self.button_release)
        canvas.mpl_connect('motion_notify_event', self.motion_notify)

    def boxes(self, run = 1):
        # value to judge closeness to the means
        v = self.base.get_ydata()[0]

        pos = np.arange(len(self.bars_data['names']))
        means = [np.mean(x) for x in self.bars_data['ci95']]
        err = [np.abs(s1-s0) for s0, s1 in self.bars_data['ci95']]
        reds = [1 - np.min([np.abs(m-v), e])/e for m, e in zip(means, err)]
        colors = [(r, 0, 1-r) for r in reds]

        if run == 0:
            self.bars = self.axes.bar(pos, means, color=colors, width=0.5, align='center', zorder=-1, yerr=err)
        else:
            for b, c in zip(self.bars, colors):
                b.set_color(c)

    def annotate(self, run = 1):
        v = self.base.get_ydata()[0]
        if run == 0:
            self.text = self.axes.annotate('{:.2f}'.format(v), (-1.4, v))
        else:
            self.text.set_y(v)
            self.text.set_text('{:.2f}'.format(v))

    def draw(self):
        self.annotate()
        self.base.figure.canvas.draw()
        self.boxes()


    def is_clicked(self, event):
        'get the index of the vertex under point if within epsilon tolerance'
        # display coords
        xy = np.asarray(self.base.get_xydata())
        xyt = self.base.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        return np.any(np.sqrt((xt - event.x)**2 + (yt - event.y)**2) < 2)

    def button_press(self, event):
        'whenever a mouse button is pressed'
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        self.base_dragged = self.is_clicked(event)

    def motion_notify(self, event):
        'on mouse movement'
        if not self.base_dragged:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        self.base.set_ydata([event.ydata]*2)
        self.draw()

    def button_release(self, event):
        'whenever a mouse button is released'
        if event.button != 1:
            return
        self.base_dragged = False

np.random.seed(12345)

df = pd.DataFrame([
    np.random.normal(32000, 200000, 3650),
    np.random.normal(43000, 100000, 3650),
    np.random.normal(43500, 140000, 3650),
    np.random.normal(48000, 70000, 3650)], index = [1992, 1993, 1994, 1995])

# confidence intervals
ci95 = df.apply(lambda x:st.t.interval(0.95, len(x)-1, loc = np.mean(x), scale = st.sem(x)), axis=1)

values = np.hstack(ci95.values)
M = np.max(values)
M += 0.2*M

fig, ax=plt.subplots(figsize=(10, 7))
ax.set_ylim(0, M)
ax.set_xlim(-2, len(ci95.values)+1)

bars={'names': tuple(ci95.index), 'ci95': ci95.values}
be=BarEx(ax, bars, fig.canvas)

plt.show()
