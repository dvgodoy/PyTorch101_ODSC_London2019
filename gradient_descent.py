import numpy as np
import plotly.graph_objs as go
from plotly.tools import make_subplots
from ipywidgets import VBox, interactive, IntSlider, Checkbox, FloatLogSlider, Button, FloatSlider, HBox, Dropdown

def build_figure_deriv():
    w = np.linspace(-2, 2, 100)

    def convex_j(w):
        return w ** 2

    def convex_djdw(w):
        return 2 * w

    def nonconvex_j(w):
        return np.sin(w*3) + w**2

    def nonconvex_djdw(w):
        return 3*np.cos(w*3) + 2*w

    j = convex_j
    djdw = convex_djdw

    w0 = -2
    lr = .2
    delta = lr * djdw(w0)

    traces = [go.Scatter(x=w, y=j(w), mode='lines', line={'color': 'black'}, showlegend=False),
             go.Scatter(x=[w0], y=[j(w0)], marker={'color': 'black'}, name='w before'),
             go.Scatter(x=[w0-delta], y=[j(w0-delta)], marker={'color': 'red'}, name='w after'),
             go.Scatter(x=[w0-delta, w0-delta], y=[j(w0), j(w0-delta)], showlegend=False, mode='lines', line={'color': 'gray', 'dash': 'dot'}),
             go.Scatter(x=[w0, w0-delta], y=[j(w0-delta), j(w0-delta)], showlegend=False, mode='lines', line={'color': 'gray', 'dash': 'dot'}),
             go.Scatter(x=[w0, w0-delta], y=[j(w0), j(w0-delta)], showlegend=False, mode='lines', line={'color': 'red', 'width': 2, 'dash': 'dash'}),
             go.Scatter(x=[w0, w0-delta], y=[j(w0), j(w0)], mode='lines', line={'color': 'red'}, name='- lr * dJ/dw(w)'),
             go.Scatter(x=[w0, w0], y=[j(w0), j(w0-delta)], showlegend=False, mode='lines', line={'color': 'gray'})]

    fig = go.Figure(traces, layout={'title': 'Gradient Descent',
                                    'width': 600, 'height': 600, 'xaxis': {'zeroline': False, 'title': 'Feature'},
                                    'yaxis': {'title': 'Loss'}})
    f = go.FigureWidget(fig)

    functype = Dropdown(description='Function', options=['Convex', 'Non-convex'], value='Convex')
    bt_reset = Button(description='Reset')
    bt_step = Button(description='Step')
    lrate = FloatSlider(description='Learning Rate', value=.05, min=.05, max=1.1, step=.05)

    def update2(functype):
        if functype == 'Convex':
            j = convex_j
        else:
            j = nonconvex_j

        #w0 = np.random.rand() * 4 - 2
        w0 = -1.5
        j0 = j(w0)
        with f.batch_update():
            f.data[0].y = j(f.data[0].x)
            for i in range(2, 8):
                f.data[i].visible = False
            f.data[1].x = [w0]
            f.data[1].y = [j0]
            f.data[2].x = [w0]
            f.data[2].y = [j0]

    def update(b):
        if functype.value == 'Convex':
            j = convex_j
            djdw = convex_djdw
        else:
            j = nonconvex_j
            djdw = nonconvex_djdw

        if b == bt_reset:
            ##w0 = np.random.rand() * 4 - 2
            w0 = -1.5
            j0 = j(w0)
            with f.batch_update():
                f.data[0].y = j(f.data[0].x)
                for i in range(2, 8):
                    f.data[i].visible = False
                f.data[1].x = [w0]
                f.data[1].y = [j0]
                f.data[2].x = [w0]
                f.data[2].y = [j0]
        else:
            w0, j0 = f.data[2].x[0], f.data[2].y[0]
            lr = lrate.value
            delta = lr * djdw(w0)
            w1 = w0-delta
            j1 = j(w1)
            with f.batch_update():
                for i in range(2, 8):
                    f.data[i].visible = True
                f.data[1].x = [w0]
                f.data[1].y = [j0]
                f.data[2].x = [w1]
                f.data[2].y = [j1]
                f.data[3].x = [w1, w1]
                f.data[3].y = [j0, j1]
                f.data[4].x = [w0, w1]
                f.data[4].y = [j1, j1]
                f.data[5].x = [w0, w1]
                f.data[5].y = [j0, j1]
                f.data[6].x = [w0, w1]
                f.data[6].y = [j0, j0]
                f.data[7].x = [w0, w0]
                f.data[7].y = [j0, j1]

    bt_step.on_click(update)
    bt_reset.on_click(update)

    update(bt_reset)
    return (f, interactive(update2, functype=functype), HBox((bt_reset, bt_step)), lrate)

def interactive_gd():
    interactive_gd = VBox(build_figure_deriv())
    interactive_gd.layout.align_items = 'center'
    return interactive_gd