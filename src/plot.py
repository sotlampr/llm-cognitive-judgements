#!/usr/bin/env python
import texfig
import pandas as pd
import numpy as np
from operator import itemgetter
import tabulate

from utils import tasks

data = dict(
    cakes=dict(
        x_data=[10, 20, 35, 50, 70],
        y_data=[40, 35, 45, 60, 90],
        y_pred=[36, 41, 51, 61, 76],
        x_model=[0.0, 4.0, 5.0, 9.0, 10.0, 15.0, 16.0, 30.0, 31.0, 35.0, 36.0, 45.0, 46.0, 60.0, 61.0, 70.0, 71.0, 80.0],
        y_model=[31.0, 31.0, 36.0, 36.0, 39.0, 39.0, 41.0, 41.0, 51.0, 51.0, 59.0, 59.0, 61.0, 61.0, 76.0, 76.0, 86.0, 86.0],
        xlim=[0, 80],
        xticks=[0, 40, 80],
        ylim=[0, 160],
        yticks=[0, 40, 80, 120, 160],
        extent=[-24.5, 89.5, -36, 181.5],
        y_err=[
            (5, 5, 0, 0, 0),
            (0, 5, 5, 0, 0)
        ],
        title="Cakes"
    ),
    life_spans=dict(
        x_data=[18, 39, 61, 83, 96],
        y_data=[75.0, 75.0, 78.0, 90.0, 99.0],
        y_pred=[77, 78, 81, 89, 98],
        x_model=[1.0, 2.0, 6.0, 7.0, 25.0, 27.0, 39.0, 41.0, 51.0, 52.0, 59.0, 62.0, 64.0, 67.0, 68.0, 71.0, 75.0, 78.0, 80.0, 87.0, 91.0, 93.0, 99.0, 102.0, 103.0, 104.0, 107.0, 109.0],
        y_model=[60.0, 75.0, 76.0, 77.0, 77.0, 78.0, 78.0, 79.0, 80.0, 80.0, 81.0, 81.0, 82.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 92.0, 94.0, 96.0, 101.0, 103.0, 104.0, 105.0, 107.0, 110.0],
        xlim=[0, 110],
        xticks=[0, 50, 100],
        ylim=[0, 220],
        yticks=[0, 50, 100, 150, 200],
        extent=[-33, 109.5, -26, 221],
        y_err=[
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0)
        ],
        title="Life Spans",
    ),
    movie_grosses=dict(
        x_data=[1, 6, 10, 40, 100],
        y_data=[5.0, 13.0, 20.0, 60.0, 150.0],
        y_pred=[4, 9, 15, 54, 127],
        y_err=[
            (0, 0, 0, 0, 0),
            (0, 0, 10, 0, 0),
        ],
        xlim=[0, 110],
        xticks=[0, 50, 100],
        ylim=[0, 220],
        yticks=[0, 50, 100, 150, 200],
        extent=[-37, 117.5, -40.5, 259.5],
        x_model=[1.0, 3.0, 5.0, 6.0, 7.0, 9.0, 11.0, 13.0, 14.0, 15.0, 16.0, 17.0, 19.0, 20.0, 22.0, 23.0, 25.0, 27.0, 29.0, 30.0, 31.0, 32.0, 33.0, 35.0, 36.0, 37.0, 39.0, 40.0, 41.0, 43.0, 46.0, 48.0, 49.0, 50.0, 51.0, 52.0, 54.0, 56.0, 58.0, 59.0, 60.0, 61.0, 62.0, 65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 76.0, 77.0, 79.0, 80.0, 81.0, 83.0, 83.0, 85.0, 86.0, 87.0, 88.0, 90.0, 92.0, 93.0, 93.0, 95.0, 96.0, 97.0, 97.0, 101.0, 102.0, 103.0, 104.0, 106.0, 106.0, 107.0, 109.0],
        y_model=[2.0, 4.0, 7.0, 9.0, 11.0, 15.0, 18.0, 20.0, 22.0, 23.0, 25.0, 27.0, 27.0, 29.0, 32.0, 34.0, 37.0, 39.0, 40.0, 42.0, 42.0, 44.0, 46.0, 48.0, 50.0, 51.0, 53.0, 54.0, 57.0, 59.0, 62.0, 64.0, 66.0, 67.0, 68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 81.0, 83.0, 86.0, 88.0, 89.0, 92.0, 92.0, 94.0, 95.0, 100.0, 101.0, 102.0, 103.0, 104.0, 106.0, 107.0, 109.0, 109.0, 113.0, 114.0, 115.0, 116.0, 118.0, 119.0, 120.0, 121.0, 123.0, 123.0, 125.0, 126.0, 127.0, 127.0, 129.0, 131.0, 132.0, 132.0, 132.0, 137.0, 137.0],
        title="Movie Grosses",
    ),
    poems=dict(
        x_data=[2, 5, 12, 32, 67],
        y_data=[10.0, 15.0, 20.0, 40.0, 95.0],
        y_pred=[10, 16, 21, 42, 84],
        y_err=[
            (0, 0, 0, 0, 5),
            (0, 0, 0, 0, 5),
        ],
        xlim=[0, 80],
        xticks=[0, 40, 80],
        ylim=[0, 160],
        yticks=[0, 40, 80, 120, 160],
        extent=[-26, 89, -26.5, 190.5],
        x_model=[0.0, 2.0, 4.0, 6.0, 11.0, 13.0, 15.0, 15.0, 16.0, 16.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 41.0, 43.0, 44.0, 45.0, 47.0, 49.0, 50.0, 51.0, 52.0, 53.0, 55.0, 55.0, 55.0, 57.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, 65.0, 65.0, 66.0, 67.0, 68.0, 69.0, 69.0, 70.0, 71.0, 72.0, 72.0, 73.0, 73.0, 75.0, 76.0, 77.0, 77.0, 78.0, 78.0, 80.0],
        y_model=[9.0, 11.0, 14.0, 16.0, 16.0, 21.0, 21.0, 24.0, 25.0, 27.0, 29.0, 30.0, 31.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 39.0, 41.0, 41.0, 41.0, 45.0, 46.0, 47.0, 47.0, 49.0, 50.0, 51.0, 53.0, 54.0, 55.0, 58.0, 59.0, 60.0, 61.0, 64.0, 65.0, 66.0, 68.0, 69.0, 69.0, 71.0, 72.0, 73.0, 73.0, 75.0, 76.0, 77.0, 78.0, 78.0, 79.0, 79.0, 81.0, 83.0, 85.0, 86.0, 87.0, 88.0, 90.0, 91.0, 92.0, 93.0, 94.0, 100.0, 101.0, 101.0, 103.0, 105.0, 107.0, 108.0, 109.0],
        title="Poems"
    ),
    representatives=dict(
        x_data=[1, 3, 7, 15, 31],
        y_data=[4.0, 6.0, 8.0, 20.0, 40.0],
        y_pred=[3, 6, 11, 19, 35],
        y_err=[
            (0, 0, 0, 0, 2),
            (0, 0, 0, 0, 0),
        ],
        xlim=[0, 40],
        xticks=[0, 20, 40],
        ylim=[0, 80],
        yticks=[0, 20, 40, 60, 80],
        extent=[-12, 45.5, -15.5, 94],
        x_model=[0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 30.0, 31.0, 34.0, 35.0, 37.0, 37.0, 38.0, 39.0, 40.0],
        y_model=[3.0, 4.0, 4.0, 5.0, 7.0, 8.0, 9.0, 11.0, 11.0, 12.0, 13.0, 14.0, 15.0, 17.0, 17.0, 18.0, 19.0, 21.0, 21.0, 23.0, 23.0, 25.0, 25.0, 27.0, 27.0, 29.0, 29.0, 31.0, 31.0, 35.0, 35.0, 37.0, 37.0, 41.0, 41.0, 43.0, 43.0],
        title="Representatives",
    ),
    waiting_time=dict(
        x_data=[1, 3, 7, 11, 23],
        y_data=[3.0, 5.0, 10.0, 15.0, 30.0],
        y_pred=[1, 4, 9, 14, 30],
        y_err=[
            (0, 0, 0, 2, 0),
            (0, 0, 0, 2, 0),
        ],
        xlim=[0, 30],
        xticks=[0, 15, 30],
        ylim=[0, 60],
        yticks=[0, 15, 30, 45, 60],
        extent=[-9.5, 33.5, -12, 70],
        x_model=[0, 30],
        y_model=[0, 39],
        title="Waiting Times"
    )
)


coordinates = list()


def onclick(event):
    coordinates.append((round(event.xdata, 0), round(event.ydata, 0)))
    print(coordinates[-1])


class Cursor:
    """
    A cross hair cursor.
    """
    def __init__(self, ax):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        # text location in axes coordinates
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        if not event.inaxes:
            need_redraw = self.set_cross_hair_visible(False)
            if need_redraw:
                self.ax.figure.canvas.draw()
        else:
            self.set_cross_hair_visible(True)
            x, y = round(event.xdata, 0), round(event.ydata, 0)
            # update the line positions
            self.horizontal_line.set_ydata([y])
            self.vertical_line.set_xdata([x])
            self.text.set_text('x=%1.2f, y=%1.2f' % (x, y))
            self.ax.figure.canvas.draw()


def plot(ax, name, label=False):
    # im = plt.imread(f"graphics/{name}.png")
    ax.set_xlim(data[name]["xlim"])
    ax.set_xticks(data[name]["xticks"])
    ax.set_ylim(data[name]["xlim"])
    ax.set_yticks(data[name]["yticks"])
    # ax.imshow(im, extent=data[name]["extent"])
    ax.errorbar(
        data[name]["x_data"],
        data[name]["y_data"],
        data[name]["y_err"],
        fmt="o", label="Human Participants" if label else None, color="k",
        ms=4
    )
    ax.plot(
        data[name]["x_model"],
        data[name]["y_model"],
        label="Bayesian Estimate" if label else None, color="k"
    )
    ax.set_title(data[name]["title"], pad=6)


# models = ["gpt3", "gpt35", "huggingchat"]
models = ["gpt3", "gpt35"]

markers = dict(
    gpt3="x",
    gpt35="+",
    huggingchat="x"
)

colors = dict(
    gpt3="#1b9e77",
    gpt35="#d95f02",
    huggingchat="#7570b3"
)

labels = dict(
    gpt3="GPT-3",
    gpt35="ChatGPT",
    huggingchat="HuggingChat"
)

x_adjust = dict(
    gpt3=-1,
    gpt35=1,
    huggingchat=0
)


if __name__ == "__main__":
    # plt.ion()
    # for task in tasks:
    #     fig, ax = plt.subplots()
    #     cursor = Cursor(ax)
    #     fig.canvas.mpl_connect('motion_notify_event', cursor.on_mouse_move)
    #     cid = fig.canvas.mpl_connect('button_press_event', onclick)
    #     plot(ax, task)
    #     input()
    #     plt.close("all")
    fig, axes = texfig.subplots(width=4.9, ratio=1, nrows=2, ncols=3)
    axes = np.ravel(axes)
    label = True
    for task, ax in zip(tasks, axes):
        plot(ax, task, label)
        label = False
    # fig.subplots_adjust(vspace=0.5)
    dfs = list()
    for model in models:
        dfs.append((model, pd.read_csv(f"results/{model}.csv")))
        df = dfs[-1][-1]
        for ax, task in zip(axes, tasks):
            x_data = []
            y_data = []
            y_err = [[], []]
            for x, group in df[df.task == task].groupby("value"):
                median = group.num_value.dropna().median()
                bootstrap = np.array([
                    group.sample(
                        frac=1, replace=True
                    ).num_value.dropna().median()
                    for i in range(1000)
                ])
                ci = np.percentile(bootstrap, [0.16, 0.84])
                x_data.append(x)
                y_data.append(median)
                y_err[0].append(median-ci[0])
                if ci[1] > median:
                    y_err[1].append(ci[1]-median)
                else:
                    y_err[1].append(median-ci[1])
            x_data = np.array(x_data) + x_adjust[model]
            label = None if task != "cakes" else labels[model]
            ax.errorbar(
                x_data, y_data, y_err, fmt=markers[model],
                label=label,
                color=colors[model], ms=4

            )
            x_data, y_data = zip(
                *sorted(zip(x_data, y_data), key=itemgetter(0))
            )
            data[task][f"y_{model}"] = y_data
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.00), ncols=2)
    # fig.subplots_adjust(hspace=1)
    fig.supxlabel(r'$t$', y=0.14)
    fig.supylabel(r'$t_\text{total}')
    fig.tight_layout(h_pad=0.6, rect=(0, 0.1, 1.0, 1.0))
    texfig.savefig("test")
    rows = []
    for task in tasks:
        print(task)
        human = np.array(data[task]["y_data"])
        bayesian = np.array(data[task]["y_pred"])
        gpt3 = np.array(data[task]["y_gpt3"])
        gpt35 = np.array(data[task]["y_gpt35"])
        # hug = np.array(data[task]["y_huggingchat"])
        rows.append((
            task, 100*(np.abs(human - bayesian)/human).mean(),
            100*(np.abs(gpt3 - human)/human).mean(),
            100*(np.abs(gpt35 - human)/human).mean(),
        ))
    print(tabulate.tabulate(
        rows, headers=["Human Participants", "GPT-3", "ChatGPT"],
        tablefmt="latex_booktabs", floatfmt=".1f"
    ))
    rows = zip(*[
        tasks,
        100*dfs[0][-1].groupby("task").num_value.apply(
            lambda x: x.isna().mean()
        ).values,
        100*dfs[1][-1].groupby("task").num_value.apply(
            lambda x: x.isna().mean()
        ).values
    ])
    print(tabulate.tabulate(
        rows, headers=["Task", "GPT-3", "ChatGPT"],
        floatfmt=".1f", tablefmt="latex_booktabs"
    ))

    # # im = plot_waiting_time(ax)
