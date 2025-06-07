import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from datetime import datetime, timedelta
import os

class Plotter:
    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None.")
        self.df = df
        self.fig = None

    def preprocess_data(self, start_date=None):
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        if start_date is None:
            start_date = datetime.today() - timedelta(days=7)
        else:
            start_date = pd.to_datetime(start_date)

        df_recent = self.df[self.df['Date'] >= start_date.strftime('%Y-%m-%d')]
        if df_recent.empty or len(df_recent) < 2:
            raise ValueError("Not enough data in the given date range.")

        df_recent = df_recent.iloc[[0, -1]]  # Select first and last row in range
        df_recent = df_recent.dropna(axis=1).set_index('Date').T
        return df_recent

    def _add_texts(self, ax, fig, title, subtitle, source):
        ax.plot([0.05, .95], [.95, .95], transform=fig.transFigure, clip_on=False, color='#E3120B', linewidth=.6)
        ax.add_patch(plt.Rectangle((0.05, .95), 0.04, -0.02, facecolor='#E3120B',
                                   transform=fig.transFigure, clip_on=False, linewidth=0))
        ax.text(x=0.05, y=.9, s=title, transform=fig.transFigure, ha='left', fontsize=14, weight='bold', alpha=1)
        ax.text(x=0.05, y=.87, s=subtitle, transform=fig.transFigure, ha='left', fontsize=12, alpha=1)
        ax.text(x=0.05, y=0.04, s=source, transform=fig.transFigure, ha='left', fontsize=10, alpha=.7)

    def _style_plot(self, ax, fig):
        ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
        ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
        ax.xaxis.set_tick_params(pad=2, labelbottom=True, bottom=True, labelsize=12)
        ax.yaxis.set_tick_params(pad=2, labelsize=12)
        ax.spines[['top', 'right', 'bottom']].set_visible(False)
        ax.spines['left'].set_color('black')
        ax.spines['left'].set_linewidth(1.1)
        plt.subplots_adjust(bottom=0.12, top=0.85, left=0.08, right=0.97)
        fig.patch.set_facecolor('white')

    def plot_yield_curve(self, title: str, subtitle: str, source: str, start_date=None):
        df_recent = self.preprocess_data(start_date)

        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=96)
        colors = ['#808080', '#006BA2']
        for i, date in enumerate(df_recent.columns):
            ax.plot(df_recent.index, df_recent[date], linestyle='-', linewidth=1.2,
                    color=colors[i % len(colors)], label=date.strftime('%Y-%m-%d'))

        self._style_plot(ax, fig)
        self._add_texts(ax, fig, title, subtitle, source)
        ax.legend(loc="best", fontsize=8)
        self.fig = fig
        plt.show()

    def plot_column_trend(self, start_date, column_name, title, subtitle, source):
        if column_name not in self.df.columns:
            print(f"Column '{column_name}' not found.")
            return

        self.df['Date'] = pd.to_datetime(self.df['Date']).dt.date
        start_date = pd.to_datetime(start_date).date()
        df_column = self.df[self.df['Date'] >= start_date][['Date', column_name]]

        if df_column.empty:
            print("No data available for the selected date range.")
            return

        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=96)
        ax.plot(df_column['Date'], df_column[column_name], linestyle='-', color='#006BA2', linewidth=1.2)

        # Min/Max annotations
        min_val = df_column[column_name].min()
        max_val = df_column[column_name].max()
        min_date = df_column[df_column[column_name] == min_val]['Date'].values[0]
        max_date = df_column[df_column[column_name] == max_val]['Date'].values[0]

        ax.annotate(f'Min: {min_val:.2f}', xy=(min_date, min_val), xytext=(-30, -20),
                    textcoords='offset points', fontsize=10, color='red',
                    arrowprops=dict(facecolor='red', arrowstyle='->'))
        ax.annotate(f'Max: {max_val:.2f}', xy=(max_date, max_val), xytext=(-30, 15),
                    textcoords='offset points', fontsize=10, color='green',
                    arrowprops=dict(facecolor='green', arrowstyle='->'))

        ax.set_ylim(min_val * 0.98, max_val * 1.02)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

        self._style_plot(ax, fig)
        self._add_texts(ax, fig, title, subtitle, source)
        self.fig = fig
        plt.show()

    def save_plot(self, name):
        if not hasattr(self, 'fig') or self.fig is None:
            raise ValueError("No figure to save. Call a plot method first.")
        directory = os.path.join(os.getcwd(), 'source')
        os.makedirs(directory, exist_ok=True)
        self.fig.savefig(os.path.join(directory, name), bbox_inches='tight')

    def get_df(self):
        return self.df


    @staticmethod
    def fplot(df: pd.DataFrame, column_name: str, start_date=None,
              title="Line Chart", subtitle="", source=""):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty or None.")
        if 'Date' not in df.columns or column_name not in df.columns:
            raise ValueError("Required columns 'Date' or specified column not found in DataFrame.")

        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.date

        if start_date:
            start_date = pd.to_datetime(start_date).date()
            df = df[df['Date'] >= start_date]

        if df.empty:
            raise ValueError("No data available for the selected date range.")

        fig, ax = plt.subplots(figsize=(12, 7.5), dpi=96)
        ax.plot(df['Date'], df[column_name], linestyle='-', color='#006BA2', linewidth=1.2)

        # Min/Max annotations
        min_val = df[column_name].min()
        max_val = df[column_name].max()
        min_date = df[df[column_name] == min_val]['Date'].values[0]
        max_date = df[df[column_name] == max_val]['Date'].values[0]

        ax.annotate(f'Min: {min_val:.2f}', xy=(min_date, min_val), xytext=(-30, -20),
                    textcoords='offset points', fontsize=10, color='red',
                    arrowprops=dict(facecolor='red', arrowstyle='->'))
        ax.annotate(f'Max: {max_val:.2f}', xy=(max_date, max_val), xytext=(-30, 15),
                    textcoords='offset points', fontsize=10, color='green',
                    arrowprops=dict(facecolor='green', arrowstyle='->'))

        ax.set_ylim(min_val * 0.98, max_val * 1.02)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))

        # Use dummy instance to call instance styling methods
        temp = Plotter(df)
        temp._style_plot(ax, fig)
        temp._add_texts(ax, fig, title, subtitle, source)

        plt.show()
