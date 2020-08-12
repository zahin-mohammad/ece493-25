import matplotlib.pyplot as plt

'''
    Each series in data is a tuple
    series[0]: series label
    series[1]: series data points
'''


def graph(title, data):
    plt.figure()
    for series in data:
        has_legend = False
        x = range(1, len(series[1])+1)
        y = series[1]
        if series[0] == '':
            plt.plot(x, y)
        else:
            has_legend = True
            plt.plot(x, y, label=series[0])

    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    if has_legend:
        plt.legend()
    plt.title(title)

    plt.savefig(f'./plots/{title}', dpi=500)
    plt.clf()
