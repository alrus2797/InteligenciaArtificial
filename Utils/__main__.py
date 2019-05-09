import matplotlib.pyplot as plt

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.hsv, alpha = 1.0):
    plt.matshow(df_confusion,cmap = 'winter') # imshow
    #plt.title(title)
    for (i, j), z in np.ndenumerate(df_confusion):
        plt.text(j, i, '{:0.3f}%'.format(z*100), ha='center', va='center')
        plt.text(j, i, '{:0.3f}'.format(z*100), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)