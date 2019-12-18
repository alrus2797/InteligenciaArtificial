import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_confusion_matrix(df_confusion, title='Confussion Matrix', cmap=plt.cm.hsv, alpha = 1.0):
    plt.matshow(df_confusion,cmap = 'winter') # cambiar el cmap
    plt.title(title)
    for (i, j), z in np.ndenumerate(df_confusion):
        plt.text(j, i, '{:0.3f}%'.format(z*100), ha='center', va='center')
        plt.text(j, i, '{:0.3f}'.format(z*100), ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.colorbar() ##Cambiar
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)

'''
/** verif_target: Desired class
/** verif_res   : Obtained class
'''
def make_confussion_matrix(verif_target,verif_res, title = 'Confussion Matrix', names = {}, verbose= False):
    verif_target = np.array(verif_target)
    verif_res = np.array(verif_res)
    confussion_matrix = pd.crosstab(verif_target, verif_res, rownames=['Actual'], colnames=['Predicted'], margins=False, normalize= True, dropna=False)
    confussion_matrix =  confussion_matrix.rename(columns=names, index=names)
    if verbose:
        print(confussion_matrix)
        print("Predicted:\t",verif_res)
        print("Real:\t\t",verif_target)
    plot_confusion_matrix(confussion_matrix, title = title)
    return confussion_matrix