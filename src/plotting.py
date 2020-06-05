import pandas as pd
import glob
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams.update({'font.size':20})


# from data_pipelines import ImagePipeline

def metadata_heatmap():
    pass

def sat_histogram():
    '''
    Make 
    '''
    pass

def tt_holdout_error_curves(tb_csv_dir):
    '''
    Test, train and holdout error curves.
    
    '''
    test = pd.read_csv('{}/test_friday_am.csv'.format(tb_csv_dir))
    val = pd.read_csv('{}/validation_friday_am.csv'.format(tb_csv_dir))

    x_vals = test['Test']
    
    fig, axes = plt.subplots(1,2,figsize=(24,10))
    axes[0].plot(x_vals,test['avg'], label='Train')
    axes[0].plot(x_vals,val['avg'], label='Test')
    axes[0].axvline(4,color='k', label='Chosen Epoch')
    axes[0].set_title('RGB CNN Error vs Epoch')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('RMSE')
    axes[0].legend()
    axes[0].set_ylim([0.03,0.08])

    axes[1].plot(x_vals,test['gavg'], label='Train')
    axes[1].plot(x_vals,val['gavg'], label='Test')
    axes[1].axvline(2, color='k', label='Chosen Epoch')
    axes[1].set_title('Gray CNN Error vs Epoch')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('RMSE')
    axes[1].legend()
    axes[1].set_ylim([0.03,0.08])

    plt.savefig('../images/tt_error.png',dpi=200)
    plt.close('all')
    return val
    

if __name__ == "__main__":
    path = '/home/conslag/Documents/galvanize/capstones/homes-like-this/data/models'
    df = tt_holdout_error_curves(path)
    