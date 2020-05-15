from skimage.io import imread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os
from skimage import io
import seaborn as sb
from data_pipelines import ImagePipeline

plt.style.use('ggplot')
plt.rcParams.update({'font.size': 20})

def bar_chart_desc(tick_labels, values, xlabel, ylabel, title):
    desc_idx = np.argsort(np.array(values))
    desc_y = values[desc_idx][::-1]
    desc_x = tick_labels[desc_idx][::-1]


    fig, ax = plt.subplots(1,figsize=(10,8))
    ax.bar(desc_x, desc_y)
    ax.set_xticklabels(desc_x, rotation=25, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

def plot_1x4_resize_img(sq_imgs, size=(20,5)):
    '''
    Plots 1x4 of varying resizing
    '''
    shape_lst = ['32 x 32','64 x 64','128 x 128','256 x 256']

    fig, axs = plt.subplots(1,4,figsize=size)
    for idx, (ax, img) in enumerate(zip(axs, sq_imgs)):
        ax.imshow(img, cmap='gray')
        ax.set_xlabel(shape_lst[idx])
        ax.set_xticks([]); ax.set_yticks([])


def plot_1x2_img(imgs, labels, size=(20,8)):
    '''
    Plots 1x2 of varying resizing
    '''

    fig, axs = plt.subplots(1,2,figsize=size)
    for idx, (ax, img) in enumerate(zip(axs, imgs)):
        ax.imshow(img)
        ax.set_xlabel(labels[idx])
        ax.set_xticks([]); ax.set_yticks([])


def cm_plot(cm,title):
    '''
    Plots heatmap of Confusion matrix for NB classifier
    '''
    cats = ['Denver','Arvada','Aurora','Lakewood','Centennial','Westminster','Thornton']
    y_range = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]

    cm_round = np.round(cm,decimals=2)
    heat_map = sb.heatmap(cm_round, annot=True)
    plt.xticks(range(len(cats)), labels = cats, rotation=25)
    plt.title(title)
    plt.yticks(y_range, labels = cats, rotation=0, ha='right')

def save_fig(image_name):
    plt.savefig(f'../images/{image_name}', dpi=200)
    plt.close('all')

if __name__ == "__main__":
    df = pd.read_csv('../data/metadata/2020-05-14_pg1_3_all.csv')

    ''''
    Desc bar chart
    '''
    
    # count_by_city = df.groupby('city').count()['image_file']
    # bar_chart_desc(count_by_city.index, count_by_city.values, 'City',"Listing Images (#)", "Number of Listing Images by City")
    # save_fig('count_by_city.png')


    '''
    Plotting 1x2 by city
    '''
    
    cats = ['Denver','Arvada','Aurora','Lakewood','Centennial','Westminster','Thornton']
    for cat in cats:
        test = df.image_file[df.city == cat]
        
    
        fnames = []
        for _ in range(2):
            rand_idx = np.random.randint(0,500)
            fnames.append(list(test)[rand_idx])
        color_imgs = [io.imread(f'../data/listing_images/full/{file}') for file in fnames]
        plot_1x2_img(color_imgs,[f'{cat} #1',f'{cat} #2'])
        save_fig(f'{cat}_1x2.png')

    '''
    Plotting 1x4 gray
    '''
    # shape_lst = [32, 64, 128, 256]
    # img_idx = 55
    # resized_imgs = []
    # for shape in shape_lst:
    #     img_pipe = ImagePipeline(f'../data/proc_images/{shape}')
    #     img_pipe.read()
    #     # breakpoint()
    #     resized_imgs.append(img_pipe.img_lst2[55])

    # plot_1x4_resize_img(resized_imgs)
    # save_fig('1x4_gray.png')
    '''
    Plotting 1x2 showing square
    '''
    # shape = 256
    # img_pipe = ImagePipeline(f'../data/proc_images/{shape}')
    # img_pipe.read()
    # fname = img_pipe.img_names2[0][55]
    # # img_pipe._square_image()

    # color_img = io.imread(f'../data/listing_images/full/{fname}')

    # y_len, x_len, _ = color_img.shape

    # crop_len = min([x_len,y_len])
    # x_crop = [int((x_len/2) - (crop_len/2)), int((x_len/2) + (crop_len/2))]
    # y_crop = [int((y_len/2) - (crop_len/2)), int((y_len/2) + (crop_len/2))]
    # if y_len >= crop_len:
    #     sq_color = color_img[y_crop[0]:y_crop[1], x_crop[0]:x_crop[1]]
    # else:
    #     sq_color = color_img[x_crop[0]:x_crop[1], y_crop[0]:y_crop[1]]

    # color_lst = [color_img, sq_color]
    # plot_1x2_img(color_lst)
    # save_fig('1x2_color.png')

    '''
    Plotting CM (on train & on holdout)
    '''
    # # on test from NB_classifier.py
    # cm_test = np.array([[0.12169312, 0.07407407, 0.07407407, 0.22751323, 0.21164021,
    #     0.24867725, 0.04232804],
    #    [0.05369128, 0.09395973, 0.04697987, 0.26845638, 0.19463087,
    #     0.30872483, 0.03355705],
    #    [0.06521739, 0.08695652, 0.07971014, 0.27536232, 0.17391304,
    #     0.23188406, 0.08695652],
    #    [0.05691057, 0.09756098, 0.05691057, 0.29268293, 0.2195122 ,
    #     0.2195122 , 0.05691057],
    #    [0.04672897, 0.10280374, 0.07476636, 0.25233645, 0.18691589,
    #     0.27102804, 0.06542056],
    #    [0.08695652, 0.07608696, 0.05434783, 0.19565217, 0.2826087 ,
    #     0.2826087 , 0.02173913],
    #    [0.05376344, 0.12903226, 0.02150538, 0.15053763, 0.16129032,
    #     0.44086022, 0.04301075]])

    # cm_plot(cm_test,'NB Confusion Matrix - Test Split')
    # save_fig('cm_test.png')

    # cm_holdout = np.array([[0.0503876 , 0.08527132, 0.06976744, 0.25968992, 0.20930233,
    #     0.24418605, 0.08139535],
    #    [0.04      , 0.13714286, 0.02857143, 0.23428571, 0.22857143,
    #     0.24571429, 0.08571429],
    #    [0.05298013, 0.07284768, 0.0397351 , 0.25165563, 0.16556291,
    #     0.31125828, 0.10596026],
    #    [0.09815951, 0.08588957, 0.04907975, 0.36809816, 0.11656442,
    #     0.2208589 , 0.06134969],
    #    [0.10714286, 0.09285714, 0.07142857, 0.24285714, 0.22142857,
    #     0.20714286, 0.05714286],
    #    [0.07207207, 0.13513514, 0.04504505, 0.18018018, 0.17117117,
    #     0.32432432, 0.07207207],
    #    [0.05982906, 0.16239316, 0.05128205, 0.18803419, 0.21367521,
    #     0.25641026, 0.06837607]])

    # cm_plot(cm_holdout,'NB Confusion Matrix - Holdout Split')
    # save_fig('cm_holdout.png')

    