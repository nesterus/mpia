import os
import seaborn as sns
from PIL import Image
from pandas import DataFrame
from pandas import read_csv
import matplotlib.pyplot as plt
from metadata.metadata import object_classes, object_type, object_type_ru

def _create_stat_plots(df, save_dir, data_dir):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # part statistics
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    plot_ind = 0
    
    # Alpha horizons
    sns.kdeplot(
        data=df,
        x='alpha_horizons',
        hue='class_type',
    ).set_title('Alpha horizons, parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    # sns.kdeplot(
    #     data=df,
    #     x='alpha_horizons',
    #     hue='class'
    # ).set_title('Alpha horizons, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    # plot_ind += 1
    # plt.close()
    
    # centroids
    sns.jointplot(
        data=df,
        x='centroids_norm_x',
        y='centroids_norm_y',
        hue='class_type'
    ).fig.suptitle('Centroids, parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    # sns.jointplot(
    #     data=df,
    #     x='centroids_norm_x',
    #     y='centroids_norm_y',
    #     hue='class'
    # ).fig.suptitle('Centroids, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    # plot_ind += 1
    # plt.close()
    
    # main diagonal
    sns.jointplot(
        data=df,
        x='main_diag_norm_x0',
        y='main_diag_norm_y0',
        hue='class_type'
    ).fig.suptitle('Normalized main diagonal (point x0, y0), parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    sns.jointplot(
        data=df,
        x='main_diag_norm_x1',
        y='main_diag_norm_y1',
        hue='class_type'
    ).fig.suptitle('Normalized main diagonal (point x1, y1), parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    # sns.jointplot(
    #     data=df,
    #     x='main_diag_norm_x0',
    #     y='main_diag_norm_y0',
    #     hue='class'
    # ).fig.suptitle('Normalized main diagonal (point x0, y0), classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    # plot_ind += 1
    # plt.close()
    
#     sns.jointplot(
#         data=df,
#         x='main_diag_norm_x1',
#         y='main_diag_norm_y1',
#         hue='class'
#     ).fig.suptitle('Normalized main diagonal (point x1, y1), classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # height_norm width_norm
    sns.jointplot(
        data=df,
        x='width_norm',
        y='height_norm',
        hue='class_type'
    ).fig.suptitle('Normalized (image scale) heights and width, parts types').figure.savefig("{}/output_9.png".format(save_dir))

    plt.close()
    
#     sns.jointplot(
#         data=df,
#         x='width_norm',
#         y='height_norm',
#         hue='class'
#     ).fig.suptitle('Normalized (image scale) heights and width, classes').figure.savefig("{}/output_10.png".format(save_dir))

#     plt.close()
    
    # area_bb_norm
    sns.kdeplot(
        data=df,
        x='area_bb_norm',
        hue='class_type'
    ).set_title('Normalized (image scale) area of part\'s bounding box, parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_bb_norm',
#         hue='class'
#     ).set_title('Normalized (image scale) area of part\'s bounding box, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # area_mask_norm
    sns.kdeplot(
        data=df,
        x='area_mask_norm',
        hue='class_type'
    ).set_title('Normalized (image scale) per-pixel mask area, parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_mask_norm',
#         hue='class'
#     ).set_title('Normalized (image scale) per-pixel mask area, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # height_norm_plant width_norm_plant
    sns.jointplot(
        data=df,
        x='width_norm_plant',
        y='height_norm_plant',
        hue='class_type'
    ).fig.suptitle('Normalized (object scale) heights and width, parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.jointplot(
#         data=df,
#         x='width_norm_plant',
#         y='height_norm_plant',
#         hue='class'
#     ).fig.suptitle('Normalized (object scale) heights and width, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # area_bb_norm_plant
    sns.kdeplot(
        data=df,
        x='area_bb_norm_plant',
        hue='class_type'
    ).set_title('Normalized (object scale) area of part\'s bounding box, parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_bb_norm_plant',
#         hue='class'
#     ).set_title('Normalized (plant scale) area of part\'s bounding box, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # area_mask_norm_plant
    sns.kdeplot(
        data=df,
        x='area_mask_norm_plant',
        hue='class_type'
    ).set_title('Normalized (object scale) per-pixel mask area, parts types').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_mask_norm_plant',
#         hue='class'
#     ).set_title('Normalized (object scale) per-pixel mask area, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1   
#     plt.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # object statistics
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df = DataFrame() # initialize data frame
    df = read_csv(data_dir + 'object_statistics.csv', header = 0)
    
    sns.jointplot(
        data=df,
        x='width_norm',
        y='height_norm',
        hue='tag_nums'
    ).fig.suptitle('Normalized heights and width for each object').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    sns.kdeplot(
        data=df,
        x='area_bb_norm',
        hue='tag_nums'
    ).set_title('Normalized bounding box area for each object').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    sns.kdeplot(
        data=df,
        x='area_mask_norm',
        hue='tag_nums'
    ).set_title('Normalized per-pixel mask area for each object').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    return plot_ind
  
def create_stat_plots(df, save_dir, data_dir):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # part statistics
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    plot_ind = 0
    
    # Alpha horizons
    ax = sns.kdeplot(
        data=df,
        x='alpha_horizons',
        hue='Тип подобъекта'
    )
    ax.set(xlabel='Угол',
       ylabel='Плотность')
    sns.move_legend(ax, title='Тип подобъекта', loc='best')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    # sns.kdeplot(
    #     data=df,
    #     x='alpha_horizons',
    #     hue='class'
    # ).set_title('Alpha horizons, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    # plot_ind += 1
    # plt.close()
    
    # centroids
    ax = sns.jointplot(
        data=df,
        x='centroids_norm_x',
        y='centroids_norm_y',
        hue='Тип подобъекта'
    )
    ax.ax_joint.set_xlabel('Координата x')
    ax.ax_joint.set_ylabel('Координата y')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    # sns.jointplot(
    #     data=df,
    #     x='centroids_norm_x',
    #     y='centroids_norm_y',
    #     hue='class'
    # ).fig.suptitle('Centroids, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    # plot_ind += 1
    # plt.close()
    
    # main diagonal
    ax = sns.jointplot(
        data=df,
        x='main_diag_norm_x0',
        y='main_diag_norm_y0',
        hue='Тип подобъекта'
    )
    ax.ax_joint.set_xlabel(xlabel='Координата х0')
    ax.ax_joint.set_ylabel('Координата у0')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    ax = sns.jointplot(
        data=df,
        x='main_diag_norm_x1',
        y='main_diag_norm_y1',
        hue='Тип подобъекта'
    )
    ax.ax_joint.set_xlabel(xlabel='Координата х1')
    ax.ax_joint.set_ylabel('Координата у1')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    # sns.jointplot(
    #     data=df,
    #     x='main_diag_norm_x0',
    #     y='main_diag_norm_y0',
    #     hue='class'
    # ).fig.suptitle('Normalized main diagonal (point x0, y0), classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    # plot_ind += 1
    # plt.close()
    
#     sns.jointplot(
#         data=df,
#         x='main_diag_norm_x1',
#         y='main_diag_norm_y1',
#         hue='class'
#     ).fig.suptitle('Normalized main diagonal (point x1, y1), classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # height_norm width_norm
    ax = sns.jointplot(
        data=df,
        x='width_norm',
        y='height_norm',
        hue='Тип подобъекта'
    )
    ax.ax_joint.set_xlabel(xlabel='Ширина')
    ax.ax_joint.set_ylabel(ylabel='Высота')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.jointplot(
#         data=df,
#         x='width_norm',
#         y='height_norm',
#         hue='class'
#     ).fig.suptitle('Normalized (image scale) heights and width, classes').figure.savefig("{}/output_10.png".format(save_dir))

#     plt.close()
    
    # area_bb_norm
    ax = sns.kdeplot(
        data=df,
        x='area_bb_norm',
        hue='Тип подобъекта'
    )
    ax.set(xlabel='Нормированная площадь окаймляющей рамки',
       ylabel='Плотность')
    sns.move_legend(ax, title='Тип подобъекта', loc='best')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_bb_norm',
#         hue='class'
#     ).set_title('Normalized (image scale) area of part\'s bounding box, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # area_mask_norm
    ax = sns.kdeplot(
        data=df,
        x='area_mask_norm',
        hue='Тип подобъекта'
    )
    ax.set(xlabel='Нормированная площадь попиксельной маски',
       ylabel='Плотность')
    sns.move_legend(ax, title='Тип подобъекта', loc='best')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_mask_norm',
#         hue='class'
#     ).set_title('Normalized (image scale) per-pixel mask area, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # height_norm_plant width_norm_plant
    ax = sns.jointplot(
        data=df,
        x='width_norm_plant',
        y='height_norm_plant',
        hue='Тип подобъекта'
    )
    ax.ax_joint.set_xlabel(xlabel='Нормированная ширина подобъекта')
    ax.ax_joint.set_ylabel(ylabel='Нормированная высота подобъекта')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.jointplot(
#         data=df,
#         x='width_norm_plant',
#         y='height_norm_plant',
#         hue='class'
#     ).fig.suptitle('Normalized (object scale) heights and width, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # area_bb_norm_plant
    ax = sns.kdeplot(
        data=df,
        x='area_bb_norm_plant',
        hue='Тип подобъекта'
    )
    ax.set(xlabel='Нормированная площадь окаймляющей рамки',
       ylabel='Плотность')
    sns.move_legend(ax, title='Тип подобъекта', loc='best')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_bb_norm_plant',
#         hue='class'
#     ).set_title('Normalized (plant scale) area of part\'s bounding box, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1
#     plt.close()
    
    # area_mask_norm_plant
    ax = sns.kdeplot(
        data=df,
        x='area_mask_norm_plant',
        hue='Тип подобъекта'
    )
    ax.set(xlabel='Нормированная попиксельная площадь',
       ylabel='Плотность')
    sns.move_legend(ax, title='Тип подобъекта', loc='best')
    ax.figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
#     sns.kdeplot(
#         data=df,
#         x='area_mask_norm_plant',
#         hue='class'
#     ).set_title('Normalized (object scale) per-pixel mask area, classes').figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
#     plot_ind += 1   
#     plt.close()

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # object statistics
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    df = DataFrame() # initialize data frame
    df = read_csv(data_dir + 'object_statistics.csv', header = 0)
    
    sns.jointplot(
        data=df,
        x='width_norm',
        y='height_norm',
        hue='tag_nums'
    ).figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    sns.kdeplot(
        data=df,
        x='area_bb_norm',
        hue='tag_nums'
    ).figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    sns.kdeplot(
        data=df,
        x='area_mask_norm',
        hue='tag_nums'
    ).figure.savefig("{}/output_{}.png".format(save_dir, plot_ind))
    plot_ind += 1
    plt.close()
    
    return plot_ind

def create_stat_report(dataset_name):
    stat_dir = './reports/' + dataset_name
    # create
    if 'reports' not in os.listdir('./'):
        os.mkdir('reports')
    if dataset_name not in os.listdir('./reports/'):
        os.mkdir(stat_dir)
    
    data_dir = './datasets/{}/{}/'.format(dataset_name, dataset_name)
    df = DataFrame() # initialize data frame
    df = read_csv(data_dir + 'part_statistics.csv', header = 0)
    
    object_classes_switched = {y: x for x, y in object_classes.items()}
    object_type_switched = {y: x for x, y in object_type_ru.items()}
    df['class_type'] = df['class_type'].map(object_type_switched)
    df['class'] = df['class'].map(object_classes_switched)

    df = df.rename(columns={'class_type': 'Тип подобъекта'})
    plot_num = create_stat_plots(df, stat_dir, data_dir)
    
    #save pdf with report
    images = [
        Image.open("{}/output_{}.png".format(stat_dir, ind)).convert('RGB')
        for ind in range(plot_num)
    ]

    pdf_path = stat_dir + "/stat_report.pdf"

    images[0].save(
        pdf_path, "PDF" ,resolution=100.0, save_all=True, append_images=images[1:]
    ) 