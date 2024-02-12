# Описание используемых данных

Список подготовленных и размеченных датасетов:

1. [Cassava Leaf Disease](https://github.com/nesterus/mpia/tree/main/datasets/Cassava%20Leaf%20Disease/Cassava%20Leaf%20Disease) ([архив](https://disk.yandex.ru/d/ubDyzqG8x8vfNg))[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1dtBwF3C1ozAfCC6_U-RoEI5_4wU8-QHX?usp=sharing) 
2. [Corn or Maize Leaf Disease](https://github.com/nesterus/mpia/tree/main/datasets/Corn%20or%20Maize%20Leaf%20Disease/Corn%20or%20Maize%20Leaf%20Disease) ([архив](https://disk.yandex.ru/d/FtydQWrF6kWoEQ)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1oJuxEi1UMIUbBvcHoURcFDslAIHIREau?usp=sharing) 
3. [Flower classification](https://github.com/nesterus/mpia/tree/main/datasets/flower_classification/flower_classification) ([архив](https://disk.yandex.ru/d/CXNN63kmrpGSaA)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zoHT5B72s-yymMGxJmupg4-ES0x0_2Hw?usp=sharing) 
4. [Fruit plants](https://github.com/nesterus/mpia/tree/main/datasets/fruit_plants/fruit_plants) ([архив](https://disk.yandex.ru/d/MHDJiO2Nxe8q6A)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uRtSfoKhB8lTsTrpAk8Y-UutIykLmwaw?usp=sharing) 
5. [Herbarium](https://github.com/nesterus/mpia/tree/main/datasets/Herbarium/Herbarium) ([архив](https://disk.yandex.ru/d/z0F-1IRUuelS8A)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UNLepp1oVanuq2VRpzbZyqq2lOYoNJgT?usp=sharing) 
6. [Plant Pathology](https://github.com/nesterus/mpia/tree/main/datasets/Plant%20Pathology/Plant%20Pathology) ([архив](https://disk.yandex.ru/d/1RZHS2sCBgRpTA)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1uu3dq3zXXtmTm2HmrxSRxLBGXuVIeo_X?usp=sharing) 
7. [Satellite images](https://github.com/nesterus/mpia/tree/main/datasets/satellite_images/satellite_images) ([архив](https://disk.yandex.ru/d/8Nb5E0DyQ4PlHQ)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1zCWc6s6OFdOhBNdp7sglReUOOgOZn26i?usp=sharing) 
8. [Tomato detection](https://github.com/nesterus/mpia/tree/main/datasets/Tomato%20detection/Tomato%20detection) ([архив](https://disk.yandex.ru/d/jNBpqd_KPPTb6Q)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UU_p00ELm6T15suTiBAYJup3gSb9zxeP?usp=sharing) 
9. [Wild Edible Plants](https://github.com/nesterus/mpia/tree/main/datasets/Wild%20Edible%20Plants/Wild%20Edible%20Plants) ([архив](https://disk.yandex.ru/d/xaaJ0GBx2_TtAQ)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KWU0eEx3KENUhi1G5HUdQ-RBV6Je4792?usp=sharing) 
10. [Soy](https://github.com/nesterus/mpia/tree/main/datasets/soy/soy) ([архив](https://disk.yandex.ru/d/n5TYf7jQPjIyCA)) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1t9Peq6sHzTFfbmo4VmWz77oapFqdft4_?usp=sharing) 

## Описание общей структуры датасетов:

Каждый датасет содержит папку “ann” и “img”. 

“img” — изображения

“ann” — аннотация в общем для всех датасетов формате (расширение файлы json); каждому изображения соответствует свой файл с разметкой. Обработанные файлы с разметкой содержат в имени файла “_mpta” перед указанием расширения. В папке “ann” также содержатся файлы со статистиками по датасету “stats_dists.csv”  и “norm_stats_dists.csv”. В директории кроме папок с аннотацией и изображениями также находятся файлы с рассчитанными статистиками “object_statistics.csv”, “part_statistics.csv”.

## Описание датасетов

1. Cassava Leaf Disease

Состоит из 21400 изображений больных и здоровых кустов Юки. Данный датасет был выбран, так как на одном изображении хорошо различимы больные и здоровые элементы; необычно расположение стеблей и листьев.

Размечено:

- растений: 59
- стеблей: 637
- листьев: 558
- больных: 346

2. Corn or Maize Leaf Disease

Датасет состоит из 4188 изображений больных и здоровых листьев кукурузы. Нам необходимо покрыть свойство “больной”, а на большинстве датасетов болезни разглядеть очень тяжело, поэтому был выбран данный датасет.  На отобранных 75 изображениях болезни легко различимы.

Размечено:

- растений: 90
- стеблей: 14
- листьев: 100
- больных: 91

3. Flower classification

Состоит из 18500 изображений 102 видов цветов. Мы выбрали этот датасет, так как кроме цветов на изображениях можно выделить другие элементы: листья и стебли. Мы отобрали 55 изображений, на которых разнообразны расположение и группировка цветов.

Размечено:

- растений: 78
- стеблей: 238
- листьев: 401
- цветов: 924
- больных: 14

4. Fruit plants

В датасете изображения винограда, яблонь и грушевых деревьев в саду. На отобранных 36 изображениях можно выделить стебель, листья, плоды, а также свойства: больной и усохший.

Размечено:

- растений: 56
- стеблей: 2470
- листьев: 2691
- плодов: 284
- усохших: 474
- больных: 689

5. Herbarium

В датасете более 2 миллионов изображений 15501 вида засохших растений. Этот датасет был выбран, так как это единственный крупный датасет с засохшими растениями. Изображения растений отбирали так, что у них можно выделить все необходимые элементы, кроме плода. Отобрали 80 изображений разных видов растений.

Размечено:

- растений: 185
- стеблей: 2154
- листьев: 2328
- цветов: 579
- корней: 100

6. Plant Pathology

Датасет состоит из 3642 изображений болезней растений. Датасет был выбран, так как на изображениях хорошо видны болезни, и можно выделить элементы растений, такие как стебель и лист. Мы отобрали 80 изображений.

Размечено:

- растений: 85
- стеблей: 249
- листьев: 271
- усохших: 1
- больных: 166

7. Satellite images

В наборе данных представлена композиционная разметка, связывающая отдельные домам с дорогами.

Спутниковые снимки:

- Здание: 2902
- Бассейн: 79
- Дерево: 649
- Дорога: 603
- Групп деревьев: 363

8. Tomato detection

Состоит из 895 изображений кустов помидоров. Он был выбран, так как это один из немногих датасетов в общем доступе, где есть куст с хорошо различимыми плодами. Также необычно само расположение плодов. У данного датасета уже была разметка, но она нам не подходит, так как выделены только bounding boxes у плодов. Отобрали 80 изображений, на которых отчетливо можно выделить стебель и принадлежащие ему плоды и листья

Размечено:

- растений: 148
- стеблей: 670
- листьев: 338
- плодов: 834
- усохших: 6
- больных: 48

9. Wild Edible Plants

Датасет состоит из 16535 изображений 35 видов растений. Данный датасет был выбран, так как необходимы изображения с цветами. Отобрали 84 изображения, на которых хорошо различимы элементы растений: стебель, лист и цветок. Также есть несколько больных элементов.

Размечено:

- растений: 154
- стеблей: 536
- листьев: 781
- цветов: 560
- больных: 31

10. Soy

Фотографии кустов, на которых можно выделить стебель и листья. Интересна структура расположения листьев на стебле. Есть немного засохших элементов. Была отобрана 251 фотография.

Размечено:

- растений: 1394
- стеблей: 1493
- листьев: 7674
- усохших: 169


## Описание вспомогательных файлов и содержащихся в них параметров:

Файл part_statistics.csv расположен во второй директории с названием датасета, например, “Cassava Leaf Disease / Cassava Leaf Disease”. Файл содержит статистики по каждому подобъекту в датасете. Кроме некоторых базовых значений файл содержит дополнительную информацию и нормированные статистики для каждого отдельного подобъекта. Характеристики подобъектов можно разделить на следующие категории:

Базовые характеристики

- 'id_names', 'condition', 'type', 'ierarchy', 'group', 'kind', 'ripeness', 'stage', 'integrity', 'class', 'class_type', 'tag_nums', 'alpha_horizons' – их описание приведено выше

Дополнительные данные о подобъекте

- ‘part_height’, ‘part_width’ – высота и ширина подобъектов
- ‘obj_id’ – уникальные id объектов в датасете, id состоит из названия изображения и наименования объекта на изображении

Нормированные характеристики

- 'centroids_norm_x', 'centroids_norm_y' – цетроиды, нормированные на размер всего изображения;
- ‘main_diag_norm_x0', 'main_diag_norm_x1', 'main_diag_norm_y0', 'main_diag_norm_y1' – координаты диагоналей, нормированные на размер изображения
- 'height_norm', 'width_norm' – высота и ширина окаймляющей рамки подобъекта, нормированная на размер изображения

Площадь

- 'area' – кол-во пикселей маски подобъекта
- 'area_bb_norm' – площадь окаймляющей рамки подобъекта, нормированная на площадь всего изображения
- 'area_mask_norm' – попиксельная площадь маски подобъекта, нормированная на площадь всего изображения

Нормированные статистики относительно всего объекта, которому принадлежит подобъект

- 'x_coord_object', 'y_coord_object' – координаты центра подобъекта относительно объекта
- 'x_coord_object_norm', 'y_coord_object_norm' – нормированные координаты центра подобъекта относительно объекта
- 'height_norm_plant', 'width_norm_plant' – высота и ширина окаймляющей рамки подобъекта, нормированная на размер всего объекта, которому принадлежит подобъект
- 'area_bb_norm_plant' – площадь окаймляющей рамки подобъекта, нормированная на размер всего объекта, которому принадлежит подобъект
- 'area_mask_norm_plant' – попиксельная площадь маски подобъекта, нормированная на размер всего объекта, которому принадлежит подобъект

Файл object_statistics.csv расположен во второй директории с названием датасета, например, “Cassava Leaf Disease / Cassava Leaf Disease”. Файл содержит статистики по каждому объекту, который является объединением подобъектов, в датасете.

Площадь

- 'area' – кол-во пикселей всех масок подобъектов, составляющих данный объект
- 'area_bb_norm' – площадь окаймляющей рамки объекта, нормированная на размер всего изображения
- 'area_mask_norm' – попиксельная площадь маски объекта, нормированная на размер всего изображения

Нормированные статистики

- 'centroids_norm_x', 'centroids_norm_y' – центр объекта, нормированный относительно изображения целиком
- 'height_norm', 'width_norm' – высота и ширина объекта, нормированная на размер изображения

Дополнительные данные об объектах

- ‘tag_nums’ – идентификатор отдельного объекта, содержащего подобъекты. (определение соответствует описанию выше). Может быть “plant_{id}” либо тегом аномалий
- 'height', 'width' – высота и ширина объекта
- 'img_id' – уникальный идентификатор изображения в датасете (название изображения без расширения)
- 'x_min', 'y_min' – верхняя левая координата объекта. Используется для выполнения последующих функций обработки данных

