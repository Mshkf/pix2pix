# pix2pix
ввод            |  вывод
:-------------------------:|:-------------------------:
![3](https://github.com/Mshkf/pix2pix/assets/93014053/28d4859c-82fb-4f0f-bff8-4f84bda7a63a)  |  ![outp_3](https://github.com/Mshkf/pix2pix/assets/93014053/fdcd3737-b745-4363-a09e-1af0adddbb7b)

Реализация архитектуры pix2pix и создание интерфейса для генерации портретов на её основе
# Обучение и инференс
Всё обучениие проходило на каггл ноутбуке: первый ноутбук для повтора экспемимента из статьи (фасады), второй для создания веба (портреты)

Посмотреть обучение

фасадов на ([гитхабе](https://github.com/Mshkf/pix2pix/blob/main/exploration_pix2pix.ipynb)|[каггле](https://www.kaggle.com/code/mshkf7/pix2pix/notebook))

портретов на ([гитхабе](https://github.com/Mshkf/pix2pix/blob/main/portraits_pix2pix.ipynb)|[каггле](https://www.kaggle.com/code/mshkf7/pix2pix-portraits-dataset))
# Веса и чекпоинты
Все веса вместе с картинками (картинки почему-то странно отбражаются: хуже, чем на самом деле) на этой эпохе сохранены в выводе ноутбука

[вывод для фасадов](https://www.kaggle.com/code/mshkf7/pix2pix/output)

[вывод для портретов](https://www.kaggle.com/code/mshkf7/pix2pix-portraits-dataset/output)
# Веб-версия
Готовая версия лежит на хостинге [hugging face](https://huggingface.co/spaces/Mshkf/Sketch_2_portrait_pix2pix), можно загрузить и скачать фотографию или просто выбрать пример

Рисунки можно взять [здесь](https://github.com/Mshkf/pix2pix/blob/main/edges.zip), а если рисовать самому, то нужно чётко следовать шаблону: чёткие линии "фломастером" в правильных масштабах, губы и нас как на картинках, обязательно брови

Лучше всего работала неправильная версия, где применяется батчнорм на размере 1 на 1 (что не удивительно, так как у авторов статьи тот же косяк), вот [реализация](https://github.com/Mshkf/pix2pix/blob/main/app.py) веба и для него брались веса из 100-й эпохи [старого](https://www.kaggle.com/code/mshkf7/pix2pix-portraits-dataset/output?scriptVersionId=161597314) ноутбука


