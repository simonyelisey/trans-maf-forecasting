# trans-maf-forecasting

## Описание задачи

Прогнозирование многомерных временных рядов с помощью условных нормализующих потоков [Trans-MAF](https://arxiv.org/pdf/2002.06103.pdf). 

В ходе исследования проводятся следующие эксперименты:
  - прогнозирование регулярных временных рядов с использованием модели Trans-MAF;
  - сравнение Trans-MAF с диффузионной моделью для прогнозирования временных рядов [TimeGrad](https://arxiv.org/abs/2305.00624.pdf);
  - прогнозирование нерегулярных временных рядов с долей пропусков в данных от 10% до 80%.


## Данные

Используются датасеты из открытых источников:

1. Solar:
   - гранулярность: H;
   - число компонент: 137;
   - обучающая выборка: 2006-01-01 00:00:00 - 2006-10-20 00:00:00 (7033 точки);
   - тестовая выборка: 2006-10-20 01:00:00 - 2006-10-27 00:00:00 (168 точек);
   - длина прогноза: 24.
3. Electricity:
   - гранулярность: H;
   - число компонент: 370;
   - обучающая выборка: 2014-03-19 09:00 - 2014-09-01 00:00 (4000 точки);
   - тестовая выборка: 2014-09-01 01:00 - 2014-09-08 00:00:00 (168 точек);
   - длина прогноза: 24.
5. Exchange:
   - гранулярность: B;
   - число компонент: 8;
   - обучающая выборка: 1990-01-01 - 2013-04-08 (6101 точки);
   - тестовая выборка: 2013-04-09 - 2013-11-04 (150 точек);
   - длина прогноза: 30.

## Итоговый продукт

Итоговым продуктом является библиотека на языке Python для прогноза многомерных рядов с помощью модели Trans-MAF.

## План работы
1. [2024-02-29] - Завершение всех экспериментов;
2. [2024-03-14] - Создание библиотеки;
3. [2024-04-31] - Завершение оформления статьи.
   
## Контакты

**Автор работы**: Елисеев Семен, студент НИУ ВШЭ, tg: @simonyelisey;

**Научный руководитель**: Гущин Михаил, старший научный сотрудник НИУ ВШЭ, tg: @mikhail_h91
