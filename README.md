
# Formularity RFS — пакет для автоматической оценки формульности фольклорных песен на русском языке

Пакет создан для поиска «формул» в фольклорных песнях и для оценки формульности народных лирических произведений. Под «формулами» понимаются повторяющиеся элементы, присущие фольклорным песням на русском языке, по А. Дауру: *«под понятие формульности следует подводить все то, что, часто повторяясь, заключает в себе понятие типического для жанра»*. 

Пакет рассчитает коэффициент формульности ваших текстов и на выходе предоставит размеченный файл для визуальной демонстрации «формул» в текстах.

## Содержание
- [Коэффициент формульности](#коэффициент-формульности)
- [Технологии и инструменты](#технологии-и-инструменты)
- [Подготовка данных](#подготовка-данных)
- [Установка и начало работы](#установка-и-начало-работы)
- [Результаты](#результаты)
- [Предложить изменения](#предложить-изменения)
- [Контакты](#контакты)
- [Подробнее](#подробнее)

## Коэффициент формульности

Коэффициент формульности состоит из нескольких элементов: 
1) коэффициент VocD; 
2) коэффициент для фразеологизмов;
3) коэффициент для междометий;
4) коэффициент для «биномов»;
5) коэффициент для n-грамм.

**Коэффициент VocD** — это коэффициент лексического разнообразия текста, усовершенствованная версия критикуемого коэффициента TTR (отношение повторяющихся лемм к общему количеству слов в тексте). В коде VocD рассчитывается как среднее из 1000 значений на отрезках из 35 лексических единиц текстов. Именно поэтому **тексты, содержащие менее 35 слов, нельзя проанализировать этим пакетом**.

**Коэффициент для фразеологизмов** — это добавочный коэффициент для текстов, содержащих «формулы»-фразеологизмы. Такие «формулы» упоминаются в работах *А.Т. Хроленко «Поэтическая фразеология русской народ­ной лирической песни»* как «устойчивые словесные комплексы», *Б. Фера «Die formelhaften Elemente in den alten englischen Balladen»* как «формульные словосочетания», а также *Г.И. Мальцева «Русские народные формулы в необрядовой лирике»*. Формулы в текстовом файле пакета [collocations.txt](https://github.com/valsidnenko/formularity_rfs/blob/main/collocations.txt) были получены благодаря анализу повторяющихся словосочетаний в русских народных песнях, которые были получены путем парсинга страниц [отсюда](https://ru.wikisource.org/wiki/Русские_народные_песни). В файл с фразеологизмами вошли, например, *«добрый молодец», «красная девица»* и другие. **Список найденных фразеологизмов будет доступен в окне выходных данных, а их разметка — в файле results.html на выходе. Каждый уникальный фразеологизм в тексте прибавляет 0.1 к итоговому коэффициенту.**

**Коэффициент для междометий** — это добавочный коэффициент для текстов, содержащих междометия. Междометия, повторяемые с определенной периодичностью, служат своего рода «структурными элементами» песни, не только выражая эмоции, но и помогая «сказителям» начинать песню и поддерживать ее ритм. *С.Б. Адоньева в работе «Звуковые формулы в ритуальном фольклоре»* определяет звуковые комплексы, устойчиво сохраняющиеся в песенных фольклорных произведениях и выполняющие в них определенные конструктивные функции, которые оказываются устойчивыми в отношении набора и последовательности звуков, ритмической организации, позиции в тексте. **Список найденных междометий будет доступен в окне выходных данных, а их разметка — в файле results.html на выходе. Каждое уникальное междометие в тексте прибавляет 0.1 к итоговому коэффициенту.**

**Коэффициент для «биномов»** — это добавочный коэффициент для текстов, содержащих сложные слова с дефисным написанием. Под «биномами» понимаются все слова с дефисным написанием кроме местоимений. Такие «биномы», с одной стороны, позволяют «сказителям» применить сравнение в ограниченных условиях ритма (*«Волга матушка-река»*), а с другой — поддержать звуковую и ритмическую симметрию (*«Поутру раным-раненько,/ По утру раным-ранешенько»*) , иногда — дуальность (*«Красавица-изменница», «ласки-взоры»*) и параллелизм (*«Он кудрями, кудрями/ Потрясёт-потрясёт, / Жницам горелочки/ Поднесёт-поднесёт»*). **Список найденных «биномов» будет доступен в окне выходных данных, а их разметка — в файле results.html на выходе. Каждый уникальный «бином» в тексте прибавляет 0.1 к итоговому коэффициенту.**

**Коэффициент для n-грамм** — это добавочный коэффициент для текстов, содержащих повторяющуюся последовательность от 2 до 14 лемм. Алгоритм для автоматического выявления формул в поэтическом тексте был взят из [доклада Дмитрия Николаева](https://www.academia.edu/6304149/). Стоп-лист исследователя был расширен — туда вошли и междометия для корректой работы пакета. **Стоп-лист ([stopwords.txt](https://github.com/valsidnenko/formularity_rfs/blob/main/stopwords.txt)), как и список фразеологизмов ([collocations.txt](https://github.com/valsidnenko/formularity_rfs/blob/main/collocations.txt)), необходимы для начала работы с пакетом.** Пакет ищет n-граммы от большего к меньшему, исключая вхождение одного и того же слова в несколько n-грамм. **Список найденных n-грамм и их разметка будет доступна в файле results.html на выходе. Каждая уникальная n-грамма в тексте прибавляет 0.1 к итоговому коэффициенту.**

Таким образом, итоговый коэффициент формульности фольклорной песни можно описать формулой: **Коэффициент VocD + 0.1 * (кол-во фразеологизмов) + 0.1 * (кол-во междометий) + 0.1 * (кол-во «биномов») + 0.1 * (кол-во уникальных n-грамм).**

## Технологии и инструменты

Проект использует ряд библиотек и инструментов для обработки текстов, включая:

- *Python 3* — язык программирования проекта.
- *spaCy* — библиотека для обработки естественного языка, используемая для токенизации и определения частей речи (POS-тегов).
- *ru_core_news_sm* — модель spaCy для русского языка.
- *pymorphy2* — библиотека для морфологического анализа русского языка, используемая для лемматизации слов.
- *NumPy* — библиотека для работы с массивами и числовыми данными.
- *scikit-learn* — библиотека для машинного обучения, используется для поиска n-грамм.
- *collections* — стандартный модуль Python, используемый для работы с коллекциями данных.
- *re (регулярные выражения)* — для поиска и замены текста по шаблону.
- *os* — для работы с файловой системой.

## Подготовка данных

Перед началом работы с пакетом необходимо подготовить ваши данные для анализа. Вы можете создать файл songs.txt или воспользоваться пустым файлом songs.txt, который будет создан при первом запуске пакета. Данные следует оформить, используя знак «+» между разными песнями.

✅ Конец одной песни + Начало новой песни

❌ Конец одной песни Начало новой песни

Также учитывайте, что пакет не сможет проанализировать тексты, содержащие менее 35 слов.

## Установка и начало работы

**1) Перед установкой пакета загрузите библиотеки, которые он использует.**

Код для облачной среды Google Colab:

    !pip install -U spacy
    !pip install -U spacy-lookups-data
    !python -m spacy download ru_core_news_sm
    !pip install pymorphy2

**2) Установите пакет formularity-rfs.**

Код для облачной среды Google Colab:
    
    !pip install formularity-rfs==0.3

или

    !pip install --upgrade git+https://github.com/valsidnenko/formularity_rfs.git

**3) Загрузите необходимые файлы для работы пакета collocations.txt и stopwords.txt вручную или с помощью команд.**

Код для облачной среды Google Colab:

    !wget https://raw.githubusercontent.com/valsidnenko/formularity_rfs/main/stopwords.txt
    !wget https://raw.githubusercontent.com/valsidnenko/formularity_rfs/main/collocations.txt

**4) Импортируйте загруженные библиотеки.**

Код для облачной среды Google Colab:

    import re
    import spacy
    import pymorphy2
    import numpy as np
    from collections import defaultdict
    from sklearn.feature_extraction.text import CountVectorizer
    nlp = spacy.load("ru_core_news_sm")

**5) Импортируйте функцию анализа из пакета formularity_rfs.**

Код для облачной среды Google Colab:

    from formularity.your_module import analyse

**6) Введите команду для анализа текстов.**

Код для облачной среды Google Colab:

    analyse()

**Что произойдет дальше?**

- Если у вас не было файла songs.txt, такой пустой файл будет создан в течение минуты. Туда следует добавить тексты согласно [инструкции](#подготовка-данных), а затем вновь вызвать команду analyse().
- Если файл songs.txt содержит менее 35 слов, пакет не сможет проанализировать его и появится текст *«Невозможно посчитать VOCD»*. Почему так происходит — [здесь](#коэффициент-формульности).
- Если песни из songs.txt были проанализированы, в выходных данных для каждой песни вы увидите:

  1) список лемматизированных слов
  2) список найденных междометий
  3) список найденных «биномов»
  4) список найденных фразеологизмов
  5) список найденных n-грамм
  6) коэффициент для междометий
  7) коэффициент для фразеологизмов
  8) коэффициент для «биномов»
  9) коэффициент для n-грамм
  10) коэффициент VOCD
  
В конце будет надпись *«Результаты записаны в файл results.html, он появится в течение минуты.»* Подробнее о результатах можно узнать [здесь](#результаты).

## Результаты

В размеченном файле results.html на выходе вы увидите результаты анализа песен и коэффициенты их формульности. 

🟢 <span style="color: green;">Зелёным цветом</span> будут выделены найденные n-граммы. Для точности они также будут продублированы ниже списком.

🟣 <span style="color: purple;">Фиолетовым цветом</span> будут выделены найденные «биномы».

🔴 <span style="color: red;">Красным цветом</span> будут выделены найденные междометия.

🔵 <span style="color: blue;">Синим цветом</span> будут выделены найденные фразеологизмы.

Подробнее о том, что означает коэффициент и разметка, можете узнать [здесь же](#коэффициент-формульности) или в [тексте моей магистерской диссертации](#подробнее).

## Предложить изменения

Буду рада вашим предложениям и улучшениям для этого проекта! Пожалуйста, следуйте нижеприведённым шагам, чтобы предложить свои изменения:

1. Форкните этот репозиторий. Это можно сделать, нажимая кнопку «Fork» в верхней части страницы репозитория на GitHub.
   
2. Клонируйте форкнутый репозиторий на ваш локальный компьютер:

Код:

    git clone https://github.com/valsidnenko/formularity_rfs.git
   
3. Создайте новую ветку для ваших изменений. Рекомендуется назвать ветку так, чтобы отражать суть изменения:

Код:

    git checkout -b имя-вашей-ветки
   
4. Внесите изменения и закоммитьтесь:

Код:

    git add .
    git commit -m "Краткое описание ваших изменений"
   
5. Запушьте ваши изменения на ваш форк:

Код:

    git push origin имя-вашей-ветки
   
6. Перейдите на страницу вашего форка на GitHub и создайте Pull Request. Подробнее о том, как это сделать, можно прочитать в [официальной документации GitHub](https://docs.github.com/en/pull-requests).

Пожалуйста, обязательно включите следующие детали в ваш Pull Request:

- Краткое описание изменений.
- Обоснование, почему эти изменения необходимы.
- Любые связанные вопросы.

Спасибо за вклад в развитие проекта!

**Если у вас возникнут какие-либо вопросы или затруднения, не стесняйтесь обращаться ко мне [через контакты](#контакты).**

## Контакты

По любым вопросам и предложениям можно написать мне:

- на почту: valsidnenko@gmail.com
- в Telegram: @necha1

## Подробнее

Здесь будет ссылка на работу, в рамках которой был сделан этот пакет!
