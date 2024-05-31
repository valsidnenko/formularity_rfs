
# Formularity RFS — пакет для автоматической оценки формульности фольклорных песен на русском языке

Пакет создан для поиска «формул» в фольклорных песнях и для оценки формульности народных лирических произведений. Под «формулами» понимаются повторяющиеся элементы, присущие фольклорным песням на русском языке, по А. Дауру: *«под понятие формульности следует подводить все то, что, часто повторяясь, заключает в себе понятие типического для жанра»*. 

Пакет рассчитает коэффициент формульности ваших текстов и на выходе предоставит размеченный файл для визуальной демонстрации «формул» в текстах.

## Содержание
- [Коэффициент формульности](#коэффициент-формульности)
- [Технологии и инструменты](#технологии-и-инструменты)
- Подготовка данных
- Начало работы
- Результаты
- Предложить изменения
- Контакты
- Подробнее

## Коэффициент формульности

Коэффициент формульности состоит из нескольких элементов: 
1) коэффициент VocD 
2) коэффициент для фразеологизмов
3) коэффициент для междометий
4) коэффициент для "биномов"
5) коэффициент для n-грамм

**Коэффициент VocD** — это коэффициент лексического разнообразия текста, усовершенствованная версия критикуемого коэффициента TTR (отношение повторяющихся лемм к общему количеству слов в тексте). В коде VocD рассчитывается как среднее из 1000 значений на отрезках из 35 лексических единиц текстов. Именно поэтому **тексты, содержащие менее 35 слов, нельзя проанализировать этим пакетом**.

**Коэффициент для фразеологизмов** — это добавочный коэффициент для текстов, содержащих «формулы»-фразеологизмы. Такие «формулы» упоминаются в работах *А.Т. Хроленко «Поэтическая фразеология русской народ­ной лирической песни»* как «устойчивые словесные комплексы», *Б. Фера «Die formelhaften Elemente in den alten englischen Balladen»* как «формульные словосочетания», а также *Г.И. Мальцева «Русские народные формулы в необрядовой лирике»*. Формулы в текстовом файле пакета [collocations.txt](https://github.com/valsidnenko/formularity_rfs/blob/main/collocations.txt) были получены благодаря анализу повторяющихся словосочетаний в русских народных песнях, которые были получены путем парсинга страниц [отсюда](https://ru.wikisource.org/wiki/Русские_народные_песни). В файл с фразеологизмами вошли, например, *«добрый молодец», «красная девица»* и другие. **Список найденных фразеологизмов будет доступен в окне выходных данных, а их разметка — в файле results.html на выходе. Каждый уникальный фразеологизм в тексте прибавляет 0.1 к итоговому коэффициенту.**

**Коэффициент для междометий** — это добавочный коэффициент для текстов, содержащих междометия. Междометия, повторяемые с определенной периодичностью, служат своего рода «структурными элементами» песни, не только выражая эмоции, но и помогая «сказителям» начинать песню и поддерживать ее ритм. *С.Б. Адоньева в работе «Звуковые формулы в ритуальном фольклоре»* определяет звуковые комплексы, устойчиво сохраняющиеся в песенных фольклорных произведениях и выполняющие в них определенные конструктивные функции, которые оказываются устойчивыми в отношении набора и последовательности звуков, ритмической организации, позиции в тексте. **Список найденных междометий будет доступен в окне выходных данных, а их разметка — в файле results.html на выходе. Каждое уникальное междометие в тексте прибавляет 0.1 к итоговому коэффициенту.**

**Коэффициент для «биномов»** — это добавочный коэффициент для текстов, содержащих сложные слова с дефисным написанием. Под «биномами» понимаются все слова с дефисным написанием кроме местоимений. Такие «биномы», с одной стороны, позволяют «сказителям» применить сравнение в ограниченных условиях ритма (*«Волга матушка-река»*), а с другой — поддержать звуковую и ритмическую симметрию (*«Поутру раным-раненько,/ По утру раным-ранешенько»*) , иногда — дуальность (*«Красавица-изменница», «ласки-взоры»*) и параллелизм (*«Он кудрями, кудрями/ Потрясёт-потрясёт, / Жницам горелочки/ Поднесёт-поднесёт»*). **Список найденных «биномов» будет доступен в окне выходных данных, а их разметка — в файле results.html на выходе. Каждый уникальный «бином» в тексте прибавляет 0.1 к итоговому коэффициенту.**

**Коэффициент для n-грамм** — это добавочный коэффициент для текстов, содержащих повторяющуюся последовательность от 2 до 14 лемм. Алгоритм для автоматического выявления формул в поэтическом тексте был взят из [доклада Дмитрия Николаева](https://www.academia.edu/6304149/). Стоп-лист исследователя был расширен — туда вошли и междометия для корректой работы пакета. **Стоп-лист ([stopwords.txt](https://github.com/valsidnenko/formularity_rfs/blob/main/stopwords.txt)), как и список фразеологизмов ([collocations.txt](https://github.com/valsidnenko/formularity_rfs/blob/main/collocations.txt)), необходимы для начала работы с пакетом.** Пакет ищет n-граммы от большего к меньшему, исключая вхождение одного и того же слова в несколько n-грамм. **Список найденных n-грамм и их разметка будет доступна в файле results.html на выходе. Каждая уникальная n-грамма в тексте прибавляет 0.1 к итоговому коэффициенту.**

Таким образом, итоговый коэффициент формульности фольклорной песни можно описать формулой: **Коэффициент VocD + 0.1 * (кол-во фразеологизмов) + 0.1 * (кол-во междометий) + 0.1 * (кол-во «биномов») + 0.1 * (кол-во уникальных n-грамм).**

## Технологии и инструменты

Проект использует ряд библиотек и инструментов для обработки текстов, включая:

- *Python 3* - язык программирования проекта.
- *spaCy* - библиотека для обработки естественного языка, используемая для токенизации и определения частей речи (POS-тегов).
- *ru_core_news_sm* — модель spaCy для русского языка.
- *pymorphy2* - библиотека для морфологического анализа русского языка, используемая для лемматизации слов.
- *NumPy* - библиотека для работы с массивами и числовыми данными.
- *scikit-learn* - библиотека для машинного обучения, используется для поиска n-грамм.
- *collections* - стандартный модуль Python, используемый для работы с коллекциями данных.
- *re (регулярные выражения)* - для поиска и замены текста по шаблону.
- *os* - для работы с файловой системой.

## Подготовка данных

Перед началом работы с пакетом необходимо подготовить ваши данные для анализа. Вы можете создать файл songs.txt или воспользоваться пустым файлом songs.txt, который будет создан при первом запуске пакета. Данные следует оформить следующим образом:

| ✅                                                             | ❌                                                              |
|---------------------------------------------------------------|----------------------------------------------------------------|
| Ой, мороз-мороз, не морозь меня, Не морозь меня, моего коня.  | Ой, мороз-мороз, не морозь меня, 
Не морозь меня, моего коня.  |
| Конец одной песни + Начало новой песни                        | Конец одной песни Начало новой песни                              |


