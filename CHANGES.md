# Changes

Журнал доработок поверх исходного состояния `add gan` (commit `23bc609`).

## 1. Воспроизводимость пайплайна

- Добавлен пример набора данных `data/synthetic_training_data.csv` (15 строк,
  колонки `cement, sand, gravel, water, strength_28`) и исключение в
  `.gitignore`, чтобы образец трекался, а артефакты экспериментов — нет.
- В `requirements.txt` добавлен `scikit-learn` — нужен для TSNE-визуализации
  (`materialgen/visualization.py`).
- Пакет `materialgen/__init__.py` переписан на ленивые экспорты:
  `import materialgen` больше не тянет PyTorch/Pyro, символы (`run_train_neat`,
  `run_make_neat_to_bnn`, `run_train_gan`, `run_evaluate_metrics`)
  подгружаются по обращению.
- `main_work.py` запускает все четыре стадии CLI подряд одной командой
  (`python main_work.py`).
- `README.md` и `main.py` приведены в соответствие с фактическим CLI.

## 2. Стадия `train_gan` подключена к обученной BNN

Раньше дискриминатор учился на «фейках» от заглушки `torch.rand(...)`. Сейчас:

- В `materialgen/train_gan.py` стадия загружает `bnn_model.pt` из
  `artifacts/make_neat_to_bnn/`, читает манифест и проверяет, что порядок
  колонок `components` / `properties` в конфиге совпадает с манифестом
  (`_alignment_check`). При расхождении — явная ошибка, чтобы не получить
  «молчаливо неправильный» прогон.
- Замороженная NEAT→BNN используется как генератор: по каждой пачке свойств
  возвращает MC-среднее предсказание состава.
- `examples/train_gan.json` приведён к тому же CSV и тем же колонкам, что
  `backward.json` / `make_neat_to_bnn.json`. Добавлены параметры
  `random_seed`, `generator_mc_samples`, `bnn_model_filename` для
  воспроизводимости.
- Вход дискриминатора стандартизуется через `StandardScaler` по обучающему
  фолду на конкатенации `[components, properties]`. Без этого BCELoss
  насыщался (наблюдалось `D_loss ≈ 70`), теперь сходится в районе
  теоретического равновесия `2·ln2 ≈ 1.39`. Параметры скейлера сохраняются в
  `artifacts/train_gan/input_scaler.json`.
- Сохранение артефактов: `discriminator.pt`, `history.json`,
  `gan_generator_meta.json` (ссылается на `bnn_model.pt` и манифест),
  `train_gan.json` (сводка).

## 3. Метрики кейса

Добавлен модуль `materialgen/evaluate_metrics.py` и CLI-команда
`evaluate_metrics` (см. `examples/evaluate_metrics.json`):

- Загружает `bnn_model.pt`.
- Делает MC-предсказания на отложенной доле данных (`validation_split`).
- Считает MAE, RMSE, MAPE, R² по каждому компоненту и агрегаты,
  плюс среднее `pred_std` по MC-сэмплам — оценка неопределённости.

Это закрывает требования кейса по метрикам (MAE/RMSE/MAPE/R²) и даёт явный
способ сравнивать варианты обучения.

## 4. Конфиги для быстрой проверки

`examples/*_smoke.json` — урезанные параметры (меньше эпох, генераций,
mc_samples), чтобы прогон всех четырёх стадий на синтетике укладывался в
~1 минуту:

```bash
python main.py train_neat        --config examples/backward_smoke.json
python main.py make_neat_to_bnn  --config examples/make_neat_to_bnn_smoke.json
python main.py train_gan         --config examples/train_gan_smoke.json
python main.py evaluate_metrics  --config examples/evaluate_metrics.json
```

## 5. Тесты

`tests/test_end_to_end.py` обновлён под актуальный CLI: проверяет, что в
парсере зарегистрированы все четыре подкоманды, и что для каждой из них есть
JSON-пример.

## 6. Поведение «GAN» — как читать результат

Архитектура — не классический GAN со взаимным обучением, а **дискриминатор
реализма**: генератор (BNN) заморожен, а дискриминатор учится отличать
реальные пары `(состав, свойство)` от предсказаний BNN. Возможные исходы:

| Состояние BNN          | `acc_real` | `acc_fake` | Интерпретация                          |
|-----------------------|-----------:|-----------:|----------------------------------------|
| Хорошо обучена         | ~0.5       | ~0.5       | Дискриминатор не отличает → реалистично |
| Плохо/не обучена       | ~1.0       | ~1.0       | Подделки распознаются легко             |
| Численная поломка      | колебания  | колебания  | Признак бага (был до фикса входа)       |

На `data/synthetic_training_data.csv` (15 строк) BNN обучается на тех же
данных, поэтому на дымовом прогоне дискриминатор естественно сходится к ~0.5
по обоим классам — это успех генератора, а не провал GAN.

## 7. Что осталось «бонусом»

- Прогон с реальным датасетом GlavStroy (когда будет доступен), включая
  `time` / прочность во времени.
- Multi-output (например, удобоукладываемость) — `dataset.properties` уже
  принимает список любой длины.
- Сравнение BNN с базовым регрессором (например, GradientBoosting) на тех
  же `evaluate_metrics` для отчёта.
