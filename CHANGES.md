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

## 7. Forward-стадия: `состав → прочность` (well-determined регрессия)

Поскольку обратная задача `1 свойство → 4 компонента` принципиально
underdetermined (R²≈0 на 192 строках), добавлена **прямая** ветка пайплайна.

- Новый модуль `materialgen/forward_model.py` — `ForwardBayesianMLP`
  (PyroModule с Normal-приорами на все веса) + `ForwardBNNRegressor`
  (scaling, fit с SVI/AutoNormal, MC-prediction, save/load).
- Новая стадия `materialgen/train_forward.py` (CLI: `train_forward`) —
  обучает forward BNN, считает MAE/RMSE/MAPE/R² на train/val.
- Гиперпараметры, благодаря которым модель действительно сходится на
  средних датасетах (>1k строк):
  - `kl_weight` (= `1 / likelihood_scale`): дата-член ELBO усиливается,
    чтобы KL на весах не давил likelihood. Без этого модель остаётся у
    приора и R²≈0.
  - `sigma` шума наблюдения — обучаемый `pyro.param`, не сэмпл из
    HalfNormal. Стабильнее на структурированных данных.
  - `Trace_ELBO(num_particles=2)` для менее шумного градиента.
- `_predict_scaled` использует ручной guide-replay цикл, а не
  `Predictive(...)`: параллельная развёртка Predictive ломает
  PyroSample-обёрнутые `Linear` после `load()` (лишняя batch-dim).

Результаты:

- `forward_normal.json` (Normal_Concrete_DB.csv, 2412 строк, 9 фич →
  CS_28d): **val R²=0.879, MAE=4.23 МПа**. Sklearn baselines на том же
  сплите: Ridge 0.47, RandomForest 0.905, GBR 0.898 — BNN наравне и плюс
  калиброванная неопределённость.
- `forward_synthetic.json` (synthetic_training_data.csv, 192 строки,
  14 фич → 4 таргета `strength_1/_3/_7/_28`):
  **val R²=0.974, MAPE=1.88%**. Закрывает бонус ТЗ «прочность во
  времени» одной multi-output моделью.

## 8. Поддержка проблемных CSV

- `DatasetInputConfig` получил параметры `skiprows` (для CSV с
  комментариями в шапке, как `Normal_Concrete_DB.csv`) и
  `component_aliases` (переименование колонок при чтении).
- `_select_numeric_columns` теперь толерантен к NBSP / пробелам в
  числах-тысячах (`"1 013,00"`) и десятичной запятой при любом
  обнаруженном dtype колонки (StringDtype не пускался прежней проверкой
  `dtype == object`).

## 9. ГОСТ-валидация (`validate_gost`)

Новый модуль `materialgen/validate_gost.py` + CLI `validate_gost`
(`examples/validate_gost.json`). Загружает forward-модель, читает
`data/ГОСТы.csv` (B3.5–B60, диапазоны `R_min ≤ R_сж ≤ R_max`), для каждой
строки датасета считает:

- предсказанная и истинная марка по `[R_min, R_max]`;
- `exact_match` (точная марка);
- `within_one_class` (±1 марка);
- `predictive_2sigma_coverage` (доля случаев, когда истинная прочность
  попадает в `[μ−2σ, μ+2σ]` модели).

Сохраняет per-row `predictions.csv` и confusion-матрицу.

На `Normal_Concrete_DB.csv`: exact_match = **58.0%**,
within ±1 brand = **92.5%**.

## 10. Что осталось «бонусом»

- Transfer-learning: продолжить `synthetic_training_data` с инициализации
  весов из `Normal_Concrete_DB`-модели (поле `pretrained_model_path` уже
  предусмотрено в `ForwardStageConfig`).
- Калибровка предиктивной неопределённости (scale-tempering posterior
  или MC-Dropout / NUTS), чтобы 2σ-coverage приблизился к ~95%.
- Multi-output на `Normal_Concrete_DB` (Slump + CS_28d) на подмножестве из
  873 строк, где есть оба таргета.
