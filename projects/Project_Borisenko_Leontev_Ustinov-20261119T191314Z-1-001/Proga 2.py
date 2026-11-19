import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

# Настройка визуального стиля для  графиков
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (14, 8)


# ЧАСТЬ 1: ЗАГРУЗКА И ОЧИСТКА ДАННЫХ
print("Скачивание и подготовка данных")
path = kagglehub.dataset_download("franciscogcc/financial-data")

# Находим CSV файл
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))

# Превращаем дату в правильный формат и делаем индексом
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Заполняем дырки (протягиваем данные вперед) и удаляем пустые строки
df_clean = df.ffill().dropna()

# Выбираем главные активы для анализа
main_assets = ['sp500 close', 'nasdaq close', 'gold close', 'oil close', 'eur_usd']
print(f"Период: {df_clean.index.min().date()} — {df_clean.index.max().date()}")

# ЧАСТЬ 2: АНАЛИТИКА (EDA) И МАТРИЦА

print("\n Строим Матрицу Корреляций и Анализ")

#ГРАФИК 1: Матрица Корреляций
plt.figure(figsize=(10, 8))
correlation_matrix = df_clean[main_assets].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", center=0)
plt.title('Матрица корреляций: Связи активов', fontsize=16)
plt.show()

#ГРАФИК 2: Сравнение доходности (Нормализация)
# Приводим все цены к старту = 1.0
normalized_df = df_clean[main_assets] / df_clean[main_assets].iloc[0]

plt.figure(figsize=(14, 6))
sns.lineplot(data=normalized_df)
plt.title('Сравнение доходности активов (Рост с единого старта)', fontsize=16)
plt.ylabel('Рост капитала (x раз)')
plt.show()

# ЧАСТЬ 3:AI
# Создаем копию данных для ИИ
data_ai = df_clean.copy()

# ИНЖЕНЕРИЯ ПРИЗНАКОВ (Учим ИИ видеть тренды)
# 1. Считаем ДОХОДНОСТЬ (Процент изменения за день). ИИ легче понять %, чем цену.
data_ai['Gold_Return'] = data_ai['gold close'].pct_change()
data_ai['Oil_Return'] = data_ai['oil close'].pct_change()
data_ai['SP500_Return'] = data_ai['sp500 close'].pct_change()
data_ai['Dollar_Return'] = data_ai['eur_usd'].pct_change()

# 2. ТРЕНД (Скользящая средняя за 20 дней)
data_ai['SMA_20'] = data_ai['gold close'].rolling(window=20).mean()
# 3. ОТКЛОНЕНИЕ (Насколько цена оторвалась от средней)
data_ai['Dist_from_SMA'] = data_ai['gold close'] / data_ai['SMA_20'] - 1
# 4. ВОЛАТИЛЬНОСТЬ
data_ai['Volatility'] = data_ai['gold close'].rolling(window=5).std()

#ЦЕЛЬ
# Мы хотим предсказать Gold_Return на ЗАВТРА (сдвигаем колонку назад на -1)
data_ai['Target_Next_Day_Return'] = data_ai['Gold_Return'].shift(-1)

# Удаляем пустые строки, появившиеся после расчетов
data_ai = data_ai.dropna()

# Выбираем факторы, на которые будет смотреть ИИ
features = ['Gold_Return', 'Oil_Return', 'SP500_Return', 'Dollar_Return', 'Dist_from_SMA', 'Volatility']
X = data_ai[features]
y = data_ai['Target_Next_Day_Return']

# Делим данные: 95% на обучение, 5% на тест (экзамен)
split = int(len(data_ai) * 0.95)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# Обучаем модель (Random Forest)
# max_depth=5 ограничивает дерево ИИ, чтобы он не заучивал шум
model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=69)
model.fit(X_train, y_train)

# ЧАСТЬ 4: ПРОВЕРКА ИИ И ГРАФИКИ
# ИИ делает прогноз (он выдает ПРОЦЕНТЫ изменения)
predicted_returns = model.predict(X_test)

# ВОССТАНОВЛЕНИЕ ЦЕН
# Чтобы нарисовать график, превращаем % обратно в цену ($)
# Цена Завтра = Цена Сегодня * (1 + Прогноз ИИ)
last_known_prices = data_ai['gold close'].iloc[split:] # Цена "сегодня" для теста
predicted_prices_curve = last_known_prices * (1 + predicted_returns)

# Реальная цена "завтра" (для проверки)
real_prices_curve = df_clean['gold close'].shift(-1).loc[X_test.index]

# ОЦЕНКА ТОЧНОСТИ НАПРАВЛЕНИЯ
# Угадал ли ИИ знак? (Рост или Падение)
true_direction = np.sign(y_test)
pred_direction = np.sign(predicted_returns)
acc = accuracy_score(true_direction, pred_direction)

print(f" Точность угадывания направления: {acc*100:.2f}%")
print("(Если > 50%, значит ИИ работает лучше, чем подбрасывание монетки)")

# ГРАФИК ПРОГНОЗА
plt.figure(figsize=(15, 7))

# Рисуем последние 150 дней для наглядности
subset = -150
plt.plot(real_prices_curve.index[subset:], real_prices_curve.iloc[subset:], label='РЕАЛЬНАЯ ЦЕНА', color='blue', linewidth=2)
plt.plot(real_prices_curve.index[subset:], predicted_prices_curve.iloc[subset:], label='ПРОГНОЗ ИИ', color='red', linestyle='--', linewidth=2)

plt.title(f'ИИ против Рынка (Прогноз Золота). Точность направления: {acc*100:.1f}%', fontsize=16)
plt.xlabel('Дата')
plt.ylabel('Цена золота ($)')
plt.legend()
plt.show()

# ГРАФИК ВАЖНОСТИ ФАКТОРОВ
plt.figure(figsize=(10, 5))
pd.Series(model.feature_importances_, index=features).sort_values().plot(kind='barh', color='purple')
plt.title('Важность факторов', fontsize=16)
plt.show()