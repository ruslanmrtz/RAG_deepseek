# 🧠 RAG-Система на основе RoSBERTa и DeepSeek

Этот проект реализует Retrieval-Augmented Generation (RAG) систему, которая позволяет отвечать на вопросы, опираясь на внешние данные — вашу базу знаний.  
Используется векторная БД Qdrant, модель RoSBERTa для эмбеддингов и API от DeepSeek в качестве генеративной модели.

---

## 📁 Структура проекта

```
rag_system/
├── README.md
├── main.py                 # Основной модуль с логикой вопрос-ответ
├── data.py                 # Работа с векторной БД Qdrant и поиском контекста
├── config.py               # Конфигурационные данные (например, токен)
├── RoSBERTa.py             # Модель для получения эмбеддингов текста
└── result.json             # Пример данных для загрузки в БД
```

---

## 🔧 Технологии

- **Python 3.10+**
- **Qdrant** — векторная БД для хранения эмбеддингов
- **DeepSeek API** — генерация ответов
- **RoSBERTa (ai-forever/ru-en-RoSBERTa)** — модель для создания эмбеддингов
- **LangChain TextSplitter** — разделение текста на чанки
- **NLTK** — предобработка текста (опционально)

---

## 🚀 Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/yourusername/rag_system.git
cd rag_system
```

### 2. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 3. Настройка окружения

Создайте файл `config.py`:

```python
deepseek_token = "ваш_токен_от_DeepSeek"
```

> Получить токен можно на [https://platform.deepseek.com](https://platform.deepseek.com)

### 4. Запуск Qdrant (локально)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 5. Подготовка данных

Поместите данные в формате JSON в файл `result.json`.  
Пример структуры:

```json
{
  "messages": [
    {
      "id": 1,
      "text_entities": [{"text": "Текст документа..."}]
    }
  ]
}
```

### 6. Загрузка данных в Qdrant

Запустите:

```bash
python data.py
```

Он проиндексирует все документы из `result.json`.

### 7. Запуск RAG-системы

```bash
python main.py
```

---

## 💬 Пример использования

```python
prompt = "Как заработать миллион молодому человеку?"
context = data.find_text(prompt)
response = ask_context(prompt, context)
print(response)
```

Ответ будет сгенерирован на основе найденного контекста из вашей базы знаний.

---

## 📌 Возможности расширения

- Добавление поддержки других LLM (например, YandexGPT, GPT, Claude)
- Интеграция с Telegram / Web-интерфейсом
- Использование более сложного ранжирования при поиске
- Логирование запросов и ответов
- Асинхронная обработка

---

## 🤝 Автор

Ruslan Murtazin

📧 ruslanmrtzn@yadnex.ru
