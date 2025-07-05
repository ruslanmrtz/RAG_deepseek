from openai import OpenAI

import data
from config import deepseek_token


def simple_ask(prompt):
    client = OpenAI(api_key=deepseek_token,
                    base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content


def ask_context(prompt, context):
    system_prompt = f"""
    Ты профессиональный продавец-консультант компании. Отвечай ТОЛЬКО на основе предоставленного контекста. 
    Если информации недостаточно – вежливо откажись отвечать. Сохраняй дружелюбный и уверенный тон.

    **Роль:**
    - Эксперт по продуктам/услугам компании
    - Мастер вежливого общения
    - Специалист по решению проблем клиентов

    **Инструкции:**
    1. Анализируй контекст из базы знаний: <CONTEXT_START>{context}<CONTEXT_END>
    2. Отвечай максимально конкретно на вопрос: {prompt}
    3. Если в контексте нет ответа: "Извините, эта информация временно недоступна. Уточните детали у менеджера"
    4. Для сложных вопросов разбивай ответ на пункты
    5. Избегай технического жаргона

    **Стиль ответа:**
    - Используй эмоджи для эмоциональной окраски (1-2 в сообщении)
    - Подчеркивай выгоды клиента
    - Предлагай дополнительные варианты: "Возможно вас также заинтересует..."
    - Завершай вопросом: "Хотите уточнить какие-то детали?"

    **Пример ответа:**
    "Добрый день! <основной ответ из контекста>. 
    Нужны дополнительные подробности? 😊"

    **Технические примечания:**
    - Контекст обрезан до 512 токенов
    - Избегай markdown-разметки
    - Ответ держи в пределах 3 предложений
    """

    client = OpenAI(api_key=deepseek_token,
                    base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        stream=False
    )

    return response.choices[0].message.content


if __name__ == '__main__':

    prompt = 'Как заработать миллион молодому человеку'

    print(simple_ask(prompt))

    print('-\n\n' * 3)

    context = data.find_text(prompt)
    print(ask_context(prompt, context))
