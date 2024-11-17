import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSequenceClassification
import rouge
from rouge import Rouge
import torch

# девайс для торча
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# загрузка модели для сентимента
def load_sentiment_model():
    sentiment_model_id = "blanchefort/rubert-base-cased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(sentiment_model_id)
    model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_id)
    sentiment_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return sentiment_pipe

# готовим нашу модель для суммаризации
model_name = "Gnider/mix_6ep_15k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to(device)

# генерация заголовка
def generate_headline(article_text):
    text_tokens = tokenizer(
        article_text,
        max_length=600,
        add_special_tokens=False,
        padding=False,
        truncation=True
    )["input_ids"]
    input_ids = text_tokens + [tokenizer.sep_token_id]
    input_ids = torch.LongTensor([input_ids]).to(device)
    output_ids = model.generate(
        input_ids=input_ids,
        no_repeat_ngram_size=0
    )
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    summary = summary.split(tokenizer.sep_token)[1]
    summary = summary.split(tokenizer.eos_token)[0]
    return summary.strip()

# посчитаем скоры руж для заголовков
def calculate_rouge(predicted_title, original_title):
    predicted_title = predicted_title.strip()
    original_title = original_title.strip()  
    
    rouge = Rouge()
    scores = rouge.get_scores(predicted_title, original_title, avg=True)
    f_scores = {k: v['f'] for k, v in scores.items()}  #берем только ф-меры
    
    # посчитаем среднее ф-мер для усредненной оценки заголовка
    final_score = sum(f_scores.values()) / len(f_scores)
    return f_scores, final_score
# сентимент
def analyze_sentiment(sentiment_pipe, text):
    return sentiment_pipe(text)

# интерфйс стримлит
def main():
    st.title("Генерация новостных заголовков (наука и спорт) и сентимент-анализ")

    # инициализируем модель для сентимента
    sentiment_pipe = load_sentiment_model()

    # поля ввода
    article_text = st.text_area("Введите текст новости на русском языке:", height=200)
    original_title = st.text_input("Введите оригинальный заголовок статьи для рассчета формальной близости сгенерированного заголовка (опционально):")

    if st.button("Сгенерировать"):
        if article_text.strip():
            # генерируем заголовок
            predicted_title = generate_headline(article_text)

            # отображение заголовка
            st.subheader("Сгенерированный заголовок:")
            st.write(predicted_title)

            # сентимент анализ
            sentiment_results = analyze_sentiment(sentiment_pipe, predicted_title)
            st.subheader("Сентимент-анализ сгенерированного заголовка:")
            for result in sentiment_results:
                for sentiment in result:
                    label, score = sentiment['label'], sentiment['score']
                    st.write(f"{label}: {score:.2f}")

            # руж
            if original_title.strip():
                rouge_scores, final_score = calculate_rouge(predicted_title, original_title)
                st.subheader("Оценка ROUGE:")
                for k, v in rouge_scores.items():
                    st.write(f"{k.upper()}: {v:.2f}")
                st.subheader("Среднее F-мер:")
                st.write(f"Сгенерированный заголовок похож на оригинальный в среднем на {final_score * 100:.0f} %") # выводим в виде процентов

        else:
            st.error("Введите текст новости")

if __name__ == "__main__":
    main()

