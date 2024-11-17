# Сервис на langchain и streamlit для:
- генерации заголовков новостных статей по тексту
- сентимент-анализа заголовка
- оценки сгенерированного заголовка с оригинальным по метрике ROUGE
- 
Модель для генерации заголовка на HF - [Gnider/mix_6ep_15k]([url](https://huggingface.co/Gnider/mix_6ep_15k))

Модель для оценки сентимента на HF - [blanchefort/rubert-base-cased-sentiment]([url](https://huggingface.co/blanchefort/rubert-base-cased-sentiment))

Для запуска сервиса:
```
streamlit run app.py
```
