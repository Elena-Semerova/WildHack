## Эксперименты на 10 классах

Датасет представлен в 10 категориях животных: `another, bear, deer, fox, hog, lynx, saiga, steppe eagle, tiger, wolf`.

Наблюдается дисбаланс классов (суммарно для тренировочных и валидацинных данных):

![Alt-текст](https://sun9-27.userapi.com/impg/CK7WL0bcic_IczR9BnHHGD2KHg_pYT-DGWf1NA/GF6SKRokX1A.jpg?size=864x432&quality=96&sign=bbe2f6a18b51ae5c1db3d47ceed236ec&type=album)

В качестве моделей использовались различные предобученные версии ResNet, обучение происходило на 10 эпохах. Результаты:

| Модель | Accuracy |
|----------------|:----------------:|
| ResNet18 | 0.5950 | 
| ResNet50 | 0.6725 | 
| ResNet152 | 0.7150 | 
