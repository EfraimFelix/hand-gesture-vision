## 🤖 Webots Gesture Detector

Este projeto utiliza técnicas de visão computacional para controlar robôs no simulador Webots. Através do processamento de imagens capturadas pela câmera, o sistema é capaz de interpretar comandos e executar ações no ambiente simulado.


## 🎯 Treinando o Modelo

Para coletar dados para o treinamento do modelo, execute o seguinte comando:

```bash
python generate_dataset.py
```

#### 📌 Controles Disponíveis:

- `R` → Iniciar/Parar gravação
- `1` → Movimento para FRENTE
- `2` → Movimento para TRÁS
- `3` → Movimento para DIREITA
- `4` → Movimento para ESQUERDA
- `S` → Salvar dados em um arquivo CSV
- `Q` → Sair

#### 🚩 Como Treinar um Gesto:
- Selecione um dos movimentos disponíveis (`1` a `4`).
- Pressione `R` para iniciar a gravação do gesto.
- Após concluir o gesto, pressione `R` novamente para parar a gravação.
- Repita o processo para treinar mais gestos, se necessário.
- Pressione `S` para salvar os dados coletados em um arquivo CSV.
- Pressione `Q` para sair do programa.


Para treinar o modelo, execute o seguinte comando:

```bash
python gesture_classifier.py
```

Por fim, para testar o modelo, execute o seguinte comando:

```bash
python gesture_recognition.py
```

## 🫡 Integração com Webots

Faça o [download](https://cyberbotics.com/) do Webots e abra o mundo disponível na pasta:

`` 
file > open world > .\hand-gesture-vision\webots\worlds
``

Aperte no play para iniciar a simulação com os gestos treinados.
