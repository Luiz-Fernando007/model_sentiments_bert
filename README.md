# 🤖 Classificador de Sentimentos com BERT

Este projeto utiliza um modelo BERT treinado para classificar textos em diferentes sentimentos, como Frustração, Satisfação, Confusão, Neutro, Urgência e Raiva.

## 🚀 Como rodar

### ✅ Requisitos

- Python 3.8+
- PyTorch
- Transformers (`pip install transformers`)
- GPU (opcional, mas recomendado para desempenho)

### 📦 Instalando dependências

No terminal, execute:

```sh
pip install torch transformers
```

### ▶️ Executando o código

1. Certifique-se de que a pasta `app/modelo_sentimentos` contém os arquivos do modelo treinado (incluindo `config.json`, `pytorch_model.bin`, `vocab.txt`).
2. No terminal, execute um script Python ou use o exemplo abaixo:

```python
from app.models import analise_sentimento

texto = "Seu texto para análise aqui."
sentimento = analise_sentimento(texto)
print("Sentimento identificado:", sentimento)
```

Você verá no terminal se o CUDA está disponível e qual dispositivo está sendo usado.

### 🌐 Suporte ao Português

O modelo foi treinado para compreender textos em **português**.

---

## 📁 Estrutura do Projeto

- `app/models.py`: Função principal para análise de sentimentos.
- `app/modelo_sentimentos/`: Diretório com o modelo BERT treinado.
- `README.md`: Este arquivo.

---

## 📝 Exemplo de Saída

```sh
CUDA disponível? True
Dispositivo: NVIDIA GeForce RTX 3060
Sentimento identificado: Satisfação
```

---

## 📬 Contato

Contribuições, sugestões ou dúvidas são bem-vindas!  
Abra uma issue ou entre em contato diretamente pelo GitHub.
