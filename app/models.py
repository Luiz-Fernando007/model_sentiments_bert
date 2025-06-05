from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

# Mapeamento reverso dos índices para sentimentos
sentimento_map_reverso = {
    0: "Sentimento",
    1: "Frustração",
    2: "Satisfação",
    3: "Confusão",
    4: "Neutro",
    5: "Urgência",
    6: "Raiva"
}

def analise_sentimento(texto: str) -> str:
    print("CUDA disponível?", torch.cuda.is_available())
    print("Dispositivo:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    # Carregar o tokenizador e modelo do caminho local onde você salvou após o treinamento
    tokenizer = BertTokenizer.from_pretrained("app/modelo_sentimentos", local_files_only=True)
    model = BertForSequenceClassification.from_pretrained("app/modelo_sentimentos", local_files_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Tokenizar o texto
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Fazer inferência
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    # Retornar o sentimento correspondente
    return sentimento_map_reverso.get(predicted_class, "Desconhecido")
  # durante o treino
