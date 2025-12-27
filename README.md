# MLMusics

Recomendador simples de músicas usando **embeddings (Sentence-Transformers)** + **busca por similaridade (ChromaDB)**.

O objetivo é: dado um input (texto livre ou `title/artist/keywords`), retornar as faixas mais próximas no espaço vetorial.

## Como rodar

### 1) Instalar dependências

No Windows (PowerShell) dentro da pasta do projeto:

- `python -m pip install -r requirements.txt`

### 2) Gerar a base de embeddings (index)

Use o CSV do repositório (ex.: `itunes_songs.csv`) e gere o banco local do Chroma:

- `python embedder.py --csv itunes_songs.csv --chroma_dir ./chroma_db --collection songs`

Isso cria (ou atualiza) a coleção `songs` em `./chroma_db`.

### 3) Consultar recomendações (modo interativo)

Recomendações “puras” (top-K por similaridade):

- `python query_interface.py --chroma_dir ./chroma_db --collection songs --top_k 5 --no_diverse_artists`

Recomendações com **artistas diferentes** (pega mais candidatos e filtra por artista):

- (padrão) `python query_interface.py --chroma_dir ./chroma_db --collection songs --top_k 5 --max_per_artist 1 --fetch_multiplier 8`

### 4) UI básica (Streamlit)

Uma interface simples para pesquisar e ver resultados em tabela:

- Recomendado (Windows/Conda-proof): `./.venv/Scripts/python.exe -m streamlit run app.py`
- Alternativa (mais à prova de PATH/Conda no Windows): `./.venv/Scripts/python.exe run_ui.py`

Se preferir ativar o venv antes:

- PowerShell: `./.venv/Scripts/Activate.ps1`
- Depois: `python -m streamlit run app.py`

Se você estiver com `(base)` do Conda ativo no terminal e der erro de `torch`/DLL (ex.: `shm.dll`), isso normalmente significa que o Streamlit está rodando com o Python do Conda em vez da `.venv` do projeto.

Exemplos de consulta:

- `title="Bohemian Rhapsody" artist=Queen`
- `keywords=rock vibe=energetic`
- `lofi chill study beats` (texto livre)

## (Opcional) Baixar dados do iTunes novamente

Atenção: gerar termos de 2 letras pode buscar **muitos** registros.

- `python songs.py --letters 2 --limit 200 --country us --out itunes_songs.csv`

## Notas

- O Chroma pode variar APIs entre versões; o código tenta `PersistentClient` e faz fallback para o modo antigo.
- A “similaridade” exibida é uma conversão simples da distância retornada pelo Chroma (quanto menor a distância, mais similar).
- Diversificação por artista pode reduzir um pouco a similaridade média, mas melhora variedade.
