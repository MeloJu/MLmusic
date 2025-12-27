Removendo arquivos já comitados que agora estão em `.gitignore`

Use os comandos abaixo no diretório do projeto para limpar o índice do Git
e remover arquivos ignorados (como `__pycache__` e `.venv`) do controle de versão.

OBS: faça um backup/commit antes se tiver alterações não commitadas.

1) Verifique o estado atual:

   git status --porcelain

2) Para remover os arquivos ignorados já versionados e reindexar conforme `.gitignore`:

   git rm -r --cached .
   git add .
   git commit -m "Remove ignored files (pycache, venv, chroma_db) and add .gitignore"

Esse fluxo limpa o índice do Git (sem deletar os arquivos locais) e depois adiciona
apenas os arquivos que não estão listados no `.gitignore`.

Windows / PowerShell (alternativa):

   # remove cached files matching patterns
   git rm -r --cached __pycache__ || echo 'no __pycache__ tracked'
   git rm -r --cached .venv || echo 'no .venv tracked'
   git add .gitignore
   git commit -m "Add .gitignore and remove cached artifacts"

Depois faça `git push`.
